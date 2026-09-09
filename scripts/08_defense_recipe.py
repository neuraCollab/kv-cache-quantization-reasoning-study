"""Phase 6c: per-channel defense recipe — TF measurement + AR validation
(mechanistic-analysis report §5).

Protects the top-N outlier channels per layer in bf16 while quantizing the
rest, first in a single teacher-forced forward pass (the "lab" measurement,
report §5.2/§5.3 — looks like it works), then in real autoregressive
`model.generate()` (report §5.4 — the "calibrated" row of Table 13, where
the report found the lab benefit mostly disappears: Ф.2, the lab→production
gap). Runs both by default; `--mode` restricts to one.

Only meaningful for FP8 variants — report §5.3 found this recipe makes HQQ
*worse* (it fights HQQ's own per-group calibration), so `--quant` doesn't
accept hqq_int4/hqq_int2 here.

GPU-required in practice; `teacher_forced_defense_effect` and
`ar_defense_validation` are verified against a tiny real model on CPU by
`pytest -m network`.

Reads:
  outputs/traces/{model}_bf16.jsonl   (needs "token_ids" per record)

Writes (outputs/kv_capture/{model}/{quant}/):
  defense_tf.json   per-problem {"undefended_k_error": {...}, "defended_k_error": {...}}
  defense_ar.json   per-problem {"undefended": {...}, "defended": {...}}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.config import load_all_configs

log = logging.getLogger("defense_recipe")


def _load_baseline_traces(traces_dir: Path, model: str, n_problems: int) -> list[dict]:
    path = traces_dir / f"{model}_bf16.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"baseline traces missing: {path} (run Phase 1 --config bf16)")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_problems:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key in config/models.yaml")
    parser.add_argument("--quant", required=True, choices=["fp8_e4m3", "fp8_e5m2"])
    parser.add_argument("--mode", choices=["tf", "ar", "both"], default="both")
    parser.add_argument("--top_n", type=int, default=10, help="channels to protect per layer")
    parser.add_argument("--traces_dir", default="outputs/traces")
    parser.add_argument("--out_dir", default="outputs/kv_capture")
    parser.add_argument("--n_problems", type=int, default=5, help="how many problems for TF mode")
    parser.add_argument("--n_problems_ar", type=int, default=10, help="report §5.4 used 10")
    parser.add_argument("--max_tokens", type=int, default=2048, help="TF replay truncation")
    parser.add_argument(
        "--max_new_tokens", type=int, default=100, help="AR generation length (report §5.4: 100)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    import torch
    from transformers import AutoModelForCausalLM

    from kvtrace.kv_capture.capture_generator import (
        ar_defense_validation,
        capture_forward_pass,
        select_defense_channels,
        teacher_forced_defense_effect,
    )

    models, _, _ = load_all_configs(Path("config"))
    if args.model not in models:
        log.error("unknown model %r; available: %s", args.model, sorted(models))
        return 2
    mcfg = models[args.model]

    model = AutoModelForCausalLM.from_pretrained(
        mcfg.hf_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=mcfg.trust_remote_code,
        attn_implementation="eager",
    )
    model.eval()

    out_dir = Path(args.out_dir) / args.model / args.quant
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("tf", "both"):
        problems = _load_baseline_traces(Path(args.traces_dir), args.model, args.n_problems)
        log.info("TF defense effect: %d problems, top_n=%d", len(problems), args.top_n)
        tf_results = {}
        for row in problems:
            token_ids = row["token_ids"][: args.max_tokens]
            input_ids = torch.tensor([token_ids], device=model.device)
            attention_mask = torch.ones_like(input_ids)
            result = teacher_forced_defense_effect(
                model, input_ids, attention_mask, args.quant, args.top_n
            )
            tf_results[str(row.get("idx"))] = {
                "undefended_k_error": result["undefended_k_error"],
                "defended_k_error": result["defended_k_error"],
            }
        (out_dir / "defense_tf.json").write_text(json.dumps(tf_results, indent=2), encoding="utf-8")
        log.info("wrote %s", out_dir / "defense_tf.json")

    if args.mode in ("ar", "both"):
        problems = _load_baseline_traces(Path(args.traces_dir), args.model, args.n_problems_ar)
        log.info(
            "AR defense validation: %d problems, %d new tokens, top_n=%d",
            len(problems), args.max_new_tokens, args.top_n,
        )
        ar_results = {}
        for row in problems:
            token_ids = row["token_ids"][: args.max_tokens]
            input_ids = torch.tensor([token_ids], device=model.device)
            attention_mask = torch.ones_like(input_ids)

            baseline = capture_forward_pass(model, input_ids, attention_mask, "bf16")
            protected_channels = select_defense_channels(baseline.kv_snapshot, args.top_n)

            result = ar_defense_validation(
                model, input_ids, attention_mask, args.quant, protected_channels, args.max_new_tokens
            )
            ar_results[str(row.get("idx"))] = result
            log.info(
                "problem idx=%s: undefended first_divergence=%s, defended first_divergence=%s",
                row.get("idx"),
                result["undefended"]["first_divergence_step"],
                result["defended"]["first_divergence_step"],
            )
        (out_dir / "defense_ar.json").write_text(json.dumps(ar_results, indent=2), encoding="utf-8")
        log.info("wrote %s", out_dir / "defense_ar.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
