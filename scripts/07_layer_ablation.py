"""Phase 6b: single-layer ablation (mechanistic-analysis report §4.5).

Quantizes exactly one layer at a time (all others stay bf16) and ranks
layers by how much that shifts final logits from the all-bf16 baseline —
the "logit-impact" ranking the report compares against the per-layer
attention-shift-KL ranking from `06_kv_capture.py`'s full-model quantization.

The underlying `single_layer_ablation_kl` is verified via `pytest -m network`.

Reads:
  outputs/traces/{model}_bf16.jsonl   (needs "token_ids" per record)

Writes (outputs/kv_capture/{model}/{quant}/):
  layer_ablation.json   {"problem_idx": {"layer_idx": mean_logit_kl, ...}, ...}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.config import load_all_configs

log = logging.getLogger("layer_ablation")


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
    parser.add_argument(
        "--quant", required=True, choices=["fp8_e4m3", "fp8_e5m2"],
        help="report §5.3: the defense/ablation recipes are FP8-specific",
    )
    parser.add_argument("--traces_dir", default="outputs/traces")
    parser.add_argument("--out_dir", default="outputs/kv_capture")
    parser.add_argument("--n_problems", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    import torch
    from transformers import AutoModelForCausalLM

    from kvtrace.kv_capture.capture_generator import single_layer_ablation_kl

    models, _, _ = load_all_configs(Path("config"))
    if args.model not in models:
        log.error("unknown model %r; available: %s", args.model, sorted(models))
        return 2
    mcfg = models[args.model]

    problems = _load_baseline_traces(Path(args.traces_dir), args.model, args.n_problems)
    log.info("ablating %d layers x %d problems for %s x %s", 0, len(problems), args.model, args.quant)

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

    results: dict[str, dict[str, float]] = {}
    for row in problems:
        token_ids = row["token_ids"][: args.max_tokens]
        input_ids = torch.tensor([token_ids], device=model.device)
        attention_mask = torch.ones_like(input_ids)

        scores = single_layer_ablation_kl(model, input_ids, attention_mask, args.quant)
        results[str(row.get("idx"))] = {str(k): v for k, v in scores.items()}
        top_layer = max(scores, key=scores.get)
        log.info(
            "problem idx=%s: top logit-impact layer=%d (kl=%.4f)",
            row.get("idx"), top_layer, scores[top_layer],
        )

    (out_dir / "layer_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("wrote layer ablation results to %s", out_dir / "layer_ablation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
