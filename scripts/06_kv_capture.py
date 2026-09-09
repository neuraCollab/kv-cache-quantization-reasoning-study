"""Phase 6 (mechanistic capture, GPU-required): teacher-force already-
generated bf16 traces through the real model under a quantized KV cache,
capturing per-layer attention/logit/KV divergence from baseline.

This reconstructs the *infrastructure* behind
research/kv-cache-reasoning-divergence-study/mechanistic-analysis/ — whose
original generating code was not recovered, only its output artifacts.
Only a subset of that directory's analyses are covered here (attention-shift
KL, per-layer KV stats, logit-KL trajectory, outlier-channel scores); layer
ablation, counterfactual skip-K, the CNN failure-predictor, and multi-seed
variance runs are not — see
research/kv-cache-reasoning-divergence-study/README.md.

GPU-required in practice (real 1.5B-7B models); the capture plumbing itself
is verified against a tiny real model on CPU by `pytest -m network`
(tests/test_kv_capture_generator.py) — this script has not been run against
the actual study models.

Reads:
  outputs/traces/{model}_bf16.jsonl   (needs "token_ids" per record)

Writes (outputs/kv_capture/{model}/{quant}/):
  per_problem.npz         attention_shift_kl, logit_kl_trajectory,
                           kv_stats_{baseline,quant}, outlier_channels_*
                           for each of the first --n_problems problems
  summary.json             mean attention_shift_kl per layer across problems
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from kvtrace.config import load_all_configs

log = logging.getLogger("kv_capture")


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
        "--quant",
        required=True,
        choices=["fp8_e4m3", "fp8_e5m2", "hqq_int4", "hqq_int2"],
        help="cache variant to compare against the bf16 baseline",
    )
    parser.add_argument("--traces_dir", default="outputs/traces")
    parser.add_argument("--out_dir", default="outputs/kv_capture")
    parser.add_argument("--n_problems", type=int, default=5, help="how many problems to replay")
    parser.add_argument("--max_tokens", type=int, default=2048, help="truncate replay to this many tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    import torch
    from transformers import AutoModelForCausalLM

    from kvtrace.kv_capture.capture_generator import compare_baseline_vs_quant

    models, _, _ = load_all_configs(Path("config"))
    if args.model not in models:
        log.error("unknown model %r; available: %s", args.model, sorted(models))
        return 2
    mcfg = models[args.model]

    problems = _load_baseline_traces(Path(args.traces_dir), args.model, args.n_problems)
    log.info("replaying %d problems for %s x %s", len(problems), args.model, args.quant)

    # token_ids come straight from the already-generated bf16 trace — no
    # tokenizer needed, we're replaying the exact same token sequence.
    model = AutoModelForCausalLM.from_pretrained(
        mcfg.hf_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=mcfg.trust_remote_code,
        attn_implementation="eager",  # required for output_attentions=True
    )
    model.eval()

    out_dir = Path(args.out_dir) / args.model / args.quant
    out_dir.mkdir(parents=True, exist_ok=True)

    per_problem_results = []
    for row in problems:
        token_ids = row["token_ids"][: args.max_tokens]
        input_ids = torch.tensor([token_ids], device=model.device)
        attention_mask = torch.ones_like(input_ids)

        result = compare_baseline_vs_quant(model, input_ids, attention_mask, args.quant)
        per_problem_results.append(result)
        log.info(
            "problem idx=%s: mean attention_shift_kl=%.4f",
            row.get("idx"),
            float(np.mean(result["attention_shift_kl"])),
        )

    n_layers = len(per_problem_results[0]["attention_shift_kl"])
    shift_matrix = np.array([r["attention_shift_kl"] for r in per_problem_results])  # [n_problems, n_layers]
    trajectory_arrays = {
        f"logit_kl_trajectory_{i}": np.array(r["logit_kl_trajectory"])
        for i, r in enumerate(per_problem_results)
    }
    np.savez(
        out_dir / "per_problem.npz",
        attention_shift_kl=shift_matrix,
        **trajectory_arrays,  # type: ignore[arg-type]  # numpy stubs don't model **kwargs here
    )

    summary = {
        "model": args.model,
        "quant": args.quant,
        "n_problems": len(problems),
        "n_layers": n_layers,
        "attention_shift_kl_mean_per_layer": shift_matrix.mean(axis=0).tolist(),
        "attention_shift_kl_std_per_layer": shift_matrix.std(axis=0).tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("wrote kv_capture results to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
