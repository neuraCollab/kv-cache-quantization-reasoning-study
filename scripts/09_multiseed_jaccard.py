"""Phase 6d: multi-seed outlier-channel identity check (mechanistic-analysis
report §4.3).

Generates `--n_seeds` independent bf16 continuations of the same prompt at
`--temperature > 0` (a different seed each), picks each run's top-N outlier
K-channels per layer, and measures pairwise Jaccard overlap. A high median
(report: 0.879) means outlier-channel identity is a property of the model's
learned weights, not an artifact of one sampled continuation — which is what
justifies calibrating the defense recipe's channel set once instead of
per-generation.

GPU-required in practice; `multiseed_channel_jaccard` is verified against a
tiny real model on CPU by `pytest -m network`.

Reads:
  outputs/traces/{model}_bf16.jsonl   (only used for prompts — "token_ids"
                                        sliced to the prompt, not replayed)

Writes (outputs/kv_capture/{model}/):
  multiseed_jaccard.json   {"median_jaccard": ..., "per_problem": {...}}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.config import load_all_configs

log = logging.getLogger("multiseed_jaccard")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key in config/models.yaml")
    parser.add_argument("--traces_dir", default="outputs/traces")
    parser.add_argument("--out_dir", default="outputs/kv_capture")
    parser.add_argument("--n_problems", type=int, default=3, help="prompts to check")
    parser.add_argument("--prompt_tokens", type=int, default=64, help="prompt length to reuse")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--temperature", type=float, default=0.6, help="report §4.3 used T=0.6")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    import torch
    from transformers import AutoModelForCausalLM

    from kvtrace.kv_capture.capture_generator import multiseed_channel_jaccard

    models, _, _ = load_all_configs(Path("config"))
    if args.model not in models:
        log.error("unknown model %r; available: %s", args.model, sorted(models))
        return 2
    mcfg = models[args.model]

    traces_path = Path(args.traces_dir) / f"{args.model}_bf16.jsonl"
    if not traces_path.exists():
        log.error("baseline traces missing: %s (run Phase 1 --config bf16)", traces_path)
        return 2
    problems = []
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))
            if len(problems) >= args.n_problems:
                break

    model = AutoModelForCausalLM.from_pretrained(
        mcfg.hf_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=mcfg.trust_remote_code,
        attn_implementation="eager",
    )
    model.eval()

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    per_problem = {}
    for row in problems:
        prompt_ids = row["token_ids"][: args.prompt_tokens]
        input_ids = torch.tensor([prompt_ids], device=model.device)
        attention_mask = torch.ones_like(input_ids)

        result = multiseed_channel_jaccard(
            model,
            input_ids,
            attention_mask,
            top_n=args.top_n,
            seeds=tuple(args.seeds),
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        per_problem[str(row.get("idx"))] = result
        log.info("problem idx=%s: median Jaccard=%.3f", row.get("idx"), result["median_jaccard"])

    all_medians = [r["median_jaccard"] for r in per_problem.values()]
    overall = {
        "median_jaccard": sorted(all_medians)[len(all_medians) // 2] if all_medians else 0.0,
        "per_problem": per_problem,
    }
    (out_dir / "multiseed_jaccard.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    log.info("wrote %s", out_dir / "multiseed_jaccard.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
