"""Experiment: can text/metadata features of the already-generated bf16
trace predict where the quantized trace first diverges (FDP), without any
K-matrices or GPU?

This is a deliberately different, weaker-signal experiment from the
mechanistic-analysis report's own FDP predictor (a 1D-CNN over raw K
matrices, R²=0.98 in-distribution on Qwen3-1.7B — see
research/kv-cache-reasoning-divergence-study/mechanistic-analysis/). We
don't have K-matrices for the real study models here (no GPU), so this asks
a narrower question: is there *any* signal in cheap, CPU-only proxies —
n-gram repetition rate, trace length, finish_reason, baseline correctness —
run per (model, quant) group with 5-fold CV, analytical (1-feature linear)
vs. gradient-boosted (sklearn) models, reported via Spearman rank
correlation (R² is unstable on these small, heavy-tailed targets — see
`cross_validate_spearman`'s docstring).

No judge fields (category/confidence/...) are used as features — see
`kvtrace.analysis.fdp_predictor`'s module docstring for why that would be
circular.

Reads:
  outputs/fdps/*.jsonl, outputs/traces/*.jsonl (or --fdps_dir/--traces_dir)

Writes:
  outputs/fdp_text_predictor.json   per-group CV results
  outputs/fdp_text_predictor.md     human-readable summary table
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from kvtrace.analysis.fdp_predictor import build_feature_table, cross_validate_spearman

log = logging.getLogger("fdp_text_predictor")

FULL_FEATURES = [
    "num_generated_tokens", "num_prompt_tokens", "finish_reason_length",
    "think_complete", "baseline_correct", "char_length",
    "repetition_score_4gram", "repetition_score_8gram",
]
REDUCED_FEATURES = ["repetition_score_4gram", "num_generated_tokens"]
ANALYTICAL_FEATURES = ["repetition_score_4gram"]


def _load_jsonl_dir(directory: Path) -> list[dict]:
    out: list[dict] = []
    for f in sorted(directory.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            out.extend(json.loads(line) for line in fh)
    return out


def _group_by(records: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        out[tuple(r.get(k) for k in keys)].append(r)
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdps_dir", default="outputs/fdps")
    parser.add_argument("--traces_dir", default="outputs/traces")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    fdps_by_key = _group_by(_load_jsonl_dir(Path(args.fdps_dir)), ("model", "quant_method"))
    traces_by_key = _group_by(_load_jsonl_dir(Path(args.traces_dir)), ("model", "quant_method"))
    rows = build_feature_table(fdps_by_key, traces_by_key)
    if not rows:
        log.error("no usable feature rows built from %s / %s", args.fdps_dir, args.traces_dir)
        return 3
    log.info("built %d feature rows", len(rows))

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_group[(r["model"], r["quant_method"])].append(r)

    def col(group_rows: list[dict], feats: list[str]) -> list[list[float]]:
        return [[float(r[f]) for f in feats] for r in group_rows]

    results: dict[str, dict] = {}
    for (model, quant), group_rows in sorted(by_group.items()):
        y = [r["fdp_token_idx"] for r in group_rows]
        analytical = cross_validate_spearman(
            col(group_rows, ANALYTICAL_FEATURES), y,
            lambda: LinearRegression(), args.n_splits, args.seed,
        )
        gbm_reduced = cross_validate_spearman(
            col(group_rows, REDUCED_FEATURES), y,
            lambda: GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=0),
            args.n_splits, args.seed,
        )
        gbm_full = cross_validate_spearman(
            col(group_rows, FULL_FEATURES), y,
            lambda: GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=0),
            args.n_splits, args.seed,
        )
        key = f"{model}__{quant}"
        results[key] = {
            "n": len(group_rows),
            "fdp_mean": sum(y) / len(y),
            "analytical_1feature": analytical,
            "gbm_2feature": gbm_reduced,
            "gbm_8feature": gbm_full,
        }
        log.info(
            "%s: n=%d spearman analytical=%.3f gbm2f=%.3f gbm8f=%.3f",
            key, len(group_rows),
            analytical["spearman_mean"], gbm_reduced["spearman_mean"], gbm_full["spearman_mean"],
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fdp_text_predictor.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# FDP prediction from text/metadata features only (no K-matrices)\n",
        "| model x quant | n | analytical (1f) | GBM (2f) | GBM (8f) |",
        "|---|---|---|---|---|",
    ]
    for key, r in results.items():
        lines.append(
            f"| {key} | {r['n']} | {r['analytical_1feature']['spearman_mean']:.3f} "
            f"| {r['gbm_2feature']['spearman_mean']:.3f} | {r['gbm_8feature']['spearman_mean']:.3f} |"
        )
    (out_dir / "fdp_text_predictor.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log.info("wrote %s and .md", out_dir / "fdp_text_predictor.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
