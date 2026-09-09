"""Phase 5: post-hoc paper analysis. Reads Phase 1-3 artifacts (traces, FDPs,
judgments) and writes the tables/plots referenced by the study write-up
(research/kv-cache-reasoning-divergence-study/paper/supervisor_report.md §5).

No GPU, no network, no tokenizer — pure local aggregation over already
generated JSONL, same as Phase 4. Safe to run repeatedly; it never mutates
its inputs and always overwrites its outputs deterministically.

Reads:
  outputs/traces/{model}_{config}.jsonl
  outputs/fdps/{model}_{config}.jsonl
  outputs/judgments/{model}_{config}.jsonl

Writes (outputs/paper/):
  accuracy_bars.png
  divergence_position.png, divergence_position.md
  fdp_rate.md
  finish_reason.md
  judge_confidence.md
  per_model_chi2.md, per_model_chi2.csv
  quant_only_deepdives.md
  token_efficiency.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kvtrace.analysis.paper import (
    compute_accuracy_table,
    compute_divergence_position_table,
    compute_fdp_rate_table,
    compute_finish_reason_table,
    compute_judge_confidence_table,
    compute_per_model_chi2,
    compute_quant_only_deepdives,
    compute_token_efficiency_table,
    load_all_records,
    load_records_by_key,
)
from kvtrace.analysis.paper_report import (
    render_divergence_position_table,
    render_fdp_rate_table,
    render_finish_reason_table,
    render_judge_confidence_table,
    render_per_model_chi2,
    render_quant_only_deepdives,
    render_token_efficiency_table,
    write_accuracy_bars_plot,
    write_divergence_position_plot,
    write_per_model_chi2_csv,
)

log = logging.getLogger("paper_analysis")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_dir", default="outputs/traces")
    parser.add_argument("--fdps_dir", default="outputs/fdps")
    parser.add_argument("--judgments_dir", default="outputs/judgments")
    parser.add_argument("--out_dir", default="outputs/paper")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    traces_dir = Path(args.traces_dir)
    fdps_dir = Path(args.fdps_dir)
    judgments_dir = Path(args.judgments_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not traces_dir.exists():
        log.error("traces dir missing: %s (run Phase 1 first)", traces_dir)
        return 2
    if not fdps_dir.exists():
        log.error("fdps dir missing: %s (run Phase 2 first)", fdps_dir)
        return 2
    if not judgments_dir.exists():
        log.error("judgments dir missing: %s (run Phase 3 first)", judgments_dir)
        return 2

    traces_by_key = load_records_by_key(traces_dir)
    fdps_by_key = load_records_by_key(fdps_dir)
    judgments_by_key = load_records_by_key(judgments_dir)
    judgments = load_all_records(judgments_dir)

    log.info(
        "loaded %d trace groups, %d fdp groups, %d judgments",
        len(traces_by_key), len(fdps_by_key), len(judgments),
    )

    accuracy_rows = compute_accuracy_table(traces_by_key)
    write_accuracy_bars_plot(accuracy_rows, out_dir / "accuracy_bars.png")

    divergence_rows = compute_divergence_position_table(fdps_by_key, traces_by_key)
    write_divergence_position_plot(divergence_rows, out_dir / "divergence_position.png")
    (out_dir / "divergence_position.md").write_text(
        render_divergence_position_table(divergence_rows), encoding="utf-8"
    )

    fdp_rate_rows = compute_fdp_rate_table(fdps_by_key)
    (out_dir / "fdp_rate.md").write_text(render_fdp_rate_table(fdp_rate_rows), encoding="utf-8")

    finish_reason_rows = compute_finish_reason_table(traces_by_key)
    (out_dir / "finish_reason.md").write_text(
        render_finish_reason_table(finish_reason_rows), encoding="utf-8"
    )

    judge_confidence_rows = compute_judge_confidence_table(judgments)
    (out_dir / "judge_confidence.md").write_text(
        render_judge_confidence_table(judge_confidence_rows), encoding="utf-8"
    )

    chi2_result = compute_per_model_chi2(judgments)
    (out_dir / "per_model_chi2.md").write_text(render_per_model_chi2(chi2_result), encoding="utf-8")
    write_per_model_chi2_csv(chi2_result, out_dir / "per_model_chi2.csv")

    deepdives = compute_quant_only_deepdives(fdps_by_key, judgments_by_key)
    (out_dir / "quant_only_deepdives.md").write_text(
        render_quant_only_deepdives(deepdives), encoding="utf-8"
    )

    token_eff_rows = compute_token_efficiency_table(traces_by_key)
    (out_dir / "token_efficiency.md").write_text(
        render_token_efficiency_table(token_eff_rows), encoding="utf-8"
    )

    log.info("wrote paper analysis to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
