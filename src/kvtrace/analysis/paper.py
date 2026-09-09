"""Phase 5 post-hoc analysis: per-model/per-quant tables beyond the Phase 4
global report.md. Pure computation over already-generated traces/fdps/
judgments — no GPU, no network, no tokenizer.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from kvtrace.analysis.signatures import (
    CATEGORY_ORDER,
    aggregate_counts,
    chi_square_test,
    cramers_v,
    effect_size_label,
    natural_dof,
    standardized_residuals,
)

RecordsByKey = dict[tuple[str, str], list[dict]]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_all_records(directory: Path) -> list[dict]:
    """Flat list of every JSON record across every *.jsonl file in directory."""
    out: list[dict] = []
    for f in sorted(Path(directory).glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            out.extend(json.loads(line) for line in fh if line.strip())
    return out


def load_records_by_key(
    directory: Path, key_fields: tuple[str, str] = ("model", "quant_method")
) -> RecordsByKey:
    """Group every record across *.jsonl files in directory by (model, quant)."""
    by_key: RecordsByKey = defaultdict(list)
    for rec in load_all_records(directory):
        key = (rec[key_fields[0]], rec[key_fields[1]])
        by_key[key].append(rec)
    return dict(by_key)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def compute_accuracy(traces: list[dict]) -> float:
    if not traces:
        return 0.0
    correct = sum(
        1
        for t in traces
        if t.get("boxed_answer") is not None
        and str(t["boxed_answer"]).strip() == str(t.get("ground_truth", "")).strip()
    )
    return correct / len(traces)


def compute_accuracy_table(traces_by_key: RecordsByKey) -> list[dict]:
    rows = []
    for (model, quant), traces in traces_by_key.items():
        rows.append(
            {
                "model": model,
                "quant": quant,
                "n": len(traces),
                "accuracy_pct": compute_accuracy(traces) * 100,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# finish_reason distribution
# ---------------------------------------------------------------------------


def compute_finish_reason_table(traces_by_key: RecordsByKey) -> list[dict]:
    rows = []
    for (model, quant), traces in traces_by_key.items():
        n = len(traces)
        counts = Counter(t.get("finish_reason") for t in traces)
        raw = dict(counts.most_common())
        rows.append(
            {
                "model": model,
                "quant": quant,
                "n": n,
                "pct_stop": round(counts.get("stop", 0) / n * 100) if n else 0,
                "pct_length": round(counts.get("length", 0) / n * 100) if n else 0,
                "raw": raw,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Token efficiency
# ---------------------------------------------------------------------------


def compute_token_efficiency_table(traces_by_key: RecordsByKey) -> list[dict]:
    baselines: dict[str, float] = {}
    for (model, quant), traces in traces_by_key.items():
        if quant == "bf16" and traces:
            baselines[model] = statistics.mean(t["num_generated_tokens"] for t in traces)

    finish = {
        (r["model"], r["quant"]): r for r in compute_finish_reason_table(traces_by_key)
    }

    rows = []
    for (model, quant), traces in traces_by_key.items():
        avg = statistics.mean(t["num_generated_tokens"] for t in traces) if traces else 0.0
        baseline = baselines.get(model)
        if quant == "bf16" or baseline is None:
            delta, delta_pct = None, None
        else:
            # Compute on the unrounded means, not round(avg) - round(baseline):
            # independently rounding each side first can be off by 1 from
            # rounding the true difference directly.
            delta = round(avg - baseline)
            delta_pct = (avg - baseline) / baseline * 100 if baseline else None
        rows.append(
            {
                "model": model,
                "quant": quant,
                "n": len(traces),
                "avg_gen_tokens": round(avg),
                "delta": delta,
                "delta_pct": delta_pct,
                "pct_length": finish[(model, quant)]["pct_length"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Divergence position
# ---------------------------------------------------------------------------


def compute_divergence_position_table(
    fdps_by_key: RecordsByKey, traces_by_key: RecordsByKey
) -> list[dict]:
    rows = []
    for (model, quant), fdps in fdps_by_key.items():
        baseline_by_idx = {
            t["idx"]: t for t in traces_by_key.get((model, "bf16"), [])
        }
        positions = []
        for rec in fdps:
            fdp_idx = rec.get("fdp_token_idx")
            if fdp_idx is None:
                continue
            baseline = baseline_by_idx.get(rec["problem_idx"])
            if baseline is None or not baseline.get("num_generated_tokens"):
                continue
            positions.append(fdp_idx / baseline["num_generated_tokens"])

        if positions:
            row = {
                "model": model,
                "quant": quant,
                "n": len(positions),
                "median": statistics.median(positions),
                "mean": statistics.mean(positions),
                "std": statistics.pstdev(positions) if len(positions) > 1 else 0.0,
                "min": min(positions),
                "max": max(positions),
            }
        else:
            row = {
                "model": model,
                "quant": quant,
                "n": 0,
                "median": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# FDP rate / boxed_match breakdown
# ---------------------------------------------------------------------------


def compute_fdp_rate_table(fdps_by_key: RecordsByKey) -> list[dict]:
    rows = []
    for (model, quant), fdps in fdps_by_key.items():
        n_pairs = len(fdps)
        diverged = sum(1 for r in fdps if r.get("fdp_token_idx") is not None)
        match_counts = Counter(r.get("boxed_match") for r in fdps)
        rows.append(
            {
                "model": model,
                "quant": quant,
                "n_pairs": n_pairs,
                "diverged": diverged,
                "diverged_pct": round(diverged / n_pairs * 100) if n_pairs else 0,
                "cosmetic_skipped_count": sum(
                    1 for r in fdps if r.get("cosmetic_skipped", 0) > 0
                ),
                "both_correct": match_counts.get("both_correct", 0),
                "baseline_only": match_counts.get("baseline_only", 0),
                "quant_only": match_counts.get("quant_only", 0),
                "both_wrong": match_counts.get("both_wrong", 0),
                "no_boxed": match_counts.get("no_boxed", 0),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Judge confidence validation
# ---------------------------------------------------------------------------


def compute_judge_confidence_table(judgments: list[dict]) -> list[dict]:
    by_cat: dict[str, list[float]] = defaultdict(list)
    for j in judgments:
        cat = j.get("category")
        if cat in CATEGORY_ORDER:
            by_cat[cat].append(float(j["confidence"]))

    rows = []
    for cat in CATEGORY_ORDER:
        confs = by_cat.get(cat)
        if not confs:
            continue
        rows.append(
            {
                "category": cat,
                "n": len(confs),
                "mean_conf": statistics.mean(confs),
                "std_conf": statistics.pstdev(confs) if len(confs) > 1 else 0.0,
                "min": min(confs),
                "max": max(confs),
                "n_below_0.5": sum(1 for c in confs if c < 0.5),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Per-model chi-square
# ---------------------------------------------------------------------------


def compute_per_model_chi2(judgments: list[dict]) -> dict[str, dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for j in judgments:
        by_model[j["model"]].append(j)

    result: dict[str, dict] = {}
    for model, model_judgments in by_model.items():
        methods, counts = aggregate_counts(model_judgments)
        chi2, p, _ = chi_square_test(counts)
        result[model] = {
            "methods": methods,
            "counts": counts,
            "chi2": chi2,
            "p": p,
            "dof": natural_dof(counts),
            "cramers_v": cramers_v(counts),
            "n": int(counts.sum()),
            "effect_size": effect_size_label(cramers_v(counts)),
            "residuals": standardized_residuals(counts),
        }
    return result


# ---------------------------------------------------------------------------
# quant_only deep-dives
# ---------------------------------------------------------------------------


def compute_quant_only_deepdives(
    fdps_by_key: RecordsByKey, judgments_by_key: RecordsByKey
) -> list[dict]:
    cases: list[dict] = []
    for key in sorted(fdps_by_key):
        model, quant = key
        judgments_by_idx = {
            j["problem_idx"]: j for j in judgments_by_key.get(key, [])
        }
        for rec in sorted(fdps_by_key[key], key=lambda r: r["problem_idx"]):
            if rec.get("boxed_match") != "quant_only":
                continue
            judgment = judgments_by_idx.get(rec["problem_idx"], {})
            cases.append(
                {
                    "model": model,
                    "quant": quant,
                    "problem_idx": rec["problem_idx"],
                    "source": rec.get("source"),
                    "problem": rec.get("problem"),
                    "ground_truth": rec.get("ground_truth"),
                    "fdp_token_idx": rec.get("fdp_token_idx"),
                    "baseline_boxed": rec.get("baseline_boxed"),
                    "quant_boxed": rec.get("quant_boxed"),
                    "common_prefix": rec.get("common_prefix"),
                    "baseline_context": rec.get("baseline_context"),
                    "quant_context": rec.get("quant_context"),
                    "category": judgment.get("category"),
                    "confidence": judgment.get("confidence"),
                    "rationale": judgment.get("rationale"),
                    "affected_span": judgment.get("affected_span"),
                }
            )
    return cases
