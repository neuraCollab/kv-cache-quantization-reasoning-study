from __future__ import annotations

import pytest

from kvtrace.analysis.paper import (
    compute_accuracy,
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

# ---------------------------------------------------------------------------
# Synthetic unit tests — precise control over edge cases.
# ---------------------------------------------------------------------------


def _trace(idx, boxed, gt, finish_reason="stop", gen_tokens=100, model="m", quant="bf16"):
    return {
        "idx": idx,
        "boxed_answer": boxed,
        "ground_truth": gt,
        "finish_reason": finish_reason,
        "num_generated_tokens": gen_tokens,
        "model": model,
        "quant_method": quant,
    }


def _fdp(idx, fdp_idx, boxed_match, cosmetic_skipped=0, model="m", quant="q"):
    return {
        "problem_idx": idx,
        "fdp_token_idx": fdp_idx,
        "boxed_match": boxed_match,
        "cosmetic_skipped": cosmetic_skipped,
        "model": model,
        "quant_method": quant,
    }


def _judgment(idx, category, confidence, model="m", quant="q"):
    return {
        "problem_idx": idx,
        "category": category,
        "confidence": confidence,
        "model": model,
        "quant_method": quant,
    }


def test_compute_accuracy_counts_matching_boxed_answers():
    traces = [_trace(0, "42", "42"), _trace(1, "7", "8"), _trace(2, " 5 ", "5")]
    assert compute_accuracy(traces) == pytest.approx(2 / 3)


def test_compute_accuracy_none_boxed_is_wrong():
    traces = [_trace(0, None, "42")]
    assert compute_accuracy(traces) == 0.0


def test_compute_accuracy_table_groups_by_model_quant():
    traces_by_key = {
        ("m1", "bf16"): [_trace(0, "1", "1"), _trace(1, "2", "3")],
        ("m1", "fp8"): [_trace(0, "1", "1"), _trace(1, "3", "3")],
    }
    rows = compute_accuracy_table(traces_by_key)
    by_key = {(r["model"], r["quant"]): r for r in rows}
    assert by_key[("m1", "bf16")]["accuracy_pct"] == pytest.approx(50.0)
    assert by_key[("m1", "fp8")]["accuracy_pct"] == pytest.approx(100.0)
    assert by_key[("m1", "bf16")]["n"] == 2


def test_finish_reason_table_orders_raw_by_descending_count():
    traces = [_trace(i, "1", "1", finish_reason="length") for i in range(78)] + [
        _trace(i, "1", "1", finish_reason="stop") for i in range(78, 80)
    ]
    rows = compute_finish_reason_table({("m", "q"): traces})
    row = rows[0]
    assert list(row["raw"].keys()) == ["length", "stop"]
    assert row["raw"] == {"length": 78, "stop": 2}


def test_finish_reason_table_rounds_half_to_even_like_python():
    # 6/80 = 7.5% exactly -> Python round() gives 8 (nearest even).
    traces = [_trace(i, "1", "1", finish_reason="stop") for i in range(74)] + [
        _trace(i, "1", "1", finish_reason="length") for i in range(74, 80)
    ]
    row = compute_finish_reason_table({("m", "q"): traces})[0]
    assert row["pct_length"] == 8
    assert row["pct_stop"] == 92


def test_token_efficiency_table_computes_delta_vs_bf16_baseline():
    traces_by_key = {
        ("m", "bf16"): [_trace(0, "1", "1", gen_tokens=100), _trace(1, "1", "1", gen_tokens=200)],
        ("m", "fp8"): [_trace(0, "1", "1", gen_tokens=180), _trace(1, "1", "1", gen_tokens=220)],
    }
    rows = compute_token_efficiency_table(traces_by_key)
    by_key = {(r["model"], r["quant"]): r for r in rows}
    assert by_key[("m", "bf16")]["avg_gen_tokens"] == 150
    assert by_key[("m", "bf16")]["delta"] is None
    fp8 = by_key[("m", "fp8")]
    assert fp8["avg_gen_tokens"] == 200
    assert fp8["delta"] == 50
    assert fp8["delta_pct"] == pytest.approx(33.333, abs=0.01)


def test_divergence_position_table_uses_per_problem_baseline_length():
    traces_by_key = {
        ("m", "bf16"): [_trace(0, "1", "1", gen_tokens=100), _trace(1, "1", "1", gen_tokens=200)],
    }
    fdps_by_key = {
        ("m", "q"): [_fdp(0, 50, "both_wrong"), _fdp(1, 100, "both_wrong")],
    }
    rows = compute_divergence_position_table(fdps_by_key, traces_by_key)
    row = rows[0]
    assert row["n"] == 2
    assert row["mean"] == pytest.approx((0.5 + 0.5) / 2)


def test_divergence_position_table_skips_null_fdp():
    traces_by_key = {("m", "bf16"): [_trace(0, "1", "1", gen_tokens=100)]}
    fdps_by_key = {("m", "q"): [_fdp(0, None, "no_boxed")]}
    rows = compute_divergence_position_table(fdps_by_key, traces_by_key)
    assert rows[0]["n"] == 0


def test_fdp_rate_table_counts_boxed_match_categories():
    fdps = [
        _fdp(0, 10, "both_correct"),
        _fdp(1, 20, "baseline_only"),
        _fdp(2, None, "no_boxed"),
        _fdp(3, 30, "quant_only", cosmetic_skipped=2),
    ]
    row = compute_fdp_rate_table({("m", "q"): fdps})[0]
    assert row["n_pairs"] == 4
    assert row["diverged"] == 3
    assert row["cosmetic_skipped_count"] == 1
    assert row["both_correct"] == 1
    assert row["baseline_only"] == 1
    assert row["quant_only"] == 1
    assert row["no_boxed"] == 1
    assert row["both_wrong"] == 0


def test_judge_confidence_table_aggregates_by_category():
    judgments = [
        _judgment(0, "F", 0.9),
        _judgment(1, "F", 1.0),
        _judgment(2, "F", 0.4),
        _judgment(3, "A", 0.8),
    ]
    rows = compute_judge_confidence_table(judgments)
    by_cat = {r["category"]: r for r in rows}
    assert by_cat["F"]["n"] == 3
    assert by_cat["F"]["mean_conf"] == pytest.approx(0.7667, abs=0.001)
    assert by_cat["F"]["n_below_0.5"] == 1
    assert by_cat["A"]["n"] == 1


def test_per_model_chi2_uses_natural_dof_not_nominal():
    # Only 2 of 6 categories ever appear -> natural dof (2-1)*(2-1)=1,
    # not the nominal (2-1)*(6-1)=5.
    judgments = (
        [_judgment(i, "A", 0.8, model="m1", quant="q1") for i in range(10)]
        + [_judgment(i, "F", 0.9, model="m1", quant="q1") for i in range(10, 15)]
        + [_judgment(i, "A", 0.8, model="m1", quant="q2") for i in range(15, 18)]
        + [_judgment(i, "F", 0.9, model="m1", quant="q2") for i in range(18, 30)]
    )
    result = compute_per_model_chi2(judgments)
    assert result["m1"]["dof"] == 1
    assert result["m1"]["n"] == 30
    assert result["m1"]["effect_size"] in {
        "negligible",
        "small-to-moderate",
        "moderate-to-large",
        "large",
    }


def test_quant_only_deepdives_filters_and_joins_fdp_and_judgment():
    fdps_by_key = {
        ("m", "q"): [
            _fdp(0, 10, "quant_only"),
            _fdp(1, 20, "both_wrong"),
        ],
    }
    judgments_by_key = {
        ("m", "q"): [
            _judgment(0, "C", 0.85),
            _judgment(1, "F", 0.99),
        ],
    }
    cases = compute_quant_only_deepdives(fdps_by_key, judgments_by_key)
    assert len(cases) == 1
    assert cases[0]["problem_idx"] == 0
    assert cases[0]["category"] == "C"
    assert cases[0]["confidence"] == 0.85


# ---------------------------------------------------------------------------
# Real-data regression tests — verified against the actual committed
# research/kv-cache-reasoning-divergence-study/tables/*.md ground truth.
# ---------------------------------------------------------------------------


def test_load_records_by_key_real_traces(study_data_dir):
    traces_by_key = load_records_by_key(study_data_dir / "traces")
    row = traces_by_key[("deepseek-r1-distill-qwen-1.5b", "bf16")]
    assert len(row) == 80


def test_accuracy_table_matches_headline_numbers(study_data_dir):
    traces_by_key = load_records_by_key(study_data_dir / "traces")
    rows = compute_accuracy_table(traces_by_key)
    by_key = {(r["model"], r["quant"]): r for r in rows}
    # From supervisor_report.md §5.1
    assert by_key[("deepseek-r1-distill-qwen-1.5b", "bf16")]["accuracy_pct"] == pytest.approx(
        35.0, abs=0.1
    )
    assert by_key[("deepseek-r1-distill-qwen-1.5b", "fp8_e4m3")][
        "accuracy_pct"
    ] == pytest.approx(0.0, abs=0.1)
    assert by_key[("qwen3-1.7b", "bf16")]["accuracy_pct"] == pytest.approx(52.5, abs=0.1)
    assert by_key[("qwen3-1.7b", "fp8_e4m3")]["accuracy_pct"] == pytest.approx(53.7, abs=0.15)
    assert by_key[("deepseek-r1-distill-qwen-7b", "bf16")]["accuracy_pct"] == pytest.approx(
        60.0, abs=0.1
    )


def test_finish_reason_table_matches_ground_truth(study_data_dir):
    traces_by_key = load_records_by_key(study_data_dir / "traces")
    rows = compute_finish_reason_table(traces_by_key)
    by_key = {(r["model"], r["quant"]): r for r in rows}
    row = by_key[("deepseek-r1-distill-qwen-1.5b", "bf16")]
    assert row["raw"] == {"stop": 42, "length": 38}
    assert row["pct_stop"] == 52
    assert row["pct_length"] == 48
    row7b = by_key[("deepseek-r1-distill-qwen-7b", "fp8_e4m3")]
    assert row7b["raw"] == {"length": 74, "stop": 6}
    assert row7b["pct_length"] == 92


def test_token_efficiency_table_matches_ground_truth(study_data_dir):
    traces_by_key = load_records_by_key(study_data_dir / "traces")
    rows = compute_token_efficiency_table(traces_by_key)
    by_key = {(r["model"], r["quant"]): r for r in rows}
    row = by_key[("deepseek-r1-distill-qwen-1.5b", "bf16")]
    assert row["avg_gen_tokens"] == 17077
    row_e4m3 = by_key[("deepseek-r1-distill-qwen-1.5b", "fp8_e4m3")]
    assert row_e4m3["avg_gen_tokens"] == 16180
    assert row_e4m3["delta"] == -896
    row_7b_e4m3 = by_key[("deepseek-r1-distill-qwen-7b", "fp8_e4m3")]
    assert row_7b_e4m3["avg_gen_tokens"] == 15960
    assert row_7b_e4m3["delta_pct"] == pytest.approx(166.1, abs=0.1)


def test_divergence_position_table_matches_ground_truth(study_data_dir):
    traces_by_key = load_records_by_key(study_data_dir / "traces")
    fdps_by_key = load_records_by_key(study_data_dir / "fdps")
    rows = compute_divergence_position_table(fdps_by_key, traces_by_key)
    by_key = {(r["model"], r["quant"]): r for r in rows}
    row = by_key[("qwen3-1.7b", "fp8_e4m3")]
    assert row["n"] == 80
    assert row["median"] == pytest.approx(0.07, abs=0.01)
    assert row["mean"] == pytest.approx(0.16, abs=0.01)
    assert row["max"] == pytest.approx(0.78, abs=0.01)


def test_fdp_rate_table_matches_ground_truth(study_data_dir):
    fdps_by_key = load_records_by_key(study_data_dir / "fdps")
    rows = compute_fdp_rate_table(fdps_by_key)
    by_key = {(r["model"], r["quant"]): r for r in rows}
    row = by_key[("deepseek-r1-distill-qwen-1.5b", "fp8_e4m3")]
    assert row["n_pairs"] == 80
    assert row["diverged"] == 80
    assert row["cosmetic_skipped_count"] == 4
    assert row["both_correct"] == 0
    assert row["baseline_only"] == 28
    assert row["quant_only"] == 0
    assert row["both_wrong"] == 15
    assert row["no_boxed"] == 37


def test_judge_confidence_table_matches_ground_truth(study_data_dir):
    judgments = load_all_records(study_data_dir / "judgments")
    rows = compute_judge_confidence_table(judgments)
    by_cat = {r["category"]: r for r in rows}
    assert by_cat["F"]["n"] == 412
    assert by_cat["F"]["mean_conf"] == pytest.approx(0.95, abs=0.01)
    assert by_cat["A"]["n"] == 21
    assert by_cat["A"]["mean_conf"] == pytest.approx(0.80, abs=0.01)


def test_per_model_chi2_matches_ground_truth(study_data_dir):
    judgments = load_all_records(study_data_dir / "judgments")
    result = compute_per_model_chi2(judgments)
    ds15 = result["deepseek-r1-distill-qwen-1.5b"]
    assert ds15["dof"] == 9
    assert ds15["chi2"] == pytest.approx(64.33, abs=0.05)
    assert ds15["cramers_v"] == pytest.approx(0.259, abs=0.002)
    assert ds15["n"] == 320
    qwen3 = result["qwen3-1.7b"]
    assert qwen3["dof"] == 5
    assert qwen3["p"] == pytest.approx(0.348, abs=0.002)


def test_quant_only_deepdives_matches_ground_truth_count(study_data_dir):
    fdps_by_key = load_records_by_key(study_data_dir / "fdps")
    judgments_by_key = load_records_by_key(study_data_dir / "judgments")
    cases = compute_quant_only_deepdives(fdps_by_key, judgments_by_key)
    # supervisor_report.md §5.3: 5 quant_only cases, all on qwen3-1.7b.
    assert len(cases) == 5
    assert {c["model"] for c in cases} == {"qwen3-1.7b"}
