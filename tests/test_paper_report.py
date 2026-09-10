from __future__ import annotations

from kvtrace.analysis.paper import (
    compute_fdp_rate_table,
    compute_finish_reason_table,
    compute_judge_confidence_table,
    compute_per_model_chi2,
    compute_quant_only_deepdives,
    compute_token_efficiency_table,
)
from kvtrace.analysis.paper_report import (
    render_fdp_rate_table,
    render_finish_reason_table,
    render_judge_confidence_table,
    render_per_model_chi2,
    render_quant_only_deepdives,
    render_token_efficiency_table,
    write_accuracy_bars_plot,
    write_divergence_position_plot,
)


def test_render_finish_reason_table_contains_row_and_raw_dict():
    rows = compute_finish_reason_table(
        {
            ("m1", "bf16"): [
                {"finish_reason": "stop"},
                {"finish_reason": "stop"},
                {"finish_reason": "length"},
            ]
        }
    )
    text = render_finish_reason_table(rows)
    assert "| model | quant |" in text
    assert "m1" in text and "bf16" in text
    assert "67%" in text  # 2/3 stop


def test_render_token_efficiency_table_shows_dash_for_baseline():
    rows = compute_token_efficiency_table(
        {
            ("m1", "bf16"): [{"num_generated_tokens": 100, "finish_reason": "stop"}],
            ("m1", "fp8"): [{"num_generated_tokens": 150, "finish_reason": "stop"}],
        }
    )
    text = render_token_efficiency_table(rows)
    assert "—" in text
    assert "+50" in text


def test_render_fdp_rate_table_has_quant_only_column():
    rows = compute_fdp_rate_table(
        {
            ("m1", "q"): [
                {"problem_idx": 0, "fdp_token_idx": 1, "boxed_match": "quant_only", "cosmetic_skipped": 0}
            ]
        }
    )
    text = render_fdp_rate_table(rows)
    assert "quant_only" in text


def test_render_judge_confidence_table_has_all_columns():
    rows = compute_judge_confidence_table(
        [{"category": "F", "confidence": 0.9}, {"category": "F", "confidence": 0.95}]
    )
    text = render_judge_confidence_table(rows)
    assert "n_below_0.5" in text
    assert "F" in text


def test_render_per_model_chi2_includes_observed_and_residuals():
    judgments = [
        {"model": "m1", "quant_method": "q1", "category": "A"},
        {"model": "m1", "quant_method": "q1", "category": "F"},
        {"model": "m1", "quant_method": "q2", "category": "A"},
        {"model": "m1", "quant_method": "q2", "category": "F"},
    ]
    result = compute_per_model_chi2(judgments)
    text = render_per_model_chi2(result)
    assert "chi²" in text or "chi2" in text
    assert "Standardized residuals" in text
    assert "q1" in text


def test_render_quant_only_deepdives_lists_each_case():
    cases = compute_quant_only_deepdives(
        {("m1", "q1"): [{"problem_idx": 5, "boxed_match": "quant_only", "problem": "2+2?",
                          "ground_truth": "4", "fdp_token_idx": 3}]},
        {("m1", "q1"): [{"problem_idx": 5, "category": "A", "confidence": 0.9,
                          "rationale": "arith slip", "affected_span": "oops"}]},
    )
    text = render_quant_only_deepdives(cases)
    assert "Case 1" in text
    assert "m1" in text and "q1" in text
    assert "2+2?" in text


def test_write_accuracy_bars_plot_creates_file(tmp_path):
    rows = [
        {"model": "m1", "quant": "bf16", "accuracy_pct": 50.0},
        {"model": "m1", "quant": "fp8", "accuracy_pct": 40.0},
    ]
    out = tmp_path / "accuracy_bars.png"
    write_accuracy_bars_plot(rows, out)
    assert out.exists() and out.stat().st_size > 0


def test_write_divergence_position_plot_creates_file(tmp_path):
    rows = [
        {"model": "m1", "quant": "fp8", "median": 0.1, "mean": 0.2, "std": 0.1,
         "min": 0.0, "max": 0.5, "n": 10},
    ]
    out = tmp_path / "divergence_position.png"
    write_divergence_position_plot(rows, out)
    assert out.exists() and out.stat().st_size > 0
