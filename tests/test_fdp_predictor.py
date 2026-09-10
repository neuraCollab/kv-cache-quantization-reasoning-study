from __future__ import annotations

import pytest

from kvtrace.analysis.fdp_predictor import (
    build_feature_row,
    cross_validate_spearman,
    ngram_repetition_score,
)

# ---------------------------------------------------------------------------
# ngram_repetition_score — analytical proxy for "loop"-iness (report Category F)
# ---------------------------------------------------------------------------


def test_ngram_repetition_score_zero_for_no_repeats():
    text = "one two three four five six seven eight nine ten"
    assert ngram_repetition_score(text, n=4) == pytest.approx(0.0)


def test_ngram_repetition_score_high_for_obvious_loop():
    text = "let me try x equals two " * 10
    assert ngram_repetition_score(text, n=4) > 0.7


def test_ngram_repetition_score_partial_repetition():
    # "a b c d" repeated once among otherwise-unique 4-grams
    text = "a b c d e f g h a b c d i j k l"
    score = ngram_repetition_score(text, n=4)
    assert 0.0 < score < 1.0


def test_ngram_repetition_score_handles_short_text():
    assert ngram_repetition_score("only two words", n=4) == pytest.approx(0.0)
    assert ngram_repetition_score("", n=4) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_feature_row — joins one fdp record with its baseline trace, no leakage
# ---------------------------------------------------------------------------


def _trace(**overrides):
    base = {
        "idx": 0,
        "raw_output": "Let me think. The answer is \\boxed{4}.",
        "boxed_answer": "4",
        "ground_truth": "4",
        "finish_reason": "stop",
        "think_complete": True,
        "num_generated_tokens": 120,
        "num_prompt_tokens": 30,
        "source": "aime-24",
    }
    base.update(overrides)
    return base


def _fdp(**overrides):
    base = {
        "problem_idx": 0,
        "model": "qwen3-1.7b",
        "quant_method": "fp8_e4m3",
        "fdp_token_idx": 60,
    }
    base.update(overrides)
    return base


def test_build_feature_row_includes_target_and_no_judge_fields():
    row = build_feature_row(_fdp(), _trace())
    assert row["fdp_token_idx"] == 60
    assert row["fdp_position"] == pytest.approx(60 / 120)
    assert row["model"] == "qwen3-1.7b"
    assert row["quant_method"] == "fp8_e4m3"
    # must not leak judge-derived fields (category/confidence/rationale come
    # from judging *the FDP itself* — using them to predict FDP is circular)
    assert "category" not in row
    assert "confidence" not in row
    assert "rationale" not in row


def test_build_feature_row_computes_baseline_correctness():
    correct = build_feature_row(_fdp(), _trace(boxed_answer="4", ground_truth="4"))
    wrong = build_feature_row(_fdp(), _trace(boxed_answer="5", ground_truth="4"))
    assert correct["baseline_correct"] is True
    assert wrong["baseline_correct"] is False


def test_build_feature_row_returns_none_for_missing_fdp():
    assert build_feature_row(_fdp(fdp_token_idx=None), _trace()) is None


def test_build_feature_row_returns_none_for_zero_length_trace():
    assert build_feature_row(_fdp(), _trace(num_generated_tokens=0)) is None


# ---------------------------------------------------------------------------
# cross_validate_spearman — small-sample-safe K-fold CV wrapper
# ---------------------------------------------------------------------------


class _MeanModel:
    """A model that always predicts the training-fold mean — the same
    baseline predictor used in the report and in our own experiment."""

    def fit(self, X, y):
        self._mean = sum(y) / len(y)
        return self

    def predict(self, X):
        return [self._mean] * len(X)


class _LinearModel:
    """y = x (identity) — a model that recovers a perfect linear signal."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [row[0] for row in X]


def test_cross_validate_spearman_perfect_signal_scores_high():
    n = 30
    X = [[i] for i in range(n)]
    y = [float(i) for i in range(n)]
    result = cross_validate_spearman(X, y, model_factory=lambda: _LinearModel(), n_splits=5, seed=0)
    assert result["spearman_mean"] > 0.9


def test_cross_validate_spearman_constant_predictions_are_nan_safe():
    n = 30
    X = [[i] for i in range(n)]
    y = [float(i % 3) for i in range(n)]
    # A constant-output model can't be rank-correlated with anything; the
    # wrapper must not raise or return NaN as the aggregate.
    result = cross_validate_spearman(X, y, model_factory=lambda: _MeanModel(), n_splits=5, seed=0)
    assert result["spearman_mean"] == result["spearman_mean"]  # not NaN
    assert "r2_mean" in result and "mae_mean" in result
