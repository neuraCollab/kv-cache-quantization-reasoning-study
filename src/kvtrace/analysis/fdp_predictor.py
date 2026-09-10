"""Analytical / lightweight-ML First-Divergence-Point (FDP) prediction from
data already on disk — no K-matrices, no GPU, no captured internals.

This is a deliberately different experiment from the mechanistic-analysis
report's own FDP predictor (report Part 7: a 1D-CNN over raw K-matrices,
R²=0.98 in-distribution). That approach needs real per-layer K tensors from
a forward pass; we don't have those for the actual study models (no GPU
here), and re-deriving them isn't in scope (see repo root README's Phase 6
notes). Instead: can *text/metadata features of the already-generated bf16
trace* — repetition rate, length, finish_reason, correctness — explain any
of the variance in where the quantized trace first diverges?

The report's own "prompt-only" experiment (features from the prompt/prefill
only, no decode signal) got R²<0 — worse than predicting the mean — and
concluded FDP is an emergent property of decoding, not derivable from static
features. Our features are a *different* kind of "static" (whole-trace
properties available after generation, not just the prompt), so this is a
genuine follow-up question, not a repeat of their failed experiment — but
expect it to land somewhere between "prompt-only" and the real K-matrix
model, if it works at all.

No judge-derived fields (category/confidence/rationale/affected_span) are
used as features — those are *outputs* of classifying the FDP itself, so
using them to predict the FDP's position would be circular.
"""
from __future__ import annotations

import random
import re
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import spearmanr

_WORD_RE = re.compile(r"\S+")


def ngram_repetition_score(text: str, n: int = 4) -> float:
    """Fraction of word n-grams that are exact repeats of an earlier n-gram
    in the same text. A cheap, GPU-free proxy for report Category F
    (Repetition/loop): a trace stuck re-emitting the same phrase scores high.
    """
    words = _WORD_RE.findall(text)
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for g in ngrams:
        if g in seen:
            repeats += 1
        else:
            seen.add(g)
    return repeats / len(ngrams)


def build_feature_row(fdp: dict, baseline_trace: dict) -> dict | None:
    """One row of (features, target) from a real FDP record + its baseline
    (bf16) trace. Returns None for rows with no usable target (no divergence,
    or a zero-length baseline trace makes normalized position undefined).

    Deliberately excludes judge fields (category/confidence/...) — see
    module docstring.
    """
    fdp_token_idx = fdp.get("fdp_token_idx")
    n_tokens = baseline_trace.get("num_generated_tokens") or 0
    if fdp_token_idx is None or n_tokens <= 0:
        return None

    raw = baseline_trace.get("raw_output", "") or ""
    boxed = (baseline_trace.get("boxed_answer") or "").strip()
    ground_truth = (baseline_trace.get("ground_truth") or "").strip()

    return {
        "problem_idx": fdp.get("problem_idx"),
        "model": fdp.get("model"),
        "quant_method": fdp.get("quant_method"),
        "source": baseline_trace.get("source"),
        "fdp_token_idx": fdp_token_idx,
        "fdp_position": fdp_token_idx / n_tokens,
        "num_generated_tokens": n_tokens,
        "num_prompt_tokens": baseline_trace.get("num_prompt_tokens") or 0,
        "finish_reason_length": baseline_trace.get("finish_reason") == "length",
        "think_complete": bool(baseline_trace.get("think_complete")),
        "baseline_correct": boxed == ground_truth and boxed != "",
        "char_length": len(raw),
        "repetition_score_4gram": ngram_repetition_score(raw, n=4),
        "repetition_score_8gram": ngram_repetition_score(raw, n=8),
    }


def build_feature_table(
    fdps_by_key: dict[tuple[str, str], list[dict]],
    traces_by_key: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    """Join every (model, quant) pair's FDP records with their bf16 baseline
    trace (keyed by `problem_idx`/`idx`), skipping rows `build_feature_row`
    rejects. `traces_by_key` must contain a `(model, "bf16")` entry for every
    model referenced in `fdps_by_key`.
    """
    baseline_by_model: dict[str, dict[int, dict]] = {}
    for (model, quant), rows in traces_by_key.items():
        if quant == "bf16":
            baseline_by_model[model] = {r["idx"]: r for r in rows}

    out: list[dict] = []
    for (model, _quant), fdp_rows in fdps_by_key.items():
        baseline = baseline_by_model.get(model, {})
        for fdp in fdp_rows:
            problem_idx = fdp.get("problem_idx")
            trace = baseline.get(problem_idx) if problem_idx is not None else None
            if trace is None:
                continue
            row = build_feature_row(fdp, trace)
            if row is not None:
                out.append(row)
    return out


def _kfold_indices(n: int, n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Deterministic K-fold train/test index splits (no sklearn dependency —
    this module only needs KFold's index bookkeeping, not its full API)."""
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    folds = [indices[i::n_splits] for i in range(n_splits)]
    splits = []
    for i, test in enumerate(folds):
        train = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        splits.append((train, test))
    return splits


def cross_validate_spearman(
    X: Any,
    y: Any,
    model_factory: Callable[[], Any],
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """K-fold CV for a small (dozens-of-rows) regression problem, reporting
    R²/MAE/Spearman per fold and averaged.

    `model_factory()` must return a fresh object with `.fit(X, y)` and
    `.predict(X)` — any duck-typed regressor works, including sklearn's
    (this module doesn't import sklearn itself; pass one in from the caller
    if you have it installed).

    Spearman is the headline number, not R² — on the small, often heavy-
    tailed FDP-position targets here, R² computed per tiny test fold is
    numerically unstable (a single outlier in a 16-row fold can send it to
    -1000), the same pathology the mechanistic-analysis report ran into on
    real K-matrix features for DeepSeek (R²=-3.9 there, Spearman=0.94).
    `spearman_mean` is nan-safe: folds where predictions or targets are
    constant (undefined correlation) are dropped rather than poisoning the
    average; if *every* fold is degenerate the average is 0.0, not NaN.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_splits = min(n_splits, n)

    r2s: list[float] = []
    maes: list[float] = []
    sps: list[float] = []
    for train_idx, test_idx in _kfold_indices(n, n_splits, seed):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[test_idx]), dtype=float)
        y_test = y[test_idx]

        ss_res = float(np.sum((y_test - pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        maes.append(float(np.mean(np.abs(y_test - pred))))

        if len(set(pred.tolist())) > 1 and len(set(y_test.tolist())) > 1:
            sp = spearmanr(y_test, pred).correlation
            if sp == sp:  # not NaN
                sps.append(float(sp))

    return {
        "r2_mean": float(np.nanmean(r2s)) if r2s else float("nan"),
        "mae_mean": float(np.mean(maes)) if maes else float("nan"),
        "spearman_mean": float(np.mean(sps)) if sps else 0.0,
    }
