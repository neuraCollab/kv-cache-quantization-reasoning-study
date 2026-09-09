"""Aggregate judgments into failure-signature matrices and run statistics."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency

CATEGORY_ORDER = ("A", "B", "C", "D", "E", "F")


def aggregate_counts(
    judgments: list[dict],
) -> tuple[list[str], np.ndarray]:
    """Build a [n_methods × 6] matrix of counts.

    Methods are ordered by first appearance in `judgments`.
    Each row sums to the number of judgments for that quant method.
    """
    if not judgments:
        raise ValueError("judgments must be non-empty")

    methods: list[str] = []
    rows: dict[str, np.ndarray] = {}
    for j in judgments:
        m = j["quant_method"]
        c = j["category"]
        if c not in CATEGORY_ORDER:
            continue
        if m not in rows:
            methods.append(m)
            rows[m] = np.zeros(6, dtype=float)
        rows[m][CATEGORY_ORDER.index(c)] += 1

    matrix = np.stack([rows[m] for m in methods], axis=0)
    return methods, matrix


def row_normalize(m: np.ndarray) -> np.ndarray:
    """Per-row normalization; all-zero rows stay zero."""
    out = np.zeros_like(m, dtype=float)
    for i in range(m.shape[0]):
        s = m[i].sum()
        if s > 0:
            out[i] = m[i] / s
    return out


def chi_square_test(m: np.ndarray) -> tuple[float, float, int]:
    """Return (chi2, p_value, dof).

    All-zero columns are dropped before calling chi2_contingency to avoid
    degenerate stats, but the reported `dof` is the nominal (rows-1)*(cols-1)
    of the *original* 6-category matrix so reports stay comparable across runs.
    """
    nonzero_cols = (m.sum(axis=0) > 0)
    m_clean = m[:, nonzero_cols]
    chi2, p, _, _ = chi2_contingency(m_clean)
    dof_full = (m.shape[0] - 1) * (m.shape[1] - 1)
    return float(chi2), float(p), int(dof_full)


def cramers_v(m: np.ndarray) -> float:
    """Cramér's V for a contingency matrix."""
    nonzero_cols = (m.sum(axis=0) > 0)
    m_clean = m[:, nonzero_cols]
    chi2, _, _, _ = chi2_contingency(m_clean)
    n = m_clean.sum()
    if n == 0:
        return 0.0
    k = min(m_clean.shape[0], m_clean.shape[1])
    if k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))


def natural_dof(m: np.ndarray) -> int:
    """Degrees of freedom scipy would report on the all-zero-column-dropped matrix.

    Unlike `chi_square_test`'s `dof` (fixed at the nominal (rows-1)*(cols-1) of
    the *original* 6-category matrix, so it stays comparable across runs where
    different categories happen to be empty), this reflects the actual number
    of categories that appear for this particular slice of data — used for
    per-model chi-square reporting where categories legitimately differ in
    which columns are ever nonzero.
    """
    nonzero_cols = int((m.sum(axis=0) > 0).sum())
    return (m.shape[0] - 1) * (nonzero_cols - 1)


def standardized_residuals(m: np.ndarray) -> np.ndarray:
    """Simple standardized residuals (O-E)/sqrt(E) for each cell.

    E_ij = row_total_i * col_total_j / N. Cells where E_ij == 0 (an all-zero
    column) get residual 0.0 rather than NaN from a 0/0 division.
    """
    n = m.sum()
    row_totals = m.sum(axis=1, keepdims=True)
    col_totals = m.sum(axis=0, keepdims=True)
    expected = row_totals * col_totals / n
    residuals = np.zeros_like(m, dtype=float)
    nonzero = expected > 0
    residuals[nonzero] = (m[nonzero] - expected[nonzero]) / np.sqrt(expected[nonzero])
    return residuals


def effect_size_label(v: float) -> str:
    """Bucket a Cramér's V value into a human-readable effect-size label.

    Fixed thresholds (not Cohen's df-adjusted table): <0.1 negligible,
    <0.3 small-to-moderate, <0.5 moderate-to-large, else large.
    """
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small-to-moderate"
    if v < 0.5:
        return "moderate-to-large"
    return "large"
