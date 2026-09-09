"""
Test Suite 1 — Basic invariants (Section 4.5 of the roadmap).

Four invariants that every building block of the theory must satisfy:
  INV-1: Delta^2/12 noise model (fake-quant variance == theoretical)
  INV-2: Anchor score is rotation-invariant under basis change
  INV-3: b -> infinity limit recovers zero error
  INV-4: T=1 single-step Hanson-Wright concentration

These must pass before we trust any of the downstream theorem tests.
"""
import numpy as np
from kv_math import (
    subtractive_dither_quantize, sigma_sq_from_bits, quant_step,
    anchor_score, query_covariance_eigenbasis,
)
from test_runner import test, run_all, approx


# ---------------------------------------------------------------------------
# INV-1: Delta^2/12 invariant
# ---------------------------------------------------------------------------

@test
def test_inv1_quant_noise_variance_matches_theory():
    """Empirical MSE of subtractive-dithered quantization matches Delta^2/12.

    Important: the Schuchman-Gray-Stockham result requires that x + dither
    does not exceed the quantizer range. We sample x in [-0.8, 0.8] with
    range_val=1.0 so that x + U(-Delta/2, Delta/2) stays within bounds for
    all b >= 2 (since Delta/2 <= 1/(2^b - 1) <= 1/3 at b=2, so x_max + Delta/2 <= 1.13
    and we use a slightly safer margin of 0.7 at b=2).
    """
    rng = np.random.default_rng(42)
    N = 200_000
    range_val = 1.0

    # At b=2 with Delta = 2/3, dither extends x by +- 1/3; x in [-0.6, 0.6] keeps
    # x+dither within +-0.93, safely inside quantizer range.
    signal_ranges = {2: 0.6, 3: 0.75, 4: 0.85, 5: 0.9, 6: 0.9, 8: 0.95}

    for b in [2, 3, 4, 5, 6, 8]:
        sr = signal_ranges[b]
        x = rng.uniform(-sr, sr, size=N)
        x_q = subtractive_dither_quantize(x, b, rng, range_val=range_val)
        err = x_q - x
        empirical_var = float(np.var(err))
        theoretical_var = sigma_sq_from_bits(range_val, b)
        rel_err = abs(empirical_var - theoretical_var) / theoretical_var
        assert rel_err < 0.05, (
            f"b={b}: empirical var {empirical_var:.5g} vs "
            f"theory {theoretical_var:.5g} (rel err {rel_err:.2%})"
        )


@test
def test_inv1_quant_noise_mean_is_zero():
    """Quantization error is zero-mean (critical for sub-Gaussian assumption)."""
    rng = np.random.default_rng(123)
    N = 500_000
    x = rng.standard_normal(N) * 0.5
    for b in [3, 4, 8]:
        range_val = 2.0
        x_q = subtractive_dither_quantize(x, b, rng, range_val=range_val)
        err = x_q - x
        # std of sample mean ~ sigma / sqrt(N)
        sigma = np.sqrt(sigma_sq_from_bits(range_val, b))
        se = sigma / np.sqrt(N)
        assert abs(err.mean()) < 4 * se, (
            f"b={b}: mean={err.mean():.3e}, 4*SE={4*se:.3e}"
        )


@test
def test_inv1_quant_noise_independence_of_signal():
    """Subtractive dither: error is independent of x (Schuchman 1964)."""
    rng = np.random.default_rng(7)
    N = 100_000
    x = rng.standard_normal(N) * 0.3
    b = 4
    range_val = 2.0
    x_q = subtractive_dither_quantize(x, b, rng, range_val=range_val)
    err = x_q - x
    # Pearson correlation between x and err should be ~0
    corr = float(np.corrcoef(x, err)[0, 1])
    assert abs(corr) < 0.02, f"|corr(x, err)| = {abs(corr):.4f}, expected ~0"


# ---------------------------------------------------------------------------
# INV-2: Anchor score rotation invariance (V_r U has same score as V_r)
# ---------------------------------------------------------------------------

@test
def test_inv2_anchor_score_rotation_invariant():
    """Anchor score is a property of the subspace, not the basis."""
    rng = np.random.default_rng(2024)
    d, r = 128, 16
    # Random orthonormal V_r in R^d
    G = rng.standard_normal((d, d))
    V_full, _ = np.linalg.qr(G)
    V_r = V_full[:, :r]

    # Rotate inside subspace: V_r' = V_r @ U, where U is r x r orthogonal
    U, _ = np.linalg.qr(rng.standard_normal((r, r)))
    V_r_rot = V_r @ U

    # Sample keys
    k = rng.standard_normal((50, d))
    s1 = anchor_score(k, V_r)
    s2 = anchor_score(k, V_r_rot)
    max_diff = float(np.max(np.abs(s1 - s2)))
    assert max_diff < 1e-10, f"max diff {max_diff:.3e}, not rotation invariant"


@test
def test_inv2_anchor_score_in_unit_interval():
    """Score is in [0, 1] since V_r^T projection has norm <= ||k||."""
    rng = np.random.default_rng(99)
    d, r = 64, 8
    G, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_r = G[:, :r]
    k = rng.standard_normal((200, d))
    s = anchor_score(k, V_r)
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-10, (
        f"score range: [{s.min():.4f}, {s.max():.4f}]"
    )


@test
def test_inv2_anchor_in_subspace_has_score_one():
    """Keys lying entirely in span(V_r) should have anchor_score == 1."""
    rng = np.random.default_rng(5)
    d, r = 32, 4
    G, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_r = G[:, :r]
    # Keys = V_r @ (random r-dim coefficients), i.e. live in span(V_r)
    coeffs = rng.standard_normal((100, r))
    k = coeffs @ V_r.T
    s = anchor_score(k, V_r)
    assert np.allclose(s, 1.0, atol=1e-10), f"scores = {s[:5]}..., not all 1.0"


@test
def test_inv2_anchor_orthogonal_to_subspace_has_score_zero():
    """Keys orthogonal to V_r should have anchor_score == 0."""
    rng = np.random.default_rng(6)
    d, r = 32, 4
    G, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_r = G[:, :r]
    V_perp = G[:, r:]                     # orthogonal complement
    coeffs = rng.standard_normal((100, d - r))
    k = coeffs @ V_perp.T
    s = anchor_score(k, V_r)
    assert np.allclose(s, 0.0, atol=1e-10), f"scores = {s[:5]}..., not all 0.0"


# ---------------------------------------------------------------------------
# INV-3: b -> infinity recovers zero error
# ---------------------------------------------------------------------------

@test
def test_inv3_large_bits_small_error():
    """At high bit-width, max absolute error drops below 2/(2^b - 1)."""
    rng = np.random.default_rng(31)
    N = 50_000
    x = rng.uniform(-1.0, 1.0, size=N)
    for b in [12, 14, 16]:
        x_q = subtractive_dither_quantize(x, b, rng, range_val=1.0)
        err = x_q - x
        max_err = float(np.max(np.abs(err)))
        theoretical_max = quant_step(1.0, b)           # Delta
        assert max_err < theoretical_max + 1e-10, (
            f"b={b}: max_err={max_err:.3e} > Delta={theoretical_max:.3e}"
        )


@test
def test_inv3_mse_decays_as_4_neg_b():
    """MSE scales as 4^-b (since Delta ~ 2^-b, MSE ~ Delta^2)."""
    rng = np.random.default_rng(17)
    N = 100_000
    x = rng.uniform(-1.0, 1.0, size=N)
    mses = []
    for b in [4, 6, 8, 10]:
        x_q = subtractive_dither_quantize(x, b, rng, range_val=1.0)
        mses.append(float(np.mean((x_q - x) ** 2)))
    # Log ratios should be ~-2*log(2) per bit
    log_mses = np.log(mses)
    slope = np.polyfit([4, 6, 8, 10], log_mses, 1)[0]
    expected = -2 * np.log(2)
    rel_err = abs(slope - expected) / abs(expected)
    assert rel_err < 0.05, (
        f"slope={slope:.4f}, expected={expected:.4f}, rel_err={rel_err:.2%}"
    )


# ---------------------------------------------------------------------------
# INV-4: T=1 single-step Hanson-Wright concentration
# ---------------------------------------------------------------------------

@test
def test_inv4_single_step_hanson_wright():
    """At T=1, E[(q^T e)^2] = sigma^2 ||q||^2.

    This is the atomic per-step bound from Hanson-Wright (linear form).
    """
    rng = np.random.default_rng(1000)
    d = 128
    b = 4
    range_val = 1.0
    sigma2 = sigma_sq_from_bits(range_val, b)
    n_trials = 15_000

    # Fixed query
    q = rng.standard_normal(d)
    norm_q_sq = float(np.sum(q ** 2))

    dot_products = np.empty(n_trials)
    for i in range(n_trials):
        e = subtractive_dither_quantize(np.zeros(d), b, rng, range_val=range_val)
        # e is quant error for x=0, which is just -dither. E[e^2] = Delta^2/12.
        dot_products[i] = float(q @ e)
    empirical = float(np.mean(dot_products ** 2))
    theoretical = sigma2 * norm_q_sq
    rel_err = abs(empirical - theoretical) / theoretical
    # Tight tolerance: 15k trials * 128 dims gives SE ~ 1/sqrt(15k) ~ 0.8%
    assert rel_err < 0.05, (
        f"E[(q^T e)^2]: empirical={empirical:.5g}, theory={theoretical:.5g}, "
        f"rel_err={rel_err:.2%}"
    )


@test
def test_inv4_quadratic_form_hanson_wright():
    """E[||e||^2] = d * sigma^2 (Hanson-Wright quadratic form expectation)."""
    rng = np.random.default_rng(2000)
    d = 256
    b = 4
    range_val = 1.0
    n_trials = 5_000
    sigma2 = sigma_sq_from_bits(range_val, b)

    norms = np.empty(n_trials)
    for i in range(n_trials):
        e = subtractive_dither_quantize(np.zeros(d), b, rng, range_val=range_val)
        norms[i] = float(e @ e)
    empirical = float(norms.mean())
    theoretical = d * sigma2
    rel_err = abs(empirical - theoretical) / theoretical
    assert rel_err < 0.02, (
        f"E[||e||^2]: empirical={empirical:.5g}, theory={theoretical:.5g}, "
        f"rel_err={rel_err:.2%}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Test Suite 1: Basic invariants")
    print("=" * 70)
    n_fail = run_all()
    exit(n_fail)
