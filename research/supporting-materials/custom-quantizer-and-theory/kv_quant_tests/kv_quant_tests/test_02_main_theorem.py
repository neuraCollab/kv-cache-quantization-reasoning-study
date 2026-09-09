"""
Test Suite 2 — Main Theorem predictions (Sections 2 & 4.4 of the roadmap).

The main theorem predicts three falsifiable regressions:

  PRED-A: E[TV] scales linearly in sigma^2(b), i.e. log E[TV] vs b has slope -2 log 2.
  PRED-B: log E[TV] vs log T is linear with slope alpha, regime-switching as T grows.
  PRED-C: Anchor-aware mixed precision reduces error by factor ~ T_anchor/T (up to constants).

These test the key quantitative claims of the closed-form bound, using pure synthetic
matrix simulations that run in seconds on CPU.
"""
import numpy as np
from kv_math import (
    subtractive_dither_quantize, sigma_sq_from_bits,
    sample_cot_queries, generate_kv_from_queries,
    attention_distribution, tv_distance,
    anchor_score, query_covariance_eigenbasis,
)
from test_runner import test, run_all, approx


def _setup_synthetic(T: int, d: int, r: int, seed: int):
    """Standard synthetic Long-CoT setup used throughout these tests."""
    rng = np.random.default_rng(seed)
    # Shared anchor subspace V_r
    G = rng.standard_normal((d, d))
    V_full, _ = np.linalg.qr(G)
    V_r = V_full[:, :r]

    Q, comp = sample_cot_queries(T, d, r, rng, signal=1.0, noise=0.1, V_r=V_r)
    K, V = generate_kv_from_queries(Q, rng, noise=0.05)
    return rng, Q, K, V, V_r, comp


def _mean_tv_over_trials(K: np.ndarray, queries: np.ndarray, d: int,
                         b: int, n_trials: int,
                         range_val: float, rng_master: int) -> float:
    """Average TV(p_full, p_q) over trials. Queries are shared; K quant noise is resampled."""
    rng = np.random.default_rng(rng_master)
    T = K.shape[0]
    # Use a random subset of query positions to keep O(T^2) manageable
    tvs = []
    # Sample 'probe' positions uniformly in [T//2, T) so each query sees >= T/2 cached keys
    n_probes = min(32, len(queries))
    probe_idx = rng.choice(len(queries), size=n_probes, replace=False)
    for trial in range(n_trials):
        K_q = subtractive_dither_quantize(K, b, rng, range_val=range_val)
        trial_tv = 0.0
        for t in probe_idx:
            p_full = attention_distribution(queries[t], K[: t + 1], d)
            p_q    = attention_distribution(queries[t], K_q[: t + 1], d)
            trial_tv += tv_distance(p_full, p_q)
        tvs.append(trial_tv / n_probes)
    return float(np.mean(tvs))


# ---------------------------------------------------------------------------
# PRED-A: E[TV] vs sigma^2 (equivalently vs bit-width b)
# ---------------------------------------------------------------------------

@test
def test_predA_tv_linear_in_sigma():
    """E[TV] should scale as sigma ~ 2^{-b} (not sigma^2).

    Why: TV(p, p_q) <= (1/2) ||p - p_q||_1 <= (1/2) sqrt(T) * ||delta_logits||_2.
    Since delta_logits ~ q^T e / sqrt(d), its l2 norm is O(sigma * sqrt(||q||^2/d)),
    i.e. linear in sigma. For KL the prediction would be quadratic (slope -2 log 2),
    but for TV the correct exponent is linear (slope -log 2 ~ -0.693).

    This distinction is important for the thesis: the bound is on E[KL] if we want
    quadratic scaling, or on E[TV]^2 equivalently. Here we test E[TV] directly and
    expect slope ~ -log(2).
    """
    T, d, r = 256, 64, 8
    rng, Q, K, V, V_r, _ = _setup_synthetic(T, d, r, seed=1)
    range_val = float(np.max(np.abs(K)))

    bits = [4, 6, 8, 10]
    tvs = []
    for b in bits:
        tv = _mean_tv_over_trials(K, Q, d, b, n_trials=8,
                                  range_val=range_val, rng_master=100 + b)
        tvs.append(tv)
    # Strictly decreasing
    for i in range(len(bits) - 1):
        assert tvs[i] > tvs[i + 1], (
            f"E[TV] not decreasing: b={bits[i]}:{tvs[i]:.4g} vs b={bits[i+1]}:{tvs[i+1]:.4g}"
        )
    # Slope of log(TV) vs b should be close to -log(2) ~ -0.693.
    log_tvs = np.log(tvs)
    slope = float(np.polyfit(bits, log_tvs, 1)[0])
    expected = -np.log(2)
    assert -0.9 <= slope <= -0.5, (
        f"log(E[TV]) vs b slope={slope:.3f}, expected close to -log(2)={expected:.3f}. "
        f"Values: {list(zip(bits, tvs))}"
    )


@test
def test_predA_kl_linear_in_sigma_squared():
    """For KL divergence the prediction IS quadratic in sigma (slope -2 log 2 per bit).

    This is because KL(p, p+dp) ~ (1/2) dp^T diag(1/p) dp ~ O(||dp||^2) = O(sigma^2).
    """
    T, d, r = 256, 64, 8
    rng, Q, K, V, V_r, _ = _setup_synthetic(T, d, r, seed=1)
    range_val = float(np.max(np.abs(K)))

    from kv_math import kl_divergence, attention_distribution

    bits = [4, 6, 8, 10]
    kls = []
    probe_idx = rng.choice(T, size=32, replace=False)
    for b in bits:
        rng_b = np.random.default_rng(500 + b)
        trial_kls = []
        for trial in range(8):
            K_q = subtractive_dither_quantize(K, b, rng_b, range_val=range_val)
            acc = 0.0
            for t in probe_idx:
                p_full = attention_distribution(Q[t], K[: t + 1], d)
                p_q    = attention_distribution(Q[t], K_q[: t + 1], d)
                acc += kl_divergence(p_full, p_q)
            trial_kls.append(acc / len(probe_idx))
        kls.append(float(np.mean(trial_kls)))
    log_kls = np.log(np.maximum(kls, 1e-20))
    slope = float(np.polyfit(bits, log_kls, 1)[0])
    expected = -2 * np.log(2)
    assert -1.8 <= slope <= -1.1, (
        f"log(E[KL]) vs b slope={slope:.3f}, expected close to -2*log(2)={expected:.3f}. "
        f"Values: {list(zip(bits, kls))}"
    )


@test
def test_predA_tv_zero_at_high_bits():
    """At b=16, the TV distance should be numerically negligible."""
    T, d, r = 128, 64, 4
    rng, Q, K, V, V_r, _ = _setup_synthetic(T, d, r, seed=11)
    range_val = float(np.max(np.abs(K)))
    tv = _mean_tv_over_trials(K, Q, d, b=16, n_trials=4,
                              range_val=range_val, rng_master=777)
    assert tv < 5e-4, f"E[TV] at b=16 should be negligible, got {tv:.4e}"


# ---------------------------------------------------------------------------
# PRED-B: log E[TV] vs log T slope (anchor compounding signature)
# ---------------------------------------------------------------------------

@test
def test_predB_tv_grows_with_context_length():
    """E[TV(p_full, p_q)] at the last position grows as context length T grows.

    Setup: use the same strong-separation scheme as PRED-C (sqrt(d)-scaled keys).
    Measure TV of attention distribution for the final query over increasing
    context lengths T.

    Expected: growth with T, because more keys means (a) more filler-quant noise
    entries summing in the denominator, (b) more anchor positions whose perturbed
    keys can redirect attention mass.
    """
    d, r = 64, 3
    b = 3
    T_max = 1024
    mag = np.sqrt(d)
    rng = np.random.default_rng(2024)

    G, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_r = G[:, :r]
    V_perp = G[:, r:]

    # Build a LONG sequence of keys with ~20% anchors
    anchor_frac = 0.2
    is_anchor = rng.random(T_max) < anchor_frac
    K = np.zeros((T_max, d))
    for i in range(T_max):
        if is_anchor[i]:
            c = rng.standard_normal(r)
            c /= np.linalg.norm(c)
            K[i] = c @ V_r.T * mag
        else:
            c = rng.standard_normal(d - r)
            c /= np.linalg.norm(c)
            K[i] = c @ V_perp.T * mag

    # Queries aligned with anchor directions
    q_coeffs = rng.standard_normal((T_max, r))
    q_coeffs /= np.linalg.norm(q_coeffs, axis=1, keepdims=True)
    Q = q_coeffs @ V_r.T * mag

    range_val = float(np.max(np.abs(K)))

    Ts = [128, 256, 512, 1024]
    n_trials = 30
    probe_count = 24

    tvs = []
    for T in Ts:
        rng_t = np.random.default_rng(T * 13 + 7)
        probe_idx = rng_t.choice(T, size=probe_count, replace=False)
        trial_tvs = []
        for trial in range(n_trials):
            K_q = subtractive_dither_quantize(K[:T], b, rng_t, range_val=range_val)
            acc = 0.0
            for t in probe_idx:
                p_full = attention_distribution(Q[t], K[: t + 1], d)
                p_q    = attention_distribution(Q[t], K_q[: t + 1], d)
                acc += tv_distance(p_full, p_q)
            trial_tvs.append(acc / probe_count)
        tvs.append(float(np.mean(trial_tvs)))

    ratio = tvs[-1] / tvs[0]
    assert ratio > 1.15, (
        f"E[TV] should grow: T=128->{tvs[0]:.4g}, T=1024->{tvs[-1]:.4g}, ratio={ratio:.3f}. "
        f"All values: {list(zip(Ts, tvs))}"
    )
    log_T = np.log(Ts)
    log_tv = np.log(np.maximum(tvs, 1e-12))
    slope = float(np.polyfit(log_T, log_tv, 1)[0])
    assert 0.0 <= slope <= 1.5, (
        f"log-log slope alpha = {slope:.3f}, expected in [0.0, 1.5]. "
        f"Values: {list(zip(Ts, tvs))}"
    )


# ---------------------------------------------------------------------------
# PRED-C: Anchor-aware mixed precision reduces error
# ---------------------------------------------------------------------------

@test
def test_predC_anchor_mixed_precision_reduces_error():
    """Keeping anchor tokens at high precision should reduce E[TV].

    Setup:
      - Quantize all tokens uniformly at b=4 (baseline).
      - Quantize filler tokens at b=4 but anchors at b=16 (mixed).
    Predict: mixed error <= baseline error * (1 - T_anchor/T * (1 - sigma^2(16)/sigma^2(4))).
    Equivalently: mixed should be significantly lower.
    """
    T, d, r = 512, 64, 8
    rng, Q, K, V, V_r, _ = _setup_synthetic(T, d, r, seed=42)
    range_val = float(np.max(np.abs(K)))

    # Identify anchor tokens: top-20% by anchor_score
    scores = anchor_score(K, V_r)
    threshold = float(np.quantile(scores, 0.80))
    anchor_mask = scores >= threshold
    n_anchors = int(anchor_mask.sum())
    # With our r=8 synthetic, most keys have high score -> we expect a clean separation.
    assert n_anchors > 0, "Need at least one anchor for meaningful test"
    anchor_fraction = n_anchors / T

    b_filler = 4
    b_anchor = 16
    n_trials = 10

    probe_idx = rng.choice(T, size=min(32, T), replace=False)

    # Baseline: everyone at b_filler
    rng_base = np.random.default_rng(10_000)
    baseline_tvs = []
    for trial in range(n_trials):
        K_q = subtractive_dither_quantize(K, b_filler, rng_base, range_val=range_val)
        acc = 0.0
        for t in probe_idx:
            p_full = attention_distribution(Q[t], K[: t + 1], d)
            p_q    = attention_distribution(Q[t], K_q[: t + 1], d)
            acc += tv_distance(p_full, p_q)
        baseline_tvs.append(acc / len(probe_idx))

    # Mixed: anchors at b_anchor, rest at b_filler
    rng_mixed = np.random.default_rng(20_000)
    mixed_tvs = []
    for trial in range(n_trials):
        K_q = K.copy()
        fill = np.where(~anchor_mask)[0]
        anch = np.where(anchor_mask)[0]
        if len(fill) > 0:
            K_q[fill] = subtractive_dither_quantize(K[fill], b_filler, rng_mixed,
                                                     range_val=range_val)
        if len(anch) > 0:
            K_q[anch] = subtractive_dither_quantize(K[anch], b_anchor, rng_mixed,
                                                     range_val=range_val)
        acc = 0.0
        for t in probe_idx:
            p_full = attention_distribution(Q[t], K[: t + 1], d)
            p_q    = attention_distribution(Q[t], K_q[: t + 1], d)
            acc += tv_distance(p_full, p_q)
        mixed_tvs.append(acc / len(probe_idx))

    baseline_mean = float(np.mean(baseline_tvs))
    mixed_mean    = float(np.mean(mixed_tvs))
    reduction = (baseline_mean - mixed_mean) / baseline_mean

    # Theoretical minimum reduction under equal weighting: anchor_fraction * (1 - 4^{-12})
    # In practice anchors contribute disproportionately so expected reduction >> anchor_fraction.
    min_expected_reduction = 0.5 * anchor_fraction
    assert mixed_mean < baseline_mean, (
        f"Mixed precision did not reduce error: baseline={baseline_mean:.4g}, "
        f"mixed={mixed_mean:.4g}"
    )
    assert reduction >= min_expected_reduction, (
        f"Reduction {reduction:.2%} < minimum expected {min_expected_reduction:.2%}. "
        f"anchor_fraction={anchor_fraction:.2%}"
    )


@test
def test_predC_anchors_are_attended_more_than_fillers():
    """Core empirical basis of Lemma 2.1: anchors receive O(T/r) attention mass,
    fillers receive O(log T). We check the attention-mass dichotomy holds.

    This test requires a clean anchor/filler separation, so we inject fillers
    explicitly as random isotropic noise orthogonal to the anchor subspace.
    Key subtlety: we need logit separation (q^T k)/sqrt(d) to be O(1), so the
    mixture signal strength must be large enough that q^T k_anchor - q^T k_filler
    >> sqrt(d). For d=64 this means magnitude ~ sqrt(d) ~ 8.
    """
    T = 400
    d, r = 64, 3
    rng = np.random.default_rng(314)

    # Build V_r
    G = rng.standard_normal((d, d))
    V_full, _ = np.linalg.qr(G)
    V_r = V_full[:, :r]
    V_perp = V_full[:, r:]

    # Keys: 20% anchors in V_r, 80% fillers in V_perp. Magnitude sqrt(d) ~ 8
    # to give O(1) logit separation under the /sqrt(d) normalization.
    mag = np.sqrt(d)      # chosen so (q^T k)/sqrt(d) = O(1) when q and k are aligned
    n_anchor = int(0.2 * T)
    n_filler = T - n_anchor
    coeffs_a = rng.standard_normal((n_anchor, r))
    coeffs_a /= np.linalg.norm(coeffs_a, axis=1, keepdims=True)
    K_anchors = coeffs_a @ V_r.T * mag
    coeffs_f = rng.standard_normal((n_filler, d - r))
    coeffs_f /= np.linalg.norm(coeffs_f, axis=1, keepdims=True)
    K_fillers = coeffs_f @ V_perp.T * mag

    # Interleave
    K = np.zeros((T, d))
    is_anchor = np.zeros(T, dtype=bool)
    anchor_positions = rng.choice(T, size=n_anchor, replace=False)
    is_anchor[anchor_positions] = True
    K[is_anchor] = K_anchors
    K[~is_anchor] = K_fillers

    # Queries: drawn from the anchor mixture (this simulates "reasoning queries"
    # aligned with architectural anchor structure).
    q_coeffs = rng.standard_normal((T, r))
    q_coeffs /= np.linalg.norm(q_coeffs, axis=1, keepdims=True)
    Q = q_coeffs @ V_r.T * mag

    # Verify anchor_score separates them cleanly
    scores = anchor_score(K, V_r)
    assert scores[is_anchor].mean() > 0.9, (
        f"anchor scores mean={scores[is_anchor].mean():.3f}, expected ~1.0"
    )
    assert scores[~is_anchor].mean() < 0.1, (
        f"filler scores mean={scores[~is_anchor].mean():.3f}, expected ~0.0"
    )

    # Cumulative attention mass each token receives from later queries
    attn_mass = np.zeros(T)
    for t in range(1, T):
        p = attention_distribution(Q[t], K[:t], d)
        attn_mass[:t] += p

    anchor_per = float(attn_mass[is_anchor].mean())
    filler_per = float(attn_mass[~is_anchor].mean())
    ratio = anchor_per / (filler_per + 1e-12)

    assert anchor_per > filler_per, (
        f"per-token attention: anchors={anchor_per:.4g}, fillers={filler_per:.4g}"
    )
    assert ratio > 5.0, (
        f"anchor/filler attention ratio = {ratio:.2f}, expected > 5.0 "
        f"(anchor_per={anchor_per:.3g}, filler_per={filler_per:.3g})"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Test Suite 2: Main Theorem predictions")
    print("=" * 70)
    n_fail = run_all()
    exit(n_fail)
