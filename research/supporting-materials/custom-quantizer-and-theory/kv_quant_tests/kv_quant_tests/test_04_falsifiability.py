"""
Test Suite 4 — Falsifiability Guards.

These tests deliberately construct scenarios where the theorems MUST NOT hold,
to ensure our test fixtures actually discriminate. If all of these pass, we have
evidence that the test suites in 01-03 are testing real effects, not artifacts.

Each test here either:
  (a) Flips a precondition of a theorem and verifies the conclusion FAILS, OR
  (b) Verifies the converse of a theorem under a controlled counter-example.
"""
import numpy as np
from kv_math import (
    subtractive_dither_quantize, sigma_sq_from_bits, quant_step,
    attention_distribution, tv_distance,
    anchor_score, query_covariance_eigenbasis,
)
from test_runner import test, run_all


# ---------------------------------------------------------------------------
# Guard for INV: does our quantizer really add noise?
# ---------------------------------------------------------------------------

@test
def test_falsify_no_quantization_gives_zero_error():
    """Sanity: passing b=inf (or not quantizing) must give zero TV error.
    If this test fails, the pipeline has a bug unrelated to quantization."""
    rng = np.random.default_rng(0)
    T, d = 50, 32
    K = rng.standard_normal((T, d))
    q = rng.standard_normal(d)
    p_a = attention_distribution(q, K, d)
    p_b = attention_distribution(q, K, d)     # same K, same q
    tv = tv_distance(p_a, p_b)
    assert tv < 1e-15, f"Zero-noise TV should be exactly 0, got {tv}"


@test
def test_falsify_tv_zero_for_identical_distributions():
    """Sanity: TV(p, p) = 0 for any distribution p."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        p = rng.random(100)
        p /= p.sum()
        assert tv_distance(p, p) < 1e-15


# ---------------------------------------------------------------------------
# Guard for PRED-A: bits matter
# ---------------------------------------------------------------------------

@test
def test_falsify_fewer_bits_gives_more_error():
    """Sanity: E[TV] at b=2 must be strictly greater than at b=8.

    If this fails, it means our quantization is ineffective or the softmax is
    saturating everything — either way, tests 02 are measuring nothing.
    """
    rng = np.random.default_rng(42)
    T, d, r = 128, 64, 4
    mag = np.sqrt(d)
    G, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_r = G[:, :r]

    # Strong anchor structure
    coeffs = rng.standard_normal((T, r))
    coeffs /= np.linalg.norm(coeffs, axis=1, keepdims=True)
    K = coeffs @ V_r.T * mag
    q = rng.standard_normal(r); q /= np.linalg.norm(q)
    q = q @ V_r.T * mag

    range_val = float(np.max(np.abs(K)))
    p_full = attention_distribution(q, K, d)

    def mean_tv_at_b(b, n=50):
        rng_b = np.random.default_rng(b * 99)
        tvs = []
        for _ in range(n):
            K_q = subtractive_dither_quantize(K, b, rng_b, range_val=range_val)
            p_q = attention_distribution(q, K_q, d)
            tvs.append(tv_distance(p_full, p_q))
        return float(np.mean(tvs))

    tv_2 = mean_tv_at_b(2)
    tv_8 = mean_tv_at_b(8)
    ratio = tv_2 / max(tv_8, 1e-10)
    assert ratio > 10, (
        f"TV at b=2 ({tv_2:.4g}) should be >>10x TV at b=8 ({tv_8:.4g}); ratio={ratio:.1f}x"
    )


# ---------------------------------------------------------------------------
# Guard for PRED-C: anchor mixed precision is NOT always a win
# ---------------------------------------------------------------------------

@test
def test_falsify_mixed_precision_useless_when_no_anchor_structure():
    """If there is NO anchor structure (keys uniformly random), keeping some
    random subset at higher precision should give only marginal reduction
    proportional to the subset fraction.

    This ensures our Theorem I claim (anchor subspace matters) is non-trivial.
    """
    T, d = 400, 64
    rng = np.random.default_rng(777)

    # Isotropic random keys (NO anchor structure)
    K = rng.standard_normal((T, d))
    q = rng.standard_normal(d)
    range_val = float(np.max(np.abs(K)))
    p_full = attention_distribution(q, K, d)

    b_low, b_high = 4, 16
    # Randomly select 20% of keys as "fake anchors"
    fake_anchor_mask = np.zeros(T, dtype=bool)
    fake_anchor_mask[rng.choice(T, size=int(0.2 * T), replace=False)] = True

    def mean_err_mixed(mask):
        rng_m = np.random.default_rng(2222)
        tvs = []
        for _ in range(30):
            K_q = K.copy()
            a_idx = np.where(mask)[0]
            f_idx = np.where(~mask)[0]
            if len(f_idx) > 0:
                K_q[f_idx] = subtractive_dither_quantize(K[f_idx], b_low, rng_m, range_val=range_val)
            if len(a_idx) > 0:
                K_q[a_idx] = subtractive_dither_quantize(K[a_idx], b_high, rng_m, range_val=range_val)
            p_q = attention_distribution(q, K_q, d)
            tvs.append(tv_distance(p_full, p_q))
        return float(np.mean(tvs))

    def mean_err_uniform():
        rng_u = np.random.default_rng(3333)
        tvs = []
        for _ in range(30):
            K_q = subtractive_dither_quantize(K, b_low, rng_u, range_val=range_val)
            p_q = attention_distribution(q, K_q, d)
            tvs.append(tv_distance(p_full, p_q))
        return float(np.mean(tvs))

    err_mixed  = mean_err_mixed(fake_anchor_mask)
    err_uniform = mean_err_uniform()
    # Without anchor structure, random 20% protection should only reduce error by ~20%,
    # far less than the 70%+ we see in the real anchor case.
    reduction = (err_uniform - err_mixed) / err_uniform
    assert reduction < 0.5, (
        f"Without anchor structure, mixed precision shouldn't give dramatic benefit: "
        f"reduction={reduction:.2%}, uniform={err_uniform:.4g}, mixed={err_mixed:.4g}. "
        f"A reduction >> 20% would suggest our test is too permissive."
    )


# ---------------------------------------------------------------------------
# Guard for failure Theorem A: eviction is fine when anchors aren't evicted
# ---------------------------------------------------------------------------

@test
def test_falsify_eviction_ok_when_budget_covers_anchors():
    """Theorem A predicts catastrophic error when anchors are evicted. The
    converse: when the eviction budget is large enough to keep ALL anchors,
    error should be small. This validates that Theorem A is about anchor LOSS,
    not about eviction per se.
    """
    T = 200
    d, r = 64, 3
    mag = np.sqrt(d)
    rng = np.random.default_rng(2024)
    G, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_r = G[:, :r]
    V_perp = G[:, r:]

    n_anchor = 20
    is_anchor = np.zeros(T, dtype=bool)
    is_anchor[rng.choice(T, size=n_anchor, replace=False)] = True

    K = np.zeros((T, d))
    V_cache = np.zeros((T, d))
    for i in range(T):
        if is_anchor[i]:
            c = rng.standard_normal(r); c /= np.linalg.norm(c)
            K[i] = (c @ V_r.T) * mag
            V_cache[i] = c @ V_r.T
        else:
            c = rng.standard_normal(d - r); c /= np.linalg.norm(c)
            K[i] = (c @ V_perp.T) * mag
            V_cache[i] = rng.standard_normal(d) * 0.1

    # Eviction budget >> n_anchor means anchors are safe (keep all of them + some fillers)
    keep_budget = n_anchor + 50    # generous budget

    # Build mask keeping all anchors + 50 random fillers
    keep_mask = is_anchor.copy()
    filler_idx = np.where(~is_anchor)[0]
    additional = rng.choice(filler_idx, size=50, replace=False)
    keep_mask[additional] = True

    # Test attention with this eviction at a query aligned to V_r
    q_coeffs = rng.standard_normal(r); q_coeffs /= np.linalg.norm(q_coeffs)
    q = (q_coeffs @ V_r.T) * mag

    p_full = attention_distribution(q, K, d)
    logits = K @ q / np.sqrt(d)
    logits[~keep_mask] = -1e12
    p_evict = np.exp(logits - logits.max()); p_evict /= p_evict.sum()

    o_full  = p_full  @ V_cache
    o_evict = p_evict @ V_cache
    err = float(np.linalg.norm(o_full - o_evict))

    # With all anchors kept, error should be TINY
    assert err < 0.1, (
        f"Error with all anchors kept: {err:.4g}. Theorem A should NOT fire here. "
        f"This confirms the theorem's specificity: it fires only on anchor LOSS."
    )


# ---------------------------------------------------------------------------
# Guard for failure Theorem B: clustering is fine when cal distribution matches
# ---------------------------------------------------------------------------

@test
def test_falsify_clustering_fine_when_cal_matches_online():
    """Theorem B says offline clustering fails under distribution shift. Converse:
    if P_cal == P_test exactly, clustering should work well. We verify this by
    running both cal and test queries from the same distribution.
    """
    T = 300
    d, r = 32, 3
    mag = np.sqrt(d)
    rng = np.random.default_rng(4040)
    G, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_r = G[:, :r]

    K = np.zeros((T, d))
    for i in range(T):
        c = rng.standard_normal(r); c /= np.linalg.norm(c)
        K[i] = (c @ V_r.T) * mag + rng.standard_normal(d) * 0.1

    # Simple k-means (inlined to avoid cross-suite test registration)
    def _kmeans_simple(X, K, n_iter=20, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        N, dd = X.shape
        idx = rng.choice(N, size=K, replace=False)
        C = X[idx].copy()
        for _ in range(n_iter):
            dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    C[k] = X[mask].mean(axis=0)
        return C
    n_clusters = 8
    centroids = _kmeans_simple(K, K=n_clusters, rng=rng)
    dists = np.linalg.norm(K[:, None, :] - centroids[None, :, :], axis=2)
    key_cluster = dists.argmin(axis=1)

    # Both cal and test queries are SAME distribution: mixture across ALL directions
    def make_queries(n, seed):
        rng_q = np.random.default_rng(seed)
        c = rng_q.standard_normal((n, r)); c /= np.linalg.norm(c, axis=1, keepdims=True)
        return c @ V_r.T * mag + rng_q.standard_normal((n, d)) * 0.2

    Q_cal = make_queries(100, 7)
    # Compute cluster hotness from cal
    cluster_hotness = np.zeros(n_clusters)
    for q in Q_cal:
        p = attention_distribution(q, K, d)
        for c_idx in range(n_clusters):
            cluster_hotness[c_idx] += float(p[key_cluster == c_idx].sum())
    cluster_hotness /= len(Q_cal)
    retained = set(np.argsort(-cluster_hotness)[:4].tolist())

    def squeezed(q, top_m=2):
        cent_scores = centroids @ q
        masked = cent_scores.copy()
        for c_idx in range(n_clusters):
            if c_idx not in retained:
                masked[c_idx] = -1e18
        top = np.argsort(-masked)[:top_m]
        keep = np.isin(key_cluster, top)
        logits = K @ q / np.sqrt(d)
        logits[~keep] = -1e12
        ex = np.exp(logits - logits.max())
        return ex / ex.sum()

    # Same-distribution test queries
    Q_test = make_queries(50, 11)
    errs = []
    for q in Q_test:
        p_full = attention_distribution(q, K, d)
        p_sq = squeezed(q)
        errs.append(tv_distance(p_full, p_sq))
    err_same = float(np.mean(errs))

    # When P_cal == P_test, error should be modest — the clustering works as intended.
    # We expect err_same to be around 0.3 at top_m=2/8 (retrieved fraction 25%).
    assert err_same < 0.6, (
        f"When P_cal matches P_test, error should be modest: {err_same:.3f}. "
        f"If this exceeds 0.6, our clustering implementation is broken."
    )


# ---------------------------------------------------------------------------
# Guard for failure Theorem C: when attention contracts, rho(M) < 1 is possible
# ---------------------------------------------------------------------------

@test
def test_falsify_spectral_radius_can_be_small_in_contractive_regime():
    """Theorem C claims NO heuristic contracts. Verify the converse: an
    artificially constructed "contractive" scheme (e.g., scaling the
    attention output by 0.1) DOES give rho(M) < 1. This shows the theorem
    is meaningful — contraction IS possible, it's just that none of the
    baseline heuristics implements it.
    """
    T = 128
    d, r = 32, 2
    mag = np.sqrt(d)
    rng = np.random.default_rng(9999)
    G, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_r = G[:, :r]

    K = np.zeros((T, d))
    V_cache = np.zeros((T, d))
    for i in range(T):
        c = rng.standard_normal(r); c /= np.linalg.norm(c)
        K[i] = (c @ V_r.T) * mag
        V_cache[i] = (c @ V_r.T) * 0.5
    q = rng.standard_normal(r); q /= np.linalg.norm(q)
    q = q @ V_r.T * mag

    p = attention_distribution(q, K, d)
    v_mean = p @ V_cache

    # Unscaled M
    M_full = np.outer(v_mean, q) / np.sqrt(d)
    # Contractive M: scale output by 0.01
    M_contract = 0.01 * M_full
    norm_full = float(np.linalg.norm(M_full, ord=2))
    norm_contract = float(np.linalg.norm(M_contract, ord=2))

    assert norm_contract < 0.1, (
        f"Artificially scaled-down M must have small norm: {norm_contract:.4g}. "
        f"If it doesn't, the Lyapunov tests would be invalid."
    )
    assert norm_full > norm_contract * 50, (
        f"Scaling by 0.01 should reduce norm by ~100x: full={norm_full:.4g}, "
        f"contracted={norm_contract:.4g}, ratio={norm_full/norm_contract:.2f}x"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Test Suite 4: Falsifiability Guards")
    print("=" * 70)
    n_fail = run_all()
    exit(n_fail)
