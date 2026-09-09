"""
Core mathematical primitives for Long-CoT KV cache quantization error analysis.

Implements:
- Subtractive-dithered uniform quantization (rigorous Hanson-Wright assumption)
- Query-covariance eigenspace computation (anchor subspace)
- Anchor score: ||V_r^T k|| / ||k||
- Synthetic Long-CoT generative process (mixture-of-r-Gaussians)
- Attention distributions and TV distance
- Quantization-propagation operator M and its spectral radius

All operations pure NumPy, CPU, no external deps beyond numpy/scipy.
"""
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. Quantization primitives
# ---------------------------------------------------------------------------

def quant_step(range_val: float, b: int) -> float:
    """Quantization step Delta = range / (2^b - 1). Matches dithered quant theory."""
    return 2.0 * range_val / (2 ** b - 1)


def sigma_sq_from_bits(range_val: float, b: int) -> float:
    """sigma^2(b) = Delta^2 / 12 for subtractive-dithered uniform quantizer.

    This is the exact per-entry variance of the quantization error e = x_hat - x,
    where e ~ Uniform(-Delta/2, Delta/2), mean zero, independent of x.
    """
    delta = quant_step(range_val, b)
    return delta ** 2 / 12.0


def subtractive_dither_quantize(x: np.ndarray, b: int, rng: np.random.Generator,
                                 range_val: float = None) -> np.ndarray:
    """Subtractive-dithered uniform quantizer.

    Produces error e = x_hat - x which is:
      - zero-mean
      - uniformly distributed on (-Delta/2, Delta/2)
      - independent of x (Schuchman 1964, Gray-Stockham 1993)
      - i.i.d. across entries

    This is the quantization model assumed in the main theorem.
    """
    if range_val is None:
        range_val = float(np.max(np.abs(x)))
        if range_val < 1e-12:
            return x.copy()

    delta = quant_step(range_val, b)
    qmax = 2 ** b - 1
    u = (rng.uniform(size=x.shape) - 0.5) * delta         # dither ~ U(-Delta/2, Delta/2)
    x_int = np.round((x + u) / delta)
    x_int = np.clip(x_int, -(qmax // 2), qmax // 2)
    x_q = x_int * delta - u                                # subtract dither back
    return x_q


# ---------------------------------------------------------------------------
# 2. Anchor subspace and anchor score
# ---------------------------------------------------------------------------

def query_covariance_eigenbasis(Q: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-r eigenbasis V_r of the empirical query covariance C_Q = (1/T) Q^T Q.

    Returns (V_r, eigenvalues) where V_r is d x r orthonormal.
    Computed via SVD of Q for numerical stability (faster when T << d^2).
    """
    # SVD: Q = U S V^T, so Q^T Q = V S^2 V^T, eigenvectors of C_Q are columns of V
    _, S, Vt = np.linalg.svd(Q, full_matrices=False)
    T = Q.shape[0]
    eigenvalues = (S ** 2) / T
    V_r = Vt[:r, :].T   # d x r
    return V_r, eigenvalues[:r]


def anchor_score(k: np.ndarray, V_r: np.ndarray) -> np.ndarray:
    """Anchor score for key vectors: ||V_r^T k|| / ||k||.

    k: either (d,) for a single key or (N, d) for a batch.
    V_r: (d, r) orthonormal basis.
    Returns: scalar or (N,) of scores in [0, 1].
    """
    if k.ndim == 1:
        proj = V_r.T @ k
        norm_k = np.linalg.norm(k)
        return np.linalg.norm(proj) / (norm_k + 1e-12)
    else:
        proj = k @ V_r                             # (N, r)
        norm_proj = np.linalg.norm(proj, axis=1)
        norm_k = np.linalg.norm(k, axis=1)
        return norm_proj / (norm_k + 1e-12)


# ---------------------------------------------------------------------------
# 3. Synthetic Long-CoT generator (mixture of r Gaussians aligned with V_r)
# ---------------------------------------------------------------------------

def sample_cot_queries(T: int, d: int, r: int, rng: np.random.Generator,
                       signal: float = 1.0, noise: float = 0.1,
                       V_r: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sample T queries from a mixture of r Gaussians aligned with V_r directions.

    This simulates the architectural anchor-subspace structure:
      q_t = v_{c(t)} * signal + eta_t * noise,  where c(t) ~ Uniform{1..r}
    Anchor tokens (drawn near the v_j's) dominate top-r eigenbasis of C_Q.

    Returns (Q, component_labels) where Q is (T, d) and labels in {0..r-1}.
    """
    if V_r is None:
        # Generate a random orthonormal basis via QR
        G = rng.standard_normal((d, d))
        Q_qr, _ = np.linalg.qr(G)
        V_r = Q_qr[:, :r]

    comp = rng.integers(0, r, size=T)                # component assignment
    means = V_r[:, comp].T                            # (T, d)
    noise_mat = rng.standard_normal((T, d)) * noise
    Q = signal * means + noise_mat
    return Q, comp


def generate_kv_from_queries(Q: np.ndarray, rng: np.random.Generator,
                             W_K: np.ndarray = None, W_V: np.ndarray = None,
                             noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Generate K, V caches from queries.

    K = Q @ W_K + small_noise,  V = random unit vectors.
    If W_K / W_V are None, uses identity + rotation so anchors in Q map to anchors in K.
    """
    T, d = Q.shape
    if W_K is None:
        # Orthogonal rotation: preserves anchor subspace structure
        G = rng.standard_normal((d, d))
        W_K, _ = np.linalg.qr(G)
    K = Q @ W_K + rng.standard_normal((T, d)) * noise

    if W_V is None:
        V_raw = rng.standard_normal((T, d))
        V = V_raw / (np.linalg.norm(V_raw, axis=1, keepdims=True) + 1e-12)
    else:
        V = Q @ W_V
    return K, V


# ---------------------------------------------------------------------------
# 4. Attention and distances
# ---------------------------------------------------------------------------

def softmax_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax."""
    X = X - X.max(axis=-1, keepdims=True)
    ex = np.exp(X)
    return ex / (ex.sum(axis=-1, keepdims=True) + 1e-30)


def attention_distribution(q: np.ndarray, K: np.ndarray, d: int) -> np.ndarray:
    """Attention distribution p(i) = softmax(q^T K / sqrt(d)) for one query."""
    logits = (K @ q) / np.sqrt(d)
    return softmax_rows(logits[None, :])[0]


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance = 0.5 * L1."""
    return 0.5 * float(np.abs(p - q).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q)."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float((p_safe * np.log(p_safe / q_safe)).sum())


# ---------------------------------------------------------------------------
# 5. Spectral quantities (the core architectural constants)
# ---------------------------------------------------------------------------

def propagation_operator_norm(K: np.ndarray, V: np.ndarray, W_Q: np.ndarray,
                               p_bar: np.ndarray, d: int) -> float:
    """Operator norm of the per-step attention-propagation operator M_t.

    M_t h = (1/sqrt(d)) * sum_i p_t(i) * v_i * (q_t^T h) + J_other
    For the leading-term analysis:  ||M_t|| ~ (1/sqrt(d)) * ||W_Q|| * sum_i p_t(i) * ||v_i||.
    """
    contrib = np.sum(p_bar[:, None] * V, axis=0)        # (d,)
    return float(np.linalg.norm(contrib) * np.linalg.norm(W_Q, ord=2) / np.sqrt(d))


def top_lyapunov_estimate(Ms: list, n_iters: int = 50,
                          rng: np.random.Generator = None) -> float:
    """Estimate the top Lyapunov exponent via QR-based product-of-matrices iteration.

    lambda_1 = lim_T (1/T) log || M_T ... M_1 ||
    Uses QR normalization to avoid overflow (standard Benettin et al. algorithm).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = Ms[0].shape[0]
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    log_norm_sum = 0.0
    n = 0
    for _ in range(n_iters):
        for M in Ms:
            v = M @ v
            nrm = np.linalg.norm(v)
            if nrm < 1e-300:
                return -np.inf
            log_norm_sum += np.log(nrm)
            v /= nrm
            n += 1
    return log_norm_sum / n


# ---------------------------------------------------------------------------
# 6. Autoregressive CoT with and without quantization
# ---------------------------------------------------------------------------

def simulate_cot_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                           b: int, rng: np.random.Generator,
                           quantize_anchors: bool = True,
                           anchor_mask: np.ndarray = None,
                           b_anchor: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """For each step t, compute p_t^full and p_t^q under KV quantization.

    Returns (P_full, P_q) as lists padded to T x T (upper-triangular invalid).
    When `quantize_anchors` is False, anchor tokens are quantized at b_anchor bits.
    """
    T, d = Q.shape
    P_full = np.zeros((T, T))
    P_q = np.zeros((T, T))

    K_q = K.copy()
    if quantize_anchors or anchor_mask is None:
        # Quantize everything uniformly at b bits
        K_q = subtractive_dither_quantize(K, b, rng)
    else:
        # Mixed precision: anchors at b_anchor, rest at b
        anchor_idx = np.where(anchor_mask)[0]
        filler_idx = np.where(~anchor_mask)[0]
        if len(filler_idx) > 0:
            K_q[filler_idx] = subtractive_dither_quantize(K[filler_idx], b, rng)
        if len(anchor_idx) > 0:
            K_q[anchor_idx] = subtractive_dither_quantize(K[anchor_idx], b_anchor, rng)

    for t in range(T):
        q_t = Q[t]
        p_full = attention_distribution(q_t, K[: t + 1], d)
        p_q    = attention_distribution(q_t, K_q[: t + 1], d)
        P_full[t, : t + 1] = p_full
        P_q[t,    : t + 1] = p_q
    return P_full, P_q
