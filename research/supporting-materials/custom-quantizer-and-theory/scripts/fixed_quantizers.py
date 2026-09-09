"""
Исправленные квантизаторы для quantization_workbench.ipynb

Формат данных в ноутбуке:
  W: (d_out, d_in)     = (256, 256)
  X: (d_in, n_samples) = (256, 128)   ← строки = каналы, столбцы = примеры
  H = X @ X.T / n      = (d_in, d_in)
  matmul: W @ X         = (d_out, n_samples)

Скопируй содержимое каждого класса в соответствующую ячейку ноутбука.
"""

# ═══════════════════════════════════════════════════════════
# 4.2 GPTQ — ИСПРАВЛЕН
# ═══════════════════════════════════════════════════════════

class GPTQQuantizer(BaseQuantizer):
    """GPTQ: последовательное квантование с коррекцией через Hessian."""
    
    def _compute_hessian(self, X):
        """X: (d_in, n_samples) → H: (d_in, d_in)"""
        return (X @ X.T) / X.shape[1]   # ← X.T и делим на n_samples (dim=1)
    
    def quantize_vector(self, w):
        """Symmetric quantization одного вектора."""
        qmax = 2 ** (self.bits - 1) - 1
        scale = w.abs().max() / qmax
        scale = scale.clamp(min=1e-10)
        q = (w / scale).round().clamp(-qmax, qmax)
        return q * scale
    
    def quantize(self, W, X=None, **kwargs):
        assert X is not None, "GPTQ требует калибровочные данные X"
        
        d_out, d_in = W.shape
        
        H = self._compute_hessian(X)  # (d_in, d_in)
        H_reg = H + 1e-6 * torch.eye(d_in, device=H.device)
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H_reg))
        
        W = W.clone()
        
        for i in range(d_in):
            q_i = self.quantize_vector(W[:, i])
            err_i = W[:, i] - q_i
            W[:, i] = q_i
            
            if i < d_in - 1:
                W[:, i+1:] -= err_i.unsqueeze(1) * (H_inv[i, i+1:] / (H_inv[i, i] + 1e-12)).unsqueeze(0)
        
        return W, {
            "effective_bits": self.bits,
            "hessian_diag": torch.diag(H).cpu().numpy(),
            "hessian_cond": torch.linalg.cond(H).item()
        }


# ═══════════════════════════════════════════════════════════
# 4.3 AWQ — ИСПРАВЛЕН
# ═══════════════════════════════════════════════════════════

class AWQQuantizer(BaseQuantizer):
    """AWQ: защита salient каналов через масштабирование."""
    
    def _fake_quantize(self, w):
        n_bits = self.bits
        qmin = -(2**(n_bits - 1))
        qmax = 2**(n_bits - 1) - 1
        scale = w.abs().max() / qmax
        scale = scale.clamp(min=1e-8)
        W_q = (w / scale).round().clamp(qmin, qmax)
        return W_q * scale
    
    def quantize(self, W, X=None, **kwargs):
        assert X is not None, "AWQ требует калибровочные данные X"
        
        # X: (d_in, n_samples)
        # s_j = mean(|X[j, :]|) — среднее по примерам для каждого канала
        s = X.abs().mean(dim=1)  # (d_in,) ← dim=1 = по столбцам (примерам)
        
        best_loss = float('inf')
        best_scale = None
        
        for alpha in torch.linspace(0, 1, 11):
            channel_scale = s.pow(alpha).clamp(min=1e-8)  # (d_in,)
            
            # W: (d_out, d_in), channel_scale: (d_in,) → broadcast по столбцам
            w_scaled = W * channel_scale.unsqueeze(0)       # (d_out, d_in)
            
            W_q = self._fake_quantize(w_scaled)
            W_deq = W_q / channel_scale.unsqueeze(0)
            
            # loss = ||W @ X - W_deq @ X||²
            # W @ X: (d_out, d_in) @ (d_in, n_samples) = (d_out, n_samples)
            loss = (W @ X - W_deq @ X).pow(2).sum()
            
            if loss < best_loss:
                best_loss = loss
                best_scale = channel_scale.clone()
        
        W_scaled = W * best_scale.unsqueeze(0)
        W_q = self._fake_quantize(W_scaled)
        W_deq = W_q / best_scale.unsqueeze(0)
        
        return W_deq, {
            "effective_bits": self.bits,
            "channel_scale": best_scale.cpu(),
        }


# ═══════════════════════════════════════════════════════════
# 4.4 SmoothQuant — ИСПРАВЛЕН
# ═══════════════════════════════════════════════════════════

class SmoothQuantQuantizer(BaseQuantizer):
    """SmoothQuant: перенос сложности квантования с активаций на веса."""
    
    def __init__(self, bits: int = 8, alpha: float = 0.5, **kwargs):
        super().__init__(bits, **kwargs)
        self.alpha = alpha
    
    def _fake_quantize(self, w):
        n_bits = self.bits
        qmin = -(2**(n_bits - 1))
        qmax = 2**(n_bits - 1) - 1
        scale = w.abs().max() / qmax
        scale = scale.clamp(min=1e-8)
        W_q = (w / scale).round().clamp(qmin, qmax)
        return W_q * scale
    
    def quantize(self, W, X=None, **kwargs):
        assert X is not None, "SmoothQuant требует активации"
        
        # X: (d_in, n_samples)
        # W: (d_out, d_in)
        
        # act_scale[j] = max по примерам для канала j
        act_scale = X.abs().amax(dim=1)   # (d_in,) ← max по столбцам (примерам)
        
        # w_scale[j] = max по выходным нейронам для канала j
        w_scale = W.abs().amax(dim=0)     # (d_in,) ← max по строкам (выходам)
        
        s = act_scale.pow(self.alpha) / w_scale.pow(1 - self.alpha)
        s = s.clamp(min=1e-8)
        
        # Сглаживаем: W столбцы умножаем на s, X строки делим на s
        W_smooth = W * s.unsqueeze(0)       # (d_out, d_in) * (1, d_in)
        X_smooth = X / s.unsqueeze(1)       # (d_in, n_samples) / (d_in, 1)
        
        W_smooth_q = self._fake_quantize(W_smooth)
        X_smooth_q = self._fake_quantize(X_smooth)
        
        # Возвращаем W в исходном масштабе для сравнения
        W_deq = W_smooth_q / s.unsqueeze(0)
        
        return W_deq, {
            "effective_bits": self.bits,
            "smoothing_scale": s.cpu(),
        }


# ═══════════════════════════════════════════════════════════
# 4.5 QuIP — ИСПРАВЛЕН
# ═══════════════════════════════════════════════════════════

class QuIPQuantizer(BaseQuantizer):
    """QuIP: Random rotation + LDLQ квантование."""
    
    def _fake_quantize_vector(self, w):
        n_bits = self.bits
        qmax = 2 ** (n_bits - 1) - 1
        qmin = -(2 ** (n_bits - 1))
        scale = w.abs().max() / qmax
        scale = scale.clamp(min=1e-8)
        q = (w / scale).round().clamp(qmin, qmax)
        return q * scale
    
    def quantize(self, W, X=None, **kwargs):
        assert X is not None, "QuIP требует калибровочные данные"
        
        d_out, d_in = W.shape
        
        # Случайные ортогональные матрицы
        U = random_orthogonal_matrix(d_out, device=W.device)  # (d_out, d_out)
        V = random_orthogonal_matrix(d_in, device=W.device)   # (d_in, d_in)
        
        # Вращаем веса
        W_rot = U @ W @ V                    # (d_out, d_in)
        
        # Hessian: X (d_in, n_samples) → H = X @ X.T / n → (d_in, d_in)
        H = (X @ X.T) / X.shape[1]
        H_rot = V.T @ H @ V                  # (d_in, d_in)
        
        # Обращаем Hessian для GPTQ-коррекции
        H_reg = H_rot + 1e-6 * torch.eye(d_in, device=W.device)
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H_reg))
        
        # Последовательное квантование с компенсацией
        W_rot = W_rot.clone()
        for i in range(d_in):
            q_i = self._fake_quantize_vector(W_rot[:, i])
            err_i = W_rot[:, i] - q_i
            W_rot[:, i] = q_i
            
            if i < d_in - 1:
                W_rot[:, i+1:] -= err_i.unsqueeze(1) * (H_inv[i, i+1:] / (H_inv[i, i] + 1e-12)).unsqueeze(0)
        
        # Обратное вращение
        W_deq = U.T @ W_rot @ V.T
        
        return W_deq, {
            "effective_bits": self.bits,
        }


# ═══════════════════════════════════════════════════════════
# 4.7 TurboQuant — ИСПРАВЛЕН
# ═══════════════════════════════════════════════════════════

class TurboQuantQuantizer(BaseKVQuantizer):
    """TurboQuant: PolarQuant + 1-bit QJL residual."""
    
    def __init__(self, bits: float = 3.5, use_qjl: bool = True, **kwargs):
        super().__init__(bits, **kwargs)
        self.use_qjl = use_qjl
        self._codebooks = {}
    
    def _get_codebook(self, d: int, bits_per_coord: int):
        key = (d, bits_per_coord)
        if key not in self._codebooks:
            from scipy.stats import beta as beta_dist
            a_param, b_param = 0.5, (d - 1) / 2
            pdf = lambda x: beta_dist.pdf(x, a_param, b_param)
            n_levels = 2 ** bits_per_coord
            centroids, boundaries = lloyd_max_quantizer(
                pdf, (0.0, 1.0), n_levels
            )
            self._codebooks[key] = (
                torch.tensor(centroids, dtype=torch.float32),
                torch.tensor(boundaries, dtype=torch.float32),
            )
        return self._codebooks[key]
    
    def quantize_vectors(self, V, **kwargs):
        # V: [n_vectors, d]
        d = V.shape[-1]
        R = random_orthogonal_matrix(d, device=V.device)
        centroids, boundaries = self._get_codebook(d, int(self.bits))
        centroids = centroids.to(V.device)
        boundaries = boundaries.to(V.device)
        
        # === STAGE 1: PolarQuant ===
        V_rot = V @ R.T
        
        norms = V_rot.norm(dim=-1, keepdim=True)
        U = V_rot / (norms + 1e-12)
        
        signs = U.sign()
        U_abs = U.abs()
        
        # boundaries от lloyd_max содержит [0, b1, b2, ..., 1.0]
        # bucketize нужны только внутренние границы
        inner_boundaries = boundaries[1:-1]  # убираем 0.0 и 1.0
        bucket_indices = torch.bucketize(U_abs, inner_boundaries)
        bucket_indices = bucket_indices.clamp(max=centroids.shape[0] - 1)
        U_q = centroids[bucket_indices]
        
        V_hat_rot = signs * U_q * norms
        
        # === STAGE 2: QJL Residual ===
        if self.use_qjl:
            residual = V_rot - V_hat_rot
            m = max(1, d // 4)
            
            S = torch.randint(0, 2, (m, d), device=V.device).float() * 2 - 1
            S = S / (m ** 0.5)
            
            sketch = S @ residual.T
            sketch_q = sketch.sign()
            
            correction = (S.T @ sketch_q).T
            
            residual_scale = residual.norm(dim=-1, keepdim=True) / (d ** 0.5)
            correction = correction * residual_scale
            
            V_hat_rot = V_hat_rot + correction
        
        # === Inverse rotation ===
        V_hat = V_hat_rot @ R
        
        polar_bits = (int(self.bits) + 1) * d + 16
        qjl_bits = (d // 4) if self.use_qjl else 0
        actual_bits = (polar_bits + qjl_bits) / d
        
        return V_hat, {
            "effective_bits": actual_bits,
            "rotation_matrix": R.cpu(),
            "is_kv_cache": True,
        }
