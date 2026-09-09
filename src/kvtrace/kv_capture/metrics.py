"""Pure tensor-math metrics computed from captured KV/attention/logits.

No model, no I/O — takes tensors already captured by `capture_generator.py`
(or `CapturingCache.snapshot()`), returns plain numbers/arrays. This is the
part of the mechanistic-analysis harness that's fully testable without a GPU
or even a real model.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

_EPS = 1e-8


def kv_stats_per_layer(layers: list[torch.Tensor]) -> list[dict[str, float]]:
    """Per-layer summary stats (mean/std/max of |value|) for a K or V cache.

    `layers[i]` is that layer's K (or V) tensor, any shape — flattened before
    computing stats.
    """
    out = []
    for t in layers:
        flat = t.float().abs().flatten()
        out.append(
            {
                "mean_abs": float(flat.mean()),
                "std": float(t.float().flatten().std(unbiased=False)),
                "max_abs": float(flat.max()),
            }
        )
    return out


def _kl_from_probs(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q) for two already-normalized probability tensors, summed over
    the last dim and averaged over every other dim."""
    p = p.float().clamp_min(_EPS)
    q = q.float().clamp_min(_EPS)
    kl = (p * (p / q).log()).sum(dim=-1)
    return float(kl.mean())


def attention_kl_per_layer(
    baseline_layers: list[torch.Tensor], quant_layers: list[torch.Tensor]
) -> list[float]:
    """KL(baseline || quantized) attention distribution, per layer.

    Each `layers[i]` is a tensor of post-softmax attention weights (any
    leading batch/head/query dims, last dim = key positions — already a
    valid probability distribution along that axis).
    """
    return [_kl_from_probs(b, q) for b, q in zip(baseline_layers, quant_layers, strict=True)]


def logits_kl(baseline_logits: torch.Tensor, quant_logits: torch.Tensor) -> float:
    """KL(softmax(baseline) || softmax(quant)) over the vocab dim (last dim)."""
    p = F.softmax(baseline_logits.float(), dim=-1)
    q = F.softmax(quant_logits.float(), dim=-1)
    return _kl_from_probs(p, q)


def outlier_channel_scores(k: torch.Tensor) -> torch.Tensor:
    """Per-channel (last dim) outlier score: max |value| in that channel
    divided by the median max-|value| across channels.

    Channels that are `Liu et al. 2024`-style persistent outliers score far
    above 1.0; a uniform tensor scores ~1.0 on every channel.
    """
    max_per_channel = k.float().abs().amax(dim=tuple(range(k.dim() - 1)))
    denom = max_per_channel.median().clamp_min(_EPS)
    return max_per_channel / denom
