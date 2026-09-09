"""KV-cache wrappers for the mechanistic-analysis capture harness.

`FakeQuantCache` fake-quantizes K/V on write using torch's native FP8
dtypes (`float8_e4m3fn` / `float8_e5m2`) — a quantize→dequantize round
trip, same idea as `hf_gen.py`'s real HQQ `QuantizedCache`, just for FP8.

vLLM (used for real FP8 generation in `generators/vllm_gen.py`) doesn't
expose per-layer K/V for introspection, so this harness always runs
through HF Transformers eager attention — even for the FP8 variants —
and *simulates* FP8 numerically via this cache instead of vLLM's actual
FP8 kernels. HQQ capture, by contrast, can wrap the real
`transformers.cache_utils.QuantizedCache(backend="HQQ")` used in
production, since that already runs through HF (see `CapturingQuantizedCache`
in `capture_generator.py`).

Both `update()` overrides also *capture* the post-quantization K/V written
for each layer (`.snapshot()`), for later analysis (kv_stats_per_layer,
attention-shift KL, outlier-channel impact, ...).

Note: these must be real `transformers.cache_utils.Cache` subclasses, not a
duck-typed wrapper — Transformers' modeling code does
`isinstance(past_key_values, Cache)` and raises otherwise.
"""
from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover — torch optional at import time
    torch = None  # type: ignore[assignment]

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = object  # type: ignore[assignment, misc]

_FP8_DTYPES: dict[str, Any] = {}
if torch is not None:
    _FP8_DTYPES = {
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }

VALID_VARIANTS = ("bf16", "fp8_e4m3", "fp8_e5m2")


class CaptureMixin:
    """Adds a per-layer K/V `.snapshot()` to any Cache subclass's `update()`.

    Mix in *before* the Cache subclass (`class X(CaptureMixin, DynamicCache)`)
    so `super().update(...)` in `_record_and_return` reaches the real cache
    implementation.
    """

    def _record_and_return(
        self, key_states: Any, value_states: Any, layer_idx: int, cache_kwargs: dict | None
    ) -> tuple[Any, Any]:
        out_k, out_v = super().update(key_states, value_states, layer_idx, cache_kwargs)  # type: ignore[misc]
        if not hasattr(self, "_captured"):
            self._captured: dict[int, dict[str, Any]] = {}
        self._captured[layer_idx] = {"key": out_k, "value": out_v}
        return out_k, out_v

    def snapshot(self) -> dict[int, dict[str, Any]]:
        """Latest captured (post-quantization) K/V tensor per layer."""
        return getattr(self, "_captured", {})


class FakeQuantCache(CaptureMixin, DynamicCache):
    """DynamicCache that fake-quantizes K/V to FP8 (or passes through for
    bf16) and captures the post-quantization K/V per layer."""

    def __init__(self, variant: str, *args: Any, **kwargs: Any) -> None:
        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; expected one of {VALID_VARIANTS}"
            )
        super().__init__(*args, **kwargs)
        self.variant = variant
        self._fp8_dtype = _FP8_DTYPES.get(variant)

    def _quantize(self, t: torch.Tensor) -> torch.Tensor:
        if self._fp8_dtype is None:
            return t
        orig_dtype = t.dtype
        return t.to(self._fp8_dtype).to(orig_dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._record_and_return(
            self._quantize(key_states), self._quantize(value_states), layer_idx, cache_kwargs
        )
