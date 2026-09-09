"""Ties model + cache + hooks together: one teacher-forced forward pass over
an already-generated sequence, capturing per-layer attention, hidden states,
logits, and post-quantization K/V — then diffing a baseline (bf16) capture
against a quantized one.

Deliberately *not* live multi-step generation: `compare_baseline_vs_quant`
takes a token sequence that was already generated once under bf16 (e.g. from
`outputs/traces/{model}_bf16.jsonl`), and replays it through the model twice
— once per cache variant — under teacher forcing. That gives position-by-
position ("trajectory") divergence at every layer in a single forward pass
each, rather than needing to drive `model.generate()` with attention capture
across many incremental decode steps.

GPU-only in practice (real models won't fit/run at usable speed on CPU) but
every code path here is exercised in `tests/test_kv_capture_generator.py`
against a tiny real Qwen2-architecture model on CPU (`pytest -m network`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers.cache_utils import QuantizedCache, QuantizedCacheConfig
except Exception:  # pragma: no cover
    QuantizedCache = None  # type: ignore[assignment, misc]
    QuantizedCacheConfig = None  # type: ignore[assignment, misc]

from kvtrace.kv_capture.cache import VALID_VARIANTS, CaptureMixin, FakeQuantCache
from kvtrace.kv_capture.metrics import (
    attention_kl_per_layer,
    kv_stats_per_layer,
    logits_kl,
    outlier_channel_scores,
)

HQQ_VARIANTS = ("hqq_int4", "hqq_int2")
_HQQ_NBITS = {"hqq_int4": 4, "hqq_int2": 2}


if QuantizedCache is not None:

    class CapturingQuantizedCache(CaptureMixin, QuantizedCache):  # type: ignore[misc]
        """The real production HQQ `QuantizedCache`, with `.snapshot()` added.

        Unlike `FakeQuantCache`, this isn't a numerical simulation — it's the
        same cache class `generators/hf_gen.py` passes to `model.generate()`
        via `cache_implementation="quantized"` in real Phase-1 runs. Needs the
        `hqq` package and, in practice, a GPU; not exercised by
        `pytest -m network` (only bf16/fp8 are, via `FakeQuantCache`).
        """

        def update(
            self,
            key_states: Any,
            value_states: Any,
            layer_idx: int,
            cache_kwargs: dict | None = None,
        ) -> tuple[Any, Any]:
            return self._record_and_return(key_states, value_states, layer_idx, cache_kwargs)

else:  # pragma: no cover
    CapturingQuantizedCache = None  # type: ignore[assignment, misc]


def build_cache(variant: str) -> Any:
    """Construct the (capturing) cache for a variant.

    bf16/fp8_e4m3/fp8_e5m2 -> `FakeQuantCache` (numerically simulated, since
    vLLM's real FP8 kernels aren't introspectable). hqq_int4/hqq_int2 -> the
    *real* `transformers.cache_utils.QuantizedCache(backend="HQQ", ...)` used
    in production by `generators/hf_gen.py` — this path needs the `hqq`
    package and is not exercised by the CPU-only network test.
    """
    if variant in VALID_VARIANTS:
        return FakeQuantCache(variant)
    if variant in HQQ_VARIANTS:
        if CapturingQuantizedCache is None:
            raise RuntimeError("transformers.cache_utils.QuantizedCache unavailable")
        cache_config = QuantizedCacheConfig(backend="HQQ", nbits=_HQQ_NBITS[variant])
        return CapturingQuantizedCache(cache_config=cache_config)
    raise ValueError(f"unknown cache variant {variant!r}")


@dataclass
class CaptureResult:
    attentions: list[Any]
    hidden_states: list[Any]
    logits: Any
    kv_snapshot: dict[int, dict[str, Any]]


def capture_forward_pass(
    model: Any, input_ids: Any, attention_mask: Any, variant: str
) -> CaptureResult:
    """One teacher-forced forward pass, capturing attentions/hidden_states/
    logits/K,V for every layer under the given cache variant."""
    cache = build_cache(variant)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )
    return CaptureResult(
        attentions=list(out.attentions),
        hidden_states=list(out.hidden_states),
        logits=out.logits,
        kv_snapshot=cache.snapshot(),
    )


def compare_baseline_vs_quant(
    model: Any, input_ids: Any, attention_mask: Any, quant_variant: str
) -> dict[str, Any]:
    """Run the same sequence under bf16 and `quant_variant`, diff the results.

    Returns a dict matching the shape of the (lost) mechanistic-analysis
    artifacts this reconstructs: `attention_shift_kl` (per layer),
    `logit_kl_trajectory` (per position), `kv_stats_{baseline,quant}` (per
    layer), and `outlier_channels_{baseline,quant}` (per layer, per channel).
    """
    baseline = capture_forward_pass(model, input_ids, attention_mask, "bf16")
    quant = capture_forward_pass(model, input_ids, attention_mask, quant_variant)

    seq_len = baseline.logits.shape[1]
    logit_kl_trajectory = [
        logits_kl(baseline.logits[0, i], quant.logits[0, i]) for i in range(seq_len)
    ]

    n_layers = len(baseline.kv_snapshot)
    baseline_keys = [baseline.kv_snapshot[i]["key"] for i in range(n_layers)]
    quant_keys = [quant.kv_snapshot[i]["key"] for i in range(n_layers)]

    return {
        "attention_shift_kl": attention_kl_per_layer(baseline.attentions, quant.attentions),
        "logit_kl_trajectory": logit_kl_trajectory,
        "kv_stats_baseline": kv_stats_per_layer(baseline_keys),
        "kv_stats_quant": kv_stats_per_layer(quant_keys),
        "outlier_channels_baseline": [outlier_channel_scores(k) for k in baseline_keys],
        "outlier_channels_quant": [outlier_channel_scores(k) for k in quant_keys],
    }
