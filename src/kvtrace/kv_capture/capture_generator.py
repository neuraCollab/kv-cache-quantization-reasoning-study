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
    channel_jaccard,
    kv_stats_per_layer,
    logits_kl,
    outlier_channel_scores,
    relative_frobenius_error,
    top_n_outlier_channels,
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


def build_cache(
    variant: str,
    *,
    layers: set[int] | None = None,
    protected_channels: dict[int, list[int]] | None = None,
) -> Any:
    """Construct the (capturing) cache for a variant.

    bf16/fp8_e4m3/fp8_e5m2 -> `FakeQuantCache` (numerically simulated, since
    vLLM's real FP8 kernels aren't introspectable). hqq_int4/hqq_int2 -> the
    *real* `transformers.cache_utils.QuantizedCache(backend="HQQ", ...)` used
    in production by `generators/hf_gen.py` — this path needs the `hqq`
    package and is not exercised by the CPU-only network test.

    `layers`/`protected_channels` (single-layer ablation / per-channel
    defense, report §4.5/§5) only apply to the `FakeQuantCache` path — the
    report found the defense recipe actively *hurts* HQQ (§5.3, it breaks
    HQQ's own per-group calibration), so there's no reason to wire it in there.
    """
    if variant in VALID_VARIANTS:
        return FakeQuantCache(variant, layers=layers, protected_channels=protected_channels)
    if variant in HQQ_VARIANTS:
        if layers is not None or protected_channels is not None:
            raise ValueError(
                "layers/protected_channels are not supported for HQQ variants "
                "(report §5.3: the defense recipe makes HQQ worse, not better)"
            )
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
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    variant: str,
    *,
    layers: set[int] | None = None,
    protected_channels: dict[int, list[int]] | None = None,
) -> CaptureResult:
    """One teacher-forced forward pass, capturing attentions/hidden_states/
    logits/K,V for every layer under the given cache variant."""
    cache = build_cache(variant, layers=layers, protected_channels=protected_channels)
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


# ---------------------------------------------------------------------------
# Single-layer ablation (report §4.5)
# ---------------------------------------------------------------------------


def single_layer_ablation_kl(
    model: Any, input_ids: Any, attention_mask: Any, variant: str
) -> dict[int, float]:
    """Quantize exactly one layer at a time: rank layers by how much
    quantizing *only* that layer shifts final logits from the all-bf16
    baseline (mean KL over positions).

    The report compares this "logit-impact" ranking against
    `compare_baseline_vs_quant`'s per-layer attention-shift-KL ranking (from
    quantizing *every* layer) and finds they disagree — logit-impact reflects
    downstream cascade depth (27 subsequent layers can amplify or dampen a
    layer's local shift), not just that layer's own local attention shift.
    """
    baseline = capture_forward_pass(model, input_ids, attention_mask, "bf16")
    n_layers = len(baseline.kv_snapshot)
    seq_len = baseline.logits.shape[1]

    scores: dict[int, float] = {}
    for layer in range(n_layers):
        ablated = capture_forward_pass(model, input_ids, attention_mask, variant, layers={layer})
        kls = [logits_kl(baseline.logits[0, t], ablated.logits[0, t]) for t in range(seq_len)]
        scores[layer] = float(sum(kls) / len(kls))
    return scores


# ---------------------------------------------------------------------------
# Per-channel defense recipe (report §5)
# ---------------------------------------------------------------------------


def select_defense_channels(
    kv_snapshot: dict[int, dict[str, Any]], top_n: int
) -> dict[int, list[int]]:
    """Per-layer top-N outlier-channel indices from a baseline (bf16) K
    snapshot — the channel set the defense recipe (report §5.1) protects in
    bf16 while the rest of that layer quantizes normally.
    """
    return {layer: top_n_outlier_channels(snap["key"], top_n) for layer, snap in kv_snapshot.items()}


def teacher_forced_defense_effect(
    model: Any, input_ids: Any, attention_mask: Any, variant: str, top_n: int
) -> dict[str, Any]:
    """Single-forward (teacher-forced) defense check (report §5.2/§5.3):
    protect the top-N outlier channels per layer in bf16, quantize the rest,
    and compare relative K-error (report §3.3 metric 1) against the same
    quantization with no protection at all.

    This is the *lab* measurement, where it looks like it works (report:
    -34% K-error at N=10 on fp8_e4m3). See `ar_defense_validation` for why
    that mostly doesn't hold up in real autoregressive generation (Ф.2, the
    lab→production gap) — and note this recipe is FP8-specific: applying it
    to an HQQ variant makes K-error *worse*, not better (§5.3), because it
    fights HQQ's own per-group calibration; this function doesn't guard
    against being called with an HQQ variant, `build_cache` does.
    """
    baseline = capture_forward_pass(model, input_ids, attention_mask, "bf16")
    protected_channels = select_defense_channels(baseline.kv_snapshot, top_n)

    undefended = capture_forward_pass(model, input_ids, attention_mask, variant)
    defended = capture_forward_pass(
        model, input_ids, attention_mask, variant, protected_channels=protected_channels
    )

    n_layers = len(baseline.kv_snapshot)
    undefended_k_error = {
        layer: relative_frobenius_error(
            baseline.kv_snapshot[layer]["key"], undefended.kv_snapshot[layer]["key"]
        )
        for layer in range(n_layers)
    }
    defended_k_error = {
        layer: relative_frobenius_error(
            baseline.kv_snapshot[layer]["key"], defended.kv_snapshot[layer]["key"]
        )
        for layer in range(n_layers)
    }
    return {
        "protected_channels": protected_channels,
        "undefended_k_error": undefended_k_error,
        "defended_k_error": defended_k_error,
    }


def _generate_with_scores(
    model: Any, input_ids: Any, attention_mask: Any, cache: Any, max_new_tokens: int
) -> list[Any]:
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return list(out.scores)


def _score_trajectory_stats(baseline_scores: list[Any], other_scores: list[Any]) -> dict[str, Any]:
    n = min(len(baseline_scores), len(other_scores))
    first_divergence_step = n  # sentinel: no divergence observed in range(n)
    matches = 0
    kls: list[float] = []
    for t in range(n):
        b, o = baseline_scores[t], other_scores[t]
        if torch.argmax(b, dim=-1).item() == torch.argmax(o, dim=-1).item():
            matches += 1
        elif first_divergence_step == n:
            first_divergence_step = t
        kls.append(logits_kl(b, o))
    return {
        "first_divergence_step": first_divergence_step,
        "token_agreement": matches / n if n else 0.0,
        "mean_logit_kl": float(sum(kls) / len(kls)) if kls else 0.0,
    }


def ar_defense_validation(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    variant: str,
    protected_channels: dict[int, list[int]],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Real `model.generate()` comparison (report §5.4 — the "calibrated"
    row of Table 13: channels selected once from the prefill, fixed for the
    whole generation). Unlike `teacher_forced_defense_effect`, this drives
    real incremental decoding, where each decode step sees only *one* new K
    token — this is the check that surfaced the lab→production gap (Ф.2):
    the TF-measured -34% K-error / -15% attention-KL benefit shrinks to
    -2.4% logit-KL here (within noise), because single-position K-error
    recovery doesn't propagate through the compound softmax→cascade→logits
    chain during autoregressive decoding.

    Note: the report's other Table-13 row ("top-10 per-step", where the
    protected channel set is *re-selected at every decode step* from live K)
    isn't implemented here — the report itself lists dynamic per-step
    outlier re-identification as future work, not a settled method.
    """
    baseline_cache = build_cache("bf16")
    baseline_scores = _generate_with_scores(
        model, input_ids, attention_mask, baseline_cache, max_new_tokens
    )

    undefended_cache = build_cache(variant)
    undefended_scores = _generate_with_scores(
        model, input_ids, attention_mask, undefended_cache, max_new_tokens
    )

    defended_cache = build_cache(variant, protected_channels=protected_channels)
    defended_scores = _generate_with_scores(
        model, input_ids, attention_mask, defended_cache, max_new_tokens
    )

    return {
        "undefended": _score_trajectory_stats(baseline_scores, undefended_scores),
        "defended": _score_trajectory_stats(baseline_scores, defended_scores),
    }


# ---------------------------------------------------------------------------
# Multi-seed outlier-channel identity (report §4.3)
# ---------------------------------------------------------------------------


def multiseed_channel_jaccard(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    top_n: int = 10,
    seeds: tuple[int, ...] = (1, 2, 3),
    temperature: float = 0.6,
    max_new_tokens: int = 20,
) -> dict[str, Any]:
    """Is outlier-channel identity a property of the model's *weights*, or
    an artifact of one particular sampled continuation? Generate
    `len(seeds)` independent bf16 continuations at `temperature>0` (a
    different seed each), pick each run's top-N outlier channels per layer,
    and measure pairwise Jaccard overlap. A high median (report: 0.879)
    means the channels are stable across seeds — architectural, not sampling
    noise — which is what justifies calibrating them once (§4.3/§5.1)
    instead of per-generation.
    """
    per_seed_channels: list[dict[int, list[int]]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        cache = build_cache("bf16")
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )
        per_seed_channels.append(select_defense_channels(cache.snapshot(), top_n))

    n_layers = len(per_seed_channels[0])
    per_layer: dict[int, list[float]] = {layer: [] for layer in range(n_layers)}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            for layer in range(n_layers):
                per_layer[layer].append(
                    channel_jaccard(per_seed_channels[i][layer], per_seed_channels[j][layer])
                )

    all_scores = [v for scores in per_layer.values() for v in scores]
    all_scores.sort()
    if all_scores:
        mid = len(all_scores) // 2
        median = (
            all_scores[mid]
            if len(all_scores) % 2
            else (all_scores[mid - 1] + all_scores[mid]) / 2
        )
    else:
        median = 0.0
    return {"median_jaccard": median, "per_layer": per_layer}
