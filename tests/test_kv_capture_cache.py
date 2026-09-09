from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kvtrace.kv_capture.cache import FakeQuantCache  # noqa: E402


def _kv(shape=(1, 2, 4, 8), seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(shape, generator=g)
    v = torch.randn(shape, generator=g)
    return k, v


# ---------------------------------------------------------------------------
# FakeQuantCache
# ---------------------------------------------------------------------------


def test_fake_quant_cache_bf16_passthrough_is_lossless():
    cache = FakeQuantCache(variant="bf16")
    k, v = _kv()
    out_k, out_v = cache.update(k, v, layer_idx=0)
    assert torch.equal(out_k, k)
    assert torch.equal(out_v, v)


def test_fake_quant_cache_e4m3_introduces_bounded_rounding_error():
    cache = FakeQuantCache(variant="fp8_e4m3")
    k, v = _kv()
    out_k, _ = cache.update(k, v, layer_idx=0)
    assert not torch.equal(out_k, k)
    # e4m3 has ~3 mantissa bits -> relative error should stay small for
    # values well within its representable range.
    rel_err = ((out_k - k).abs() / k.abs().clamp_min(1e-6)).max()
    assert rel_err < 0.2


def test_fake_quant_cache_e4m3_saturates_out_of_range_to_nan():
    cache = FakeQuantCache(variant="fp8_e4m3")
    k = torch.tensor([[[[1000.0]]]])  # beyond e4m3fn's ~448 max
    out_k, _ = cache.update(k, k.clone(), layer_idx=0)
    assert torch.isnan(out_k).all()


def test_fake_quant_cache_appends_across_generation_steps():
    cache = FakeQuantCache(variant="bf16")
    k1, v1 = _kv(shape=(1, 2, 1, 8), seed=1)
    k2, v2 = _kv(shape=(1, 2, 1, 8), seed=2)
    cache.update(k1, v1, layer_idx=0)
    out_k, out_v = cache.update(k2, v2, layer_idx=0)
    assert out_k.shape == (1, 2, 2, 8)
    assert torch.equal(out_k[:, :, :1, :], k1)
    assert torch.equal(out_k[:, :, 1:, :], k2)


def test_fake_quant_cache_rejects_unknown_variant():
    with pytest.raises(ValueError):
        FakeQuantCache(variant="int3_made_up")


def test_fake_quant_cache_is_a_real_transformers_cache():
    # transformers' modeling code does `isinstance(past_key_values, Cache)` —
    # a duck-typed wrapper (not a real Cache subclass) gets rejected outright.
    from transformers.cache_utils import Cache

    assert isinstance(FakeQuantCache(variant="bf16"), Cache)


# ---------------------------------------------------------------------------
# capture (snapshot) behaviour, mixed into FakeQuantCache itself
# ---------------------------------------------------------------------------


def test_snapshot_records_final_kv_per_layer():
    cache = FakeQuantCache(variant="bf16")
    k, v = _kv(shape=(1, 2, 3, 8))
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)

    snapshot = cache.snapshot()
    assert set(snapshot.keys()) == {0, 1}
    assert snapshot[0]["key"].shape == (1, 2, 3, 8)
    assert snapshot[0]["value"].shape == (1, 2, 3, 8)


def test_snapshot_reflects_post_quantization_values():
    cache = FakeQuantCache(variant="fp8_e5m2")
    k, v = _kv()
    out_k, _ = cache.update(k, v, layer_idx=0)
    assert not torch.equal(out_k, k)  # quantization noise applied
    assert torch.equal(cache.snapshot()[0]["key"], out_k)  # captured post-quant


def test_snapshot_empty_before_any_update():
    cache = FakeQuantCache(variant="bf16")
    assert cache.snapshot() == {}
