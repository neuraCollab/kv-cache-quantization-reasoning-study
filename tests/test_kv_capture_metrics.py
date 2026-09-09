from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from kvtrace.kv_capture.metrics import (  # noqa: E402
    attention_kl_per_layer,
    kv_stats_per_layer,
    logits_kl,
    outlier_channel_scores,
)

# ---------------------------------------------------------------------------
# kv_stats_per_layer
# ---------------------------------------------------------------------------


def test_kv_stats_per_layer_computes_mean_std_max():
    layer0 = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])  # shape (1,1,2,2)
    layer1 = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]]]])
    stats = kv_stats_per_layer([layer0, layer1])
    assert len(stats) == 2
    assert stats[0]["mean_abs"] == pytest.approx(2.5)
    assert stats[0]["max_abs"] == pytest.approx(4.0)
    assert stats[1]["std"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# attention_kl_per_layer
# ---------------------------------------------------------------------------


def test_attention_kl_per_layer_zero_for_identical_distributions():
    attn = torch.tensor([[[[0.7, 0.3]]]])  # (batch=1, heads=1, q=1, k=2)
    kl = attention_kl_per_layer([attn], [attn])
    assert kl[0] == pytest.approx(0.0, abs=1e-6)


def test_attention_kl_per_layer_matches_manual_kl_divergence():
    p = torch.tensor([[[[0.9, 0.1]]]])
    q = torch.tensor([[[[0.5, 0.5]]]])
    kl = attention_kl_per_layer([p], [q])
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert kl[0] == pytest.approx(expected, abs=1e-4)


def test_attention_kl_per_layer_handles_multiple_layers():
    a1 = torch.tensor([[[[1.0, 0.0]]]])
    a2 = torch.tensor([[[[0.5, 0.5]]]])
    kl = attention_kl_per_layer([a1, a1], [a1, a2])
    assert kl[0] == pytest.approx(0.0, abs=1e-6)
    assert kl[1] > 0.0


# ---------------------------------------------------------------------------
# logits_kl
# ---------------------------------------------------------------------------


def test_logits_kl_zero_for_identical_logits():
    logits = torch.tensor([2.0, 1.0, 0.1])
    assert logits_kl(logits, logits) == pytest.approx(0.0, abs=1e-6)


def test_logits_kl_positive_for_different_logits():
    a = torch.tensor([5.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 5.0])
    assert logits_kl(a, b) > 1.0


# ---------------------------------------------------------------------------
# outlier_channel_scores
# ---------------------------------------------------------------------------


def test_outlier_channel_scores_flags_the_high_magnitude_channel():
    # shape (batch, heads, seq, channels); channel 2 is a clear outlier.
    k = torch.randn(1, 1, 20, 4) * 0.1
    k[..., 2] = 50.0
    scores = outlier_channel_scores(k)
    assert scores.shape == (4,)
    assert int(scores.argmax()) == 2


def test_outlier_channel_scores_uniform_channels_are_close():
    k = torch.ones(1, 1, 10, 3)
    scores = outlier_channel_scores(k)
    assert scores[0] == pytest.approx(scores[1], abs=1e-4)
    assert scores[1] == pytest.approx(scores[2], abs=1e-4)
