from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from kvtrace.kv_capture.metrics import (  # noqa: E402
    attention_kl_per_layer,
    channel_jaccard,
    kv_stats_per_layer,
    logits_kl,
    outlier_channel_scores,
    relative_frobenius_error,
    top_n_outlier_channels,
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


# ---------------------------------------------------------------------------
# top_n_outlier_channels (report §5.1: "N channels with largest max_t|K_pre|")
# ---------------------------------------------------------------------------


def test_top_n_outlier_channels_returns_highest_scoring_indices():
    k = torch.randn(1, 1, 20, 6) * 0.1
    k[..., 4] = 50.0
    k[..., 1] = 30.0
    assert top_n_outlier_channels(k, n=2) == [4, 1]


def test_top_n_outlier_channels_n_larger_than_channels_returns_all():
    k = torch.randn(1, 1, 5, 3)
    assert sorted(top_n_outlier_channels(k, n=10)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# channel_jaccard (report §4.3: multi-seed outlier-channel identity check)
# ---------------------------------------------------------------------------


def test_channel_jaccard_identical_sets_is_one():
    assert channel_jaccard([1, 2, 3], [3, 2, 1]) == pytest.approx(1.0)


def test_channel_jaccard_disjoint_sets_is_zero():
    assert channel_jaccard([1, 2], [3, 4]) == pytest.approx(0.0)


def test_channel_jaccard_partial_overlap():
    # intersection {2,3} = 2, union {1,2,3,4} = 4 -> 0.5
    assert channel_jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# relative_frobenius_error (report §3.3 metric 1: eps = ||K_pre-K_post||_F / ||K_pre||_F)
# ---------------------------------------------------------------------------


def test_relative_frobenius_error_zero_for_identical_tensors():
    k = torch.randn(1, 2, 5, 8)
    assert relative_frobenius_error(k, k.clone()) == pytest.approx(0.0, abs=1e-6)


def test_relative_frobenius_error_matches_manual_computation():
    k_pre = torch.tensor([[[[3.0, 4.0]]]])  # norm = 5
    k_post = torch.tensor([[[[0.0, 0.0]]]])  # diff norm = 5 -> ratio = 1.0
    assert relative_frobenius_error(k_pre, k_post) == pytest.approx(1.0, abs=1e-6)


def test_relative_frobenius_error_scales_with_perturbation_size():
    k_pre = torch.ones(1, 1, 1, 100)
    small_noise = k_pre + 0.01
    big_noise = k_pre + 0.1
    assert relative_frobenius_error(k_pre, small_noise) < relative_frobenius_error(
        k_pre, big_noise
    )
