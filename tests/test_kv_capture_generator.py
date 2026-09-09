"""Real-model integration test for the kv_capture hook wiring.

Downloads a tiny (untrained, ~KB-scale) real Qwen2-architecture model —
same architecture family as DeepSeek-R1-Distill-Qwen — to verify the
capture plumbing (attention/hidden_states/logits capture, CapturingCache
wired into a real forward pass, baseline-vs-quant diffing) actually works
end-to-end on CPU. The real 1.5B-7B study models are far too large to run
here; this is the closest verification possible without a GPU.

Run with: pytest -m network
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.network

TINY_MODEL = "yujiepan/qwen2-tiny-random"


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    except Exception as e:  # pragma: no cover — network/hub availability
        pytest.skip(f"could not download {TINY_MODEL}: {e}")
    model.eval()
    return model, tokenizer


@pytest.fixture
def sample_input(tiny_model_and_tokenizer):
    _, tokenizer = tiny_model_and_tokenizer
    enc = tokenizer("The quick brown fox jumps over", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def test_capture_forward_pass_shapes_are_consistent(tiny_model_and_tokenizer, sample_input):
    from kvtrace.kv_capture.capture_generator import capture_forward_pass

    model, _ = tiny_model_and_tokenizer
    input_ids, attention_mask = sample_input
    seq_len = input_ids.shape[1]

    result = capture_forward_pass(model, input_ids, attention_mask, "bf16")

    n_layers = model.config.num_hidden_layers
    assert len(result.attentions) == n_layers
    assert result.attentions[0].shape[0] == 1  # batch
    assert result.attentions[0].shape[-2:] == (seq_len, seq_len)
    assert result.logits.shape == (1, seq_len, model.config.vocab_size)
    assert len(result.kv_snapshot) == n_layers
    assert result.kv_snapshot[0]["key"].shape[-2] == seq_len


def test_fp8_capture_differs_numerically_from_bf16(tiny_model_and_tokenizer, sample_input):
    from kvtrace.kv_capture.capture_generator import capture_forward_pass

    model, _ = tiny_model_and_tokenizer
    input_ids, attention_mask = sample_input

    baseline = capture_forward_pass(model, input_ids, attention_mask, "bf16")
    quant = capture_forward_pass(model, input_ids, attention_mask, "fp8_e4m3")

    assert not torch.equal(baseline.kv_snapshot[0]["key"], quant.kv_snapshot[0]["key"])
    # Fake-quantizing K/V changes attention -> logits should shift too.
    assert not torch.allclose(baseline.logits, quant.logits)


def test_compare_baseline_vs_quant_end_to_end(tiny_model_and_tokenizer, sample_input):
    from kvtrace.kv_capture.capture_generator import compare_baseline_vs_quant

    model, _ = tiny_model_and_tokenizer
    input_ids, attention_mask = sample_input
    seq_len = input_ids.shape[1]
    n_layers = model.config.num_hidden_layers

    result = compare_baseline_vs_quant(model, input_ids, attention_mask, "fp8_e5m2")

    assert len(result["attention_shift_kl"]) == n_layers
    assert all(v >= 0.0 for v in result["attention_shift_kl"])

    assert len(result["logit_kl_trajectory"]) == seq_len
    assert all(v >= -1e-6 for v in result["logit_kl_trajectory"])

    assert len(result["kv_stats_baseline"]) == n_layers
    assert len(result["kv_stats_quant"]) == n_layers
    for stats in result["kv_stats_baseline"]:
        assert set(stats.keys()) == {"mean_abs", "std", "max_abs"}

    assert len(result["outlier_channels_baseline"]) == n_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    assert result["outlier_channels_baseline"][0].shape == (head_dim,)
