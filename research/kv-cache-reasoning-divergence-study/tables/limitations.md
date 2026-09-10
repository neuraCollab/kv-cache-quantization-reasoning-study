# Methodology Limitations & Future Work

This section enumerates the constraints that affected the experimental
design, results that should be read with caveats, and concrete follow-up
work that would close the gaps.

## 1. Excluded configurations

### 1.1 `qwen3-1.7b` × HQQ INT4/INT2 — excluded

The original study design called for the full Cartesian product
{3 models} × {bf16, fp8_e4m3, fp8_e5m2, hqq_int4, hqq_int2} = 15
configurations. Only 11 were collected. The four missing cells:

| model | quant | reason |
|---|---|---|
| qwen3-1.7b | hqq_int4 | transformers 4.55.x regression (see §1.3) |
| qwen3-1.7b | hqq_int2 | transformers 4.55.x regression |
| deepseek-r1-distill-qwen-7b | hqq_int4 | wall-clock budget (eager attention path) |
| deepseek-r1-distill-qwen-7b | hqq_int2 | wall-clock budget |

### 1.2 Why HQQ on these models was not collected

vLLM does not implement HQQ KV cache quantization, so HQQ runs use the
HuggingFace `transformers` generation loop with `QuantizedCacheConfig(backend="HQQ")`.
On `transformers >= 4.55.0` this code path crashes:

```
RuntimeError: The size of tensor a (N) must match the size of tensor b (2)
              at non-singleton dimension 3
```

The error originates in `eager_attention_forward` when `attn_weights` (shape
`[B, H, Q, K]`) is added to the causal mask (broadcast shape `[B, 1, Q, 2]`).
We confirmed the bug reproduces on both Qwen2 (DeepSeek-R1-Distill) and
Qwen3 architectures with a 15-token prompt — it is **not** model-specific.
On `transformers == 4.51.3` the same code path runs cleanly.

The codebase pins `transformers>=4.51,<4.52` (see `requirements.txt` and
commits `47b8d3a` / `8594ffb`). At the time of writing, neither
`huggingface/transformers` nor `mobiusml/hqq` has shipped a fix in the
4.5x → 4.55+ window, so we have not been able to bisect to a single
offending commit.

### 1.3 Why HQQ on the 7B model was not retried

After resolving §1.2 by downgrading transformers, we reattempted
`qwen3-1.7b × hqq_int4` and observed throughput of **~33 tokens/sec
aggregate** on an L40S (vs ~615 tok/s for the same model on vLLM bf16).
The eager attention path runs a per-step dequantization of the entire
growing KV cache; with reasoning traces averaging 7-16k tokens, one
chunk of 4 problems took 1975 seconds. The complete `qwen3-1.7b × hqq_int4`
run was projected at ≈11 hours; `deepseek-r1-distill-qwen-7b × hqq_*`
would be longer still. Given a fixed compute budget (Vast.ai L40S at
≈$1.10–1.40/hour, $5/10-hour project ceiling), we accepted partial HQQ
coverage rather than re-allocate >12 hours of GPU to a single cell.

### 1.4 Implication for downstream stats

The global chi² test in `report.md` and the per-method aggregate in
`accuracy_bars.png` lump observations across whatever models contributed
to that quant column. HQQ rows in those tables are entirely
deepseek-r1-distill-qwen-1.5b — a model that catastrophically fails under
*every* quantization scheme tested. The HQQ column therefore reflects
"this model is fragile" more than "HQQ is harmful." Per-model breakdowns
(`per_model_chi2.md`, `accuracy_bars.png`) should be the primary read.

## 2. Sampling parameters

### 2.1 `repetition_penalty = 1.05` was used uniformly

All runs use `repetition_penalty=1.05` — chosen as a mild value that
mostly preserves the model's natural distribution. In practice, the
DeepSeek-R1-Distill family proves heavily loop-prone under quantization:
80/80 problems on `deepseek-r1-distill-qwen-7b × fp8_e5m2` land in
Category F (Repetition/loop). It is plausible that a stronger penalty
(1.10 / 1.15 / 1.20) would suppress the loops and surface the underlying
arithmetic / logical errors that quantization is causing — i.e. the
current data over-counts F at the expense of A / B.

### 2.2 Greedy vs sampled

All generations use `temperature=0` (greedy). This guarantees that any
trace divergence is attributable to KV-cache numerics — there is no
sampling noise. The cost is that we cannot average over multiple seeds
to estimate the *probability* of a divergence; each (model, quant,
problem) cell is a single deterministic observation. A sampled study
(e.g. n=8 seeds at `temperature=0.6`) would let us bound the variance,
at 8× the compute cost.

### 2.3 HF generator does not expose `finish_reason` cleanly

HQQ runs use the HuggingFace generation path (`src/kvtrace/generators/hf_gen.py`),
which unconditionally writes `finish_reason="stop"` to every record because
`transformers.generate()` does not return a per-sample termination signal in
the form vLLM does. As a result, the `finish_reason` column in
`finish_reason.md` shows 100% `stop` for both HQQ cells on
deepseek-r1-distill-qwen-1.5b — that figure does **not** mean those traces
all terminated with EOS. Compare against `avg_gen_tokens`:
`hqq_int4` averages 16384 tokens (= `max_tokens`), strongly implying every
sample hit the length cap; `hqq_int2` averages 5516 tokens, suggesting real
early termination. Future work: reimplement the HF path to compare
generated length against `max_new_tokens` and emit `length` when they match.

### 2.4 `max_tokens = 16384`

Reasoning traces from these models routinely exceed 8k tokens. The 16384
ceiling means `finish_reason=length` events dominate Category F counts
— the model is stuck in a loop, but we observe it as "ran out of
budget" not "generated infinitely." The choice was forced by GPU memory
on L40S; a larger budget would shift counts within F but is unlikely to
change category boundaries.

## 3. Dataset

### 3.1 Size & composition

We use a fixed seed of **80 problems**: 30 from AIME-24 (`HuggingFaceH4/aime_2024`)
and 50 from MATH-500 (`HuggingFaceH4/MATH-500`). Both are heavily
contaminated in the training data of all three target models, so
absolute accuracy figures should not be read as "performance on unseen
math" — they measure how often quantization keeps the model on the
already-memorized track. This is a feature, not a bug, for our
purpose (controlled comparison) but limits external generalization.

### 3.2 Difficulty & length skew

AIME-24 problems generate reasoning chains 2-5× longer than MATH-500
problems. Divergence-position statistics (`divergence_position.md`) are
computed on the union; the mode at low normalized positions is
disproportionately driven by AIME chains where the absolute FDP token
is large but the chain is even longer.

### 3.3 Asymmetry of `quant_only` cases

The 5 `quant_only` cases observed on `qwen3-1.7b × fp8` (where the
quantized branch arrived at the correct answer and the bf16 baseline
did not) are interesting *anomalies*, not a *trend*. With 80 problems
and ~50% baseline accuracy, the binomial probability that quantization
noise rescues the model on at least 5 problems by pure chance is
non-trivial. We do not claim FP8 KV cache quantization improves Qwen3-1.7B
performance — we claim it is **statistically indistinguishable** from
bf16 on this benchmark, with both directional anomalies present
(`baseline_only` cases also exist at similar rates).

## 4. Judge model

### 4.1 Single-judge classification

Failure-mode taxonomy is assigned by Anthropic Claude (a single judge
model) using a fixed 6-category prompt with prompt caching for the
taxonomy block. We did not collect human inter-rater agreement.
`judge_confidence.md` reports per-category mean confidence as a proxy
for taxonomy ambiguity — categories with a meaningful share of
`confidence < 0.5` should be treated as soft labels rather than
ground truth.

### 4.2 Cost vs coverage trade-off

Phase 3 invoked the judge on every divergent (model, quant, problem)
cell where both branches produced a final boxed answer or where one
branch did. This kept costs at ~$3 (≈640 cached calls) but means
divergent cases that hit `max_tokens` on *both* sides without producing
a boxed answer are scored as `no_boxed` rather than judged. These cases
are visible in the `boxed_match` distribution in `fdp_rate.md`.

## 5. Reproducibility

### 5.1 Stochastic environment

vLLM 0.7.3 with greedy decoding (`temperature=0`, `top_p=1`) is
deterministic given fixed `seed` and fixed CUDA / cuBLAS configuration.
We have not verified bit-equality across hardware (L40S vs A100 vs
H100) — different cards may produce different traces even with
matched seeds, due to different reduction orders in fused kernels.
Cross-card reproduction is open future work.

### 5.2 Library pins

The pinned dependency window (`vllm 0.7.x`, `transformers 4.51.x`,
`hqq >= 0.2.5`) reflects the only combination we could get all five
quantization backends running simultaneously. Any deviation has been
observed to break at least one path. We document the exact pins in
`requirements.txt`; future reruns should use those exact versions or
expect breakage.

### 5.3 Hardware

All runs except the bf16 vLLM 7B run were collected on a single Vast.ai
L40S instance (Ada Lovelace, sm_89, 48 GB VRAM, ~$1.10–1.40/hour
spot). The 7B bf16 vLLM run also ran on L40S; HQQ runs that did
complete used the same card with the eager-attention HF path.

## 6. Future work — concrete experiments

In order of priority and ease:

1. **Repetition-penalty sweep on DeepSeek family.** Re-run
   deepseek-r1-distill-qwen-{1.5,7}b × {fp8_e4m3, fp8_e5m2} at
   `repetition_penalty ∈ {1.05, 1.10, 1.15, 1.20}`. Hypothesis: at 1.15+
   the loops break and Category F gives way to A / B / D — i.e. the
   *real* effects of FP8 KV quantization on these models become
   measurable.
2. **HQQ on Qwen3 + 7B once `transformers` ships a fix.** Track the
   `huggingface/transformers` release notes for HQQ QuantizedCache
   compatibility on 4.55+. When fixed, fill in the four missing cells.
3. **Multi-seed sampling at `temperature=0.6`.** Estimate variance
   bounds on accuracy and on `quant_only` / `baseline_only` rates.
   8 seeds × 11 configs × 80 problems = ~7k generations; ~$30 of GPU.
4. **Cross-architecture replication.** Add Llama-3.2-3B and Qwen2.5-3B
   to the model set. Test whether the "Qwen3 robust, DeepSeek collapses"
   result is family-specific or generalizes.
5. **Judge inter-rater study.** Have a second judge model (e.g. GPT-4o)
   re-classify a 100-divergence subset; compute Cohen's kappa to
   establish taxonomy reliability.
6. **Per-layer KV-quant ablation.** Quantize only layers 0–N or only the
   last N layers; the current results quantize all layers uniformly. May
   reveal whether divergences originate from a specific layer's noise.
