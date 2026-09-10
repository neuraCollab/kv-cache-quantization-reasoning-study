# Addenda — new experiments beyond the recovered report

Unlike the rest of `kv-cache-reasoning-divergence-study/`, nothing here
reconstructs something from the author's original NIR report. This is new
analysis, run on the same real study data, in response to a follow-up
question the report itself raises but couldn't answer without K-matrices we
don't have access to here (no GPU, and the raw capture safetensors were
never uploaded anywhere — checked the author's 3 public HuggingFace
datasets, none contain them).

## FDP prediction from text/metadata alone (`fdp_text_predictor.{json,md}`)

**Question:** the report's own FDP predictor (Part 7) needs real per-layer K
matrices from a GPU forward pass — a 1D-CNN gets R²=0.98 in-distribution on
Qwen3-1.7B. Without a GPU, is there *any* signal left in cheap, CPU-only
proxies of the already-generated bf16 trace — n-gram repetition rate,
length, `finish_reason`, baseline correctness?

**Method:** `scripts/10_fdp_text_predictor.py` +
`src/kvtrace/analysis/fdp_predictor.py` (tested, TDD — see
`tests/test_fdp_predictor.py`). For each of the 8 real (model, quant) pairs
in `data/fdps/` (80 problems each, 640 rows total), 5-fold CV comparing:
an analytical 1-feature linear fit (`repetition_score_4gram` only), a
2-feature GBM, and an 8-feature GBM. Reported via **Spearman rank
correlation**, not R² — R² is numerically unstable on these small,
heavy-tailed targets (the same pathology the report itself found: -3.9 R²
but 0.94 Spearman on real K-matrix features for DeepSeek). No judge-derived
fields (category/confidence/...) are used — those come from classifying the
FDP itself, so using them as a predictive feature would be circular.

**Result — real, reproducible, run via `python scripts/10_fdp_text_predictor.py`:**

| model × quant | Spearman (8-feature GBM) |
|---|---|
| deepseek-1.5b × fp8_e5m2 | **0.70** |
| deepseek-1.5b × hqq_int2 | 0.32 |
| deepseek-1.5b × fp8_e4m3 | 0.27 |
| deepseek-7b × fp8_e5m2 | 0.34 |
| deepseek-7b × fp8_e4m3 | 0.24 |
| deepseek-1.5b × hqq_int4 | -0.03 |
| qwen3-1.7b × fp8_e5m2 | -0.08 |
| qwen3-1.7b × fp8_e4m3 | -0.22 |

**Honest interpretation:**

- **DeepSeek family: real, non-trivial signal**, strongest on
  `fp8_e5m2` (Spearman 0.70). The single-feature analytical model alone
  (`fdp_token_idx ≈ 28.1 − 26.1 × repetition_score_4gram` on
  deepseek-1.5b×fp8_e5m2) is statistically significant on the full 80-row
  fit (Spearman −0.29, p=0.0099 between the raw feature and the target,
  before sign-flipping through the negative coefficient) — **traces whose
  bf16 baseline is already more repetitive diverge earlier under
  quantization**. Mechanistically this fits the main study's own finding
  that DeepSeek's quantized failures are overwhelmingly Category F
  (Repetition/loop, [`../tables/per_model_chi2.md`](../tables/per_model_chi2.md)):
  a baseline trace that's already drifting toward repetition is the one
  quantization noise tips over fastest. Treat the *sign and rank* as the
  reliable part, not the exact coefficients — this is one linear fit on 80
  points, and testing 8 groups for the best one carries a real
  multiple-comparisons risk (p≈0.01 is not far above a Bonferroni-corrected
  ~0.006 threshold for 8 comparisons).
- **Qwen3-1.7B: no signal**, consistent with (and an independent
  replication of) the report's own §7.2 "prompt-only" finding — FDP is an
  emergent property of decoding for this model family, not something static
  trace properties predict. The 8-feature GBM does *worse* than random rank
  order here (-0.22), i.e. actively misleading if trusted.
- **hqq_int4 on DeepSeek: no signal** — unlike the two FP8 variants on the
  same model.
- More features doesn't reliably help: the 8-feature GBM beats the
  1-feature analytical model on DeepSeek (where there's real structure to
  find) but is *worse* than the simple linear fit on Qwen3 (where there
  isn't — the extra features just give it more ways to overfit 64 training
  rows).

**Caveat that applies to everything above:** this used only 8 of the
15 possible (model, quant) cells (the ones with real FDP data — see the
main study's `limitations.md` on the HQQ gap), and even the best result is
correlational, not causal, and unverified against the report's own
(inaccessible) K-matrix-based numbers for a sanity check. Worth taking as a
lead for a real K-matrix follow-up, not a finished result.
