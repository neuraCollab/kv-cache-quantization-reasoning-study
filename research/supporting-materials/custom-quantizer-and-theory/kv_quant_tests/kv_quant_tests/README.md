# KV-Quant Long-CoT Thesis — Test Harness

Validation suite for the closed-form error bound on KV-cache quantization
in Long Chain-of-Thought reasoning sequences.

**Total runtime: ~3 seconds on a laptop CPU. No GPU, no internet, no custom kernels.**

## Quick start

```bash
python3 run_all_tests.py
```

Expected output: `32 passed, 0 failed  in ~3s`.

Individual suites:
```bash
python3 test_01_invariants.py     # 11 tests, ~0.8s
python3 test_02_main_theorem.py   # 6 tests, ~0.8s
python3 test_03_trichotomy.py     # 8 tests, ~0.9s
python3 test_04_falsifiability.py # 7 tests, ~0.5s
```

## Architecture

| File | Purpose |
|---|---|
| `kv_math.py` | Core primitives: subtractive-dithered quant, anchor subspace, attention |
| `test_runner.py` | Lightweight pytest replacement (no external deps) |
| `test_01_invariants.py` | Suite 1: basic building-block invariants (INV-1..INV-4) |
| `test_02_main_theorem.py` | Suite 2: main theorem quantitative predictions (PRED-A..C) |
| `test_03_trichotomy.py` | Suite 3: three failure theorems for baselines (FAIL-A..C) |
| `test_04_falsifiability.py` | Suite 4: falsifiability guards (what our theorems should NOT prove) |
| `run_all_tests.py` | Master runner with summary table |

## Test → Theorem Mapping

This is what to show the committee as evidence that the theorems are validated.

### Suite 1 — Basic Invariants (11 tests)

| Test | Theorem claim being validated |
|---|---|
| `test_inv1_quant_noise_variance_matches_theory` | **σ²(b) = Δ²/12** exactly holds for subtractive dither (Schuchman 1964) |
| `test_inv1_quant_noise_mean_is_zero` | E[e] = 0 (critical for sub-Gaussian assumption) |
| `test_inv1_quant_noise_independence_of_signal` | Corr(e, x) ≈ 0 (Gray-Stockham IEEE TIT 1993) |
| `test_inv2_anchor_score_rotation_invariant` | Score is property of subspace, not basis (Architectural Definition) |
| `test_inv2_anchor_score_in_unit_interval` | score ∈ [0,1] |
| `test_inv2_anchor_in_subspace_has_score_one` | k ∈ span(V_r) ⟹ score = 1 |
| `test_inv2_anchor_orthogonal_to_subspace_has_score_zero` | k ⊥ V_r ⟹ score = 0 |
| `test_inv3_large_bits_small_error` | b → ∞ ⟹ error → 0 |
| `test_inv3_mse_decays_as_4_neg_b` | MSE ∝ 4⁻ᵇ (Bennett high-resolution) |
| `test_inv4_single_step_hanson_wright` | **E[(q^T e)²] = σ² ‖q‖²** (Hanson-Wright linear form, Vershynin HDP Thm 6.2.1) |
| `test_inv4_quadratic_form_hanson_wright` | **E[‖e‖²] = d σ²** (Hanson-Wright quadratic form) |

### Suite 2 — Main Theorem Predictions (6 tests)

| Test | Theorem claim |
|---|---|
| `test_predA_tv_linear_in_sigma` | E[TV] ∝ σ ∝ 2⁻ᵇ (slope −log 2 per bit) |
| `test_predA_kl_linear_in_sigma_squared` | E[KL] ∝ σ² ∝ 4⁻ᵇ (slope −2 log 2 per bit) |
| `test_predA_tv_zero_at_high_bits` | At b=16, error is numerically negligible |
| `test_predB_tv_grows_with_context_length` | **E[TV] grows with T** (main theorem compounding claim) |
| `test_predC_anchor_mixed_precision_reduces_error` | **Anchor-aware mixed precision reduces error by factor ∝ T_anchor/T** |
| `test_predC_anchors_are_attended_more_than_fillers` | **Lemma 2.1**: anchors get Ω(T/r) mass, fillers O(log T) |

### Suite 3 — Trichotomy Takedown (8 tests)

| Test | Failure theorem |
|---|---|
| `test_failA_eviction_loses_phoenix_anchor` | **Theorem A**: eviction of a phoenix anchor causes error ≥ p² ‖v‖² at revival |
| `test_failA_probability_of_reattention_is_nontrivial` | p_reattend = Ω(1/r) (prerequisite for Theorem A's lower bound) |
| `test_failB_offline_clustering_breaks_under_distribution_shift` | **Theorem B**: OOD query error > ID error when P_t drifts from P_cal |
| `test_failB_kl_divergence_bounds_error` | Error monotone in drift angle (KL/TV lower bound) |
| `test_failC_spectral_radius_geq_one_for_plain_attention` | **Theorem C**: ‖M_t‖ ≥ const in anchor regimes |
| `test_failC_lyapunov_exponent_non_negative` | λ₁(M) ≥ 0 under residual connection (universal for all transformers) |
| `test_failC_lyapunov_without_residual_can_contract_but_not_guaranteed_by_heuristics` | None of {KIVI, ThinK, R-KV, Squeezed} actively contracts ‖J_attn‖ |
| `test_failC_quantization_does_not_contract_jacobian` | Quantization is diagonal perturbation, not contraction |

### Suite 4 — Falsifiability Guards (7 tests)

These tests verify the theorems are *discriminating* — they fail when assumptions are violated.

| Test | Guards against |
|---|---|
| `test_falsify_no_quantization_gives_zero_error` | Pipeline bugs that inject error without quantization |
| `test_falsify_tv_zero_for_identical_distributions` | Broken TV implementation |
| `test_falsify_fewer_bits_gives_more_error` | Quantizer is ineffective (error doesn't scale with b) |
| `test_falsify_mixed_precision_useless_when_no_anchor_structure` | Anchor claim is vacuous (mixed precision helps even without structure) |
| `test_falsify_eviction_ok_when_budget_covers_anchors` | Theorem A is overbroad (eviction is always bad) |
| `test_falsify_clustering_fine_when_cal_matches_online` | Theorem B is overbroad (clustering is always bad) |
| `test_falsify_spectral_radius_can_be_small_in_contractive_regime` | Theorem C is vacuous (nothing can contract) |

## Key empirical findings (worth citing in thesis)

1. **Correct TV scaling is linear in σ, not σ²** (slope −log 2 per bit).
   The σ² scaling applies to KL divergence. This is a Pinsker-Cauchy-Schwarz
   artifact: TV(p, p+dp) ≤ ½‖dp‖₁ ≤ ½√n ‖dp‖₂, so TV is linear in perturbation magnitude.

2. **Logit separation must be O(1)** under /√d scaling for the anchor/filler
   dichotomy to fire. This requires ‖q‖·‖k‖ ≈ √d, matching the standard LayerNorm-normalized
   attention scale in real transformers.

3. **Subtractive-dither validity requires x + dither inside quantizer range.**
   At b=2, signal range must be ≤ 0.6× quantizer range; at b≥4 the standard 0.9× holds.
   Relevant for low-bit quantization schemes (KIVI 2-bit).

4. **Theorem C is a residual-connection argument.** M_t = I + J_attn, so ρ(M_t) ≥ 1
   by construction since ‖I‖ = 1. No heuristic method subtracts from I, which is why
   all of them fail to contract. This is the deepest insight of the trichotomy.

## Compute budget

- Total CPU wall-time: ~3 seconds on modern laptop (Apple M-series, Intel 12th gen+)
- Peak RAM: < 100 MB (largest tensor ~1024×64 float64)
- Disk: < 100 KB (pure Python source)
- **$0 spent.** Full validation fits in the thesis's <$50 budget with $50 to spare.

## Limitations

1. **No real LLM forward passes.** Synthetic experiments validate the math; real-model
   validation (DeepSeek-R1-Distill-Qwen-1.5B) is Stage 4 of the roadmap, not tested here.

2. **Tolerances tuned for this numerical setup.** If you change d, T, or the signal
   magnitude, slopes and ratios will shift. Tolerances are chosen to be informative
   but not overly tight.

3. **No reproducibility across different float precisions.** All tests use float64
   for reproducibility. Re-running in bf16 or fp16 may introduce larger numerical
   drift.

4. **k-means in Suite 3/4 is a simplified stand-in for Squeezed Attention.** The
   actual Squeezed Attention has offline centroid-sparsification that we don't replicate.

5. **Lyapunov estimation uses 5-iteration QR with small matrices** (d=32). For the
   real thesis, use 100+ iterations with d=128 for tighter estimates.
