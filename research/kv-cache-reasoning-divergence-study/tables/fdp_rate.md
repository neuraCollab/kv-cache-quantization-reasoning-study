# Per-model FDP (First Divergence Point) rate

For each (model, quant) pair: out of the 80 problems, how many produced a non-trivial divergence between baseline and quantized trace (i.e. `fdp_token_idx is not None`). A 100% FDP rate means every problem diverged somewhere; a 0% rate would mean the quantization had no observable effect on any trace.

The complementary statistic is `boxed_match` — even when traces diverge token-wise, the quantized branch can still arrive at the same final answer. Both are reported below.

| model | quant | n_pairs | diverged | cosmetic_skipped | both_correct | baseline_only | quant_only | both_wrong | no_boxed |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-r1-distill-qwen-1.5b | fp8_e4m3 | 80 | 80 (100%) | 4 | 0 | 28 | 0 | 15 | 37 |
| deepseek-r1-distill-qwen-1.5b | fp8_e5m2 | 80 | 80 (100%) | 1 | 0 | 28 | 0 | 14 | 38 |
| deepseek-r1-distill-qwen-1.5b | hqq_int4 | 80 | 80 (100%) | 2 | 0 | 28 | 0 | 14 | 38 |
| deepseek-r1-distill-qwen-1.5b | hqq_int2 | 80 | 80 (100%) | 0 | 0 | 28 | 0 | 14 | 38 |
| qwen3-1.7b | fp8_e4m3 | 80 | 80 (100%) | 61 | 41 | 1 | 2 | 22 | 14 |
| qwen3-1.7b | fp8_e5m2 | 80 | 80 (100%) | 45 | 38 | 4 | 3 | 23 | 12 |
| deepseek-r1-distill-qwen-7b | fp8_e4m3 | 80 | 80 (100%) | 8 | 0 | 48 | 0 | 17 | 15 |
| deepseek-r1-distill-qwen-7b | fp8_e5m2 | 80 | 80 (100%) | 0 | 0 | 48 | 0 | 17 | 15 |

> The `quant_only` column is the headline anomaly: quantization noise produced the right answer in cases where the deterministic bf16 baseline did not. All such cases observed in this study come from `qwen3-1.7b` × fp8 (2 + 3 = 5 cases out of 160 fp8 trials). See `quant_only_deepdives.md` for full traces.
