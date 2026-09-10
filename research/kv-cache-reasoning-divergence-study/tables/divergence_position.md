# Divergence position summary

Where in the trace does quantization first cause a divergence? Position is `fdp_token_idx / num_generated_tokens_baseline`. 0.0 = diverges at the very first generated token; 1.0 = diverges right at the end. The plot is `divergence_position.png`.

**Reading guide.** A median near 0.0 means quantization noise hits early — the model goes off the rails almost immediately. A median near 1.0 means quantization tolerates most of the reasoning chain and only causes problems near the end (compounding error). A flat histogram means no positional pattern.

| model | quant | n | median | mean | std | min | max |
|---|---|---|---|---|---|---|---|
| deepseek-r1-distill-qwen-1.5b | fp8_e4m3 | 80 | 0.00 | 0.01 | 0.06 | 0.00 | 0.48 |
| deepseek-r1-distill-qwen-1.5b | fp8_e5m2 | 80 | 0.00 | 0.01 | 0.07 | 0.00 | 0.62 |
| deepseek-r1-distill-qwen-1.5b | hqq_int2 | 80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| deepseek-r1-distill-qwen-1.5b | hqq_int4 | 80 | 0.00 | 0.02 | 0.11 | 0.00 | 1.00 |
| deepseek-r1-distill-qwen-7b | fp8_e4m3 | 80 | 0.00 | 0.05 | 0.16 | 0.00 | 0.96 |
| deepseek-r1-distill-qwen-7b | fp8_e5m2 | 80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| qwen3-1.7b | fp8_e4m3 | 80 | 0.07 | 0.16 | 0.19 | 0.00 | 0.78 |
| qwen3-1.7b | fp8_e5m2 | 80 | 0.03 | 0.13 | 0.18 | 0.00 | 0.65 |

