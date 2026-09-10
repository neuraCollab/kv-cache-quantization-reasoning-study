# FDP prediction from text/metadata features only (no K-matrices)

| model x quant | n | analytical (1f) | GBM (2f) | GBM (8f) |
|---|---|---|---|---|
| deepseek-r1-distill-qwen-1.5b__fp8_e4m3 | 80 | 0.093 | 0.067 | 0.267 |
| deepseek-r1-distill-qwen-1.5b__fp8_e5m2 | 80 | 0.203 | 0.408 | 0.701 |
| deepseek-r1-distill-qwen-1.5b__hqq_int2 | 80 | -0.029 | 0.102 | 0.324 |
| deepseek-r1-distill-qwen-1.5b__hqq_int4 | 80 | -0.101 | -0.062 | -0.030 |
| deepseek-r1-distill-qwen-7b__fp8_e4m3 | 80 | 0.100 | 0.175 | 0.235 |
| deepseek-r1-distill-qwen-7b__fp8_e5m2 | 80 | 0.238 | 0.193 | 0.335 |
| qwen3-1.7b__fp8_e4m3 | 80 | 0.074 | -0.196 | -0.224 |
| qwen3-1.7b__fp8_e5m2 | 80 | 0.097 | 0.095 | -0.075 |
