# KV Cache Quantization — Failure-Signature Report

Total judgments: **637**

Chi-square: chi2=66.58, dof=15, **p=1.8e-08**

Cramér's V: **0.187**


## Raw counts (rows = quant method, cols = category A..F)

| method | A | B | C | D | E | F | total |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 10 | 5 | 37 | 31 | 24 | 133 | 240 |
| fp8_e5m2 | 7 | 12 | 27 | 9 | 20 | 162 | 237 |
| hqq_int2 | 0 | 0 | 0 | 6 | 18 | 56 | 80 |
| hqq_int4 | 4 | 0 | 0 | 5 | 10 | 61 | 80 |

## Normalized failure signatures (rows sum to 1)

| method | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| fp8_e4m3 | 0.04 | 0.02 | 0.15 | 0.13 | 0.10 | 0.55 |
| fp8_e5m2 | 0.03 | 0.05 | 0.11 | 0.04 | 0.08 | 0.68 |
| hqq_int2 | 0.00 | 0.00 | 0.00 | 0.07 | 0.23 | 0.70 |
| hqq_int4 | 0.05 | 0.00 | 0.00 | 0.06 | 0.12 | 0.76 |
