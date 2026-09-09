# Per-model chi² and Cramér's V

Tests whether the failure-category distribution depends on quantization method, **independently for each model**. The global chi² in `report.md` lumps all models together; this view shows whether the dependency is uniform or model-specific.

## deepseek-r1-distill-qwen-1.5b

- **chi² = 64.33**, dof = 9, **p = 1.94e-10**
- **Cramér's V = 0.259** (0 = independent, 1 = perfect dependence)
- N judgments = 320
- Effect size: **small-to-moderate**

**Observed counts:**

| quant | A | B | C | D | E | F | total |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 5 | 0 | 0 | 20 | 0 | 55 | 80 |
| fp8_e5m2 | 1 | 0 | 0 | 2 | 1 | 76 | 80 |
| hqq_int4 | 4 | 0 | 0 | 5 | 10 | 61 | 80 |
| hqq_int2 | 0 | 0 | 0 | 6 | 18 | 56 | 80 |

**Standardized residuals** (|r|>2 marks unusually high/low cells):

| quant | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| fp8_e4m3 | +1.58 | +0.00 | +0.00 | +4.09 | -2.69 | -0.89 |
| fp8_e5m2 | -0.95 | +0.00 | +0.00 | -2.18 | -2.32 | +1.78 |
| hqq_int4 | +0.95 | +0.00 | +0.00 | -1.13 | +1.02 | -0.13 |
| hqq_int2 | -1.58 | +0.00 | +0.00 | -0.78 | +3.99 | -0.76 |

## qwen3-1.7b

- **chi² = 5.60**, dof = 5, **p = 3.48e-01**
- **Cramér's V = 0.189** (0 = independent, 1 = perfect dependence)
- N judgments = 157
- Effect size: **small-to-moderate**

**Observed counts:**

| quant | A | B | C | D | E | F | total |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 5 | 5 | 37 | 4 | 22 | 7 | 80 |
| fp8_e5m2 | 6 | 12 | 27 | 7 | 19 | 6 | 77 |

**Standardized residuals** (|r|>2 marks unusually high/low cells):

| quant | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| fp8_e4m3 | -0.26 | -1.24 | +0.77 | -0.68 | +0.24 | +0.15 |
| fp8_e5m2 | +0.26 | +1.27 | -0.78 | +0.69 | -0.25 | -0.15 |

## deepseek-r1-distill-qwen-7b

- **chi² = 9.54**, dof = 2, **p = 8.50e-03**
- **Cramér's V = 0.244** (0 = independent, 1 = perfect dependence)
- N judgments = 160
- Effect size: **small-to-moderate**

**Observed counts:**

| quant | A | B | C | D | E | F | total |
|---|---|---|---|---|---|---|---|
| fp8_e4m3 | 0 | 0 | 0 | 7 | 2 | 71 | 80 |
| fp8_e5m2 | 0 | 0 | 0 | 0 | 0 | 80 | 80 |

**Standardized residuals** (|r|>2 marks unusually high/low cells):

| quant | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| fp8_e4m3 | +0.00 | +0.00 | +0.00 | +1.87 | +1.00 | -0.52 |
| fp8_e5m2 | +0.00 | +0.00 | +0.00 | -1.87 | -1.00 | +0.52 |

