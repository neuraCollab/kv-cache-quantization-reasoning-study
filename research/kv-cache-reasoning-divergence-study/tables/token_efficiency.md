# Token-efficiency table

Average generated tokens per problem, by (model, quant). The Δ column is relative to the model's bf16 baseline. Large positive Δ correlates with category F (Repetition/loop) hits and with `finish_reason=length` — the model gets stuck and consumes its entire `max_tokens` budget without finishing.

| model | quant | n | avg_gen_tokens | Δ vs bf16 | % finish=length |
|---|---|---|---|---|---|
| deepseek-r1-distill-qwen-1.5b | bf16 | 80 | 17077 | — | 48% |
| deepseek-r1-distill-qwen-1.5b | fp8_e4m3 | 80 | 16180 | -896 (-5.2%) | 98% |
| deepseek-r1-distill-qwen-1.5b | fp8_e5m2 | 80 | 16094 | -983 (-5.8%) | 89% |
| deepseek-r1-distill-qwen-1.5b | hqq_int4 | 80 | 16384 | -693 (-4.1%) | 0% |
| deepseek-r1-distill-qwen-1.5b | hqq_int2 | 80 | 5516 | -11561 (-67.7%) | 0% |
| qwen3-1.7b | bf16 | 80 | 7748 | — | 22% |
| qwen3-1.7b | fp8_e4m3 | 80 | 7949 | +201 (+2.6%) | 26% |
| qwen3-1.7b | fp8_e5m2 | 80 | 8099 | +351 (+4.5%) | 25% |
| deepseek-r1-distill-qwen-7b | bf16 | 80 | 5997 | — | 19% |
| deepseek-r1-distill-qwen-7b | fp8_e4m3 | 80 | 15960 | +9963 (+166.1%) | 92% |
| deepseek-r1-distill-qwen-7b | fp8_e5m2 | 80 | 15805 | +9808 (+163.5%) | 92% |

> Interpretation: every catastrophic configuration in the DeepSeek family ends with `finish_reason=length` rates ≥80%, confirming Category F dominance. Qwen3 is the only family where quantization does not inflate token consumption.
