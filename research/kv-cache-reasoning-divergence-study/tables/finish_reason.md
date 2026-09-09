# `finish_reason` distribution by (model, quant)

vLLM reports `finish_reason ∈ {stop, length, ...}`. `stop` means the model emitted the EOS token cleanly; `length` means it hit `max_tokens` first. A configuration with high `length` rate (>50%) is producing runaway / looping reasoning; this is the underlying mechanism behind Category F.

**Caveat — HQQ rows.** The HQQ runs use the HuggingFace generation loop, which in `src/kvtrace/generators/hf_gen.py` unconditionally reports `finish_reason='stop'` (HF's `model.generate()` does not expose per-sample termination reasons in a clean form). For HQQ cells, the column `% length` is therefore always 0% by construction and should not be read literally. Use `avg_gen_tokens` from `token_efficiency.md` as a proxy: rows where avg ≈ max_tokens (16384) almost certainly hit the length cap regardless of the reported label.

| model | quant | n | % stop | % length | raw |
|---|---|---|---|---|---|
| deepseek-r1-distill-qwen-1.5b | bf16 | 80 | 52% | 48% | {'stop': 42, 'length': 38} |
| deepseek-r1-distill-qwen-1.5b | fp8_e4m3 | 80 | 2% | 98% | {'length': 78, 'stop': 2} |
| deepseek-r1-distill-qwen-1.5b | fp8_e5m2 | 80 | 11% | 89% | {'length': 71, 'stop': 9} |
| deepseek-r1-distill-qwen-1.5b | hqq_int4 | 80 | 100% | 0% | {'stop': 80} |
| deepseek-r1-distill-qwen-1.5b | hqq_int2 | 80 | 100% | 0% | {'stop': 80} |
| qwen3-1.7b | bf16 | 80 | 78% | 22% | {'stop': 62, 'length': 18} |
| qwen3-1.7b | fp8_e4m3 | 80 | 74% | 26% | {'stop': 59, 'length': 21} |
| qwen3-1.7b | fp8_e5m2 | 80 | 75% | 25% | {'stop': 60, 'length': 20} |
| deepseek-r1-distill-qwen-7b | bf16 | 80 | 81% | 19% | {'stop': 65, 'length': 15} |
| deepseek-r1-distill-qwen-7b | fp8_e4m3 | 80 | 8% | 92% | {'length': 74, 'stop': 6} |
| deepseek-r1-distill-qwen-7b | fp8_e5m2 | 80 | 8% | 92% | {'length': 74, 'stop': 6} |

> Note: `hqq_int4` and `hqq_int2` on deepseek-1.5b show 100% `stop` despite 0% accuracy — these configurations produce *short, confident, wrong* outputs rather than loops. Different failure mode from fp8 on the same model.
