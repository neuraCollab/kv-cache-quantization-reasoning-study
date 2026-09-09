# General KV-Cache / Weight Quantization Benchmark

A standalone benchmark of off-the-shelf quantization schemes on
`unsloth/llama-3-8b`, independent of the reasoning-trace divergence study.
Where the main study ([`../kv-cache-reasoning-divergence-study/`](../kv-cache-reasoning-divergence-study/))
asks "how does KV quantization change *reasoning behavior*", this benchmark
asks the more conventional question — VRAM, throughput, and output-quality
tradeoffs across quantization schemes in general.

## Contents

- **`bench.py`** — generation-quality benchmark: runs a handful of
  summarization/translation prompts through FP16, `bitsandbytes` INT8
  (`LLM.int8()`), NF4, and NF4 + double-quant, scoring outputs against
  reference text (via the `evaluate` library, e.g. ROUGE/BLEU-style metrics).
- **`kv_cache_perf.py`** — performance benchmark across context lengths
  (512 / 2048 / 4096 tokens): VRAM usage, tokens/sec, and perplexity on
  WikiText-2 for the same set of quantization schemes.
- **`diploma_final_stats.json`, `diploma_full_stats.json`,
  `quant_results_fixed.json`** — recorded results from those two scripts.
  Example (`diploma_final_stats.json`): FP16 baseline uses 8906 MB / 1.68
  tok/s; NF4 drops to 7009 MB at 21.46 tok/s — the expected
  memory-for-speed-and-footprint tradeoff, with double-quant landing between
  the two.
- **`simple_test.py`, `test.py`** — smaller/earlier standalone test scripts,
  precursors to `bench.py` / `kv_cache_perf.py`.
- **`self_created.ipynb`** — exploratory notebook version of the same
  benchmarking work.

This is generic quantization-methods survey work (the kind reported in any
LLM-compression benchmark), not specific to reasoning traces or to the models
studied in the main pipeline (DeepSeek-R1-Distill, Qwen3) — it uses
Llama-3-8B throughout. Kept here as supporting/background material for the
same diploma project.
