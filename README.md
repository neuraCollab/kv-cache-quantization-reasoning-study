# KV-Cache Quantization × Reasoning Trace Stability

What actually breaks when an LLM's KV-cache gets quantized — not just how
much accuracy drops, but *where* the reasoning chain snaps and *why*.

## The finding

![Accuracy by model and quantization](research/kv-cache-reasoning-divergence-study/figures/accuracy_bars.png)

DeepSeek-R1-Distill collapses to **0% accuracy** under the mildest scheme
tested (FP8-E4M3) — both the 1.5B and 7B variants. Qwen3-1.7B, a model of
comparable size, loses essentially nothing (52.5% → 53.7%, within noise).
Same hardware, same quantization, same 80 problems. The only variable that
predicts the outcome is model family.

![Divergence position distribution](research/kv-cache-reasoning-divergence-study/figures/divergence_position.png)

DeepSeek doesn't drift into error — it diverges from the unquantized
baseline in the first few tokens, almost every time. Qwen3's divergences,
when they happen, spread across the whole trace. That's the mechanistic
difference: DeepSeek can't tolerate KV noise at all; Qwen3 absorbs it and
keeps reasoning.

![Failure signature heatmap](research/kv-cache-reasoning-divergence-study/figures/signatures_all.png)

And it's not random breakage — it's one specific failure mode. Across every
quantization scheme, the dominant category is **repetition loops**: the
model gets stuck re-emitting the same reasoning step until it runs out of
tokens. HQQ INT4/INT2 push almost everything into this bucket.

A follow-up: the obvious fix — keep the ~10 "outlier" K-channels that carry
most of the quantization error in full precision — recovers 34% of the
error in a single forward pass. Run the same fix in *real* autoregressive
generation and the benefit drops to noise level (−2.4%). Defenses measured
teacher-forced don't transfer to production decoding: each decode step only
sees one new K-token, and outlier-channel identity isn't stable across
steps.

→ full findings + reproduction commands:
[`research/kv-cache-reasoning-divergence-study/RESULTS.md`](research/kv-cache-reasoning-divergence-study/RESULTS.md)

## Engineering notes

- **vLLM doesn't expose KV-cache internals** — no hook to introspect
  per-layer K/V during generation. Worked around it by fake-quantizing
  through torch's native `float8_e4m3fn`/`float8_e5m2` dtypes (a real
  quantize→dequantize round trip) on HF Transformers' eager-attention path
  instead, which does let you hook `Cache.update()`.
- **A bug that only showed up against a real model.** A first cut at the
  KV-cache wrapper duck-typed the `Cache` interface instead of subclassing
  it — passed every unit test, then broke immediately against a real model,
  because Transformers does `isinstance(past_key_values, Cache)`
  internally and rejects anything else. Fixed with real inheritance; kept
  as a regression test.
- **Bisected a transformers regression.** `transformers>=4.52` silently
  breaks HQQ's quantized KV-cache on an unrelated code path. Reproduced on
  a 15-token prompt, confirmed it wasn't architecture-specific, pinned
  `<4.52` with the exact repro documented inline.
- **CI caught a real cross-version numerical difference.** A test assumed
  FP8 overflow produces NaN; a newer torch release saturates to the max
  representable value instead. Both are valid — the test now checks the
  invariant that matters, not one library version's specific choice.
- **Quantization sometimes *helps*.** In 5 of 160 FP8 runs on Qwen3, the
  quantized trace reached the correct answer where the deterministic bf16
  baseline didn't — noise nudged the model off a stuck verification loop
  onto a shorter path to the same answer.
- **A lost analysis script, rebuilt from its own output.** The script that
  generated the paper's tables and plots was gone — only the output files
  survived. Rebuilt it by reverse-engineering the exact formulas from those
  files, then verified it reproduces every number byte-for-byte.

## More

- [`RESULTS.md`](research/kv-cache-reasoning-divergence-study/RESULTS.md) — every finding, ranked by significance, with a reproduction command
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — setup, all 10 pipeline phases, testing, repo layout
- [`research/README.md`](research/README.md) — full index: raw data, the original report, prior prototype, theory work
