# Research — KV-Cache Quantization Programme

This directory holds the full research programme behind the reasoning-trace
KV-cache quantization study whose production pipeline lives at the repo root
(`src/kvtrace/`, `scripts/`, see the top-level [README.md](../README.md)).
It was consolidated from a working "nir" (НИР — научно-исследовательская
работа) folder that accumulated over the course of the project; personal and
unrelated material (an unrelated ML model, personal photos/notes) was removed
by the author before this pass, and this pass deduplicated redundant snapshots
and gave the remainder a consistent structure.

## Contents

| Directory | What it is | Status |
|---|---|---|
| [`kv-cache-reasoning-divergence-study/`](kv-cache-reasoning-divergence-study/) | **The main study.** Full results of the 3-model × 5-quant × 80-problem reasoning-trace divergence experiment described at the repo root: raw traces, FDP records, LLM-judge classifications, the compiled report, and a deeper mechanistic-analysis pass (attention shift, layer ablation, failure prediction). | Complete — this is the finished deliverable |
| [`prior-prototype/`](prior-prototype/) | An earlier, simpler iteration of the pipeline (Day 1–6 plan, KIVI/KVQuant-oriented) that predates and was superseded by `src/kvtrace/` at the repo root. | Historical / superseded |
| [`general-kv-cache-quantization-bench/`](general-kv-cache-quantization-bench/) | A separate, model-accuracy-agnostic benchmark of off-the-shelf quantization schemes (bitsandbytes INT8/NF4, double-quant) on Llama-3-8B: VRAM, throughput, perplexity, generation quality. Not about reasoning traces — a general quantization-methods survey done in parallel. | Standalone benchmark |
| [`custom-quantizer-and-theory/`](custom-quantizer-and-theory/) | Theoretical and from-scratch implementation work: a literature review of SOTA KV-cache quantization (SmoothQuant → KIVI/KVQuant → QuaRot/SpinQuant → sub-2-bit 2025–26 schemes), a formal (numerically-validated) closed-form error-bound theory for KV quantization under long chain-of-thought, and from-scratch quantizer implementations (naive RTN, GPTQ, and a novel "WaterSIC" water-filling bit allocator). | Theory / prototypes |
| [`literature-and-notes/`](literature-and-notes/) | Literature-review draft and working notes kept by the author. | Reference material |
| [`misc/`](misc/) | Leftover files not clearly tied to one subproject (a slide deck, photos). | Unsorted |

## Reading order

If you only read one thing, read
[`kv-cache-reasoning-divergence-study/paper/supervisor_report.md`](kv-cache-reasoning-divergence-study/paper/supervisor_report.md) —
it is the complete write-up (hypotheses, methodology, results, limitations,
future work) for the study that this whole repository is built around. The
other directories are supporting work: an earlier prototype, a parallel
general-purpose quantization benchmark, and the theoretical background that
motivated the KV-cache-quantization angle in the first place.

## Note on `quanrization/_redundant_safe_to_delete/`

Sitting next to `research/` (one level up from here) is
`quanrization/_redundant_safe_to_delete/` — confirmed byte-identical
duplicate trace files, superseded intermediate snapshots (`kv-trace/`,
`kv-trsh/`), a stripped `.git/` from the prototype, and `__pycache__` /
`.ipynb_checkpoints` clutter. Everything of value was moved into `research/`
before that directory was set aside; it was not deleted outright (destructive
deletes are held back for you to confirm). Delete it once you've spot-checked
that nothing is missing:

```bash
rm -rf quanrization/_redundant_safe_to_delete quanrization/nir
```
