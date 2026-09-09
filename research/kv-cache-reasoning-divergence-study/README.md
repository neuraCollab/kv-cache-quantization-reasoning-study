# KV Cache Quantization × Reasoning Trace Stability — Results

Results of the completed study described in the repo-root
[README.md](../../README.md) and implemented by `src/kvtrace/` + `scripts/`:
how KV-cache quantization (FP8 E4M3/E5M2, HQQ INT4/INT2) affects chain-of-thought
reasoning in DeepSeek-R1-Distill-Qwen-1.5B/7B and Qwen3-1.7B, on 80 problems
(30 AIME-24 + 50 MATH-500).

**Start here:** [`paper/supervisor_report.md`](paper/supervisor_report.md) —
the full write-up: hypotheses (H1 monotonic degradation, H2 quant-dependent
failure type, H3 architecture-dependent sensitivity), methodology, the
technical obstacles hit during collection (transformers 4.55 regression
breaking HQQ, SDPA incompatibility, eager-attention slowdown), results, and
limitations/future work. The compiled version is
[`paper/report.pdf`](paper/report.pdf) (source: `paper/report.tex`).

**Headline result:** the DeepSeek-R1-Distill family collapses to 0% accuracy
under *any* KV quantization (even the mildest FP8-E4M3), dominated by
Repetition/loop failures; Qwen3-1.7B is statistically indistinguishable from
its bf16 baseline under FP8 (and even beats it in 2/2 FP8 configs by 1
problem). Model family, not quantization aggressiveness, is the dominant
factor — see §5–6 of the report.

## Layout

```
data/
├── traces/        11 raw generation traces (model × quant), from scripts/01_generate_traces.py
├── fdps/          10 First-Divergence-Point records, from scripts/02_find_fdps.py
├── judgments/     8 Claude-judge classifications (category A-F), from scripts/03_judge_fdps.py
├── report.md/json Phase-4 global aggregate (chi-square, Cramér's V), from scripts/04_analyze.py
└── run_all_failures.log   operational log from the run_all.sh orchestrator

paper/
├── supervisor_report.md   full write-up (see above)
├── report.tex/.pdf        compiled version (+ .aux/.log/.out/.toc/.synctex.gz build artifacts)

figures/          accuracy_bars, divergence_position, per-model signature heatmaps
tables/           per-model chi-square, token efficiency, finish_reason breakdown,
                  judge-confidence validation, quant_only deep-dives, limitations

mechanistic-analysis/    deeper per-model internals probe (see below)
```

Note: `traces/`, `fdps/`, and `judgments/` don't line up 1:1 — HQQ configs for
Qwen3-1.7B and DeepSeek-7B were never collected (documented in
`paper/limitations.md` §2.3: a `transformers>=4.55` regression broke HQQ, and
the eager-attention workaround was too slow to afford within budget for those
two cells). This is the same gap the repo-root pipeline's
`scripts/verify_run.py` is built to detect.

## `mechanistic-analysis/`

Per-model internals probing that goes beyond the FDP/judge pipeline: per-layer
KV statistics, attention-shift KL divergence, layer-ablation and
counterfactual (skip-K) experiments, a trained failure-prediction model, and
multi-seed variance runs. One subfolder per model
(`deepseek-r1-distill-qwen-1.5b/`, `qwen3-1.7b/`, `qwen3-1.7b_multiseed/`,
`qwen3-4b/`, `qwen2.5-1.5b/`, `smollm2-1.7b/` — note some of these models are
not part of the main 3-model study; they appear to be one-off probes).

**The scripts that generated these artifacts were not found anywhere in the
source material this was consolidated from** — only the output JSON/NPZ/PNG
files survived. Treat this directory as results without accompanying code;
reproducing it would mean re-deriving the methodology from the artifact
filenames and JSON schemas (e.g. `attention_shift_summary_*.json`,
`failure_prediction_*.json`, `layer_ablation_*.npz`) rather than running an
existing script.
