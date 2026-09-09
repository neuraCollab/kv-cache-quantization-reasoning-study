# KV Cache Quantization × Reasoning Trace Stability — Results

Results of the completed study described in the repo-root
[README.md](../../README.md) and implemented by `src/kvtrace/` + `scripts/`:
how KV-cache quantization (FP8 E4M3/E5M2, HQQ INT4/INT2) affects chain-of-thought
reasoning in DeepSeek-R1-Distill-Qwen-1.5B/7B and Qwen3-1.7B, on 80 problems
(30 AIME-24 + 50 MATH-500).

## Суть эксперимента

Исследовалось, как различные методы сжатия KV-кэша (FP8, HQQ INT4 и INT2)
влияют на качество решения 80 математических задач (из AIME-24 и MATH-500).
Сравнивались модели: DeepSeek-R1-Distill-Qwen (1.5B и 7B) и базовая Qwen3
(1.7B). Базовым (эталонным) форматом был BF16.

### Главные выводы

- **Архитектура важнее метода сжатия.** Устойчивость модели к квантованию
  зависит в первую очередь от её архитектуры/семейства, а не от
  агрессивности алгоритма квантования.
- **DeepSeek не справляется с квантованием (проблема зацикливания).** Модели
  DeepSeek (и 1.5B, и 7B) крайне чувствительны к сжатию кэша. Даже при
  мягком квантовании (FP8) они полностью теряют точность. Основная причина
  деградации — бесконечные циклы в рассуждениях. Из-за этого модель быстро
  расходует весь лимит токенов (например, у DeepSeek 7B в FP8 длина
  генерации вырастает в 2.6 раза, и 92% ответов обрываются по лимиту).
- **Qwen3 демонстрирует высокую стабильность.** Qwen3 1.7B сохраняет логику
  рассуждений почти на уровне эталонного BF16. Использование FP8
  практически не влияет на длину генерации (рост всего на 2.6%–4.5%), и
  модель не уходит в циклы.
- **Экстремальное сжатие (HQQ INT4/INT2) работает плохо.** INT4 почти всегда
  приводит к исчерпанию лимита токенов (даже если система логирует это как
  нормальное завершение — см. `paper/limitations.md` §4.4 про баг
  `finish_reason` в HF-генераторе), а INT2 выдаёт слишком короткие и, как
  правило, неверные ответы.

**Итог:** квантование KV-кэша критично ломает логику рассуждений у моделей
семейства DeepSeek из-за потери контекста и последующего зацикливания, в то
время как Qwen3 показывает отличную устойчивость к таким оптимизациям без
существенной потери качества.

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
├── fdps/          8 First-Divergence-Point records, from scripts/02_find_fdps.py
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

**Data-integrity note (found and fixed during the reproducibility pass
below):** `data/fdps/` originally also contained `qwen3-1.7b_hqq_int2.jsonl`
and `qwen3-1.7b_hqq_int4.jsonl` — byte-identical copies of
`qwen3-1.7b_fp8_e4m3.jsonl` mislabeled under the wrong filename (their
internal `quant_method` field even still said `"fp8_e4m3"`). Both files were
removed; they were never real HQQ data, just a leftover copy-paste artifact
from whoever assembled this backup, and their presence directly contradicted
`paper/limitations.md`'s own "never collected" claim.

## Reproducing `paper/` and `tables/`

Phases 1-4 reproduce `data/`. **Phase 5** (`scripts/05_paper_analysis.py` at
the repo root) reproduces the tables and figures in `paper/` and `tables/` —
`accuracy_bars.png`, `divergence_position.png/.md`, `fdp_rate.md`,
`finish_reason.md`, `judge_confidence.md`, `per_model_chi2.md/.csv`,
`quant_only_deepdives.md`, `token_efficiency.md`. That script (and the
`kvtrace.analysis.paper`/`paper_report` modules behind it) did not exist
anywhere in the source material this repo was consolidated from — only
these output files survived — so it was rewritten from scratch by reverse-
engineering the exact formulas from these checked-in ground-truth files
(e.g. the per-model χ² table's `dof` turned out to use the scipy-natural
degrees of freedom on the zero-column-dropped matrix, not the nominal
`(rows-1)*(cols-1)` the Phase-4 global report uses).

`tests/test_paper_analysis.py` and `tests/test_paper_report.py` at the repo
root verify every recomputed number against these exact files (accuracy
percentages, χ²/dof/Cramér's V, standardized residuals, FDP rates,
`finish_reason` distributions, etc.) — run
`python scripts/05_paper_analysis.py --traces_dir research/kv-cache-reasoning-divergence-study/data/traces --fdps_dir research/kv-cache-reasoning-divergence-study/data/fdps --judgments_dir research/kv-cache-reasoning-divergence-study/data/judgments --out_dir /tmp/phase5_check`
to regenerate them yourself and diff against this directory. Row order in
the regenerated tables is alphabetical rather than matching the original
(lost) script's byte-for-byte order — only the values were verified, not
row order or the hand-written prose paragraphs.

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
