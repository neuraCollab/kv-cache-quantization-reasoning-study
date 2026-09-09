# KV Cache Quantization — Trace-Level Diagnostic Study

Reproducible pipeline for diagnosing **how** KV cache quantization methods
break reasoning in compact (1.5B–7B) reasoning models, not just how much
they degrade accuracy. Output is a markdown + JSON report with failure
**signatures** for 5 quantization configurations across 3 models.

## Design documents

- Design spec: [docs/superpowers/specs/2026-04-25-kv-trace-study-design.md](docs/superpowers/specs/2026-04-25-kv-trace-study-design.md)
- Implementation plan: [docs/superpowers/plans/2026-04-25-kv-trace-study.md](docs/superpowers/plans/2026-04-25-kv-trace-study.md)

## What it does

Five idempotent phases:

1. **GENERATE** — run each of 3 models × 5 KV-cache configurations × 80
   math problems (30 AIME-24 + 50 MATH-500) through its own generator
   (vLLM for BF16 / FP8 E5M2 / FP8 E4M3, HuggingFace Transformers + HQQ
   for INT4 / INT2).
2. **FIND FDP** — for each quantized trace, locate the First Divergence
   Point from the BF16 baseline using token-level exact matching plus a
   MiniLM semantic re-sync filter to skip cosmetic divergences.
3. **JUDGE** — ask Claude Sonnet 4.6 (with Anthropic prompt caching and a
   local SHA256 prompt cache) to classify each FDP into one of six error
   categories: A-Arithmetic, B-Logical, C-Strategy-switch,
   D-Hallucination, E-Premature-termination, F-Repetition/loop.
4. **ANALYZE** — build a confusion matrix (method × category), run a
   chi-square independence test and Cramér's V, and emit a deterministic
   markdown + JSON report plus heatmap plots.
5. **PAPER ANALYSIS** (post-hoc, CPU-only) — per-model chi-square/Cramér's V
   breakdowns, accuracy-by-model plots, divergence-position histograms,
   token-efficiency and `finish_reason` tables, judge-confidence validation,
   and `quant_only` case deep-dives — the tables and figures behind
   [`research/kv-cache-reasoning-divergence-study/paper/supervisor_report.md`](research/kv-cache-reasoning-divergence-study/paper/supervisor_report.md).
   Verified to reproduce that report's numbers exactly from the checked-in
   Phase 1-3 data (see `tests/test_paper_analysis.py`).
6. **KV CAPTURE** (mechanistic, GPU-required) — `src/kvtrace/kv_capture/` +
   `scripts/06_kv_capture.py` teacher-force an already-generated bf16 trace
   through the real model under a quantized KV cache (FP8 fake-quantized via
   torch's native `float8_e4m3fn`/`float8_e5m2` dtypes, since vLLM doesn't
   expose per-layer K/V for introspection; HQQ via the real production
   `QuantizedCache`), capturing per-layer attention-shift KL, KV statistics,
   logit-KL trajectory, and outlier-channel scores, plus three follow-on
   analyses grounded in the author's recovered NIR report:
   `scripts/07_layer_ablation.py` (single-layer ablation, quantizing one
   layer at a time), `scripts/08_defense_recipe.py` (the "protect top-N
   outlier channels in bf16" recipe — both a teacher-forced measurement and
   a real autoregressive `model.generate()` validation, since the report
   found the lab-measured benefit mostly disappears in real generation),
   and `scripts/09_multiseed_jaccard.py` (outlier-channel identity across
   sampling seeds). Reconstructs this much of
   [`research/kv-cache-reasoning-divergence-study/mechanistic-analysis/`](research/kv-cache-reasoning-divergence-study/mechanistic-analysis/)
   (whose original code wasn't recovered, only its outputs) — the CNN
   failure-predictor is deliberately not implemented (needs further work
   first), and "counterfactual skip-K" isn't described anywhere in the
   recovered report, so nothing was reconstructed for it. The hook/capture
   plumbing (including real `model.generate()` with a custom cache) is
   verified end-to-end against a tiny real model on CPU
   (`pytest -m network`); it has not been run against the actual study
   models, which need a GPU this repo doesn't have.

Each phase is resumable from HuggingFace Hub snapshots, so a Vast.ai
instance death in the middle of the run is cheap to recover from.

## Hardware

- **Single RTX 4090 (24 GB, Ada, CC 8.9)** on Vast.ai (~$0.40/hour).
  Ada is chosen over Ampere because it has native FP8 tensor cores,
  making `fp8_e5m2` and `fp8_e4m3` qualitatively distinct — this is
  essential for the "different methods break differently" thesis.
- ~14 GPU-hours total for the full 3-model × 5-config × 80-problem run.

## Budget

| Line item | Cost |
|---|---|
| Compute (Vast.ai, ~14 GPU-hours) | ~$5.60 |
| Claude Sonnet 4.6 judge (prompt cached) | ~$3.00 |
| HuggingFace Hub (public dataset) | $0 |
| Contingency | ~$3.00 |
| **Total** | **~$12** |

## Install

```bash
git clone <this repo>
cd kv-trace-study

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Dev dependencies (tests, linters):

```bash
pip install -r requirements-dev.txt
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | for Phase 3 | Claude Sonnet 4.6 judge |
| `HF_REPO_ID` | optional | full HF dataset repo id (e.g. `me/my-kv-study`) |
| `HF_USER` | optional | HF username; dataset is `{HF_USER}/kv-trace-study` |
| `HF_TOKEN` | for HF upload | write access to the dataset repo |

If no HF variable is set the pipeline still runs locally — uploads are
silently skipped.

## Run the full study

```bash
# Set at least ANTHROPIC_API_KEY; HF_USER is strongly recommended.
export ANTHROPIC_API_KEY="sk-ant-..."
export HF_USER="your-hf-username"
export HF_TOKEN="hf_..."

# Calibrate the judge FIRST (takes ~30s on live API).
# Must pass ≥7/10 — if it fails, re-check the taxonomy prompt.
pytest -m live_api tests/test_judge_calibration.py -v

# Now the real thing.
bash scripts/run_all.sh

# Output:
#   outputs/report.md
#   outputs/report.json
#   outputs/plots/*.png
#   HF:  {HF_USER}/kv-trace-study with 15 trace + 12 FDP + 12 judgment revisions
```

For a faster iteration run that drops the most experimental config:

```bash
bash scripts/run_all.sh --light
```

## Run individual phases

```bash
# Phase 1 — one (model, config) at a time, resumable
python scripts/01_generate_traces.py \
    --model deepseek-r1-distill-qwen-1.5b \
    --config fp8_e5m2 \
    --resume

# Phase 2 — needs baseline bf16 already generated
python scripts/02_find_fdps.py --model deepseek-r1-distill-qwen-1.5b

# Phase 3 — all FDPs at once; cached
python scripts/03_judge_fdps.py

# Phase 4 — CPU only, <1 min
python scripts/04_analyze.py

# Phase 5 — CPU only, no GPU/network; post-hoc tables + plots for the paper
python scripts/05_paper_analysis.py

# Phase 6 — GPU required; mechanistic KV-capture vs. the bf16 baseline
python scripts/06_kv_capture.py --model deepseek-r1-distill-qwen-1.5b --quant fp8_e4m3
```

## Testing

```bash
# CI default — no GPU, no live API, ≥85% coverage gate
make test

# GPU-dependent tests (run on the rented 4090 before Phase 1)
make test-gpu

# Live-API calibration (run before Phase 3)
make test-live

# Downloads a tiny real HF model to verify the kv_capture hook wiring end-to-end
make test-network
```

Four pytest markers:

| Marker | When to run |
|---|---|
| (none) | always; CI default |
| `@pytest.mark.gpu` | before renting GPU time |
| `@pytest.mark.live_api` | before each Phase 3 run (catches Anthropic drift) |
| `@pytest.mark.network` | verifying `kv_capture/` against a real (tiny) model without a GPU |

## Repository layout

```
kv-trace-study/
├── config/                   # 3 YAML files — models, quant methods, pipeline
├── src/kvtrace/
│   ├── generators/           # vLLM + HF (HQQ) backends behind one ABC
│   ├── fdp/                  # hybrid token + semantic re-sync finder
│   ├── judge/                # taxonomy, prompt, Claude client, golden set
│   ├── hf_hub/               # idempotent upload / download
│   ├── analysis/             # signatures + markdown report + paper tables/plots
│   └── kv_capture/           # mechanistic capture: fake-quant cache, KL/outlier metrics
├── scripts/                  # 01…06 phase CLIs + run_all.sh
├── tests/                    # CPU, GPU, live-API, and network suites
└── outputs/                  # runtime artifacts (gitignored)
```

## Reproducibility

- Greedy decoding (`temperature=0.0`, `top_p=1.0`) + fixed `seed=42`.
- Chat templates taken verbatim from each model's HF tokenizer.
- `requirements.txt` pins `vllm==0.6.6`, `transformers==4.45.0`,
  `anthropic==0.40.0`.
- Prompt and taxonomy are versioned (`PROMPT_V1`, `TAXONOMY_V1`); a
  change bumps the SHA256 cache key, so stale judgments never silently
  reappear.
- Each JSONL row is self-describing — model, config, seed, timestamp,
  prompt version.

## Results

The completed study — the actual traces, FDP records, judge classifications,
compiled report, and a deeper mechanistic-analysis pass (attention shift,
layer ablation, failure prediction) — lives in
[`research/kv-cache-reasoning-divergence-study/`](research/kv-cache-reasoning-divergence-study/).
Start with its
[`paper/supervisor_report.md`](research/kv-cache-reasoning-divergence-study/paper/supervisor_report.md)
for the full write-up. `research/` also holds an earlier prototype, a general
(non-reasoning) quantization benchmark, and the theoretical background work —
see [`research/README.md`](research/README.md) for the full index.

## Citation

The compiled report (`research/kv-cache-reasoning-divergence-study/paper/report.pdf`)
and dataset are the citable artifacts: cite via the HuggingFace dataset id
and the GitHub commit hash.
