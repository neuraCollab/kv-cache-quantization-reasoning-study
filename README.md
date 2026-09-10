# KV-Cache Quantization × Reasoning Trace Stability

Mechanistic study of *how* — not just how much — KV-cache quantization
(FP8 E4M3/E5M2, HQQ INT4/INT2) breaks chain-of-thought reasoning in compact
LLMs. Full pipeline: generate reasoning traces under each quantization
scheme → locate where a quantized trace first diverges from its bf16
baseline → classify the failure with an LLM judge → analyze the resulting
failure signatures → capture the mechanism (attention, KV statistics,
per-channel outliers) that produced them.

## Headline result

**Quantization robustness is set by model family, not by how aggressive the
quantization is.** DeepSeek-R1-Distill-Qwen (1.5B and 7B) collapses to 0%
accuracy under the *mildest* tested scheme (FP8-E4M3) — it gets stuck in
repetition loops and burns its entire token budget. Qwen3-1.7B, a model of
comparable size, loses essentially nothing under the same quantization
(53.7% vs. 52.5% bf16 baseline — within noise). A follow-up finding:
per-channel defenses that look effective in a single forward pass mostly
stop working in real autoregressive generation — the "lab→production gap."

→ **[`research/kv-cache-reasoning-divergence-study/RESULTS.md`](research/kv-cache-reasoning-divergence-study/RESULTS.md)**
for the full list of findings, ranked by novelty/significance, each with an
actual reproduction command.

## Pipeline

Ten phases, each independently runnable and (mostly) idempotent:

1. **GENERATE** (GPU) — run 3 models × 5 KV-cache configurations × 80 math
   problems (30 AIME-24 + 50 MATH-500) through vLLM (bf16/FP8) or HF
   Transformers + HQQ (INT4/INT2).
2. **FIND FDP** (CPU) — locate each quantized trace's First Divergence
   Point from its bf16 baseline via token-exact matching + a MiniLM
   semantic re-sync filter that skips cosmetic (non-substantive) diffs.
3. **JUDGE** (API) — classify each divergence with Claude Sonnet into one
   of six failure categories (A-Arithmetic, B-Logical, C-Strategy-switch,
   D-Hallucination, E-Premature-termination, F-Repetition/loop), with
   prompt caching and a local SHA256 result cache.
4. **ANALYZE** (CPU) — build the method × category confusion matrix, run a
   χ² independence test + Cramér's V, emit a markdown/JSON report and
   heatmaps.
5. **PAPER ANALYSIS** (CPU) — per-model χ²/Cramér's V, accuracy-by-model
   plots, divergence-position histograms, token-efficiency and
   `finish_reason` tables, judge-confidence validation, `quant_only`
   deep-dives. Verified to reproduce the study's own numbers exactly from
   the checked-in Phase 1-3 data.
6. **KV CAPTURE** (GPU) — teacher-force an already-generated bf16 trace
   through the model under a quantized KV cache (FP8 fake-quantized via
   torch's native `float8_e4m3fn`/`float8_e5m2` dtypes, since vLLM doesn't
   expose per-layer K/V; HQQ via the real production `QuantizedCache`),
   capturing per-layer attention-shift KL, KV statistics, logit-KL
   trajectory, and outlier-channel scores.
7. **LAYER ABLATION** (GPU) — quantize one layer at a time, rank layers by
   logit-impact on the final output.
8. **DEFENSE RECIPE** (GPU) — test "protect the top-N outlier channels in
   bf16": both a single-forward measurement (looks effective) and a real
   `model.generate()` validation (mostly doesn't hold up — this is the
   lab→production gap in the headline result).
9. **MULTI-SEED JACCARD** (GPU) — check whether outlier-channel identity is
   stable across sampling seeds (a weight property) or just noise.
10. **FDP TEXT PREDICTOR** (CPU) — can text/metadata features of the bf16
    trace alone (no K-matrices) predict where a quantized trace will
    diverge? Real signal for DeepSeek, none for Qwen3.

Phases 6-9 reconstruct the mechanistic-capture infrastructure behind
[`research/kv-cache-reasoning-divergence-study/mechanistic-analysis/`](research/kv-cache-reasoning-divergence-study/mechanistic-analysis/)
— see that directory's README for exactly what is and isn't covered.

Phases 1-5 are resumable from HuggingFace Hub snapshots, so an interrupted
GPU rental is cheap to recover from.

## Install

```bash
git clone https://github.com/neuraCollab/kv-cache-quantization-reasoning-study.git
cd kv-cache-quantization-reasoning-study

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
| `ANTHROPIC_API_KEY` | for Phase 3 | Claude judge |
| `HF_REPO_ID` | optional | full HF dataset repo id (e.g. `me/my-kv-study`) |
| `HF_USER` | optional | HF username; dataset is `{HF_USER}/kv-trace-study` |
| `HF_TOKEN` | for HF upload | write access to the dataset repo |

If no HF variable is set the pipeline still runs locally — uploads are
silently skipped.

## Run the full study (Phases 1-4)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export HF_USER="your-hf-username"   # optional but recommended
export HF_TOKEN="hf_..."            # optional but recommended

# Calibrate the judge first (~30s, live API). Must pass ≥7/10.
pytest -m live_api tests/test_judge_calibration.py -v

bash scripts/run_all.sh          # full run
bash scripts/run_all.sh --light  # drops the most experimental config
```

## Run individual phases

```bash
# Phase 1 — one (model, config) at a time, resumable
python scripts/01_generate_traces.py --model deepseek-r1-distill-qwen-1.5b --config fp8_e5m2 --resume

# Phase 2 — needs baseline bf16 already generated
python scripts/02_find_fdps.py --model deepseek-r1-distill-qwen-1.5b

python scripts/03_judge_fdps.py     # Phase 3 — all FDPs at once; cached
python scripts/04_analyze.py        # Phase 4 — CPU only, <1 min
python scripts/05_paper_analysis.py # Phase 5 — CPU only

# Phase 6 — mechanistic KV-capture vs. the bf16 baseline
python scripts/06_kv_capture.py --model deepseek-r1-distill-qwen-1.5b --quant fp8_e4m3
python scripts/07_layer_ablation.py --model qwen3-1.7b --quant fp8_e4m3
python scripts/08_defense_recipe.py --model qwen3-1.7b --quant fp8_e4m3 --mode both --top_n 10
python scripts/09_multiseed_jaccard.py --model qwen3-1.7b --seeds 1 2 3 --temperature 0.6

# Phase 10 — CPU only, no GPU
python scripts/10_fdp_text_predictor.py \
    --fdps_dir research/kv-cache-reasoning-divergence-study/data/fdps \
    --traces_dir research/kv-cache-reasoning-divergence-study/data/traces
```

## Testing

```bash
make test           # CI default — no GPU, no live API, no network; ≥85% coverage gate
make test-gpu        # GPU-dependent tests (run before renting GPU time)
make test-live       # Live-API judge calibration (run before Phase 3)
make test-network    # Downloads a tiny real HF model to verify kv_capture hook wiring
```

| Marker | When to run |
|---|---|
| (none) | always; CI default |
| `@pytest.mark.gpu` | before renting GPU time |
| `@pytest.mark.live_api` | before each Phase 3 run (catches Anthropic drift) |
| `@pytest.mark.network` | verifying `kv_capture/` hook wiring against a real model |

## Repository layout

```
kv-cache-quantization-reasoning-study/
├── config/                # 3 YAML files — models, quant methods, pipeline
├── src/kvtrace/
│   ├── generators/        # vLLM + HF (HQQ) backends behind one ABC
│   ├── fdp/               # hybrid token + semantic re-sync finder
│   ├── judge/             # taxonomy, prompt, Claude client, golden set
│   ├── hf_hub/             # idempotent upload / download
│   ├── analysis/           # signatures, report, paper tables/plots, FDP text predictor
│   └── kv_capture/         # mechanistic capture: fake-quant cache, KL/outlier metrics
├── scripts/                # 01…10 phase CLIs + run_all.sh
├── tests/                  # CPU, GPU, live-API, and network suites
├── research/               # completed study results — see research/README.md
└── outputs/                # runtime artifacts (gitignored)
```

## Reproducibility

- Greedy decoding (`temperature=0.0`, `top_p=1.0`) + fixed `seed=42`.
- Chat templates taken verbatim from each model's HF tokenizer.
- `requirements.txt` pins `vllm>=0.7.3,<0.9` and `transformers>=4.51.0,<4.52`
  deliberately — 4.52-4.55 broke the HQQ `QuantizedCache` → eager-attention
  path; see the inline comment in `requirements.txt` for the exact repro.
- Prompt and taxonomy are versioned (`PROMPT_V1`, `TAXONOMY_V1`); a change
  bumps the SHA256 cache key, so stale judgments never silently reappear.
- Each JSONL row is self-describing — model, config, seed, timestamp,
  prompt version.

## Hardware & budget

- **Single RTX 4090 (24 GB, Ada, CC 8.9)** on Vast.ai (~$0.40/hour). Ada
  over Ampere for native FP8 tensor cores — needed for `fp8_e5m2` and
  `fp8_e4m3` to be qualitatively distinct.
- ~14 GPU-hours for the full 3-model × 5-config × 80-problem run (Phases
  1-4). Phases 6-9 (mechanistic capture) run on a handful of problems per
  (model, quant) pair, not the full 80 — see each script's `--n_problems`.

| Line item | Cost |
|---|---|
| Compute (Vast.ai, ~14 GPU-hours) | ~$5.60 |
| Claude judge (prompt cached) | ~$3.00 |
| HuggingFace Hub (public dataset) | $0 |
| Contingency | ~$3.00 |
| **Total** | **~$12** |

## Results

The completed study — traces, FDP records, judge classifications, a
compiled report, mechanistic-analysis captures, and a new text-only FDP
predictor experiment — lives in
[`research/kv-cache-reasoning-divergence-study/`](research/kv-cache-reasoning-divergence-study/).
Start with [`RESULTS.md`](research/kv-cache-reasoning-divergence-study/RESULTS.md)
for findings + reproduction commands, or
[`paper/supervisor_report.md`](research/kv-cache-reasoning-divergence-study/paper/supervisor_report.md)
for the full original write-up. `research/` also holds an earlier
prototype, a general (non-reasoning) quantization benchmark, and
theoretical background work — see [`research/README.md`](research/README.md)
for the full index.

## Design documents

- Design spec: [docs/superpowers/specs/2026-04-25-kv-trace-study-design.md](docs/superpowers/specs/2026-04-25-kv-trace-study-design.md)
- Implementation plan: [docs/superpowers/plans/2026-04-25-kv-trace-study.md](docs/superpowers/plans/2026-04-25-kv-trace-study.md)

## Citation

The compiled report (`research/kv-cache-reasoning-divergence-study/paper/report.pdf`)
and dataset are the citable artifacts: cite via the HuggingFace dataset id
and the GitHub commit hash.
