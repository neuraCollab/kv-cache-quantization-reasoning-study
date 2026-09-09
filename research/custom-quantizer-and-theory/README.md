# Custom Quantizers & Theoretical Background

The theoretical and from-scratch-implementation groundwork behind the whole
project: a literature survey of state-of-the-art KV-cache quantization, a
formal (numerically-validated) error-bound theory for KV quantization under
long chain-of-thought, and hand-built quantizer implementations built up from
naive rounding to a novel bit-allocation scheme.

## Contents

### Literature review

- **[`kv_cache_quantization_ru.md`](kv_cache_quantization_ru.md)** — a
  detailed (Russian-language) survey of the field: why KV cache, not weights,
  is the dominant memory bottleneck at long context; why keys and values need
  different quantization axes (per-channel K vs. per-token V); the
  pre-RoPE-vs-post-RoPE design tension; and an algorithmic walkthrough from
  SmoothQuant (2022) through Atom, KIVI, KVQuant, WKVQuant, QuaRot/SpinQuant,
  to sub-2-bit 2025–26 schemes (CommVQ, XQuant, H1B-KV, TurboQuant), plus how
  each interacts with FlashAttention-3 / PagedAttention / NVFP4 Tensor Cores.
- **`kv_quant_novelty_ru.html`** — companion novelty pitch (same subject,
  HTML rendering, presumably for presentation).

### Formal theory + validation suite

- **[`kv_quant_tests/kv_quant_tests/`](kv_quant_tests/kv_quant_tests/)** — a
  32-test, pure-Python/no-GPU validation harness (`run_all_tests.py`, ~3s
  total) for a closed-form error-bound theorem on KV-cache quantization in
  long-CoT sequences. See its own
  [README](kv_quant_tests/kv_quant_tests/README.md) for the full
  test-to-theorem mapping; in short it validates:
  - **Suite 1 (invariants):** quantization-noise variance/mean/independence
    match theory (Schuchman, Gray-Stockham, Hanson-Wright).
  - **Suite 2 (main theorem):** total-variation error scales linearly in
    quantization noise σ (not σ², which is the KL-divergence scaling —
    flagged in the README as a result worth citing), and an "anchor" subspace
    of tokens can be kept at higher precision to cut error by a factor
    proportional to `T_anchor/T`.
  - **Suite 3 (trichotomy):** three failure theorems against baseline
    eviction/clustering/heuristic-compression methods (KIVI, ThinK, R-KV,
    Squeezed Attention) — e.g. evicting an "anchor" token causes bounded-below
    error on re-attention; the residual-connection structure of attention
    (`M_t = I + J_attn`) means no heuristic actually contracts the relevant
    operator norm.
  - **Suite 4 (falsifiability):** guards that the theorems are discriminating
    rather than vacuous (they fail to hold when their own assumptions are
    violated).
  - Explicitly scoped limitations: synthetic math only, no real LLM forward
    passes yet (that's flagged as the next roadmap stage against
    DeepSeek-R1-Distill-Qwen-1.5B — i.e., this theory work and the main
    reasoning-trace study were meant to eventually connect).

### From-scratch quantizer implementations

- **[`scripts/`](scripts/)** — hand-rolled PyTorch implementations, building
  up in sophistication:
  - naive round-to-nearest fake quantization (`fixed_quantizers.py`,
    `quantization_workbench*.ipynb`) — INT8/INT4 weight quantization on
    `facebook/opt-125m`, including two documented early bugs (shared
    quantizer state across layers; `deepcopy` copying an already-mutated
    model) and the finding that naive PyTorch INT4/INT8 gives no real
    speedup without hardware-native low-bit kernels;
  - GPTQ from scratch (`gptq_fixed.ipynb`) — Optimal-Brain-Surgeon-style
    error compensation via the inverse Hessian, including a documented
    critical bug (using H instead of H⁻¹, which amplified rather than
    compensated error, PPL exploding to 4615);
  - AWQ (`quantization_awq_fixed.py`);
  - **`watersic_quantizer.py` / `bd_watersic.py` / `watersic_notebook.ipynb`**
    — **WaterSIC**, a novel extension of GPTQ that replaces uniform bit-width
    allocation with a water-filling scheme over the Hessian's eigenvalues, so
    bits are spent on high-curvature (high-importance) channels rather than
    split evenly — maximizing reconstruction SNR under an average bit-width
    budget.

### `oleg_quant/`

A parallel copy of the general quantization benchmark (mirrors
[`../general-kv-cache-quantization-bench/`](../general-kv-cache-quantization-bench/):
same `bench.py`/`kv_cache_perf.py`/diploma-stats files), plus two files not
present in that folder — `kv_cache.py` and `gptq_project_context.md`. The
name suggests this may be a collaborator's ("Oleg's") variant rather than a
pure accidental duplicate, so it was kept as-is rather than merged/deduped;
worth checking with whoever "Oleg" is before removing it.

### Top-level notebooks

- `quantization_benchmark_v4.ipynb`, `self_created.ipynb` — further
  exploratory notebook iterations of the quantizer-building work above.
