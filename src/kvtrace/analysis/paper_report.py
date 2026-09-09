"""Render Phase 5 tables (markdown/CSV) and plots (PNG) from computed rows.

Row order in every table is deterministic — sorted by (model, quant) — but is
not guaranteed to match the byte order of the original (lost) script that
first produced research/kv-cache-reasoning-divergence-study/tables/*.md; only
the underlying values were verified against that ground truth, not row order.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kvtrace.analysis.signatures import CATEGORY_ORDER  # noqa: E402


def _fmt_pct(x: float | None, decimals: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:.{decimals}f}%"


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["model"], r["quant"]))


# ---------------------------------------------------------------------------
# finish_reason.md
# ---------------------------------------------------------------------------


def render_finish_reason_table(rows: list[dict]) -> str:
    lines = [
        "# `finish_reason` distribution by (model, quant)",
        "",
        "vLLM reports `finish_reason ∈ {stop, length, ...}`. `stop` means the "
        "model emitted the EOS token cleanly; `length` means it hit "
        "`max_tokens` first.",
        "",
        "| model | quant | n | % stop | % length | raw |",
        "|---|---|---|---|---|---|",
    ]
    for r in _sorted_rows(rows):
        lines.append(
            f"| {r['model']} | {r['quant']} | {r['n']} | {r['pct_stop']}% | "
            f"{r['pct_length']}% | {r['raw']} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# token_efficiency.md
# ---------------------------------------------------------------------------


def render_token_efficiency_table(rows: list[dict]) -> str:
    lines = [
        "# Token-efficiency table",
        "",
        "Average generated tokens per problem, by (model, quant). The Δ "
        "column is relative to the model's bf16 baseline.",
        "",
        "| model | quant | n | avg_gen_tokens | Δ vs bf16 | % finish=length |",
        "|---|---|---|---|---|---|",
    ]
    for r in _sorted_rows(rows):
        if r["delta"] is None:
            delta_str = "—"
        else:
            sign = "+" if r["delta"] >= 0 else ""
            delta_str = f"{sign}{r['delta']} ({sign}{r['delta_pct']:.1f}%)"
        lines.append(
            f"| {r['model']} | {r['quant']} | {r['n']} | {r['avg_gen_tokens']} | "
            f"{delta_str} | {r['pct_length']}% |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# divergence_position.md
# ---------------------------------------------------------------------------


def render_divergence_position_table(rows: list[dict]) -> str:
    lines = [
        "# Divergence position summary",
        "",
        "Where in the trace does quantization first cause a divergence? "
        "Position is `fdp_token_idx / num_generated_tokens_baseline`.",
        "",
        "| model | quant | n | median | mean | std | min | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in _sorted_rows(rows):
        lines.append(
            f"| {r['model']} | {r['quant']} | {r['n']} | {r['median']:.2f} | "
            f"{r['mean']:.2f} | {r['std']:.2f} | {r['min']:.2f} | {r['max']:.2f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# fdp_rate.md
# ---------------------------------------------------------------------------


def render_fdp_rate_table(rows: list[dict]) -> str:
    lines = [
        "# Per-model FDP (First Divergence Point) rate",
        "",
        "For each (model, quant) pair: out of the problems, how many produced "
        "a non-trivial divergence between baseline and quantized trace.",
        "",
        "| model | quant | n_pairs | diverged | cosmetic_skipped | "
        "both_correct | baseline_only | quant_only | both_wrong | no_boxed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in _sorted_rows(rows):
        lines.append(
            f"| {r['model']} | {r['quant']} | {r['n_pairs']} | "
            f"{r['diverged']} ({r['diverged_pct']}%) | {r['cosmetic_skipped_count']} | "
            f"{r['both_correct']} | {r['baseline_only']} | {r['quant_only']} | "
            f"{r['both_wrong']} | {r['no_boxed']} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# judge_confidence.md
# ---------------------------------------------------------------------------


def render_judge_confidence_table(rows: list[dict]) -> str:
    lines = [
        "# Judge confidence × category cross-tabulation",
        "",
        "Sanity check on the taxonomy: how confident was the judge per "
        "category?",
        "",
        "| category | n | mean_conf | std_conf | min | max | n_below_0.5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: CATEGORY_ORDER.index(r["category"])):
        lines.append(
            f"| {r['category']} | {r['n']} | {r['mean_conf']:.2f} | "
            f"{r['std_conf']:.2f} | {r['min']:.2f} | {r['max']:.2f} | "
            f"{r['n_below_0.5']} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# per_model_chi2.md / .csv
# ---------------------------------------------------------------------------


def render_per_model_chi2(result: dict[str, dict]) -> str:
    lines = [
        "# Per-model chi² and Cramér's V",
        "",
        "Tests whether the failure-category distribution depends on "
        "quantization method, independently for each model.",
        "",
    ]
    for model in sorted(result):
        r = result[model]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"- **chi² = {r['chi2']:.2f}**, dof = {r['dof']}, **p = {r['p']:.3g}**")
        lines.append(f"- **Cramér's V = {r['cramers_v']:.3f}**")
        lines.append(f"- N judgments = {r['n']}")
        lines.append(f"- Effect size: **{r['effect_size']}**")
        lines.append("")
        lines.append("**Observed counts:**")
        lines.append("")
        lines.append("| quant | " + " | ".join(CATEGORY_ORDER) + " | total |")
        lines.append("|---|" + "|".join(["---"] * 6) + "|---|")
        for i, method in enumerate(r["methods"]):
            row = r["counts"][i].astype(int)
            lines.append(f"| {method} | " + " | ".join(str(x) for x in row) + f" | {int(row.sum())} |")
        lines.append("")
        lines.append("**Standardized residuals** (|r|>2 marks unusually high/low cells):")
        lines.append("")
        lines.append("| quant | " + " | ".join(CATEGORY_ORDER) + " |")
        lines.append("|---|" + "|".join(["---"] * 6) + "|")
        for i, method in enumerate(r["methods"]):
            row = r["residuals"][i]
            lines.append(f"| {method} | " + " | ".join(f"{x:+.2f}" for x in row) + " |")
        lines.append("")
    return "\n".join(lines)


def write_per_model_chi2_csv(result: dict[str, dict], out_path: Path) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["model", "chi2", "dof", "p_value", "cramers_v", "n", "quants"])
    for model in sorted(result):
        r = result[model]
        w.writerow(
            [
                model,
                f"{r['chi2']:.4f}",
                r["dof"],
                f"{r['p']:.6e}",
                f"{r['cramers_v']:.4f}",
                r["n"],
                "+".join(r["methods"]),
            ]
        )
    Path(out_path).write_text(buf.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# quant_only_deepdives.md
# ---------------------------------------------------------------------------


def render_quant_only_deepdives(cases: list[dict]) -> str:
    lines = [
        "# `quant_only` deep-dives",
        "",
        "Cases where the **quantized** trace produced the correct boxed "
        "answer but the **bf16 baseline** did not.",
        "",
        f"**Total `quant_only` cases found**: {len(cases)}",
        "",
    ]
    for i, c in enumerate(cases, start=1):
        lines.append(f"## Case {i}: `{c['model']}` × `{c['quant']}` × problem {c['problem_idx']}")
        lines.append("")
        lines.append(f"- **Source**: {c.get('source')}")
        lines.append(f"- **FDP token idx**: {c.get('fdp_token_idx')}")
        lines.append(f"- **Ground truth**: `{c.get('ground_truth')}`")
        lines.append(f"- **Baseline boxed**: `{c.get('baseline_boxed')}`")
        lines.append(f"- **Quant boxed**: `{c.get('quant_boxed')}`")
        cat = c.get("category")
        conf = c.get("confidence")
        if cat is not None:
            lines.append(f"- **Judge category**: **{cat}**, confidence {conf}")
        if c.get("rationale"):
            lines.append(f"- **Judge rationale**: {c['rationale']}")
        if c.get("affected_span"):
            lines.append(f"- **Affected span**: `{c['affected_span']}`")
        lines.append("")
        if c.get("problem"):
            lines.append("**Problem:**")
            lines.append("")
            lines.append(f"> {c['problem']}")
            lines.append("")
        if c.get("common_prefix"):
            lines.append("**Common prefix (last part — same in both):**")
            lines.append("")
            lines.append("```")
            lines.append(c["common_prefix"])
            lines.append("```")
            lines.append("")
        if c.get("baseline_context"):
            lines.append("**Baseline branch (bf16) around FDP:**")
            lines.append("")
            lines.append("```")
            lines.append(c["baseline_context"])
            lines.append("```")
            lines.append("")
        if c.get("quant_context"):
            lines.append(f"**Quantized branch ({c['quant']}) around FDP:**")
            lines.append("")
            lines.append("```")
            lines.append(c["quant_context"])
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def write_accuracy_bars_plot(rows: list[dict], out_path: Path) -> None:
    models = sorted({r["model"] for r in rows})
    quants = sorted({r["quant"] for r in rows})
    by_key = {(r["model"], r["quant"]): r["accuracy_pct"] for r in rows}

    fig, ax = plt.subplots(figsize=(max(6, len(quants) * 1.2), 5))
    width = 0.8 / max(len(models), 1)
    x = range(len(quants))
    for i, model in enumerate(models):
        heights = [by_key.get((model, q), 0.0) for q in quants]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, heights, width=width, label=model)

    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(quants, rotation=30, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy by model × quantization method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_divergence_position_plot(rows: list[dict], out_path: Path) -> None:
    rows = _sorted_rows(rows)
    labels = [f"{r['model']}\n{r['quant']}" for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.9), 5))
    ax.bar(range(len(rows)), means, yerr=stds, capsize=4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("normalized divergence position (mean ± std)")
    ax.set_title("Where in the trace does quantization first diverge?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
