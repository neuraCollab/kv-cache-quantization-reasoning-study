# Prior Prototype (superseded)

An earlier, simpler iteration of the reasoning-trace pipeline, organized as a
"Day 1–6" plan rather than the four-phase (generate → FDP → judge → analyze)
architecture used at the repo root. It targeted KIVI/KVQuant instead of the
vLLM-FP8 / HF-HQQ split the final pipeline uses, and had no FDP-finder, judge,
or statistical-analysis stage yet — see [`PROTOTYPE_PLAN.md`](PROTOTYPE_PLAN.md)
(the original README written for this prototype) for the plan as it stood.

This folder is kept for historical reference only. It is **not** wired into
the repo-root pipeline and should not be run as-is — `src/kvtrace/` at the
repo root is the actively maintained, tested, and CI-gated implementation of
the same idea.

This prototype originally had its own git history (a separate `.git/`
directory with a GitHub remote). That history was not preserved when folding
this into the parent repository — carrying a nested `.git/` in would have
produced a broken submodule reference rather than usable history. If the
original history matters, it should still be reachable from wherever this
prototype's own GitHub remote points.

`data/test_fp16_traces.jsonl` is a leftover pilot-run output (5–30 samples)
from this prototype's "Day 1" FP16 baseline collection.
