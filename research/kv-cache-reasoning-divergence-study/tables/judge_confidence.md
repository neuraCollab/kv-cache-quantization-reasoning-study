# Judge confidence × category cross-tabulation

Sanity check on the 6-category taxonomy. Each Anthropic judgment carries a `confidence` ∈ [0, 1] reflecting how sure the model is about the assigned category. Categories that consistently get **low** average confidence are taxonomically ambiguous (the judge can't decide); those with **high** confidence are well-discriminated.

| category | n | mean_conf | std_conf | min | max | n_below_0.5 |
|---|---|---|---|---|---|---|
| A (Arithmetic) | 21 | 0.80 | 0.10 | 0.55 | 0.95 | 0 |
| B (Logical) | 17 | 0.79 | 0.08 | 0.60 | 0.92 | 0 |
| C (Strategy-switch) | 64 | 0.74 | 0.10 | 0.55 | 0.92 | 0 |
| D (Hallucination) | 51 | 0.84 | 0.07 | 0.72 | 0.95 | 0 |
| E (Premature-termination) | 72 | 0.83 | 0.13 | 0.40 | 0.97 | 1 |
| F (Repetition/loop) | 412 | 0.95 | 0.05 | 0.72 | 1.00 | 0 |

> Interpretation: F (Repetition/loop) is the most syntactically obvious failure — easy for the judge to spot via repeated n-grams. C (Strategy-switch) tends to be more contested because what counts as 'unmotivated' is subjective. A categories with `n_below_0.5 > 5` should be treated cautiously when reading downstream stats.
