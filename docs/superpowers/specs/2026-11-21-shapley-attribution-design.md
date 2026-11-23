# Design: Shapley attribution figure (current product → copilot design)

## Intent
A new paper figure that explains *why* the copilot design outperforms the current product: which design changes produce the change in each objective. It answers the expected reviewer question "is the gain real pattern design or just a better compound?" and complements Fig 7 (which shows *what* changed).

## Goals
- Split the change in RRC, wet braking and treadwear (current product → copilot case design) into contributions from 7 design-variable groups.
- Contributions add up exactly to the total change and do not depend on any ordering.
- Show line-average contributions, with each of the 3 sizes visible.

## Non-goals / anti-goals
- No change to the simulator, brief, existing results, existing figures or `published/`.
- No surrogate (GP) in the attribution; the simulator itself is evaluated.
- Manuscript text and figure renumbering in the TST submission are not part of this change.

## Approaches (picked: 1)
1. **Grouped Shapley over 7 variable groups**: exact, order-independent, sums to total; 2⁷ = 128 simulator runs. *Picked.*
2. Sequential waterfall: 8 runs, but depends on the order chosen.
3. One-at-a-time swaps: bars don't sum to the total; interactions unexplained.

## Design
- **Groups** (indices into `T.NAMES`): compound {tand60, tand0}; circumferential grooves {n_circ, w_circ}; lateral grooves {w_lat, ang_lat, inner_frac}; sipes {sipes}; tread depth {depth}; pitching {n_pitch, pitch_ratio}; crown {crown}.
- **Pair**: `T.BASELINE` → `results/case_design.npy` (written by `analysis.py`, so the new step runs after it).
- **Computation**: for each of the 128 coalitions, build a design taking the copilot values for the groups in the coalition and the baseline values otherwise; `T.evaluate(x, detail=True)`. Exact Shapley per group, per size, for rrc, wet, wear. Line-average contribution = load-weighted average of the per-size contributions (Shapley is linear, so this equals Shapley on the line metric). Assert that the contributions sum to the total change (tolerance 1e-6). Noise and hydroplaning contributions are also stored in the JSON, but not plotted.
- **New file** `code/attribution.py` → `results/attribution.json` and `figs/fig_attribution.png` (300 dpi, 7.8 in wide, colors from `viz_common.py`, matching the other figures).
- **Figure**: 3 panels (RRC ↓, wet braking ↓, treadwear ↑). Each is a horizontal waterfall from the current-product value through the 7 group bars to the copilot-design value (line average); each bar carries 3 markers for the per-size contributions. Bars are colored by whether they help or hurt the objective.
- **Pipeline**: add an `attribution` step to `run_pipeline.py` after `analysis` in the `figures`, `quick` and `full` modes. `verify_results.py` is unchanged; the number in the figure title is the paper's figure number, left to the author.

## Left to the builder
Exact bar ordering (suggest sorted by absolute line-average contribution per panel), marker styling, label wording.
