# Worn-Tire Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Evaluate current product vs copilot design from new to 1.6 mm remaining tread under two worn models (extrapolated, calibrated by depth remapping), produce `results/worn.json`, `figs/fig_worn.png`, and a FIG. 10 + Experiment 5 + Results subsection in the TST manuscript.

**Architecture:** New flat module `code/worn.py` (same conventions as `attribution.py`: paths from `__file__`, pure functions + `__main__`). Depth set on the decoded dict, `T.evaluate_size` per size (bypasses `decode` clipping; `tiresim.py` untouched). Calibration = monotone piecewise-linear effective-depth maps φ_wet, φ_rr fitted by bisection on the current product. Manuscript text in `tst/content_tst.py`, numbers from `worn.json`, asserts on stated signs.

**Tech Stack:** Python 3.13, numpy, matplotlib; stdlib unittest; python-docx (scratchpad) for the manuscript build.

**Spec:** `docs/superpowers/specs/2026-11-21-worn-performance-design.md` (read "Verified anchors" and "Decisions after the builder check": they override earlier provisional values).

## Global Constraints
- Do not modify `tiresim.py`, `problem.py`, `analysis.py`, `attribution.py`, `published/`, existing results; `verify_results.py` must still report all numbers match.
- Depths: `DEPTHS = [None, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.6]` (None = each design's own new depth).
- Anchors (inputs, with source keys): wet `{4.0: 0.0, 2.0: +0.17}` (`acl2026`, `dekra2022`), rrc `{1.6: -0.20}` (`trb286`).
- Line = load-weighted mean over sizes (normalised `T.SIZES` weights); hydro = worst size.
- "Lead holds" = copilot better at every depth under both models. Hydro: report gap, never "lead".
- No git: no commits.

## Review Focus
- φ must be the identity at the design's own new depth, so worn results at "new" equal `T.evaluate` exactly → test.
- A calibrated anchor must reproduce its target for the current product within 1e-3 relative → test.
- φ monotone (deeper remaining tread never maps to a shallower effective depth) and D_eff ≥ 0.5 mm → test.
- Copilot design and baseline have different new depths (8.57 vs 8.50): φ is defined on *remaining* depth D, the same function for both designs; at D = new each design uses its own new depth → covered by the identity test.
- Manuscript asserts must fail the build if a rerun flips a stated sign → asserts in content.

---

### Task 1: worn evaluation + calibration (`code/worn.py`, `code/test_worn.py`)
Interfaces produced:
- `evaluate_at(x, depth: float|None) -> dict` keys `wet, hydro, rrc` (line / worst-size) and `wet_sizes` (list of 3); `depth=None` → design's own depth.
- `EffDepth(knots: list[tuple[float,float]])` callable `D -> D_eff` (piecewise linear in D, knots sorted by D descending, linear extrapolation below the last knot, clipped ≥ 0.5).
- `calibrate(x_ref) -> dict` → `{"wet": EffDepth, "rrc": EffDepth, "knots": {...}}`: wet knots (new→new, 4→d4, 2→d2) so that ref line wet at d4 = wet_new·(1+0.0), at d2 = wet_new·1.17; rrc knots (new→new, 1.6→d16) so ref line rrc at d16 = rrc_new·0.80. Bisection on the effective depth in [0.5, new].
- `evaluate_worn(x, depth, model, cal) -> dict` with model in `{"extrapolated", "calibrated"}`; calibrated uses φ_wet for wet+hydro, φ_rr for rrc.
Tests (write first, watch fail): identity at new depth equals `T.evaluate` (wet, rrc, hydro_min) for both designs; clipping bypass (`evaluate_at(BASELINE, 3.0)["wet"] > evaluate_at(BASELINE, None)["wet"]`); anchors reproduced on BASELINE (4 mm wet ratio 1.000, 2 mm 1.170, 1.6 mm rrc 0.800, tol 1e-3); EffDepth monotone over a 0.1 mm grid and ≥ 0.5.

### Task 2: results JSON + figure + pipeline
- `run(x0, x1) -> dict` over DEPTHS × models × designs; `worn.json` also stores anchors with source keys, knots, `lead_holds` per output per model (copilot better at every depth; for hydro the boolean is stored but the text never calls it a lead), and the brief hydro limit.
- `plot(res, path) -> Figure`: 2×3 per spec; anchors drawn on wet and RRC panels for the current product; grey band for D < 7.5 mm; dotted hydro limit line "brief limit (new tire)".
- Tests: JSON values at "new" equal `T.evaluate`; `lead_holds` computed correctly on a synthetic result; plot smoke test returns a Figure with 6 axes.
- Pipeline step `worn` after `attribution` in all modes; README + CLAUDE.md lines.
- Run `run_pipeline.py --mode figures` → verify_results all match. Look at the PNG.

### Task 3: manuscript
- In `tst/content_tst.py`: RUNIN "Experiment 5: worn tires." after Experiment 4; H2 "Performance When Worn" after the Case Study (after the FIG. 9 paragraph), FIG. 10 via `fig(...)` there; one Limitations sentence; REFS `acl2026`, `dekra2022`, `trb286`; reuse existing `r117` key for the worn-state requirement (check its text fits; add `eu2025_1453` if not).
- Asserts for every sign the text states (lead holds for wet and rrc under both models; hydro gap size; both designs fall below the brief hydro limit when worn).
- Build to scratch, diff text vs current submitted manuscript (only intended additions), render PDF, look at the new pages; then back up and write manuscript, preview PDF, `FIG10.png`, updated code zip (add `worn.py`, `test_worn.py`, `worn.json`, updated `tst/`).
- Update `docs/progress.html` + `tarefas.json`.

### Final: independent reviewer agent (Opus) over code, figure, JSON and manuscript text.
