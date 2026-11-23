# Design: Worn-tire performance — does the copilot design keep its lead?

## Intent
A new paper result answering a question reviewers and regulators care about: the copilot design beats the current product when new — does it still beat it as the tread wears? UN R117 (04 series) now has a worn-state wet grip requirement for C1 tires, and "good when new, poor when worn" is a known industry complaint. A design method that improves only the new tire would be weaker than one whose gain holds over tread life.

## Goals
- Compare current product (`T.BASELINE`) and selected copilot design (`results/case_design.npy`) from new depth down to 1.6 mm remaining tread, on all 3 sizes, for wet braking, hydroplaning onset (worst size, as in the brief), and rolling resistance.
- Do it under **two worn models** and report whether the copilot's lead holds under both (same logic as the existing sensitivity study: a conclusion that holds across model variants).
- Make the plausibility check against published worn-tire data part of the result, not hidden.
- New figure, new short Experimental Design entry, new Results subsection, caption and references in the TST manuscript, built through `tst/build_tst.py`.

## Non-goals / anti-goals
- No change to new-tire simulator behaviour or to any published number (`verify_results.py` must still match; `published/` untouched).
- No worn treadwear index (the index means life from new; meaningless part-worn). No worn noise result (no data to check it; noise is not part of the worn regulation).
- No re-run of the benchmark on worn tires (different, bigger question).
- Anti-goal: presenting worn absolute values as predictions. The claim is about the *lead* between two designs, and the paper states plainly where the simulator disagrees with data.

## Constraints
- `tiresim.decode` clips depth to its design range [7.5, 10] mm. Worn evaluation must set the depth on the decoded design dict and call `T.evaluate_size` directly; `tiresim.py` is not modified.
- Remaining depths evaluated: new (each design's own new depth), 7, 6, 5, 4, 3, 2, 1.6 mm. Line value = load-weighted mean over sizes with normalised `T.SIZES` weights (as `T.evaluate`); hydroplaning = worst size (min), as in the brief.
- Every cited source is opened and verified before it goes into the paper; a figure that cannot be traced to a primary or clearly identified secondary source is not used.

## Plausibility check (done during brainstorming; to be verified from the sources)
Simulator, current product, new → ~2 mm: wet braking +24 %, steadily (+12 % at 4 mm); RRC −44 %.
Published: wet braking about +16–18 % at 2 mm and roughly flat down to 4 mm (DEKRA; European motoring-club study via ACL: 31.9 m new, 31.4 m at 4 mm, 37.3 m at 2 mm; Continental: +2.7 m at 3 mm, +6.9 m at 1.6 mm vs 8 mm); RR about −20 % over tread life (National Academies Special Report 286; Tire Rack). Hydroplaning: decline is well documented qualitatively; no numeric onset-vs-depth data found.
Verdict: direction right; wet shape wrong (too early), RR change about twice too strong.

## Approaches (picked: 1)
1. **Two worn models; the lead is the result.** *Picked.* (a) *extrapolated*: simulator as is, depth below design range; (b) *calibrated*: same simulator evaluated at an **effective depth** that reproduces the published trends for the current product, applied identically to both designs. New-tire behaviour unchanged.
2. As-is simulator plus plausibility table, claims limited to the lead — rejected: the "lead grows" claim would rest on a wet curve of the wrong shape.
3. Recalibrate the simulator's depth relations — rejected: changes every published number.

## Design
**Calibrated model = depth remapping.** For a remaining depth D, outputs are computed at an effective depth D_eff = φ_out(D), a monotone piecewise-linear map in D (identity at new depth), separate for wet-related outputs and for RR:
- φ_wet (used for wet braking and hydroplaning — both driven by drainage): knots chosen so that the current product's line-average wet braking is unchanged (0 %) at 4 mm and +17 % at 2 mm relative to new; linear between knots; extrapolate the 3→2 mm slope to 1.6 mm (and clip D_eff to ≥ 0.5 mm).
- φ_rr: chosen so that the current product's line-average RRC is −20 % at 2 mm relative to new, linear in D between new and 2 mm, same slope to 1.6 mm.
- Knot values are found numerically (bisection on depth) for the current product only, stored in the JSON, and then applied unchanged to the copilot design. Because each design keeps its own drainage, stiffness and pattern, design differences survive the remapping; only the depth axis is bent.
- Rationale to state in the paper: one-parameter-per-anchor calibration, anchored on the current product (as the EPREL wet-grip mapping already is), not tuned to the copilot design.

**Outputs.** `code/worn.py` → `results/worn.json` (per model, per depth, per design: line wet, worst-size hydro, line RRC, per-size wet; the φ knots; the published anchor values with source keys) and `figs/fig_worn.png`. Unit tests `code/test_worn.py` (stdlib unittest), pipeline step `worn` after `attribution` in all modes.

**Figure (one figure, 2 rows × 3 columns, 7.8 in wide, 300 dpi, `viz_common` colours).**
- Top row — wet braking (m), hydroplaning onset worst size (km/h), RRC (N/kN) against remaining depth (x from new on the left to 1.6 mm on the right). Current product NAVY, copilot ORANGE; calibrated model solid, extrapolated dashed. Grey band below 7.5 mm labelled "outside the simulator's design range".
- Bottom row — the copilot's lead (copilot − current, sign so that up = copilot better) per output and model, with a thin per-size range band for wet braking; zero line.
- Plausibility check: in the top-left (wet) and top-right (RRC) panels, published anchor points for the current product drawn as % change converted to the current product's new value (black open markers, legend "published worn-tire tests"). Exact visual layout left to the builder; must stay readable at print width.

**Manuscript (via `tst/content_tst.py`, numbers from `results/worn.json`, asserts on every sign the text states).**
- Experimental Design: "Experiment 5: worn tires" — what was evaluated, the two models, the calibration anchors and sources.
- Results: new H2 "Performance When Worn" after "Case Study": result for each output under both models, and the plausibility verdicts stated plainly (wet shape, RR magnitude, no numeric hydroplaning check).
- Limitations: one sentence on the calibration's thin data base.
- FIG. 10 caption; new references (UN R117 04 series; DEKRA; motoring-club study primary report if found, else ACL; Continental; National Academies SR 286). Placement after FIG. 9, so no renumbering of existing figures.
- If the lead does **not** hold for some output under some model, the text says so; the build asserts encode whichever result the data gives at writing time.

## Verified anchors (sources opened 2026-11-21)
- Wet braking: joint European motoring-club study reported by ACL Autotouring (J. Zabatta, 8 Sept 2026; five 185/65 R15 winter tyres): 31.9 m new, 31.4 m at 4 mm, 37.3 m at 2 mm (+17 %). DEKRA (M. Gaul, 7 Dec 2022; two sizes, from 100 km/h): +16–18 % at 2–3 mm. → anchors **0 % at 4 mm** ("barely moved"; the −1.6 % is treated as no change) and **+17 % at 2 mm**.
- Rolling resistance: TRB Special Report 286 (2006), ch. 4: "a new tire's rolling resistance typically declines by 20 percent or more as the tread diminishes to its worn-out depth"; literature range 20–40 % (Schuring 1980, via SR 286 fn. 10). → anchor **−20 % at 1.6 mm** (worn-out depth). The extrapolated model (−47 % at 1.6 mm) lies just beyond the upper end of the 20–40 % range, so the two models bracket the published range.
- Continental TechShow page: 404, dropped.
- Regulation: Regulation (EU) 2025/1453 (OJ L, 7.8.2025), the consolidated UN R117 04 series, covers "class C1 tyres in worn state" for wet adhesion (Annex 9). Entry-into-force dates differ between sources, so no date is stated in the paper.
- This replaces the provisional RR anchor (−20 % at 2 mm) in the Design section below: the RR anchor is at 1.6 mm.

## Decisions after the builder check
- **"The lead holds"** (agreed with the author): the copilot design is better than the current product at every evaluated depth, under both worn models. Figure gaps are in absolute units; the text also gives RRC gaps in percent. Hydroplaning is reported honestly as *no lead when new* (copilot −0.3 km/h) and the size of the gap when worn; it is never called a lead.
- *(Author's call: builder)* Calibration targets are **inputs derived from verified sources**, stored in `worn.json` with source keys, not constants. If a target loses all sources, that anchor is dropped, the corresponding output is reported under the extrapolated model only, and the text says so.
- *(Builder's call)* Wet calibration has **two anchors** (4 mm → 0 %, 2 mm → +x % from sources); φ_wet is linear between new and 4 mm (identity-like segment fitted to 0 %), between 4 and 2 mm, and continues the 4→2 mm slope to 1.6 mm. Continental's 3 mm figure is not an anchor (metres without the new-tire base distance); it may be cited qualitatively.
- *(Builder's call)* The brief's hydroplaning limit (76.5 km/h) is drawn as a dotted line labelled "brief limit (new tire)"; one sentence in Results notes that both designs fall below it when worn and that the brief sets it for new tires.

## Left to the builder
Exact knot search tolerances, marker styles, figure annotations, the per-size band styling, wording of captions within the decisions above.
