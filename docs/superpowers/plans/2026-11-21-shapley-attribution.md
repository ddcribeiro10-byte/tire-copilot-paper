# Shapley Attribution Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a figure + JSON that split the current-product → copilot-design change in RRC, wet braking and treadwear into exact grouped-Shapley contributions of 7 design-variable groups.

**Architecture:** New flat module `code/attribution.py` (same style as the other `code/` scripts: run with `code/` as cwd, relative `../results/`, `../figs/`). Pure functions for coalitions and Shapley values, a `__main__` block that evaluates 128 coalition designs on the simulator, writes `results/attribution.json`, draws `figs/fig_attribution.png`. Hooked into `run_pipeline.py` after `analysis`.

**Tech Stack:** Python 3.13, numpy, matplotlib (existing pinned deps); tests with stdlib `unittest` (no new dependency).

**Spec:** `docs/superpowers/specs/2026-11-21-shapley-attribution-design.md`

## Global Constraints

- Do not modify `code/tiresim.py`, `code/problem.py`, `code/analysis.py`, `verify_results.py`, `published/`, or any existing result/figure file.
- No GP/surrogate in the attribution; call `T.evaluate(x, detail=True)` only.
- Groups (indices into `T.NAMES`): compound [8, 9]; circumferential grooves [0, 1]; lateral grooves [3, 4, 5]; sipes [6]; tread depth [7]; pitching [2, 10]; crown [11].
- Pair: `T.BASELINE` → `np.load("../results/case_design.npy")`.
- Efficiency check: contributions sum to total change within 1e-6, else raise.
- Figure: 300 dpi, 7.8 in wide, colors from `viz_common` (`GREEN` = helps, `RED` = hurts, `NAVY` text), RRC unit label `"N/kN"` as in `analysis.py`.
- Project is not a git repo: no commit steps.

## Review Focus

- Integer variables (n_circ, n_pitch, sipes) are rounded inside `decode`; a coalition mixing a rounded and non-rounded value must still be a valid design → covered by evaluating real coalitions in Task 2's check.
- Line average must use the normalized `weight` of `T.SIZES` exactly as `T.evaluate` does, so the line totals match `S["case"]["line_ai"]`/`line_base` in `results/summary.json` → test in Task 2.
- Wear is "higher is better": helps/hurts coloring must flip for wear → test in Task 2 (`helps_sign`).
- Running `attribution.py` before `case_design.npy` exists must fail with a clear message, not a traceback deep in numpy → handled in Task 2 main.
- A group whose baseline and copilot values are identical must get exactly 0 contribution (Shapley dummy property) → test in Task 1.

---

### Task 0: Environment

- [ ] Create venv with Python 3.13 and install pinned deps: `py -V:Astral\CPython3.13.17 -m venv .venv` then `.venv\Scripts\python -m pip install -r requirements.txt`.
- [ ] Verify: `cd code && ..\.venv\Scripts\python tiresim.py` prints three size rows and a line-metrics dict.
- [ ] Verify: `..\.venv\Scripts\python ..\verify_results.py` (from `code/`, or from root without `..`) reports the saved results match the published ones.

### Task 1: Shapley core (pure functions)

**Files:**
- Create: `code/attribution.py`
- Test: `code/test_attribution.py`

**Interfaces:**
- Produces:
  - `GROUPS: dict[str, list[int]]` — ordered as in Global Constraints, keys `"Compound"`, `"Circumferential grooves"`, `"Lateral grooves"`, `"Sipes"`, `"Tread depth"`, `"Pitching"`, `"Crown"`.
  - `coalition_design(x0: np.ndarray, x1: np.ndarray, mask: int) -> np.ndarray` — copy of `x0` with the variables of group `i` taken from `x1` when bit `i` of `mask` is set.
  - `shapley(values: np.ndarray) -> np.ndarray` — `values` has shape `(2**n, ...)` indexed by mask; returns shape `(n, ...)` exact Shapley values (weight `|S|!(n-|S|-1)!/n!` on `v(S∪{i}) - v(S)`).

- [ ] **Step 1: Write failing tests** in `code/test_attribution.py` (`unittest`, imports `attribution as A`):
  - `test_coalition_design_masks`: `x0 = zeros(12)`, `x1 = ones(12)`; mask 0 → all 0; mask `2**7-1` → all 1; mask `1` → ones exactly at indices `[8, 9]`.
  - `test_shapley_additive_game`: `v(S) = sum of w[i] for i in S` with `w = [1,2,3]` → `shapley` returns `[1,2,3]`.
  - `test_shapley_efficiency_interaction`: `n=3`, `v(S) = 1 if S == {0,1} or S ⊇ {0,1} else 0` → result `[0.5, 0.5, 0.0]`, sum equals `v(full) - v(empty)`.
  - `test_shapley_vector_values`: values shape `(8, 2)` stacking two games → result shape `(3, 2)` and each column matches its scalar game.
  - `test_dummy_group_zero`: game where player 2 never changes `v` → contribution of 2 is exactly `0.0`.
- [ ] **Step 2:** Run `cd code && ..\.venv\Scripts\python -m unittest test_attribution -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `GROUPS`, `coalition_design`, `shapley` in `code/attribution.py` (module docstring in the style of the other `code/` files; import `tiresim as T`).
- [ ] **Step 4:** Rerun → all PASS.

### Task 2: Simulator run, JSON, figure

**Files:**
- Modify: `code/attribution.py`
- Test: `code/test_attribution.py`

**Interfaces:**
- Consumes: Task 1 `GROUPS`, `coalition_design`, `shapley`.
- Produces:
  - `OBJ = ["rrc", "wet", "wear"]`, `EXTRA = ["noise", "hydro"]`.
  - `helps_sign(key: str) -> int` — `-1` for rrc, wet, noise (lower is better), `+1` for wear, hydro.
  - `attribute(x0, x1) -> dict` — evaluates all `2**7` coalitions with `T.evaluate(x, detail=True)`; returns `{"groups": [...names], "sizes": [...size names], "weights": [...normalized], "base": {key: {"line": float, "per_size": [3]}}, "copilot": {same}, "contrib": {key: {"line": [7], "per_size": [[7] x 3]}}}` for keys `OBJ + EXTRA`. Line contribution = weighted average of per-size contributions with normalized `T.SIZES` weights. Raises `AssertionError` if `abs(sum(contrib line) - (copilot line - base line)) > 1e-6` for any key.
  - `plot(att: dict, path: str) -> None`.
  - `__main__`: load `../results/case_design.npy` (if missing: `sys.exit("results/case_design.npy not found - run analysis.py first (run_pipeline.py --mode figures).")`), `attribute(T.BASELINE, xb)`, dump `../results/attribution.json` (indent 1, floats rounded to 4 decimals), `plot(att, "../figs/fig_attribution.png")`, print the line-level table (group × objective).

- [ ] **Step 1: Write failing tests:**
  - `test_helps_sign`: `helps_sign("wear") == 1`, `helps_sign("wet") == -1`, `helps_sign("rrc") == -1`.
  - `test_attribute_identity_pair`: `attribute(T.BASELINE, T.BASELINE)` → every contribution `== 0.0` (takes ~128 evals; acceptable, mark the test's docstring "slow").
  - `test_attribute_matches_evaluate` (skip if `../results/case_design.npy` missing): `att = attribute(T.BASELINE, xb)`; for each key in `OBJ`, `att["base"][key]["line"]` equals `T.evaluate(T.BASELINE)[key]` and `att["copilot"][key]["line"]` equals `T.evaluate(xb)[key]` (abs 1e-9); `sum(att["contrib"][key]["line"])` equals their difference (abs 1e-6).
- [ ] **Step 2:** Run tests → new ones FAIL.
- [ ] **Step 3:** Implement `helps_sign`, `attribute`, `__main__` JSON part. Evaluate each coalition once and read all keys from the `per_size` list (one `T.evaluate` call per mask).
- [ ] **Step 4:** Implement `plot`: `fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.6))`, one panel per `OBJ` with titles `f"RRC, N/kN (↓)"`, `"Wet braking 80–20 km/h, m (↓)"`, `"Treadwear index (↑)"`. In each panel: groups sorted by `|line contribution|` descending; horizontal waterfall — top row "Current product" bar at base value, then one floating bar per group from running total to running total + contribution (color `GREEN` if `contrib * helps_sign(key) > 0` else `RED`, `GREY` if exactly 0), last row "Copilot design" bar at copilot value; 3 small markers per group bar at `running_start + per_size contribution` (markers `o`, `s`, `^` for the 3 sizes, black edge, shared legend with short size names below the figure); x-limits padded so all bars and markers are visible; group names on the y-axis of the left panel only (`sharey` style), `viz_common` styling. `fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight")`.
- [ ] **Step 5:** Run tests → all PASS.
- [ ] **Step 6:** Run `cd code && ..\.venv\Scripts\python attribution.py` → prints the table, writes `results/attribution.json` and `figs/fig_attribution.png`. Open the PNG and check: base/copilot bars equal the summary.json `case.line_base`/`case.line_ai` values; bar labels readable; markers visible.
- [ ] **Step 7:** Cross-check with the existing split: `compound` line contribution has the same sign as `case.compound_only − case.line_base` in `results/summary.json` for each objective (report, don't assert — interactions can differ).

### Task 3: Pipeline hook + docs

**Files:**
- Modify: `run_pipeline.py` (`STEPS`, `MODES`, module docstring)
- Modify: `CLAUDE.md`, `README.md` (folder-layout list: one line for `attribution.py`)

- [ ] **Step 1:** Add `"attribution": (["attribution.py"], "Shapley attribution: current product -> copilot design")` to `STEPS` and insert `"attribution"` right after `"analysis"` in all three `MODES` lists.
- [ ] **Step 2:** Run `.venv\Scripts\python run_pipeline.py --mode figures --workers 4` → all steps finish, attribution step included, and `verify_results.py` at the end still reports the published numbers matched (attribution doesn't touch `summary.json`).
- [ ] **Step 3:** Add `attribution.py` line to README folder layout and the CLAUDE.md architecture list; mention `python -m unittest test_attribution` (from `code/`) as the test command in CLAUDE.md.
