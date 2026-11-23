# Tire Design Copilot: runnable project

This folder holds the complete code behind the paper *"An Agent-Orchestrated, Surrogate-Assisted Design Copilot for Replacement-Market Tire Lines"* (Diego de Carvalho Ribeiro, Tire Society 2027). Open the folder in VS Code to run every experiment yourself, watch the copilot work, and try your own tread designs.

> The virtual tire (`code/tiresim.py`) is a simplified, physics-informed model built to test AI design methods.
> It has **not** been calibrated against real tires, so its numbers show how the method behaves. They are not product performance.

## What the project does

1. A **virtual tire** (`code/tiresim.py`) takes 12 numbers describing a tread design (grooves, pitches, sipes, depth, compound, and so on). It draws the tread and computes the footprint pressure. From those it estimates rolling resistance, wet braking, treadwear, noise and hydroplaning on three sizes of one replacement line: 205/55R16, 225/65R17 and 265/70R17.
2. A **design brief** (`code/problem.py`) sets the goals and limits. The goals are lower rolling resistance, shorter wet braking and longer wear. The limits cover noise, hydroplaning, void ratio and a realistic compound.
3. The **design copilot** (`code/copilot.py`) runs a loop of six steps:
   - **generate**: a first spread of designs;
   - **verify**: test each design on every size;
   - **learn**: fit Gaussian-process predictors;
   - **optimize**: NSGA-II on the predictors;
   - **select**: pick a batch of 10;
   - **engineer gate**: someone approves the batch.

   The loop then repeats until the budget of 150 checks is spent.
4. **Experiments** compare the copilot with:
   - a conventional, expert-style process;
   - random sampling;
   - NSGA-II;
   - ParEGO.

   They also test lead-size versus whole-line design, check the virtual tire against 5,955 real EU-labelled tires, and run a sensitivity study on 8 perturbed versions of the simulator.

## Setup (once)

1. Install **Python 3.13** from python.org. On Windows, tick "Add python.exe to PATH".
2. Open this folder in VS Code and install the recommended **Python** extension when VS Code offers it.
3. Choose `Terminal > Run Task... > Setup: create .venv and install packages (Windows)`.
4. Choose `Ctrl+Shift+P > Python: Select Interpreter > .venv`.

If the pinned versions fail to install (for example on a newer Python), run `pip install numpy scipy pandas scikit-learn pymoo matplotlib` instead. Everything still runs, but the last digits may differ from the paper.

## Run (Run and Debug panel, Ctrl+Shift+D, pick a configuration, F5)

| Configuration | Time | What you get |
|---|---|---|
| 1. Play with a design | seconds | Edit `MY_DESIGN` in `play_design.py`. You get a table against the current product on 3 sizes, pass/fail on the brief, and `output/my_design.png` with the tread and footprints. |
| 2. Watch the copilot run | under 1 min to a few min | Prints each loop iteration (checks, hypervolume, feasible designs). Saves `output/copilot_live.png`. |
| 3. Copilot with YOU as engineer gate | a few min | Same loop, but each batch is shown in the terminal so you approve it or reject designs. |
| 4. Rebuild figures from saved results | ~2 min | Rebuilds every paper figure in `figs/` and `results/summary.json`. |
| 5. Quick reproduction | ~15 min | Also recomputes the 3,000-design dataset and the predictor learning curves. |
| 6. FULL reproduction | 1.5–3 h | Recomputes everything from scratch, including the benchmark (5 methods × 5 seeds) and the sensitivity study. |
| 7. Compare my results with the paper | seconds | Prints the key numbers next to the published ones. |

Configurations 4–6 finish by running the comparison automatically. `--workers N` sets how many CPU cores are used; the results do not depend on it.

From a terminal you can also run, for example, `python run_copilot_live.py --budget 80 --seed 3 --gate`.

## Folder layout

```
play_design.py        try your own design
run_copilot_live.py   watch / steer the copilot loop
run_pipeline.py       reproduce the paper (figures | quick | full)
verify_results.py     compare your numbers with the paper
code/                 the research code (unchanged from the paper)
  tiresim.py          virtual tire: pattern, footprint, performance relations
  problem.py          brief: objectives, constraints, hypervolume
  copilot.py          the agent loop (tools + orchestration + engineer gate)
  benchmark.py        the race: copilot vs 4 other methods
  reference.py        long NSGA-II runs for the best-known fronts (whole line / lead size)
  make_dataset.py     3,000 random designs
  surrogates.py       predictor learning curves
  sanity.py           qualitative checks of the simulator
  eprel_check.py      comparison with EU tire-label data (EPREL)
  eprel_download.js   browser script used to download the EPREL counts
  sensitivity.py      method ranking on 8 perturbed simulators
  analysis.py         all figures + summary.json
  attribution.py      Shapley attribution: why the copilot design beats the current product (fig_attribution.png)
  worn.py             worn-tire performance, new to 1.6 mm, two worn models (fig_worn.png)
  benchmark_qnehvi.py copilot with a qNEHVI (BoTorch) engine; extra packages in requirements-qnehvi.txt
  compare_qnehvi.py   qNEHVI engine vs the published strategies (fig_hv_qnehvi.png, qnehvi_summary.json)
  fig_framework.py    framework diagram
results/              raw results (your runs overwrite these)
figs/                 figures (your runs overwrite these)
published/            untouched copy of the paper's results and figures, for comparison
output/               pictures and logs from the play / live scripts
```

## Good places to start reading the code

- `code/tiresim.py`, function `evaluate_size`: every performance relation is in one place, with its coefficients.
- `code/copilot.py`, method `run`: the whole agent loop in about 15 lines.
- `code/benchmark.py`, function `expert`: how the conventional, expert-style process is emulated.

All random seeds are fixed, so with the versions in `requirements.txt` the results match the paper exactly.

## Desktop app

If you only want to evaluate your own designs, the **Tire Tread Simulator** is a free Windows program built on the same
simulator and copilot (no Python needed; it can also import a tread drawing as DXF):
https://github.com/ddcribeiro10-byte/tire-tread-simulator

## License and citation

MIT license (see `LICENSE`). If you use this code, please cite the paper; `CITATION.cff` has the details.

Contact: Diego de Carvalho Ribeiro, ddcribeiro@hotmail.com
