"""
run_pipeline.py  -  reproduce the paper's results and figures.

Modes:
  --mode figures   (~2 min)  rebuild every figure and summary.json from the saved raw results
  --mode quick     (~15 min) also re-run the 3,000-design dataset and the surrogate learning curves
  --mode full      (~1.5-3 h) re-run EVERYTHING from scratch: dataset, benchmark (5 methods x 5 seeds),
                              reference fronts, surrogates, sanity, figures, attribution, worn tires, EPREL check, sensitivity

Results are written to results/ and figs/. The published versions stay untouched in published/,
and verify_results.py compares the two at the end.

Speed: set the number of CPU cores used with --workers (default 2). Results do not depend on it.
"""
import os, sys, time, subprocess, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")

STEPS = {
    "dataset":     (["make_dataset.py"], "3,000 random designs evaluated on the virtual tire"),
    "benchmark":   (["benchmark.py"], "the race: 5 methods x 5 seeds x 150 checks"),
    "ref_line":    (["reference.py", "line"], "best-known front, whole line (long NSGA-II)"),
    "ref_lead":    (["reference.py", "lead"], "best-known front, lead size only"),
    "surrogates":  (["surrogates.py"], "how fast 4 predictor types learn"),
    "sanity":      (["sanity.py"], "qualitative checks of the simulator"),
    "framework":   (["fig_framework.py"], "framework diagram"),
    "analysis":    (["analysis.py"], "all figures + summary.json"),
    "attribution": (["attribution.py"], "Shapley attribution: current product -> copilot design"),
    "worn":        (["worn.py"], "worn-tire performance: current product vs copilot design, two worn models"),
    "qnehvi_run":  (["benchmark_qnehvi.py"], "copilot with the qNEHVI engine, 5 seeds (needs requirements-qnehvi.txt; ~1.5 h)"),
    "qnehvi_cmp":  (["compare_qnehvi.py"], "qNEHVI engine vs published strategies + hypervolume figure"),
    "eprel":       (["eprel_check.py"], "comparison with 5,955 real EU-labelled tires"),
    "sensitivity": (["sensitivity.py"], "ranking on 8 perturbed simulators"),
}
MODES = {
    "figures": ["sanity", "framework", "analysis", "attribution", "worn", "qnehvi_cmp", "eprel"],
    "quick": ["dataset", "surrogates", "sanity", "framework", "analysis", "attribution", "worn", "qnehvi_cmp", "eprel"],
    "full": ["dataset", "benchmark", "ref_line", "ref_lead", "surrogates", "sanity", "framework",
             "analysis", "attribution", "worn", "qnehvi_run", "qnehvi_cmp", "eprel", "sensitivity"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="figures")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    env = dict(os.environ, TIRE_WORKERS=str(a.workers), MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True); os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)

    print(f"Mode '{a.mode}', {a.workers} worker(s)\n")
    t_all = time.time()
    for i, key in enumerate(MODES[a.mode], 1):
        args, what = STEPS[key]
        print(f"[{i}/{len(MODES[a.mode])}] {' '.join(args):<24} {what} ...", flush=True)
        t0 = time.time()
        p = subprocess.run([sys.executable] + args, cwd=CODE, env=env)
        if p.returncode != 0:
            sys.exit(f"\nStep '{key}' failed (exit code {p.returncode}). See the error above.")
        print(f"      done in {time.time() - t0:.0f} s\n", flush=True)
    print(f"All steps finished in {(time.time() - t_all) / 60:.1f} min.\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "verify_results.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
