"""
benchmark_qnehvi.py — The design copilot with its qNEHVI engine (copilot_qnehvi.QnehviCopilot) under the benchmark protocol.

Same protocol as benchmark.py: 150 simulator evaluations, the copilot's 40 Latin-hypercube initial designs (same seed),
batches of 10, seeds 0-4, the default engineer gate (approves every batch), same hypervolume metric and compromise
score. Written to results/benchmark_qnehvi.json; results/benchmark.json is not touched.
Needs the extra packages in requirements-qnehvi.txt (torch, botorch). About 90 minutes per seed on 2 CPU threads.

Run from anywhere:  python benchmark_qnehvi.py [--seeds 0 1 2 3 4] [--workers 5]
"""
import os, sys, json, time, argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problem as PB
from benchmark import BUDGET, trace, weighted_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_ROOT, "results", "")
NAME = "qNEHVI (BoTorch)"
ENGINE_VERSION = 2          # 2 = run through DesignCopilot with the compound-constraint margin


def qnehvi(seed, verbose=False, budget=BUDGET):
    from copilot_qnehvi import QnehviCopilot
    cp = QnehviCopilot(budget=budget, seed=seed, verbose=verbose).run()
    F = np.array([PB.objectives(r) for r in cp.R]); G = np.array([PB.constraints(r) for r in cp.R])
    return F, G, np.array(cp.X), cp.log


def job(seed, verbose=False):
    t0 = time.time(); F, G, X, log = qnehvi(seed, verbose)
    return dict(method=NAME, seed=seed, engine_version=ENGINE_VERSION, trace=trace(F, G), final_hv=trace(F, G)[-1],
                best_score=min(weighted_score(f, g) for f, g in zip(F, G)), n_feasible=int(np.all(G <= 0, axis=1).sum()),
                seconds=time.time() - t0, F=F.tolist(), G=G.tolist(), X=X.tolist(), log=log)


if __name__ == "__main__":
    try:
        import botorch  # noqa: F401
    except ImportError:
        print("BoTorch not installed - skipping the qNEHVI run (pip install -r requirements-qnehvi.txt to enable it).")
        sys.exit(0)
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TIRE_WORKERS", "2"))); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.workers > 1 and len(a.seeds) > 1:
        with Pool(a.workers) as p:
            res = p.map(job, a.seeds, chunksize=1)
    else:
        res = [job(s, a.verbose) for s in a.seeds]
    res = sorted(res, key=lambda r: r["seed"])
    json.dump(res, open(RES + "benchmark_qnehvi.json", "w"))
    for r in res:
        print(f"seed {r['seed']}: final HV {r['final_hv']:.4f}  feasible {r['n_feasible']}  best score {r['best_score']:.3f}  {r['seconds']:.0f} s")
