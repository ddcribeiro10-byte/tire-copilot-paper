"""Sensitivity of the method ranking to the simulator's coefficients.
Eight variants scale nine key coefficients by factors drawn by Latin hypercube from [0.8, 1.2];
every strategy is rerun with three seeds per variant. Hypervolume is normalized per variant by the
merged feasible front of all runs in that variant."""
import os
import json, numpy as np, time
from multiprocessing import Pool
from scipy.stats import qmc
import tiresim as T, problem as PB
import benchmark as BM
from copilot import DesignCopilot

KEYS = ["rr_tread", "wet_edge", "wet_drain", "wet_compound", "hydro_drain", "wear_pressure", "wear_peak", "noise_tonal", "noise_pump"]
FACT = qmc.scale(qmc.LatinHypercube(d=len(KEYS), seed=99).random(8), 0.8, 1.2)
SEEDS = [0, 1, 2]


def copilot_nosave(seed):
    cp = DesignCopilot(budget=BM.BUDGET, seed=seed).run()
    return np.array([PB.objectives(r) for r in cp.R]), np.array([PB.constraints(r) for r in cp.R])


METHODS = dict(BM.METHODS); METHODS["AI copilot (surrogate + agent)"] = copilot_nosave


F0_DEFAULT = PB.objectives(T.evaluate(T.BASELINE)).copy()
LO0, HI0 = PB.F_LO.copy(), PB.F_HI.copy()


def set_variant(v):
    """Apply coefficient factors and restate the brief relative to the current product, as in the
    main study: noise no worse than the current product (+0.07 dB(A)), hydroplaning margin 0.85 km/h,
    and the hypervolume box shifted with the current product."""
    T.PERT.clear(); T.PERT.update(dict(zip(KEYS, FACT[v])))
    r0 = T.evaluate(T.BASELINE)
    PB.C["noise_max"] = r0["noise"] + 0.07
    PB.C["hydro_min"] = r0["hydro_min"] - 0.85
    shift = PB.objectives(r0) - F0_DEFAULT
    PB.F_LO[:] = LO0 + shift; PB.F_HI[:] = HI0 + shift


def job(args):
    v, m, s = args
    set_variant(v)
    F, G = METHODS[m](s)
    return dict(variant=v, method=m, seed=s, F=F.tolist(), G=G.tolist(),
                best_score=min(BM.weighted_score(f, g) for f, g in zip(F, G)))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(v, m, s) for v in range(len(FACT)) for m in METHODS for s in SEEDS]
    with Pool(int(os.environ.get("TIRE_WORKERS", "2"))) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open("../results/sensitivity_raw.json", "w"))
    out = {"keys": KEYS, "factors": FACT.round(3).tolist(), "variants": []}
    for v in range(len(FACT)):
        rr = [r for r in res if r["variant"] == v]
        set_variant(v)
        pool = np.vstack([np.array(r["F"])[np.all(np.array(r["G"]) <= 0, axis=1)] for r in rr if np.any(np.all(np.array(r["G"]) <= 0, axis=1))])
        ref = PB.hypervolume(pool, np.ones(len(pool), bool))
        hv = {m: [PB.hypervolume(np.array(r["F"]), np.all(np.array(r["G"]) <= 0, axis=1)) / ref * 100 for r in rr if r["method"] == m] for m in METHODS}
        sc = {m: [r["best_score"] for r in rr if r["method"] == m] for m in METHODS}
        means = {m: float(np.mean(hv[m])) for m in METHODS}
        rank = sorted(METHODS, key=lambda m: -means[m])
        out["variants"].append(dict(hv_mean={m: round(means[m], 1) for m in METHODS},
                                    best_score_mean={m: round(float(np.mean(sc[m])), 3) for m in METHODS},
                                    ranking=rank, copilot_first=rank[0] == "AI copilot (surrogate + agent)",
                                    copilot_best_score=min(sc, key=lambda m: np.mean(sc[m])) == "AI copilot (surrogate + agent)"))
        print(v, out["variants"][-1]["hv_mean"], flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open("../results/sensitivity.json", "w"), indent=1)
