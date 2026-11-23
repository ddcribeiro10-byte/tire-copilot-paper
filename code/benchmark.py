"""Compare design strategies under a fixed budget of expensive evaluations."""
import os
import numpy as np, pandas as pd, json, sys, time
from scipy.stats import qmc
from multiprocessing import Pool
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import tiresim as T, problem as PB
from copilot import DesignCopilot

BUDGET = 150
SPAN = T.UB - T.LB


def trace(F, G):
    feas = np.all(G <= 0, axis=1)
    return [PB.hypervolume(F[:i], feas[:i]) for i in range(10, len(F) + 1, 10)]


def run_list(X):
    out = [PB.evaluate_full(x) for x in X]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


W_SCORE = np.array([0.35, 0.40, 0.25])
G_SCALE = np.array([0.05, 0.5, 2, 0.02, 0.02])


def weighted_score(f, g):
    """Scalar compromise used by the emulated conventional process (lower is better)."""
    return float(np.dot(W_SCORE, PB.normalise(f)[0]) + 5 * np.sum(np.maximum(g, 0) / G_SCALE))


def expert(seed):
    """Emulates the conventional process: start from the current product, change one or two
    parameters at a time in loops of 10 designs, keep the best compromise (weighted score)."""
    rng = np.random.default_rng(seed)
    score = weighted_score
    best = T.BASELINE.copy(); fb, gb, _ = PB.evaluate_full(best); sb = score(fb, gb)
    Fs, Gs = [fb], [gb]
    while len(Fs) < BUDGET:
        loop = []
        for _ in range(min(10, BUDGET - len(Fs))):
            x = best.copy()
            for j in rng.choice(12, size=rng.integers(1, 3), replace=False):
                x[j] = np.clip(x[j] + rng.normal(0, 0.15) * SPAN[j], T.LB[j], T.UB[j])
            f, g, _ = PB.evaluate_full(x); Fs.append(f); Gs.append(g); loop.append((score(f, g), x))
        s, x = min(loop, key=lambda t: t[0])
        if s < sb:
            sb, best = s, x
    return np.array(Fs), np.array(Gs)


def random_lhs(seed):
    X = qmc.scale(qmc.LatinHypercube(d=12, seed=100 + seed).random(BUDGET), T.LB, T.UB)
    return run_list(X)


class SimProblem(ElementwiseProblem):
    def __init__(self, store):
        super().__init__(n_var=12, n_obj=3, n_ieq_constr=5, xl=T.LB, xu=T.UB); self.store = store
    def _evaluate(self, x, out, *a, **k):
        f, g, _ = PB.evaluate_full(x); self.store.append((f, g)); out["F"] = f; out["G"] = g


def nsga_direct(seed):
    store = []
    minimize(SimProblem(store), NSGA2(pop_size=30), ("n_evals", BUDGET), seed=seed, verbose=False)
    F = np.array([s[0] for s in store])[:BUDGET]; G = np.array([s[1] for s in store])[:BUDGET]
    return F, G


def copilot(seed):
    cp = DesignCopilot(budget=BUDGET, seed=seed).run()
    F = np.array([PB.objectives(r) for r in cp.R]); G = np.array([PB.constraints(r) for r in cp.R])
    np.save(f"../results/copilot_X_seed{seed}.npy", np.array(cp.X))
    return F, G


def parego(seed):
    """ParEGO-style multi-objective Bayesian optimization (Knowles 2006), batch version: each batch
    member uses its own random weight vector on an augmented Tchebycheff scalarization (constraint
    violations penalized), a GP on that scalar, and expected improvement over random candidates."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
    from scipy.stats import norm
    rng = np.random.default_rng(1000 + seed)
    X = list(qmc.scale(qmc.LatinHypercube(d=12, seed=seed).random(40), T.LB, T.UB))
    FG = [PB.evaluate_full(x)[:2] for x in X]
    while len(X) < BUDGET:
        Xa = np.array(X); Xu = (Xa - T.LB) / (T.UB - T.LB)
        Fn = PB.normalise(np.array([f for f, g in FG])); Gv = np.array([g for f, g in FG])
        pen = 5 * np.sum(np.maximum(Gv, 0) / G_SCALE, axis=1)
        batch = []
        for _ in range(min(10, BUDGET - len(X))):
            w = rng.dirichlet(np.ones(3))
            y = np.max(w * Fn, axis=1) + 0.05 * np.sum(w * Fn, axis=1) + pen
            gp = GaussianProcessRegressor(Ck(1.0) * Matern(length_scale=np.full(12, 0.5), nu=2.5) + WhiteKernel(1e-4),
                                          normalize_y=True, random_state=0).fit(Xu, y)
            cand = rng.random((3000, 12))
            best = Xu[np.argsort(y)[:5]]
            cand = np.vstack([cand, np.clip(best[rng.integers(0, 5, 1000)] + rng.normal(0, 0.05, (1000, 12)), 0, 1)])
            cx = T.LB + cand * (T.UB - T.LB)
            ok = PB.compound_g(cx) <= 0
            cand = cand[ok]
            mu, sd = gp.predict(cand, return_std=True)
            imp = y.min() - mu; z = imp / np.maximum(sd, 1e-9)
            ei = imp * norm.cdf(z) + sd * norm.pdf(z)
            for xb in batch:                      # avoid duplicates inside the batch
                ei[np.linalg.norm(cand - (xb - T.LB) / (T.UB - T.LB), axis=1) < 0.02] = -1
            batch.append(T.LB + cand[int(np.argmax(ei))] * (T.UB - T.LB))
        for x in batch:
            X.append(x); FG.append(PB.evaluate_full(x)[:2])
    return np.array([f for f, g in FG]), np.array([g for f, g in FG])


METHODS = {"Conventional (expert loops)": expert, "Random sampling (LHS)": random_lhs,
           "NSGA-II on simulator": nsga_direct, "ParEGO (MOBO)": parego,
           "AI copilot (surrogate + agent)": copilot}


def job(args):
    name, seed = args
    t0 = time.time(); F, G = METHODS[name](seed)
    best_score = min(weighted_score(f, g) for f, g in zip(F, G))
    return dict(method=name, seed=seed, trace=trace(F, G), final_hv=trace(F, G)[-1], best_score=best_score,
                n_feasible=int(np.all(G <= 0, axis=1).sum()), seconds=time.time() - t0,
                F=F.tolist(), G=G.tolist())


if __name__ == "__main__":
    seeds = range(5)
    jobs = [(m, s) for m in METHODS for s in seeds]
    with Pool(int(os.environ.get("TIRE_WORKERS", "2"))) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open("../results/benchmark.json", "w"))
    df = pd.DataFrame([{k: v for k, v in r.items() if k in ("method", "seed", "final_hv", "n_feasible", "seconds")} for r in res])
    print(df.groupby("method")[["final_hv", "n_feasible", "seconds"]].agg(["mean", "std"]).round(3))
