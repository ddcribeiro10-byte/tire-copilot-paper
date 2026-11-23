"""Large-budget NSGA-II runs on the simulator:
 (a) full replacement line (3 sizes)  -> best-known Pareto front (reference for hypervolume)
 (b) lead size only (205/55R16)       -> to show what happens when a line is designed on one size."""
import os
import numpy as np, json, sys
from multiprocessing import Pool
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import tiresim as T, problem as PB


class P(ElementwiseProblem):
    def __init__(self, sizes, **kw):
        super().__init__(n_var=12, n_obj=3, n_ieq_constr=5, xl=T.LB, xu=T.UB, **kw); self.sizes = sizes
    def _evaluate(self, x, out, *a, **k):
        r = T.evaluate(x, sizes=self.sizes)
        out["F"] = PB.objectives(r); out["G"] = PB.constraints(r)


if __name__ == "__main__":
    which = sys.argv[1]
    sizes = T.SIZES if which == "line" else T.SIZES[:1]
    with Pool(int(os.environ.get("TIRE_WORKERS", "2"))) as pool:
        prob = P(sizes, elementwise_runner=StarmapParallelization(pool.starmap))
        res = minimize(prob, NSGA2(pop_size=100), ("n_gen", 60), seed=7, verbose=False)
    X = res.X; F = res.F
    json.dump(dict(X=X.tolist(), F=F.tolist(), hv=PB.hypervolume(F, np.ones(len(F), bool))),
              open(f"../results/reference_{which}.json", "w"))
    print(which, len(F), PB.hypervolume(F, np.ones(len(F), bool)))
