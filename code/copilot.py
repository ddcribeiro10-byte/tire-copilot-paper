"""
copilot.py — Agentic design-copilot loop (surrogate-assisted multi-objective optimisation).

The orchestrator owns the workflow state and calls 'tools':
  brief -> generate (space-filling) -> verify (expensive simulator) -> learn (surrogates)
        -> optimise (NSGA-II on surrogates) -> select (uncertainty-aware batch) -> human gate -> verify ...
Every step is logged so the run can be audited and a design-review report generated.
No LLM is required for the numerical loop; an LLM can sit on top for brief parsing and report writing.
"""
import numpy as np, json, time, warnings
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import tiresim as T
import problem as PB
warnings.filterwarnings("ignore")

OUT_KEYS = ["rrc", "wet", "wear", "noise", "hydro_min", "void"]


class SurrogateBank:
    """One GP per simulator output (inputs scaled to the unit cube)."""
    def __init__(self):
        self.models = {}

    def fit(self, X, R):
        Xu = (X - T.LB) / (T.UB - T.LB)
        for k in OUT_KEYS:
            y = np.array([r[k] for r in R])
            gp = GaussianProcessRegressor(Ck(1.0) * Matern(length_scale=np.full(12, 0.5), nu=2.5) + WhiteKernel(1e-4),
                                          normalize_y=True, n_restarts_optimizer=0, random_state=0)
            gp.fit(Xu, y)
            self.models[k] = gp
        return self

    def predict(self, X, std=False):
        Xu = (np.atleast_2d(X) - T.LB) / (T.UB - T.LB)
        mu, sd = {}, {}
        for k, gp in self.models.items():
            m, s = gp.predict(Xu, return_std=True)
            mu[k], sd[k] = m, s
        return (mu, sd) if std else mu


class SurrogateProblem(Problem):
    def __init__(self, bank, kappa=1.0):
        super().__init__(n_var=12, n_obj=3, n_ieq_constr=5, xl=T.LB, xu=T.UB)
        self.bank, self.kappa = bank, kappa

    def _evaluate(self, X, out, *a, **kw):
        mu, sd = self.bank.predict(X, std=True)
        k = self.kappa
        # optimistic (lower-confidence-bound) objectives encourage exploration of uncertain regions
        out["F"] = np.column_stack([mu["rrc"] - k * sd["rrc"], mu["wet"] - k * sd["wet"], -(mu["wear"] + k * sd["wear"])])
        C = PB.C
        out["G"] = np.column_stack([PB.compound_g(X), mu["noise"] - C["noise_max"], C["hydro_min"] - mu["hydro_min"],
                                    C["void_min"] - mu["void"], mu["void"] - C["void_max"]])


class DesignCopilot:
    def __init__(self, brief=PB.BRIEF, budget=150, n_init=40, batch=10, seed=0, human_gate=None, verbose=False):
        self.brief, self.budget, self.n_init, self.batch, self.seed = brief, budget, n_init, batch, seed
        self.human_gate = human_gate or (lambda cands, ctx: cands)   # default: engineer approves the batch
        self.X, self.R, self.log, self.verbose = [], [], [], verbose
        self.rng = np.random.default_rng(seed)

    # ---- tools ----
    def tool_generate(self, n, seed=None):
        seed = self.seed if seed is None else seed
        return qmc.scale(qmc.LatinHypercube(d=12, seed=seed).random(n), T.LB, T.UB)

    def tool_verify(self, Xb):
        for x in Xb:
            self.X.append(np.array(x)); self.R.append(T.evaluate(x))

    def tool_learn(self):
        return SurrogateBank().fit(np.array(self.X), self.R)

    def tool_optimise(self, bank, kappa):
        res = minimize(SurrogateProblem(bank, kappa), NSGA2(pop_size=80), ("n_gen", 40), seed=int(self.rng.integers(1e6)), verbose=False)
        return res.X if res.X is not None else self.tool_generate(self.batch, seed=int(self.rng.integers(1e9)))

    def tool_select(self, cands, bank, k):
        """Greedy batch: maximise predicted hypervolume gain, penalise points close to existing ones."""
        cands = np.atleast_2d(cands)
        mu = bank.predict(cands)
        Fc = np.column_stack([mu["rrc"], mu["wet"], -mu["wear"]])
        F, feas = self.current_front()
        chosen = []
        Xall_u = (np.array(self.X) - T.LB) / (T.UB - T.LB)
        cu = (cands - T.LB) / (T.UB - T.LB)
        for _ in range(min(k, len(cands))):
            best, bestv = None, -1
            base = PB.hypervolume(np.vstack([F[feas]] + [Fc[j][None] for j in chosen]) if (feas.any() or chosen) else np.zeros((0, 3)),
                                  np.ones(int(feas.sum()) + len(chosen), bool))
            for i in range(len(cands)):
                if i in chosen:
                    continue
                pool = np.vstack([F[feas]] + [Fc[j][None] for j in chosen] + [Fc[i][None]])
                gain = PB.hypervolume(pool, np.ones(len(pool), bool)) - base
                dmin = np.min(np.linalg.norm(np.vstack([Xall_u] + [cu[j][None] for j in chosen]) - cu[i], axis=1))
                v = gain + 0.002 * dmin
                if v > bestv:
                    best, bestv = i, v
            chosen.append(best)
        return cands[chosen]

    def current_front(self):
        F = np.array([PB.objectives(r) for r in self.R])
        G = np.array([PB.constraints(r) for r in self.R])
        return F, np.all(G <= 0, axis=1)

    def step_log(self, phase, **kw):
        F, feas = self.current_front()
        e = dict(phase=phase, evals=len(self.R), hv=PB.hypervolume(F, feas), n_feasible=int(feas.sum()), **kw)
        self.log.append(e)
        if self.verbose:
            print(e, flush=True)

    # ---- orchestration ----
    def run(self):
        self.tool_verify(self.tool_generate(self.n_init)); self.step_log("initial space-filling")
        it = 0
        while len(self.R) < self.budget:
            it += 1
            t0 = time.time()
            bank = self.tool_learn()
            kappa = 1.0 if len(self.R) < 0.7 * self.budget else 0.2            # explore early, exploit late
            cands = self.tool_optimise(bank, kappa)
            batch = self.tool_select(cands, bank, min(self.batch, self.budget - len(self.R)))
            batch = self.human_gate(batch, dict(iteration=it, bank=bank))
            self.tool_verify(batch)
            self.step_log(f"iteration {it}", kappa=kappa, seconds=round(time.time() - t0, 1))
        return self


if __name__ == "__main__":
    cp = DesignCopilot(seed=0, verbose=True).run()
    json.dump(cp.log, open("../results/copilot_demo_log.json", "w"), indent=1)
