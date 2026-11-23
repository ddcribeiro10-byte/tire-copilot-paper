"""
copilot_qnehvi.py — The design copilot with a qNEHVI optimization engine.

Subclass of copilot.DesignCopilot that replaces only the engine tools (learn, optimise, select) with BoTorch:
one SingleTaskGP per output (3 objectives as -normalised F, 4 simulator constraints scaled by G_SCALE) and batch
qLogNoisyExpectedHypervolumeImprovement with the analytic compound constraint as a linear inequality (with a small
safety margin so boundary designs are not lost to rounding). Everything else — initial Latin hypercube, multi-size
verification, the engineer gate callback, the step log, budget and batch size — is the unchanged DesignCopilot.run.
Needs requirements-qnehvi.txt (torch, botorch).
"""
import os, warnings
import numpy as np
import tiresim as T, problem as PB
from copilot import DesignCopilot

G_SCALE = np.array([0.05, 0.5, 2, 0.02, 0.02])      # same constraint scales as benchmark.py
MC_SAMPLES, RESTARTS, RAW_SAMPLES, MAXITER = 64, 4, 128, 100   # lighter than BoTorch tutorial defaults (128/10/512)
COMPOUND_MARGIN = 1e-6                              # keep proposals strictly inside the compound constraint
THREADS = int(os.environ.get("TIRE_TORCH_THREADS", "2"))


def compound_linear():
    """tand0 - 2.6 tand60 <= 0.18 in unit-cube coordinates, as BoTorch's  sum(c * u[idx]) >= rhs."""
    a9, a8 = T.UB[9] - T.LB[9], T.UB[8] - T.LB[8]
    rhs = 0.18 - T.LB[9] + 2.6 * T.LB[8]                    # a9*u9 - 2.6*a8*u8 <= rhs
    return [-a9, 2.6 * a8], -rhs


class QnehviCopilot(DesignCopilot):
    """DesignCopilot whose learn/optimise/select tools are a BoTorch qNEHVI engine."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import torch
        torch.manual_seed(self.seed); torch.set_num_threads(THREADS)
        warnings.filterwarnings("ignore")

    def _data(self):
        import torch
        X = np.array(self.X); F, feas = self.current_front()
        G = np.array([PB.constraints(r) for r in self.R])
        U = (X - T.LB) / (T.UB - T.LB)
        Y = np.column_stack([-PB.normalise(F), G[:, 1:] / G_SCALE[1:]])     # maximise -F; constraints <= 0 feasible
        tk = dict(dtype=torch.double)
        return torch.tensor(U, **tk), torch.tensor(Y, **tk)

    def tool_learn(self):
        from botorch.models import SingleTaskGP, ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import SumMarginalLogLikelihood
        Xt, Yt = self._data()
        model = ModelListGP(*[SingleTaskGP(Xt, Yt[:, i:i + 1], outcome_transform=Standardize(m=1)) for i in range(Yt.shape[1])])
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        return dict(model=model, X=Xt)

    def tool_optimise(self, bank, kappa):
        """Returns the qNEHVI batch (kappa is the default engine's exploration setting and is not used here)."""
        import torch
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
        from botorch.sampling import SobolQMCNormalSampler
        from botorch.optim import optimize_acqf
        tk = dict(dtype=torch.double)
        coef, rhs = compound_linear()
        ineq = [(torch.tensor([9, 8]), torch.tensor(coef, **tk), float(rhs) + COMPOUND_MARGIN)]
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=bank["model"], ref_point=torch.tensor([-1.0, -1.0, -1.0], **tk), X_baseline=bank["X"], prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
            constraints=[(lambda Z, j=j: Z[..., 3 + j]) for j in range(4)])
        q = min(self.batch, self.budget - len(self.R))
        cand, _ = optimize_acqf(acq, bounds=torch.stack([torch.zeros(12, **tk), torch.ones(12, **tk)]), q=q,
                                num_restarts=RESTARTS, raw_samples=RAW_SAMPLES, sequential=True,
                                inequality_constraints=ineq, options={"batch_limit": RESTARTS, "maxiter": MAXITER})
        U = np.clip(cand.detach().numpy(), 0, 1)
        return T.LB + U * (T.UB - T.LB)

    def tool_select(self, cands, bank, k):
        """qNEHVI already chose the batch jointly; pass it on."""
        return np.atleast_2d(cands)[:k]
