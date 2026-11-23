"""Benchmark surrogate models: accuracy vs training-set size (learning curves)."""
import numpy as np, pandas as pd, time, json, warnings
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import tiresim as T
warnings.filterwarnings("ignore")

TARGETS = ["rrc", "wet", "wear", "noise", "hydro_min", "void"]


def models():
    return {
        "Quadratic RSM (classical DOE)": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=np.logspace(-4, 3, 15))),
        "Gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=8),
        "Neural network (MLP)": lambda: TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128),
            alpha=1e-3, learning_rate_init=2e-3, max_iter=3000, early_stopping=True, n_iter_no_change=60, random_state=0)), transformer=StandardScaler()),
        "Gaussian process": lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
            C(1.0) * Matern(length_scale=np.ones(12), nu=2.5) + WhiteKernel(1e-3), normalize_y=True, n_restarts_optimizer=1, random_state=0)),
    }


if __name__ == "__main__":
    df = pd.read_csv("../results/dataset.csv")
    X = df[T.NAMES].values
    test = np.arange(2500, 3000)
    sizes = [30, 60, 120, 250, 500, 1000, 2000]
    rows = []
    for name, mk in models().items():
        for n in sizes:
            if name == "Gaussian process" and n > 500:
                continue
            tr = np.arange(n)
            for tgt in TARGETS:
                y = df[tgt].values
                m = mk(); t0 = time.time(); m.fit(X[tr], y[tr]); tfit = time.time() - t0
                t0 = time.time(); yp = m.predict(X[test]); tpred = (time.time() - t0) / len(test)
                rows.append(dict(model=name, n=n, target=tgt, r2=r2_score(y[test], yp), rmse=float(np.sqrt(np.mean((y[test] - yp) ** 2))),
                                 mape=100 * mean_absolute_percentage_error(y[test], yp), fit_s=tfit, pred_s=tpred))
            print(name, n, np.mean([r["r2"] for r in rows if r["model"] == name and r["n"] == n]).round(3), flush=True)
    pd.DataFrame(rows).to_csv("../results/learning_curves.csv", index=False)
