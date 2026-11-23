"""Generate the simulation dataset (Latin hypercube over the 12-D design space)."""
import os
import numpy as np, pandas as pd, time
from multiprocessing import Pool
from scipy.stats import qmc
import tiresim as T

KEYS = ["rrc", "wet", "wear", "noise", "hydro_min", "void", "cov", "g_compound", "wet_worst", "wear_worst", "rrc_spread"]


def run(x):
    r = T.evaluate(x)
    return [r[k] for k in KEYS]


if __name__ == "__main__":
    N = 3000
    X = qmc.scale(qmc.LatinHypercube(d=12, seed=2026).random(N), T.LB, T.UB)
    t0 = time.time()
    with Pool(int(os.environ.get("TIRE_WORKERS", "2"))) as p:
        Y = p.map(run, X, chunksize=50)
    dt = time.time() - t0
    df = pd.DataFrame(np.hstack([X, np.array(Y)]), columns=T.NAMES + KEYS)
    df.to_csv("../results/dataset.csv", index=False)
    print(f"{N} designs in {dt:.0f}s ({dt / N * 2:.3f} s/design single-core)")
