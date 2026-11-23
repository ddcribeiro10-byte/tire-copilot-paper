"""Replacement-line design problem definition: objectives, constraints, hypervolume."""
import numpy as np
import tiresim as T

BRIEF = dict(
    product="Premium all-season replacement line, 3 representative sizes (205/55R16 - 265/70R17)",
    objectives=["min rolling resistance coefficient (kg/t)", "min wet braking distance 80-20 km/h (m)",
                "max treadwear index (warranty mileage)"],
    constraints=dict(noise_max=71.3, hydro_min=76.5, void_min=0.20, void_max=0.36),
)
C = BRIEF["constraints"]
OBJ_KEYS = ["rrc", "wet", "wear"]
# normalisation box for hypervolume (objectives in minimisation form: rrc, wet, -wear)
F_LO = np.array([6.0, 26.0, -170.0])
F_HI = np.array([11.0, 38.0, -60.0])     # reference point = worst corner


def objectives(r):
    return np.array([r["rrc"], r["wet"], -r["wear"]])


def constraints(r, x=None):
    """g <= 0 feasible."""
    return np.array([r["g_compound"], r["noise"] - C["noise_max"], C["hydro_min"] - r["hydro_min"],
                     C["void_min"] - r["void"], r["void"] - C["void_max"]])


def compound_g(X):
    X = np.atleast_2d(X)
    return X[:, 9] - (0.18 + 2.6 * X[:, 8])


def normalise(F):
    return (np.atleast_2d(F) - F_LO) / (F_HI - F_LO)


def hypervolume(F, feasible):
    from pymoo.indicators.hv import HV
    F = np.atleast_2d(F)[np.asarray(feasible, bool)]
    if len(F) == 0:
        return 0.0
    Fn = normalise(F)
    Fn = Fn[np.all(Fn < 1.0, axis=1)]
    if len(Fn) == 0:
        return 0.0
    return float(HV(ref_point=np.ones(3))(Fn))


def evaluate_full(x):
    r = T.evaluate(x)
    return objectives(r), constraints(r), r
