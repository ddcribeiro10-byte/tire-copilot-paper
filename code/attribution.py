"""
attribution.py — Why does the copilot design beat the current product?

Exact grouped Shapley attribution of the change in each objective (current product -> copilot case design)
to 7 groups of design variables. Every mix of 'current' and 'copilot' groups (2^7 = 128 designs) is evaluated
on the simulator itself (no surrogate). Contributions sum exactly to the total change and do not depend on
the order in which changes are applied; interaction effects are shared fairly between the groups involved.

Run from code/ after analysis.py (which writes results/case_design.npy):
  python attribution.py   ->  results/attribution.json, figs/fig_attribution.png
"""
import sys, os, json
from math import factorial
import numpy as np
import tiresim as T

GROUPS = {
    "Compound": [8, 9],
    "Circumferential grooves": [0, 1],
    "Lateral grooves": [3, 4, 5],
    "Sipes": [6],
    "Tread depth": [7],
    "Pitching": [2, 10],
    "Crown": [11],
}
OBJ = ["rrc", "wet", "wear"]
EXTRA = ["noise", "hydro"]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # works from any cwd
OUT = os.path.join(_ROOT, "figs", ""); RES = os.path.join(_ROOT, "results", "")


def coalition_design(x0, x1, mask):
    """x0 with the variables of group i taken from x1 whenever bit i of mask is set."""
    x = np.array(x0, float).copy()
    for i, idx in enumerate(GROUPS.values()):
        if mask >> i & 1:
            x[idx] = np.asarray(x1, float)[idx]
    return x


def shapley(values):
    """Exact Shapley values. values[mask] = v(coalition), shape (2**n, ...). Returns shape (n, ...)."""
    values = np.asarray(values, float)
    n = int(round(np.log2(len(values))))
    phi = np.zeros((n,) + values.shape[1:])
    for i in range(n):
        for m in range(2 ** n):
            if m >> i & 1:
                continue
            s = bin(m).count("1")
            phi[i] += factorial(s) * factorial(n - s - 1) / factorial(n) * (values[m | 1 << i] - values[m])
    return phi


def helps_sign(key):
    """+1 if a larger value is better (wear, hydroplaning speed), -1 if smaller is better."""
    return 1 if key in ("wear", "hydro") else -1


def check_efficiency(key, contrib, total, tol=1e-6):
    """Shapley efficiency: contributions must add up to the total change (raises, unlike assert, also under -O)."""
    if abs(float(np.sum(contrib)) - total) > tol:
        raise ValueError(f"{key}: contributions {float(np.sum(contrib))} != total change {total}")


def attribute(x0, x1):
    """Grouped Shapley attribution of x0 -> x1 on every size and on the load-weighted line average."""
    n = len(GROUPS)
    keys = OBJ + EXTRA
    w = np.array([s["weight"] for s in T.SIZES]); w = w / w.sum()
    import problem as PB
    V = np.zeros((2 ** n, len(T.SIZES), len(keys)))          # coalition x size x key
    n_infeasible = 0
    for m in range(2 ** n):
        r = T.evaluate(coalition_design(x0, x1, m), detail=True)
        V[m] = [[p[k] for k in keys] for p in r["per_size"]]
        n_infeasible += int(np.any(PB.constraints(r) > 0))
    phi = shapley(V)                                          # group x size x key
    line_phi = np.einsum("gsk,s->gk", phi, w)
    base, cop = V[0], V[-1]
    att = dict(groups=list(GROUPS), sizes=[s["name"] for s in T.SIZES], weights=w.tolist(), base={}, copilot={}, contrib={},
               n_infeasible_coalitions=n_infeasible)
    for j, k in enumerate(keys):
        att["base"][k] = dict(line=float(w @ base[:, j]), per_size=base[:, j].tolist())
        att["copilot"][k] = dict(line=float(w @ cop[:, j]), per_size=cop[:, j].tolist())
        att["contrib"][k] = dict(line=line_phi[:, j].tolist(), per_size=phi[:, :, j].T.tolist())
        total = att["copilot"][k]["line"] - att["base"][k]["line"]
        check_efficiency(k, line_phi[:, j], total)
        if k == "hydro":   # the brief constrains the worst size, not the load-weighted mean
            att["base"][k]["worst_size"] = float(base[:, j].min()); att["copilot"][k]["worst_size"] = float(cop[:, j].min())
    return att


def group_order(att):
    """One bar order shared by all panels: largest total |contribution|, each objective scaled by its
    range in the hypervolume box (problem.F_HI - F_LO)."""
    import problem as PB
    span = PB.F_HI - PB.F_LO
    score = sum(np.abs(att["contrib"][k]["line"]) / span[j] for j, k in enumerate(OBJ))
    return [int(g) for g in np.argsort(-score, kind="stable")]


TITLES = {"rrc": "RRC, N/kN (↓)", "wet": "Wet braking 80–20 km/h, m (↓)", "wear": "Treadwear index (↑)"}


def plot(att, path):
    import matplotlib.pyplot as plt
    from viz_common import NAVY, GREEN, RED, GREY
    markers = ["o", "s", "^"]
    order = group_order(att)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.6))
    for p, (ax, key) in enumerate(zip(axes, OBJ)):
        c = np.array(att["contrib"][key]["line"]); cs = np.array(att["contrib"][key]["per_size"])  # (3, 7)
        b0, b1 = att["base"][key]["line"], att["copilot"][key]["line"]
        labels = ["Current product"] + [att["groups"][g] for g in order] + ["Copilot design"]
        y = np.arange(len(labels))[::-1]
        xs = [b0, b1]
        for yy, v in ((y[0], b0), (y[-1], b1)):
            ax.plot([v, v], [yy - 0.32, yy + 0.32], color=NAVY, lw=3, solid_capstyle="butt")
        run = b0
        for r, g in enumerate(order, start=1):
            col = GREY if c[g] == 0 else (GREEN if c[g] * helps_sign(key) > 0 else RED)
            ax.barh(y[r], c[g], left=run, color=col, height=0.6)
            ax.plot([run, run], [y[r - 1] - 0.3, y[r] + 0.3], color="k", lw=0.5)
            for s in range(cs.shape[0]):
                ax.plot(run + cs[s, g], y[r], marker=markers[s], ms=4, mfc="white", mec="k", mew=0.7, ls="none",
                        label=att["sizes"][s].split()[0] if (p == 0 and r == 1) else None)
                xs.append(run + cs[s, g])
            if c[g] == 0 and np.all(cs[:, g] == 0):
                ax.text(run, y[r], "  no change", va="center", fontsize=6.5, color=GREY)
            run += c[g]
            xs.append(run)
        ax.plot([run, run], [y[-2] - 0.3, y[-1] + 0.3], color="k", lw=0.5)   # last group -> copilot design
        lo, hi = min(xs), max(xs); pad = 0.12 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks(y); ax.set_yticklabels(labels if p == 0 else [], fontsize=8)
        ax.set_title(TITLES[key], fontsize=9, color=NAVY); ax.grid(axis="x", alpha=0.25)
        ax.text(b0, y[0], f" {b0:.1f}" if key != "rrc" else f" {b0:.2f}", va="center", fontsize=7)
        ax.text(b1, y[-1], f" {b1:.1f}" if key != "rrc" else f" {b1:.2f}", va="center", fontsize=7)
    fig.legend(loc="lower center", ncol=3, fontsize=7.5, frameon=False,
               title="Markers: where the bar would end with that size's own contribution", title_fontsize=7.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return fig


def _round(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v) for v in o]
    return o


if __name__ == "__main__":
    if not os.path.exists(RES + "case_design.npy"):
        sys.exit("results/case_design.npy not found - run analysis.py first (run_pipeline.py --mode figures).")
    xb = np.load(RES + "case_design.npy")
    att = attribute(T.BASELINE, xb)
    json.dump(_round(att), open(RES + "attribution.json", "w"), indent=1)
    plot(att, OUT + "fig_attribution.png")
    print(f"{'Group':26s}" + "".join(f"{k:>10s}" for k in OBJ))
    print(f"{'Current product':26s}" + "".join(f"{att['base'][k]['line']:10.3f}" for k in OBJ))
    for g, name in enumerate(att["groups"]):
        print(f"{name:26s}" + "".join(f"{att['contrib'][k]['line'][g]:+10.3f}" for k in OBJ))
    print(f"{'Copilot design':26s}" + "".join(f"{att['copilot'][k]['line']:10.3f}" for k in OBJ))
    print(f"Infeasible mixed designs: {att['n_infeasible_coalitions']} of {2 ** len(GROUPS)}")
