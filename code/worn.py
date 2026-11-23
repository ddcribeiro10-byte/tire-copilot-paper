"""
worn.py — Does the copilot design keep its lead as the tread wears?

Current product vs selected copilot design from new to 1.6 mm remaining tread, on all three sizes, under two
worn models:
  extrapolated  the simulator as is, with the depth set below its design range (tiresim.decode clips depth to
                [7.5, 10] mm, so the depth is set on the decoded design and evaluate_size is called directly);
  calibrated    the same simulator evaluated at an effective depth D_eff = phi(D), a monotone piecewise-linear
                map fitted so that the CURRENT PRODUCT reproduces published worn-tire trends (ANCHORS), then
                applied unchanged to the copilot design. Each design keeps its own drainage, stiffness and pattern;
                only the depth axis is bent. New-tire behaviour is unchanged (phi = identity at new depth).
phi_wet is used for wet braking and hydroplaning (both drainage-driven), phi_rrc for rolling resistance.

Run from anywhere after analysis.py:  python worn.py  ->  results/worn.json, figs/fig_worn.png
"""
import os, sys, json
import numpy as np
import tiresim as T

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "figs", ""); RES = os.path.join(_ROOT, "results", "")

DEPTHS = [None, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.6]          # remaining tread depth, mm (None = new)
MODELS = ["extrapolated", "calibrated"]
KEYS = ["wet", "hydro", "rrc"]
# relative change from new for the current product, from verified sources (see spec "Verified anchors")
ANCHORS = {
    "wet": dict(points={4.0: 0.0, 2.0: 0.17}, sources=["acl2026", "dekra2022"],
                note="joint European motoring-club study (ACL 2026): 31.9 m new, 31.4 m at 4 mm, 37.3 m at 2 mm; DEKRA 2022: +16-18 % at 2-3 mm"),
    "rrc": dict(points={1.6: -0.20}, sources=["trb286"],
                note="TRB Special Report 286: rolling resistance declines by 20 % or more to worn-out depth; literature range 20-40 %"),
}
D_MIN = 0.5
_W = np.array([s["weight"] for s in T.SIZES], float); _W = _W / _W.sum()


def evaluate_at(x, depth):
    """Line-average wet and rrc, worst-size hydro, per-size wet, with the tread depth set to `depth` mm
    (None = the design's own new depth). Bypasses decode()'s clipping of depth."""
    d = T.decode(x)
    if depth is not None:
        d["depth"] = float(depth)
    rs = [T.evaluate_size(d, s) for s in T.SIZES]
    return dict(wet=float(_W @ [r["wet"] for r in rs]), rrc=float(_W @ [r["rrc"] for r in rs]),
                hydro=float(min(r["hydro"] for r in rs)), wet_sizes=[float(r["wet"]) for r in rs])


class EffDepth:
    """Piecewise-linear map from remaining depth D to effective depth. knots = [(D, D_eff), ...] with D descending;
    identity above the first knot, linear extrapolation below the last, clipped at D_MIN."""
    def __init__(self, knots):
        self.knots = sorted(((float(a), float(b)) for a, b in knots), reverse=True)

    def __call__(self, D, new=None):
        if D is None:
            return new
        k = self.knots
        if D >= k[0][0]:
            return float(D)
        for (d0, e0), (d1, e1) in zip(k, k[1:]):
            if D >= d1:
                return e1 + (e0 - e1) * (D - d1) / (d0 - d1)
        (d0, e0), (d1, e1) = k[-2], k[-1]
        return max(D_MIN, e1 + (e0 - e1) * (D - d1) / (d0 - d1))


def _solve_depth(x, key, target, hi):
    """Effective depth in [D_MIN, hi] at which evaluate_at(x, .)[key] == target (bisection; key monotone in depth)."""
    f = lambda dd: evaluate_at(x, dd)[key] - target
    lo, flo, fhi = D_MIN, f(D_MIN), f(hi)
    if abs(fhi) < 1e-12:
        return hi
    if flo * fhi > 0:
        raise ValueError(f"{key}: target {target:.4f} not reachable between {D_MIN} and {hi} mm")
    for _ in range(50):
        mid = 0.5 * (lo + hi); fm = f(mid)
        if fm * flo > 0: lo, flo = mid, fm
        else: hi = mid
    return 0.5 * (lo + hi)


def check_monotone(f, lo, hi, increasing, name, step=0.25):
    """Bisection precondition: f strictly monotone in depth on [lo, hi] (checked on a grid)."""
    v = [f(d) for d in np.arange(lo, hi + 1e-9, step)]
    ok = all((b > a) if increasing else (b < a) for a, b in zip(v, v[1:]))
    if not ok:
        raise ValueError(f"{name} is not monotone in depth on [{lo}, {hi}] mm; the depth calibration assumes it is")


def calibrate(x_ref):
    """Fit phi_wet and phi_rrc on the reference design (the current product)."""
    new_depth = T.decode(x_ref)["depth"]; new = evaluate_at(x_ref, None)
    check_monotone(lambda dd: evaluate_at(x_ref, dd)["wet"], D_MIN, new_depth, increasing=False, name="wet braking")
    check_monotone(lambda dd: evaluate_at(x_ref, dd)["rrc"], D_MIN, new_depth, increasing=True, name="RRC")
    cal = {"knots": {}}
    for key, a in ANCHORS.items():
        knots = [(new_depth, new_depth)]
        prev = new_depth
        for D, change in sorted(a["points"].items(), reverse=True):
            e = _solve_depth(x_ref, key, new[key] * (1 + change), prev)
            knots.append((D, e)); prev = e
        cal[key] = EffDepth(knots); cal["knots"][key] = knots
    return cal


def evaluate_worn(x, depth, model, cal):
    if model == "extrapolated":
        return evaluate_at(x, depth)
    new_depth = T.decode(x)["depth"]
    w = evaluate_at(x, cal["wet"](depth, new=new_depth))
    r = evaluate_at(x, cal["rrc"](depth, new=new_depth))
    return dict(wet=w["wet"], hydro=w["hydro"], wet_sizes=w["wet_sizes"], rrc=r["rrc"])


BETTER = {"wet": -1, "hydro": 1, "rrc": -1}          # sign of "better": lower wet/rrc, higher hydroplaning speed
HYDRO_LIMIT = 76.5                                     # brief's hydroplaning limit (problem.C["hydro_min"]), new tire


def limit_crossing(x, model, cal, limit, step=0.05):
    """Largest remaining depth at which hydroplaning onset first drops below `limit`, scanning from new depth
    down to 1.6 mm (the scan tolerates small non-monotone dips) and refining by bisection. None if it never does."""
    new_depth = T.decode(x)["depth"]
    h = lambda D: evaluate_worn(x, D, model, cal)["hydro"]
    if h(new_depth) < limit:
        return float(new_depth)
    prev = new_depth
    for D in np.arange(new_depth - step, 1.6 - 1e-9, -step):
        if h(D) < limit:
            hi, lo = prev, D                       # h(hi) >= limit > h(lo)
            for _ in range(30):
                mid = 0.5 * (hi + lo)
                if h(mid) >= limit: hi = mid
                else: lo = mid
            return float(0.5 * (hi + lo))
        prev = D
    return None


def lead_holds(cur, cop):
    """Per output: is the copilot design better than the current product at every depth?"""
    return {k: bool(all((b - a) * BETTER[k] > 0 for a, b in zip(cur[k], cop[k]))) for k in cur}


def run(x0, x1):
    cal = calibrate(x0)
    res = dict(depths=["new" if d is None else d for d in DEPTHS], new_depth=dict(current=float(T.decode(x0)["depth"]),
               copilot=float(T.decode(x1)["depth"])), anchors=ANCHORS, knots=cal["knots"], hydro_limit_new_tire=HYDRO_LIMIT)
    for m in MODELS:
        res[m] = {}
        for tag, x in (("current", x0), ("copilot", x1)):
            rows = [evaluate_worn(x, d, m, cal) for d in DEPTHS]
            res[m][tag] = {k: [r[k] for r in rows] for k in KEYS + ["wet_sizes"]}
        res[m]["lead_holds"] = lead_holds({k: res[m]["current"][k] for k in KEYS}, {k: res[m]["copilot"][k] for k in KEYS})
        res[m]["hydro_limit_crossing"] = {tag: limit_crossing(x, m, cal, HYDRO_LIMIT) for tag, x in (("current", x0), ("copilot", x1))}
        res[m]["gap"] = {k: [(a - b) * -BETTER[k] for a, b in zip(res[m]["current"][k], res[m]["copilot"][k])] for k in KEYS}
    return res


TITLES = {"wet": "Wet braking 80–20 km/h, m (↓)", "hydro": "Hydroplaning onset, worst size, km/h (↑)", "rrc": "RRC, N/kN (↓)"}
GAP_TITLES = {"wet": "Copilot lead, m", "hydro": "Copilot − current, km/h", "rrc": "Copilot lead, N/kN"}


def plot(res, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from viz_common import NAVY, ORANGE, GREY
    xs = [res["new_depth"]["current"] if d == "new" else d for d in res["depths"]]
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    style = {"calibrated": "-", "extrapolated": "--"}
    for j, k in enumerate(KEYS):
        ax, axg = axes[0, j], axes[1, j]
        for m in MODELS:
            for tag, col in (("current", NAVY), ("copilot", ORANGE)):
                ax.plot(xs, res[m][tag][k], style[m], color=col, lw=1.4, marker="o", ms=2.5)
            axg.plot(xs, res[m]["gap"][k], style[m], color=ORANGE, lw=1.4, marker="o", ms=2.5)
        if k == "wet":   # per-size spread of the lead, calibrated model
            cs, ps = np.array(res["calibrated"]["current"]["wet_sizes"]), np.array(res["calibrated"]["copilot"]["wet_sizes"])
            g = cs - ps
            axg.fill_between(xs, g.min(axis=1), g.max(axis=1), color=ORANGE, alpha=0.18, lw=0)
        new_cur = res["calibrated"]["current"][k][0]
        if k in res["anchors"]:
            pts = res["anchors"][k]["points"]
            for D, ch in pts.items():
                ax.plot(float(D), new_cur * (1 + ch), "D", ms=5, mfc="white", mec="k", mew=1.0, zorder=5)
            if k == "rrc":   # published 20-40 % range at worn-out depth
                ax.plot([1.6, 1.6], [new_cur * 0.6, new_cur * 0.8], color="k", lw=1.0, zorder=4)
                ax.plot(1.6, new_cur * 0.6, "_", color="k", ms=6)
        if k == "hydro":
            ax.axhline(res["hydro_limit_new_tire"], color=GREY, ls=":", lw=1.1)
            ax.text(6.4, res["hydro_limit_new_tire"] - 0.5, "brief limit (new tire)", fontsize=6.5, color=GREY, va="top")
        axg.axhline(0, color="k", lw=0.6)
        for a in (ax, axg):
            a.axvspan(1.4, 7.5, color=GREY, alpha=0.07, lw=0)
            a.grid(alpha=0.25)
        ax.set_title(TITLES[k], fontsize=8.5, color=NAVY); axg.set_title(GAP_TITLES[k], fontsize=8.5, color=NAVY)
        axg.set_xlabel("Remaining tread depth, mm", fontsize=8)
    axes[0, 0].text(7.35, axes[0, 0].get_ylim()[1], "outside design range →", fontsize=6.5, color=GREY, va="top")
    axes[1, 0].set_xlim(8.7, 1.4)
    for a in axes[1]:
        a.set_xticks([8.5, 7, 6, 5, 4, 3, 2, 1.6]); a.set_xticklabels(["new", "7", "6", "5", "4", "3", "2", ""], fontsize=7.5)
        a.text(1.6, -0.06, "1.6", transform=a.get_xaxis_transform(), fontsize=6.5, ha="left", va="top")
    handles = [Line2D([], [], color=NAVY, lw=1.4, label="Current product"), Line2D([], [], color=ORANGE, lw=1.4, label="Copilot design"),
               Line2D([], [], color="k", ls="-", lw=1.2, label="Calibrated worn model"), Line2D([], [], color="k", ls="--", lw=1.2, label="Extrapolated simulator"),
               Line2D([], [], marker="D", ls="none", mfc="white", mec="k", label="Published worn-tire data, current-product scale"),
               Line2D([], [], color="k", lw=1.0, marker="_", ms=6, label="Published RRC range at worn-out depth (−20 to −40 %)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return fig


def _round(o):
    if isinstance(o, float): return round(o, 4)
    if isinstance(o, dict): return {str(k): _round(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_round(v) for v in o]
    return o


if __name__ == "__main__":
    if not os.path.exists(RES + "case_design.npy"):
        sys.exit("results/case_design.npy not found - run analysis.py first (run_pipeline.py --mode figures).")
    res = run(T.BASELINE, np.load(RES + "case_design.npy"))
    json.dump(_round(res), open(RES + "worn.json", "w"), indent=1)
    plot(res, OUT + "fig_worn.png")
    print("knots:", {k: [(round(a, 2), round(b, 2)) for a, b in v] for k, v in res["knots"].items()})
    for m in MODELS:
        print(f"\n[{m}]  lead holds: {res[m]['lead_holds']}")
        print("depth " + "".join(f"{k + ' cur/cop':>20s}" for k in KEYS))
        for i, d in enumerate(res["depths"]):
            print(f"{str(d):5s} " + "".join(f"{res[m]['current'][k][i]:10.2f}{res[m]['copilot'][k][i]:10.2f}" for k in KEYS))
