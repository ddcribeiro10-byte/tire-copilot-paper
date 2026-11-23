"""Produce all paper figures and the numbers quoted in the text (results/summary.json).
Figures are sized for a 6.5-inch print width so that text stays >= ~8 pt."""
import numpy as np, pandas as pd, json
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import tiresim as T, problem as PB
from viz_common import *

OUT = "../figs/"; RES = "../results/"
S = {}
NAME = {"Conventional (expert loops)": "Emulated conventional process", "Random sampling (LHS)": "Random sampling (LHS)",
        "NSGA-II on simulator": "NSGA-II on simulator", "ParEGO (MOBO)": "ParEGO (Bayesian optimization)",
        "AI copilot (surrogate + agent)": "Design copilot"}
ORDER = list(NAME)
MC = dict(zip(ORDER, [GREY, BLUE, GREEN, "#7b4ea3", ORANGE]))
UNIT_RR = "N/kN"

bm = json.load(open(RES + "benchmark.json"))
ref_line = json.load(open(RES + "reference_line.json")); ref_lead = json.load(open(RES + "reference_lead.json"))

# ---------------- best-known front: merge reference run + every feasible design from every benchmark run ----------------
def feasible_F(X):
    out = []
    for x in X:
        f, g, _ = PB.evaluate_full(np.array(x))
        if np.all(g <= 0):
            out.append(f)
    return out
pool = feasible_F(ref_line["X"])
for r in bm:
    F = np.array(r["F"]); G = np.array(r["G"]); ok = np.all(G <= 0, axis=1)
    pool += list(F[ok])
pool = np.array(pool)
HV_REF = PB.hypervolume(pool, np.ones(len(pool), bool))
S["hv_reference"] = HV_REF
S["hv_reference_run_only_pct"] = round(PB.hypervolume(np.array(feasible_F(ref_line["X"])), np.ones(len(feasible_F(ref_line["X"])), bool)) / HV_REF * 100, 1)

# ---------------- Fig 3: learning curves ----------------
lc = pd.read_csv(RES + "learning_curves.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
cols = {"Quadratic RSM (classical DOE)": GREY, "Gradient boosting": BLUE, "Neural network (MLP)": GREEN, "Gaussian process": ORANGE}
lab = {"Quadratic RSM (classical DOE)": "Quadratic RSM (DOE)", "Gradient boosting": "Gradient boosting",
       "Neural network (MLP)": "Neural network", "Gaussian process": "Gaussian process"}
for name, c in cols.items():
    g = lc[lc.model == name].groupby("n")
    axes[0].plot(g.r2.mean().index, g.r2.mean().values, "o-", color=c, label=lab[name], lw=2, ms=4)
    axes[1].plot(g.mape.mean().index, g.mape.mean().values, "o-", color=c, label=lab[name], lw=2, ms=4)
for ax in axes:
    ax.set_xscale("log"); ax.set_xlabel("Training designs"); ax.grid(alpha=0.25)
axes[0].set_ylabel("Test R² (mean of 6 outputs)"); axes[0].set_ylim(0, 1.0)
axes[1].set_ylabel("Test MAPE, % (mean of 6 outputs)")
axes[0].legend(loc="lower right", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT + "fig3_learning_curves.png", dpi=300); plt.close(fig)
S["r2_mean"] = {m: {int(n): round(v, 3) for n, v in lc[lc.model == m].groupby("n").r2.mean().items()} for m in cols}
S["mape_mean"] = {m: {int(n): round(v, 2) for n, v in lc[lc.model == m].groupby("n").mape.mean().items()} for m in cols}
gp250 = lc[(lc.model == "Gaussian process") & (lc.n == 250)].set_index("target")
S["gp250_by_target"] = {k: dict(r2=round(gp250.loc[k, "r2"], 3), mape=round(gp250.loc[k, "mape"], 2), rmse=round(gp250.loc[k, "rmse"], 4)) for k in gp250.index}

# ---------------- Fig 4: hypervolume traces + statistics ----------------
fig, ax = plt.subplots(figsize=(6.8, 4.2))
S["benchmark"] = {}
traces = {}
for m in ORDER:
    rr = sorted([r for r in bm if r["method"] == m], key=lambda r: r["seed"])
    tr = np.array([r["trace"] for r in rr]) / HV_REF * 100; traces[m] = tr
    x = np.arange(10, 10 * tr.shape[1] + 1, 10)
    mu, sd = tr.mean(0), tr.std(0, ddof=1)
    ax.plot(x, mu, color=MC[m], lw=2.2, label=NAME[m]); ax.fill_between(x, mu - sd, mu + sd, color=MC[m], alpha=0.13)
    fe = [r["n_feasible"] for r in rr]; bs = [r["best_score"] for r in rr]
    S["benchmark"][m] = dict(label=NAME[m], hv_pct_mean=round(mu[-1], 1), hv_pct_sd=round(sd[-1], 1),
                             hv_pct_min=round(tr[:, -1].min(), 1), hv_pct_max=round(tr[:, -1].max(), 1),
                             n_feasible_mean=float(np.mean(fe)), best_score_mean=round(float(np.mean(bs)), 3),
                             best_score_sd=round(float(np.std(bs, ddof=1)), 3), trace_pct=mu.round(1).tolist())
f0, g0, r0 = PB.evaluate_full(T.BASELINE)
hv0 = PB.hypervolume(f0[None], [True]) / HV_REF * 100; S["hv_baseline_pct"] = round(hv0, 1)
S["baseline_score"] = round(float(np.dot([0.35, 0.40, 0.25], PB.normalise(f0)[0])), 3)
ax.axhline(hv0, color=RED, ls=":", lw=1.4); ax.text(148, hv0 - 1.2, "current product", color=RED, va="top", ha="right", fontsize=8.5)
CONV, CP, PAR = ORDER[0], ORDER[4], ORDER[3]
# per-seed: evaluations the copilot needs to reach the paired conventional run's final hypervolume
match = []
for s in range(traces[CP].shape[0]):
    target = traces[CONV][s, -1]; hit = np.where(traces[CP][s] >= target)[0]
    match.append(int(10 * (hit[0] + 1)) if len(hit) else None)
S["copilot_match_per_seed"] = match
S["copilot_match_median"] = int(np.median([m for m in match if m]))
S["copilot_match_range"] = [min(m for m in match if m), max(m for m in match if m)]
p = mannwhitneyu(traces[CP][:, -1], traces[CONV][:, -1], alternative="greater").pvalue
p2 = mannwhitneyu(traces[CP][:, -1], traces[PAR][:, -1], alternative="greater").pvalue
S["mwu_p_copilot_vs_conv"] = round(float(p), 4); S["mwu_p_copilot_vs_parego"] = round(float(p2), 4)
ax.set_xlabel("Expensive evaluations (full simulation of the three-size line)")
ax.set_ylabel("Hypervolume, % of best-known front")
ax.set_xlim(10, 150); ax.set_ylim(0, 105); ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig4_hypervolume.png", dpi=300); plt.close(fig)

# sensitivity of the ratio to the hypervolume reference point
def ratio_with(F_hi):
    old = PB.F_HI.copy(); PB.F_HI[:] = F_hi
    def hv_runs(m):
        return np.mean([PB.hypervolume(np.array(r["F"]), np.all(np.array(r["G"]) <= 0, axis=1)) for r in bm if r["method"] == m])
    out = hv_runs(CP) / hv_runs(CONV); PB.F_HI[:] = old; return out
S["ratio_default"] = round(B := S["benchmark"][CP]["hv_pct_mean"] / S["benchmark"][CONV]["hv_pct_mean"], 2)
S["ratio_ref_relaxed"] = round(ratio_with(np.array([13.0, 41.0, -40.0])), 2)
S["ratio_ref_tight"] = round(ratio_with(np.array([10.0, 36.5, -80.0])), 2)

# ---------------- case-study design (copilot seed 0) ----------------
X0 = np.load(RES + "copilot_X_seed0.npy")
rows = [(x,) + tuple(PB.evaluate_full(x)) for x in X0]
feas = [t for t in rows if np.all(t[2] <= 0)]
dom = [t for t in feas if np.all(t[1] <= f0)]
best = max(dom if dom else feas, key=lambda t: np.min((f0 - t[1]) / np.abs(f0)))
xb, fb, gb, rb = best
S["n_dominating_baseline_seed0"] = len(dom)
rb_d = T.evaluate(xb, detail=True); r0_d = T.evaluate(T.BASELINE, detail=True)
KEYS = ["rrc", "wet", "wear", "noise", "hydro_min", "void", "cov", "g_compound"]
S["case"] = dict(design=dict(zip(T.NAMES, [round(float(v), 3) for v in xb])),
                 baseline=dict(zip(T.NAMES, [round(float(v), 3) for v in T.BASELINE])),
                 line_ai={k: round(rb[k], 3) for k in KEYS}, line_base={k: round(r0[k], 3) for k in KEYS},
                 per_size_ai=[{k: round(float(p[k]), 3) for k in ["rrc", "wet", "wear", "noise", "hydro", "cov"]} for p in rb_d["per_size"]],
                 per_size_base=[{k: round(float(p[k]), 3) for k in ["rrc", "wet", "wear", "noise", "hydro", "cov"]} for p in r0_d["per_size"]])
# attribution: compound-only and geometry-only changes
xc = T.BASELINE.copy(); xc[8:10] = xb[8:10]
xg = xb.copy(); xg[8:10] = T.BASELINE[8:10]
for tag, x in [("compound_only", xc), ("geometry_only", xg)]:
    r = T.evaluate(x)
    S["case"][tag] = {k: round(r[k], 3) for k in KEYS}
np.save(RES + "case_design.npy", xb)

# ---------------- Fig 5: objective space ----------------
fig, ax = plt.subplots(figsize=(6.8, 5.2))
nd = pool[[not np.any(np.all(pool <= q, axis=1) & np.any(pool < q, axis=1)) for q in pool]]
sc = ax.scatter(nd[:, 0], nd[:, 1], c=-nd[:, 2], cmap="viridis", s=14, alpha=0.35, label="Best-known front (merged)")
clim = sc.get_clim()
for m, mk in [(CONV, "s"), (CP, "o")]:
    r = [r for r in bm if r["method"] == m and r["seed"] == 0][0]
    F = np.array(r["F"]); G = np.array(r["G"]); ok = np.all(G <= 0, axis=1)
    ax.scatter(F[ok, 0], F[ok, 1], c=-F[ok, 2], cmap="viridis", vmin=clim[0], vmax=clim[1],
               marker=mk, s=40, edgecolor="k" if mk == "o" else GREY, lw=0.8, label=f"{NAME[m]} (feasible)")
ax.scatter(f0[0], f0[1], marker="X", s=190, c=RED, edgecolor="k", zorder=5, label="Current product")
ax.scatter(fb[0], fb[1], marker="*", s=360, c=ORANGE, edgecolor="k", zorder=6, label="Selected copilot design")
cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label("Treadwear index (higher = better)")
ax.set_xlabel(f"Rolling resistance coefficient, {UNIT_RR} (line average)"); ax.set_ylabel("Wet braking 80–20 km/h, m (line average)")
ax.set_xlim(PB.F_LO[0], PB.F_HI[0]); ax.set_ylim(PB.F_LO[1], PB.F_HI[1])
ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig5_objective_space.png", dpi=300); plt.close(fig)

# ---------------- E3: lead-size vs line ----------------
def pick_compromise(X, F):
    Fn = PB.normalise(np.array(F)); return np.array(X)[np.argmin(np.linalg.norm(Fn, axis=1))]
x_lead = pick_compromise(ref_lead["X"], ref_lead["F"]); x_line = pick_compromise(ref_line["X"], ref_line["F"])
rl = T.evaluate(x_lead, detail=True); rL = T.evaluate(x_line, detail=True)
names = [s["name"].split()[0] for s in T.SIZES]
metrics = [("wear", "Treadwear index (↑)"), ("wet", "Wet braking, m (↓)"), ("rrc", f"RRC, {UNIT_RR} (↓)"), ("hydro", "Hydroplaning, km/h (↑)")]
fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.9))
S["robustness"] = {}
for ax, (k, lab_) in zip(axes, metrics):
    a = [p[k] for p in rl["per_size"]]; b = [p[k] for p in rL["per_size"]]
    xx = np.arange(3)
    ax.plot(xx, a, "s--", color=GREY, ms=6, label="Lead size only")
    ax.plot(xx, b, "o-", color=ORANGE, ms=6, label="Whole line")
    if k == "hydro":
        ax.axhline(PB.C["hydro_min"], color=RED, ls=":", lw=1.2)
    ax.set_xticks(xx); ax.set_xticklabels(["205/55\nR16", "225/65\nR17", "265/70\nR17"], fontsize=8)
    ax.set_title(lab_, fontsize=9, color=NAVY); ax.grid(alpha=0.25); ax.set_xlim(-0.4, 2.4)
    S["robustness"][k] = dict(lead_only=[round(float(v), 2) for v in a], line=[round(float(v), 2) for v in b])
axes[0].legend(fontsize=7.5, loc="best")
fig.tight_layout(); fig.savefig(OUT + "fig6_multisize.png", dpi=300); plt.close(fig)
FL, GL, spread_lead, spread_line = [], [], [], []
for x in ref_lead["X"]:
    r = T.evaluate(np.array(x), detail=True)
    FL.append(PB.objectives(r)); GL.append(PB.constraints(r)); spread_lead.append(np.ptp([p["wear"] for p in r["per_size"]]))
for x in ref_line["X"]:
    r = T.evaluate(np.array(x), detail=True); spread_line.append(np.ptp([p["wear"] for p in r["per_size"]]))
FL, GL = np.array(FL), np.array(GL); okL = np.all(GL <= 0, axis=1)
S["robustness"]["lead_front_on_line_hv_pct"] = round(PB.hypervolume(FL, okL) / HV_REF * 100, 1)
S["robustness"]["lead_front_infeasible_pct"] = round(float(100 * (1 - okL.mean())), 1)
S["robustness"]["violation_pct"] = dict(zip(["compound", "noise", "hydro", "void_min", "void_max"], [round(float(v) * 100) for v in (GL > 0).mean(0)]))
S["robustness"]["wear_spread_mean_lead"] = round(float(np.mean(spread_lead)), 1)
S["robustness"]["wear_spread_mean_line"] = round(float(np.mean(spread_line)), 1)

# ---------------- Fig 7: case study patterns + footprints ----------------
fig = plt.figure(figsize=(7.8, 4.6))
gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1, 1, 1], wspace=0.08, hspace=0.32)
for row, (x, lab_) in enumerate([(T.BASELINE, "Current product"), (xb, "Copilot design")]):
    ax = fig.add_subplot(gs[row, 0]); draw_pattern(ax, x, title=lab_)
    for j, s in enumerate(T.SIZES):
        ax = fig.add_subplot(gs[row, j + 1])
        p = (rb_d if row else r0_d)["per_size"][j]
        im = draw_footprint(ax, x, s, title=f"{s['name'].split()[0]}\nlateral CoV {p['cov']:.3f}")
cax = fig.add_axes([0.92, 0.25, 0.015, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_label("Contact pressure, MPa")
fig.savefig(OUT + "fig7_case_study.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 2: design-space examples ----------------
df = pd.read_csv(RES + "dataset.csv")
fig, axes = plt.subplots(1, 6, figsize=(7.8, 2.6))
draw_pattern(axes[0], T.BASELINE, title="Baseline")
for ax, i in zip(axes[1:], [3, 17, 42, 108, 260]):
    draw_pattern(ax, df[T.NAMES].values[i], title=f"Sample {i}")
fig.tight_layout(); fig.savefig(OUT + "fig2_design_space.png", dpi=300); plt.close(fig)

json.dump(S, open(RES + "summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in S.items() if k not in ("r2_mean", "mape_mean")}, indent=1, default=float)[:5000])
