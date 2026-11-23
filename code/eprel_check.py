"""Plausibility check of the reduced-order simulator against EU tyre-label data (EPREL).
Real data: label classes of C1 tyres in the three study sizes (results/eprel_counts.json).
Simulator RRC maps to the label class directly (same quantity, N/kN). Simulated wet braking distance
is mapped to a wet-grip index by G = G_ref * d_ref / d (G is a braking-performance ratio, so it scales
with deceleration, i.e. inversely with distance); G_ref is anchored by placing the current product on the
lead size at the centre of class C, the most common wet class of real 3PMSF tyres in 205/55R16."""
import json, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import qmc, spearmanr
import tiresim as T
from viz_common import NAVY, ORANGE, GREY, RED, GREEN

RES = "../results/"; OUT = "../figs/"
E = json.load(open(RES + "eprel_counts.json"))
CL = "ABCDE"
RR_EDGES = [5.5, 6.55, 7.75, 9.05, 10.55, 12.5]          # class boundaries (midpoints between published limits)
G_EDGES = [1.80, 1.545, 1.395, 1.245, 1.095, 0.9]
RR_MID = [6.0, 7.15, 8.4, 9.8, 11.5]; G_MID = [1.62, 1.47, 1.32, 1.17, 1.02]
SIZES = ["205", "225", "265"]
rr_class = lambda v: CL[int(np.clip(np.searchsorted(RR_EDGES[1:-1], v, side="right"), 0, 4))]
g_class = lambda g: CL[int(np.clip(np.searchsorted(-np.array(G_EDGES[1:-1]), -g, side="right"), 0, 4))]


def real_stats(group):
    t = E["tab"][group]; n = sum(t.values())
    rr = sum(v * (CL.index(k[0]) + 1) for k, v in t.items()) / n
    wg = sum(v * (CL.index(k[1]) + 1) for k, v in t.items()) / n
    xs, ys = [], []
    for k, v in t.items():
        xs += [CL.index(k[0])] * v; ys += [CL.index(k[1])] * v
    rho = spearmanr(xs, ys).correlation
    return dict(n=n, rr_mean_class=rr, wet_mean_class=wg, spearman_rr_wet=rho)


S = {"real": {g: real_stats(g) for g in E["tab"]}}
base = T.evaluate(T.BASELINE, detail=True)["per_size"]
G_REF = 1.32 * base[0]["wet"]                              # anchor: baseline lead size -> G = 1.32 (class C centre)
Gof = lambda d: G_REF / d
S["anchor"] = dict(G_baseline_lead=1.32, d_baseline_lead=round(base[0]["wet"], 2))
# baseline per size
S["baseline_per_size"] = [dict(rrc=round(p["rrc"], 2), rr_class=rr_class(p["rrc"]), G=round(Gof(p["wet"]), 3),
                               wet_class=g_class(Gof(p["wet"])), noise=round(p["noise"], 1)) for p in base]
# random design space (per size)
X = qmc.scale(qmc.LatinHypercube(d=12, seed=7).random(400), T.LB, T.UB)
rnd = [T.evaluate(x, detail=True)["per_size"] for x in X]
S["random_range"] = {SIZES[j]: dict(rrc_p5_p95=[round(float(np.percentile([r[j]["rrc"] for r in rnd], q)), 2) for q in (5, 95)],
                                    G_p5_p95=[round(float(np.percentile([Gof(r[j]["wet"]) for r in rnd], q)), 3) for q in (5, 95)],
                                    noise_p5_p95=[round(float(np.percentile([r[j]["noise"] for r in rnd], q)), 1) for q in (5, 95)])
                     for j in range(3)}
# best-known front (line optimisation) evaluated per size
ref = json.load(open(RES + "reference_line.json"))
front = [T.evaluate(np.array(x), detail=True)["per_size"] for x in ref["X"]]
combos_real = {g: {k for k, v in E["tab"][g + "_3PMSF"].items() if v > 0} for g in SIZES}
combos_real_any = {g: {k for k, v in list(E["tab"][g + "_3PMSF"].items()) + list(E["tab"][g + "_summer"].items()) if v > 0} for g in SIZES}
S["front"] = {}
for j, g in enumerate(SIZES):
    cmb = [rr_class(p[j]["rrc"]) + g_class(Gof(p[j]["wet"])) for p in front]
    S["front"][g] = dict(n=len(cmb), share_in_real_3pmsf_combos=round(float(np.mean([c in combos_real[g] for c in cmb])) * 100, 1),
                         share_in_any_real_combo=round(float(np.mean([c in combos_real_any[g] for c in cmb])) * 100, 1),
                         best_combo_counts={c: cmb.count(c) for c in sorted(set(cmb))},
                         rrc_min=round(min(p[j]["rrc"] for p in front), 2), G_max=round(max(Gof(p[j]["wet"]) for p in front), 3))
    xs = [CL.index(c[0]) for c in cmb]; ys = [CL.index(c[1]) for c in cmb]
    S["front"][g]["spearman_rr_wet"] = round(float(spearmanr(xs, ys).correlation), 2)
# simulator size trend (baseline) vs real
S["size_trend"] = dict(real_rr=[round(S["real"][g + "_3PMSF"]["rr_mean_class"], 2) for g in SIZES],
                       real_wet=[round(S["real"][g + "_3PMSF"]["wet_mean_class"], 2) for g in SIZES],
                       real_noise_median=[E["noise_n_p10_p50_p90"][g + "_3PMSF"][2] for g in SIZES],
                       sim_rrc=[round(p["rrc"], 2) for p in base], sim_G=[round(Gof(p["wet"]), 3) for p in base],
                       sim_noise=[round(p["noise"], 1) for p in base])
case = np.load(RES + "case_design.npy"); cs = T.evaluate(case, detail=True)["per_size"]
S["case_per_size"] = [dict(rrc=round(p["rrc"], 2), rr_class=rr_class(p["rrc"]), G=round(Gof(p["wet"]), 3), wet_class=g_class(Gof(p["wet"]))) for p in cs]
json.dump(S, open(RES + "eprel_check.json", "w"), indent=1)

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axes[0]
t = E["tab"]["205_3PMSF"]; vmax = max(t.values())
for k, v in t.items():
    i, jj = CL.index(k[0]), CL.index(k[1])
    ax.add_patch(Rectangle((RR_EDGES[i], G_EDGES[jj + 1]), RR_EDGES[i + 1] - RR_EDGES[i], G_EDGES[jj] - G_EDGES[jj + 1],
                           fc=plt.cm.Blues(0.15 + 0.85 * (v / vmax) ** 0.5), ec="white", lw=1))
    ax.text(RR_MID[i], G_MID[jj], str(v), ha="center", va="center", fontsize=7, color="k" if v < vmax * 0.5 else "white")
fr = np.array([[p[0]["rrc"], Gof(p[0]["wet"])] for p in front])
rd = np.array([[r[0]["rrc"], Gof(r[0]["wet"])] for r in rnd])
ax.scatter(rd[:, 0], rd[:, 1], s=5, c=GREY, alpha=0.35, label="Random simulated designs")
ax.scatter(fr[:, 0], fr[:, 1], s=12, c=ORANGE, ec="k", lw=0.3, label="Simulated best-known front")
ax.scatter(base[0]["rrc"], Gof(base[0]["wet"]), marker="X", s=110, c=RED, ec="k", label="Current product (anchor)", zorder=5)
ax.set_xlim(RR_EDGES[0], RR_EDGES[-1]); ax.set_ylim(G_EDGES[-1], G_EDGES[0])
ax.set_xticks(RR_MID); ax.set_xticklabels(list(CL)); ax.set_yticks(G_MID); ax.set_yticklabels(list(CL))
ax.set_xlabel("Rolling resistance class (simulated RRC, N/kN)"); ax.set_ylabel("Wet grip class (mapped G)")
ax.set_title("205/55R16: 3PMSF tires in EPREL (counts)", fontsize=9, color=NAVY)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
ax = axes[1]
xx = np.arange(3)
ax.plot(xx, S["size_trend"]["real_rr"], "s-", color=NAVY, label="Real, mean RR class")
ax.plot(xx, S["size_trend"]["real_wet"], "o-", color=GREEN, label="Real, mean wet class")
ax.plot(xx, [CL.index(rr_class(v)) + 1 + (v - RR_MID[CL.index(rr_class(v))]) / 1.4 for v in S["size_trend"]["sim_rrc"]], "s--", color=NAVY, mfc="white", label="Simulator, RR (baseline)")
ax.plot(xx, [CL.index(g_class(v)) + 1 - (v - G_MID[CL.index(g_class(v))]) / 0.15 for v in S["size_trend"]["sim_G"]], "o--", color=GREEN, mfc="white", label="Simulator, wet (baseline)")
ax.set_xticks(xx); ax.set_xticklabels(["205/55\nR16", "225/65\nR17", "265/70\nR17"], fontsize=8)
ax.set_yticks(range(1, 6)); ax.set_yticklabels(list(CL)); ax.set_ylim(1.5, 4.8); ax.invert_yaxis()
ax.set_title("Class versus size (3PMSF tires)", fontsize=9, color=NAVY); ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig_eprel.png", dpi=300); plt.close(fig)
print(json.dumps({k: v for k, v in S.items() if k != "real"}, indent=1)); print(json.dumps(S["real"], indent=1))
