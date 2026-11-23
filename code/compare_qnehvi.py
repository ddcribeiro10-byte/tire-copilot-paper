"""
compare_qnehvi.py — Put the qNEHVI baseline on the same footing as the published benchmark.

Reads results/benchmark.json (published 5 methods) and results/benchmark_qnehvi.json, uses the published best-known
hypervolume (summary.json "hv_reference") so every percentage is comparable with the paper, and also reports how the
percentages move if qNEHVI's feasible designs are merged into the best-known front. Writes results/qnehvi_summary.json
and figs/fig_hv_qnehvi.png (the published hypervolume-trace figure with qNEHVI added).
"""
import os, json
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
import tiresim as T, problem as PB

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_ROOT, "results", ""); OUT = os.path.join(_ROOT, "figs", "")
CP, CONV, PAR, QN = "AI copilot (surrogate + agent)", "Conventional (expert loops)", "ParEGO (MOBO)", "qNEHVI (BoTorch)"


def method_stats(runs, hv_ref):
    runs = sorted(runs, key=lambda r: r["seed"])
    tr = np.array([r["trace"] for r in runs]) / hv_ref * 100
    bs = [r["best_score"] for r in runs]
    return dict(hv_pct_mean=round(float(tr[:, -1].mean()), 1), hv_pct_sd=round(float(tr[:, -1].std(ddof=1)), 1),
                hv_pct_min=round(float(tr[:, -1].min()), 1), hv_pct_max=round(float(tr[:, -1].max()), 1),
                n_feasible_mean=float(np.mean([r["n_feasible"] for r in runs])),
                best_score_mean=round(float(np.mean(bs)), 3), best_score_sd=round(float(np.std(bs, ddof=1)), 3),
                trace_pct=tr.mean(0).round(1).tolist(), trace_sd=tr.std(0, ddof=1).round(1).tolist(), final_pct=tr[:, -1].round(2).tolist(),
                seconds_mean=round(float(np.mean([r.get("seconds", np.nan) for r in runs])), 0))


def compare(bm, qn, hv_ref):
    by = {}
    for r in bm + qn:
        by.setdefault(r["method"], []).append(r)
    S = {"hv_reference": hv_ref, "methods": {m: method_stats(rr, hv_ref) for m, rr in by.items()}}
    f0, g0, _ = PB.evaluate_full(T.BASELINE)
    S["hv_baseline_pct"] = round(PB.hypervolume(f0[None], [True]) / hv_ref * 100, 1)
    a, b = np.array(S["methods"][CP]["final_pct"]), np.array(S["methods"][QN]["final_pct"])
    S["mwu_p_copilot_gt_qnehvi"] = round(float(mannwhitneyu(a, b, alternative="greater").pvalue), 4)
    S["mwu_p_qnehvi_gt_copilot"] = round(float(mannwhitneyu(b, a, alternative="greater").pvalue), 4)
    # the qNEHVI engine was added after the first results, so report two-sided tests and the paired view
    S["mwu_p_two_sided"] = round(float(mannwhitneyu(b, a, alternative="two-sided").pvalue), 4)
    S["paired_wins_qnehvi"] = int(np.sum(b > a))                                   # seeds share the 40 initial designs
    S["wilcoxon_p_two_sided"] = round(float(wilcoxon(b, a, alternative="two-sided", method="exact").pvalue), 4)
    sq = [r["best_score"] for r in by[QN]]; sc = [r["best_score"] for r in by[CP]]
    S["best_score_range"] = {QN: [round(min(sq), 3), round(max(sq), 3)], CP: [round(min(sc), 3), round(max(sc), 3)]}
    S["scores_overlap"] = bool(max(sq) >= min(sc))
    S["mwu_p_qnehvi_gt_parego"] = round(float(mannwhitneyu(b, np.array(S["methods"][PAR]["final_pct"]), alternative="greater").pvalue), 4)
    # robustness: best-known front that also includes qNEHVI's feasible designs
    pool = []
    for r in bm + qn:
        F, G = np.array(r["F"]), np.array(r["G"]); pool += list(F[np.all(G <= 0, axis=1)])
    hv_ref2 = PB.hypervolume(np.array(pool), np.ones(len(pool), bool))
    S["hv_reference_with_qnehvi"] = hv_ref2
    S["hv_pct_mean_with_qnehvi_front"] = {m: round(float(np.mean([r["trace"][-1] for r in rr]) / hv_ref2 * 100), 1) for m, rr in by.items()}
    return S


def plot(S, path):
    import matplotlib.pyplot as plt
    from viz_common import GREY, BLUE, GREEN, ORANGE, RED, NAVY
    col = {CONV: GREY, "Random sampling (LHS)": BLUE, "NSGA-II on simulator": GREEN, PAR: "#7b4ea3", CP: ORANGE, QN: NAVY}
    lab = {CONV: "Emulated conventional process", "Random sampling (LHS)": "Random sampling (LHS)", "NSGA-II on simulator": "NSGA-II on simulator",
           PAR: "ParEGO (Bayesian optimization)", CP: "Design copilot, default engine", QN: "Design copilot, qNEHVI engine"}
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for m in col:
        if m not in S["methods"]:
            continue
        mu, sd = np.array(S["methods"][m]["trace_pct"]), np.array(S["methods"][m]["trace_sd"]); x = np.arange(10, 10 * len(mu) + 1, 10)
        ax.plot(x, mu, color=col[m], lw=2.2, ls="--" if m == QN else "-", label=lab[m])
        ax.fill_between(x, mu - sd, mu + sd, color=col[m], alpha=0.10 if m == QN else 0.13, lw=0)
    hv0 = S["hv_baseline_pct"]
    ax.axhline(hv0, color=RED, ls=":", lw=1.4); ax.text(148, hv0 - 1.2, "current product", color=RED, va="top", ha="right", fontsize=8.5)
    ax.set_xlabel("Expensive evaluations (full simulation of the three-size line)"); ax.set_ylabel("Hypervolume, % of best-known front")
    ax.set_xlim(10, 150); ax.set_ylim(0, 105); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return fig


if __name__ == "__main__":
    bm = json.load(open(RES + "benchmark.json")); qn = json.load(open(RES + "benchmark_qnehvi.json"))
    hv_ref = json.load(open(RES + "summary.json"))["hv_reference"]
    S = compare(bm, qn, hv_ref)
    json.dump(S, open(RES + "qnehvi_summary.json", "w"), indent=1)
    plot(S, OUT + "fig_hv_qnehvi.png")
    for m, v in S["methods"].items():
        print(f"{m:32s} HV {v['hv_pct_mean']:5.1f} ± {v['hv_pct_sd']:4.1f} %   best score {v['best_score_mean']:.3f}   feasible {v['n_feasible_mean']:.1f}")
    print("p copilot > qNEHVI:", S["mwu_p_copilot_gt_qnehvi"], "  p qNEHVI > copilot:", S["mwu_p_qnehvi_gt_copilot"])
    print("with qNEHVI in the best-known front:", S["hv_pct_mean_with_qnehvi_front"])
