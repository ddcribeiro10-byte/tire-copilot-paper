"""
run_copilot_live.py  -  watch the design copilot work, one loop at a time.

It runs the same agent loop as the paper (generate -> verify -> learn -> optimize -> select ->
engineer gate) with one seed. Every iteration prints how many designs were checked, how much
of the trade-off map has been found (hypervolume) and how many designs meet the brief.

Options (edit below or pass on the command line):
  --budget 150    total simulator checks (150 = paper; 80 is a faster demo)
  --seed 0        random seed (0-4 were used in the paper)
  --gate          ask YOU to approve each batch in the terminal (the "engineer gate")

Takes well under a minute to a few minutes for 150 checks, depending on the computer.
Writes output/copilot_live_log.json and output/copilot_live.png.
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))

import numpy as np
import tiresim as T
import problem as PB
from copilot import DesignCopilot


def terminal_gate(batch, context):
    """Engineer gate in the terminal: you approve or reject each proposed batch."""
    print(f"\n  Engineer gate - iteration {context['iteration']}: {len(batch)} designs proposed.")
    for i, x in enumerate(batch):
        d = T.decode(x)
        print(f"   {i}: grooves={d['n_circ']} width={d['w_circ']:.1f} pitches={d['n_pitch']} "
              f"sipes={d['sipes']} depth={d['depth']:.1f} tan60={d['tand60']:.3f} tan0={d['tand0']:.3f}")
    ans = input("  Press Enter to approve all, or type numbers to reject (e.g. 2 5): ").strip()
    if not ans:
        return batch
    drop = {int(t) for t in ans.split() if t.isdigit()}
    kept = np.array([x for i, x in enumerate(batch) if i not in drop])
    print(f"  {len(drop)} rejected, {len(kept)} approved.")
    return kept if len(kept) else batch[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()

    print(f"Design copilot - budget {a.budget} checks, seed {a.seed}"
          f"{' (you are the engineer gate)' if a.gate else ''}\n")
    cp = DesignCopilot(budget=a.budget, seed=a.seed, human_gate=terminal_gate if a.gate else None, verbose=False)
    orig_log = cp.step_log

    def log(phase, **kw):
        orig_log(phase, **kw)
        e = cp.log[-1]
        print(f"  {phase:<24} checks={e['evals']:>4}   hypervolume={e['hv']:.4f}   feasible={e['n_feasible']:>4}"
              + (f"   ({e['seconds']} s)" if "seconds" in e else ""), flush=True)
    cp.step_log = log
    cp.run()

    F, feas = cp.current_front()
    ref = json.load(open(os.path.join(ROOT, "results", "summary.json"))).get("hv_reference")
    print("\nFinished.")
    if ref:
        print(f"  Share of best-known result: {cp.log[-1]['hv'] / ref * 100:.1f}%  " + ("  (paper: 69.7% on average over seeds 0-4)" if a.budget == 150 else ""))
    import benchmark as BM
    G = np.array([PB.constraints(r) for r in cp.R])
    i = int(np.argmin([BM.weighted_score(f, g) for f, g in zip(F, G)]))
    print(f"  Best compromise design (same weights as the paper): RRC {F[i,0]:.2f}, wet {F[i,1]:.2f} m, wear {-F[i,2]:.0f}")

    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    json.dump(cp.log, open(os.path.join(ROOT, "output", "copilot_live_log.json"), "w"), indent=1, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f0 = PB.objectives(T.evaluate(T.BASELINE))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ev = [e["evals"] for e in cp.log]; hv = [e["hv"] for e in cp.log]
    ax[0].plot(ev, hv, "o-", color="#e07a1f"); ax[0].set_xlabel("simulator checks"); ax[0].set_ylabel("hypervolume")
    ax[0].set_title("Learning progress"); ax[0].axvline(cp.n_init, color="grey", ls=":")
    sc = ax[1].scatter(F[~feas, 0], F[~feas, 1], c="lightgrey", s=12, label="fails brief")
    sc = ax[1].scatter(F[feas, 0], F[feas, 1], c=-F[feas, 2], cmap="viridis", s=22, label="meets brief")
    ax[1].plot(*f0[:2], "rX", ms=12, label="current product")
    fig.colorbar(sc, ax=ax[1], label="treadwear index")
    ax[1].set_xlabel("rolling resistance coeff."); ax[1].set_ylabel("wet braking, m"); ax[1].legend(fontsize=8)
    ax[1].set_title("All designs checked")
    out = os.path.join(ROOT, "output", "copilot_live.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Picture saved: {out}")
    if os.name == "nt":
        os.startfile(out)


if __name__ == "__main__":
    main()
