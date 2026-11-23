"""
play_design.py  -  try your own tread design on the virtual tire.

Edit MY_DESIGN below, press F5 (or run this file), and the script will:
  1. print the performance of your design vs. the current product on all three sizes,
  2. tell you which requirements of the brief pass or fail,
  3. save a picture (tread pattern + footprint pressure per size) to output/my_design.png.

Runs in a couple of seconds. Nothing here changes the published results.
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))

import numpy as np
import tiresim as T
import problem as PB

# ----------------------------------------------------------------------------------------------
# YOUR DESIGN  (start from the current product and change what you like; allowed range on the right)
# ----------------------------------------------------------------------------------------------
MY_DESIGN = dict(
    n_circ=4,          # number of circumferential grooves              [2 .. 5]
    w_circ=9.0,        # circumferential groove width, mm               [5 .. 13]
    n_pitch=64,        # pitch count around the tire                    [52 .. 78]
    w_lat=4.0,         # lateral groove width, mm                       [2 .. 6]
    ang_lat=10,        # lateral groove angle, deg                      [0 .. 40]
    inner_frac=0.0,    # how far lateral grooves cut the inner ribs     [0 = solid rib .. 1 = full]
    sipes=1,           # sipes per block                                [0 .. 4]
    depth=8.5,         # non-skid tread depth, mm                       [7.5 .. 10]
    tand60=0.14,       # compound tan(delta) at 60 C (lower = rolls easier)   [0.08 .. 0.20]
    tand0=0.52,        # compound tan(delta) at 0 C  (higher = more wet grip)  [0.35 .. 0.80]
    pitch_ratio=1.25,  # largest / smallest pitch length                [1.00 .. 1.60]
    crown=0.45,        # crown shape (0 = flat, 1 = round)              [0 .. 1]
)
# Tip: the copilot's example design from the paper is in results/case_design.npy.
# Set USE_PAPER_CASE = True to load it instead of MY_DESIGN.
USE_PAPER_CASE = False
# ----------------------------------------------------------------------------------------------


def to_vector(d):
    x = np.array([d[k] for k in T.NAMES], float)
    out = np.clip(x, T.LB, T.UB)
    for k, a, b in zip(T.NAMES, x, out):
        if a != b:
            print(f"  note: {k}={a} is outside the allowed range, using {b}")
    return out


def main():
    x = np.load(os.path.join(ROOT, "results", "case_design.npy")) if USE_PAPER_CASE else to_vector(MY_DESIGN)
    base = T.evaluate(T.BASELINE, detail=True)
    mine = T.evaluate(x, detail=True)

    print("\nYOUR DESIGN:", ", ".join(f"{k}={v:g}" for k, v in zip(T.NAMES, np.round(x, 3))))
    rows = [("Rolling resistance coeff. (lower better)", "rrc", "{:.2f}"),
            ("Wet braking 80-20 km/h, m (lower better)", "wet", "{:.2f}"),
            ("Treadwear index (higher better)", "wear", "{:.0f}"),
            ("Noise, dB(A) (lower better)", "noise", "{:.2f}"),
            ("Hydroplaning speed, km/h (higher better)", "hydro", "{:.1f}"),
            ("Void ratio", "void", "{:.3f}")]
    for s, b, m in zip(T.SIZES, base["per_size"], mine["per_size"]):
        print(f"\n  {s['name']:<16}{'current':>12}{'yours':>12}")
        for label, k, f in rows:
            print(f"  {label:<42}{f.format(b[k]):>8}{f.format(m[k]):>12}")

    print("\nLINE AVERAGE (weighted by sales mix 40/35/25)")
    for label, k, f in rows[:4]:
        delta = (mine[k] - base[k]) / abs(base[k]) * 100
        print(f"  {label:<42}{f.format(base[k]):>8}{f.format(mine[k]):>12}   {delta:+.1f}%")

    g = PB.constraints(mine)
    names = ["compound is realistic (tan d0 <= 0.18 + 2.6 tan d60)",
             f"noise <= {PB.C['noise_max']} dB(A)",
             f"hydroplaning >= {PB.C['hydro_min']} km/h on every size",
             f"void ratio >= {PB.C['void_min']}", f"void ratio <= {PB.C['void_max']}"]
    print("\nBRIEF REQUIREMENTS")
    for n, v in zip(names, g):
        print(f"  [{'PASS' if v <= 0 else 'FAIL'}] {n}")
    print("  -> design is", "FEASIBLE" if np.all(g <= 0) else "NOT feasible")

    # picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz_common import draw_pattern, draw_footprint
    fig, ax = plt.subplots(2, 4, figsize=(11, 6), gridspec_kw=dict(width_ratios=[1, 1.2, 1.2, 1.2]))
    for r, (xx, name) in enumerate([(T.BASELINE, "Current product"), (x, "Your design")]):
        draw_pattern(ax[r, 0], xx, title=name)
        for c, s in enumerate(T.SIZES):
            im = draw_footprint(ax[r, c + 1], xx, s, title=s["name"])
    fig.colorbar(im, ax=ax[:, 1:], shrink=0.7, label="contact pressure, MPa")
    out = os.path.join(ROOT, "output", "my_design.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPicture saved: {out}")
    if os.name == "nt":
        os.startfile(out)


if __name__ == "__main__":
    main()
