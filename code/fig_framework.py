"""Framework diagram (FIG. 1), sized for a 6.5-inch print width."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from viz_common import NAVY, ORANGE, GREEN

fig, ax = plt.subplots(figsize=(7.6, 4.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.45); ax.axis("off")
TF, BF = 9.5, 8.3


def box(x, y, w, h, title, body, fc="#eef2f7", ec=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=TF, weight="bold", color=NAVY)
    ax.text(x + w / 2, y + h - 0.47 - 0.24 * title.count("\n"), body, ha="center", va="top", fontsize=BF, color="#222", linespacing=1.25)


def arr(p, q, c=NAVY, rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=12, color=c, lw=lw))


# inputs (top)
box(0.125, 5.05, 2.6, 1.3, "Market brief", "label grades, magazine\ntests, warranty, cost", fc="#fdf1e6", ec=ORANGE)
box(3.0, 5.05, 2.6, 1.3, "Size range", "representative sizes\nof the line (here 3)", fc="#fdf1e6", ec=ORANGE)
# agent frame
ax.add_patch(FancyBboxPatch((0.1, 0.15), 7.55, 4.55, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f7f9fc", ec=NAVY, lw=1.1, ls="--"))
ax.text(3.88, 4.47, "Design copilot (agent): keeps state, calls tools, logs decisions", ha="center", va="center", fontsize=8.8, color=NAVY, weight="bold", zorder=10, bbox=dict(fc="#f7f9fc", ec="none", pad=1.5))
W, H = 2.25, 1.6
pos = {"gen": (0.3, 2.55), "ver": (2.75, 2.55), "learn": (5.2, 2.55), "opt": (5.2, 0.35), "sel": (2.75, 0.35), "gate": (0.3, 0.35)}
box(*pos["gen"], W, H, "1 Generate", "space-filling designs\nfrom a pattern\ngrammar")
box(*pos["ver"], W, H, "2 Verify", "expensive evaluation\non every size\n(CAE chain)")
box(*pos["learn"], W, H, "3 Learn", "Gaussian-process\nsurrogates with\nuncertainty")
box(*pos["opt"], W, H, "4 Optimize", "NSGA-II on\nsurrogates\n(explore → exploit)")
box(*pos["sel"], W, H, "5 Select", "batch with largest\npredicted Pareto\ngain + diversity")
box(*pos["gate"], W, H, "6 Engineer gate", "approve, edit or\nveto the batch", fc="#e8f4ec", ec=GREEN)
c = lambda k, dx=0.5, dy=0.5: (pos[k][0] + W * dx, pos[k][1] + H * dy)
arr(c("gen", 1, .5), c("ver", 0, .5)); arr(c("ver", 1, .5), c("learn", 0, .5)); arr(c("learn", .5, 0), c("opt", .5, 1))
arr(c("opt", 0, .5), c("sel", 1, .5)); arr(c("sel", 0, .5), c("gate", 1, .5))
arr(c("gate", .75, 1), c("ver", .25, 0), c=GREEN, lw=1.9)
ax.text(2.47, 2.33, "next batch", fontsize=8, color=GREEN, ha="center", va="center", rotation=0,
        bbox=dict(fc="#f7f9fc", ec="none", pad=0.5))
arr((1.425, 5.03), c("gen", .5, 1), c=ORANGE); arr((3.875, 5.03), c("ver", .5, 1), c=ORANGE)
# outputs (right)
box(7.95, 2.55, 1.95, 2.15, "Outputs", "Pareto set of\nline designs,\nper-size margins,\nreview report", fc="#e8f4ec", ec=GREEN)
box(7.95, 0.35, 1.95, 1.9, "Data asset", "every verified\nresult trains the\nnext line's\nsurrogates")
arr((7.67, 3.6), (7.93, 3.6), c=GREEN); arr((7.67, 1.3), (7.93, 1.3))
fig.savefig("../figs/fig1_framework.png", dpi=300, bbox_inches="tight")
