import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiresim as T

NAVY = "#1f3a5f"; ORANGE = "#e07a1f"; GREY = "#8a8f98"; GREEN = "#2e7d4f"; RED = "#b23a3a"; BLUE = "#3d6fb6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def draw_pattern(ax, x, size=T.SIZES[1], length=200, title=None):
    d = T.decode(x)
    rubber, sipe, X, Y, gx = T.tread_mask(d, size["tw"], length, np.pi * size["od"])
    img = np.where(rubber, 0.15, 0.97); img = np.where(sipe, 0.6, img)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=[-size["tw"] / 2, size["tw"] / 2, -length / 2, length / 2])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title, fontsize=9, color=NAVY, weight="bold")


def draw_footprint(ax, x, size, vmax=0.55, title=None):
    d = T.decode(x)
    P, rubber, sipe, k, delta, X, Y, gx = T.footprint(d, size)
    Pm = np.where(P > 1e-6, P, np.nan)
    im = ax.imshow(Pm, cmap="turbo", vmin=0, vmax=vmax, origin="lower",
                   extent=[X.min(), X.max(), Y.min(), Y.max()])
    ax.set_facecolor("#f3f3f3"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(-110, 110)
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title, fontsize=8, color=NAVY)
    return im
