"""
tiresim.py — Reduced-order, physics-informed simulator for passenger/SUV replacement-tire tread design.

Purpose: a transparent, reproducible stand-in for an expensive CAE chain (FEA footprint, rolling
resistance, wet braking, wear, noise). It is NOT calibrated against real tires; it reproduces the
qualitative trade-offs documented in the tire-mechanics literature so that AI design methods can be
benchmarked on a problem with realistic structure (coupled, multi-objective, multi-size).

Design vector (12 variables, all continuous; integers are rounded inside):
  0 n_circ        number of circumferential grooves          [2, 5]
  1 w_circ        circumferential groove width, mm           [5, 13]
  2 n_pitch       pitch count around circumference            [52, 78]
  3 w_lat         lateral groove width, mm                    [2, 6]
  4 ang_lat       lateral groove angle, deg                   [0, 40]
  5 inner_frac    inner-rib lateral groove penetration (0=solid rib, 1=full)   [0, 1]
  6 sipes         sipes per block                             [0, 4]
  7 depth         non-skid tread depth, mm                    [7.5, 10.0]
  8 tand60        compound tan(delta) at 60 C (RR indicator)  [0.08, 0.20]
  9 tand0         compound tan(delta) at 0 C (wet indicator)  [0.35, 0.80]
 10 pitch_ratio   largest/smallest pitch length ratio         [1.00, 1.60]
 11 crown         crown drop parameter (0 = flat, 1 = round)  [0, 1]
"""
import numpy as np
from scipy import ndimage

NAMES = ["n_circ", "w_circ", "n_pitch", "w_lat", "ang_lat", "inner_frac", "sipes", "depth",
         "tand60", "tand0", "pitch_ratio", "crown"]
LB = np.array([2, 5, 52, 2, 0, 0, 0, 7.5, 0.08, 0.35, 1.00, 0.0])
UB = np.array([5, 13, 78, 6, 40, 1, 4, 10.0, 0.20, 0.80, 1.60, 1.0])

# Representative replacement-market sizes covered by ONE tread line (load at 80% of load index)
SIZES = [
    dict(name="205/55R16 91V", tw=160., od=632., load=4800., p=0.22, aspect=0.55, weight=0.40),
    dict(name="225/65R17 102H", tw=175., od=725., load=6700., p=0.23, aspect=0.65, weight=0.35),
    dict(name="265/70R17 115T", tw=205., od=802., load=9500., p=0.24, aspect=0.70, weight=0.25),
]
DX = 2.0  # grid resolution, mm
# Coefficient multipliers for the sensitivity study (all 1.0 = published model)
PERT = {}
_K = lambda name: PERT.get(name, 1.0)


def decode(x):
    x = np.clip(np.asarray(x, float), LB, UB)
    d = dict(zip(NAMES, x))
    d["n_circ"] = int(round(d["n_circ"])); d["n_pitch"] = int(round(d["n_pitch"]))
    d["sipes"] = int(round(d["sipes"]))
    return d


def compound_feasibility(d):
    """Compound 'magic triangle': for a given technology level, low tand60 limits achievable tand0.
    Returns g <= 0 when feasible."""
    return d["tand0"] - (0.18 + 2.6 * d["tand60"])


def pitch_sequence(n_pitch, ratio, circumference):
    """Three pitch sizes, arranged with a fixed deterministic rule (same rule for every design)."""
    rng = np.random.default_rng(12345)
    classes = np.array([1.0, (1 + ratio) / 2, ratio])
    base = np.tile([0, 1, 2, 2, 1, 0, 1, 2, 0, 1], n_pitch // 10 + 1)[:n_pitch]
    seq = classes[rng.permutation(base)]
    return seq / seq.sum() * circumference


def tread_mask(d, tw, length, circumference, phase=0.0):
    """Rasterize the tread pattern on a (ny, nx) grid. Returns rubber mask, sipe mask, groove geometry."""
    nx = int(tw / DX); ny = int(length / DX)
    x = (np.arange(nx) + 0.5) * DX - tw / 2
    y = (np.arange(ny) + 0.5) * DX - length / 2
    X, Y = np.meshgrid(x, y)
    rubber = np.ones((ny, nx), bool)
    sipe = np.zeros((ny, nx), bool)
    n = d["n_circ"]
    gx = np.linspace(-0.32, 0.32, n) * tw if n > 1 else np.array([0.0])
    for g in gx:
        rubber &= np.abs(X - g) > d["w_circ"] / 2
    edges = np.concatenate([[-tw / 2], gx, [tw / 2]])
    pitches = pitch_sequence(d["n_pitch"], d["pitch_ratio"], circumference)
    yb = np.cumsum(pitches) - circumference / 2 - phase
    yb = yb[(yb > -length / 2 - 40) & (yb < length / 2 + 40)]
    ta = np.tan(np.radians(d["ang_lat"]))
    for r in range(len(edges) - 1):
        x0, x1 = edges[r], edges[r + 1]
        shoulder = r in (0, len(edges) - 2)
        frac = 1.0 if shoulder else d["inner_frac"]
        if frac <= 0.02:
            continue
        side = np.sign((x0 + x1) / 2) or 1.0
        if shoulder:
            xa, xb = x0, x1
        else:  # partial groove from one side, alternating side per rib
            xa, xb = (x0, x0 + (x1 - x0) * frac) if r % 2 else (x1 - (x1 - x0) * frac, x1)
        inrib = (X >= xa) & (X <= xb)
        yline = Y - side * ta * (X - (x0 + x1) / 2)
        for k in range(len(yb)):
            rubber &= ~(inrib & (np.abs(yline - yb[k]) < d["w_lat"] / 2))
            if d["sipes"] > 0 and k < len(yb) - 1:
                for s in range(d["sipes"]):
                    ys = yb[k] + (yb[k + 1] - yb[k]) * (s + 1) / (d["sipes"] + 1)
                    sipe |= ((X >= x0) & (X <= x1) & (np.abs(yline - ys) < 1.0))
    sipe &= rubber
    return rubber, sipe, X, Y, gx


def footprint(d, size, phase=0.0):
    """Reduced-order contact model: Winkler foundation with spatially varying stiffness
    (block-edge softening, sipes, shoulder/sidewall stiffening) + belt load spreading.
    Load equilibrium solved by bisection on tire deflection."""
    tw, od, load, p_inf, aspect = size["tw"], size["od"], size["load"], size["p"], size["aspect"]
    R = od / 2
    length = 260.0
    rubber, sipe, X, Y, gx = tread_mask(d, tw, length, np.pi * od, phase)
    dist = ndimage.distance_transform_edt(rubber) * DX
    k0 = 0.030 * (8.0 / d["depth"]) ** 0.35                         # MPa/mm, composite radial stiffness
    k = k0 * (1 - 0.45 * np.exp(-dist / 2.5)) * np.where(sipe, 0.72, 1.0)
    k *= (1 - 0.06 * d["sipes"])                                     # sipes soften blocks overall
    s_sh = 0.25 + 1.6 * (0.75 - aspect)                              # low profile = stiffer shoulder
    k *= 1 + s_sh * (np.abs(X) / (tw / 2)) ** 6
    k = np.where(rubber, k, 0.0)
    drop = 2.0 + 10.0 * d["crown"]                                    # mm crown drop at tread edge
    gap = drop * (2 * X / tw) ** 4 + Y ** 2 / (2 * R * 0.62)
    A = DX * DX

    def solve(delta):
        pw = k * np.maximum(0.0, delta - gap)
        sm = ndimage.gaussian_filter(pw, 3.0) / np.maximum(ndimage.gaussian_filter(rubber * 1.0, 3.0), 1e-3)
        pr = np.where(pw > 0, 0.55 * pw + 0.45 * sm, 0.0)
        return pr
    lo, hi = 0.0, 60.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if solve(mid).sum() * A < load: lo = mid
        else: hi = mid
    P = solve(hi)
    return P, rubber, sipe, k, hi, X, Y, gx


def evaluate_size(d, size):
    P, rubber, sipe, k, delta, X, Y, gx = footprint(d, size)
    A = DX * DX
    c = P > 1e-6
    pc = P[c]
    pm = pc.mean()
    # lateral (rib-to-rib) pressure distribution — what crown shape and shoulder design control
    colc = c.sum(axis=0) > 5
    prof = np.where(colc, (P * c).sum(axis=0) / np.maximum(c.sum(axis=0), 1), np.nan)[colc]
    cov = float(np.std(prof) / np.mean(prof)); peak = float(prof.max() / prof.mean())
    # gross (envelope) footprint: rows filled between first/last contact cell, then columns
    env = np.zeros_like(c)
    for i in np.where(c.any(axis=1))[0]:
        j = np.where(c[i])[0]; env[i, j[0]:j[-1] + 1] = True
    env = ndimage.binary_closing(env, iterations=3) | env
    for jcol in np.where(env.any(axis=0))[0]:
        ii = np.where(env[:, jcol])[0]; env[ii[0]:ii[-1] + 1, jcol] = True
    fp_len = np.any(c, axis=1).sum() * DX; fp_wid = np.any(c, axis=0).sum() * DX
    gross = env.sum() * A
    net_area = c.sum() * A
    void = 1 - net_area / max(gross, 1)
    # --- rolling resistance coefficient (kg/t) ---
    energy = np.sum(np.where(c, P ** 2 / np.maximum(k, 1e-9), 0.0)) * A / size["load"] * (d["depth"] / 8.0) ** 0.9
    # (an explicit aspect-ratio term was removed after the EPREL plausibility check: real label classes of
    #  3PMSF tyres show no systematic rolling-resistance class change from 205/55R16 to 265/70R17)
    rrc = 3.2 + 39.0 * _K("rr_tread") * d["tand60"] * (energy / 12.4) ** 0.7
    # --- wet braking 80->20 km/h (m) ---
    trans = (np.abs(np.diff(rubber.astype(int), axis=0)) > 0) & env[1:]
    sipe_len = (sipe & env).sum() * A / 2.0                                       # sipe band ~2 mm wide on grid
    edge = (trans.sum() * DX + 2 * sipe_len) / max(gross, 1)                      # edge length per area, 1/mm
    f_edge = 1 - 0.22 * _K("wet_edge") * np.exp(-edge / 0.04)
    circ_area = d["n_circ"] * d["w_circ"] * d["depth"]                            # mm^2 drainage cross-section
    lat_drain = (1 + 0.2 * np.sin(np.radians(2 * d["ang_lat"]))) * d["w_lat"] * d["depth"] * d["n_pitch"] / 60
    drain = (circ_area + 6 * lat_drain) / size["tw"]                              # mm^2 per mm width
    f_drain = 1 - 0.30 * _K("wet_drain") * np.exp(-drain / 1.6)
    mu_c = 0.8749 * (1 + 0.9 * _K("wet_compound") * (d["tand0"] - 0.52))                                      # compound wet friction
    mu_loc = (pc / 0.25) ** -0.15                                                 # rubber friction load sensitivity
    mu_eff = mu_c * f_edge * f_drain * np.sum(mu_loc * pc) / np.sum(pc) * (1 - 0.8 * max(void - 0.34, 0)) \
        * (1 - 0.6 * max(cov - 0.06, 0))
    v1, v2 = 80 / 3.6, 20 / 3.6
    wet = (v1 ** 2 - v2 ** 2) / (2 * 9.81 * mu_eff * 0.92)
    # --- hydroplaning onset speed (km/h), constraint ---
    # Horne-type scaling: onset speed rises with inflation pressure (sqrt) and with mean contact pressure
    # (a higher-pressure footprint squeezes the water film out faster); drainage capacity adds margin.
    hydro = (44 + 40 * _K("hydro_drain") * (1 - np.exp(-drain / 1.9))) * np.sqrt(size["p"] / 0.23) * (pm / 0.25) ** 0.25
    # --- treadwear life index (100 = baseline design on the 205/55R16 size) ---
    slip = 1 + 0.06 * d["sipes"] + 0.6 * void
    wear_rate = (pm / 0.25) ** (1.2 * _K("wear_pressure")) * peak ** (4 * _K("wear_peak")) * slip * (d["tand0"] / 0.52) ** 0.35 * (d["depth"] / 8.5) ** 0.8
    wear = 100 / 0.50625 * (d["depth"] - 1.6) / (8.5 - 1.6) * (1 - void) / 0.725 / wear_rate
    # --- pass-by noise proxy, dB(A) ---
    seq = pitch_sequence(d["n_pitch"], d["pitch_ratio"], np.pi * size["od"])
    v = 80 / 3.6 * 1000                                          # mm/s
    t = np.cumsum(seq) / v
    N = 1 << 15
    sig = np.zeros(N); idx = (t / t[-1] * (N - 1)).astype(int); sig[idx] = 1.0
    blur = 1 + 2.5 * np.tan(np.radians(d["ang_lat"]))            # angled grooves spread impact in time
    sig = ndimage.gaussian_filter1d(sig, blur * 3)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    spec = spec[5:]
    tonal = 10 * np.log10(spec.max() / spec.mean())
    pump = d["w_lat"] * d["depth"] * d["n_pitch"] * (1 - 0.6 * np.sin(np.radians(d["ang_lat"])))
    noise = 44.32 + 0.45 * _K("noise_tonal") * tonal + 3.5 * _K("noise_pump") * np.log10(pump) + 1.5 * np.log10(d["n_circ"] * d["w_circ"]) \
        + 2.4 * (size["tw"] / 175 - 1)
    return dict(rrc=rrc, wet=wet, wear=wear, noise=noise, hydro=hydro, void=void, cov=cov, peak=peak,
                fp_len=fp_len, fp_wid=fp_wid, delta=delta, edge=edge, drain=drain)


def evaluate(x, sizes=SIZES, detail=False):
    """Full evaluation over all sizes in the line. Returns load-weighted line metrics."""
    d = decode(x)
    res = [evaluate_size(d, s) for s in sizes]
    w = np.array([s["weight"] for s in sizes]); w = w / w.sum()
    out = {}
    for key in res[0]:
        vals = np.array([r[key] for r in res])
        out[key] = float(np.dot(w, vals))
    out["wet_worst"] = float(max(r["wet"] for r in res))
    out["wear_worst"] = float(min(r["wear"] for r in res))
    out["rrc_spread"] = float(np.ptp([r["rrc"] for r in res]))
    out["hydro_min"] = float(min(r["hydro"] for r in res))
    out["g_compound"] = float(compound_feasibility(d))
    if detail:
        out["per_size"] = res
    return out


BASELINE = np.array([4, 9.0, 64, 4.0, 10, 0.0, 1, 8.5, 0.14, 0.52, 1.25, 0.45])   # representative current RE all-season


if __name__ == "__main__":
    import time
    t0 = time.time(); r = evaluate(BASELINE, detail=True); dt = time.time() - t0
    for s, rs in zip(SIZES, r["per_size"]):
        print(s["name"], {k: round(v, 3) for k, v in rs.items()})
    print({k: round(v, 3) for k, v in r.items() if k != "per_size"}, f"{dt:.2f}s")
