"""Qualitative sanity checks: one-at-a-time changes from the baseline should move outputs in the
directions known from tire engineering practice."""
import numpy as np, json
import tiresim as T

checks = [
    ("tand60", +0.04, "rrc", +1, "Higher compound hysteresis raises rolling resistance"),
    ("tand0", +0.10, "wet", -1, "Higher loss factor at 0 °C shortens wet braking"),
    ("sipes", +2, "wet", -1, "More sipes (edges) shorten wet braking"),
    ("sipes", +2, "wear", -1, "More sipes soften blocks and reduce treadwear"),
    ("depth", +1.0, "wear", +1, "Deeper tread increases wear life"),
    ("depth", +1.0, "rrc", +1, "Deeper tread increases rolling resistance"),
    ("w_circ", +3, "hydro_min", +1, "Wider circumferential grooves raise hydroplaning speed"),
    ("pitch_ratio", +0.3, "noise", -1, "Larger pitch ratio spreads tonal noise"),
    ("ang_lat", +20, "noise", -1, "Angled lateral grooves reduce impact noise"),
    ("w_lat", +2, "noise", +1, "Wider lateral grooves increase air-pumping noise"),
]
base = T.evaluate(T.BASELINE)
rows = []
for var, dv, out, sign, txt in checks:
    x = T.BASELINE.copy(); j = T.NAMES.index(var); x[j] += dv
    r = T.evaluate(x)
    delta = r[out] - base[out]
    rows.append(dict(change=f"{var} {dv:+g}", output=out, base=round(base[out], 2), new=round(r[out], 2),
                     expected=("increase" if sign > 0 else "decrease"), ok=bool(np.sign(delta) == sign), note=txt))
json.dump(rows, open("../results/sanity.json", "w"), indent=1)
for r in rows:
    print(r)
print("passed", sum(r["ok"] for r in rows), "of", len(rows))
