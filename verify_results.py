"""
verify_results.py  -  compare your freshly computed numbers with the ones published in the paper.

Reads results/summary.json (yours) and published/results/summary.json (paper) and prints the key
numbers side by side. With the pinned package versions in requirements.txt they should match exactly;
other versions may differ in the last digits.
"""
import os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
CP, CONV = "AI copilot (surrogate + agent)", "Conventional (expert loops)"


def load(p):
    return json.load(open(os.path.join(ROOT, p, "summary.json")))


def items(S):
    B = S["benchmark"]
    yield "Copilot, % of best-known hypervolume", B[CP]["hv_pct_mean"]
    yield "Conventional, % of best-known hypervolume", B[CONV]["hv_pct_mean"]
    for m in B:
        if m not in (CP, CONV):
            yield f"{m}, %", B[m]["hv_pct_mean"]
    yield "Hypervolume ratio copilot / conventional", S["ratio_default"]
    yield "Checks for copilot to match conventional (median)", S["copilot_match_median"]
    yield "Mann-Whitney p, copilot vs conventional", S["mwu_p_copilot_vs_conv"]
    yield "GP surrogate R2 at 250 designs", S["r2_mean"]["Gaussian process"]["250"]
    yield "Lead-size optima infeasible on the line, %", S["robustness"]["lead_front_infeasible_pct"]
    c = S["case"]
    for k in ("rrc", "wet", "wear", "noise"):
        yield f"Case design, line {k}", c["line_ai"][k]


def extra_items():
    """Numbers added after the first submission (FIG. 5 qNEHVI engine, FIG. 9 attribution, FIG. 10 worn tires);
    compared only when both your file and the frozen copy in published/results exist."""
    def pair(name):
        a, b = os.path.join(ROOT, "published", "results", name), os.path.join(ROOT, "results", name)
        return (json.load(open(a)), json.load(open(b))) if os.path.exists(a) and os.path.exists(b) else (None, None)
    P, M = pair("qnehvi_summary.json")
    if P:
        q = "qNEHVI (BoTorch)"
        yield "Copilot qNEHVI engine, % of best-known hypervolume", P["methods"][q]["hv_pct_mean"], M["methods"][q]["hv_pct_mean"]
        yield "Copilot qNEHVI engine, best weighted score", P["methods"][q]["best_score_mean"], M["methods"][q]["best_score_mean"]
    P, M = pair("attribution.json")
    if P:
        yield "Attribution, compound share of RRC change", P["contrib"]["rrc"]["line"][0], M["contrib"]["rrc"]["line"][0]
    P, M = pair("worn.json")
    if P:
        i = P["depths"].index(2.0)
        yield "Worn (calibrated), copilot wet lead at 2 mm, m", P["calibrated"]["gap"]["wet"][i], M["calibrated"]["gap"]["wet"][i]


def main():
    try:
        mine, paper = load("results"), load(os.path.join("published", "results"))
    except FileNotFoundError as e:
        print("Missing file:", e); return
    print(f"{'Quantity':<54}{'paper':>12}{'yours':>12}")
    ok = True
    for (label, p), (_, m) in zip(items(paper), items(mine)):
        same = abs(float(p) - float(m)) <= 1e-6 * max(1, abs(float(p)))
        ok &= same
        print(f"{label:<54}{float(p):>12.4g}{float(m):>12.4g}   {'=' if same else 'DIFFERENT'}")
    for label, p, m in extra_items():
        same = abs(float(p) - float(m)) <= 1e-6 * max(1, abs(float(p)))
        ok &= same
        print(f"{label:<54}{float(p):>12.4g}{float(m):>12.4g}   {'=' if same else 'DIFFERENT'}")
    print("\nAll key numbers match the paper." if ok else
          "\nSome numbers differ - usually a different package version (see requirements.txt).")


if __name__ == "__main__":
    main()
