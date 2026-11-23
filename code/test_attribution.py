"""Tests for attribution.py. Run from code/:  python -m unittest test_attribution -v"""
import os, unittest
import numpy as np
import tiresim as T
import attribution as A

CASE = "../results/case_design.npy"


def game(n, v):
    """Tabulate a set function v(frozenset) over all 2**n masks."""
    return np.array([v(frozenset(i for i in range(n) if m >> i & 1)) for m in range(2 ** n)], float)


class TestCore(unittest.TestCase):
    def test_coalition_design_masks(self):
        x0, x1 = np.zeros(12), np.ones(12)
        np.testing.assert_array_equal(A.coalition_design(x0, x1, 0), x0)
        np.testing.assert_array_equal(A.coalition_design(x0, x1, 2 ** 7 - 1), x1)
        np.testing.assert_array_equal(np.flatnonzero(A.coalition_design(x0, x1, 1)), [8, 9])

    def test_groups_match_spec_and_partition(self):
        named = {g: [T.NAMES[i] for i in idx] for g, idx in A.GROUPS.items()}
        self.assertEqual(named, {"Compound": ["tand60", "tand0"], "Circumferential grooves": ["n_circ", "w_circ"],
                                 "Lateral grooves": ["w_lat", "ang_lat", "inner_frac"], "Sipes": ["sipes"],
                                 "Tread depth": ["depth"], "Pitching": ["n_pitch", "pitch_ratio"], "Crown": ["crown"]})
        self.assertEqual(sorted(i for idx in A.GROUPS.values() for i in idx), list(range(12)))

    def test_efficiency_check_raises(self):
        with self.assertRaises(ValueError):
            A.check_efficiency("rrc", np.array([1.0, 1.0]), 3.0)
        A.check_efficiency("rrc", np.array([1.0, 2.0]), 3.0)

    def test_shapley_additive_game(self):
        w = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(A.shapley(game(3, lambda S: sum(w[i] for i in S))), w)

    def test_shapley_efficiency_interaction(self):
        v = game(3, lambda S: 1.0 if {0, 1} <= S else 0.0)
        phi = A.shapley(v)
        np.testing.assert_allclose(phi, [0.5, 0.5, 0.0])
        self.assertAlmostEqual(phi.sum(), v[-1] - v[0])

    def test_shapley_vector_values(self):
        g1 = game(3, lambda S: sum([1.0, 2.0, 3.0][i] for i in S))
        g2 = game(3, lambda S: 1.0 if {0, 1} <= S else 0.0)
        phi = A.shapley(np.column_stack([g1, g2]))
        self.assertEqual(phi.shape, (3, 2))
        np.testing.assert_allclose(phi[:, 0], A.shapley(g1))
        np.testing.assert_allclose(phi[:, 1], A.shapley(g2))

    def test_dummy_group_zero(self):
        phi = A.shapley(game(3, lambda S: len(S & {0, 1}) ** 2))
        self.assertEqual(phi[2], 0.0)


class TestAttribute(unittest.TestCase):
    def test_helps_sign(self):
        self.assertEqual(A.helps_sign("wear"), 1)
        self.assertEqual(A.helps_sign("wet"), -1)
        self.assertEqual(A.helps_sign("rrc"), -1)
        self.assertEqual(A.helps_sign("hydro"), 1)
        self.assertEqual(A.helps_sign("noise"), -1)

    def test_attribute_identity_pair(self):
        """slow: 128 simulator runs"""
        att = A.attribute(T.BASELINE, T.BASELINE)
        for key in A.OBJ + A.EXTRA:
            self.assertTrue(all(c == 0.0 for c in att["contrib"][key]["line"]))
            self.assertTrue(all(c == 0.0 for row in att["contrib"][key]["per_size"] for c in row))

    def test_group_order_shared_and_normalised(self):
        """One order for all panels, ranked by sum over objectives of |contribution| / objective range."""
        att = {"groups": ["a", "b", "c"],
               "contrib": {"rrc": {"line": [1.0, 0.0, 0.0]},      # range 5   -> a: 0.2
                           "wet": {"line": [0.0, 0.0, 6.0]},      # range 12  -> c: 0.5
                           "wear": {"line": [0.0, -33.0, 0.0]}}}  # range 110 -> b: 0.3
        self.assertEqual(A.group_order(att), [2, 1, 0])

    @unittest.skipUnless(os.path.exists(CASE), "results/case_design.npy missing")
    def test_attribute_matches_evaluate(self):
        """slow: 128 simulator runs"""
        xb = np.load(CASE)
        att = A.attribute(T.BASELINE, xb)
        r0, r1 = T.evaluate(T.BASELINE), T.evaluate(xb)
        p0, p1 = T.evaluate(T.BASELINE, detail=True)["per_size"], T.evaluate(xb, detail=True)["per_size"]
        for key in A.OBJ:
            self.assertAlmostEqual(att["base"][key]["line"], r0[key], delta=1e-9)
            self.assertAlmostEqual(att["copilot"][key]["line"], r1[key], delta=1e-9)
            self.assertAlmostEqual(sum(att["contrib"][key]["line"]), r1[key] - r0[key], delta=1e-6)
            for s in range(len(T.SIZES)):   # per-size values and per-size efficiency (guards a transposed array)
                self.assertAlmostEqual(att["base"][key]["per_size"][s], p0[s][key], delta=1e-9)
                self.assertAlmostEqual(att["copilot"][key]["per_size"][s], p1[s][key], delta=1e-9)
                self.assertAlmostEqual(sum(att["contrib"][key]["per_size"][s]), p1[s][key] - p0[s][key], delta=1e-6)
        self.assertAlmostEqual(att["base"]["hydro"]["worst_size"], r0["hydro_min"], delta=1e-9)
        self.assertAlmostEqual(att["copilot"]["hydro"]["worst_size"], r1["hydro_min"], delta=1e-9)
        n = att["n_infeasible_coalitions"]
        self.assertIsInstance(n, int)
        self.assertTrue(0 < n < 2 ** len(A.GROUPS) - 1)   # both endpoints feasible; review counted many mixes infeasible

    @unittest.skipUnless(os.path.exists(CASE), "results/case_design.npy missing")
    def test_plot_no_bars_from_zero(self):
        """Start/end values must not be drawn as bars from 0 on a truncated axis: only the 7 group bars are bars,
        and every bar lies inside the visible x-range."""
        import tempfile
        from matplotlib.patches import Rectangle
        att = {k: v for k, v in __import__("json").load(open("../results/attribution.json")).items()}
        fig = A.plot(att, os.path.join(tempfile.mkdtemp(), "f.png"))
        for ax in fig.axes:
            bars = [p for p in ax.patches if isinstance(p, Rectangle)]
            self.assertEqual(len(bars), len(A.GROUPS))
            lo, hi = ax.get_xlim()
            for b in bars:
                self.assertTrue(lo <= min(b.get_x(), b.get_x() + b.get_width()) and max(b.get_x(), b.get_x() + b.get_width()) <= hi)


if __name__ == "__main__":
    unittest.main()
