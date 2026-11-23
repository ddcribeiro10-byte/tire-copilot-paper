"""Tests for worn.py. Run from code/:  python -m unittest test_worn -v"""
import os, unittest
import numpy as np
import tiresim as T
import worn as W

CASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "case_design.npy")


class TestEvaluate(unittest.TestCase):
    def test_identity_at_new_depth(self):
        designs = [T.BASELINE] + ([np.load(CASE)] if os.path.exists(CASE) else [])
        for x in designs:
            r, w = T.evaluate(x), W.evaluate_at(x, None)
            self.assertAlmostEqual(w["wet"], r["wet"], delta=1e-9)
            self.assertAlmostEqual(w["rrc"], r["rrc"], delta=1e-9)
            self.assertAlmostEqual(w["hydro"], r["hydro_min"], delta=1e-9)
            self.assertEqual(len(w["wet_sizes"]), 3)

    def test_clipping_bypassed(self):
        """decode() clips depth to >= 7.5 mm; worn evaluation must not."""
        self.assertGreater(W.evaluate_at(T.BASELINE, 3.0)["wet"], W.evaluate_at(T.BASELINE, 7.5)["wet"] + 1.0)


class TestEffDepth(unittest.TestCase):
    def test_piecewise_linear_and_extrapolation(self):
        f = W.EffDepth([(8.0, 8.0), (4.0, 6.0), (2.0, 3.0)])
        self.assertAlmostEqual(f(8.0), 8.0)
        self.assertAlmostEqual(f(6.0), 7.0)
        self.assertAlmostEqual(f(3.0), 4.5)
        self.assertAlmostEqual(f(1.0), 1.5)      # continues the 4->2 slope
        self.assertAlmostEqual(f(0.1), 0.5)      # clipped at 0.5 mm

    def test_new_depth_is_identity_for_any_design(self):
        f = W.EffDepth([(8.5, 8.5), (2.0, 1.0)])
        self.assertEqual(f(None, new=8.57), 8.57)


class TestCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = W.calibrate(T.BASELINE)

    def test_anchors_reproduced(self):
        new = W.evaluate_at(T.BASELINE, None)
        for key, anchors in W.ANCHORS.items():
            for depth, change in anchors["points"].items():
                v = W.evaluate_worn(T.BASELINE, depth, "calibrated", self.cal)[key]
                self.assertAlmostEqual(v / new[key], 1 + change, delta=1e-3, msg=f"{key} at {depth} mm")

    def test_maps_monotone_and_bounded(self):
        for key in ("wet", "rrc"):
            f = self.cal[key]
            grid = np.arange(8.5, 1.5, -0.1)
            vals = [f(d) for d in grid]
            self.assertTrue(all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])), key)
            self.assertTrue(min(vals) >= 0.5)

    def test_extrapolated_model_is_plain_depth(self):
        a = W.evaluate_worn(T.BASELINE, 3.0, "extrapolated", self.cal)
        b = W.evaluate_at(T.BASELINE, 3.0)
        self.assertEqual(a["wet"], b["wet"]); self.assertEqual(a["rrc"], b["rrc"])


class TestRun(unittest.TestCase):
    def test_lead_holds_logic(self):
        cur = {"wet": [30, 35], "hydro": [80, 70], "rrc": [9, 6]}
        cop = {"wet": [29, 34], "hydro": [79, 71], "rrc": [8, 6.5]}
        out = W.lead_holds(cur, cop)
        self.assertEqual(out, {"wet": True, "hydro": False, "rrc": False})

    @unittest.skipUnless(os.path.exists(CASE), "results/case_design.npy missing")
    def test_run_new_equals_evaluate_and_plot(self):
        """slow"""
        xb = np.load(CASE)
        res = W.run(T.BASELINE, xb)
        r0, r1 = T.evaluate(T.BASELINE), T.evaluate(xb)
        for m in W.MODELS:
            self.assertAlmostEqual(res[m]["current"]["wet"][0], r0["wet"], delta=1e-9)
            self.assertAlmostEqual(res[m]["copilot"]["rrc"][0], r1["rrc"], delta=1e-9)
            self.assertAlmostEqual(res[m]["copilot"]["hydro"][0], r1["hydro_min"], delta=1e-9)
            self.assertEqual(len(res[m]["current"]["wet"]), len(W.DEPTHS))
        self.assertIn("lead_holds", res["extrapolated"])
        import tempfile
        fig = W.plot(res, os.path.join(tempfile.mkdtemp(), "f.png"))
        self.assertEqual(len(fig.axes), 6)


class TestReviewFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = W.calibrate(T.BASELINE)

    def test_calibrated_hydro_uses_wet_map_and_rrc_its_own(self):
        for D in (5.0, 2.0):
            w = W.evaluate_worn(T.BASELINE, D, "calibrated", self.cal)
            self.assertEqual(w["hydro"], W.evaluate_at(T.BASELINE, self.cal["wet"](D))["hydro"])
            self.assertEqual(w["wet"], W.evaluate_at(T.BASELINE, self.cal["wet"](D))["wet"])
            self.assertEqual(w["rrc"], W.evaluate_at(T.BASELINE, self.cal["rrc"](D))["rrc"])

    def test_solve_depth_unreachable_raises(self):
        with self.assertRaises(ValueError):
            W._solve_depth(T.BASELINE, "wet", 1.0, 8.5)     # 1 m wet braking is unreachable

    def test_monotone_check_raises_on_bad_metric(self):
        with self.assertRaises(ValueError):
            W.check_monotone(lambda d: (d - 5.0) ** 2, 0.5, 8.5, increasing=True, name="bad")
        W.check_monotone(lambda d: d, 0.5, 8.5, increasing=True, name="ok")

    def test_identity_between_reference_and_copilot_new_depth(self):
        self.assertEqual(self.cal["wet"](8.55), 8.55)

    def test_limit_crossing(self):
        limit = 76.5
        d = W.limit_crossing(T.BASELINE, "calibrated", self.cal, limit)
        h = lambda D: W.evaluate_worn(T.BASELINE, D, "calibrated", self.cal)["hydro"]
        self.assertTrue(1.6 < d < 8.5)
        self.assertGreaterEqual(h(d + 0.02), limit)
        self.assertLess(h(d - 0.02), limit)


if __name__ == "__main__":
    unittest.main()
