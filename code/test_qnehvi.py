"""Tests for the qNEHVI baseline and its comparison. Run from code/:  python -m unittest test_qnehvi -v"""
import os, json, unittest
import numpy as np
import tiresim as T, problem as PB
import benchmark_qnehvi as Q, compare_qnehvi as C, copilot_qnehvi as CQ
from copilot import DesignCopilot

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "")


class TestQnehvi(unittest.TestCase):
    def test_compound_linear_matches_problem(self):
        c, r = CQ.compound_linear()
        U = np.random.default_rng(0).random((20000, 12)); X = T.LB + U * (T.UB - T.LB)
        np.testing.assert_array_equal(U[:, 9] * c[0] + U[:, 8] * c[1] >= r, PB.compound_g(X) <= 0)

    def test_compare_reproduces_published_stats(self):
        """Recomputing the published methods through compare() must give the numbers in summary.json."""
        bm = json.load(open(RES + "benchmark.json")); S0 = json.load(open(RES + "summary.json"))
        fake_qn = [dict(r, method=C.QN) for r in bm if r["method"] == C.PAR]       # stand-in so compare() runs
        S = C.compare(bm, fake_qn, S0["hv_reference"])
        for m, v in S0["benchmark"].items():
            for k in ("hv_pct_mean", "hv_pct_sd", "best_score_mean", "n_feasible_mean"):
                self.assertAlmostEqual(S["methods"][m][k], v[k], places=6, msg=f"{m} {k}")
        self.assertAlmostEqual(S["mwu_p_copilot_gt_qnehvi"], S0["mwu_p_copilot_vs_parego"], places=4)


class TestEngine(unittest.TestCase):
    def test_engine_runs_inside_the_copilot_loop(self):
        """qNEHVI replaces only learn/optimise/select: same initial designs, gate called on every batch, log kept,
        and every proposal strictly satisfies the compound constraint (margin)."""
        saved = (CQ.MC_SAMPLES, CQ.RESTARTS, CQ.RAW_SAMPLES, CQ.MAXITER)
        CQ.MC_SAMPLES, CQ.RESTARTS, CQ.RAW_SAMPLES, CQ.MAXITER = 8, 1, 16, 5       # fast smoke settings
        try:
            calls = []
            gate = lambda batch, ctx: (calls.append(np.array(batch)), batch)[1]
            cp = CQ.QnehviCopilot(budget=50, seed=0, human_gate=gate).run()
        finally:
            CQ.MC_SAMPLES, CQ.RESTARTS, CQ.RAW_SAMPLES, CQ.MAXITER = saved
        self.assertIsInstance(cp, DesignCopilot)
        ref = DesignCopilot(seed=0).tool_generate(40)
        np.testing.assert_allclose(np.array(cp.X[:40]), ref)
        self.assertEqual(len(calls), 1); self.assertEqual(calls[0].shape, (10, 12))
        np.testing.assert_allclose(np.array(cp.X[40:]), calls[0])
        self.assertEqual(len(cp.log), 2)
        self.assertTrue(np.all(PB.compound_g(calls[0]) < 0))


class TestSavedRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = RES + "benchmark_qnehvi.json"
        cls.runs = json.load(open(f)) if os.path.exists(f) else []

    def test_saved_runs_consistent(self):
        runs = [r for r in self.runs if r.get("engine_version") == Q.ENGINE_VERSION]
        if not runs:
            self.skipTest("no engine_version-2 qNEHVI results yet")
        for r in runs:
            X, F, G = np.array(r["X"]), np.array(r["F"]), np.array(r["G"])
            self.assertEqual(len(X), 150)
            self.assertTrue(np.all(PB.compound_g(X[40:]) <= 0), f"seed {r['seed']}: proposal violates the compound constraint")
            for i in (0, 45, 90, 149):
                f, g, _ = PB.evaluate_full(X[i])
                np.testing.assert_allclose(f, F[i]); np.testing.assert_allclose(g, G[i])


if __name__ == "__main__":
    unittest.main()
