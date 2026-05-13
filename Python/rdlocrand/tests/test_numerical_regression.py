from __future__ import annotations

import contextlib
import io
import inspect
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import ks_2samp, ranksums

import rdlocrand as rdlocrand_pkg
from rdlocrand import rdrandinf, rdrbounds, rdsensitivity, rdwinselect
from rdlocrand.rdlocrand_fun import (
    ksmirnov_statistic,
    ranksum_statistic,
    rdrandinf_bernoulli_ranksum_pvalue,
)


def make_regression_data(n: int = 60):
    idx = np.arange(1, n + 1, dtype=float)
    R = np.linspace(-1.5, 1.5, n)
    D = (R >= 0).astype(float)
    T = 0.15 + 0.65 * D + 0.05 * np.cos(idx)
    Y = 1 + 0.8 * R - 0.2 * R**2 + 1.5 * (R >= 0) + np.sin(idx / 3)
    Y_fuzzy = 1 + 0.8 * R - 0.2 * R**2 + 1.5 * T + np.sin(idx / 3)
    X = np.column_stack((np.cos(idx), np.sin(idx)))
    return Y, Y_fuzzy, R, X, T


def assert_rng_unchanged(test_case: unittest.TestCase, func):
    np.random.seed(1900)
    before = np.random.get_state()
    func()
    after = np.random.get_state()
    test_case.assertEqual(before[0], after[0])
    assert_array_equal(before[1], after[1])
    test_case.assertEqual(before[2:], after[2:])


class NumericalRegressionTests(unittest.TestCase):
    def test_public_api_surface_and_return_keys_are_stable(self):
        self.assertEqual(
            rdlocrand_pkg.__all__,
            ["rdrandinf", "rdwinselect", "rdsensitivity", "rdrbounds"],
        )
        for name in rdlocrand_pkg.__all__:
            self.assertIs(getattr(rdlocrand_pkg, name), globals()[name])

        self.assertEqual(list(inspect.signature(rdrandinf).parameters)[:5], ["Y", "R", "cutoff", "wl", "wr"])
        self.assertEqual(list(inspect.signature(rdwinselect).parameters)[:3], ["R", "X", "cutoff"])
        self.assertEqual(list(inspect.signature(rdsensitivity).parameters)[:4], ["Y", "R", "cutoff", "wlist"])
        self.assertEqual(list(inspect.signature(rdrbounds).parameters)[:6], ["Y", "R", "cutoff", "wlist", "gamma", "expgamma"])

        Y, _, R, X, _ = make_regression_data()
        with contextlib.redirect_stdout(io.StringIO()):
            inference = rdrandinf(Y, R, wl=-0.65, wr=0.65, reps=5, seed=123, quietly=True)
            windows = rdwinselect(R, X, wmin=0.45, wstep=0.2, nwindows=1, reps=5, seed=123, quietly=True)
            sensitivity = rdsensitivity(
                Y, R,
                wlist=np.array([0.65]),
                tlist=np.array([0, 1]),
                reps=5, seed=123, nodraw=True, quietly=True,
            )
            bounds = rdrbounds(Y, R, expgamma=[1.5], wlist=[0.65], reps=5, seed=123)

        self.assertEqual(set(inference), {"sumstats", "obs.stat", "p.value", "asy.pvalue", "window"})
        self.assertEqual(set(windows), {"w_left", "w_right", "wlist_left", "wlist_right", "results", "summary"})
        self.assertEqual(set(sensitivity), {"tlist", "wlist", "wlist_left", "results"})
        self.assertEqual(set(bounds), {"gamma", "expgamma", "wlist", "p.values", "lower.bound", "upper.bound"})

    def test_rdwinselect_fixed_seed_output_is_stable(self):
        _, _, R, X, _ = make_regression_data()
        out = rdwinselect(
            R, X,
            wmin=0.45, wstep=0.2, nwindows=4,
            reps=40, seed=123, quietly=True,
        )

        assert_allclose([out["w_left"], out["w_right"]], [-1.05, 1.05], atol=1e-12)
        assert_allclose(
            out["results"].to_numpy(),
            np.array([
                [0.375, 0, 1, 9, 9, -0.45, 0.45],
                [0.925, 0, 1, 13, 13, -0.65, 0.65],
                [0.475, 0, 1, 17, 17, -0.85, 0.85],
                [0.575, 0, 1, 21, 21, -1.05, 1.05],
            ]),
            atol=1e-12,
        )

    def test_rdwinselect_missing_covariate_filtering_is_stable(self):
        _, _, R, X, _ = make_regression_data()
        X[[19, 23, 36], 0] = np.nan
        X[[24, 40], 1] = np.nan

        with contextlib.redirect_stdout(io.StringIO()):
            default = rdwinselect(
                R, X,
                wmin=0.45, wstep=0.2, nwindows=4,
                reps=30, seed=123, quietly=True,
            )
            dropmissing = rdwinselect(
                R, X,
                wmin=0.45, wstep=0.2, nwindows=4,
                reps=30, seed=123, quietly=True, dropmissing=True,
            )

        assert_allclose([default["w_left"], default["w_right"]], [np.nan, np.nan])
        assert_allclose([dropmissing["w_left"], dropmissing["w_right"]], [np.nan, np.nan])
        assert_allclose(
            default["results"].to_numpy(),
            np.array([
                [0.1, 0, 1, 7, 8, -0.45, 0.45],
                [0.366666666666667, 0, 1, 10, 11, -0.65, 0.65],
                [0.166666666666667, 0, 1, 14, 15, -0.85, 0.85],
                [0.333333333333333, 0, 1, 18, 19, -1.05, 1.05],
            ]),
            atol=1e-12,
        )
        assert_allclose(dropmissing["results"].to_numpy(), default["results"].to_numpy(), atol=1e-12)
        assert_allclose(
            default["summary"].to_numpy(),
            np.array([[30, 30], [0, 0], [1, 1], [3, 3], [5, 5]]),
            atol=1e-12,
        )
        assert_allclose(
            dropmissing["summary"].to_numpy(),
            np.array([[27, 28], [0, 0], [1, 1], [3, 3], [6, 5]]),
            atol=1e-12,
        )

    def test_rdrandinf_fixed_seed_outputs_are_stable(self):
        Y, Y_fuzzy, R, _, T = make_regression_data()

        out = rdrandinf(
            Y, R,
            wl=-0.85, wr=0.85, statistic="all",
            reps=40, seed=123, quietly=True,
        )
        assert_allclose(
            out["obs.stat"],
            [2.143401733273603, 0.823529411764706, -4.735985648132588],
            atol=1e-12,
        )
        assert_allclose(out["p.value"], [0, 0, 0], atol=1e-12)
        assert_allclose(
            out["asy.pvalue"],
            [9.048374510988886e-13, 5.128543066704716e-06, 2.179930150728572e-06],
            atol=1e-18,
        )

        out = rdrandinf(
            Y, R,
            wl=-0.85, wr=0.85,
            bernoulli=np.repeat(0.55, len(R)),
            interfci=0.1,
            reps=40, seed=123, quietly=True,
        )
        assert_allclose(out["obs.stat"], [2.143401733273603], atol=1e-12)
        self.assertEqual(out["p.value"], 0)
        assert_allclose(out["asy.pvalue"], [9.048374510988886e-13], atol=1e-18)
        assert_allclose(out["interf.ci"], [1.36144455343675, 3.001019130949376], atol=1e-12)

        out = rdrandinf(
            Y, R,
            wl=-0.85, wr=0.85,
            ci=[0.1, 0, 1, 2],
            reps=40, seed=123, quietly=True,
        )
        assert_allclose(out["ci"], [[2, 2]], atol=1e-12)

        out = rdrandinf(
            Y_fuzzy, R,
            wl=-0.85, wr=0.85,
            fuzzy=[T, "itt"],
            reps=40, seed=123, quietly=True,
        )
        assert_allclose(out["obs.stat"], [1.627708522122618], atol=1e-12)
        self.assertEqual(out["p.value"], 0)
        assert_allclose(out["asy.pvalue"], [2.661349516867963e-08], atol=1e-18)

    def test_rdsensitivity_fixed_seed_output_is_stable(self):
        Y, _, R, _, _ = make_regression_data()
        out = rdsensitivity(
            Y, R,
            wlist=np.array([0.65, 0.85]),
            tlist=np.array([0, 1, 2]),
            ci=np.array([-0.85, 0.85]),
            reps=40, seed=123, nodraw=True, quietly=True,
        )

        assert_allclose(out["results"], [[0, 0], [0.05, 0], [0.125, 0.6]], atol=1e-12)
        assert_allclose(out["ci"], [[2, 2]], atol=1e-12)

    def test_rdrbounds_fixed_seed_output_is_stable(self):
        Y, _, R, _, _ = make_regression_data()
        with contextlib.redirect_stdout(io.StringIO()):
            out = rdrbounds(Y, R, expgamma=[1.5], wlist=[0.65], reps=20, seed=123)

        assert_allclose(out["p.values"], [[0]], atol=1e-12)
        assert_allclose(out["lower.bound"], [[0]], atol=1e-12)
        assert_allclose(out["upper.bound"], [[0]], atol=1e-12)

    def test_rdrbounds_fmpval_false_keeps_bernoulli_pvalues(self):
        n = 40
        idx = np.arange(1, n + 1, dtype=float)
        R = np.linspace(-1, 1, n)
        Y = np.sin(idx / 4) + 0.2 * (R >= 0) + 0.1 * R
        args = dict(
            expgamma=[1.5],
            wlist=[0.7],
            reps=200,
            seed=123,
            statistic="diffmeans",
        )

        with contextlib.redirect_stdout(io.StringIO()):
            bernoulli_only = rdrbounds(Y, R, fmpval=False, **args)
            with_fixed_margins = rdrbounds(Y, R, fmpval=True, **args)

        assert_allclose(bernoulli_only["p.values"], with_fixed_margins["p.values"][:1, :], atol=1e-12)
        assert_allclose(bernoulli_only["p.values"], [[0.02]], atol=1e-12)

    def test_documented_python_paths_remain_callable(self):
        Y, _, R, X, _ = make_regression_data()

        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            sensitivity = rdsensitivity(Y, R, reps=5, seed=123, nodraw=True, quietly=True)
            bounds = rdrbounds(Y, R, reps=5, seed=123)
            hotelling = rdwinselect(
                R, X,
                wmin=0.45, wstep=0.2, nwindows=2,
                statistic="hotelling", reps=5, seed=123, quietly=True,
            )

        assert_allclose(
            sensitivity["wlist"],
            [0.483050847457627, 0.737288135593221, 0.991525423728814, 1.245762711864407],
            atol=1e-12,
        )
        self.assertEqual(sensitivity["results"].shape, (12, 4))
        assert_allclose(sensitivity["results"][:3, :3], [[0, 0, 0], [0.4, 0, 0], [0.4, 0, 0]], atol=1e-12)
        assert_allclose(bounds["wlist"], [0.48, 0.74, 0.99, 1.25], atol=1e-12)
        assert_allclose(bounds["p.values"], [[0, 0, 0, 0]], atol=1e-12)
        assert_allclose(
            hotelling["results"].to_numpy(),
            [[0.8, np.nan, 1, 9, 9, -0.45, 0.45], [1, np.nan, 1, 13, 13, -0.65, 0.65]],
            atol=1e-12,
        )

        triangular = rdrandinf(
            Y, R,
            wl=-0.85, wr=0.85,
            kernel="triangular",
            reps=5, seed=123, quietly=True,
        )
        assert_allclose(triangular["obs.stat"], [1.360066454465187], atol=1e-12)
        self.assertEqual(triangular["p.value"], 0)
        assert_allclose(triangular["asy.pvalue"], [1.74732273695206e-07], atol=1e-18)

    def test_fast_statistics_match_scipy_reference(self):
        x = np.array([-1.2, -0.4, 0.1, 0.7, 1.5])
        y = np.array([-1.1, -0.8, 0.2, 0.4, 1.0, 1.7])
        x_ties = np.array([-1, -1, 0, 0.5, 1])
        y_ties = np.array([-0.5, 0, 0, 1, 1.5])

        self.assertAlmostEqual(ksmirnov_statistic(x, y), ks_2samp(x, y).statistic)
        self.assertAlmostEqual(ksmirnov_statistic(x_ties, y_ties), ks_2samp(x_ties, y_ties).statistic)
        self.assertAlmostEqual(ranksum_statistic(x, y), ranksums(x, y).statistic)
        self.assertAlmostEqual(ranksum_statistic(x_ties, y_ties), ranksums(x_ties, y_ties).statistic)

    def test_fast_bernoulli_ranksum_matches_public_path(self):
        Y, _, R, _, _ = make_regression_data()
        in_window = (R >= -0.85) & (R <= 0.85)
        Yw = Y[in_window]
        Rw = R[in_window]
        prob = np.linspace(0.25, 0.75, len(Rw))
        ref = rdrandinf(
            Yw, Rw,
            wl=-0.85, wr=0.85,
            bernoulli=prob,
            statistic="ranksum",
            reps=50, seed=666, quietly=True,
        )
        fast = rdrandinf_bernoulli_ranksum_pvalue(
            Yw, Rw, prob,
            reps=50, nulltau=0, seed=666,
        )

        self.assertAlmostEqual(fast, ref["p.value"])

    def test_seeded_calls_restore_numpy_rng_state(self):
        Y, _, R, X, _ = make_regression_data()

        assert_rng_unchanged(
            self,
            lambda: rdwinselect(
                R, X,
                wmin=0.45, wstep=0.2, nwindows=2,
                reps=5, seed=123, quietly=True,
            ),
        )
        assert_rng_unchanged(
            self,
            lambda: rdrandinf(
                Y, R,
                wl=-0.85, wr=0.85,
                reps=5, seed=123, quietly=True,
            ),
        )
        assert_rng_unchanged(
            self,
            lambda: rdsensitivity(
                Y, R,
                wlist=np.array([0.65]),
                tlist=np.array([0, 1]),
                reps=5, seed=123, nodraw=True, quietly=True,
            ),
        )
        assert_rng_unchanged(
            self,
            lambda: self._run_quiet_rdrbounds(Y, R),
        )

        assert_rng_unchanged(
            self,
            lambda: rdrandinf(
                Y, R,
                wl=-0.85, wr=0.85,
                reps=5, seed=-1, quietly=True,
            ),
        )

    @staticmethod
    def _run_quiet_rdrbounds(Y, R):
        with contextlib.redirect_stdout(io.StringIO()):
            return rdrbounds(
                Y, R,
                expgamma=[1.5], wlist=[0.65],
                reps=5, seed=123,
            )


if __name__ == "__main__":
    unittest.main()
