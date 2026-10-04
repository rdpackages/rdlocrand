"""Independent numerical checks for the window, KS, and sharp-null corrections."""
import contextlib
import io
import unittest
import warnings

import numpy as np
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy.stats import ks_2samp, norm

from rdlocrand import rdrandinf, rdwinselect
from rdlocrand.rdlocrand_fun import ksmirnov_statistic


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return func(*args, **kwargs)


class NumericalRegression(unittest.TestCase):
    def test_window_membership(self):
        r = np.r_[-np.arange(1, 41), np.arange(40) + .5]
        x = np.random.default_rng(1).normal(size=(80, 1))
        out = quiet(rdwinselect, r, x, nwindows=3, approx=True)
        assert_allclose(np.asarray(out['results'])[0, 3:7], [10, 10, -10, 10])
        for wmin in (10, [-10, 10]):
            out = quiet(rdwinselect, r, x, wmin=wmin, wasymmetric=True,
                        nwindows=3, approx=True)
            for row in np.asarray(out['results']):
                lo, hi = row[5:7]
                assert_allclose(row[3:5], [np.sum((r >= lo) & (r < 0)),
                                           np.sum((r >= 0) & (r <= hi))])

    def test_masspoint_windows(self):
        r = np.r_[np.repeat(-np.arange(1, 11), 3), np.repeat(np.arange(10)+.5, 2)]
        x = np.random.default_rng(3).normal(size=(50, 1))
        out = quiet(rdwinselect, r, x, wmasspoints=True, nwindows=3, approx=True)
        expected = np.column_stack((3*np.arange(1, 4), 2*np.arange(1, 4),
                                    -np.arange(1, 4), np.arange(3)+.5))
        assert_allclose(np.asarray(out['results'])[:, 3:7], expected)

    def test_ks_ties(self):
        rng = np.random.default_rng(41)
        cases = [(np.array([0, 0, 0, 1, 1]), np.array([0, 0, 0, 1, 1]))]
        cases += [(rng.integers(0, 4, 7), rng.integers(0, 4, 9)) for _ in range(40)]
        for x, y in cases:
            assert_allclose(ksmirnov_statistic(x, y), ks_2samp(x, y).statistic, atol=1e-14)
        r = rng.uniform(-1, 1, 60)
        y = rng.binomial(1, .3+.2*(r >= 0)).astype(float)
        d = (r >= 0).astype(int)
        observed = ks_2samp(y[d == 0], y[d == 1]).statistic
        for seed in (1, 3, 10):
            np.random.seed(seed)
            simulated = []
            for _ in range(199):
                ds = np.random.choice(d, size=len(d), replace=False)
                simulated.append(ks_2samp(y[ds == 0], y[ds == 1]).statistic)
            expected = np.mean(np.asarray(simulated) >= observed)
            out = quiet(rdrandinf, y, r, wl=-1, wr=1, statistic='ksmirnov', reps=199, seed=seed)
            assert_allclose(out['p.value'], expected)

    def test_polynomial_balance(self):
        rng = np.random.default_rng(9)
        r = rng.uniform(-5, 5, 400)
        x = np.column_stack((rng.integers(1, 4, 400), rng.poisson(2, 400), rng.poisson(1, 400)))
        for p in (0, 1, 2):
            for kernel in ('uniform', 'triangular', 'epan'):
                out = quiet(rdwinselect, r, x, wmin=.5, wstep=.25, nwindows=4,
                            approx=True, p=p, kernel=kernel, vce="HC2")
                for row in np.asarray(out['results']):
                    w = row[6]
                    inside = np.abs(r) <= w
                    rw = r[inside]; d = (rw >= 0).astype(float)
                    weights = {'uniform': np.ones(len(rw)), 'triangular': 1-np.abs(rw/w),
                               'epan': .75*(1-(rw/w)**2)}[kernel]
                    weights[weights == 0] = np.finfo(float).eps
                    design = np.column_stack([np.ones(len(rw)), d] +
                                             [rw**j for j in range(1, p+1)] +
                                             [d*rw**j for j in range(1, p+1)])
                    reference = []
                    for k in range(x.shape[1]):
                        fit = sm.WLS(x[inside, k], design, weights=weights).fit(cov_type='HC2')
                        reference.append(2*norm.cdf(-abs(fit.params[1]/fit.bse[1])))
                    assert_allclose(row[0], min(reference), rtol=1e-9, atol=1e-12)

    def test_multiple_adjusted_covariates(self):
        rng = np.random.default_rng(109)
        r = rng.uniform(-1, 1, 80)
        x = rng.normal(size=(80, 3)) + np.column_stack((r, 2*r, -r))
        opts = dict(wmin=1, wstep=.1, nwindows=1, p=1, reps=99, seed=20)
        out = quiet(rdwinselect, r, x, **opts)
        separate = [np.asarray(quiet(rdwinselect, r, x[:, [k]], **opts)['results'])[0, 0]
                    for k in range(x.shape[1])]
        assert_allclose(np.asarray(out['results'])[0, 0], min(separate))

    def test_nonzero_null(self):
        rng = np.random.default_rng(2)
        r = rng.uniform(-1, 1, 80)
        d = (r >= 0).astype(float)
        y = 2*d + rng.normal(size=80)
        for p in (1, 2):
            for kernel in ('uniform', 'triangular', 'epan'):
                weights = {'uniform': np.ones(len(r)), 'triangular': 1-np.abs(r),
                           'epan': .75*(1-r**2)}[kernel]
                design = np.column_stack([np.ones(len(r)), d] +
                                         [r**j for j in range(1, p+1)] +
                                         [d*r**j for j in range(1, p+1)])
                fit = sm.WLS(y, design, weights=weights).fit(cov_type='HC2')
                for tau in (0, 2, fit.params[1]):
                    out = quiet(rdrandinf, y, r, wl=-1, wr=1, p=p, kernel=kernel,
                                nulltau=tau, reps=29, vce="HC2")
                    expected = 2*norm.cdf(-abs((fit.params[1]-tau)/fit.bse[1]))
                    assert_allclose(out['asy.pvalue'], expected, rtol=1e-9, atol=1e-12)

    def test_rng_preservation(self):
        r = np.linspace(-1, 1, 80)
        y = np.sin(7*r)
        np.random.seed(902)
        before = np.random.get_state()
        one = quiet(rdrandinf, y, r, wl=-1, wr=1, reps=29, seed=10)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        two = quiet(rdrandinf, y, r, wl=-1, wr=1, reps=29, seed=10)
        assert_allclose(one['p.value'], two['p.value'])


if __name__ == '__main__':
    unittest.main()
