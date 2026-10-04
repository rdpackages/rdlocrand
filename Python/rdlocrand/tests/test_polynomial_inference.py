"""Polynomial OLS/IV inference against independent sandwich calculations."""
import contextlib
import io
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from numpy.testing import assert_allclose
from scipy.stats import norm
from rdlocrand import rdrandinf, rdwinselect, rdsensitivity, rdrbounds
from rdlocrand.rdlocrand_fun import find_CI


def call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def data():
    r = 2 + np.r_[np.linspace(-.9, -.015, 48), np.linspace(.01, 1.1, 63)]
    d = (r >= 2).astype(float)
    i = np.arange(1, len(r)+1)
    t = .7*d + .4*np.sin(i*.41) + .2*(r-2) + .1*d*(r-2)
    y = 1.2*t + .5*(r-2) - .25*d*(r-2) + (np.sin(i*1.7)+.4*np.cos(i*.33))*(1+.5*(r-2))
    return y, r, t


def reference(y, r, t, p, kernel, vce, evall=2, evalr=2):
    d = (r >= 2).astype(float)
    u = (r-2)/np.where(d == 1, 1.1, -.9)
    w = {'uniform': np.ones(len(y)), 'triangular': np.maximum(1-np.abs(u), 0),
         'epan': .75*np.maximum(1-u*u, 0)}[kernel]
    keep = w > 1e-14
    y, r, d, w = y[keep], r[keep], d[keep], w[keep]
    z = np.column_stack([np.ones(len(y)), d] + [(r-d*evalr-(1-d)*evall)**j for j in range(1, p+1)] +
                        [d*(r-d*evalr-(1-d)*evall)**j for j in range(1, p+1)])
    if t is None:
        fit = sm.WLS(y, z, weights=w).fit(cov_type=vce)
        return fit.params[1], fit.bse[1]
    x = z.copy(); x[:, 1] = t[keep]
    root = np.sqrt(w[:, None]); zw = z*root; xw = x*root
    projected = zw @ np.linalg.solve(zw.T @ zw, zw.T @ xw)
    bread = np.linalg.inv(projected.T @ projected)
    beta = bread @ projected.T @ (y*np.sqrt(w))
    residual = (y-x@beta)*np.sqrt(w)
    h = np.einsum('ij,jk,ik->i', projected, bread, projected)
    scale = np.full(len(y), len(y)/(len(y)-x.shape[1])) if vce == 'HC1' else (1-h)**(-int(vce[-1])+1)
    meat = projected.T @ ((residual**2*scale)[:, None]*projected)
    se = np.sqrt((bread @ meat @ bread)[1, 1])
    if vce == 'HC1':
        independent = IV2SLS(y, z[:, np.r_[0, 2:z.shape[1]]], x[:, 1], z[:, 1], weights=w).fit(cov_type='robust', debiased=True)
        assert_allclose([beta[1], se], [independent.params.iloc[-1], independent.std_errors.iloc[-1]], rtol=1e-8)
    return beta[1], se


class PolynomialInference(unittest.TestCase):
    def setUp(self):
        self.y, self.r, self.t = data()
        self.opts = dict(cutoff=2, wl=1.1, wr=3.1, quietly=True, nulltau=.6)

    def test_sharp_ar_and_iv_sandwich(self):
        for p in (1, 2):
            for kernel in ('uniform', 'triangular', 'epan'):
                for vce in ('HC1', 'HC2', 'HC3'):
                    for kind in ('sharp', 'ar', 'tsls'):
                        with self.subTest(p=p, kernel=kernel, vce=vce, kind=kind):
                            opts = dict(self.opts, p=p, kernel=kernel, vce=vce)
                            yref = self.y-.6*(self.r >= 2) if kind == 'sharp' else self.y-.6*self.t
                            tref = None
                            if kind == 'ar': opts['fuzzy'] = self.t
                            if kind == 'tsls':
                                opts['fuzzy'] = [self.t, 'tsls']; yref = self.y; tref = self.t
                            estimate, se = reference(yref, self.r, tref, p, kernel, vce)
                            out = call(rdrandinf, self.y, self.r, **opts)
                            assert_allclose([out['obs.stat'], out['se']], [estimate, se], rtol=1e-8, atol=1e-11)
                            expected = 2*norm.cdf(-abs((estimate-(.6 if kind == 'tsls' else 0))/se))
                            assert_allclose(out['asy.pvalue'], expected, rtol=1e-8, atol=1e-11)
                            self.assertTrue(np.isnan(out['p.value']))
                            self.assertEqual(out['inference'], 'large-sample')

    def test_default_custom_evaluation_and_no_permutations(self):
        with patch('numpy.random.choice', side_effect=AssertionError('permutation called')):
            out = call(rdrandinf, self.y, self.r, p=2, evall=1.85, evalr=2.15, **self.opts)
            estimate, se = reference(self.y-.6*(self.r >= 2), self.r, None, 2, 'uniform', 'HC3', 1.85, 2.15)
            assert_allclose([out['obs.stat'], out['se']], [estimate, se], rtol=1e-9)
            self.assertEqual(out['vce'], 'HC3')
            call(rdwinselect, self.r, np.column_stack((self.y, self.t)), cutoff=2,
                 wmin=.8, wstep=.1, nwindows=2, p=1, quietly=True)

    def test_confidence_sets_and_sensitivity(self):
        for kind in ('sharp', 'tsls'):
            opts = dict(self.opts, p=1, ci=.1)
            if kind == 'tsls': opts['fuzzy'] = [self.t, 'tsls']
            out = call(rdrandinf, self.y, self.r, **opts)
            estimate = out['obs.stat'] + (.6 if kind == 'sharp' else 0)
            assert_allclose(out['ci'], [[estimate-norm.ppf(.95)*out['se'], estimate+norm.ppf(.95)*out['se']]])
        grid = np.arange(-2, 4.01, .25)
        for fuzzy in (None, self.t):
            opts = dict(self.opts, p=2, fuzzy=fuzzy, ci=np.r_[.1, grid], kernel='triangular')
            out = call(rdrandinf, self.y, self.r, **opts)
            sens = call(rdsensitivity, self.y, self.r, cutoff=2, wlist=[3.1], wlist_left=[1.1],
                        p=2, fuzzy=fuzzy, tlist=grid, kernel='triangular', nodraw=True, quietly=True)
            assert_allclose(out['ci'], find_CI(sens['results'][:, 0], .1, grid))
        assert_allclose(find_CI([.2, 0, .4, .4, 0], .1, np.arange(5)), [[0, 0], [2, 3]])
        sens = call(rdsensitivity, self.y, self.r, cutoff=2, wlist=[3.1], wlist_left=[1.1],
                    p=1, tlist=[.6], evalat='means', nodraw=True, quietly=True)
        out = call(rdrandinf, self.y, self.r, p=1, evall=np.mean(self.r[self.r<2]),
                   evalr=np.mean(self.r[self.r>=2]), **self.opts)
        assert_allclose(sens['results'][0, 0], out['asy.pvalue'])

    def test_missing_fuzzy_observations(self):
        y, r, t = self.y.copy(), self.r.copy(), self.t.copy()
        y[3] = np.nan; r[11] = np.nan; t[16] = np.nan
        keep = np.isfinite(y+r+t)
        for kind in ('ar', 'tsls'):
            fuzzy = t if kind == 'ar' else [t, 'tsls']
            clean_fuzzy = t[keep] if kind == 'ar' else [t[keep], 'tsls']
            a = call(rdrandinf, y, r, fuzzy=fuzzy, p=1, **self.opts)
            b = call(rdrandinf, y[keep], r[keep], fuzzy=clean_fuzzy, p=1, **self.opts)
            assert_allclose([a['obs.stat'], a['se']], [b['obs.stat'], b['se']])

    def assert_fallback(self, func, args, opts, keys):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            a = call(func, *args, p=2, **opts)
        messages = [str(w.message) for w in seen if 'Polynomial adjustment' in str(w.message)]
        self.assertEqual(len(messages), 1)
        self.assertIn('p=0', messages[0])
        self.assertEqual((a['p.requested'], a['p']), (2, 0))
        b = call(func, *args, p=0, **opts)
        for key in keys: assert_allclose(a[key], b[key], equal_nan=True)

    def test_fallbacks(self):
        for stat in ('ksmirnov', 'ranksum', 'all'):
            self.assert_fallback(rdrandinf, (self.y, self.r), dict(self.opts, statistic=stat, reps=29), ['obs.stat', 'p.value', 'asy.pvalue'])
        self.assert_fallback(rdrandinf, (self.y, self.r), dict(self.opts, interfci=.1, reps=29), ['interf.ci', 'p.value'])
        for stat in ('ksmirnov', 'ranksum', 'hotelling'):
            self.assert_fallback(rdwinselect, (self.r, np.column_stack((self.y, self.t))),
                                 dict(cutoff=2, wmin=.8, wstep=.1, nwindows=2, statistic=stat, reps=29, quietly=True), ['results'])
        self.assert_fallback(rdsensitivity, (self.y, self.r), dict(cutoff=2, wlist=[3.1], wlist_left=[1.1], tlist=[0, 1],
                             statistic='ksmirnov', reps=29, nodraw=True, quietly=True), ['results'])
        self.assert_fallback(rdrbounds, (self.y, self.r-2), dict(wlist=[.8], expgamma=[1.2], reps=19), ['p.values', 'lower.bound', 'upper.bound'])

    def test_nested_window_warning_and_zero_weight_balance(self):
        x = np.cos(np.arange(len(self.r))*.71)[:, None]
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            out = call(rdrandinf, self.y, self.r, cutoff=2, covariates=x, p=1,
                       rdwstat='ksmirnov', wmin=.5, wstep=.1, nwindows=2,
                       level=0, rdwreps=29, quietly=True)
        self.assertEqual(out['p'], 1)
        self.assertEqual(sum('Polynomial adjustment' in str(w.message) for w in seen), 1)
        r = np.linspace(-1, 1, 80)
        y = np.sin(np.arange(80))
        for vce in ('HC1', 'HC2', 'HC3'):
            balance = call(rdwinselect, r, y[:, None], p=1, vce=vce, kernel='triangular',
                           wmin=1, wstep=.1, nwindows=1, quietly=True)
            direct = call(rdrandinf, y, r, p=1, vce=vce, kernel='triangular', wl=-1, wr=1, quietly=True)
            assert_allclose(np.asarray(balance['results'])[0, 0], direct['asy.pvalue'])

    def test_p0_fuzzy_with_current_pandas(self):
        for fuzzy in (self.t, [self.t, 'tsls']):
            out = call(rdrandinf, self.y, self.r, fuzzy=fuzzy, **self.opts)
            self.assertTrue(np.all(np.isfinite(out['asy.pvalue'])))
        # The p=0 TSLS variance convention is preserved.
        fit = IV2SLS(self.y, np.ones(len(self.y)), self.t,
                     (self.r >= 2).astype(float)).fit(cov_type='robust')
        assert_allclose(out['obs.stat'], fit.params.iloc[1])
        assert_allclose(out['asy.pvalue'], 2*norm.cdf(-abs((fit.params.iloc[1]-.6)/fit.std_errors.iloc[1])))

    def test_p0_tsls_nulls_and_confidence_levels(self):
        d = (self.r >= 2).astype(float)
        u = (self.r-2)/np.where(d == 1, 1.2, -1.)
        for kernel in ('uniform', 'triangular', 'epan'):
            weights = {'uniform': np.ones(len(u)), 'triangular': 1-np.abs(u), 'epan': .75*(1-u*u)}[kernel]
            fit = IV2SLS(self.y, np.ones(len(self.y)), self.t, d, weights=weights).fit(cov_type='robust')
            beta, se = fit.params.iloc[1], fit.std_errors.iloc[1]
            for tau in (0, .6, beta):
                for alpha in (.2, .1, .05):
                    out = call(rdrandinf, self.y, self.r, cutoff=2, wl=1, wr=3.2,
                               fuzzy=[self.t, 'tsls'], kernel=kernel, nulltau=tau,
                               ci=alpha, quietly=True)
                    assert_allclose(out['obs.stat'], beta)
                    assert_allclose(out['asy.pvalue'], 2*norm.cdf(-abs((beta-tau)/se)), atol=1e-12)
                    assert_allclose(out['ci'], [[beta-norm.ppf(1-alpha/2)*se, beta+norm.ppf(1-alpha/2)*se]])

    def test_invalid_and_unidentified_models(self):
        for opts in ({'p': -1}, {'p': .5}, {'p': 1, 'vce': 'HC0'}):
            with self.assertRaises(ValueError): call(rdrandinf, self.y, self.r, **dict(self.opts, **opts))
        for r in (np.r_[-np.ones(10), np.ones(10)], np.r_[-.2, -.1, np.linspace(.1, 1, 10)]):
            with self.assertRaises(ValueError):
                call(rdrandinf, np.sin(np.arange(len(r))), r, p=1, wl=-1, wr=1, quietly=True)
        with self.assertRaises(ValueError):
            call(rdrandinf, self.y, self.r, fuzzy=[self.r-2, 'tsls'], p=1, **self.opts)


if __name__ == '__main__':
    unittest.main()
