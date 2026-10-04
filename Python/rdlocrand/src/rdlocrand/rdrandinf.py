#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from rdlocrand.rdlocrand_fun import rdlocrand_inference, rdlocrand_hc_fit
from scipy.stats import norm
import statsmodels.api as sm
from scipy.special import comb
from linearmodels import IV2SLS
from rdlocrand.rdwinselect import rdwinselect
from rdlocrand.rdlocrand_fun import rdrandinf_model, find_CI, rdlocrand_preserve_rng

@rdlocrand_preserve_rng
def rdrandinf(Y, R, cutoff=0, wl=None, wr=None, statistic='diffmeans', p=0, evall=None, evalr=None, kernel='uniform',
              fuzzy=None, nulltau=0, d=None, dscale=None, ci=None, interfci=None, bernoulli=None, reps=1000, seed=666,
              quietly=False, covariates=None, obsmin=None, wmin=None, wobs=None, wstep=None, wasymmetric=False,
              wmasspoints=False, nwindows=10, dropmissing=False, rdwstat='diffmeans', approx=False, rdwreps=1000,
              level=0.15, plot=False, firststage=False, obsstep=None, vce='HC3'):
    
    """
    Randomization Inference for RD Designs under Local Randomization

    rdrandinf implements randomization inference and related methods for RD designs,
    using observations in a specified or data-driven selected window around the cutoff where
    local randomization is assumed to hold.

    Authors:
    Matias D. Cattaneo, Princeton University. Email: matias.d.cattaneo@gmail.com
    Ricardo Masini, UC Davis. Email: ricardo.masini@gmail.com
    Rocio Titiunik, Princeton University. Email: rocio.titiunik@gmail.com
    Gonzalo Vazquez-Bare, UC Santa Barbara. Email: gvazquezbare@gmail.com

    References:
    Cattaneo, M.D., B. Frandsen, and R. Titiunik. (2015).
    Randomization Inference in the Regression Discontinuity Design:
    An Application to Party Advantages in the U.S. Senate.
    Journal of Causal Inference 3(1): 1-24.
    URL: https://rdpackages.github.io/references/Cattaneo-Frandsen-Titiunik_2015_JCI.pdf

    Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2016).
    Inference in Regression Discontinuity Designs under Local Randomization.
    Stata Journal 16(2): 331-367.
    URL: https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf

    Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2017).
    Comparing Inference Approaches for RD Designs:
    A Reexamination of the Effect of Head Start on Child Mortality.
    Journal of Policy Analysis and Management 36(3): 643-681.
    URL: https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2017_JPAM.pdf

    Notes:
    With p > 0, mean contrasts, fuzzy Anderson-Rubin tests, and fuzzy TSLS/Wald
    use large-sample normal inference from the full polynomial regression, with
    HC3 standard errors by default; no randomization p-value is computed.
    HC3 does not guarantee finite-sample size control; TSLS/Wald also requires a sufficiently strong first stage.
    TSLS uses structural residuals and leverage from projected second-stage regressors.
    Fuzzy Anderson-Rubin confidence sets require an explicit treatment-effect grid in ci.
    For ksmirnov, ranksum, all, or interfci, a positive p is replaced by p = 0
    and a warning is issued at the end. With p = 0, existing inference methods
    and variance defaults are retained; TSLS tests honor nulltau and the requested CI level.

    Parameters:
    -----------
    Y : array-like
        A vector containing the values of the outcome variable.
    R : array-like
        A vector containing the values of the running variable.
    cutoff : float, optional
        The RD cutoff (default is 0).
    wl : float, optional
        The left limit of the window. The default takes the minimum of the running variable.
    wr : float, optional
        The right limit of the window. The default takes the maximum of the running variable.
    statistic : str, optional
        The randomization test statistic to be used. Allowed options are 'diffmeans' (difference in means statistic),
        'ksmirnov' (Kolmogorov-Smirnov statistic), 'ranksum' (Wilcoxon-Mann-Whitney standardized statistic), and 'all'.
        Default option is 'diffmeans'. The statistic 'ttest' is equivalent to 'diffmeans' and included for backward compatibility.
    p : int, optional
        The order of the polynomial for the outcome adjustment model (default is 0).
    evall : float, optional
        The point to the left of the cutoff at which the adjusted outcome is evaluated. Default is the cutoff value.
    evalr : float, optional
        The point to the right of the cutoff at which the adjusted outcome is evaluated. Default is the cutoff value.
    kernel : str, optional
        Specifies the type of kernel to use as a weighting scheme. Allowed kernel types are 'uniform' (uniform kernel),
        'triangular' (triangular kernel), and 'epan' (Epanechnikov kernel). Default is 'uniform'.
    fuzzy : None or tuple or array-like, optional
        Indicates that the RD design is fuzzy. If fuzzy is None, the RD design is not fuzzy. If fuzzy is a list,
        the first element should be the vector of endogenous treatment values, and the second element should be a string
        containing the statistic to be used. Allowed statistics are 'ar' or 'itt' (Anderson-Rubin/intention-to-treat
        statistic) and 'tsls' (2SLS statistic). Default statistic is 'ar'. The 'tsls' statistic relies on a large-sample
        approximation.
    nulltau : float, optional
        The value of the treatment effect under the null hypothesis (default is 0).
    d : float, optional
        The effect size for asymptotic power calculation. Default is 0.5 times the standard deviation of the outcome variable for the control group.
    dscale : float, optional
        The fraction of the standard deviation of the outcome variable for the control group used as an alternative hypothesis
        for asymptotic power calculation. Default is 0.5.
    ci : float or array-like, optional
        Calculates a confidence interval for the treatment effect. The first element specifies alpha
        (typically 0.05 or 0.01); remaining elements specify a treatment-effect grid. TSLS uses normal
        intervals. Polynomial sharp-design mean contrasts use normal intervals unless a grid is supplied.
        Other cases invert tests over a grid; polynomial fuzzy Anderson-Rubin requires an explicit grid.
        At p = 0, test inversion uses rdsensitivity. Use a sufficiently wide grid to avoid truncating the confidence set.
    interfci : float, optional
        The level for Rosenbaum's confidence interval under arbitrary interference between units.
    bernoulli : array-like, optional
        The probabilities of treatment for each unit when the assignment mechanism is a Bernoulli trial.
        This option should be specified as a vector of length equal to the length of the outcome and running variables.
    reps : int, optional
        The number of replications (default is 1000).
    seed : int, optional
        The seed to be used for the randomization test.
    quietly : bool, optional
        Suppresses the output table.
    covariates : array-like, optional
        The covariates used by rdwinselect to choose the window when wl and wr are not specified.
        This should be a matrix of size n x k where n is the total sample size and k is the number of covariates.
    obsmin : int, optional
        The minimum number of observations above and below the cutoff in the smallest window used by the companion command rdwinselect.
        Default is 10.
    wmin : float, optional
        The smallest window to be used (if obsmin is not specified) by the companion command rdwinselect.
        Specifying both wmin and obsmin returns an error.
    wobs : int, optional
        The number of observations to be added on each side of the cutoff at each step.
    wstep : float, optional
        The increment in window length (if obsstep is not specified) by the companion command rdwinselect.
        Specifying both obsstep and wstep returns an error.
    wasymmetric : bool, optional
        Allows for asymmetric windows around the cutoff when wobs is specified.
    wmasspoints : bool, optional
        Specifies that the running variable is discrete and each mass point should be used as a window.
    nwindows : int, optional
        The number of windows to be used by the companion command rdwinselect. Default is 10.
    dropmissing : bool, optional
        Drop rows with missing values in covariates when calculating windows.
    rdwstat : str, optional
        The statistic to be used by the companion command rdwinselect (see the corresponding help file for options).
        Default option is 'diffmeans'.
    approx : bool, optional
        Forces the companion command rdwinselect to conduct the covariate balance tests using a large-sample approximation
        instead of finite-sample exact randomization inference methods.
    rdwreps : int, optional
        The number of replications to be used by the companion command rdwinselect. Default is 1000.
    level : float, optional
        The minimum accepted value of the p-value from the covariate balance tests to be used by the companion command rdwinselect.
        Default is 0.15.
    plot : bool, optional
        Draws a scatter plot of the minimum p-value from the covariate balance test against window length implemented
        by the companion command rdwinselect.
    firststage : bool, optional
        Reports the results from the first step when using tsls.
    obsstep : int, optional
        The minimum number of observations to be added on each side of the cutoff for the sequence of fixed-increment nested windows.
        Default is 2. This option is deprecated and only included for backward compatibility.
    
    vce : str, optional
        Variance estimator for p > 0: 'HC1', 'HC2', or 'HC3' (default).
        Ignored when p = 0.

    Returns
    -------
    dict
        Dictionary containing:

        - ``sumstats``: full-sample and window-specific summary statistics.
        - ``obs.stat``: observed statistic or statistics.
        - ``p.value``: randomization p-value or p-values; NaN when p > 0.
        - ``asy.pvalue``: asymptotic p-value or p-values.
        - ``window``: chosen window endpoints.
        - ``ci``: confidence interval; included only when ``ci`` is specified.
        - ``interf.ci``: confidence interval under interference; included only
          when ``interfci`` is specified.
        - ``p.requested``, ``p``: requested and effective polynomial degrees.
        - ``vce``: HC estimator, or None when p = 0.
        - ``inference``: inference method used.
        - ``se``: standard error when the effective p > 0.

    Example
    ------- 

    import numpy as np
    from rdlocrand import rdrandinf

    np.random.seed(123)
    X = np.random.normal(size=(100, 2))
    R = X[:, 0] + X[:, 1] + np.random.normal(size=100)
    Y = 1 + R - 0.5 * R**2 + 0.3 * R**3 + (R >= 0) + np.random.normal(size=100)

    # Randomization inference in window (-.75,.75)
    tmp = rdrandinf(Y, R, wl=-0.75, wr=0.75, quietly=True)

    # Randomization inference in window (-.75,.75), all statistics
    tmp = rdrandinf(Y, R, wl=-0.75, wr=0.75, statistic='all', quietly=True)

    # Randomization inference with window selection
    # Note: low number of replications to speed up the process.
    # The user should increase the number of replications.
    tmp = rdrandinf(
        Y, R,
        statistic='all', covariates=X,
        wmin=0.5, wstep=0.125, rdwreps=500,
        level=0, quietly=True,
    )
    """

    if isinstance(fuzzy, str) and fuzzy == '':
        fuzzy = None
    Y, R = np.asarray(Y, dtype=float), np.asarray(R, dtype=float)
    if ci is not None:
        ci = np.atleast_1d(ci).astype(float)
    randmech = 'fixed margins'
    Rc_long = R - cutoff

    if fuzzy is not None:
        statistic = ''
        if isinstance(fuzzy, list) and (len(fuzzy)==2):
            fuzzy_tr = np.array(fuzzy[0])
            if (fuzzy[1] == 'ar') or (fuzzy[1] == 'itt'):
                fuzzy_stat = 'ar'
            elif fuzzy[1] == 'tsls':
                fuzzy_stat = 'wald'
            else:
                raise ValueError('Invalid fuzzy statistic')
        else:
            fuzzy_stat = 'ar'
            fuzzy_tr = np.array(fuzzy)
    else:
        fuzzy_stat = ''
    inference = rdlocrand_inference(p, statistic, vce, 'interfci' if interfci is not None else None)
    p = inference['p']
    if p > 0:
        vce = inference['vce']

    if fuzzy is None:
        if bernoulli is None:
            data = np.column_stack((Y, R))
            data = data[~np.isnan(data).any(axis=1)]
            Y = data[:, 0]
            R = data[:, 1]
        else:
            data = np.column_stack((Y, R, bernoulli))
            data = data[~np.isnan(data).any(axis=1)]
            Y = data[:, 0]
            R = data[:, 1]
            bernoulli = data[:, 2]
    else:
        if bernoulli is None:
            data = np.column_stack((Y, R, fuzzy_tr))
            data = data[~np.isnan(data).any(axis=1)]
            Y = data[:, 0]
            R = data[:, 1]
            fuzzy_tr = data[:, 2]
        else:
            data = np.column_stack((Y, R, bernoulli, fuzzy_tr))
            data = data[~np.isnan(data).any(axis=1)]
            Y = data[:, 0]
            R = data[:, 1]
            bernoulli = data[:, 2]
            fuzzy_tr = data[:, 3]

    if cutoff < np.min(R) or cutoff > np.max(R):
        raise ValueError('Cutoff must be within the range of the running variable')

    if p < 0:
        raise ValueError('p must be a positive integer')

    if fuzzy is None:
        if statistic != 'diffmeans' and statistic != 'ttest' and statistic != 'ksmirnov' and statistic != 'ranksum' and statistic != 'all':
            raise ValueError('Invalid statistic')
    
    if kernel != 'uniform' and kernel != 'triangular' and kernel != 'epan':
        raise ValueError('Invalid kernel')
    
    if kernel != 'uniform' and evall is not None and evalr is not None:
        if evall != cutoff or evalr != cutoff:
            raise ValueError('Kernel only allowed when evall=evalr=cutoff')
    
    if kernel != 'uniform' and statistic != 'ttest' and statistic != 'diffmeans' and fuzzy is None:
        raise ValueError('Kernel only allowed for diffmeans')

    if ci is not None:
        if ci[0] > 1 or ci[0] < 0:
            raise ValueError('ci must be in [0,1]')
    
    if interfci is not None:
        if interfci > 1 or interfci < 0:
            raise ValueError('interfci must be in [0,1]')
        if statistic != 'diffmeans' and statistic != 'ttest' and statistic != 'ksmirnov' and statistic != 'ranksum':
            raise ValueError('interfci only allowed with ttest, ksmirnov or ranksum')
    
    if bernoulli is not None:
        randmech = 'Bernoulli'
        if np.max(bernoulli) > 1 or np.min(bernoulli) < 0:
            raise ValueError('bernoulli probabilities must be in [0,1]')
        if len(bernoulli) != len(R):
            raise ValueError('bernoulli should have the same length as the running variable')
    
    if wl is not None and wr is not None:
        wselect = 'set by user'
        if wl >= wr:
            raise ValueError('wl has to be smaller than wr')
        if wl > cutoff or wr < cutoff:
            raise ValueError('window does not include cutoff')
    
    if wl is None and wr is not None:
        raise ValueError('wl not specified')
    
    if wl is not None and wr is None:
        raise ValueError('wr not specified')
    
    if evall is not None and evalr is None:
        raise ValueError('evalr not specified')
    
    if evall is None and evalr is not None:
        raise ValueError('evall not specified')
    
    if d is not None and dscale is not None:
        raise ValueError('Cannot specify both d and dscale')

    Rc = R - cutoff
    D = np.array(Rc >= 0, dtype=float)

    n = len(D)
    n1 = np.sum(D)
    n0 = n - n1

    ###############################################################################
    # Window selection
    ###############################################################################

    if wl is None and wr is None:
        if covariates is None:
            wl = np.min(R, axis=0, initial=np.inf)
            wr = np.max(R, axis=0, initial=-np.inf)
            wselect = 'run. var. range'
        else:
            wselect = 'rdwinselect'
            if not quietly:
                print('\nRunning rdwinselect...\n')
            rdwlength = rdwinselect(Rc_long, covariates, obsmin=obsmin, obsstep=obsstep, wmin=wmin, wstep=wstep, wobs=wobs,
                                wasymmetric=wasymmetric, wmasspoints=wmasspoints, dropmissing=dropmissing, nwindows=nwindows,
                                statistic=rdwstat, p=p, vce=vce, approx=approx, reps=rdwreps, plot=plot, level=level, seed=seed, quietly=True)
            wl = cutoff + rdwlength['w_left']
            wr = cutoff + rdwlength['w_right']
            if not quietly:
                print('\nrdwinselect complete.\n')

    if not quietly:
        print(f'\nSelected window = [{round(wl, 3)};{round(wr, 3)}] \n')

    if evall is not None and evalr is not None:
        if evall < wl or evalr > wr:
            raise ValueError('evall and evalr need to be inside window')

    ww = (np.round(R, 8) >= np.round(wl, 8)) & (np.round(R, 8) <= np.round(wr, 8))

    Yw = Y[ww]
    Rw = Rc[ww]
    Dw = D[ww]

    if fuzzy is not None:
        Tw = fuzzy_tr[ww]

    if bernoulli is None:
        data = np.column_stack((Yw, Rw, Dw))
        data = data[~np.isnan(data).any(axis=1)]
        Yw = data[:, 0]
        Rw = data[:, 1]
        Dw = data[:, 2]
    else:
        Bew = bernoulli[ww]
        data = np.column_stack((Yw, Rw, Dw, Bew))
        data = data[~np.isnan(data).any(axis=1)]
        Yw = data[:, 0]
        Rw = data[:, 1]
        Dw = data[:, 2]
        Bew = data[:, 3]

    n_w = len(Dw)
    n1_w = int(np.sum(Dw))
    n0_w = n_w - n1_w

    ###############################################################################
    # Summary statistics
    ###############################################################################

    sumstats = np.zeros((5, 2))
    sumstats[0, :] = [n0, n1]
    sumstats[1, :] = [n0_w, n1_w]
    mean0 = np.mean(Yw[Dw == 0], axis=0)
    mean1 = np.mean(Yw[Dw == 1], axis=0)
    sd0 = np.nanstd(Yw[Dw == 0], axis=0, ddof =1)
    sd1 = np.nanstd(Yw[Dw == 1], axis=0, ddof =1)
    sumstats[2, :] = [mean0, mean1]
    sumstats[3, :] = [sd0, sd1]
    sumstats[4, :] = [wl, wr]

    if d is None and dscale is None:
        delta = 0.5 * sd0
    if d is not None and dscale is None:
        delta = d
    if d is None and dscale is not None:
        delta = dscale * sd0

    ###############################################################################
    # Weights
    ###############################################################################

    kweights = np.ones(n_w)

    if kernel == 'triangular':
        bwt = wr - cutoff
        bwc = wl - cutoff
        kweights[Dw == 1] = (1 - np.abs(Rw[Dw == 1] / bwt)) * (np.abs(Rw[Dw == 1] / bwt) < 1)
        kweights[Dw == 0] = (1 - np.abs(Rw[Dw == 0] / bwc)) * (np.abs(Rw[Dw == 0] / bwc) < 1)
    elif kernel == 'epan':
        bwt = wr - cutoff
        bwc = wl - cutoff
        kweights[Dw == 1] = 0.75 * (1 - (Rw[Dw == 1] / bwt) ** 2) * (np.abs(Rw[Dw == 1] / bwt) < 1)
        kweights[Dw == 0] = 0.75 * (1 - (Rw[Dw == 0] / bwc) ** 2) * (np.abs(Rw[Dw == 0] / bwc) < 1)

    ###############################################################################
    # Outcome adjustment: model and null hypothesis
    ###############################################################################

    Y_adj = Yw.copy()

    if p > 0:
        if evall is None and evalr is None:
            evall = cutoff
            evalr = cutoff
        R_adj = Rw + cutoff - Dw * evalr - (1 - Dw) * evall
        Rpoly = np.transpose(np.vstack([R_adj**k for k in range(1,p+1)]))
    
    if fuzzy is None:
        Y_adj_null = Y_adj - nulltau * Dw
    else:
        Y_adj_null = Y_adj - nulltau * Tw

    ###############################################################################
    # Observed statistics and asymptotic p-values
    ###############################################################################

    if p == 0:
        if fuzzy is None:
            results = rdrandinf_model(Y_adj_null, Dw, statistic=statistic, pvalue=True, kweights=kweights, delta=delta)
        else:
            results = rdrandinf_model(Y_adj_null, Dw, statistic=fuzzy_stat, endogtr=Tw, pvalue=True, kweights=kweights, delta=delta)

        obs_stat = results['statistic']

        if fuzzy_stat == 'wald':
            firststagereg = sm.OLS(Tw, sm.add_constant(Dw)).fit()
            aux = IV2SLS(dependent = Yw, 
                            exog = np.ones(n_w),
                            endog = Tw,
                            instruments = Dw,
                            weights = kweights).fit(cov_type = 'robust')
            obs_stat = aux.params.iloc[1]
            se = aux.std_errors.iloc[1]
            critical = 1.96 if ci is None else norm.ppf(1-ci[0]/2)
            ci_lb = obs_stat - critical * se
            ci_ub = obs_stat + critical * se
            tstat = (obs_stat-nulltau) / se
            asy_pval = 2 * norm.cdf(-np.abs(tstat))
            asy_power = 1 - norm.cdf(1.96 - delta / se) + norm.cdf(-1.96 - delta / se)
        else:
            asy_pval = results['p_value']
            asy_power = results['asy_power']
    else:
        if fuzzy_stat == 'wald':
            hc = rdlocrand_hc_fit(Yw, Dw, Rpoly, kweights, vce, Tw)
            obs_stat = hc['estimate']
            tstat = (obs_stat-nulltau)/hc['se']
            firststagereg = sm.WLS(Tw, np.column_stack((np.ones(n_w), Dw, Rpoly, Dw[:, None]*Rpoly)), weights=kweights).fit()
        else:
            Y_null = Yw-nulltau*(Dw if fuzzy is None else Tw)
            hc = rdlocrand_hc_fit(Y_null, Dw, Rpoly, kweights, vce)
            obs_stat = hc['estimate']
            tstat = obs_stat/hc['se']
        se = hc['se']
        asy_pval = 2*norm.cdf(-abs(tstat))
        asy_power = 1-norm.cdf(1.96-delta/se)+norm.cdf(-1.96-delta/se)

    ###############################################################################
    # Randomization-based inference
    ###############################################################################

    if statistic == 'all': stats_distr = np.empty((reps, 3))
    else: stats_distr  = np.empty((reps, 1))
    
    if not quietly and p == 0 and fuzzy_stat != 'wald':
        print('')
        print('Running randomization-based test...')
    
    if p == 0 and fuzzy_stat != 'wald':
        if bernoulli is None:
            max_reps = comb(n_w, n1_w)
            reps = int(min(reps, max_reps))
            if max_reps < reps:
                print(f'Chosen no. of reps > total no. of permutations.\nreps set to {reps}.')
            
            for i in range(reps):
                D_sample = np.random.choice(Dw, size = len(Dw), replace=False)
                if fuzzy is None:
                    obs_stat_sample = rdrandinf_model(Y_adj_null, D_sample, statistic, kweights=kweights, delta=delta)['statistic']
                else:
                    obs_stat_sample = rdrandinf_model(Y_adj_null, D_sample, statistic=fuzzy_stat, endogtr=Tw, kweights=kweights, delta=delta)['statistic']
                stats_distr[i,:] = obs_stat_sample
        else:
            for i in range(reps):
                D_sample = np.random.uniform(0, 1, n_w) <= Bew
                if (np.mean(D_sample) == 1) or (np.mean(D_sample) == 0):
                    stats_distr[i] = np.nan # ignore cases where bernoulli assignment mechanism gives no treated or no controls
                else:
                    obs_stat_sample = rdrandinf_model(Y_adj_null, D_sample, statistic, kweights=kweights, delta=delta)['statistic']
                    stats_distr[i, :] = obs_stat_sample

        if not quietly:
            print('Randomization-based test complete.')

        if statistic == 'all':
            p_value1 = np.mean(np.abs(stats_distr[:, 0]) >= np.abs(obs_stat[0]), axis=0)
            p_value2 = np.mean(np.abs(stats_distr[:, 1]) >= np.abs(obs_stat[1]), axis=0)
            p_value3 = np.mean(np.abs(stats_distr[:, 2]) >= np.abs(obs_stat[2]), axis=0)
            p_value = [p_value1, p_value2, p_value3]
        else:
            p_value = np.nanmean(np.abs(stats_distr) >= np.abs(obs_stat))

    else:
        p_value = np.nan

    ###############################################################################
    # Confidence interval
    ###############################################################################

    if ci is not None:
        ci_alpha = ci[0]
        if p > 0:
            if len(ci) > 1 and fuzzy_stat != 'wald':
                grid = np.unique(ci[1:])
                pv = []
                for tau in grid:
                    yy = Yw-tau*(Dw if fuzzy is None else Tw)
                    fit = rdlocrand_hc_fit(yy, Dw, Rpoly, kweights, vce)
                    pv.append(2*norm.cdf(-abs(fit['estimate']/fit['se'])))
                conf_int = find_CI(pv, ci_alpha, grid)
            elif fuzzy_stat == 'ar':
                raise ValueError('For fuzzy Anderson-Rubin confidence sets, supply the treatment-effect grid in ci.')
            else:
                estimate = obs_stat+(0 if fuzzy_stat == 'wald' else nulltau)
                conf_int = np.array([[estimate-norm.ppf(1-ci_alpha/2)*se,
                                      estimate+norm.ppf(1-ci_alpha/2)*se]])
            ci_lb, ci_ub = conf_int[0]
        elif fuzzy_stat != 'wald':
            wr_c, wl_c = wr-cutoff, wl-cutoff
            fuzzy_ci = None if fuzzy is None else fuzzy_tr
            opts = dict(p=p, wlist=[wr_c], wlist_left=[wl_c], statistic=statistic,
                        kernel=kernel, vce=vce, fuzzy=fuzzy_ci, ci=[wl_c, wr_c],
                        ci_alpha=ci_alpha, reps=reps, quietly=quietly, seed=seed)
            if len(ci) > 1:
                opts['tlist'] = ci[1:]
            conf_int = rdsensitivity_inner(Y, Rc, **opts)['ci']
        else:
            conf_int = np.array([[ci_lb, ci_ub]])
        if np.any(np.isnan(conf_int)):
            print('No grid points accepted. Consider a larger tlist in ci() option.')

    ###############################################################################
    # Confidence interval under interference
    ###############################################################################
    
    if interfci is not None:
        p_low = interfci / 2
        p_high = 1 - interfci / 2
        qq = np.quantile(stats_distr, [p_low, p_high])
        interf_ci = np.array([float(np.asarray(obs_stat).reshape(-1)[0]) - qq[1], float(np.asarray(obs_stat).reshape(-1)[0]) - qq[0]])

    ###############################################################################
    # Output and display results
    ###############################################################################

    output = {}

    if ci is None and interfci is None:
        output['sumstats'] = sumstats
        output['obs.stat'] = obs_stat
        output['p.value'] = p_value
        output['asy.pvalue'] = asy_pval
        output['window'] = [wl, wr]

    if ci is not None and interfci is None:
        output['sumstats'] = sumstats
        output['obs.stat'] = obs_stat
        output['p.value'] = p_value
        output['asy.pvalue'] = asy_pval
        output['window'] = [wl, wr]
        output['ci'] = conf_int

    if ci is None and interfci is not None:
        output['sumstats'] = sumstats
        output['obs.stat'] = obs_stat
        output['p.value'] = p_value
        output['asy.pvalue'] = asy_pval
        output['window'] = [wl, wr]
        output['interf.ci'] = interf_ci

    if ci is not None and interfci is not None:
        output['sumstats'] = sumstats
        output['obs.stat'] = obs_stat
        output['p.value'] = p_value
        output['asy.pvalue'] = asy_pval
        output['window'] = [wl, wr]
        output['ci'] = conf_int
        output['interf.ci'] = interf_ci

    if not quietly:
        if statistic == 'diffmeans' or statistic == 'ttest':
            statdisp = 'Diff. in means'
        elif statistic == 'ksmirnov':
            statdisp = 'Kolmogorov-Smirnov'
        elif statistic == 'ranksum':
            statdisp = 'Rank sum z-stat'
        elif fuzzy_stat == 'ar':
            statdisp = 'ITT'
        elif fuzzy_stat == 'wald':
            statdisp = 'TSLS'

        print('\n')
        print(f'{"Number of obs =":18}{n:14.0f}')
        print(f'{"Order of poly =":18}{p:14.0f}')
        print(f'{"Kernel type =":18}{kernel:>14}')
        if p == 0:
            print(f'{"Reps =":18}{reps:14.0f}')
        print(f'{"Window =":18}{wselect:>14}')
        print(f'{"H0:     tau  =":18}{nulltau:14.3f}')
        if p == 0:
            print(f'{"Randomization =":18}{randmech:>14}')
        print('\n')

        print(f'{"Cutoff c = ":10}{cutoff:^9.3f}{"Left of c":>12}{"Right of c":>12}')
        print(f'{"Number of obs":19}{n0:12.0f}{n1:12.0f}')
        print(f'{"Eff. number of obs":19}{n0_w:12.0f}{n1_w:12.0f}')
        print(f'{"Mean of outcome":19}{mean0:12.3f}{mean1:12.3f}')
        print(f'{"S.d. of outcome":19}{sd0:12.3f}{sd1:12.3f}')
        print(f'{"Window":19}{wl:12.3f}{wr:12.3f}')
      
        print('\n' + '=' * 80)

        if firststage and fuzzy_stat == 'wald':
            print("First stage regression")
            print(firststagereg.summary())
            print('\n' + '=' * 80 )

        if p > 0:
            print(f'Large-sample inference, {vce}')
            print(f'{"Statistic":19}{"T":>11}{"P>|T|":>12}{"Std. error":>12}')
            print(f'{statdisp:19}{obs_stat:11.3f}{asy_pval:12.3f}{se:12.3f}')
        else:
            print(f'{"":31}{"Finite sample":^20}{"Large sample":^29}')
            print(f'{"":31}{"-" * 18:18}{"":2}{"-" * 29:29}')
            print(f'{"Statistic":19}{"T":>11}{"P>|T|":^21}{"P>|T|":^9}{"Power vs d = ":>15}{delta:4.3f}')
       
            print('=' * 80)

            if statistic != 'all':
                if not np.isscalar(asy_pval): asy_pval = asy_pval[0]
                if not np.isscalar(asy_power): asy_power = asy_power[0]
                if not np.isscalar(obs_stat): obs_stat = obs_stat[0]
                print(f'{statdisp:19}{obs_stat:11.3f}{p_value:^21.3f}{asy_pval:^9.3f}{asy_power:20.3f}')

            if statistic == 'all':
                print(f"{'Diff. in means':19}{obs_stat[0]:11.3f}{p_value[0]:^21.3f}{asy_pval[0]:<9.3f}{asy_power[0]:>20.3f}")
                print(f"{'Kolmogorov-Smirnov':19}{obs_stat[1]:>11.3f}{p_value[1]:^21.3f}{asy_pval[1]:<9.3f}{asy_power[1]:>20.3f}")
                print(f"{'Rank sum z-stat':19}{obs_stat[2]:>11.3f}{p_value[2]:^21.3f}{asy_pval[2]:<9.3f}{asy_power[2]:>20.3f}")

        print('=' * 80)
        if ci is not None:
            print()
            if fuzzy_stat != 'wald':
                if len(conf_int) == 1:
                    print(f"{(1 - ci_alpha) * 100:.0f}% confidence interval: [{round(conf_int[0,0],3)},{round(conf_int[0,1],3)}]")
                else:
                    print(f"{(1 - ci_alpha) * 100:.0f}% confidence interval:")
                    print(np.round(conf_int, 3))
                    print()
                    print("Note: CI is disconnected - each row is a subset of the CI")
            else:
                print(f"{(1 - ci_alpha) * 100:.0f}% confidence interval: [{round(ci_lb, 3):.3f}, {round(ci_ub, 3):.3f}]")
                print("CI based on asymptotic approximation")

        if interfci is not None:
            print()
            print(f"{(1 - interfci) * 100:.0f}% confidence interval under interference: [{round(interf_ci[0], 3):.3f}, {round(interf_ci[1], 3):.3f}]")

    output.update(inference)
    output['inference'] = 'large-sample' if p > 0 or fuzzy_stat == 'wald' else 'randomization'
    if p > 0:
        output['se'] = se
    return output


def rdsensitivity_inner(*args, **kwargs):
    # Import at call time to avoid the public modules' import cycle.
    from rdlocrand.rdsensitivity import rdsensitivity
    return rdsensitivity(*args, nodraw=True, **kwargs)
