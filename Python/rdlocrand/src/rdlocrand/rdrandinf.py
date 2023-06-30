#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from linearmodels import IV2SLS
from rdlocrand.rdwinselect import rdwinselect
from rdlocrand.rdlocrand_fun import rdrandinf_model, find_CI

def rdrandinf(Y, R, cutoff=0, wl=None, wr=None, statistic='diffmeans', p=0, evall=None, evalr=None, kernel='uniform',
              fuzzy=None, nulltau=0, d=None, dscale=None, ci=None, interfci=None, bernoulli=None, reps=1000, seed=666,
              quietly=False, covariates=None, obsmin=None, wmin=None, wobs=None, wstep=None, wasymmetric=False,
              wmasspoints=False, nwindows=10, dropmissing=False, rdwstat='diffmeans', approx=False, rdwreps=1000,
              level=0.15, plot=False, firststage=False, obsstep=None):
    
    """
    Randomization Inference for RD Designs under Local Randomization

    rdrandinf implements randomization inference and related methods for RD designs,
    using observations in a specified or data-driven selected window around the cutoff where
    local randomization is assumed to hold.

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
        The statistic to be used in the balance tests. Allowed options are 'diffmeans' (difference in means statistic),
        'ksmirnov' (Kolmogorov-Smirnov statistic), and 'ranksum' (Wilcoxon-Mann-Whitney standardized statistic).
        Default option is 'diffmeans'. The statistic 'ttest' is equivalent to 'diffmeans' and included for backward compatibility.
    p : int, optional
        The order of the polynomial for the outcome transformation model (default is 0).
    evall : float, optional
        The point at the left of the cutoff at which to evaluate the transformed outcome. Default is the cutoff value.
    evalr : float, optional
        The point at the right of the cutoff at which the transformed outcome is evaluated. Default is the cutoff value.
    kernel : str, optional
        Specifies the type of kernel to use as a weighting scheme. Allowed kernel types are 'uniform' (uniform kernel),
        'triangular' (triangular kernel), and 'epan' (Epanechnikov kernel). Default is 'uniform'.
    fuzzy : None or tuple or array-like, optional
        Indicates that the RD design is fuzzy. If fuzzy is None, the RD design is not fuzzy. If fuzzy is a tuple or array-like,
        the first element should be the vector of endogenous treatment values, and the second element should be a string
        containing the name of the statistic to be used. Allowed statistics are 'itt' (intention-to-treat statistic)
        and 'tsls' (2SLS statistic). Default statistic is 'ar'. The 'tsls' statistic relies on large-sample approximation.
    nulltau : float, optional
        The value of the treatment effect under the null hypothesis (default is 0).
    d : float, optional
        The effect size for asymptotic power calculation. Default is 0.5 * standard deviation of outcome variable for the control group.
    dscale : float, optional
        The fraction of the standard deviation of the outcome variable for the control group used as an alternative hypothesis
        for asymptotic power calculation. Default is 0.5.
    ci : float or array-like, optional
        Calculates a confidence interval for the treatment effect by test inversion. ci can be specified as a scalar or a vector,
        where the first element indicates the value of alpha for the confidence interval (typically 0.05 or 0.01),
        and the remaining elements, if specified, indicate the grid of treatment effects to be evaluated.
        This option uses rdsensitivity to calculate the confidence interval. See corresponding help for details.
        Note: the default tlist can be narrow in some cases, which may truncate the confidence interval.
        We recommend the user to manually set a large enough tlist.
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
        The minimum number of observations above and below the cutoff in the smallest window employed by the companion command rdwinselect.
        Default is 10.
    wmin : float, optional
        The smallest window to be used (if obsmin is not specified) by the companion command rdwinselect.
        Specifying both wmin and obsmin returns an error.
    wobs : int, optional
        The number of observations to be added at each side of the cutoff at each step.
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
        The statistic to be used by the companion command rdwinselect (see corresponding help for options).
        Default option is 'ttest'.
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
    
    Returns
    -------
    sumstats : DataFrame
        Summary statistics.
    obs_stat : float or array-like
        Observed statistic(s).
    p_value : float or array-like
        Randomization p-value(s).
    asy_pvalue : float or array-like
        Asymptotic p-value(s).
    window : tuple
        Chosen window.
    ci : DataFrame, optional
        Confidence interval (only if ci option is specified).
    interf_ci : DataFrame, optional
        Confidence interval under interference (only if interfci is specified).

    Example
    ------- 

    X = array(rnorm(200),dim=c(100,2))
    R = X[1,] + X[2,] + rnorm(100)
    Y = 1 + R -.5*R^2 + .3*R^3 + (R>=0) + rnorm(100)

    # Randomization inference in window (-.75,.75)
    tmp = rdrandinf(Y, R, wl=-.75, wr=.75)

    # Randomization inference in window (-.75,.75), all statistics
    tmp = rdrandinf(Y, R, wl=-.75, wr=.75, statistic='all')

    # Randomization inference with window selection
    # Note: low number of replications to speed up the process.
    # The user should increase the number of replications.
    tmp = rdrandinf(Y, R, statistic='all', covariates=X, wmin=.5, wstep=.125, rdwreps=500)
    """

    randmech = 'fixed margins'
    Rc_long = R - cutoff

    if (fuzzy is not None) and (fuzzy !=''):
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

    if (fuzzy is None) or (fuzzy ==''):
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
    
    if kernel != 'uniform' and statistic != 'ttest' and statistic != 'diffmeans':
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

    if seed > 0:
        np.random.seed(seed)
    elif seed != -1:
        raise ValueError('Seed has to be a positive integer or -1 for system seed')
    
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
                                statistic=rdwstat, approx=approx, reps=rdwreps, plot=plot, level=level, seed=seed, quietly=True)
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

    if (fuzzy is not None) and (fuzzy != ''):
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
        lfit_t = sm.WLS(Yw[Dw == 1], sm.add_constant(Rpoly[Dw == 1,:]), weights=kweights[Dw == 1]).fit()
        Y_adj[Dw == 1] = lfit_t.resid + lfit_t.params[0]
        lfit_c = sm.WLS(Yw[Dw == 0], sm.add_constant(Rpoly[Dw == 0,:]), weights=kweights[Dw == 0]).fit()
        Y_adj[Dw == 0] = lfit_c.resid + lfit_c.params[0]
    
    if (fuzzy is None) or (fuzzy == ''):
        Y_adj_null = Y_adj - nulltau * Dw
    else:
        Y_adj_null = Y_adj - nulltau * Tw

    ###############################################################################
    # Observed statistics and asymptotic p-values
    ###############################################################################

    if (fuzzy is None) or (fuzzy == ''):
        results = rdrandinf_model(Y_adj_null, Dw, statistic=statistic, pvalue=True, kweights=kweights, delta=delta)
    else:
        results = rdrandinf_model(Y_adj_null, Dw, statistic=fuzzy_stat, endogtr=Tw, pvalue=True, kweights=kweights, delta=delta)
    
    obs_stat = results['statistic']
    
    if p == 0:
        if fuzzy_stat == 'wald':
            firststagereg = sm.OLS(Tw, sm.add_constant(Dw)).fit()
            aux = IV2SLS(dependent = Yw, 
                            exog = None,
                            endog = sm.add_constant(Tw),
                            instruments = sm.add_constant(Dw),
                            weights = kweights).fit(cov_type = 'robust')
            obs_stat = aux.params[1]
            se = aux.std_errors[1]
            ci_lb = obs_stat - 1.96 * se
            ci_ub = obs_stat + 1.96 * se
            tstat = obs_stat / se
            asy_pval = 2 * norm.cdf(-np.abs(tstat))
            asy_power = 1 - norm.cdf(1.96 - delta / se) + norm.cdf(-1.96 - delta / se)
        else:
            asy_pval = results['p_value']
            asy_power = results['asy_power']
    else:
        if statistic == 'diffmeans' or statistic == 'ttest' or statistic == 'all':
            X_inter = sm.add_constant(np.column_stack((Dw.reshape(-1,1),Rpoly,Dw.reshape(-1,1)*Rpoly)))
            lfit = sm.WLS(Yw, X_inter, weights=kweights).fit()
            se = lfit.HC2_se[1]
            tstat = lfit.params[1] / se
            asy_pval = 2 * norm.cdf(-np.abs(tstat))
            asy_power = 1 - norm.cdf(1.96 - delta / se) + norm.cdf(-1.96 - delta / se)
        if statistic == 'ksmirnov' or statistic == 'ranksum':
            asy_pval = np.nan
            asy_power = np.nan
        if statistic == 'all':
            asy_pval = [float(asy_pval), np.nan, np.nan]
            asy_power = [float(asy_power), np.nan, np.nan]
        
        if fuzzy_stat == 'wald':
            inter = Rpoly * Dw
            firststagereg = sm.OLS(Tw, sm.add_constant(Dw)).fit()
            aux = IV2SLS(dependent = Yw, 
                            exog = sm.add_constant([Rpoly, inter]),
                            endog = Tw,
                            instruments = Dw,
                            weights = kweights).fit(cov_type = 'robust')
            obs_stat = aux.params[-1]
            se = aux.std_errors[-1]
            ci_lb = obs_stat - 1.96 * se
            ci_ub = obs_stat + 1.96 * se
            tstat = aux.params['Tw'] / se
            asy_pval = 2 * norm.cdf(-np.abs(tstat))
            asy_power = 1 - norm.cdf(1.96 - delta / se) + norm.cdf(-1.96 - delta / se)

    ###############################################################################
    # Randomization-based inference
    ###############################################################################

    if statistic == 'all': stats_distr = np.empty((reps, 3))
    else: stats_distr  = np.empty((reps, 1))
    
    if not quietly:
        print('')
        print('Running randomization-based test...')
    
    if fuzzy_stat != 'wald':
        if bernoulli is None:
            max_reps = np.math.comb(n_w, n1_w)
            reps = min(reps, max_reps)
            if max_reps < reps:
                print(f'Chosen no. of reps > total no. of permutations.\nreps set to {reps}.')
            
            for i in range(reps):
                D_sample = np.random.choice(Dw, size = len(Dw), replace=False)
                if (fuzzy is None) or (fuzzy ==''):
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
                    obs_stat_sample = float(rdrandinf_model(Y_adj_null, D_sample, statistic, kweights=kweights, delta=delta)['statistic'])
                    stats_distr[i] = obs_stat_sample

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
        if fuzzy_stat != 'wald':
            wr_c = wr - cutoff
            wl_c = wl - cutoff
            if not np.isscalar(ci):
                t_list = ci[1:]
                aux = rdsensitivity_inner(Y, Rc, p=p, wlist=wr_c, wlist_left=wl_c, tlist=t_list, fuzzy=fuzzy_stat, ci=[wl_c, wr_c], ci_alpha=ci_alpha, reps=reps, quietly=quietly, seed=seed)
            else:
                aux = rdsensitivity_inner(Y, Rc, p=p, wlist=wr_c, wlist_left=wl_c, fuzzy=fuzzy_stat, ci=[wl_c, wr_c], ci_alpha=ci_alpha, reps=reps, quietly=quietly, seed=seed)
            conf_int = aux['ci']
        else:
            conf_int = np.array([[ci_lb, ci_ub]])
        if np.any(np.isnan(conf_int)):
            print('Consider a larger tlist in ci() option.')

    ###############################################################################
    # Confidence interval under interference
    ###############################################################################
    
    if interfci is not None:
        p_low = interfci / 2
        p_high = 1 - interfci / 2
        qq = np.quantile(stats_distr, [p_low, p_high])
        interf_ci = np.array([obs_stat[0] - qq[1], obs_stat[0] - qq[0]])

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
        print(f'{"Reps =":18}{reps:14.0f}')
        print(f'{"Window =":18}{wselect:>14}')
        print(f'{"H0:     tau  =":18}{nulltau:14.3f}')
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

    return output










def rdsensitivity_inner(Y, R, cutoff=0, wlist=None, wlist_left=None,
                   tlist=None, statistic='diffmeans', p=0,
                    evalat='cutoff', kernel='uniform', fuzzy=None,
                    ci=None, ci_alpha=0.05, reps=1000, seed=666, quietly=False):
    
    """
    This function is a copy of rdsensitivity to be called inside inside rdlocrand
    and avoid the circular reference when imporing the modules
    """
    
    ###############################################################################
    # Parameters and error checking
    ###############################################################################
    
    if cutoff < np.min(R) or cutoff > np.max(R):
        raise ValueError('Cutoff must be within the range of the running variable')
    if statistic not in ['diffmeans', 'ttest', 'ksmirnov', 'ranksum']:
        raise ValueError(statistic + ' not a valid statistic')
    if evalat not in ['cutoff', 'means']:
        raise ValueError('evalat only admits means or cutoff')
    if wlist_left is not None:
        if np.isscalar(wlist_left): wlist_left = np.array([wlist_left])
        if wlist is None:
            raise ValueError('Need to specify wlist when wlist_left is specified')
        elif np.isscalar(wlist): wlist= np.array([wlist])
        if len(wlist) != len(wlist_left):
            raise ValueError('Lengths of wlist and wlist_left need to coincide')
    if ci is not None and len(ci) != 2:
        raise ValueError('Need to specify wleft and wright in CI option')

    if seed > 0:
        np.random.seed(seed)
    elif seed != -1:
        raise ValueError('Seed has to be a positive integer or -1 for system seed')

    data = np.column_stack((Y, R))
    data = data[~np.isnan(data).any(axis=1)]
    Y = data[:, 0]
    R = data[:, 1]

    Rc = R - cutoff

    ###############################################################################
    # Default window list
    ###############################################################################

    if wlist is None:
        aux = rdwinselect(Rc, wobs=5, quietly=True)
        wlist = aux['results'][:, 6]
        wlist_left = aux['results'][:, 5]
    else:
        wlist_orig = wlist
        wlist = wlist - cutoff
        if wlist_left is None:
            wlist_left = -wlist
            wlist_left_orig = wlist_left
        else:
            wlist_left_orig = wlist_left
            wlist_left = wlist_left - cutoff

    wnum = len(wlist)
    

    ###############################################################################
    # Default tau list
    ###############################################################################

    if tlist is None:
        D = (Rc >= 0).astype(int)
        wfirst = max(wlist[0], abs(wlist_left[0]))
        if (fuzzy is None) or (fuzzy==''):
            Yaux = Y[np.abs(Rc) <= wfirst]
            Daux = D[np.abs(Rc) <= wfirst]
            model = sm.OLS(Yaux, sm.add_constant(Daux))
            results = model.fit()
            ci_lb = round(results.params[1] - 1.96 * np.sqrt(results.cov_params()[1, 1]), 2)
            ci_ub = round(results.params[1] + 1.96 * np.sqrt(results.cov_params()[1, 1]), 2)
        else:
            Yaux = Y[np.abs(Rc) <= wfirst]
            Daux = D[np.abs(Rc) <= wfirst]
            Taux = fuzzy[np.abs(Rc) <= wfirst]
            model = IV2SLS(dependent = Yaux, 
                            exog = None,
                            endog = sm.add_constant(Taux),
                            instruments = sm.add_constant(Daux))
            instrument_results = model.fit(cov_type = 'robust')
            ci_lb = round(instrument_results.params[1] - 1.96 * aux.std_errors[1], 2)
            ci_ub = round(instrument_results.params[1] + 1.96 * aux.std_errors[1], 2)

        wstep = round((ci_ub - ci_lb) / 10, 2)
        tlist = np.arange(ci_lb, ci_ub + wstep, wstep)

    ###############################################################################
    # Sensitivity analysis
    ###############################################################################

    if np.isscalar(tlist): tlist = np.array([tlist])

    results = np.empty((len(tlist), len(wlist)))
    if not quietly:
        print('')
        print('Running sensitivity analysis...', end="")

    for row, t in enumerate(tlist):
        for w in range(wnum):
            wright = wlist[w]
            wleft = wlist_left[w]
            if evalat == 'means':
                ww = (np.round(Rc, 8) >= np.round(wleft, 8)) & (np.round(Rc, 8) <= np.round(wright, 8))
                Rw = R[ww]
                Dw = D[ww]
                evall = np.mean(Rw[Dw == 0])
                evalr = np.mean(Rw[Dw == 1])
            else:
                evall = None
                evalr = None

            aux = rdrandinf(Y, Rc, wl=wleft, wr=wright, p=p, reps=reps, nulltau=t,
                               statistic=statistic, kernel=kernel, evall=evall, evalr=evalr,
                               fuzzy=fuzzy, seed=seed, quietly=True)
            results[row, w] = aux['p.value']

    if not quietly:
        print('Sensitivity analysis complete.\n')

    ###############################################################################
    # Confidence interval
    ###############################################################################

    conf_int = None
    if ci is not None:
        ci_window_l = ci[0] - cutoff
        ci_window_r = ci[1] - cutoff

        if np.isin(ci_window_r, wlist) and np.isin(ci_window_l, wlist_left):
            col = np.where(wlist == ci_window_r)[0][0]
            aux = results[:, col]

            conf_int = find_CI(aux, ci_alpha, tlist)
        else:
            raise ValueError('Window specified in ci not in wlist')
        
    ###############################################################################
    # Output
    ###############################################################################

    output = {'tlist': tlist, 'wlist': wlist_orig, 'wlist_left': wlist_left_orig, 'results': results}
    
    if conf_int is not None:
        output['ci'] = conf_int

    return output