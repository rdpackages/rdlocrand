#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm, binomtest
import statsmodels.api as sm
import matplotlib.pyplot as plt
from rdlocrand.rdlocrand_fun import findwobs, findwobs_sym, findstep, hotelT2, rdrandinf_model

def rdwinselect(R, X=None, cutoff=0, obsmin=None, wmin=None, wobs=None, wstep=None,
                wasymmetric=False, wmasspoints=False, dropmissing=False, nwindows=10,
                statistic='diffmeans', p=0, evalat='cutoff', kernel='uniform',
                approx=False, level=0.15, reps=1000, seed=666, plot=False, quietly=False,
                obsstep=None):
    
    """
    Window selection for RD designs under local randomization

    rdwinselect implements the window-selection procedure based on balance tests for RD designs
    under local randomization. Specifically, it constructs a sequence of nested windows around
    the RD cutoff and reports binomial tests for the running variable runvar and covariate
    balance tests for covariates (if specified). The recommended window is the largest window
    around the cutoff such that the minimum p-value of the balance test is larger than a
    prespecified level for all nested (smaller) windows. By default, the p-values are calculated
    using randomization inference methods.

    Parameters:
    ----------
    R : array-like
        A vector containing the values of the running variable.
    X : array-like, optional
        The matrix of covariates to be used in the balancing tests. The matrix is optional but
        the recommended window is only provided when at least one covariate is specified.
        This should be a matrix of size n x k where n is the total sample size and k is the number
        of covariates.
    cutoff : float, optional
        The RD cutoff (default is 0).
    obsmin : int, optional
        The minimum number of observations above and below the cutoff in the smallest window.
        Default is 10.
    wmin : float, optional
        The smallest window to be used.
    wobs : int, optional
        The number of observations to be added at each side of the cutoff at each step.
        Default is 5.
    wasymmetric : bool, optional
        Allows for asymmetric windows around the cutoff when wobs is specified.
    wmasspoints : bool, optional
        Specifies that the running variable is discrete and each masspoint should be used as a window.
    wstep : float, optional
        The increment in window length.
    nwindows : int, optional
        The number of windows to be used. Default is 10.
    dropmissing : bool, optional
        Drop rows with missing values in covariates when calculating windows.
    statistic : str, optional
        The statistic to be used in the balance tests. Allowed options are 'diffmeans'
        (difference in means statistic), 'ksmirnov' (Kolmogorov-Smirnov statistic),
        'ranksum' (Wilcoxon-Mann-Whitney standardized statistic), and 'hotelling'
        (Hotelling's T-squared statistic). Default option is 'diffmeans'. The statistic
        'ttest' is equivalent to 'diffmeans' and included for backward compatibility.
    p : int, optional
        The order of the polynomial for outcome adjustment model (for covariates). Default is 0.
    evalat : str, optional
        Specifies the point at which the adjusted variable is evaluated. Allowed options are
        'cutoff' and 'means'. Default is 'cutoff'.
    kernel : str, optional
        Specifies the type of kernel to use as a weighting scheme. Allowed kernel types are
        'uniform' (uniform kernel), 'triangular' (triangular kernel), and 'epan' (Epanechnikov kernel).
        Default is 'uniform'.
    approx : bool, optional
        Forces the command to conduct the covariate balance tests using a large-sample approximation
        instead of finite-sample exact randomization inference methods.
    level : float, optional
        The minimum accepted value of the p-value from the covariate balance tests. Default is 0.15.
    reps : int, optional
        Number of replications. Default is 1000.
    seed : int, optional
        The seed to be used for the randomization tests.
    plot : bool, optional
        Draws a scatter plot of the minimum p-value from the covariate balance test against window length.
    quietly : bool, optional
        Suppress output.
    obsstep : int, optional
        The minimum number of observations to be added on each side of the cutoff for the sequence
        of fixed-increment nested windows. This option is deprecated and only included for backward
        compatibility.

    Returns:
    -------
    window : float or None
        Recommended window (None if covariates are not specified).
    wlist : list
        List of window lengths.
    results : DataFrame
        Table including window lengths, minimum p-value in each window, corresponding number of
        the variable with the minimum p-value (i.e., column of covariate matrix), Binomial test
        p-value, and sample sizes to the left and right of the cutoff in each window.
    summary : dict
        Summary statistics.

    Examples:
    ---------
    X = np.random.randn(100, 2)
    R = X[0] + X[1] + np.random.randn(100)

    # Window selection adding 5 observations at each step
    # Note: low number of replications to speed up process.
    tmp = rdwinselect(R, X, obsmin=10, wobs=5, reps=500)

    # Window selection setting initial window and step
    # The user should increase the number of replications.
    tmp = rdwinselect(R, X, wmin=0.5, wstep=0.125, reps=500)

    # Window selection with approximate (large sample) inference and p-value plot
    tmp = rdwinselect(R, X, wmin=0.5, wstep=0.125, approx=True, nwin=80, quietly=True, plot=True)
    """
    
    ###############################################################################
    # Parameters and error checking
    ###############################################################################
    
    if cutoff <= np.min(R) or cutoff >= np.max(R):
        raise ValueError('Cutoff must be within the range of the running variable')
    if p < 0:
        raise ValueError('p must be a positive integer')
    if p > 0 and approx and (statistic != 'ttest' and statistic != 'diffmeans'):
        raise ValueError('approximate and p > 1 can only be combined with diffmeans')
    valid_statistics = ['diffmeans', 'ttest', 'ksmirnov', 'ranksum', 'hotelling']
    if statistic not in valid_statistics:
        raise ValueError(f'{statistic} not a valid statistic')
    valid_evalat = ['cutoff', 'means']
    if evalat not in valid_evalat:
        raise ValueError('evalat only admits means or cutoff')
    valid_kernels = ['uniform', 'triangular', 'epan']
    if kernel not in valid_kernels:
        raise ValueError(f'{kernel} not a valid kernel')
    if kernel != 'uniform' and evalat != 'cutoff':
        raise ValueError('kernel can only be combined with evalat(cutoff)')
    if kernel != 'uniform' and statistic != 'ttest' and statistic != 'diffmeans':
        raise ValueError('kernel only allowed for diffmeans')
    if obsmin is not None and wmin is not None:
        raise ValueError('cannot set both obsmin and wmin')
    if wobs is not None and wstep is not None:
        raise ValueError('cannot set both wobs and wstep')
    if wmasspoints:
        if obsmin is not None:
            raise ValueError('obsmin not allowed with wmasspoints')
        if wmin is not None:
            raise ValueError('wmin not allowed with wmasspoints')
        if wobs is not None:
            raise ValueError('wobs not allowed with wmasspoints')
        if wstep is not None:
            raise ValueError('wstep not allowed with wmasspoints')
    
    Rc = pd.DataFrame(np.array(R - cutoff), columns =['Rc'])
    D = pd.DataFrame(1*np.array(R >= cutoff), columns =['D'])

    if X is not None:
        X = pd.DataFrame(X)
        colnames_X = X.columns.values
        data = pd.concat([Rc, D, X], axis = 1)
        if not dropmissing: data = data.dropna(subset=['Rc', 'D'])
        else: data = data.dropna()
        data = data.sort_values('Rc')
        X = data.drop(['Rc', 'D'],axis=1).values
    else:
        colnames_X = None
        data = pd.concat([Rc, D], axis = 1)
        data = data.dropna()
        data = data.sort_values('Rc')
    
    Rc = data['Rc'].values
    D = data['D'].values
    
    if seed > 0:
        np.random.seed(seed)
    elif seed != -1:
        raise ValueError('Seed has to be a positive integer or -1 for system seed')
    
    testing_method = 'rdrandinf' if not approx else 'approximate'
    
    n = len(Rc)
    n1 = np.sum(D)
    n0 = n - n1
    Rc_df = pd.DataFrame({'Rc': Rc})
    count = Rc_df.groupby('Rc').size().to_frame(name = "count")
    dups = pd.merge(Rc_df, count, on='Rc', sort = True)['count'].values
    
    if np.max(dups) > 1:
        print('Mass points detected in running variable')
        print('You may use wmasspoints option for constructing windows at each mass point')
        mp_left = np.unique(Rc[D == 0])
        mp_right = np.unique(Rc[D == 1])
        if wmasspoints:
            nmax = min(max(len(mp_left), len(mp_right)), nwindows)
            wlist = np.empty((2, nmax))
    
    ###############################################################################
    # Define initial window
    ###############################################################################
    
    if wmin is None:
        posl = n0
        posr = n0 + 1
        
        if obsmin is None:
            obsmin = 10
        if wmasspoints:
            obsmin = 1
            wasymmetric = True
        if obsstep is not None:
            wmin = findwobs_sym(obsmin, 1, posl, posr, Rc, dups)
        if wasymmetric:
            tmp = findwobs(obsmin, 1, posl, posr, Rc, dups)
            wmin_left = tmp['wlength_left']
            posmin_left = tmp['poslist_left'][0]-1
            wmin_right = tmp['wlength_right']
            posmin_right = tmp['poslist_right'][0]-1
        else:
            wmin_right = findwobs_sym(obsmin, 1, posl, posr, Rc, dups)
            wmin_left = -wmin_right
    
    else:
        if np.isscalar(wmin):
            wmin_right = [wmin]
            wmin_left = [-wmin]
            posmin_right = n0 + np.sum(np.logical_and(Rc <= wmin, Rc >= 0))
            posmin_left = n0 - np.sum(np.logical_and(Rc < 0, Rc >= -wmin)) + 1
        elif len(wmin) == 2:
            wmin_left = [wmin[0]]
            wmin_right = [wmin[1]]
            posmin_right = n0 + np.sum(np.logical_and(Rc <= wmin_right, Rc >= 0))
            posmin_left = n0 - np.sum(np.logical_and(Rc < 0, Rc >= wmin_left)) + 1
        else:
            raise ValueError('wmin option incorrectly specified')
    
    ###############################################################################
    # Define window list
    ###############################################################################
    
    if obsstep is not None:
        warnings.warn('obsstep included for backward compatibility only.\nThe use of wstep and wobs is recommended.')
        wstep = findstep(Rc, D, obsmin, obsstep, 10)
        wlist_right = np.linspace(wmin[0], wmin[0] + wstep * (nwindows - 1), num=nwindows)
        wlist_left = None
    elif wstep is not None:
        wmax_left = max(wmin_left - wstep * (nwindows - 1), min(Rc))
        wmax_right = min(wmin_right + wstep * (nwindows - 1), max(Rc))
        wlist_left = np.sort(np.arange(wmax_left, wmin_left+wstep, step=wstep))[::-1]
        wlist_right = np.arange(wmin_right, wmax_right + wstep, step=wstep)
    else:
        if wobs is None:
            wobs = 5
        if wmasspoints:
            wobs = 1
        posl = max(n0 - np.sum(np.logical_and(Rc < 0, Rc >= wmin_left)), 1)
        posr = min(n0 + 1 + np.sum(np.logical_and(Rc >= 0, Rc <= wmin_right)), n)
        if wasymmetric:
            tmp = findwobs(wobs, nwindows - 1, posl, posr, Rc, dups)
            wlist_left = np.concatenate(([wmin_left], tmp['wlist_left']))
            poslist_left = np.concatenate(([posmin_left], np.array(tmp['poslist_left'])-1))
            wlist_right = np.concatenate(([wmin_right], tmp['wlist_right']))
            poslist_right = np.concatenate(([posmin_right], np.array(tmp['poslist_right'])-1))
        else:
            wlist = findwobs_sym(wobs, nwindows - 1, posl, posr, Rc, dups)
            wlist_right = np.concatenate((wmin_right, wlist))
            wlist_left = np.concatenate((wmin_left, wlist))
    
    nmax = min(nwindows, len(wlist_right))
    if nmax < nwindows:
        print()
        warnings.warn('Not enough observations to calculate all windows. '
                      'Consider changing wmin(), wobs(), or wstep().')
        
    ###############################################################################
    # Summary statistics
    ###############################################################################

    table_sumstats = np.empty((5, 2))
    table_sumstats[0, :] = [n0, n1]

    qq0 = np.round(np.quantile(np.abs(Rc[D == 0]), q=[0.01, 0.05, 0.1, 0.2], interpolation='lower'), 5)
    qq1 = np.round(np.quantile(Rc[D == 1], q=[0.01, 0.05, 0.1, 0.2], interpolation='lower'), 5)

    n0_q1 = np.sum((Rc >= -qq0[0]) & (Rc < 0))
    n0_q2 = np.sum((Rc >= -qq0[1]) & (Rc < 0))
    n0_q3 = np.sum((Rc >= -qq0[2]) & (Rc < 0))
    n0_q4 = np.sum((Rc >= -qq0[3]) & (Rc < 0))
    n1_q1 = np.sum((Rc <= qq1[0]) & (Rc >= 0))
    n1_q2 = np.sum((Rc <= qq1[1]) & (Rc >= 0))
    n1_q3 = np.sum((Rc <= qq1[2]) & (Rc >= 0))
    n1_q4 = np.sum((Rc <= qq1[3]) & (Rc >= 0))

    table_sumstats[1, :] = [n0_q1, n1_q1]
    table_sumstats[2, :] = [n0_q2, n1_q2]
    table_sumstats[3, :] = [n0_q3, n1_q3]
    table_sumstats[4, :] = [n0_q4, n1_q4]

    ###############################################################################
    ## Display upper-right panel
    ###############################################################################

    if not quietly:
        print('\n')
        print('Window selection for RD under local randomization')
        print('\n')
        print(f"{'Number of obs':18}= {n:14}")
        print(f"{'Order of poly':18}= {p:14}")
        print(f"{'Kernel type':18}= {kernel:>14}")
        print(f"{'Reps':18}= {reps:14}")
        print(f"{'Testing method':18}= {testing_method:>14}")
        print(f"{'Balance test':18}= {statistic:>14}")
        print('\n')

    ###############################################################################
    ## Display upper left panel
    ###############################################################################
    
    if not quietly:
        print(format("Cutoff c = ", '10s'), format(f'{cutoff:.3f}', '8s'), format("Left of c", '12s'),format("Right of c", '12s'))
        print(format("Number of obs", '19s'), format(f'{n0:6.0f}', '12s'), format(f'{n1:6.0f}', '12s'))
        print(format("1st percentile", '19s'), format(f'{n0_q1:6.0f}', '12s'), format(f'{n1_q1:6.0f}', '12s'))
        print(format("5th percentile", '19s'), format(f'{n0_q2:6.0f}', '12s'), format(f'{n1_q2:6.0f}', '12s'))
        print(format("10th percentile", '19s'), format(f'{n0_q3:6.0f}', '12s'), format(f'{n1_q3:6.0f}', '12s'))
        print(format("20th percentile", '19s'), format(f'{n0_q4:6.0f}', '12s'), format(f'{n1_q4:6.0f}', '12s'))
        print('\n')

    ###############################################################################
    # Balance tests
    ###############################################################################

    table_rdw = np.full((nmax, 7),np.nan)

    ## Being main panel display

    if not quietly:
        print("=" * 80)
        print(f"{'Window':^18}{'p-value':>11}{'Var. name':>16}{'Bin.test':>12}{'Obs<c':>11}{'Obs>=c':>11}")
        print("=" * 80)

    for j in range(nmax):
        if wasymmetric and wstep is None and obsstep is None:
            wlower = wlist_left[j]
            wupper = wlist_right[j]

            position_l = poslist_left[j]
            position_r = poslist_right[j]

            ww = (Rc >= Rc[position_l]) & (Rc <= Rc[position_r])

        else:
            wupper = wlist_right[j]
            wlower = -wupper
            if wlist_left is not None: wlist_left[j] = wlower
            else: wlist_left = None

            ww = (Rc >= wlower) & (Rc <= wupper)

        Dw = D[ww]
        Rw = Rc[ww]

        # Drop NA values

        if X is not None:
            Xw = X[ww, :]
            data = np.column_stack((Rw, Dw, Xw))
            data = data[~np.isnan(data).any(axis=1)]
            Rw = data[:, 0]
            Dw = data[:, 1]
            Xw = data[:, 2:]
        else:
            data = np.column_stack((Rw, Dw))
            data = data[~np.isnan(data).any(axis=1)]
            Rw = data[:, 0]
            Dw = data[:, 1]

        # Sample sizes

        n0_w = np.sum(Dw == 0)
        n1_w = np.sum(Dw == 1)
        n_w = n0_w + n1_w
        table_rdw[j, 3] = n0_w
        table_rdw[j, 4] = n1_w

        if n0_w == 0 or n1_w == 0:
            table_rdw[j, 0] = np.nan
            table_rdw[j, 1] = np.nan
            varname = ''
        else:

            # Binomial test

            p_value = binomtest(int(np.sum(Dw)), len(Dw), p=0.5).pvalue
            table_rdw[j, 2] = p_value

            if X is not None:

                # Weights

                kweights = np.ones(n_w)

                if kernel == 'triangular':
                    kweights = (1 - np.abs(Rw / wupper)) * (np.abs(Rw / wupper) <= 1)
                    kweights[kweights == 0] = np.finfo(float).eps
                elif kernel == 'epan':
                    kweights = 0.75 * (1 - (Rw / wupper) ** 2) * (np.abs(Rw / wupper) <= 1)
                    kweights[kweights == 0] = np.finfo(float).eps

                # Model adjustment

                if p > 0:
                    X_adj = np.empty_like(Xw)

                    if evalat == 'cutoff':
                        evall = cutoff
                        evalr = cutoff
                    elif evalat == 'means':
                        evall = np.mean(Rw[Dw == 0]) + cutoff
                        evalr = np.mean(Rw[Dw == 1]) + cutoff

                    R_adj = Rw + cutoff - Dw * evalr - (1 - Dw) * evall
                    Rpoly = np.polynomial.polynomial.polyvander(R_adj, deg=p)

                    X_adj = np.zeros_like(Xw)

                    for k in range(Xw.shape[1]):
                        lfit_t = sm.WLS(Xw[Dw == 1, k], sm.add_constant(Rpoly[Dw == 1]), weights=kweights[Dw == 1]).fit()
                        X_adj[Dw == 1, k] = lfit_t.resid + lfit_t.params[0]

                        lfit_c = sm.WLS(Xw[Dw == 0, k], sm.add_constant(Rpoly[Dw == 0]), weights=kweights[Dw == 0]).fit()
                        X_adj[Dw == 0, k] = lfit_c.resid + lfit_c.params[0]

                        Xw = X_adj

                # Statistics and p-values
                if statistic == 'hotelling':
                    obs_stat = hotelT2(Xw, Dw)['statistic']
                    if not approx:
                        stat_distr = np.empty(reps)
                        for i in range(reps):
                            D_sample = np.random.choice(Dw, replace=False)
                            obs_stat_sample = hotelT2(Xw, D_sample).statistic
                            stat_distr[i] = obs_stat_sample
                        p_value = np.mean(np.abs(stat_distr) >= np.abs(obs_stat))
                    else:
                        p_value = hotelT2(Xw, Dw)['p.value']
                    table_rdw[j, 0] = p_value
                    varname = np.nan
                else:
                    result = rdrandinf_model(Xw, Dw, statistic=statistic, kweights=kweights, pvalue=True)
                    obs_stat = result['statistic']
                    if not approx:
                        stat_distr = np.empty((reps, X.shape[1]))
                        for i in range(reps):
                            D_sample = np.random.choice(Dw, size = len(Dw), replace=False)
                            obs_stat_sample = rdrandinf_model(Xw, D_sample, statistic=statistic, kweights=kweights)['statistic']
                            stat_distr[i, :] = obs_stat_sample
                        p_value = np.mean(np.abs(stat_distr) >= np.abs(obs_stat), axis=0)
                    else:
                        if p == 0:
                            p_value = result['p_value']
                        else:
                            p_value = np.zeros(X.shape[1])
                            for k in range(X.shape[1]):
                                lfit = sm.WLS(Xw[:, k], sm.add_constant(np.column_stack((Dw, Rpoly, Dw * Rpoly))), weights=kweights).fit()
                                tstat = lfit.params[1] / np.sqrt(lfit.cov_HC2.loc[1, 1])
                                p_value[k] = 2 * norm.cdf(-np.abs(tstat))

                    table_rdw[j, 0] = np.min(p_value)
                    tmp = np.argmin(p_value)
                    table_rdw[j, 1] = tmp
                    if colnames_X[tmp] is not None:
                        if colnames_X[tmp] != '':
                            varname = colnames_X[tmp]
                        else:
                            varname = tmp
                    else:
                        varname = tmp
            else:
                table_rdw[j, 0] = np.nan
                table_rdw[j, 1] = np.nan
                varname = np.nan

        table_rdw[j, 5] = wlower
        table_rdw[j, 6] = wupper

        if not quietly:
            print(f"{wlower+cutoff:9.4f}{wupper+cutoff:9.4f}{table_rdw[j, 0]:11.3f}{str(varname)[:15]:>16}{table_rdw[j, 2]:12.3f}{table_rdw[j, 3]:11.0f}{table_rdw[j, 4]:11.0f}")

    if not quietly: print('=' * 80)

###############################################################################
# Find recommended window
###############################################################################

    if X is not None:
        Pvals = table_rdw[:, 0]

        if (not np.isnan(Pvals[0])) and (Pvals[0] < level):
            print('Smallest window does not pass covariate test.')
            print('Decrease smallest window or reduce level.')
            tmp = -1
            rec_length = np.full((2,),np.nan)
            rec_window = np.full((2,),np.nan)
        elif np.all(Pvals >= level):
            tmp = len(Pvals)-1
            rec_window = [cutoff + table_rdw[tmp, 5], cutoff + table_rdw[tmp, 6]]
        else:
            tmp = np.min(np.where(Pvals < level))
            tmp -= 1
            rec_window = [cutoff + table_rdw[tmp, 5], cutoff + table_rdw[tmp, 6]]

        if (not quietly) and (tmp != -1):
            print(f"Recommended window is [{round(rec_window[0], 4)};{round(rec_window[1], 4)}] with "
                f"{table_rdw[tmp, 3] + table_rdw[tmp, 4]:.0f} observations "
                f"({table_rdw[tmp, 3]:.0f} below, {table_rdw[tmp, 4]:.0f} above).")
            print('\n\n')
    else:
        if not quietly:
            print('Note: no covariates specified.')
            print('Need to specify covariates to find recommended length.')
        rec_window = np.full((2,),np.nan)

    ###############################################################################
    # Plot p-values
    ###############################################################################

    if plot:
        if 'X' in locals():
            rdwinselect_plot = plt.figure().gca()
            rdwinselect_plot.set_xlabel('Windows Right')
            rdwinselect_plot.set_ylabel('p-values')
            rdwinselect_plot.scatter(wlist_right, Pvals)
            plt.show()
        else:
            raise ValueError('Cannot draw plot without covariates')

    ###############################################################################
    # Output
    ###############################################################################

    table_sumstats_index = ['Number of obs', '1th percentile', '5th percentile', '10th percentile', '20th percentile']
    table_sumstats_columns = ['Left of c', 'Right of c']
    table_sumstats = pd.DataFrame(table_sumstats, 
                         index = table_sumstats_index, 
                         columns = table_sumstats_columns)
    
    table_rdw_columns = ['p-value', 'Variable', 'Bi.test', 'Obs<c', 'Obs>=c', 'w_left', 'w_right']
    table_rdw = pd.DataFrame(table_rdw, 
                         columns = table_rdw_columns)
    output = {
        'w_left': rec_window[0],
        'w_right': rec_window[1],
        'wlist_left': wlist_left,
        'wlist_right': wlist_right,
        'results': table_rdw,
        'summary': table_sumstats
    }

    return output