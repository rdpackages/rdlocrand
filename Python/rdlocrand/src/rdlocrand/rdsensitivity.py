#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.api as sm
from linearmodels import IV2SLS
import matplotlib.pyplot as plt
from rdlocrand.rdwinselect import rdwinselect
from rdlocrand.rdrandinf import rdrandinf
from rdlocrand.rdlocrand_fun import find_CI

def rdsensitivity(Y, R, cutoff=0, wlist=None, wlist_left=None,
                   tlist=None, statistic='diffmeans', p=0,
                    evalat='cutoff', kernel='uniform', fuzzy=None,
                    ci=None, ci_alpha=0.05, reps=1000, seed=666, 
                    nodraw=False, quietly=False):
    
    """
    Sensitivity analysis for RD designs under local randomization

    rdsensitivity analyzes the sensitivity of randomization p-values
    and confidence intervals to different window lengths.

    Author:
    Matias Cattaneo, Princeton University. Email: cattaneo@princeton.edu
    Rocio Titiunik, Princeton University. Email: titiunik@princeton.edu
    Gonzalo Vazquez-Bare, UC Santa Barbara. Email: gvazquez@econ.ucsb.edu

    References:
    Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2016).
    Inference in Regression Discontinuity Designs under Local Randomization.
    Stata Journal 16(2): 331-367.
    URL: https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf

    Parameters:
    - Y: A vector containing the values of the outcome variable.
    - R: A vector containing the values of the running variable.
    - cutoff: The RD cutoff (default is 0).
    - wlist: The list of windows to the right of the cutoff. By default, the program constructs 10 windows around the cutoff with 5 observations each.
    - wlist_left: The list of windows to the left of the cutoff. If not specified, the windows are constructed symmetrically around the cutoff based on the values in wlist.
    - tlist: The list of values of the treatment effect under the null to be evaluated. By default, the program employs ten evenly spaced points within the asymptotic confidence interval for a constant treatment effect in the smallest window to be used.
    - statistic: The statistic to be used in the balance tests. Allowed options are diffmeans (difference in means statistic), ksmirnov (Kolmogorov-Smirnov statistic), and ranksum (Wilcoxon-Mann-Whitney standardized statistic). Default option is diffmeans. The statistic ttest is equivalent to diffmeans and included for backward compatibility.
    - p: The order of the polynomial for the outcome adjustment model. Default is 0.
    - evalat: Specifies the point at which the adjusted variable is evaluated. Allowed options are cutoff and means. Default is cutoff.
    - kernel: Specifies the type of kernel to use as a weighting scheme. Allowed kernel types are uniform (uniform kernel), triangular (triangular kernel), and epan (Epanechnikov kernel). Default is uniform.
    - fuzzy: Indicates that the RD design is fuzzy. fuzzy can be specified as a vector containing the values of the endogenous treatment variable or as a list where the first element is the vector of endogenous treatment values and the second element is a string containing the name of the statistic to be used. Allowed statistics are ar (Anderson-Rubin statistic) and tsls (2SLS statistic). The default statistic is ar. The tsls statistic relies on a large-sample approximation.
    - ci: Returns the confidence interval corresponding to the indicated window length. ci has to be a two-dimensional vector indicating the left and right limits of the window. The default alpha is 0.05 (95% level CI).
    - ci_alpha: Specifies the value of alpha for the confidence interval. The default alpha is 0.05 (95% level CI).
    - reps: Number of replications. Default is 1000.
    - seed: The seed to be used for the randomization tests.
    - nodraw: Suppresses contour plot.
    - quietly: Suppresses the output table.

    Returns:
    - tlist: Treatment effects grid
    - wlist: Window grid
    - results: Table with corresponding p-values for each window and treatment effect pair.
    - ci: Confidence interval (if ci is specified).

    Examples:
    # Toy dataset
    R = np.random.uniform(-1, 1, size=100)
    Y = 1 + R - 0.5 * R**2 + 0.3 * R**3 + (R >= 0) + np.random.normal(size=100)
    # Sensitivity analysis
    # Note: low number of replications to speed up the process.
    # The user should increase the number of replications.
    tmp = rdsensitivity(Y, R, wlist=np.arange(0.75, 2.25, 0.25), tlist=np.arange(0, 5.5, 1), reps=500)
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
        if wlist is None:
            raise ValueError('Need to specify wlist when wlist_left is specified')
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
        if fuzzy is None:
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

    ###############################################################################
    # Plot
    ###############################################################################

    if not nodraw:
        if results.shape[1] == 1:
            print('Need a window grid to draw plot')
        elif results.shape[0] == 1:
            print('Need a tau grid to draw plot')
        else:
            X, Y = np.meshgrid(wlist, tlist)
            plt.contourf(X, Y, results, levels=np.arange(0, 1.01, 0.01), cmap='gray')
            plt.xlabel('window')
            plt.ylabel('treatment effect')
            plt.colorbar(label='p-value')
            plt.title('Sensitivity Analysis')
            plt.show()

    return output