#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from rdlocrand.rdwinselect import rdwinselect
from rdlocrand.rdrandinf import rdrandinf

def rdrbounds(Y, R, cutoff=0, wlist=None, gamma=None, expgamma=None,
              bound='both', statistic='ranksum', p=0, evalat='cutoff',
              kernel='uniform', fuzzy=None, nulltau=0, prob=None,
              fmpval=False, reps=1000, seed=666):

    """
    Rosenbaum bounds for RD designs under local randomization

    rdrbounds calculates lower and upper bounds for the randomization p-value under different degrees of departure from a local randomized experiment, as suggested by Rosenbaum (2002).

    Author:
    Matias Cattaneo, Princeton University. Email: cattaneo@princeton.edu
    Rocio Titiunik, Princeton University. Email: titiunik@princeton.edu
    Gonzalo Vazquez-Bare, UC Santa Barbara. Email: gvazquez@econ.ucsb.edu

    References:
    Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2016).
    Inference in Regression Discontinuity Designs under Local Randomization.
    Stata Journal 16(2): 331-367.
    URL: https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf

    Rosenbaum, P. (2002). Observational Studies. Springer.

    Parameters:
    - Y: A vector containing the values of the outcome variable.
    - R: A vector containing the values of the running variable.
    - cutoff: The RD cutoff (default is 0).
    - wlist: The list of window lengths to be evaluated. By default, the program constructs 10 windows around the cutoff, the first one including 10 treated and control observations and adding 5 observations to each group in subsequent windows.
    - gamma: The list of values of gamma to be evaluated.
    - expgamma: The list of values of exp(gamma) to be evaluated. Default is [1.5, 2, 2.5, 3].
    - bound: Specifies which bounds the command calculates. Options are 'upper' for upper bound, 'lower' for lower bound, and 'both' for both upper and lower bounds. Default is 'both'.
    - statistic: The statistic to be used in the balance tests. Allowed options are 'diffmeans' (difference in means statistic), 'ksmirnov' (Kolmogorov-Smirnov statistic), and 'ranksum' (Wilcoxon-Mann-Whitney standardized statistic). Default option is 'diffmeans'. The statistic 'ttest' is equivalent to 'diffmeans' and included for backward compatibility.
    - p: The order of the polynomial for the outcome adjustment model. Default is 0.
    - evalat: Specifies the point at which the adjusted variable is evaluated. Allowed options are 'cutoff' and 'means'. Default is 'cutoff'.
    - kernel: Specifies the type of kernel to use as a weighting scheme. Allowed kernel types are 'uniform' (uniform kernel), 'triangular' (triangular kernel), and 'epan' (Epanechnikov kernel). Default is 'uniform'.
    - fuzzy: Indicates that the RD design is fuzzy. 'fuzzy' can be specified as a vector containing the values of the endogenous treatment variable or as a list where the first element is the vector of endogenous treatment values and the second element is a string containing the name of the statistic to be used. Allowed statistics are 'ar' (Anderson-Rubin statistic) and 'tsls' (2SLS statistic). Default statistic is 'ar'. The 'tsls' statistic relies on a large-sample approximation.
    - nulltau: The value of the treatment effect under the null hypothesis. Default is 0.
    - prob: The probabilities of treatment for each unit when the assignment mechanism is a Bernoulli trial. This option should be specified as a vector of length equal to the length of the outcome and running variables.
    - fmpval: Reports the p-value under fixed margins randomization, in addition to the p-value under Bernoulli trials.
    - reps: Number of replications. Default is 1000.
    - seed: The seed to be used for the randomization tests.

    Returns:
    - gamma: List of gamma values.
    - expgamma: List of exp(gamma) values.
    - wlist: Window grid.
    - p_values: p-values for each window (under gamma = 0).
    - lower_bound: List of lower bound p-values for each window and gamma pair.
    - upper_bound: List of upper bound p-values for each window and gamma pair.

    Examples:
    # Toy dataset
    import numpy as np

    R = np.random.uniform(-1, 1, size=100)
    Y = 1 + R - 0.5 * R**2 + 0.3 * R**3 + (R >= 0) + np.random.normal(size=100)

    # Rosenbaum bounds
    # Note: low number of replications and windows to speed up the process.
    # The user should increase these values.
    rdrbounds(Y, R, expgamma=[1.5, 2], wlist=[0.3], reps=100)

    """

    ###############################################################################
    # Parameters and error checking
    ###############################################################################

    if cutoff <= np.nanmin(R) or cutoff >= np.nanmax(R):
        raise ValueError('Cutoff must be within the range of the running variable')

    if bound != 'both' and bound != 'upper' and bound != 'lower':
        raise ValueError('bound option incorrectly specified')

    data = np.column_stack((Y, R))
    data = data[~np.isnan(data).any(axis=1)]
    Y = data[:, 0]
    R = data[:, 1]

    Rc = R - cutoff
    D = (Rc >= 0).astype(int)

    if gamma is None and expgamma is None:
        gammalist = [1.5, 2, 2.5, 3]
    elif gamma is None and expgamma is not None:
        gammalist = expgamma
    elif gamma is not None and expgamma is None:
        gammalist = np.exp(gamma)
    else:
        raise ValueError('gamma and expgamma cannot be specified simultaneously')

    if wlist is None:
        aux = rdwinselect(Rc, wobs=5, nwindows=5, quietly=True)
        wlist = np.round(aux['results'][:, 0], 2)

    if seed > 0:
        np.random.seed(seed)
    elif seed != -1:
        raise ValueError('Seed has to be a positive integer or -1 for system seed')

    evall = cutoff
    evalr = cutoff

    ###############################################################################
    # Randomization p-value
    ###############################################################################

    print('\nCalculating randomization p-value...\n')

    P = np.zeros((2, len(wlist)))

    count = 0

    if not fmpval:
        for w in wlist:
            ww = (np.round(Rc, 8) >= np.round(-w, 8)) & (np.round(Rc, 8) <= np.round(w, 8))
            Dw = D[ww]
            Rw = Rc[ww]

            if prob is None:
                prob_be = np.repeat(np.mean(Dw), len(R))
            else:
                prob_be = prob

            if evalat == 'means':
                evall = np.mean(Rw[Dw == 0])
                evalr = np.mean(Rw[Dw == 1])

            aux = rdrandinf(Y, Rc, wl=-w, wr=w, bernoulli=prob_be, reps=reps, p=p,
                            nulltau=nulltau, statistic=statistic,
                            evall=evall, evalr=evalr, kernel=kernel, fuzzy=fuzzy,
                            quietly=True)

            P[0, count] = aux['p.value']

            print('Bernoulli p-value (w = {}) = {}'.format(w, round(P[0, count], 3)))

            count += 1
    else:
        for w in wlist:
            ww = (np.round(Rc, 8) >= np.round(-w, 8)) & (np.round(Rc, 8) <= np.round(w, 8))
            Dw = D[ww]
            Rw = Rc[ww]

            if prob is None:
                prob_be = np.repeat(np.mean(Dw), len(R))
            else:
                prob_be = prob

            if evalat == 'means':
                evall = np.mean(Rw[Dw == 0])
                evalr = np.mean(Rw[Dw == 1])

            aux_be = rdrandinf(Y, Rc, wl=-w, wr=w, bernoulli=prob_be, reps=reps, p=p,
                               nulltau=nulltau, statistic=statistic,
                               evall=evall, evalr=evalr, kernel=kernel, fuzzy=fuzzy,
                               quietly=True)

            P[0, count] = aux_be['p.value']

            aux_fm = rdrandinf(Y, Rc, wl=-w, wr=w, reps=reps, p=p,
                               nulltau=nulltau, statistic=statistic,
                               evall=evall, evalr=evalr, kernel=kernel, fuzzy=fuzzy,
                               quietly=True)

            P[1, count] = aux_fm['p.value']

            print('Bernoulli p-value (w = {}) = {}'.format(w, round(P[0, count], 3)))
            print('Fixed margins p-value (w = {}) = {}'.format(w, round(P[1, count], 3)))

            count += 1

    print('\n')

    ###############################################################################
    # Sensitivity analysis
    ###############################################################################

    print('Running sensitivity analysis...\n')

    if bound == 'upper':
        p_ub = np.zeros((len(gammalist), len(wlist)))
        count_g = 0

        for G in gammalist:
            plow = 1 / (1 + G)
            phigh = G / (1 + G)
            count_w = 0

            for w in wlist:
                ww = (np.round(Rc, 8) >= np.round(-w, 8)) & (np.round(Rc, 8) <= np.round(w, 8))
                Dw = D[ww]
                Yw = Y[ww]
                Rw = Rc[ww]

                data_w = np.column_stack((Yw, Rw, Dw))
                jj = np.argsort(data_w[:, 0])[::-1]
                data_dec = data_w[jj, :]
                Yw_dec = data_dec[:, 0]
                Rw_dec = data_dec[:, 1]

                nw = len(Rw)
                nw1 = np.sum(Dw)
                nw0 = nw - nw1
                pvals_ub = []

                for u in range(1, nw + 1):
                    uplus = np.concatenate((np.ones(u), np.zeros(nw - u)))
                    p_aux = phigh * uplus + plow * (1 - uplus)
                    aux = rdrandinf(Yw_dec, Rw_dec, wl=-w, wr=w, bernoulli=p_aux, reps=reps, p=p,
                                    nulltau=nulltau, statistic=statistic,
                                    evall=evall, evalr=evalr, kernel=kernel, fuzzy=fuzzy,
                                    quietly=True)
                    pvals_ub.append(aux['p.value'])

                p_ub_w = max(pvals_ub)
                p_ub[count_g, count_w] = p_ub_w

                count_w += 1

            count_g += 1

    if bound == 'both':
        p_ub = np.zeros((len(gammalist), len(wlist)))
        p_lb = np.zeros((len(gammalist), len(wlist)))
        count_g = 0

        for G in gammalist:
            plow = 1 / (1 + G)
            phigh = G / (1 + G)
            count_w = 0

            for w in wlist:
                ww = (np.round(Rc, 8) >= np.round(-w, 8)) & (np.round(Rc, 8) <= np.round(w, 8))
                Dw = D[ww]
                Yw = Y[ww]
                Rw = Rc[ww]

                data_w = np.column_stack((Yw, Rw, Dw))
                ii = np.argsort(data_w[:, 0])
                data_inc = data_w[ii, :]
                Yw_inc = data_inc[:, 0]
                Rw_inc = data_inc[:, 1]
                jj = np.argsort(data_w[:, 0])[::-1]
                data_dec = data_w[jj, :]
                Yw_dec = data_dec[:, 0]
                Rw_dec = data_dec[:, 1]

                nw = len(Rw)
                nw1 = np.sum(Dw)
                nw0 = nw - nw1
                pvals_ub = []
                pvals_lb = []

                for u in range(1, nw + 1):
                    uplus = np.concatenate((np.ones(u), np.zeros(nw - u)))
                    p_aux = phigh * uplus + plow * (1 - uplus)
                    aux = rdrandinf(Yw_dec, Rw_dec, wl=-w, wr=w, bernoulli=p_aux, reps=reps, p=p,
                                    nulltau=nulltau, statistic=statistic,
                                    evall=evall, evalr=evalr, kernel=kernel, fuzzy=fuzzy,
                                    quietly=True)
                    pvals_ub.append(aux['p.value'])

                    uminus = np.concatenate((np.zeros(nw - u), np.ones(u)))
                    p_aux = phigh * uminus + plow * (1 - uminus)
                    aux = rdrandinf(Yw_inc, Rw_inc, wl=-w, wr=w, bernoulli=p_aux, reps=reps, p=p,
                                    nulltau=nulltau, statistic=statistic,
                                    evall=evall, evalr=evalr, kernel=kernel, fuzzy=fuzzy,
                                    quietly=True)
                    pvals_lb.append(aux['p.value'])

                p_ub_w = max(pvals_ub)
                p_lb_w = min(pvals_lb)
                p_ub[count_g, count_w] = p_ub_w
                p_lb[count_g, count_w] = p_lb_w

                count_w += 1

            count_g += 1

    if bound == 'lower':
        p_lb = np.zeros((len(gammalist), len(wlist)))
        count_g = 0

        for G in gammalist:
            plow = 1 / (1 + G)
            phigh = G / (1 + G)
            count_w = 0

            for w in wlist:
                ww = (np.round(Rc, 8) >= np.round(-w, 8)) & (np.round(Rc, 8) <= np.round(w, 8))
                Dw = D[ww]
                Yw = Y[ww]
                Rw = Rc[ww]

                data_w = np.column_stack((Yw, Rw, Dw))
                ii = np.argsort(data_w[:, 0])
                data_inc = data_w[ii, :]
                Yw_inc = data_inc[:, 0]
                Rw_inc = data_inc[:, 1]

                nw = len(Rw)
                nw1 = np.sum(Dw)
                nw0 = nw - nw1
                pvals_lb = []

                for u in range(1, nw + 1):
                    uminus = np.concatenate((np.zeros(nw - u), np.ones(u)))
                    p_aux = phigh * uminus + plow * (1 - uminus)
                    aux = rdrandinf(Yw_inc, Rw_inc, wl=-w, wr=w, bernoulli=p_aux, reps=reps, p=p,
                                    nulltau=nulltau, statistic=statistic,
                                    evall=evall, evalr=evalr, kernel=kernel, fuzzy=fuzzy,
                                    quietly=True)
                    pvals_lb.append(aux['p.value'])

                p_lb_w = min(pvals_lb)
                p_lb[count_g, count_w] = p_lb_w

                count_w += 1

            count_g += 1

    output = {}

    print('\nSensitivity analysis complete.')

    ###############################################################################
    # Output
    ###############################################################################

    if not fmpval:
        P = P[1:, :]

    if bound == 'both':
        output['gamma'] = np.log(gammalist)
        output['expgamma'] = gammalist
        output['wlist'] = wlist
        output['p.values'] = P
        output['lower.bound'] = p_lb
        output['upper.bound'] = p_ub

    if bound == 'upper':
        output['gamma'] = np.log(gammalist)
        output['expgamma'] = gammalist
        output['wlist'] = wlist
        output['p.values'] = P
        output['upper.bound'] = p_ub

    if bound == 'lower':
        output['gamma'] = np.log(gammalist)
        output['expgamma'] = gammalist
        output['wlist'] = wlist
        output['p.values'] = P
        output['lower.bound'] = p_lb

    return output