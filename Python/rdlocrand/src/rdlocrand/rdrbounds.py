#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from rdlocrand.rdwinselect import rdwinselect
from rdlocrand.rdrandinf import rdrandinf
from rdlocrand.rdlocrand_fun import (
    rdrandinf_bernoulli_ranksum_pvalue,
    rdlocrand_preserve_rng,
)

@rdlocrand_preserve_rng
def rdrbounds(Y, R, cutoff=0, wlist=None, gamma=None, expgamma=None,
              bound='both', statistic='ranksum', p=0, evalat='cutoff',
              kernel='uniform', fuzzy=None, nulltau=0, prob=None,
              fmpval=False, reps=1000, seed=666):

    """
    Rosenbaum bounds for RD designs under local randomization

    rdrbounds calculates lower and upper bounds for the randomization p-value under different degrees of departure from a local randomized experiment, as suggested by Rosenbaum (2002).

    Authors:
    Matias D. Cattaneo, Princeton University. Email: matias.d.cattaneo@gmail.com
    Ricardo Masini, UC Davis. Email: ricardo.masini@gmail.com
    Rocio Titiunik, Princeton University. Email: rocio.titiunik@gmail.com
    Gonzalo Vazquez-Bare, UC Santa Barbara. Email: gvazquezbare@gmail.com

    References:
    Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2016).
    Inference in Regression Discontinuity Designs under Local Randomization.
    Stata Journal 16(2): 331-367.
    URL: https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf

    Rosenbaum, P. (2002). Observational Studies. Springer.

    Parameters:
    -----------
    Y : array-like
        A vector containing the values of the outcome variable.
    R : array-like
        A vector containing the values of the running variable.
    cutoff : float, optional
        The RD cutoff (default is 0).
    wlist : array-like, optional
        The list of window lengths to be evaluated. By default, the program
        constructs 10 windows around the cutoff, the first one including 10
        treated and control observations and adding 5 observations to each group
        in subsequent windows.
    gamma : array-like, optional
        The list of values of gamma to be evaluated.
    expgamma : array-like, optional
        The list of values of exp(gamma) to be evaluated. Default is
        [1.5, 2, 2.5, 3].
    bound : str, optional
        Specifies which bounds the command calculates. Options are 'upper',
        'lower', and 'both'. Default is 'both'.
    statistic : str, optional
        The statistic to be used in the balance tests. Allowed options are
        'diffmeans' (difference in means statistic), 'ksmirnov'
        (Kolmogorov-Smirnov statistic), and 'ranksum'
        (Wilcoxon-Mann-Whitney standardized statistic). Default option is
        'ranksum'. The statistic 'ttest' is equivalent to 'diffmeans' and
        included for backward compatibility.
    p : int, optional
        The order of the polynomial for the outcome adjustment model. Default is
        0.
    evalat : str, optional
        Specifies the point at which the adjusted variable is evaluated. Allowed
        options are 'cutoff' and 'means'. Default is 'cutoff'.
    kernel : str, optional
        Specifies the type of kernel to use as a weighting scheme. Allowed
        kernel types are 'uniform' (uniform kernel), 'triangular' (triangular
        kernel), and 'epan' (Epanechnikov kernel). Default is 'uniform'.
    fuzzy : None, array-like, or list, optional
        Indicates that the RD design is fuzzy. ``fuzzy`` can be specified as a
        vector containing the endogenous treatment values, or as a list where
        the first element is that vector and the second element is the statistic
        name. Allowed statistics are 'ar' (Anderson-Rubin statistic) and 'tsls'
        (2SLS statistic). Default statistic is 'ar'. The 'tsls' statistic relies
        on a large-sample approximation.
    nulltau : float, optional
        The value of the treatment effect under the null hypothesis. Default is
        0.
    prob : array-like, optional
        The probabilities of treatment for each unit when the assignment
        mechanism is a Bernoulli trial. This option should be specified as a
        vector of length equal to the length of the outcome and running
        variables.
    fmpval : bool, optional
        Reports the p-value under fixed margins randomization, in addition to
        the p-value under Bernoulli trials.
    reps : int, optional
        Number of replications. Default is 1000.
    seed : int, optional
        The seed to be used for the randomization tests.

    Returns:
    --------
    dict
        Dictionary containing:

        - ``gamma``: vector of gamma values.
        - ``expgamma``: vector of exp(gamma) values.
        - ``wlist``: window grid.
        - ``p.values``: p-values for each window under gamma = 0. When
          ``fmpval=True``, this includes Bernoulli and fixed-margins p-values.
        - ``lower.bound``: lower-bound p-values for each gamma-window pair;
          included when ``bound='lower'`` or ``bound='both'``.
        - ``upper.bound``: upper-bound p-values for each gamma-window pair;
          included when ``bound='upper'`` or ``bound='both'``.

    Examples:
    ---------
    # Toy dataset
    import numpy as np

    np.random.seed(123)
    R = np.random.uniform(-1, 1, size=100)
    Y = 1 + R - 0.5 * R**2 + 0.3 * R**3 + (R >= 0) + np.random.normal(size=100)

    # Rosenbaum bounds
    # Note: low number of replications and windows to speed up the process.
    # The user should increase these values.
    tmp = rdrbounds(Y, R, expgamma=[1.5, 2], wlist=[0.3], reps=100)

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
        wlist = np.round(aux['results'].to_numpy()[:, 6], 2)

    evall = cutoff
    evalr = cutoff
    fast_ranksum = statistic == 'ranksum' and p == 0 and kernel == 'uniform' and fuzzy is None

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

            if fast_ranksum:
                prob_w = prob_be[ww] if len(prob_be) == len(R) else prob_be
                P[0, count] = rdrandinf_bernoulli_ranksum_pvalue(
                    Y[ww], Rw, prob_w, reps=reps, nulltau=nulltau
                )
            else:
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

            if fast_ranksum:
                prob_w = prob_be[ww] if len(prob_be) == len(R) else prob_be
                P[0, count] = rdrandinf_bernoulli_ranksum_pvalue(
                    Y[ww], Rw, prob_w, reps=reps, nulltau=nulltau
                )
            else:
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
                    if fast_ranksum:
                        pvals_ub.append(
                            rdrandinf_bernoulli_ranksum_pvalue(
                                Yw_dec, Rw_dec, p_aux, reps=reps, nulltau=nulltau
                            )
                        )
                    else:
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
                    if fast_ranksum:
                        pvals_ub.append(
                            rdrandinf_bernoulli_ranksum_pvalue(
                                Yw_dec, Rw_dec, p_aux, reps=reps, nulltau=nulltau
                            )
                        )
                    else:
                        aux = rdrandinf(Yw_dec, Rw_dec, wl=-w, wr=w, bernoulli=p_aux, reps=reps, p=p,
                                        nulltau=nulltau, statistic=statistic,
                                        evall=evall, evalr=evalr, kernel=kernel, fuzzy=fuzzy,
                                        quietly=True)
                        pvals_ub.append(aux['p.value'])

                    uminus = np.concatenate((np.zeros(nw - u), np.ones(u)))
                    p_aux = phigh * uminus + plow * (1 - uminus)
                    if fast_ranksum:
                        pvals_lb.append(
                            rdrandinf_bernoulli_ranksum_pvalue(
                                Yw_inc, Rw_inc, p_aux, reps=reps, nulltau=nulltau
                            )
                        )
                    else:
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
                    if fast_ranksum:
                        pvals_lb.append(
                            rdrandinf_bernoulli_ranksum_pvalue(
                                Yw_inc, Rw_inc, p_aux, reps=reps, nulltau=nulltau
                            )
                        )
                    else:
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
        P = P[:1, :]

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
