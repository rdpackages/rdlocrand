#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm, ks_2samp, ranksums, f
import statsmodels.api as sm

#################################################################
# rdrandinf observed statistics and asymptotic p-values
#################################################################

def rdrandinf_model(Y, D, statistic, pvalue=False, kweights=None, endogtr=None, delta=None):
    
    try: n = len(D)
    except: n = 1
    n1 = np.sum(D)
    n0 = n - n1

    try: lenY = len(Y)
    except: lenY = 1
    Y = np.array(Y).reshape(lenY,-1)
   
    if statistic == 'ttest' or statistic == 'diffmeans':
        if np.all(kweights == 1):
            Y1 = Y[D == 1]
            Y0 = Y[D == 0]
            M1 = np.mean(Y1, axis=0)
            M0 = np.mean(Y0, axis=0)
            stat = M1 - M0
            if pvalue:
                V1 = np.mean((Y1 - np.mean(Y1, axis=0)) ** 2, axis=0) / (n1 - 1)
                V0 = np.mean((Y0 - np.mean(Y0, axis=0)) ** 2, axis=0) / (n0 - 1)
                se = np.sqrt(V1 + V0)
                t_stat = (M1 - M0) / se
                asy_pval = 2 * norm.cdf(-np.abs(t_stat))
                if delta is not None:
                    asy_power = 1 - norm.cdf(1.96 - delta / se) + norm.cdf(-1.96 - delta / se)
                else:
                    asy_power = np.nan
        else:
            stat = np.zeros(Y.shape[1])
            asy_pval = np.zeros(Y.shape[1])
            for k in range(Y.shape[1]):
                lm_aux = sm.WLS(Y[:, k], sm.add_constant(D), weights=kweights).fit()
                stat[k] = lm_aux.params[1]
                if pvalue:
                    se = np.sqrt(lm_aux.cov_params().loc[1, 1])
                    t_stat = stat[k] / se
                    asy_pval[k] = 2 * norm.cdf(-np.abs(t_stat))
                    if delta is not None:
                        asy_power = 1 - norm.cdf(1.96 - delta / se) + norm.cdf(-1.96 - delta / se)
                    else:
                        asy_power = np.nan

    elif statistic == 'ksmirnov':
        stat = np.zeros(Y.shape[1])
        asy_pval = np.zeros(Y.shape[1])
        for k in range(Y.shape[1]):
            aux_ks = ks_2samp(Y[D == 0, k], Y[D == 1, k])
            stat[k] = aux_ks.statistic
            if pvalue:
                asy_pval[k] = aux_ks.pvalue
                asy_power = np.nan

    elif statistic == 'ranksum':
        stat = np.zeros(Y.shape[1])
        asy_pval = np.zeros(Y.shape[1])
        for k in range(Y.shape[1]):
            T_stat, _ = ranksums(Y[D == 0, k], Y[D == 1, k])
            stat[k] = T_stat
        if pvalue:
            asy_pval = 2 * norm.cdf(-np.abs(stat))
            sigma = np.std(Y, axis=0, ddof = 1)
            if delta is not None:
                asy_power = norm.cdf(np.sqrt(3 * n0 * n1 / ((n0 + n1 + 1) * np.pi)) * delta / sigma - 1.96)
            else:
                asy_power = np.nan

    elif statistic == 'all':
        stat1 = np.mean(Y[D == 1,0]) - np.mean(Y[D == 0,0])
        aux_ks = ks_2samp(Y[D == 0,0], Y[D == 1,0])
        stat2 = aux_ks.statistic
        aux_rs = ranksums(Y[D == 0,0], Y[D == 1,0])
        stat3 = aux_rs.statistic
        stat = np.array([stat1, stat2, stat3])
        if pvalue:
            Y1 = Y[D == 1,0]
            Y0 = Y[D == 0,0]
            V1 = np.mean((Y1 - np.mean(Y1, axis=0)) ** 2, axis=0) / (n1 - 1)
            V0 = np.mean((Y0 - np.mean(Y0, axis=0)) ** 2, axis=0) / (n0 - 1)
            se1 = np.sqrt(V1 + V0)
            t_stat = stat1 / se1
            asy_pval1 = 2 * norm.cdf(-np.abs(t_stat))
            asy_pval2 = aux_ks.pvalue
            asy_pval3 = 2 * norm.cdf(-np.abs(stat3))
            asy_pval = np.array([asy_pval1, asy_pval2, asy_pval3])
            if delta is not None:
                asy_power1 = 1 - norm.cdf(1.96 - delta / se1) + norm.cdf(-1.96 - delta / se1)
                asy_power2 = np.nan
                sigma = np.std(Y[:,0], axis=0, ddof = 1)
                asy_power3 = norm.cdf(np.sqrt(3 * n0 * n1 / ((n0 + n1 + 1) * np.pi)) * delta / sigma - 1.96)
                asy_power = np.array([asy_power1, asy_power2, asy_power3])
            else:
                asy_power = np.array([np.nan, np.nan, np.nan])
    
    elif statistic == 'ar':
        stat = np.mean(Y[D == 1], axis=0) - np.mean(Y[D == 0], axis=0)
        if pvalue:
            se = np.sqrt(np.var(Y[D == 1], axis=0) / n1 + np.var(Y[D == 0], axis=0) / n0)
            t_stat = stat / se
            asy_pval = 2 * norm.cdf(-np.abs(t_stat))
            if delta is not None:
                asy_power = 1 - norm.cdf(1.96 - delta / se) + norm.cdf(-1.96 - delta / se)
            else:
                asy_power = np.nan
    
    elif statistic == 'wald':
        fs = sm.OLS(endogtr, sm.add_constant(D)).fit()
        rf = sm.OLS(Y, sm.add_constant(D)).fit()
        stat = rf.params[1] / fs.params[1]
        if pvalue:
            ehat = Y - np.mean(Y) - stat * (endogtr - np.mean(endogtr))
            ehat2 = ehat ** 2
            se = np.sqrt((np.mean(ehat2) * np.var(D)) / (n * np.cov(D, endogtr) ** 2))
            t_stat = stat / se
            asy_pval = 2 * norm.cdf(-np.abs(t_stat))
            if delta is not None:
                asy_power = 1 - norm.cdf(1.96 - delta / se) + norm.cdf(-1.96 - delta / se)
            else:
                asy_power = np.nan

    if pvalue:
        output = {'statistic': stat, 'p_value': asy_pval, 'asy_power': asy_power}
    else:
        output = {'statistic': stat}

    return output

#################################################################
# Hotelling's T2 statistic
#################################################################

def hotelT2(X, D):
    n = len(D)
    n1 = np.sum(D)
    n0 = n - n1
    p = X.shape[1]

    X1 = X[D == 1, :]
    X0 = X[D == 0, :]
    X1bar = np.mean(X1, axis=0)
    X0bar = np.mean(X0, axis=0)
    S1 = np.cov(X1.T)
    S0 = np.cov(X0.T)
    Spool = (S1 * (n1 - 1) + S0 * (n0 - 1)) / (n - 2)
    SpoolInv = np.linalg.inv(Spool)

    T2 = (n0 * n1 / n) * np.dot(np.dot((X1bar - X0bar).T, SpoolInv), (X1bar - X0bar))
    Fstat = ((n - p - 1) / ((n - 2) * p)) * T2
    pval = 1 - f.cdf(Fstat, p, n - 1 - p)

    output = {'statistic': float(T2), 'Fstat': float(Fstat), 'p.value': float(pval)}

    return output

#################################################################
# Find window increments
#################################################################

def findwobs(wobs, nwin, posl, posr, R, dups):
    N = len(R)
    Nc = np.sum(R < 0)
    Nt = np.sum(R >= 0)
    mpoints_l = len(np.unique(R[:Nc]))
    mpoints_r = len(np.unique(R[Nc:N]))
    mpoints_max = max(mpoints_l, mpoints_r)
    nwin_mp = min(nwin, mpoints_max)
    poslold = posl
    posrold = posr

    win = 1
    wlist_left = []
    poslist_left = []
    wlist_right = []
    poslist_right = []

    while (win <= nwin_mp) and (wobs < max(posl, Nt - (posr - Nc - 1))):

        poslold = posl
        posrold = posr

        while (dups[posl-1] < wobs) and (np.sum(R[posl-1] <= R[(posl-1):poslold]) < wobs) and (posl > 1):
            posl = max(posl - dups[posl-1], 1)

        while (dups[posr-1] < wobs) and (np.sum(R[(posrold-1):posr] <= R[posr-1]) < wobs) and (posr < N):
            posr = min(posr + dups[posr-1], N)

        wlength_left = R[posl-1]
        wlength_right = R[posr-1]

        wlist_left.append(wlength_left)
        poslist_left.append(posl)
        wlist_right.append(wlength_right)
        poslist_right.append(posr)

        posl = max(posl - dups[posl-1], 1)
        posr = min(posr + dups[posr-1], N)

        win += 1

    output = {
        'posl': posl,
        'posr': posr,
        'wlength_left': wlength_left,
        'wlength_right': wlength_right,
        'wlist_left': wlist_left,
        'wlist_right': wlist_right,
        'poslist_left': poslist_left,
        'poslist_right': poslist_right,
    }

    return output

#################################################################
# Find symmetric window increments
#################################################################

def findwobs_sym(wobs, nwin, posl, posr, R, dups):
    N = len(R)
    Nc = np.sum(R < 0)
    Nt = np.sum(R >= 0)
    poslold = posl
    posrold = posr
    wlist = []
    win = 1

    while (win <= nwin) and (wobs < min(posl, Nt - (posr - Nc - 1))):

        poslold = posl
        posrold = posr

        while (dups[posl-1] < wobs) and (np.sum(R[posl-1] <= R[(posl-1):poslold]) < wobs):
            posl = max(posl - dups[posl-1], 1)

        while (dups[posr-1] < wobs) and (np.sum(R[(posrold-1):posr] <= R[posr-1]) < wobs):
            posr = min(posr + dups[posr-1], N)

        if abs(R[posl-1]) < R[posr-1]:
            posl = Nc + 1 - np.sum(-R[posr-1] <= R[:Nc])

        if abs(R[posl-1]) > R[posr-1]:
            posr = np.sum(R[Nc:N] <= abs(R[posl-1])) + Nc

        wlength = max(-R[posl-1], R[posr-1])
        wlist.append(wlength)

        posl = max(posl - dups[posl-1], 1)
        posr = min(posr + dups[posr-1], N)
        win += 1

    return np.array(wlist)

#################################################################
# Find CI
#################################################################

def find_CI(pvals, alpha, tlist):
    if np.all(pvals >= alpha):
        CI = np.array([[tlist[0], tlist[-1]]])
    elif np.all(pvals < alpha):
        CI = np.full((1, 2), np.nan)
    else:
        whichvec = np.where(pvals >= alpha)[0]
        index_l = np.min(whichvec)
        index_r = np.max(whichvec)
        indexmat = np.array([[index_l, index_r]])

        whichvec_cut = whichvec.copy()
        dif = np.diff(whichvec_cut)
        while np.all(dif == 1) is False:
            cut = np.min(np.where(dif != 1))
            auxvec = whichvec_cut[:cut + 1]
            indexmat = np.vstack((indexmat, [np.min(auxvec), np.max(auxvec)]))
            whichvec_cut = whichvec_cut[cut + 1:]

            dif = np.diff(whichvec_cut)

        if indexmat.shape[0] > 1:
            indexmat = indexmat[1:, :]
            indexmat = np.vstack((indexmat, [np.min(whichvec_cut), np.max(whichvec_cut)]))
        CI = np.array([[tlist[i] for i in indexmat[0]]])

    return CI

#################################################################
# Find window length - DEPRECATED: for backward compatibility
#################################################################

def wlength(R, D, num):
    X1 = np.sort(np.abs(R[D == 1]))
    X0 = np.sort(np.abs(R[D == 0]))
    m = min(len(X1), len(X0))
    if num > m:
        num = m
    xt = X1[num - 1]
    xc = X0[num - 1]
    minw = max(xc, xt)
    return minw

#################################################################
# Find default step - DEPRECATED: for backward compatibility
#################################################################

def findstep(R, D, obsmin, obsstep, times):
    S = []
    for i in range(1, times + 1):
        U = wlength(R, D, obsmin + obsstep * i)
        L = wlength(R, D, obsmin + obsstep * (i - 1))
        Snext = U - L
        S.append(Snext)
    step = max(S)
    return step