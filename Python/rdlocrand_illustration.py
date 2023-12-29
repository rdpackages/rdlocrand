###############################################################################
# rdlocrand: illustration file
# !version 1.0 29-Jun-2023
# Authors: Matias Cattaneo, Ricardo Masini, Rocio Titiunik, Gonzalo Vazquez-Bare
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdlocrand import *

###############################################################################
## Load data
###############################################################################

#data = pd.read_csv("rdlocrand_senate.csv")
data = pd.read_csv("Python/rdlocrand_senate.csv")

# Select predetermined covariates to be used for window selector

X = data[["presdemvoteshlag1", "population", "demvoteshlag1", "demvoteshlag2",
          "demwinprv1", "demwinprv2", "dopen", "dmidterm", "dpresdem"]]

X.loc[:,'population'] = X['population'].div(1000000)

# Assign names to the covariates

colnames_X = ["DemPres Vote", "Population", "DemSen Vote t-1", "DemSen Vote t-2",
              "DemSen Win t-1", "DemSen Win t-2", "Open", "Midterm", "DemPres"]

X.columns = colnames_X

# Running variable and outcome variable

R = data["demmv"].values
Y = data["demvoteshfor2"].values
D = 1*(R >= 0)

##############################################################################
# rdwinselect
##############################################################################

# Deprecated default options (Stata Journal 2016)
tmp = rdwinselect(R, X, obsstep=2)

# Window selection with default options
tmp = rdwinselect(R, X)

# Window selection with default options and symmetric windows
tmp = rdwinselect(R, X, wasymmetric=True)

# Window selection with specified window length and increments (replicate CFT)
tmp = rdwinselect(R, X, wmin=0.5, wstep=0.125, reps=10000)

# Window selection using large sample approximation and plotting p-values
tmp = rdwinselect(R, X, wmin=0.5, wstep=0.125, approx=True, nwindows = 80, quietly=True, plot=True)

###############################################################################
## rdrandinf
###############################################################################

# Randomization inference using recommended window
tmp = rdrandinf(Y, R, wl=-0.75, wr=0.75)

# Randomization inference using recommended window, all statistics
tmp = rdrandinf(Y, R, wl=-0.75, wr=0.75, statistic='all')

# Randomization inference using recommended window using rdwinselect
tmp = rdrandinf(Y, R, statistic='all', covariates=X, wmin=0.5, wstep=0.125, rdwreps=10000)

# Randomization inference using recommended window, linear adjustment
tmp = rdrandinf(Y, R, wl=-0.75, wr=0.75, statistic='all', p=1)

# Randomization inference under interference
tmp = rdrandinf(Y, R, wl=-0.75, wr=0.75, interfci=0.05)

###############################################################################
# rdsensitivity
###############################################################################

tmp = rdsensitivity(Y, R, wlist=np.arange(0.75, 10.25, 0.25), tlist=np.arange(0, 21, 1))

# Replicate contour plot
xaxis = tmp['wlist']
yaxis = tmp['tlist']
zvalues = tmp['results']
plt.contourf(xaxis, yaxis, zvalues, levels=np.arange(0, 1.01, 0.01), cmap='gray')
plt.xlabel('window')
plt.ylabel('treatment effect')
plt.colorbar(label='p-value')
plt.title('Sensitivity Analysis')
plt.show()

# Obtain 95 percent confidence interval for window [-0.75 ; 0.75]
tmp = rdsensitivity(Y, R, wlist=np.arange(0.75, 2.25, 0.25), tlist=np.arange(0, 21, 1), ci=[-0.75, 0.75])
confidence_interval = tmp['ci']
print(confidence_interval)

# rdsensitivity to calculate CI from within rdrandinf
tmp = rdrandinf(Y, R, wl=-0.75, wr=0.75, ci=[0.05] + list(np.arange(3, 21, 1)))


###############################################################################
# rdrbounds
###############################################################################

tmp = rdrbounds(Y, R, expgamma=[1.5, 2, 3], wlist=[0.5, 0.75, 1], reps=1000)
lower_bound = tmp['lower.bound']
print(lower_bound)
upper_bound = tmp['upper.bound']
print(upper_bound)

# Bernoulli and fixed margins p-values
tmp = rdrbounds(Y, R, expgamma=[1.5, 2, 3], wlist=[0.5, 0.75, 1], reps=1000, fmpval=True)
lower_bound = tmp['lower.bound']
print(lower_bound)
upper_bound = tmp['upper.bound']
print(upper_bound)

###############################################################################
# rdrandinf with eval options
###############################################################################

ii = (R >= -0.75) & (R <= 0.75) & (~np.isnan(Y)) & (~np.isnan(R))
m0 = np.mean(R[ii & (D == 0)], axis=0)
m1 = np.mean(R[ii & (D == 1)], axis=0)
tmp = rdrandinf(Y, R, wl=-0.75, wr=0.75, p=1, evall=m0, evalr=m1)