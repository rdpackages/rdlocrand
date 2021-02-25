********************************************************************************
** RDLOCRAND Stata Package 
** Empirical Illustration
** Authors: Matias D. Cattaneo, Rocio Titiunik and Gonzalo Vazquez-Bare
** Last update: 2021-02-23
********************************************************************************
* net install rdlocrand, from(https://raw.githubusercontent.com/rdpackages/rdlocrand/master/stata) replace
********************************************************************************

use rdlocrand_senate.dta, clear
global covariates presdemvoteshlag1 population demvoteshlag1 ///
                  demvoteshlag2 demwinprv1 demwinprv2 dopen dmidterm

				  
********************************************************************************
** Summary Stats
********************************************************************************

describe $covariates
summarize demmv $covariates


********************************************************************************
** rdwinselect
********************************************************************************

** Replicate first table of Stata journal article (deprecated default options - not recommended)

rdwinselect demmv $covariates, cutoff(0) obsstep(2)

** Window selection with default options

rdwinselect demmv $covariates, cutoff(0)

** Window selection with default options and symmetric windows

rdwinselect demmv $covariates, cutoff(0) wsym

** Window selection setting window length and increments (replicate CFT)

rdwinselect demmv $covariates, wmin(.5) wstep(.125) reps(10000)

** Window selection using large sample approximation and plotting p-values

quietly rdwinselect demmv $covariates, wmin(.5) wstep(.125) ///
                    nwin(80) approximate plot

					
********************************************************************************
** rdrandinf
********************************************************************************

** Randomization inference using recommended window

rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75)

** Randomization inference using recommended window, all statistics

rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) statistic(all)

** Randomization inference using recommended window using rdwinselect

rdrandinf demvoteshfor2 demmv, statistic(all) covariates($covariates) ///
	wmin(.5) wstep(.125) level(0.16) quietly rdwreps(10000)

** Randomization inference using recommended window, linear adjustment

rdrandinf demvoteshfor2 demmv, statistic(all) wl(-.75) wr(.75) p(1)

** Randomization inference under interference

rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) interfci(.05)


********************************************************************************
** rdsensitivity
********************************************************************************

rdsensitivity demvoteshfor2 demmv, wlist(.75(.25)2) tlist(0(1)20) verbose

** Obtain 95 percent confidence interval for window [-.75 ; .75]

rdsensitivity demvoteshfor2 demmv, wlist(.75(.25)2) tlist(0(1)20) nodots ci(-.75 .75)

** Replicate contour plot

rdsensitivity demvoteshfor2 demmv, wlist(.75(.25)10) tlist(0(1)20) nodots ///
                                   saving(graphdata)
preserve
use graphdata, clear
twoway contour pvalue t w, ccuts(0(0.05)1)
restore

preserve
use graphdata, clear
twoway contour pvalue t w, ccuts(0(0.05)1) ccolors(gray*0.01 gray*0.05 ///
	gray*0.1 gray*0.15 gray*0.2 gray*0.25 gray*0.3 gray*0.35 ///
	gray*0.4 gray*0.5 gray*0.6 gray*0.7 gray*0.8 gray*0.9 gray ///
	black*0.5  black*0.6 black*0.7 black*0.8 black*0.9 black) ///
	xlabel(.75(1.25)10) ylabel(0(2)20, nogrid) graphregion(fcolor(none))
restore

** rdsensitivity to calculate CI from within rdrandinf

rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) ci(.05 3(1)20)


********************************************************************************
** rdrbounds
********************************************************************************

rdrbounds demvoteshfor2 demmv, expgamma(1.5 2 3) wlist(.5 .75 1) reps(1000)

** Bernoulli and fixed margins p-values

rdrbounds demvoteshfor2 demmv, expgamma(1.5 2 3) wlist(.5 .75 1) reps(1000) fmpval


********************************************************************************
** rdrandinf with eval options
********************************************************************************

qui sum demmv if abs(demmv)<=.75 & demmv>=0 & demmv!=. & demvoteshfor2!=.
local mt = r(mean)
qui sum demmv if abs(demmv)<=.75 & demmv<0  & demmv!=. & demvoteshfor2!=.
local mc = r(mean)
rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) p(1) evall(`mc') evalr(`mt')

