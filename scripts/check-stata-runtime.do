version 13
clear all
set more off
set linesize 100

args repo_root
if `"`repo_root'"' == "" {
    local repo_root "`c(pwd)'"
}
local repo_root : subinstr local repo_root "\" "/", all
local stata_dir "`repo_root'/stata"

capture mkdir "`repo_root'/tmp"
capture mkdir "`repo_root'/tmp/stata-check"
capture log close _all
log using "`repo_root'/tmp/stata-check/check-stata-runtime.log", text replace

cd "`stata_dir'"
adopath ++ "`stata_dir'"
mata: mata mlib index

which rdwinselect
which rdrandinf
which rdsensitivity
which rdrbounds

use rdlocrand_senate.dta, clear
local covariates presdemvoteshlag1 population demvoteshlag1

quietly rdwinselect demmv `covariates', cutoff(0) wmin(.5) wstep(.125) nwindows(2) reps(19) seed(12345)
matrix RW_results = r(results)
matrix RW_left = r(wlist_left)
matrix RW_right = r(wlist_right)
scalar RW_w_left = r(w_left)
scalar RW_w_right = r(w_right)
if rowsof(RW_results) != 2 | colsof(RW_results) != 6 {
    display as error "rdwinselect returned results with unexpected dimensions"
    exit 459
}
if rowsof(RW_left) != 1 | colsof(RW_left) != 2 | rowsof(RW_right) != 1 | colsof(RW_right) != 2 {
    display as error "rdwinselect returned window lists with unexpected dimensions"
    exit 459
}
if missing(RW_w_left) | missing(RW_w_right) {
    display as error "rdwinselect did not return a recommended window"
    exit 459
}

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(19) seed(12345)
scalar RI_randpval = r(randpval)
scalar RI_obs_stat = r(obs_stat)
scalar RI_n = r(N)
if missing(RI_randpval) | RI_randpval < 0 | RI_randpval > 1 {
    display as error "rdrandinf returned an invalid randomization p-value"
    exit 459
}
if missing(RI_obs_stat) | missing(RI_n) | RI_n <= 0 {
    display as error "rdrandinf returned invalid scalar results"
    exit 459
}

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(19) seed(12345) ci(.05 -5(5)5)
matrix RI_CI = r(CI)
if rowsof(RI_CI) < 1 | colsof(RI_CI) != 2 {
    display as error "rdrandinf did not return a two-column CI matrix"
    exit 459
}

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(19) seed(12345) interfci(.05)
scalar RI_int_lb = r(int_lb)
scalar RI_int_ub = r(int_ub)
if missing(RI_int_lb) | missing(RI_int_ub) | RI_int_lb > RI_int_ub {
    display as error "rdrandinf returned invalid interference confidence limits"
    exit 459
}

quietly rdsensitivity demvoteshfor2 demmv, wlist(.75 1) tlist(0(1)20) reps(19) seed(12345) nodots nodraw ci(-.75 .75)
matrix RS_results = r(results)
matrix RS_CI = r(CI)
if rowsof(RS_results) != 21 | colsof(RS_results) != 2 {
    display as error "rdsensitivity returned results with unexpected dimensions"
    exit 459
}
if rowsof(RS_CI) < 1 | colsof(RS_CI) != 2 {
    display as error "rdsensitivity did not return a two-column CI matrix"
    exit 459
}

quietly rdrbounds demvoteshfor2 demmv, expgamma(1.5 2) wlist(.5 .75) reps(19) seed(12345)
matrix RB_lbound = r(lbound)
matrix RB_ubound = r(ubound)
matrix RB_pvals = r(pvals)
if rowsof(RB_lbound) != 2 | colsof(RB_lbound) != 2 {
    display as error "rdrbounds returned lower bounds with unexpected dimensions"
    exit 459
}
if rowsof(RB_ubound) != 2 | colsof(RB_ubound) != 2 {
    display as error "rdrbounds returned upper bounds with unexpected dimensions"
    exit 459
}
if rowsof(RB_pvals) < 1 | colsof(RB_pvals) != 2 {
    display as error "rdrbounds returned p-values with unexpected dimensions"
    exit 459
}

display as text "Stata runtime checks passed."
log close
