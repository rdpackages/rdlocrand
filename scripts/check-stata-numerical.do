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
log using "`repo_root'/tmp/stata-check/check-stata-numerical.log", text replace

cd "`stata_dir'"
adopath ++ "`stata_dir'"
mata: mata mlib index

program define assert_close
    args label actual expected
    local tol = 1e-12
    if missing(`expected') {
        if !missing(`actual') {
            display as error "`label' changed: expected missing, got " %21.16g `actual'
            exit 459
        }
    }
    else if missing(`actual') | abs(`actual' - `expected') > `tol' {
        display as error "`label' changed: expected " %21.16g `expected' ", got " %21.16g `actual'
        exit 459
    }
end

program define assert_matrix_dims
    args label matrix_name expected_rows expected_cols
    if rowsof(`matrix_name') != `expected_rows' | colsof(`matrix_name') != `expected_cols' {
        display as error "`label' dimensions changed: expected `expected_rows'x`expected_cols', got " rowsof(`matrix_name') "x" colsof(`matrix_name')
        exit 459
    }
end

program define assert_matrix_value
    args label matrix_name row col expected
    assert_close "`label'[`row',`col']" `matrix_name'[`row',`col'] `expected'
end

use rdlocrand_senate.dta, clear
local covariates presdemvoteshlag1 population demvoteshlag1

quietly rdwinselect demmv `covariates', cutoff(0) wmin(.5) wstep(.125) nwindows(2) reps(19) seed(12345)
matrix RW_results = r(results)
matrix RW_left = r(wlist_left)
matrix RW_right = r(wlist_right)
assert_matrix_dims RW_results RW_results 2 6
assert_matrix_value RW_results RW_results 1 1 .5263157894736842
assert_matrix_value RW_results RW_results 1 2 .2295229434967042
assert_matrix_value RW_results RW_results 1 3 9
assert_matrix_value RW_results RW_results 1 4 16
assert_matrix_value RW_results RW_results 1 5 -.5
assert_matrix_value RW_results RW_results 1 6 .5
assert_matrix_value RW_results RW_results 2 1 .631578947368421
assert_matrix_value RW_results RW_results 2 2 .3770855874754487
assert_matrix_value RW_results RW_results 2 3 13
assert_matrix_value RW_results RW_results 2 4 19
assert_matrix_value RW_results RW_results 2 5 -.625
assert_matrix_value RW_results RW_results 2 6 .625
assert_matrix_dims RW_left RW_left 1 2
assert_matrix_value RW_left RW_left 1 1 -.5
assert_matrix_value RW_left RW_left 1 2 -.625
assert_matrix_dims RW_right RW_right 1 2
assert_matrix_value RW_right RW_right 1 1 .5
assert_matrix_value RW_right RW_right 1 2 .625
assert_close RW_w_left r(w_left) -.625
assert_close RW_w_right r(w_right) .625
assert_close RW_minp r(minp) .631578947368421

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(19) seed(12345)
assert_close RI_randpval r(randpval) 0
assert_close RI_asy_pval r(asy_pval) .0000795467132186
assert_close RI_obs_stat r(obs_stat) 9.68949952559038
assert_close RI_n r(N) 37
assert_close RI_n_left r(N_left) 15
assert_close RI_n_right r(N_right) 22

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) statistic(ranksum) reps(19) seed(12345)
assert_close RIR_randpval r(randpval) 0
assert_close RIR_asy_pval r(asy_pval) .0012945790640629
assert_close RIR_obs_stat r(obs_stat) -3.217178769426679

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) statistic(ksmirnov) reps(19) seed(12345)
assert_close RIK_randpval r(randpval) 0
assert_close RIK_asy_pval r(asy_pval) .008803820266996901
assert_close RIK_obs_stat r(obs_stat) .5515151515151515

capture drop ri_probs
quietly generate double ri_probs = .5 if demmv >= -.75 & demmv <= .75 & demvoteshfor2 != .
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) bernoulli(ri_probs) reps(19) seed(12345)
assert_close RIB_randpval r(randpval) 0
assert_close RIB_asy_pval r(asy_pval) .0000795467132186
assert_close RIB_obs_stat r(obs_stat) 9.68949952559038

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) p(1) reps(19) seed(12345)
assert_close RIP1_randpval r(randpval) 0
assert_close RIP1_asy_pval r(asy_pval) .0659656703091592
assert_close RIP1_obs_stat r(obs_stat) 15.29651683658787

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) fuzzy(demwinprv1) reps(19) seed(12345)
assert_close RIF_randpval r(randpval) 0
assert_close RIF_asy_pval r(asy_pval) .0000795467132186
assert_close RIF_obs_stat r(obs_stat) 9.68949952559038

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(19) seed(12345) statistic(all)
matrix RIA_pval = r(p_val)
matrix RIA_asy = r(asy_pval)
matrix RIA_obs = r(obs_stat)
assert_matrix_dims RIA_pval RIA_pval 1 3
assert_matrix_value RIA_pval RIA_pval 1 1 0
assert_matrix_value RIA_pval RIA_pval 1 2 0
assert_matrix_value RIA_pval RIA_pval 1 3 0
assert_matrix_dims RIA_asy RIA_asy 1 3
assert_matrix_value RIA_asy RIA_asy 1 1 .0000795467132186
assert_matrix_value RIA_asy RIA_asy 1 2 .008803820266996901
assert_matrix_value RIA_asy RIA_asy 1 3 .0012945790640629
assert_matrix_dims RIA_obs RIA_obs 1 3
assert_matrix_value RIA_obs RIA_obs 1 1 9.68949952559038
assert_matrix_value RIA_obs RIA_obs 1 2 .5515151515151515
assert_matrix_value RIA_obs RIA_obs 1 3 -3.217178769426679

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(19) seed(12345) ci(.05 -5(5)5)
matrix RI_CI = r(CI)
assert_matrix_dims RI_CI RI_CI 1 2
assert_matrix_value RI_CI RI_CI 1 1 .
assert_matrix_value RI_CI RI_CI 1 2 .

quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(19) seed(12345) interfci(.05)
assert_close RI_int_lb r(int_lb) 3.853152422471484
assert_close RI_int_ub r(int_ub) 15.90456929640337

quietly rdsensitivity demvoteshfor2 demmv, wlist(.75 1) tlist(0(1)20) reps(19) seed(12345) nodots nodraw ci(-.75 .75)
matrix RS_results = r(results)
matrix RS_CI = r(CI)
assert_matrix_dims RS_results RS_results 21 2
assert_matrix_value RS_results RS_results 1 1 0
assert_matrix_value RS_results RS_results 1 2 0
assert_matrix_value RS_results RS_results 2 1 0
assert_matrix_value RS_results RS_results 2 2 0
assert_matrix_value RS_results RS_results 3 1 0
assert_matrix_value RS_results RS_results 3 2 0
assert_matrix_value RS_results RS_results 4 1 0
assert_matrix_value RS_results RS_results 4 2 0
assert_matrix_value RS_results RS_results 5 1 .1052631578947368
assert_matrix_value RS_results RS_results 5 2 0
assert_matrix_value RS_results RS_results 6 1 .1052631578947368
assert_matrix_value RS_results RS_results 6 2 0
assert_matrix_value RS_results RS_results 7 1 .1052631578947368
assert_matrix_value RS_results RS_results 7 2 .1052631578947368
assert_matrix_value RS_results RS_results 8 1 .1052631578947368
assert_matrix_value RS_results RS_results 8 2 .2105263157894737
assert_matrix_value RS_results RS_results 9 1 .4210526315789473
assert_matrix_value RS_results RS_results 9 2 .2105263157894737
assert_matrix_value RS_results RS_results 10 1 .7368421052631579
assert_matrix_value RS_results RS_results 10 2 .5263157894736842
assert_matrix_value RS_results RS_results 11 1 .7368421052631579
assert_matrix_value RS_results RS_results 11 2 .9473684210526315
assert_matrix_value RS_results RS_results 12 1 .5263157894736842
assert_matrix_value RS_results RS_results 12 2 .5263157894736842
assert_matrix_value RS_results RS_results 13 1 .3157894736842105
assert_matrix_value RS_results RS_results 13 2 .3157894736842105
assert_matrix_value RS_results RS_results 14 1 .2105263157894737
assert_matrix_value RS_results RS_results 14 2 .2105263157894737
assert_matrix_value RS_results RS_results 15 1 0
assert_matrix_value RS_results RS_results 15 2 0
assert_matrix_value RS_results RS_results 16 1 0
assert_matrix_value RS_results RS_results 16 2 0
assert_matrix_value RS_results RS_results 17 1 0
assert_matrix_value RS_results RS_results 17 2 0
assert_matrix_value RS_results RS_results 18 1 0
assert_matrix_value RS_results RS_results 18 2 0
assert_matrix_value RS_results RS_results 19 1 0
assert_matrix_value RS_results RS_results 19 2 0
assert_matrix_value RS_results RS_results 20 1 0
assert_matrix_value RS_results RS_results 20 2 0
assert_matrix_value RS_results RS_results 21 1 0
assert_matrix_value RS_results RS_results 21 2 0
assert_matrix_dims RS_CI RS_CI 1 2
assert_matrix_value RS_CI RS_CI 1 1 4
assert_matrix_value RS_CI RS_CI 1 2 13

quietly rdsensitivity demvoteshfor2 demmv, statistic(ranksum) wlist(.75 1) tlist(0(5)10) reps(19) seed(12345) nodots nodraw
matrix RSR_results = r(results)
assert_matrix_dims RSR_results RSR_results 3 2
assert_matrix_value RSR_results RSR_results 1 1 0
assert_matrix_value RSR_results RSR_results 1 2 0
assert_matrix_value RSR_results RSR_results 2 1 .1052631578947368
assert_matrix_value RSR_results RSR_results 2 2 0
assert_matrix_value RSR_results RSR_results 3 1 .631578947368421
assert_matrix_value RSR_results RSR_results 3 2 .631578947368421

quietly rdsensitivity demvoteshfor2 demmv, statistic(ksmirnov) wlist(.75 1) tlist(0(5)10) reps(19) seed(12345) nodots nodraw
matrix RSK_results = r(results)
assert_matrix_dims RSK_results RSK_results 3 2
assert_matrix_value RSK_results RSK_results 1 1 0
assert_matrix_value RSK_results RSK_results 1 2 0
assert_matrix_value RSK_results RSK_results 2 1 .4210526315789473
assert_matrix_value RSK_results RSK_results 2 2 .3157894736842105
assert_matrix_value RSK_results RSK_results 3 1 .3157894736842105
assert_matrix_value RSK_results RSK_results 3 2 .2105263157894737

quietly rdrbounds demvoteshfor2 demmv, expgamma(1.5 2) wlist(.5 .75) reps(19) seed(12345)
matrix RB_lbound = r(lbound)
matrix RB_ubound = r(ubound)
matrix RB_pvals = r(pvals)
assert_matrix_dims RB_lbound RB_lbound 2 2
assert_matrix_value RB_lbound RB_lbound 1 1 0
assert_matrix_value RB_lbound RB_lbound 1 2 0
assert_matrix_value RB_lbound RB_lbound 2 1 0
assert_matrix_value RB_lbound RB_lbound 2 2 0
assert_matrix_dims RB_ubound RB_ubound 2 2
assert_matrix_value RB_ubound RB_ubound 1 1 .0526315789473684
assert_matrix_value RB_ubound RB_ubound 1 2 0
assert_matrix_value RB_ubound RB_ubound 2 1 .1578947368421053
assert_matrix_value RB_ubound RB_ubound 2 2 .1578947368421053
assert_matrix_dims RB_pvals RB_pvals 1 2
assert_matrix_value RB_pvals RB_pvals 1 1 0
assert_matrix_value RB_pvals RB_pvals 1 2 0

quietly rdrbounds demvoteshfor2 demmv, expgamma(1.5) wlist(.5 .75) reps(19) seed(12345) bound(upper)
matrix RBU_ubound = r(ubound)
matrix RBU_pvals = r(pvals)
assert_matrix_dims RBU_ubound RBU_ubound 1 2
assert_matrix_value RBU_ubound RBU_ubound 1 1 .0526315789473684
assert_matrix_value RBU_ubound RBU_ubound 1 2 0
assert_matrix_dims RBU_pvals RBU_pvals 1 2
assert_matrix_value RBU_pvals RBU_pvals 1 1 0
assert_matrix_value RBU_pvals RBU_pvals 1 2 0

quietly rdrbounds demvoteshfor2 demmv, expgamma(1.5) wlist(.5 .75) reps(19) seed(12345) bound(lower)
matrix RBL_lbound = r(lbound)
matrix RBL_pvals = r(pvals)
assert_matrix_dims RBL_lbound RBL_lbound 1 2
assert_matrix_value RBL_lbound RBL_lbound 1 1 0
assert_matrix_value RBL_lbound RBL_lbound 1 2 0
assert_matrix_dims RBL_pvals RBL_pvals 1 2
assert_matrix_value RBL_pvals RBL_pvals 1 1 0
assert_matrix_value RBL_pvals RBL_pvals 1 2 0

quietly rdrbounds demvoteshfor2 demmv, expgamma(1.5) wlist(.5 .75) reps(19) seed(12345) fmpval
matrix RBF_lbound = r(lbound)
matrix RBF_ubound = r(ubound)
matrix RBF_pvals = r(pvals)
assert_matrix_dims RBF_lbound RBF_lbound 1 2
assert_matrix_value RBF_lbound RBF_lbound 1 1 0
assert_matrix_value RBF_lbound RBF_lbound 1 2 0
assert_matrix_dims RBF_ubound RBF_ubound 1 2
assert_matrix_value RBF_ubound RBF_ubound 1 1 .0526315789473684
assert_matrix_value RBF_ubound RBF_ubound 1 2 0
assert_matrix_dims RBF_pvals RBF_pvals 2 2
assert_matrix_value RBF_pvals RBF_pvals 1 1 0
assert_matrix_value RBF_pvals RBF_pvals 1 2 0
assert_matrix_value RBF_pvals RBF_pvals 2 1 0
assert_matrix_value RBF_pvals RBF_pvals 2 2 0

display as text "Stata numerical baseline checks passed."
log close
