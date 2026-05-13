version 13
clear all
set more off
set linesize 100

args repo_root mode
if `"`repo_root'"' == "" {
    local repo_root "`c(pwd)'"
}
if `"`mode'"' == "" {
    local mode "full"
}
local repo_root : subinstr local repo_root "\" "/", all
local stata_dir "`repo_root'/stata"
local profile_dir "`repo_root'/tmp/stata-profiles"

capture mkdir "`repo_root'/tmp"
capture mkdir "`profile_dir'"
capture log close _all
log using "`profile_dir'/profile-stata-hotpaths.log", text replace

if "`mode'" == "quick" {
    local ri_reps = 40
    local rdw_reps = 25
    local sensitivity_reps = 20
    local bounds_reps = 20
}
else {
    local ri_reps = 200
    local rdw_reps = 100
    local sensitivity_reps = 70
    local bounds_reps = 70
}

cd "`stata_dir'"
adopath ++ "`stata_dir'"
mata: mata mlib index

use rdlocrand_senate.dta, clear
global profile_covariates presdemvoteshlag1 population demvoteshlag1

tempname profile
file open `profile' using "`profile_dir'/stata-hotpaths.csv", write text replace
file write `profile' "workload,elapsed_seconds,reps,notes" _n

display as text "Repository: `repo_root'"
display as text "Mode:       `mode'"
display as text "Output:     `profile_dir'/stata-hotpaths.csv"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(`ri_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"rdrandinf_diffmeans,`elapsed',`ri_reps',"fixed window diffmeans""' _n
display as text "PROFILE rdrandinf_diffmeans elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) statistic(all) reps(`ri_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"rdrandinf_all,`elapsed',`ri_reps',"fixed window all statistics""' _n
display as text "PROFILE rdrandinf_all elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdwinselect demmv $profile_covariates, cutoff(0) wmin(.5) wstep(.125) nwindows(5) reps(`rdw_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"rdwinselect_balance,`elapsed',`rdw_reps',"three covariates, five windows""' _n
display as text "PROFILE rdwinselect_balance elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdsensitivity demvoteshfor2 demmv, wlist(.75 1 1.25) tlist(0(5)20) reps(`sensitivity_reps') seed(12345) nodots nodraw
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"rdsensitivity_grid,`elapsed',`sensitivity_reps',"three windows, five nulls""' _n
display as text "PROFILE rdsensitivity_grid elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdrbounds demvoteshfor2 demmv, expgamma(1.5 2) wlist(.5 .75) reps(`bounds_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"rdrbounds_both,`elapsed',`bounds_reps',"two gammas, two windows""' _n
display as text "PROFILE rdrbounds_both elapsed=" as result %8.3f `elapsed' as text " seconds"

file close `profile'

display as text "Stata hot-path profiling complete."
log close
