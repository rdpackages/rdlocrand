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
log using "`profile_dir'/profile-stata-rdrandinf.log", text replace

if "`mode'" == "quick" {
    local ri_reps = 40
    local p_reps = 25
}
else {
    local ri_reps = 200
    local p_reps = 100
}

cd "`stata_dir'"
adopath ++ "`stata_dir'"
mata: mata mlib index

use rdlocrand_senate.dta, clear
generate double ri_probs = .5 if demmv >= -.75 & demmv <= .75 & demvoteshfor2 != .

tempname profile
file open `profile' using "`profile_dir'/stata-rdrandinf.csv", write text replace
file write `profile' "workload,elapsed_seconds,reps,notes" _n

display as text "Repository: `repo_root'"
display as text "Mode:       `mode'"
display as text "Output:     `profile_dir'/stata-rdrandinf.csv"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) reps(`ri_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"fixed_diffmeans,`elapsed',`ri_reps',"fixed margins diffmeans""' _n
display as text "PROFILE fixed_diffmeans elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) statistic(ranksum) reps(`ri_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"fixed_ranksum,`elapsed',`ri_reps',"fixed margins ranksum""' _n
display as text "PROFILE fixed_ranksum elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) statistic(ksmirnov) reps(`ri_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"fixed_ksmirnov,`elapsed',`ri_reps',"fixed margins ksmirnov""' _n
display as text "PROFILE fixed_ksmirnov elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) statistic(all) reps(`ri_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"fixed_all,`elapsed',`ri_reps',"fixed margins all statistics""' _n
display as text "PROFILE fixed_all elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) bernoulli(ri_probs) reps(`ri_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"bernoulli_diffmeans,`elapsed',`ri_reps',"bernoulli diffmeans""' _n
display as text "PROFILE bernoulli_diffmeans elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) bernoulli(ri_probs) statistic(ranksum) reps(`ri_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"bernoulli_ranksum,`elapsed',`ri_reps',"bernoulli ranksum""' _n
display as text "PROFILE bernoulli_ranksum elapsed=" as result %8.3f `elapsed' as text " seconds"

timer clear 1
timer on 1
quietly rdrandinf demvoteshfor2 demmv, wl(-.75) wr(.75) p(1) reps(`p_reps') seed(12345)
timer off 1
quietly timer list 1
local elapsed = r(t1)
file write `profile' `"p1_diffmeans,`elapsed',`p_reps',"linear adjustment diffmeans""' _n
display as text "PROFILE p1_diffmeans elapsed=" as result %8.3f `elapsed' as text " seconds"

file close `profile'

display as text "Focused rdrandinf profiling complete."
log close
