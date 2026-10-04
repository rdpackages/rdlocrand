*! version 3.0 2026-10-04
*! Internal inference option resolution
program define rdlocrand_inference, rclass
    version 13
    syntax , p(integer) [STATistic(string) VCE(string) UNAvailable(string)]
    if `p'<0 {
        di as error "p must be a nonnegative integer"
        exit 198
    }
    local requested = `p'
    local vce = lower("`vce'")
    if "`vce'"=="" local vce "hc3"
    if !inlist("`vce'","hc1","hc2","hc3") {
        di as error "vce must be hc1, hc2, or hc3"
        exit 198
    }
    if "`unavailable'"=="" & inlist("`statistic'","ksmirnov","ranksum","all","hotelling") local unavailable "statistic=`statistic'"
    if `p'>0 & "`unavailable'"!="" {
        local message "Polynomial adjustment is unavailable for `unavailable'. Requested p=`p'; results were computed with p=0, without polynomial adjustment."
        local p = 0
    }
    return scalar p = `p'
    return scalar p_requested = `requested'
    return local warning "`message'"
    return local vce "`vce'"
end
