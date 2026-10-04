*! version 3.0 2026-10-04
*! Internal weighted polynomial OLS/2SLS HC calculation
program define rdlocrand_hc, rclass
    version 13
    syntax varlist(min=3 max=3 numeric) [if] [in] [aw], p(integer) ///
        [VCE(string) Cutoff(real 0) EVALL(real 0) EVALR(real 0) Treatment(varname) FIRSTstage]
    marksample touse
    tokenize `varlist'
    local y `1'
    local r `2'
    local d `3'
    if "`treatment'"!="" markout `touse' `treatment'
    if "`weight'"!="" local weights "[`weight'`exp']"
    local vcetype "`vce'"
    if "`vce'"=="hc1" local vcetype "robust"
    tempvar centered
    qui gen double `centered' = `r'-`d'*`evalr'-(1-`d')*`evall' if `touse'
    local nuisance ""
    forvalues j=1/`p' {
        tempvar power interaction
        qui gen double `power' = `centered'^`j' if `touse'
        qui gen double `interaction' = `d'*`power' if `touse'
        local nuisance "`nuisance' `power' `interaction'"
    }
    local target `d'
    local response `y'
    if "`treatment'"!="" {
        tempvar projected pseudo
        qui reg `treatment' `d' `nuisance' if `touse' `weights'
        if "`firststage'"!="" est store first_stage, title("First stage regression")
        qui predict double `projected' if e(sample), xb
        qui reg `y' `projected' `nuisance' if `touse' `weights'
        local ivbeta = _b[`projected']
        // Structural residuals with second-stage projected-regressor leverage.
        qui gen double `pseudo' = `y'-`ivbeta'*(`treatment'-`projected') if `touse'
        local target `projected'
        local response `pseudo'
    }
    qui reg `response' `target' `nuisance' if `touse' `weights'
    if "`vce'"!="hc1" {
        tempvar h
        qui predict double `h' if e(sample), leverage
        // predict reports unweighted leverage even after analytic-weight fits.
        if "`weight'"!="" {
            tempvar w
            qui gen double `w' `exp' if e(sample)
            qui sum `w' if e(sample), meanonly
            qui replace `h' = `h'*`w'/r(mean) if e(sample)
        }
        qui sum `h', meanonly
        if r(max)>=1-1e-10 {
            di as error "HC2/HC3 is undefined for a polynomial fit with unit leverage."
            exit 498
        }
    }
    qui reg `response' `target' `nuisance' if `touse' `weights', vce(`vcetype')
    if e(rank)<2*`p'+2 | e(N)<=2*`p'+2 {
        di as error "Polynomial regression needs an identified model and more observations and distinct scores than fitted coefficients."
        exit 498
    }
    return scalar estimate = _b[`target']
    return scalar se = _se[`target']
end
