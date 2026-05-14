# Local Randomization Methods for RD Designs

The `rdlocrand` package provides Python, R, and Stata estimation, inference, and visualization methods for Regression Discontinuity (RD) designs employing local randomization methods.


## Python Implementation

To install/update in Python type:
```
pip install rdlocrand
```

- Help: [PyPI repository](https://pypi.org/project/rdlocrand/).

- Replication: [py-script](Python/rdlocrand_illustration.py), [senate data](Python/rdlocrand_senate.csv).

## R Implementation

To install/update in R type:
```
install.packages('rdlocrand')
```

- Help: [R Manual](https://cran.r-project.org/web/packages/rdlocrand/rdlocrand.pdf), [CRAN repository](https://cran.r-project.org/package=rdlocrand).

- Replication: [R-script](R/rdlocrand_illustration.R), [senate data](R/rdlocrand_senate.csv).

## Stata Implementation

To install/update in Stata type:
```
net install rdlocrand, from(https://raw.githubusercontent.com/rdpackages/rdlocrand/main/stata) replace
```

- Help: [rdrandinf](stata/rdrandinf.pdf), [rdwinselect](stata/rdwinselect.pdf), [rdsensitivity](stata/rdsensitivity.pdf), [rdrbounds](stata/rdrbounds.pdf).

- Replication: [do-file](stata/rdlocrand_illustration.do), [senate data](stata/rdlocrand_senate.dta).


## References

For overviews and introductions, see [rdpackages website](https://rdpackages.github.io). Source code is available at [https://github.com/rdpackages/rdlocrand](https://github.com/rdpackages/rdlocrand).

### Software and Implementation

- Cattaneo, Titiunik and Vazquez-Bare (2016): [Inference in Regression Discontinuity Designs under Local Randomization](https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf).<br>
_Stata Journal_ 16(2): 331-367.

### Technical and Methodological

- Cattaneo, Frandsen and Titiunik (2015): [Randomization Inference in the Regression Discontinuity Design: An Application to Party Advantages in the U.S. Senate](https://rdpackages.github.io/references/Cattaneo-Frandsen-Titiunik_2015_JCI.pdf).<br>
_Journal of Causal Inference_ 3(1): 1-24.

- Cattaneo, Titiunik and Vazquez-Bare (2017): [Comparing Inference Approaches for RD Designs: A Reexamination of the Effect of Head Start on Child Mortality](https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2017_JPAM.pdf).<br>
_Journal of Policy Analysis and Management_ 36(3): 643-681.

## Funding

This work was supported in part by the National Science Foundation through grants [SES-1357561](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1357561).


<br><br>
