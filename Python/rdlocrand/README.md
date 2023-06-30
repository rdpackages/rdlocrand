# rdlocrand: Local Randomization Methods for RD Designs

## Description
The regression discontinuity (RD) design is a popular quasi-experimental design for causal inference and policy evaluation. Under the local randomization approach, RD designs can be interpreted as randomized experiments inside a window around the cutoff. The `rdlocrand` package provides tools to analyze RD designs under local randomization:

- `rdrandinf` to perform hypothesis testing using randomization inference.
- `rdwinselect` to select a window around the cutoff in which randomization is likely to hold.
- `rdsensitivity` to assess the sensitivity of the results to different window lengths and null hypotheses.
- `rdrbounds` to construct Rosenbaum bounds for sensitivity to unobserved confounders.

For more details, and related Stata and R packages useful for the analysis of RD designs, visit [https://rdpackages.github.io/](https://rdpackages.github.io/).

## References
1. Cattaneo, M.D., B. Frandsen, and R. Titiunik. (2015). [Randomization Inference in the Regression Discontinuity Design: An Application to Party Advantages in the U.S. Senate](https://rdpackages.github.io/references/Cattaneo-Frandsen-Titiunik_2015_JCI.pdf). *Journal of Causal Inference* 3(1): 1-24.
2. Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2016). [Inference in Regression Discontinuity Designs under Local Randomization](https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf). *Stata Journal* 16(2): 331-367.
3. Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2017). [Comparing Inference Approaches for RD Designs: A Reexamination of the Effect of Head Start on Child Mortality](https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2017_JPAM.pdf). *Journal of Policy Analysis and Management* 36(3): 643-681.
4. Rosenbaum, P. (2002). *Observational Studies*. Springer.

## Author
Matias Cattaneo, Princeton University. Email: cattaneo@princeton.edu
Rocio Titiunik, Princeton University. Email: titiunik@princeton.edu
Ricardo Masini, UC Davis. Email: rmasini@ucdavis.edu
Gonzalo Vazquez-Bare, UC Santa Barbara. Email: gvazquez@econ.ucsb.edu
