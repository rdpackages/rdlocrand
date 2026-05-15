# Local Randomization Methods for RD Designs

## Description

The `rdlocrand` package provides tools to analyze RD designs under local randomization:

- `rdrandinf` to perform hypothesis testing using randomization inference.
- `rdwinselect` to select a window around the cutoff in which randomization is likely to hold.
- `rdsensitivity` to assess the sensitivity of the results to different window lengths and null hypotheses.
- `rdrbounds` to construct Rosenbaum bounds for sensitivity to unobserved confounders.

For more details, and related R, Python, and Stata packages useful for the analysis of RD designs, visit [https://rdpackages.github.io/](https://rdpackages.github.io/).

Source code is available at [https://github.com/rdpackages/rdlocrand](https://github.com/rdpackages/rdlocrand).

## Authors

Matias D. Cattaneo, Princeton University. Email: matias.d.cattaneo@gmail.com

Ricardo Masini, UC Davis. Email: ricardo.masini@gmail.com

Rocio Titiunik, Princeton University. Email: rocio.titiunik@gmail.com

Gonzalo Vazquez-Bare, UC Santa Barbara. Email: gvazquezbare@gmail.com

## Installation

To install or update from PyPI:

```sh
pip install rdlocrand
```

## Usage

```python
from rdlocrand import rdrandinf, rdwinselect, rdsensitivity, rdrbounds

out = rdrandinf(Y, R, wl=-0.75, wr=0.75, quietly=True)
print(out["p.value"])
```

Package functions return dictionaries whose keys match the names documented in
the function docstrings, such as `p.value`, `obs.stat`, `results`, and
`p.values`.

## References

For overviews and introductions, see [rdpackages website](https://rdpackages.github.io).

- Cattaneo, M.D., B. Frandsen, and R. Titiunik. (2015). [Randomization Inference in the Regression Discontinuity Design: An Application to Party Advantages in the U.S. Senate](https://rdpackages.github.io/references/Cattaneo-Frandsen-Titiunik_2015_JCI.pdf). *Journal of Causal Inference* 3(1): 1-24.

- Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2016). [Inference in Regression Discontinuity Designs under Local Randomization](https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf). *Stata Journal* 16(2): 331-367.

- Cattaneo, M.D., R. Titiunik, and G. Vazquez-Bare. (2017). [Comparing Inference Approaches for RD Designs: A Reexamination of the Effect of Head Start on Child Mortality](https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2017_JPAM.pdf). *Journal of Policy Analysis and Management* 36(3): 643-681.

<br><br>
