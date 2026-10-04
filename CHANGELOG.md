# Changelog

All notable changes to RDLOCRAND will be recorded here. This changelog starts
with the May 2026 modernization baseline.

## [3.0] - Unreleased

### Fixed

- Restored R initial-window and mass-point indexing broken in 1.1, including
  removal of a hard-coded observation index for supplied asymmetric windows.
- Corrected R KS randomization statistics with tied observations. Corrected
  Python KS rounding so equal observed and simulated statistics remain tied.
- Fixed R approximate polynomial balance tests and weighted balance tests by
  using a numeric treatment indicator.
- Fixed Python supplied asymmetric windows and polynomial balance tests,
  including design dimensions, overwritten covariates, and integer truncation.
- Made R and Python asymptotic sharp-design polynomial tests honor `nulltau`.
- Fixed the documented R list input for fuzzy treatment variables and statistics.
- Corrected nonzero-null TSLS tests and confidence-interval levels and centering
  at `p=0`, retaining the existing variance defaults.

### Changed

- Polynomial mean-contrast inference now uses HC1, HC2, or HC3 standard errors
  (HC3 by default), with large-sample normal tests and confidence intervals in
  R, Python, and Stata. This includes fuzzy Anderson-Rubin and TSLS/Wald tests.
- Polynomial window selection and sensitivity analysis use these large-sample
  p-values. Polynomial permutation p-values are no longer reported.
- Unsupported polynomial requests (KS, rank sum, all statistics, Hotelling,
  interference intervals, and Rosenbaum bounds) use `p=0` with a final warning.
- Corrected singleton and disconnected confidence-set inversion in Python and
  Stata. Added independent numerical tests for HC inference and fallbacks.

## [2.0] - 2026-05-14

Modernization release prepared across May 13-15, 2026.

### Added

- Added fixed-seed numerical regression coverage and local validation scripts
  across the R, Python, and Stata implementations.
- Added repository release hardening, including metadata checks, GitHub Actions
  workflows, Dependabot and security configuration, Python build validation, and
  PyPI trusted publishing support.
- Added Stata runtime, numerical, package, and profiling checks for local
  validation with StataNow 19.

### Changed

- Bumped the R, Python, and Stata packages to version 2.0.
- Modernized package metadata with canonical website and repository URLs,
  GPL-3.0 licensing metadata, current author emails, and alphabetical author
  ordering by last name.
- Refreshed README content, R documentation, Python package documentation,
  Stata help files and PDFs, and the R, Python, and Stata illustration scripts.
- Improved performance-sensitive internals with numerical preservation intended;
  the R KS tie regression subsequently identified is corrected above. This work
  included rank-sum and Kolmogorov-Smirnov helpers, default `rdrbounds()` fast
  paths, `rdwinselect()` balance-loop setup, and Stata temporary-file handling.
- Prepared the repository for the default branch migration from `master` to
  `main` and tightened ignore rules for local-only/generated artifacts.

### Fixed

- Restored documented Python behavior under current dependencies, including lazy
  plotting imports, scoped NumPy RNG restoration, documented return keys,
  default window extraction, Hotelling permutation sampling, and weighted HC2
  covariance access.
- Corrected Python `rdrbounds()` row selection when `fmpval = False` so the
  returned p-value row matches the R implementation.
- Corrected Stata help-file metadata, companion-command links, saved-results
  documentation, package manifest metadata, and package-level website/repository
  entries.

### Validated

- Verified R tests and replication checks, Python syntax/tests/build checks and
  wheel-install smoke checks, and Stata package/runtime/numerical checks.
