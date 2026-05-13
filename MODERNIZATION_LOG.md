# RDLOCRAND Modernization Log

This log records modernization work across the R, Python, and Stata
implementations. Numerical results are expected to remain unchanged unless a
maintainer explicitly approves a substantive change.

## 2026-05-13

### Repository

- Added repository-level development scaffolding, GitHub workflows, local check
  scripts, and ignore rules for generated artifacts.
- Removed generated/local artifacts from Git tracking while leaving local files
  on disk.
- Added the canonical package URL policy to `AGENTS.md` and replaced deprecated
  `https://rdpackages.github.io/rdlocrand/` package-page links with
  `https://rdpackages.github.io/` plus the GitHub repository URL where
  appropriate in package metadata and help files.

### R

- Updated R package maintainer to Matias D. Cattaneo
  <matias.d.cattaneo@gmail.com>.
- Updated R package author emails for Rocio Titiunik and Gonzalo Vazquez-Bare.
- Added `testthat` infrastructure and fixed-seed numerical regression tests for
  `rdwinselect()`, `rdrandinf()`, `rdsensitivity()`, and `rdrbounds()`.
- Added `scripts/check-r-replication-baseline.R` to compare selected
  fixed-seed calls against external replication data.
- Improved internal allocation behavior without changing numerical formulas:
  preallocated Kolmogorov-Smirnov and rank-sum statistic vectors in
  `rdrandinf.model()`, and replaced repeated p-value vector growth in
  `rdrbounds()` sensitivity loops with preallocated list storage.
- Improved internal window helper allocation without changing selection logic:
  preallocated candidate-window storage in `findwobs()` and `findwobs_sym()`,
  and preallocated deprecated `findstep()` increments while preserving the
  existing `1:times` iteration behavior.
- Replaced iterative `rbind()` growth in `find_CI()` with a direct contiguous
  interval-index calculation and added regression coverage for split,
  all-accepted, and all-rejected confidence-interval cases.
- Expanded fixed-seed R numerical regression coverage from 18 to 46 checks,
  including Bernoulli assignment, interference confidence intervals,
  confidence interval inversion, fuzzy ITT/default AR behavior, polynomial
  covariate adjustment in `rdwinselect()`, asymmetric windows, mass-point
  windows, `rdsensitivity()` confidence intervals, and one-sided `rdrbounds()`
  outputs.
- Added `scripts/profile-r-hotpaths.R`, a base-R profiling helper for
  representative `rdrandinf()`, `rdwinselect()`, `rdsensitivity()`, and
  `rdrbounds()` workloads; documented the command in `CONTRIBUTING.md` and
  `AGENTS.md`.
- Used profiling to identify `wilcox.test()` rank-sum statistic extraction as
  the main avoidable cost in `statistic = "ranksum"` workloads. Replaced that
  call with the equivalent direct sum of control ranks in `rdrandinf.model()`,
  preserving the existing `statistic = "all"` standardization path.
- In the profiling workload, `rdrbounds_both` elapsed time dropped from about
  0.80 seconds to about 0.17 seconds after the rank-sum refactor.
- Added a private fast path for the default `rdrbounds()` case
  (`statistic = "ranksum"`, `p = 0`, uniform kernel, no fuzzy design) that
  reuses rank information and reproduces the Bernoulli p-value logic from
  `rdrandinf()` without repeating full public-function setup. Added a
  regression check comparing the helper directly against `rdrandinf()`.
- In the same profiling workload, `rdrbounds_both` elapsed time dropped further
  from about 0.17 seconds to about 0.05 seconds after the private Bernoulli
  rank-sum fast path.
- Added a private Kolmogorov-Smirnov statistic helper for permutation paths
  where only the statistic is needed. The helper matches `ks.test()`'s
  weighted cumulative-sum statistic, including tied values, while preserving
  existing `ks.test()` calls whenever asymptotic p-values are required.
- Expanded R regression coverage to 51 fixed-seed checks, including direct
  helper comparisons against `ks.test()` and the existing `rdrandinf()` path.
- Reduced `rdwinselect()` balance-loop setup overhead by computing the
  covariate-complete row mask once after sorting and applying it inside each
  window, instead of rebuilding a temporary data frame per window. Added
  regression coverage for missing-covariate filtering under both
  `dropmissing = FALSE` and `dropmissing = TRUE`.
- In the profiling workload, `rdwinselect_balance` elapsed time dropped from
  about 0.09 seconds to about 0.05 seconds after the balance-loop setup
  refactor.
- Expanded R regression coverage to 57 fixed-seed checks.
- Completed a final R documentation pass: updated roxygen return sections to
  match actual list elements, corrected stale example code, made examples
  deterministic, regenerated `man/` files and `NAMESPACE`, removed the stale
  `wilcox.test()` import, and updated `R/rdlocrand_illustration.R` to use
  `nwindows` explicitly and locate `rdlocrand_senate.csv` from the repository
  root or `R/`.
- Added a private scoped RNG helper so seeded R package calls preserve the
  existing nested reseeding behavior while restoring the caller's `.Random.seed`
  when the outermost RDLOCRAND call exits.
- Consolidated safe option validation for public R arguments such as statistic,
  kernel, evaluation point, and Rosenbaum-bound direction, improving invalid
  input errors without changing valid numerical outputs.
- Expanded R regression coverage to 65 checks, including RNG-state restoration
  for `rdwinselect()`, `rdrandinf()`, `rdsensitivity()`, `rdrbounds()`, support
  for `seed = -1`, and direct invalid-option validation errors.
- Validation after the R refactors, speed work, and coverage expansion:
  `testthat::test_local("R/rdlocrand")`, `Rscript scripts/check-r-replication-baseline.R`,
  and `Rscript scripts/check-local.R --pre-push` all passed.
