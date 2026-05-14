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
- Added a repository metadata policy requiring author lists to be alphabetical
  by last name and recording current author email addresses.
- Updated Stata package manifest/help-file author email links to the current
  Cattaneo, Titiunik, and Vazquez-Bare addresses.
- Clarified that Ricardo Masini is an author for the Python package only and
  should not be listed in R or Stata package files.

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

### Python

- Added standard-library `unittest` numerical regression coverage for
  `rdwinselect()`, `rdrandinf()`, `rdsensitivity()`, and `rdrbounds()` using
  fixed deterministic data and fixed seeds.
- Extended `scripts/check-python.py` with `--tests` to run syntax, import smoke,
  and numerical regression checks; updated the Python GitHub Actions workflow to
  run the test suite.
- Moved `matplotlib` imports into plotting branches so the package imports
  correctly with declared numerical dependencies when plotting is not requested.
- Added scoped NumPy RNG restoration for seeded Python package calls, preserving
  existing nested reseeding behavior while restoring the caller's RNG state when
  the outermost RDLOCRAND call exits.
- Replaced repeated SciPy rank-sum and KS statistic setup in non-p-value paths
  with private NumPy/SciPy helper calculations that match SciPy's reference
  statistics in regression tests.
- Added a private fast path for the default `rdrbounds()` case
  (`statistic = "ranksum"`, `p = 0`, uniform kernel, no fuzzy design) that
  computes Bernoulli rank-sum p-values without repeating full public-function
  setup.
- Reduced `rdwinselect()` balance-loop setup overhead by computing the
  covariate-complete row mask once after sorting and applying it inside each
  candidate window.
- Restored several documented Python paths that errored under current
  dependencies without changing established regression outputs: lazy plotting
  imports, default window extraction from `rdwinselect()` DataFrame results,
  Hotelling permutation sampling, and HC2 covariance access for weighted paths.
- Updated Python package metadata and README author emails, canonical package
  URLs, GPL-3.0 license metadata, and supported Python version floor.
- Standardized Python author order by last name and updated Ricardo Masini's
  email to ricardo.masini@gmail.com across metadata, README, and function
  docstrings.
- Added `scripts/profile-python-hotpaths.py` for repeatable quick profiling of
  representative Python workloads.
- Audited and corrected the Python-only `rdrbounds()` `p.values` row selection
  when `fmpval = False`: Python now returns the Bernoulli p-value row, matching
  the R implementation, instead of the unused zero-initialized fixed-margins
  row. Added regression coverage using a nonzero p-value case.
- Modernized Python packaging notes and documentation: refreshed `ToBuild.txt`
  as a current build/release checklist, updated the build-system requirement,
  cleaned package metadata with an SPDX-style license identifier and package-level
  license file, documented the Python build check command, and aligned public
  Python docstrings with the R documentation structure and actual returned
  dictionary keys.
- Added Python public API regression coverage for exported functions, leading
  argument names, and representative returned dictionary keys.
- Verified a clean wheel install in a temporary virtual environment and ran
  import plus representative `rdrandinf()`, `rdwinselect()`, `rdsensitivity()`,
  and `rdrbounds()` calls from the installed wheel.
- Validation after the first Python modernization pass:
  `python scripts/check-python.py --syntax-only`,
  `.venv\Scripts\python.exe scripts/check-python.py --tests --no-install`, and
  `.venv\Scripts\python.exe scripts/profile-python-hotpaths.py --quick` all
  passed.

### Stata

- Corrected Stata help-file metadata and saved-results documentation without
  changing `.ado` behavior: updated companion-package wording to include R,
  Python, and Stata; fixed stale companion command links; and aligned documented
  saved-result names with actual `return` values.
- Strengthened `scripts/check-stata-package.py` to reject deprecated package
  URLs, old author emails, Ricardo/Masini Stata author entries, stale companion
  links, and stale saved-result names in distributed Stata text files.
- Added `scripts/check-stata-runtime.py` and `scripts/check-stata-runtime.do`
  to run local Stata smoke checks against the bundled Senate data. The checks
  execute each public Stata command with fixed seeds and small replication
  counts, then validate documented return objects and matrix dimensions.
- Added `scripts/check-stata-numerical.py` and
  `scripts/check-stata-numerical.do` with fixed-seed numerical baselines for
  representative `rdwinselect`, `rdrandinf`, `rdsensitivity`, and `rdrbounds`
  outputs under StataNow 19.
- Added `scripts/profile-stata-hotpaths.py` and
  `scripts/profile-stata-hotpaths.do` for repeatable Stata timer profiling of
  representative workloads, with results written to `tmp/stata-profiles/`.
- Full StataNow 19 profiling before `.ado` refactoring identified
  `rdrbounds_both` as the main target: about 31.75 seconds for the profiling
  workload, versus about 1.12 seconds for `rdrandinf_diffmeans`, 0.98 seconds
  for `rdsensitivity_grid`, 0.66 seconds for `rdrandinf_all`, and 0.49 seconds
  for `rdwinselect_balance`.
- Added a private fast path in `stata/rdrbounds.ado` for the default
  rank-sum, no-polynomial, uniform-kernel, non-fuzzy `bound(both)` sensitivity
  branch. The helper computes Bernoulli rank-sum p-values directly in Mata
  while preserving the existing per-call seed behavior and falling back to
  `rdrandinf` for other options.
- After the fast path, the full StataNow 19 profiling workload reduced
  `rdrbounds_both` from about 31.75 seconds to about 0.79 seconds, with the
  fixed-seed Stata numerical baseline unchanged.
- Expanded the Stata numerical baseline to cover non-default
  `rdsensitivity` rank-sum and Kolmogorov-Smirnov paths, plus `rdrbounds`
  `bound(upper)`, `bound(lower)`, and `fmpval` branches before further Stata
  refactoring.
- Cleaned `stata/rdsensitivity.ado` setup code without changing numerical
  behavior: construct the treatment indicator once, expand `wlist`,
  `wlist_left`, and `tlist` once, and reuse those expanded lists in later
  loops.
- Expanded Stata `rdrandinf` numerical baselines for rank-sum,
  Kolmogorov-Smirnov, Bernoulli, polynomial-adjusted, and simple fuzzy paths.
- Added `scripts/profile-stata-rdrandinf.py` and
  `scripts/profile-stata-rdrandinf.do` for focused Stata profiling of
  `rdrandinf` fixed-margins, Bernoulli, statistic, and adjustment paths.
  Full StataNow 19 profiling identified fixed-margins `diffmeans` as the
  main remaining `rdrandinf` outlier at about 1.25 seconds for 200
  replications.
- Updated `stata/rdrandinf.ado` to skip writing the fixed-margins permutation
  dataset unless `interfci()` is requested. This preserves Stata's `permute`
  engine and fixed-seed p-values while avoiding unnecessary temporary-file
  writes in the common path; the focused `fixed_diffmeans` profile moved from
  about 1.25 seconds to about 1.15 seconds.
- Validation after the Stata help/static-check/runtime/numerical pass:
  `python scripts/check-stata-package.py --strict-unlisted`,
  `python scripts/check-stata-runtime.py`, and
  `python scripts/check-stata-numerical.py` all passed with StataNow 19.
- Polished Stata package distribution metadata by adding canonical website and
  repository lines to `stata/rdlocrand.pkg` and `stata/stata.toc`, extending the
  static package checker to enforce those package-level URLs, and updating the
  illustration do-file to use a temporary dataset for contour-plot replication.
- Added repository-level release hardening: a metadata checker for canonical
  URLs, author order, emails, and Ricardo's Python-only authorship; a
  cross-package `scripts/check-all.py` validation driver; a dedicated
  repository metadata workflow; stricter Stata CI; Python CI build coverage;
  `twine check` validation for built Python distributions; and Python
  3.13/3.14 classifiers and CI entries.
- Bumped R, Python, and Stata package versions to 2.0 for the modernization
  release and updated Stata package/update dates to 2026-05-13. Updated the
  Python build helper to clear stale generated build artifacts before creating
  source and wheel distributions.
- Prepared the repository for the default-branch migration from `master` to
  `main`: updated Stata raw-install instructions, restricted CI triggers to
  `main`, added repository security/dependabot configuration, removed the
  redundant Python release checklist in favor of `CONTRIBUTING.md`, and
  tightened ignore rules for local-only artifacts.
- Updated GitHub repository settings with GitHub CLI: enabled Issues, disabled
  Projects and Wiki, set the repository homepage to
  `https://rdpackages.github.io/`, enabled delete-branch-on-merge, and enabled
  Dependabot vulnerability alerts and automated security fixes.
