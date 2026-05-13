# Repository Instructions

This repository contains the `rdlocrand` package in R, Python, and Stata. The
three implementations should remain behaviorally aligned where they expose the
same statistical routines.

## Package Roots

Use these roots unless the task is explicitly about repository-level files:

- R: `R/rdlocrand`
- Python: `Python/rdlocrand`
- Stata: `stata`

Important package paths:

- `R/rdlocrand/R/`: R source code.
- `R/rdlocrand/man/`: generated Rd documentation.
- `R/rdlocrand/DESCRIPTION`: R package metadata and dependencies.
- `Python/rdlocrand/src/rdlocrand/`: Python source code.
- `Python/rdlocrand/setup.cfg`: Python package metadata and dependencies.
- `stata/rdlocrand.pkg`: Stata package manifest.
- `stata/*.ado`, `stata/*.sthlp`, and `stata/*.mo`: Stata implementation and help files.

## Canonical URLs

Use only these package-level URLs in descriptions, manuals, package metadata,
and help files:

- Frontend website: https://rdpackages.github.io/
- GitHub repository: https://github.com/rdpackages/rdlocrand

Do not use https://rdpackages.github.io/rdlocrand/; it is deprecated and now
redirects.

## Metadata Standards

List package authors in alphabetical order by last name in package metadata,
manuals, help files, READMEs, and illustration files. Use these email
addresses when authors are listed:

- Matias D. Cattaneo: matias.d.cattaneo@gmail.com
- Ricardo Masini: ricardo.masini@gmail.com
- Rocio Titiunik: rocio.titiunik@gmail.com
- Gonzalo Vazquez-Bare: gvazquezbare@gmail.com

Ricardo Masini should appear as an author only for the Python package; do not
add him to R or Stata package metadata, documentation, or help files.

## Validation Repositories:

The following repositories contain data and codes using the package:

- C:\Users\cattaneo\Dropbox\software\rdpackages-replication\CFT_2015_JCI
- C:\Users\cattaneo\Dropbox\software\rdpackages-replication\CTV_2017_JPAM
- C:\Users\cattaneo\Dropbox\software\rdpackages-replication\CIT_2024_CUP

Make sure all numerical results remain unchanged unless explicitly approved by maintainer.


## Development Commands

From the repository root:

```sh
R CMD check --no-manual R/rdlocrand
Rscript scripts/check-r-replication-baseline.R
Rscript scripts/profile-r-hotpaths.R --quick
python scripts/check-python.py --syntax-only
python scripts/check-python.py --tests --no-install
python scripts/check-python.py --build --no-install
python scripts/profile-python-hotpaths.py --quick
python scripts/check-stata-package.py --strict-unlisted
python scripts/check-stata-runtime.py
python scripts/check-stata-numerical.py
python scripts/profile-stata-hotpaths.py --quick
```

For the R GitHub Actions equivalent:

```sh
R CMD check --no-manual --as-cran R/rdlocrand
```

For a Python import smoke check after dependencies are installed:

```sh
python -m pip install -e Python/rdlocrand
python scripts/check-python.py --smoke --no-install
```

When roxygen comments or exported R functions change, regenerate package docs
from the R package root:

```sh
cd R/rdlocrand
Rscript -e "roxygen2::roxygenise()"
```

## R Library Policy

Install R package dependencies into the user's home R library, not into this
repository. Do not create or use repo-local libraries such as `.r-lib/`,
`renv/`, or `packrat/` unless explicitly requested. If dependencies are
missing, use the standard user library shown by `.libPaths()` or
`Sys.getenv("R_LIBS_USER")`, and ask before installing packages.

## Python Environment Policy

Use an external or ignored virtual environment for Python work, such as
`.venv/`. Do not commit `dist/`, `build/`, `*.egg-info/`, or `__pycache__/`
outputs. If changing package dependencies, update `Python/rdlocrand/setup.cfg`
and verify that the supported Python version range remains coherent.

## Stata Package Policy

When adding, renaming, or removing distributed Stata files, update
`stata/rdlocrand.pkg` and run:

```sh
python scripts/check-stata-package.py --strict-unlisted
```

Do not assume Stata is available in CI. Repository checks validate the package
manifest; run functional Stata checks locally when changing `.ado` behavior:

```sh
python scripts/check-stata-runtime.py
python scripts/check-stata-numerical.py
```

On this workstation, StataNow 19 is installed at
`C:\Program Files\StataNow19\StataMP-64.exe`; the runtime check auto-detects
that path. Runtime and numerical checks also accept `--stata` or `STATA_EXE` if
Stata is installed elsewhere.

For local Stata profiling before and after performance refactors, run:

```sh
python scripts/profile-stata-hotpaths.py --quick
```

## Editing Guidelines

- Keep changes scoped to the requested behavior and match the existing style.
- Prefer existing helper functions and dependencies unless a new dependency is
  clearly required and package metadata is updated.
- Add or update tests for user-facing behavior, numerical regressions, exported
  API changes, and bug fixes.
- Do not edit R `NAMESPACE` or files under `R/rdlocrand/man/` by hand when the
  change belongs in roxygen comments; regenerate them instead.
- Do not commit generated package bundles, check directories, bytecode caches,
  local IDE state, or built wheels.
- Preserve CRAN-oriented behavior: avoid noisy startup output, unguarded
  internet access, and examples/tests that are unnecessarily slow or flaky.

## Git Safety

The working tree may contain user edits. Do not revert changes you did not
make. Before broad edits, inspect the relevant files and preserve unrelated
local changes.
