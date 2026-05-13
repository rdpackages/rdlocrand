# Contributing

This repository contains the `rdlocrand` package in R, Python, and Stata.
Keep changes scoped to the implementation you are editing, and update the
shared documentation when behavior changes across implementations.

## Repository Layout

- `R/rdlocrand/`: R package source submitted to CRAN.
- `Python/rdlocrand/`: Python package source distributed on PyPI.
- `stata/`: Stata package files used by `net install`.
- `R/`, `Python/`, and `stata/` also contain illustration scripts and data.

## Local Checks

For R package checks from the repository root:

```sh
Rscript scripts/check-local.R --dev
```

For a release-style R check matching the CRAN gate:

```sh
Rscript scripts/check-local.R --release
```

For Python syntax checks:

```sh
python scripts/check-python.py --syntax-only
```

For a Python import smoke check, first install the package dependencies in your
active environment:

```sh
python -m pip install -e Python/rdlocrand
python scripts/check-python.py --tests --no-install
python scripts/profile-python-hotpaths.py --quick
```

For a Python source/wheel build check, install `build` in the active
environment and run:

```sh
python scripts/check-python.py --build --no-install
```

For Stata package manifest checks:

```sh
python scripts/check-stata-package.py --strict-unlisted
```

For optional local Stata runtime smoke checks, which require Stata:

```sh
python scripts/check-stata-runtime.py
```

For local Stata numerical regression checks, which compare fixed-seed outputs
against the current baseline:

```sh
python scripts/check-stata-numerical.py
```

For local Stata performance profiling:

```sh
python scripts/profile-stata-hotpaths.py --quick
```

For focused Stata profiling of `rdrandinf` paths:

```sh
python scripts/profile-stata-rdrandinf.py --quick
```

For the optional local R replication baseline, which requires the external
replication folders listed in `AGENTS.md`:

```sh
Rscript scripts/check-r-replication-baseline.R
```

For local R performance profiling of representative hot paths:

```sh
Rscript scripts/profile-r-hotpaths.R
```

Pass `--quick` for a smoke run or `--keep` to save raw `Rprof` traces under
`tmp/r-profiles/`.

To enable the optional pre-push hook for future local development:

```sh
git config core.hooksPath .githooks
```

The hook runs the R pre-push check, Python syntax checks, and Stata manifest
checks. Set `RDLOCRAND_SKIP_PRE_PUSH=1` only for emergency pushes where the
relevant checks have already been run separately.

## Artifact Policy

Generated package bundles, check directories, bytecode caches, local IDE state,
and built wheels should not be committed. Use GitHub releases or package
registries for archived release artifacts when needed.
