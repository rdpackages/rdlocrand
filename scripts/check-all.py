#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "AGENTS.md").exists() and (path / "scripts").exists():
            return path
    raise SystemExit("Could not find rdlocrand repository root.")


def default_python(repo_root: Path) -> str:
    for envvar in ("RDLOCRAND_PYTHON", "PYTHON"):
        value = os.environ.get(envvar)
        if value:
            return value
    candidates = [
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run(args: list[str], repo_root: Path) -> None:
    print("\n==>", " ".join(str(arg) for arg in args), flush=True)
    subprocess.run([str(arg) for arg in args], cwd=str(repo_root), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository-level rdlocrand validation checks.")
    parser.add_argument("--quick", action="store_true", help="Run the pre-push/static subset.")
    parser.add_argument("--release", action="store_true", help="Run release-oriented R and Python build checks.")
    parser.add_argument("--python", default=None, help="Python executable for Python package checks.")
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT", "Rscript"), help="Rscript executable.")
    parser.add_argument("--no-python-install", action="store_true", help="Do not install the Python package before checks.")
    parser.add_argument("--skip-r", action="store_true", help="Skip R package checks.")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python package checks.")
    parser.add_argument("--skip-stata", action="store_true", help="Skip all Stata checks.")
    parser.add_argument("--skip-stata-runtime", action="store_true", help="Skip Stata runtime smoke checks.")
    parser.add_argument("--skip-stata-numerical", action="store_true", help="Skip Stata fixed-seed numerical checks.")
    parser.add_argument("--include-r-replication", action="store_true", help="Run external R replication baselines.")
    parser.add_argument("--include-profiles", action="store_true", help="Run quick R, Python, and Stata profiling checks.")
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())
    py = args.python or default_python(repo_root)

    print(f"Repository: {repo_root}")
    print(f"Python:     {py}")
    print(f"Rscript:    {args.rscript}")

    run([sys.executable, "scripts/check-repo-metadata.py"], repo_root)

    if not args.skip_r:
        r_mode = "--pre-push" if args.quick else ("--release" if args.release else "--dev")
        run([args.rscript, "scripts/check-local.R", r_mode], repo_root)
        if args.include_r_replication:
            run([args.rscript, "scripts/check-r-replication-baseline.R"], repo_root)

    if not args.skip_python:
        if args.quick:
            py_args = ["--syntax-only"]
        elif args.release:
            py_args = ["--build"]
        else:
            py_args = ["--tests"]
        if args.no_python_install or args.quick:
            py_args.append("--no-install")
        run([py, "scripts/check-python.py", *py_args], repo_root)

    if not args.skip_stata:
        run([py, "scripts/check-stata-package.py", "--strict-unlisted"], repo_root)
        if not args.quick and not args.skip_stata_runtime:
            run([py, "scripts/check-stata-runtime.py"], repo_root)
        if not args.quick and not args.skip_stata_numerical:
            run([py, "scripts/check-stata-numerical.py"], repo_root)

    if args.include_profiles:
        if not args.skip_r:
            run([args.rscript, "scripts/profile-r-hotpaths.R", "--quick"], repo_root)
        if not args.skip_python:
            run([py, "scripts/profile-python-hotpaths.py", "--quick"], repo_root)
        if not args.skip_stata:
            run([py, "scripts/profile-stata-hotpaths.py", "--quick"], repo_root)
            run([py, "scripts/profile-stata-rdrandinf.py", "--quick"], repo_root)

    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
