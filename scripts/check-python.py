#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "Python" / "rdlocrand" / "pyproject.toml").exists():
            return path
    raise SystemExit("Could not find rdlocrand repository root.")


def run(args: list[str], cwd: Path) -> None:
    print("==>", " ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def check_syntax(src_dir: Path) -> bool:
    ok = True
    for path in sorted(src_dir.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local checks for the rdlocrand Python package.")
    parser.add_argument("--syntax-only", action="store_true", help="Only byte-compile package sources.")
    parser.add_argument("--smoke", action="store_true", help="Run syntax checks and import the public API.")
    parser.add_argument("--tests", action="store_true", help="Run syntax, smoke, and numerical regression tests.")
    parser.add_argument("--build", action="store_true", help="Run syntax, smoke, and source/wheel build checks.")
    parser.add_argument("--no-install", action="store_true", help="Do not install the package before smoke checks.")
    args = parser.parse_args()

    if not (args.syntax_only or args.smoke or args.tests or args.build):
        args.syntax_only = True

    repo_root = find_repo_root(Path.cwd())
    pkg_dir = repo_root / "Python" / "rdlocrand"
    src_dir = pkg_dir / "src"

    print(f"Repository: {repo_root}")
    print(f"Package:    {pkg_dir}")
    print(f"Python:     {sys.executable}")

    if not check_syntax(src_dir):
        return 1
    print("Python syntax checks passed.")

    if args.smoke or args.tests or args.build:
        if not args.no_install:
            run([sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)], cwd=repo_root)

        smoke_expr = (
            "from rdlocrand import rdrandinf, rdwinselect, rdsensitivity, rdrbounds; "
            "print('Python import smoke check passed.')"
        )
        run([sys.executable, "-c", smoke_expr], cwd=repo_root)

    if args.tests or args.build:
        run([sys.executable, "-m", "unittest", "discover", "-s", str(pkg_dir / "tests")], cwd=repo_root)

    if args.build:
        run([sys.executable, "-m", "build", str(pkg_dir)], cwd=repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
