#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


FRONTEND_URL = "https://rdpackages.github.io/"
REPO_URL = "https://github.com/rdpackages/rdlocrand"
BUG_URL = "https://github.com/rdpackages/rdlocrand/issues"

FORBIDDEN_TEXT = {
    "deprecated package URL": "https://rdpackages.github.io/rdlocrand/",
    "old Matias email": "cattaneo@princeton.edu",
    "old Rocio email": "titiunik@princeton.edu",
    "old Gonzalo email": "gvazquez@econ.ucsb.edu",
    "old Ricardo email": "rmasini@ucdavis.edu",
    "old Stata install branch": "raw.githubusercontent.com/rdpackages/rdlocrand/master/",
}

TEXT_SUFFIXES = {".R", ".Rd", ".ado", ".cfg", ".do", ".md", ".pkg", ".py", ".sthlp", ".toc", ".toml", ".txt", ".yml", ".yaml"}


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "AGENTS.md").exists() and (path / "R" / "rdlocrand").exists():
            return path
    raise SystemExit("Could not find rdlocrand repository root.")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def ensure_contains(errors: list[str], path: Path, text: str, label: str, pattern: str) -> None:
    if pattern not in text:
        errors.append(f"{path.as_posix()} is missing {label}: {pattern}")


def ensure_order(errors: list[str], path: Path, text: str, labels: list[str]) -> None:
    positions = [text.find(label) for label in labels]
    missing = [label for label, pos in zip(labels, positions) if pos < 0]
    if missing:
        errors.append(f"{path.as_posix()} is missing expected author text: {', '.join(missing)}")
    elif positions != sorted(positions):
        errors.append(f"{path.as_posix()} does not list authors alphabetically by last name.")


def iter_release_text_files(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "R" / "rdlocrand",
        repo_root / "Python" / "rdlocrand",
        repo_root / "stata",
        repo_root / ".github" / "workflows",
    ]
    files = [repo_root / "README.md", repo_root / "CONTRIBUTING.md", repo_root / "SECURITY.md"]
    for root in roots:
        if not root.exists():
            continue
        files.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix in TEXT_SUFFIXES
        )
    return sorted(set(files))


def main() -> int:
    repo_root = find_repo_root(Path.cwd())
    errors: list[str] = []

    for path in iter_release_text_files(repo_root):
        text = read(path)
        rel = path.relative_to(repo_root)
        for label, pattern in FORBIDDEN_TEXT.items():
            if pattern in text:
                errors.append(f"{rel.as_posix()} contains {label}: {pattern}")

    root_readme = repo_root / "README.md"
    root_text = read(root_readme)
    ensure_contains(errors, root_readme.relative_to(repo_root), root_text, "frontend URL", FRONTEND_URL)
    ensure_contains(errors, root_readme.relative_to(repo_root), root_text, "repository URL", REPO_URL)
    ensure_contains(
        errors,
        root_readme.relative_to(repo_root),
        root_text,
        "main-branch Stata install URL",
        "raw.githubusercontent.com/rdpackages/rdlocrand/main/stata",
    )

    for required in (repo_root / "SECURITY.md", repo_root / ".github" / "dependabot.yml"):
        if not required.exists():
            errors.append(f"{required.relative_to(repo_root).as_posix()} is missing.")

    if (repo_root / "Python" / "rdlocrand" / "ToBuild.txt").exists():
        errors.append("Python/rdlocrand/ToBuild.txt is obsolete; use CONTRIBUTING.md instead.")

    r_desc = repo_root / "R" / "rdlocrand" / "DESCRIPTION"
    r_text = read(r_desc)
    ensure_contains(errors, r_desc.relative_to(repo_root), r_text, "frontend URL", FRONTEND_URL)
    ensure_contains(errors, r_desc.relative_to(repo_root), r_text, "repository URL", REPO_URL)
    ensure_contains(errors, r_desc.relative_to(repo_root), r_text, "bug reports URL", BUG_URL)
    ensure_contains(errors, r_desc.relative_to(repo_root), r_text, "Matias maintainer email", "matias.d.cattaneo@gmail.com")
    ensure_order(errors, r_desc.relative_to(repo_root), r_text, ["family = \"Cattaneo\"", "family = \"Titiunik\"", "family = \"Vazquez-Bare\""])
    if "Ricardo" in r_text or "Masini" in r_text:
        errors.append("R/rdlocrand/DESCRIPTION lists Ricardo Masini, who is Python-only.")

    py_cfg = repo_root / "Python" / "rdlocrand" / "setup.cfg"
    py_text = read(py_cfg)
    ensure_contains(errors, py_cfg.relative_to(repo_root), py_text, "frontend URL", FRONTEND_URL)
    ensure_contains(errors, py_cfg.relative_to(repo_root), py_text, "repository URL", REPO_URL)
    ensure_contains(errors, py_cfg.relative_to(repo_root), py_text, "bug tracker URL", BUG_URL)
    ensure_contains(errors, py_cfg.relative_to(repo_root), py_text, "Ricardo email", "ricardo.masini@gmail.com")
    ensure_order(errors, py_cfg.relative_to(repo_root), py_text, ["Matias D. Cattaneo", "Ricardo Masini", "Rocio Titiunik", "Gonzalo Vazquez-Bare"])

    py_readme = repo_root / "Python" / "rdlocrand" / "README.md"
    py_readme_text = read(py_readme)
    ensure_contains(errors, py_readme.relative_to(repo_root), py_readme_text, "frontend URL", FRONTEND_URL)
    ensure_contains(errors, py_readme.relative_to(repo_root), py_readme_text, "repository URL", REPO_URL)
    ensure_order(errors, py_readme.relative_to(repo_root), py_readme_text, ["Matias D. Cattaneo", "Ricardo Masini", "Rocio Titiunik", "Gonzalo Vazquez-Bare"])

    stata_pkg = repo_root / "stata" / "rdlocrand.pkg"
    stata_text = read(stata_pkg)
    ensure_contains(errors, stata_pkg.relative_to(repo_root), stata_text, "frontend URL", FRONTEND_URL)
    ensure_contains(errors, stata_pkg.relative_to(repo_root), stata_text, "repository URL", REPO_URL)
    ensure_order(errors, stata_pkg.relative_to(repo_root), stata_text, ["Matias D. Cattaneo", "Rocio Titiunik", "Gonzalo Vazquez-Bare"])
    if "Ricardo" in stata_text or "Masini" in stata_text:
        errors.append("stata/rdlocrand.pkg lists Ricardo Masini, who is Python-only.")

    for root in (repo_root / "R" / "rdlocrand", repo_root / "stata"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            text = read(path)
            if "Ricardo" in text or "Masini" in text:
                errors.append(f"{path.relative_to(repo_root).as_posix()} mentions Ricardo Masini outside the Python package.")

    print(f"Repository: {repo_root}")
    if errors:
        for error in errors:
            print("ERROR:", error, file=sys.stderr)
        return 1

    print("Repository metadata checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
