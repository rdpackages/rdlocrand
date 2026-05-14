#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


PACKAGE_LINE = re.compile(r"^f\s+(.+?)\s*$")
DATE_LINE = re.compile(r"^d\s+Distribution-Date:\s+(\d{8})\s*$")
DELIVERABLE_SUFFIXES = {".ado", ".sthlp", ".mo", ".do", ".dta"}
KNOWN_UNLISTED = {
    "rdlocrand_functions.do",  # Source used to produce compiled Mata helpers.
}
TEXT_SUFFIXES = {".ado", ".do", ".pkg", ".sthlp", ".toc"}
REQUIRED_PKG_TEXT = {
    "canonical website": "https://rdpackages.github.io/",
    "canonical repository": "https://github.com/rdpackages/rdlocrand",
}
FORBIDDEN_TEXT = {
    "deprecated package URL": "https://rdpackages.github.io/rdlocrand/",
    "old Stata install branch": "raw.githubusercontent.com/rdpackages/rdlocrand/master/",
    "old Matias email": "cattaneo@princeton.edu",
    "old Rocio email": "titiunik@princeton.edu",
    "old Gonzalo email": "gvazquez@econ.ucsb.edu",
    "old Ricardo email": "rmasini@ucdavis.edu",
    "Ricardo author entry": "Ricardo",
    "Masini author entry": "Masini",
    "stale R-only companion wording": "Companion {browse \"www.r-project.org\":R} functions",
    "stale related-package wording": "Related Stata and R packages",
    "stale rdrandinf companion link": "{help rdrandinf:rdwinselect}",
    "stale rdrbounds companion link 1": "{help rdsensitivity:rdwinselect}",
    "stale rdrbounds companion link 2": "{help rdrbounds:rdsensitivity}",
    "stale rdrbounds lower return": "r(lbounds)",
    "stale rdrbounds upper return": "r(ubounds)",
    "stale rdwinselect window-list return": "r(wlist)",
    "stale scalar CI lower return": "r(ci_lb)",
    "stale scalar CI upper return": "r(ci_ub)",
}


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "stata" / "rdlocrand.pkg").exists():
            return path
    raise SystemExit("Could not find rdlocrand repository root.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the rdlocrand Stata package manifest.")
    parser.add_argument("--strict-unlisted", action="store_true", help="Fail when deliverable files are not listed.")
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())
    stata_dir = repo_root / "stata"
    pkg_file = stata_dir / "rdlocrand.pkg"

    entries: list[str] = []
    distribution_dates: list[str] = []
    for line in pkg_file.read_text(encoding="utf-8").splitlines():
        file_match = PACKAGE_LINE.match(line.strip())
        if file_match:
            entries.append(file_match.group(1))
        date_match = DATE_LINE.match(line.strip())
        if date_match:
            distribution_dates.append(date_match.group(1))

    errors: list[str] = []
    warnings: list[str] = []

    if not entries:
        errors.append("No distributed files were listed in stata/rdlocrand.pkg.")

    duplicates = sorted(name for name, count in Counter(entries).items() if count > 1)
    if duplicates:
        errors.append("Duplicate package entries: " + ", ".join(duplicates))

    missing = sorted(name for name in entries if not (stata_dir / name).exists())
    if missing:
        errors.append("Missing package files: " + ", ".join(missing))

    if len(distribution_dates) != 1:
        errors.append("Expected exactly one Distribution-Date line in stata/rdlocrand.pkg.")

    pkg_text = pkg_file.read_text(encoding="utf-8", errors="replace")
    for label, pattern in REQUIRED_PKG_TEXT.items():
        if pattern not in pkg_text:
            errors.append(f"stata/rdlocrand.pkg is missing {label}: {pattern}")

    listed = set(entries)
    unlisted = sorted(
        path.name
        for path in stata_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in DELIVERABLE_SUFFIXES
        and path.name not in listed
        and path.name not in KNOWN_UNLISTED
    )
    if unlisted:
        message = "Deliverable-looking files not listed in rdlocrand.pkg: " + ", ".join(unlisted)
        if args.strict_unlisted:
            errors.append(message)
        else:
            warnings.append(message)

    for path in sorted(stata_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN_TEXT.items():
            if pattern in text:
                errors.append(f"{path.name} contains {label}: {pattern}")

    print(f"Repository: {repo_root}")
    print(f"Package:    {pkg_file}")
    print(f"Entries:    {len(entries)}")
    if distribution_dates:
        print(f"Date:       {distribution_dates[0]}")

    for warning in warnings:
        print("WARNING:", warning)

    if errors:
        for error in errors:
            print("ERROR:", error, file=sys.stderr)
        return 1

    print("Stata package manifest checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
