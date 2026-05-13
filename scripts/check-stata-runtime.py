#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


WINDOWS_CANDIDATES = [
    r"C:\Program Files\StataNow19\StataMP-64.exe",
    r"C:\Program Files\StataNow19\StataSE-64.exe",
    r"C:\Program Files\StataNow19\StataBE-64.exe",
    r"C:\Program Files\Stata19\StataMP-64.exe",
    r"C:\Program Files\Stata19\StataSE-64.exe",
    r"C:\Program Files\Stata19\StataBE-64.exe",
]


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "stata" / "rdlocrand.pkg").exists():
            return path
    raise SystemExit("Could not find rdlocrand repository root.")


def find_stata(user_value: str | None) -> Path:
    candidates: list[str] = []
    if user_value:
        candidates.append(user_value)
    env_value = os.environ.get("STATA_EXE")
    if env_value:
        candidates.append(env_value)

    path_names = [
        "StataMP-64.exe",
        "StataSE-64.exe",
        "StataBE-64.exe",
        "stata-mp",
        "stata-se",
        "stata",
    ]
    candidates.extend(found for name in path_names if (found := shutil.which(name)))
    if os.name == "nt":
        candidates.extend(WINDOWS_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidates:
        path = Path(candidate)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            return path

    raise SystemExit(
        "Could not find Stata. Pass --stata or set STATA_EXE to the Stata executable."
    )


def read_log(log_file: Path) -> str:
    if not log_file.exists():
        return ""
    return log_file.read_text(encoding="utf-8", errors="replace")


def print_log_tail(log_file: Path, text: str | None = None) -> None:
    if text is None:
        text = read_log(log_file)
    if not text:
        return
    lines = text.splitlines()
    print(f"\nLast lines from {log_file}:")
    for line in lines[-80:]:
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Stata smoke checks for rdlocrand.")
    parser.add_argument("--stata", help="Path to the Stata executable.")
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())
    stata = find_stata(args.stata)
    do_file = repo_root / "scripts" / "check-stata-runtime.do"
    log_file = repo_root / "tmp" / "stata-check" / "check-stata-runtime.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    command = [str(stata), "/e", "do", str(do_file), str(repo_root)]
    print(f"Repository: {repo_root}")
    print(f"Stata:      {stata}")
    result = subprocess.run(command, cwd=repo_root)
    log_text = read_log(log_file)
    if result.returncode != 0 or "Stata runtime checks passed." not in log_text:
        print_log_tail(log_file, log_text)
        if result.returncode != 0:
            return result.returncode
        return 1

    print_log_tail(log_file, log_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
