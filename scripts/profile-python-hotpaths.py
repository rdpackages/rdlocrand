#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "Python" / "rdlocrand" / "pyproject.toml").exists():
            return path
    raise SystemExit("Could not find rdlocrand repository root.")


def make_data(n: int = 240):
    idx = np.arange(1, n + 1, dtype=float)
    R = np.linspace(-1.5, 1.5, n)
    D = (R >= 0).astype(float)
    Y = 1 + 0.8 * R - 0.2 * R**2 + 1.5 * D + np.sin(idx / 3)
    X = np.column_stack((np.cos(idx / 2), np.sin(idx / 3), np.cos(idx / 5)))
    return Y, R, X


def timed(name: str, func, repeat: int) -> None:
    elapsed = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        elapsed.append(time.perf_counter() - start)
    best = min(elapsed)
    median = float(np.median(elapsed))
    print(f"{name:24} best={best:.4f}s median={median:.4f}s repeat={repeat}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile representative rdlocrand Python hot paths.")
    parser.add_argument("--quick", action="store_true", help="Use smaller repetition counts for fast local checks.")
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())
    sys.path.insert(0, str(repo_root / "Python" / "rdlocrand" / "src"))

    from rdlocrand import rdrandinf, rdrbounds, rdsensitivity, rdwinselect

    Y, R, X = make_data()
    repeat = 2 if args.quick else 5
    reps = 60 if args.quick else 200

    print(f"Repository: {repo_root}")
    print(f"Python:     {sys.executable}")
    print(f"Reps:       {reps}")
    print()

    timed(
        "rdrandinf_diffmeans",
        lambda: rdrandinf(Y, R, wl=-0.8, wr=0.8, reps=reps, seed=123, quietly=True),
        repeat,
    )
    timed(
        "rdrandinf_all",
        lambda: rdrandinf(Y, R, wl=-0.8, wr=0.8, statistic="all", reps=reps, seed=123, quietly=True),
        repeat,
    )
    timed(
        "rdwinselect_balance",
        lambda: rdwinselect(R, X, wmin=0.35, wstep=0.1, nwindows=6, reps=reps, seed=123, quietly=True, level=0),
        repeat,
    )
    timed(
        "rdsensitivity_grid",
        lambda: rdsensitivity(
            Y, R,
            wlist=np.array([0.5, 0.7, 0.9]),
            tlist=np.array([0, 1, 2]),
            reps=max(20, reps // 2),
            seed=123,
            nodraw=True,
            quietly=True,
        ),
        repeat,
    )

    def run_rdrbounds():
        with contextlib.redirect_stdout(io.StringIO()):
            rdrbounds(Y, R, expgamma=[1.5], wlist=[0.7], reps=max(20, reps // 2), seed=123)

    timed("rdrbounds_both", run_rdrbounds, repeat)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
