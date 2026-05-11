#!/usr/bin/env python3
"""Run supervised + baseline experiments for table 3.3 and pick best window by strategy TR.

Example::

    python pipelines/table33_sweep.py --windows 64,96,128
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PY = ROOT / ".venv" / "bin" / "python"
PY = str(_PY) if _PY.is_file() else sys.executable
RUN_EXP = ROOT / "pipelines" / "run_experiment.py"
RUN_CL = ROOT / "pipelines" / "run_baseline_cluster.py"

LABEL = ["--labeling", "volatility", "--k-atr", "0.25", "--label-pivot", "wick"]


@dataclass(frozen=True)
class DS:
    name: str
    periods_per_year: float
    labels_session_aware: bool


DATASETS: list[DS] = [
    DS("AAPL_1hour", 1638.0, False),
    DS("AAPL_15min", 6552.0, False),
    DS("AAPL_5min", 19656.0, True),
    DS("AAPL_1min", 98280.0, True),
]


def run_supervised(
    ds: DS,
    window: int,
    *,
    heavy: bool,
    sess_trade: bool,
    blockstart: bool,
) -> None:
    cmd = [
        PY,
        str(RUN_EXP),
        "--dataset",
        ds.name,
        "--window",
        str(window),
        "--label-window",
        "4",
        "--periods-per-year",
        str(ds.periods_per_year),
        "--n-folds",
        "5",
        *LABEL,
    ]
    if heavy:
        cmd.append("--heavy-features")
    if ds.labels_session_aware:
        cmd.append("--labels-session-aware")
    if not sess_trade:
        cmd.append("--no-session-aware")
    if blockstart:
        cmd.append("--block-session-start-entry")
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def run_baseline(ds: DS, window: int, *, kind: str) -> None:
    cmd = [
        PY,
        str(RUN_CL),
        "--dataset",
        ds.name,
        "--window",
        str(window),
        "--label-window",
        "4",
        "--k-atr",
        "0.25",
        "--label-pivot",
        "wick",
        "--periods-per-year",
        str(ds.periods_per_year),
        "--n-folds",
        "5",
        "--kind",
        kind,
        "--heavy-features",
    ]
    if ds.labels_session_aware:
        cmd.append("--labels-session-aware")
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--windows",
        default="64,96,128",
        help="Comma-separated feature windows (best row = max strategy TR among these).",
    )
    args = ap.parse_args()
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]

    for ds in DATASETS:
        for w in windows:
            run_supervised(ds, w, heavy=True, sess_trade=True, blockstart=False)
            run_supervised(ds, w, heavy=False, sess_trade=True, blockstart=False)
            run_supervised(ds, w, heavy=True, sess_trade=False, blockstart=False)
            run_supervised(ds, w, heavy=False, sess_trade=False, blockstart=False)
            run_supervised(ds, w, heavy=True, sess_trade=True, blockstart=True)
            run_supervised(ds, w, heavy=False, sess_trade=True, blockstart=True)
            run_baseline(ds, w, kind="kmeans")
            run_baseline(ds, w, kind="dbscan")
    w_str = ",".join(map(str, windows))
    print(f"\nDone. Run: python pipelines/table33_aggregate.py --windows {w_str}")


if __name__ == "__main__":
    main()
