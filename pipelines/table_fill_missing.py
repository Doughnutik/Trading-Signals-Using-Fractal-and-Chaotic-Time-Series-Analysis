#!/usr/bin/env python3
"""Run only missing table-3.3 experiments (supervised + baselines) for given windows."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PY = ROOT / ".venv" / "bin" / "python"
PY = str(_PY) if _PY.is_file() else sys.executable
RUN_EXP = ROOT / "pipelines" / "run_experiment.py"
RUN_CL = ROOT / "pipelines" / "run_baseline_cluster.py"

import importlib.util

spec = importlib.util.spec_from_file_location("table33_aggregate", ROOT / "pipelines" / "table33_aggregate.py")
agg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["table33_aggregate"] = agg
spec.loader.exec_module(agg)
DS = agg.DS
DATASETS = agg.DATASETS
sup_path = agg.sup_path
baseline_path = agg.baseline_path
RESULTS = agg.RESULTS

LABEL = ["--labeling", "volatility", "--k-atr", "0.25", "--label-pivot", "wick"]


def ds_rows(ds: DS) -> list[tuple[str, callable]]:
    dsn, lsa = ds.key, ds.labels_lsa
    return [
        ("heavy_sess", lambda w: sup_path(dsn, w, heavy=True, sess_trade=True, labels_lsa=lsa, blockstart=False)),
        ("light_sess", lambda w: sup_path(dsn, w, heavy=False, sess_trade=True, labels_lsa=lsa, blockstart=False)),
        ("heavy_nosess", lambda w: sup_path(dsn, w, heavy=True, sess_trade=False, labels_lsa=lsa, blockstart=False)),
        ("light_nosess", lambda w: sup_path(dsn, w, heavy=False, sess_trade=False, labels_lsa=lsa, blockstart=False)),
        ("heavy_block", lambda w: sup_path(dsn, w, heavy=True, sess_trade=True, labels_lsa=lsa, blockstart=True)),
        ("light_block", lambda w: sup_path(dsn, w, heavy=False, sess_trade=True, labels_lsa=lsa, blockstart=True)),
        ("km", lambda w: baseline_path(dsn, w, "kmeans", lsa)),
        ("db", lambda w: baseline_path(dsn, w, "dbscan", lsa)),
    ]


def run_supervised(ds: DS, w: int, *, heavy: bool, sess: bool, block: bool) -> None:
    cmd = [
        PY,
        str(RUN_EXP),
        "--dataset",
        ds.key,
        "--window",
        str(w),
        "--label-window",
        "4",
        "--periods-per-year",
        str(ds.periods),
        "--n-folds",
        "5",
        *LABEL,
    ]
    if heavy:
        cmd.append("--heavy-features")
    if ds.labels_lsa:
        cmd.append("--labels-session-aware")
    if not sess:
        cmd.append("--no-session-aware")
    if block:
        cmd.append("--block-session-start-entry")
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def run_baseline(ds: DS, w: int, kind: str) -> None:
    cmd = [
        PY,
        str(RUN_CL),
        "--dataset",
        ds.key,
        "--window",
        str(w),
        "--label-window",
        "4",
        "--k-atr",
        "0.25",
        "--label-pivot",
        "wick",
        "--periods-per-year",
        str(ds.periods),
        "--n-folds",
        "5",
        "--kind",
        kind,
        "--heavy-features",
    ]
    if ds.labels_lsa:
        cmd.append("--labels-session-aware")
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="64,96,128")
    args = ap.parse_args()
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]

    for ds in DATASETS:
        for w in windows:
            for tag, fn in ds_rows(ds):
                rpath = fn(w)
                if (rpath.parent / "report.json").exists():
                    continue
                if tag in ("km", "db"):
                    run_baseline(ds, w, "kmeans" if tag == "km" else "dbscan")
                else:
                    heavy = tag.startswith("heavy")
                    if tag == "heavy_sess":
                        run_supervised(ds, w, heavy=heavy, sess=True, block=False)
                    elif tag == "light_sess":
                        run_supervised(ds, w, heavy=heavy, sess=True, block=False)
                    elif tag == "heavy_nosess":
                        run_supervised(ds, w, heavy=heavy, sess=False, block=False)
                    elif tag == "light_nosess":
                        run_supervised(ds, w, heavy=heavy, sess=False, block=False)
                    elif tag == "heavy_block":
                        run_supervised(ds, w, heavy=heavy, sess=True, block=True)
                    elif tag == "light_block":
                        run_supervised(ds, w, heavy=heavy, sess=True, block=True)
                    else:
                        raise RuntimeError(tag)


if __name__ == "__main__":
    main()
