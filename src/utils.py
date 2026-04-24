"""Data IO helpers."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_ohlcv(name: str) -> pd.DataFrame:
    """Load a raw OHLCV CSV (e.g. AAPL_1hour) with datetime index, sorted asc."""
    path = DATASETS_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def load_marked(name: str) -> pd.DataFrame:
    """Load a pre-marked CSV (`*_marked.csv`)."""
    path = DATASETS_DIR / f"{name}_marked.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
