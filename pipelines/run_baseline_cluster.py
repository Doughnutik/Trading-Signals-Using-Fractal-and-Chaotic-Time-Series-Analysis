"""Legacy one-class clustering baseline (DBSCAN / KMeans on positives).

Reproduces the approach from the original notebooks but fixed for
data-leakage (uses only information from closed bars plus ``open[i]``)
and with a consistent evaluation protocol (purged walk-forward split).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluation as ev  # noqa: E402
from src import labeling as lbl  # noqa: E402
from src.features import FULL_GROUPS, LIGHT_GROUPS, FeatureConfig, build_features  # noqa: E402
from src.models import ClusterBaseline  # noqa: E402
from src.utils import RESULTS_DIR, ensure_dir, load_ohlcv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="AAPL_1hour")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--k-atr", type=float, default=0.5)
    p.add_argument("--kind", choices=["kmeans", "dbscan"], default="kmeans")
    p.add_argument("--n-clusters", type=int, default=2)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--heavy-features", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    df_raw = load_ohlcv(args.dataset)
    df = lbl.mark_volatility_extrema(df_raw, k_atr=args.k_atr, session_aware=False)
    counts = lbl.label_counts(df)
    print(counts)

    groups = FULL_GROUPS if args.heavy_features else LIGHT_GROUPS
    feats = build_features(df, FeatureConfig(window=args.window, groups=tuple(groups)))
    feats = feats.replace([np.inf, -np.inf], np.nan)
    data = pd.concat([feats, df[["is_min", "is_max"]]], axis=1).dropna()
    feat_cols = [c for c in data.columns if c not in ("is_min", "is_max")]
    X = data[feat_cols].to_numpy(dtype=float)
    y_min = data["is_min"].to_numpy(dtype=int)
    y_max = data["is_max"].to_numpy(dtype=int)
    idx = data.index
    n = len(X)

    fold_size = n // (args.n_folds + 1)
    results = {"min": [], "max": []}
    for k in range(args.n_folds):
        tr_end = fold_size * (k + 1)
        te_start = tr_end + args.window
        te_end = te_start + fold_size
        if te_end > n:
            break
        tr = slice(0, tr_end)
        te = slice(te_start, te_end)

        for target, y, out in [("min", y_min, results["min"]), ("max", y_max, results["max"])]:
            X_tr_pos = X[tr][y[tr] == 1]
            if len(X_tr_pos) < max(args.n_clusters, 5):
                continue
            baseline = ClusterBaseline(
                kind=args.kind,
                n_clusters=args.n_clusters,
                eps=args.eps,
            ).fit(X_tr_pos)
            proba = baseline.score_samples(X[te])
            thr, _ = ev.best_threshold_by_f1(y[te], proba)
            rep = ev.classification_metrics(y[te], proba, threshold=thr).to_dict()
            rep["n_train_pos"] = int(len(X_tr_pos))
            out.append(rep)

    summary = {
        "min": {
            "mean_pr_auc": float(np.mean([f["pr_auc"] for f in results["min"]])) if results["min"] else 0.0,
            "mean_f1": float(np.mean([f["f1"] for f in results["min"]])) if results["min"] else 0.0,
            "folds": results["min"],
        },
        "max": {
            "mean_pr_auc": float(np.mean([f["pr_auc"] for f in results["max"]])) if results["max"] else 0.0,
            "mean_f1": float(np.mean([f["f1"] for f in results["max"]])) if results["max"] else 0.0,
            "folds": results["max"],
        },
        "config": vars(args),
        "runtime_sec": time.time() - t0,
    }
    print(json.dumps(summary, indent=2, default=str))

    subdir = f"baseline_{args.kind}_{args.dataset}_w{args.window}{'_heavy' if args.heavy_features else ''}"
    out_dir = ensure_dir(RESULTS_DIR / subdir)
    with open(out_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
