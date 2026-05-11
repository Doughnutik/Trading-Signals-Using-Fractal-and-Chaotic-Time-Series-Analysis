"""Legacy one-class clustering baseline (DBSCAN / KMeans on positives).

Reproduces the approach from the original notebooks but fixed for
data-leakage (uses only information from closed bars plus ``open[i]``)
and with a consistent evaluation protocol (purged walk-forward split).
After OOF scoring, runs the same trading back-test as ``run_experiment``.
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
from src.sessions import session_boundaries  # noqa: E402
from src.utils import RESULTS_DIR, ensure_dir, load_ohlcv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="AAPL_1hour")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--k-atr", type=float, default=0.25)
    p.add_argument(
        "--label-pivot",
        choices=["wick", "open"],
        default="wick",
        help='Labeling pivot: "wick" (denser) vs "open" (stricter).',
    )
    p.add_argument("--kind", choices=["kmeans", "dbscan"], default="kmeans")
    p.add_argument("--n-clusters", type=int, default=2)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--heavy-features", action="store_true")
    p.add_argument(
        "--no-require-candle-direction",
        action="store_true",
        help="Label without bullish/bearish close filter.",
    )
    p.add_argument(
        "--label-window",
        type=int,
        default=4,
        help="Neighbourhood bars on each side for volatility labeling.",
    )
    p.add_argument(
        "--labels-session-aware",
        action="store_true",
        help="Require labeling window to lie within one calendar session.",
    )
    p.add_argument("--periods-per-year", type=float, default=252 * 6.5)
    p.add_argument(
        "--no-session-aware",
        action="store_true",
        help="Disable session-aware trading (carry overnight).",
    )
    p.add_argument(
        "--block-session-start-entry",
        action="store_true",
        help="Do not open trades on the first bar after an overnight gap.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    df_raw = load_ohlcv(args.dataset)
    df = lbl.mark_volatility_extrema(
        df_raw,
        window=args.label_window,
        k_atr=args.k_atr,
        session_aware=args.labels_session_aware,
        require_candle_direction=not args.no_require_candle_direction,
        pivot=args.label_pivot,
    )
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
    oof_min = np.full(n, np.nan)
    oof_max = np.full(n, np.nan)

    for k in range(args.n_folds):
        tr_end = fold_size * (k + 1)
        te_start = tr_end + args.window
        te_end = te_start + fold_size
        if te_end > n:
            break
        tr = slice(0, tr_end)
        te = slice(te_start, te_end)

        for target, y, out, oof_arr in [
            ("min", y_min, results["min"], oof_min),
            ("max", y_max, results["max"], oof_max),
        ]:
            X_tr_pos = X[tr][y[tr] == 1]
            if len(X_tr_pos) < max(args.n_clusters, 5):
                continue
            baseline = ClusterBaseline(
                kind=args.kind,
                n_clusters=args.n_clusters,
                eps=args.eps,
            ).fit(X_tr_pos)
            proba = baseline.score_samples(X[te])
            oof_arr[te] = proba
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
        "runtime_sec": None,
    }

    session_aware = not args.no_session_aware
    m_mask = ~np.isnan(oof_min)
    x_mask = ~np.isnan(oof_max)
    if m_mask.any() and x_mask.any():
        thr_l, _ = ev.best_threshold_by_f1(y_min[m_mask], oof_min[m_mask])
        thr_s, _ = ev.best_threshold_by_f1(y_max[x_mask], oof_max[x_mask])
        long_sig, short_sig = ev.build_trade_signals(oof_min, oof_max, thr_l, thr_s)
        bt_df = df.loc[idx]
        sess = session_boundaries(bt_df)
        bt = ev.backtest_signals(
            bt_df,
            long_sig,
            short_sig,
            max_hold=args.window // 4 if args.window >= 16 else 4,
            periods_per_year=args.periods_per_year,
            session_start=sess["is_session_start"].to_numpy().astype(bool) if session_aware else None,
            session_end=sess["is_session_end"].to_numpy().astype(bool) if session_aware else None,
            flatten_overnight=session_aware,
            block_entry_on_session_start=args.block_session_start_entry,
        )
        bh = ev.buy_and_hold_stats(df_raw, periods_per_year=args.periods_per_year)
        summary["backtest"] = {
            "best_model": f"cluster_{args.kind}",
            "threshold_long": thr_l,
            "threshold_short": thr_s,
            "strategy": bt.to_dict(),
            "buy_and_hold": bh,
        }
        print(f"[+] strategy stats: {bt.stats}")
        print(f"[+] buy & hold:     {bh}")
    else:
        summary["backtest"] = {}
        bt = None

    summary["runtime_sec"] = time.time() - t0
    print(json.dumps({k: v for k, v in summary.items() if k != "min" and k != "max"}, indent=2, default=str))

    _k_slug = ("%g" % args.k_atr).replace(".", "p").replace("-", "m")
    _pv_tag = "w" if args.label_pivot == "wick" else "o"
    _fea = "_heavy" if args.heavy_features else "_light"
    _lsa = "_lsa" if args.labels_session_aware else ""
    _sess = "_sess" if session_aware else "_nosess"
    _blk = "_blockstart" if args.block_session_start_entry else ""
    subdir = (
        f"baseline_{args.kind}_{args.dataset}_w{args.window}"
        f"_lw{args.label_window}_k{_k_slug}_pv{_pv_tag}{_fea}{_lsa}{_sess}{_blk}"
    )
    out_dir = ensure_dir(RESULTS_DIR / subdir)
    with open(out_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if bt is not None:
        bt.equity.to_csv(out_dir / "equity.csv")
    print(f"[+] saved to {out_dir}")


if __name__ == "__main__":
    main()
