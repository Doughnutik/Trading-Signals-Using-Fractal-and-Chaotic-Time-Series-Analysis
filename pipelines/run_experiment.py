"""End-to-end research pipeline.

Usage::

    python pipelines/run_experiment.py --dataset AAPL_1hour --window 64 \
        --labeling volatility --heavy-features

The script loads OHLCV data, labels extrema, builds features and trains
several classifiers (LogReg, RandomForest, LightGBM) with a purged
walk-forward split.  It reports classification metrics and runs a
vectorised trading back-test.  All results are persisted to
``results/<dataset>/`` for the README.
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
from src import models as mdl  # noqa: E402
from src.features import (  # noqa: E402
    FEATURE_CACHE_BUMP,
    FeatureConfig,
    FULL_GROUPS,
    LIGHT_GROUPS,
    build_features,
)
from src.sessions import session_boundaries  # noqa: E402
from src.utils import RESULTS_DIR, ensure_dir, load_ohlcv  # noqa: E402

CACHE_DIR = RESULTS_DIR / "_feature_cache"


def load_or_build_features(df: pd.DataFrame, dataset: str, cfg: FeatureConfig) -> pd.DataFrame:
    ensure_dir(CACHE_DIR)
    sa = "sa1" if cfg.session_aware else "sa0"
    tag = (
        f"{dataset}_w{cfg.window}_seg{cfg.n_slope_segments}_{sa}_"
        f"{'_'.join(cfg.groups)}_{FEATURE_CACHE_BUMP}"
    )
    path = CACHE_DIR / f"{tag}.parquet"
    if path.exists():
        print(f"[cache] loaded features from {path.name}")
        return pd.read_parquet(path)
    feats = build_features(df, cfg)
    feats.to_parquet(path)
    print(f"[cache] stored features at {path.name}")
    return feats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="AAPL_1hour")
    p.add_argument("--window", type=int, default=64)
    p.add_argument(
        "--labeling",
        choices=["simple", "volatility"],
        default="volatility",
    )
    p.add_argument("--alpha", type=float, default=0.002)
    p.add_argument("--k-atr", type=float, default=0.25, help="ATR multiplier for labeling swing.")
    p.add_argument(
        "--label-pivot",
        choices=["wick", "open"],
        default="wick",
        help='Extremum pivot: "wick" = low/high vs neighbours (denser); '
        '"open" = stricter open-only pivot (legacy).',
    )
    p.add_argument(
        "--label-window",
        type=int,
        default=4,
        help="Neighbourhood size for extrema labeling (per side).",
    )
    p.add_argument("--heavy-features", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--periods-per-year", type=float, default=252 * 6.5)
    p.add_argument("--out-subdir", default=None)
    p.add_argument(
        "--no-session-aware",
        action="store_true",
        help="Disable session-aware features and trading filters.",
    )
    p.add_argument(
        "--labels-session-aware",
        action="store_true",
        help="Require extremum neighbourhood to lie inside one session.",
    )
    p.add_argument(
        "--drop-edge-extrema",
        type=int,
        default=0,
        help="Zero-out positive labels within N bars of a session boundary.",
    )
    p.add_argument(
        "--no-require-candle-direction",
        action="store_true",
        help="Label extrema without bullish/bearish close filter (legacy).",
    )
    p.add_argument(
        "--block-session-start-entry",
        action="store_true",
        help="Do not open trades on the first bar after an overnight gap.",
    )
    return p.parse_args()


def build_labels(df: pd.DataFrame, args) -> pd.DataFrame:
    require_dir = not args.no_require_candle_direction
    if args.labeling == "simple":
        labeled = lbl.mark_simple_extrema(
            df,
            window=args.label_window,
            alpha=args.alpha,
            require_candle_direction=require_dir,
            session_aware=args.labels_session_aware,
        )
    else:
        labeled = lbl.mark_volatility_extrema(
            df,
            window=args.label_window,
            k_atr=args.k_atr,
            session_aware=args.labels_session_aware,
            require_candle_direction=require_dir,
            pivot=args.label_pivot,
        )
    if args.drop_edge_extrema > 0:
        labeled = lbl.drop_session_edge_extrema(labeled, edge_bars=args.drop_edge_extrema)
    return labeled


def purged_walk_forward(n: int, n_folds: int, embargo: int):
    """Yield (train_idx, test_idx) – chronological, with embargo gap."""
    fold_size = n // (n_folds + 1)
    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        test_start = train_end + embargo
        test_end = test_start + fold_size
        if test_end > n:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx


def fit_predict(model, X_tr, y_tr, X_te, kind: str):
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier  # noqa: F401
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
    elif kind == "xgboost":
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
    else:
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
    return proba


def run_model_cv(
    name: str,
    make_model,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    kind: str,
) -> dict:
    all_proba = np.full(len(y), np.nan)
    fold_reports = []
    for f, (tr, te) in enumerate(folds):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        if kind in ("lightgbm", "xgboost"):
            m = make_model(n_pos=n_pos, n_neg=n_neg)
        else:
            m = make_model()
        proba = fit_predict(m, X_tr, y_tr, X_te, kind)
        all_proba[te] = proba
        thr, _ = ev.best_threshold_by_f1(y_te, proba)
        rep = ev.classification_metrics(y_te, proba, threshold=thr)
        fold_reports.append(rep.to_dict())

    mask = ~np.isnan(all_proba)
    y_val = y[mask]
    p_val = all_proba[mask]
    thr, _ = ev.best_threshold_by_f1(y_val, p_val)
    overall = ev.classification_metrics(y_val, p_val, threshold=thr)
    return {
        "name": name,
        "folds": fold_reports,
        "overall": overall.to_dict(),
        "proba": all_proba,
        "oof_threshold": thr,
    }


def main():
    args = parse_args()
    t0 = time.time()
    print(
        f"[+] dataset={args.dataset} window={args.window} labeling={args.labeling} "
        f"heavy={args.heavy_features} k_atr={args.k_atr} pivot={args.label_pivot} "
        f"label_window={args.label_window}"
    )

    df_raw = load_ohlcv(args.dataset)
    df = build_labels(df_raw, args)
    counts = lbl.label_counts(df)
    print(f"[+] labeled: {counts}")

    groups = FULL_GROUPS if args.heavy_features else LIGHT_GROUPS
    session_aware = not args.no_session_aware
    cfg = FeatureConfig(
        window=args.window,
        groups=tuple(groups),
        session_aware=session_aware,
    )
    print(f"[+] building features: groups={cfg.groups} session_aware={session_aware}")
    feats = load_or_build_features(df_raw, args.dataset, cfg)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    data = pd.concat([feats, df[["is_min", "is_max"]]], axis=1).dropna()
    print(f"[+] usable rows: {len(data)}  (dropped {len(df) - len(data)} due to NaN)")

    feat_cols = [c for c in data.columns if c not in ("is_min", "is_max")]
    X = data[feat_cols].to_numpy(dtype=float)
    y_min = data["is_min"].to_numpy(dtype=int)
    y_max = data["is_max"].to_numpy(dtype=int)
    idx = data.index

    folds = list(purged_walk_forward(len(X), args.n_folds, embargo=args.window))
    print(f"[+] walk-forward folds: {len(folds)} (avg train {folds[-1][0].size}, test {folds[-1][1].size})")

    results = {
        "config": vars(args),
        "labeling": counts,
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "models": {},
        "backtest": {},
        "runtime_sec": None,
    }

    model_specs = [
        ("logreg", mdl.make_logreg, "logreg"),
        ("random_forest", mdl.make_random_forest, "rf"),
        ("lightgbm", mdl.make_lightgbm, "lightgbm"),
    ]

    proba_min_by_model = {}
    proba_max_by_model = {}
    thr_min_by_model = {}
    thr_max_by_model = {}

    for name, make, kind in model_specs:
        print(f"  -- min model: {name}")
        res_min = run_model_cv(name, make, X, y_min, folds, kind)
        print(f"     overall {res_min['overall']}")
        print(f"  -- max model: {name}")
        res_max = run_model_cv(name, make, X, y_max, folds, kind)
        print(f"     overall {res_max['overall']}")
        results["models"][name] = {
            "min": {k: v for k, v in res_min.items() if k != "proba"},
            "max": {k: v for k, v in res_max.items() if k != "proba"},
        }
        proba_min_by_model[name] = res_min["proba"]
        proba_max_by_model[name] = res_max["proba"]
        thr_min_by_model[name] = res_min["oof_threshold"]
        thr_max_by_model[name] = res_max["oof_threshold"]

    # choose the best model by pr_auc (min + max average)
    best = max(
        results["models"].items(),
        key=lambda kv: np.nanmean(
            [kv[1]["min"]["overall"]["pr_auc"], kv[1]["max"]["overall"]["pr_auc"]]
        ),
    )
    best_name = best[0]
    print(f"[+] best model by PR-AUC: {best_name}")

    # backtest using out-of-fold predictions of the best model
    proba_min = proba_min_by_model[best_name]
    proba_max = proba_max_by_model[best_name]
    thr_l = thr_min_by_model[best_name]
    thr_s = thr_max_by_model[best_name]

    long_sig, short_sig = ev.build_trade_signals(proba_min, proba_max, thr_l, thr_s)
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
    # Buy & hold on the full sample so the benchmark does not depend on which rows
    # survive feature NA-drop (must match across configs of the same dataset).
    bh = ev.buy_and_hold_stats(df_raw, periods_per_year=args.periods_per_year)
    print(f"[+] strategy stats: {bt.stats}")
    print(f"[+] buy & hold:     {bh}")

    results["backtest"] = {
        "best_model": best_name,
        "threshold_long": thr_l,
        "threshold_short": thr_s,
        "strategy": bt.to_dict(),
        "buy_and_hold": bh,
    }

    results["runtime_sec"] = time.time() - t0

    _k_slug = ("%g" % args.k_atr).replace(".", "p").replace("-", "m")
    _pv_tag = "w" if args.label_pivot == "wick" else "o"
    subdir = args.out_subdir or (
        f"{args.dataset}_{args.labeling}_w{args.window}_lw{args.label_window}"
        f"_k{_k_slug}_pv{_pv_tag}"
        f"{'_heavy' if args.heavy_features else ''}"
        f"{'_sess' if session_aware else '_nosess'}"
        f"{'_lsa' if args.labels_session_aware else ''}"
        f"{f'_edge{args.drop_edge_extrema}' if args.drop_edge_extrema else ''}"
        f"{'_nodir' if args.no_require_candle_direction else ''}"
        f"{'_blockstart' if args.block_session_start_entry else ''}"
    )
    out_dir = ensure_dir(RESULTS_DIR / subdir)
    with open(out_dir / "report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    bt.equity.to_csv(out_dir / "equity.csv")
    np.save(out_dir / "proba_min.npy", proba_min)
    np.save(out_dir / "proba_max.npy", proba_max)
    print(f"[+] saved to {out_dir}")


if __name__ == "__main__":
    main()
