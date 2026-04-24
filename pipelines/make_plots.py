"""Generate publication-ready plots from the cached experiment results."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features import FULL_GROUPS, FeatureConfig, build_features  # noqa: E402
from src.labeling import mark_volatility_extrema  # noqa: E402
from src.sessions import session_boundaries  # noqa: E402
from src.utils import RESULTS_DIR, ensure_dir, load_ohlcv  # noqa: E402

PLOTS = ensure_dir(RESULTS_DIR / "plots")


def plot_equity_curves():
    fig, ax = plt.subplots(figsize=(11, 5))
    curves = {
        "1h heavy, no-session (unrealistic, keeps overnight P&L)":
            "AAPL_1hour_volatility_w64_heavy",
        "1h heavy, session-aware (flat overnight)":
            "AAPL_1hour_volatility_w64_lw4_heavy_sess",
        "15min heavy, session-aware":
            "AAPL_15min_volatility_w64_lw4_heavy_sess",
    }
    for label, folder in curves.items():
        path = RESULTS_DIR / folder / "equity.csv"
        if not path.exists():
            continue
        eq = pd.read_csv(path, index_col=0, parse_dates=True)
        eq.columns = ["equity"]
        ax.plot(eq.index, eq["equity"], label=label, linewidth=1.3)

    df = load_ohlcv("AAPL_1hour")
    bh = df["close"] / df["close"].iloc[0]
    ax.plot(bh.index, bh.values, label="Buy & Hold (1hour)", linestyle="--", color="black")

    ax.set_title("Equity curves – walk-forward out-of-sample")
    ax.set_ylabel("Equity (initial = 1.0)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "equity_curves.png", dpi=130)
    plt.close(fig)
    print("[plot] equity_curves.png")


def plot_feature_importance():
    """Retrain RF on the full AAPL_1hour heavy set to extract feature importances."""
    from sklearn.ensemble import RandomForestClassifier

    df = load_ohlcv("AAPL_1hour")
    df = mark_volatility_extrema(df, k_atr=0.5, session_aware=False)
    feats = build_features(
        df, FeatureConfig(window=64, groups=FULL_GROUPS, session_aware=True)
    )
    feats = feats.replace([np.inf, -np.inf], np.nan)
    data = pd.concat([feats, df[["is_min", "is_max"]]], axis=1).dropna()
    cols = [c for c in data.columns if c not in ("is_min", "is_max")]
    X = data[cols].to_numpy()
    y_min = data["is_min"].to_numpy()
    y_max = data["is_max"].to_numpy()

    rf_min = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ).fit(X, y_min)
    rf_max = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ).fit(X, y_max)

    imp = pd.DataFrame(
        {
            "feature": cols,
            "min_imp": rf_min.feature_importances_,
            "max_imp": rf_max.feature_importances_,
        }
    )
    imp["avg"] = (imp["min_imp"] + imp["max_imp"]) / 2
    imp = imp.sort_values("avg", ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(imp))
    ax.barh(y - 0.2, imp["min_imp"], height=0.4, label="is_min", color="#2b8cbe")
    ax.barh(y + 0.2, imp["max_imp"], height=0.4, label="is_max", color="#de2d26")
    ax.set_yticks(y)
    ax.set_yticklabels(imp["feature"])
    ax.set_xlabel("Gini importance")
    ax.set_title("Top-25 features (AAPL 1hour, heavy set)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_importance.png", dpi=130)
    plt.close(fig)

    imp.sort_values("avg", ascending=False).to_csv(PLOTS / "feature_importance.csv", index=False)
    print("[plot] feature_importance.png")


def plot_label_examples():
    df = load_ohlcv("AAPL_1hour")
    df = mark_volatility_extrema(df, k_atr=0.5, session_aware=False)
    sess = session_boundaries(df)
    sl = df.iloc[3000:3200]
    sl_sess = sess.iloc[3000:3200]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sl.index, sl["close"], color="black", linewidth=0.8, label="close")
    # shade session boundaries
    start_idx = sl.index[sl_sess["is_session_start"].astype(bool).to_numpy()]
    for t in start_idx:
        ax.axvline(t, color="grey", alpha=0.25, linewidth=0.7)
    mins = sl[sl["is_min"] == 1]
    maxs = sl[sl["is_max"] == 1]
    ax.scatter(mins.index, mins["open"], marker="^", color="green", s=60, label="is_min")
    ax.scatter(maxs.index, maxs["open"], marker="v", color="red", s=60, label="is_max")
    ax.set_title("Extrema labels vs. session boundaries — almost every label coincides with a gap bar")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "labels_example.png", dpi=130)
    plt.close(fig)
    print("[plot] labels_example.png")


def plot_session_gap_distribution():
    df = load_ohlcv("AAPL_1hour")
    sess = session_boundaries(df)
    cl_prev = df["close"].shift(1)
    gap = (df["open"] - cl_prev) / cl_prev
    gap_start = gap[sess["is_session_start"].astype(bool) & cl_prev.notna()]
    gap_other = gap[~sess["is_session_start"].astype(bool) & cl_prev.notna()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axes[0].hist(gap_other.clip(-0.02, 0.02) * 100, bins=60, color="#2b8cbe", alpha=0.85)
    axes[0].set_title(f"Intra-session (bar-to-bar) Δ, % — n={len(gap_other)}")
    axes[0].set_xlabel("return, %")
    axes[0].grid(alpha=0.3)
    axes[1].hist(gap_start.clip(-0.02, 0.02) * 100, bins=60, color="#de2d26", alpha=0.85)
    axes[1].set_title(f"Overnight gap, % — n={len(gap_start)}")
    axes[1].set_xlabel("return, %")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Overnight gaps have a heavier-tailed distribution than intraday bars (AAPL 1h)")
    fig.tight_layout()
    fig.savefig(PLOTS / "gap_distribution.png", dpi=130)
    plt.close(fig)
    print("[plot] gap_distribution.png")


def main():
    plot_equity_curves()
    plot_feature_importance()
    plot_label_examples()
    plot_session_gap_distribution()


if __name__ == "__main__":
    main()
