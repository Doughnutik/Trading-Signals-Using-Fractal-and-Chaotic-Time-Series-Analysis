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
import mplfinance as mpf  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

from src.features import FULL_GROUPS, FeatureConfig, build_features  # noqa: E402
from src.labeling import mark_volatility_extrema  # noqa: E402
from src.sessions import session_boundaries  # noqa: E402
from src.utils import RESULTS_DIR, ensure_dir, load_ohlcv  # noqa: E402

PLOTS = ensure_dir(RESULTS_DIR / "plots")


def plot_equity_four_separate() -> None:
    """Four separate figures: Buy & Hold vs strategy equity from saved experiments.

    Uses ``equity.csv`` from ``run_experiment``. For 15m/1h, labeling may use
    extrema whose neighbourhood crosses session boundaries (no
    ``--labels-session-aware``), while the back-test still flattens before
    overnight (``_sess`` runs, not ``_nosess``)~-- positions do not carry
    across sessions. Fine TFs use label + trade session-aware (``_lsa`` + ``_sess``).
    """
    configs: list[tuple[str, str, str]] = [
        (
            "AAPL_1min_volatility_w64_lw4_k0p25_pvw_heavy_sess_lsa",
            "AAPL_1min",
            "AAPL 1 min · heavy — labels session-aware · trade: flat overnight",
        ),
        (
            "AAPL_5min_volatility_w64_lw4_k0p25_pvw_heavy_sess_lsa",
            "AAPL_5min",
            "AAPL 5 min · heavy — labels session-aware · trade: flat overnight",
        ),
        (
            "AAPL_15min_volatility_w64_lw4_k0p25_pvw_heavy_sess",
            "AAPL_15min",
            "AAPL 15 min · heavy — labels: window may cross sessions · trade: flat overnight",
        ),
        (
            "AAPL_1hour_volatility_w64_lw4_k0p25_pvw_heavy_sess",
            "AAPL_1hour",
            "AAPL 1 h · heavy — labels: window may cross sessions · trade: flat overnight",
        ),
    ]

    for out_slug, (subdir, ds, title_line) in zip(
        ("1min", "5min", "15min", "1hour"),
        configs,
        strict=True,
    ):
        eq_path = RESULTS_DIR / subdir / "equity.csv"
        if not eq_path.exists():
            print(f"[plot] SKIP missing equity: {eq_path}")
            continue
        strat = pd.read_csv(eq_path, index_col=0, parse_dates=True)
        strat = strat.squeeze()
        strat.name = "strategy"

        df = load_ohlcv(ds)
        ix = strat.index.intersection(df.index)
        strat = strat.loc[ix]
        bh = df.loc[ix, "close"].astype(float)
        bh = bh / bh.iloc[0]

        fig, ax = plt.subplots(figsize=(10.5, 4.25))
        ax.plot(strat.index, strat.values, label="strategy (heavy)", color="#08519c", linewidth=1.2)
        ax.plot(bh.index, bh.values, label="buy & hold (close)", color="#252525", linestyle="--", linewidth=1.0)
        ax.set_title(title_line)
        ax.set_ylabel("Equity ($t_0 \\rightarrow 1$)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.9)
        fig.autofmt_xdate()
        fig.tight_layout()
        outp = PLOTS / f"equity_{out_slug}_heavy_vs_bh.png"
        fig.savefig(outp, dpi=130)
        plt.close(fig)
        print(f"[plot] {outp}")


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
    r"""Retrain RF on the full AAPL_5min heavy set to extract feature importances.

    Aligns with основной прогон: volatility/wick labels, ``labels_session-aware``,
    trade/session-aware признаки, окно :math:`w=96` (лучшее по доходности среди
    :math:`\{64,96,128\}` для heavy session-aware в агрегированной таблице результатов).
    """
    from sklearn.ensemble import RandomForestClassifier

    fi_window = 96
    df = load_ohlcv("AAPL_5min")
    df = mark_volatility_extrema(
        df,
        window=4,
        k_atr=0.25,
        session_aware=True,
        pivot="wick",
    )
    feats = build_features(
        df,
        FeatureConfig(
            window=fi_window, groups=FULL_GROUPS, session_aware=True
        ),
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
    ax.set_title("Top-25 features AAPL 5min, heavy, session-aware")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_importance.png", dpi=130)
    plt.close(fig)

    imp.sort_values("avg", ascending=False).to_csv(PLOTS / "feature_importance.csv", index=False)
    print("[plot] feature_importance.png")


def plot_label_examples():
    data = "AAPL_15min"
    df = load_ohlcv(data)
    df = mark_volatility_extrema(
        df, k_atr=0.25, session_aware=False, pivot="wick"
    )
    print(df.is_min.value_counts())
    print(df.is_max.value_counts())
    sess = session_boundaries(df)
    sl = df.iloc[4000:4200].copy()
    sl_sess = sess.iloc[4000:4200]

    cols = {"Open": sl["open"], "High": sl["high"], "Low": sl["low"], "Close": sl["close"]}
    vol_on = False
    if "volume" in sl.columns:
        cols["Volume"] = sl["volume"]
        vol_on = True
    ohlcv = pd.DataFrame(cols, index=sl.index)

    is_min_arr = pd.to_numeric(sl["is_min"], errors="coerce").fillna(0).astype(int)
    is_max_arr = pd.to_numeric(sl["is_max"], errors="coerce").fillna(0).astype(int)
    mins_y = sl["low"].where(is_min_arr == 1)
    maxs_y = sl["high"].where(is_max_arr == 1)

    addplots = []
    if mins_y.notna().any():
        addplots.append(
            mpf.make_addplot(
                mins_y,
                type="scatter",
                marker="^",
                markersize=30,
                color="green",
            )
        )
    if maxs_y.notna().any():
        addplots.append(
            mpf.make_addplot(
                maxs_y,
                type="scatter",
                marker="v",
                markersize=30,
                color="red",
            )
        )

    kw: dict = {
        "type": "candle",
        "style": "yahoo",
        "figsize": (12, 6 if vol_on else 4.5),
        "title": f"Extrema labels + session boundaries — {data}",
        "ylabel": "",
        "returnfig": True,
        "tight_layout": True,
        "warn_too_much_data": len(sl) + 100,
        "volume": vol_on,
    }
    if addplots:
        kw["addplot"] = addplots
    vline_times = list(
        sl.index[sl_sess["is_session_start"].astype(bool).to_numpy()]
    )
    if vline_times:
        kw["vlines"] = dict(
            vlines=vline_times,
            linewidths=[0.75] * len(vline_times),
            colors=["b"] * len(vline_times),
            alpha=0.25,
        )

    fig, axes = mpf.plot(ohlcv, **kw)
    legend_handles = []
    if mins_y.notna().any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="^",
                color="green",
                markersize=9,
                label="is_min @ low",
            )
        )
    if maxs_y.notna().any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="v",
                color="red",
                markersize=9,
                label="is_max @ high",
            )
        )
    if legend_handles:
        axes[0].legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=9,
            framealpha=0.75,
        )
    fig.savefig(PLOTS / "labels_example.png", dpi=130, bbox_inches="tight")
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
    import argparse

    ap = argparse.ArgumentParser(description="Publication plots.")
    ap.add_argument(
        "--equity-four",
        action="store_true",
        help="Write four PNGs: B&H vs heavy for 1m/5m (LSA) and 15m/1h (no label LSA).",
    )
    ap.add_argument(
        "--feature-importance",
        action="store_true",
        help="Retrain RF on AAPL 5min heavy (w=96) and write feature_importance.png/csv.",
    )
    args = ap.parse_args()
    if args.feature_importance:
        plot_feature_importance()
        return
    if args.equity_four:
        plot_equity_four_separate()
        return
    plot_label_examples()
    


if __name__ == "__main__":
    main()
