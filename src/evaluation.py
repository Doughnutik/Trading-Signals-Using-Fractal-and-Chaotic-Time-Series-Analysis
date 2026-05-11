"""Classification metrics and simple trading back-test."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    n: int
    n_pos: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> ClassificationReport:
    y_pred = (proba >= threshold).astype(int)
    try:
        roc = float(roc_auc_score(y_true, proba))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, proba))
    except Exception:
        pr = float("nan")
    return ClassificationReport(
        n=int(len(y_true)),
        n_pos=int(y_true.sum()),
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        pr_auc=pr,
    )


def best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.concatenate([[0.0, 1.0], np.quantile(proba, np.linspace(0, 1, 200))]))
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# trading back-test
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    side: str  # "long" or "short"
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float

    @property
    def pnl(self) -> float:
        if self.side == "long":
            return self.exit_price / self.entry_price - 1.0
        return self.entry_price / self.exit_price - 1.0


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    stats: dict

    def to_dict(self) -> dict:
        return {
            "n_trades": len(self.trades),
            **self.stats,
        }


def _drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def backtest_signals(
    df: pd.DataFrame,
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    max_hold: int = 8,
    tp: float | None = None,
    sl: float | None = None,
    periods_per_year: float = 252 * 6.5,
    session_start: np.ndarray | None = None,
    session_end: np.ndarray | None = None,
    flatten_overnight: bool = True,
    block_entry_on_session_start: bool = False,
) -> BacktestResult:
    """Simple back-test.

    A trade is opened at ``open[i]`` whenever the corresponding signal
    fires.  The position is closed at the first of the following events:

    * another signal of the opposite direction fires,
    * ``max_hold`` bars elapsed,
    * ``tp`` / ``sl`` hit (evaluated on the bar's high/low).

    The function walks through bars sequentially – no look-ahead.
    """
    assert len(long_signal) == len(short_signal) == len(df)
    op = df["open"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    cl = df["close"].to_numpy()
    idx = df.index
    n = len(df)
    if session_start is None:
        session_start = np.zeros(n, dtype=bool)
    else:
        session_start = np.asarray(session_start, dtype=bool)
    if session_end is None:
        session_end = np.zeros(n, dtype=bool)
    else:
        session_end = np.asarray(session_end, dtype=bool)

    trades: list[Trade] = []
    position: str | None = None
    entry_price = 0.0
    entry_time = idx[0]
    bars_in = 0

    equity = [1.0]
    for i in range(n):
        bar_ret = 0.0
        if position is not None:
            bars_in += 1
            # evaluate TP/SL intra-bar
            exit_reason = None
            force_close = flatten_overnight and session_end[i]
            if position == "long":
                if tp is not None and hi[i] >= entry_price * (1 + tp):
                    exit_price = entry_price * (1 + tp)
                    exit_reason = "tp"
                elif sl is not None and lo[i] <= entry_price * (1 - sl):
                    exit_price = entry_price * (1 - sl)
                    exit_reason = "sl"
                elif bars_in >= max_hold:
                    exit_price = cl[i]
                    exit_reason = "hold"
                elif short_signal[i] and i > 0 and not session_start[i]:
                    exit_price = op[i]
                    exit_reason = "flip"
                elif force_close:
                    exit_price = cl[i]
                    exit_reason = "session_end"
                else:
                    exit_reason = None
            else:  # short
                if tp is not None and lo[i] <= entry_price * (1 - tp):
                    exit_price = entry_price * (1 - tp)
                    exit_reason = "tp"
                elif sl is not None and hi[i] >= entry_price * (1 + sl):
                    exit_price = entry_price * (1 + sl)
                    exit_reason = "sl"
                elif bars_in >= max_hold:
                    exit_price = cl[i]
                    exit_reason = "hold"
                elif long_signal[i] and i > 0 and not session_start[i]:
                    exit_price = op[i]
                    exit_reason = "flip"
                elif force_close:
                    exit_price = cl[i]
                    exit_reason = "session_end"
                else:
                    exit_reason = None

            if exit_reason is not None:
                trade = Trade(
                    side=position,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=idx[i],
                    exit_price=exit_price,
                )
                trades.append(trade)
                bar_ret = trade.pnl
                position = None
                bars_in = 0

        # open new position only if flat (optionally skip session_start)
        blocked = block_entry_on_session_start and session_start[i]
        if position is None and not blocked:
            if long_signal[i]:
                position = "long"
                entry_price = op[i]
                entry_time = idx[i]
                bars_in = 0
            elif short_signal[i]:
                position = "short"
                entry_price = op[i]
                entry_time = idx[i]
                bars_in = 0

        equity.append(equity[-1] * (1 + bar_ret))

    # force-close at end
    if position is not None:
        trade = Trade(
            side=position,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=idx[-1],
            exit_price=cl[-1],
        )
        trades.append(trade)
        equity[-1] = equity[-1] * (1 + trade.pnl)

    eq = pd.Series(equity[1:], index=idx)
    returns = eq.pct_change().fillna(0.0).to_numpy()

    win = [t.pnl > 0 for t in trades]
    total_long = int((long_signal).sum())
    total_short = int((short_signal).sum())
    long_on_start = int(((long_signal) & session_start).sum())
    short_on_start = int(((short_signal) & session_start).sum())
    stats = {
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
        "sharpe": _sharpe(returns, periods_per_year),
        "max_drawdown": _drawdown(eq.to_numpy()) if len(eq) else 0.0,
        "win_rate": float(np.mean(win)) if win else 0.0,
        "avg_trade_pnl": float(np.mean([t.pnl for t in trades])) if trades else 0.0,
        "best_trade": float(np.max([t.pnl for t in trades])) if trades else 0.0,
        "worst_trade": float(np.min([t.pnl for t in trades])) if trades else 0.0,
        "n_long_signals": total_long,
        "n_short_signals": total_short,
        "long_signals_on_session_start": long_on_start,
        "short_signals_on_session_start": short_on_start,
    }
    return BacktestResult(trades=trades, equity=eq, stats=stats)


def buy_and_hold_stats(df: pd.DataFrame, periods_per_year: float = 252 * 6.5) -> dict:
    cl = df["close"].to_numpy()
    eq = cl / cl[0]
    returns = np.concatenate([[0.0], np.diff(eq) / eq[:-1]])
    return {
        "total_return": float(eq[-1] - 1.0),
        "sharpe": _sharpe(returns, periods_per_year),
        "max_drawdown": _drawdown(eq),
    }


def build_trade_signals(
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    thr_l: float,
    thr_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn long/short scores into boolean entry arrays (used by experiment + baseline)."""
    long_sig = np.where(~np.isnan(proba_long) & (proba_long >= thr_l), 1, 0).astype(bool)
    short_sig = np.where(~np.isnan(proba_short) & (proba_short >= thr_s), 1, 0).astype(bool)
    both = long_sig & short_sig
    if both.any():
        keep_long = (proba_long - thr_l) >= (proba_short - thr_s)
        long_sig[both & ~keep_long] = False
        short_sig[both & keep_long] = False
    return long_sig, short_sig
