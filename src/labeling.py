"""Local extrema labeling.

The label uses both past and future bars and, when directional filters are
enabled, the bar's ``close`` vs ``open``---this is target construction only;
at prediction time features use closed history plus ``open[i]`` only.

Two variants are provided:

``mark_simple_extrema``
    Original rule from ``mark_points.ipynb``: strict fractal rule with
    neighborhood of size 4 and a relative threshold ``alpha``.

``mark_volatility_extrema``
    A more permissive rule: a bar is a local min if its low (or close) is
    the lowest over ``window`` bars on each side and the surrounding move
    exceeds ``k * ATR`` where ATR is an estimate of bar volatility.  This
    adapts the threshold to the asset/time frame instead of a fixed alpha.
    By default a minimum also requires ``close > open`` and a maximum
    ``close < open`` (targets only; not used as features at ``open[i]``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sessions import session_boundaries


def _session_id_array(df: pd.DataFrame, session_aware: bool) -> np.ndarray:
    if not session_aware:
        return np.zeros(len(df), dtype=np.int64)
    return session_boundaries(df)["session_id"].to_numpy()


def mark_simple_extrema(
    df: pd.DataFrame,
    window: int = 4,
    alpha: float = 0.001,
    require_candle_direction: bool = True,
    session_aware: bool = True,
) -> pd.DataFrame:
    """Mark local min/max using the strict fractal rule.

    Parameters
    ----------
    df
        Input OHLCV frame.
    window
        Number of bars on each side used to check the fractal condition.
    alpha
        Minimum relative move that must be observed vs. the bar that is
        ``window`` steps away (on each side).
    require_candle_direction
        If True (default), label a local min only when
        ``close[i] > open[i]`` (long idea) and a max only when
        ``close[i] < open[i]`` (short idea).  Set False to match older
        notebooks that used structure alone.
    """
    out = df.copy()
    n = len(out)
    is_min = np.zeros(n, dtype=np.int8)
    is_max = np.zeros(n, dtype=np.int8)
    op = out["open"].to_numpy()
    cl = out["close"].to_numpy()
    sid = _session_id_array(out, session_aware)

    for i in range(window, n - window - 1):
        if sid[i - window] != sid[i + window]:
            continue  # window crosses a session boundary
        can_min = (not require_candle_direction) or (cl[i] > op[i])
        if can_min:
            minL = min(cl[i - window : i].min(), op[i - window : i].min())
            minR = min(
                op[i + 1 : i + window + 1].min(), cl[i + 1 : i + window + 1].min()
            )
            if op[i] <= minL and op[i] <= minR:
                if (
                    op[i] <= op[i - window] * (1 - alpha)
                    and op[i] <= cl[i + window] * (1 - alpha)
                ):
                    is_min[i] = 1

        can_max = (not require_candle_direction) or (cl[i] < op[i])
        if can_max:
            maxL = max(cl[i - window : i].max(), op[i - window : i].max())
            maxR = max(
                op[i + 1 : i + window + 1].max(), cl[i + 1 : i + window + 1].max()
            )
            if op[i] >= maxL and op[i] >= maxR:
                if (
                    op[i] >= op[i - window] * (1 + alpha)
                    and op[i] >= cl[i + window] * (1 + alpha)
                ):
                    is_max[i] = 1

    out["is_min"] = is_min
    out["is_max"] = is_max
    return out


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return atr


def mark_volatility_extrema(
    df: pd.DataFrame,
    window: int = 4,
    k_atr: float = 0.5,
    atr_period: int = 14,
    session_aware: bool = True,
    require_candle_direction: bool = True,
) -> pd.DataFrame:
    """Volatility-adaptive extrema labeling.

    Uses ATR to adapt the threshold instead of a fixed relative alpha.
    When ``session_aware`` is True, candidate bars whose neighbourhood
    spans a session boundary are skipped – otherwise an intraday low
    could be compared against the opening gap of the next day, which
    has no exploitable meaning.

    When ``require_candle_direction`` is True (default), a local minimum
    is labelled only if ``close_i > open_i`` and a maximum only if
    ``close_i < open_i``.  This uses the bar's close **only** when
    building the training target; it is never used as a feature at
    decision time (where only ``open_i`` and past bars are available).
    """
    out = df.copy()
    n = len(out)
    is_min = np.zeros(n, dtype=np.int8)
    is_max = np.zeros(n, dtype=np.int8)
    op = out["open"].to_numpy()
    cl = out["close"].to_numpy()
    hi = out["high"].to_numpy()
    lo = out["low"].to_numpy()
    atr = _atr(out, atr_period)
    sid = _session_id_array(out, session_aware)

    for i in range(window, n - window - 1):
        if sid[i - window] != sid[i + window]:
            continue
        left_lo = lo[i - window : i].min()
        right_lo = lo[i + 1 : i + window + 1].min()
        left_hi = hi[i - window : i].max()
        right_hi = hi[i + 1 : i + window + 1].max()
        thr = k_atr * atr[i]

        can_min = (not require_candle_direction) or (cl[i] > op[i])
        if can_min and op[i] <= left_lo and op[i] <= right_lo:
            if (left_hi - op[i]) >= thr and (right_hi - op[i]) >= thr:
                is_min[i] = 1
        can_max = (not require_candle_direction) or (cl[i] < op[i])
        if can_max and op[i] >= left_hi and op[i] >= right_hi:
            if (op[i] - left_lo) >= thr and (op[i] - right_lo) >= thr:
                is_max[i] = 1

    out["is_min"] = is_min
    out["is_max"] = is_max
    return out


def drop_session_edge_extrema(
    df: pd.DataFrame, edge_bars: int = 1
) -> pd.DataFrame:
    """Zero-out ``is_min`` / ``is_max`` that sit on session boundaries.

    An extremum spotted on the very first (or last) bar of a session
    is almost always a gap artefact: by construction the fractal rule
    compares it against neighbouring bars that belong to a *different*
    session, so it cannot be exploited anyway (we cannot open a
    position mid-gap).  ``edge_bars`` controls how many bars near each
    boundary are invalidated.
    """
    sess = session_boundaries(df)
    start_mask = sess["bars_since_start"] < edge_bars
    bars_to_end = (
        sess.groupby("session_id")["bars_since_start"]
        .transform("max")
        - sess["bars_since_start"]
    )
    end_mask = bars_to_end < edge_bars

    out = df.copy()
    on_edge = (start_mask | end_mask).to_numpy()
    if "is_min" in out.columns:
        out.loc[on_edge, "is_min"] = 0
    if "is_max" in out.columns:
        out.loc[on_edge, "is_max"] = 0
    return out


def label_counts(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "n_min": int(df["is_min"].sum()),
        "n_max": int(df["is_max"].sum()),
        "share_min": float(df["is_min"].mean()),
        "share_max": float(df["is_max"].mean()),
    }
