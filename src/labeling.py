"""Local extrema labeling.

Two variants are provided:

``mark_simple_extrema``
    Original rule from ``mark_points.ipynb``: strict fractal rule with
    neighborhood of size 4 and a relative threshold ``alpha``.

``mark_volatility_extrema``
    Neighbourhood lows/highs plus ATR swing vs. opposite side.  Pivot for
    the strict centre comparison is either the bar's ``open`` or its
    ``low``/``high`` (wick) — wick matches the usual pivot definition and
    yields more positives than open-only.
"""
from __future__ import annotations

from typing import Literal

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
        ``close[i] < open[i]`` (short idea).
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
    k_atr: float = 0.25,
    atr_period: int = 14,
    session_aware: bool = True,
    require_candle_direction: bool = True,
    pivot: Literal["open", "wick"] = "wick",
) -> pd.DataFrame:
    """Volatility-adaptive extrema labeling.

    Uses ATR to adapt the swing threshold.  When ``session_aware`` is True,
    candidate bars whose neighbourhood spans a session boundary are skipped.

    ``pivot``
        * ``"wick"`` (default): local min if ``low[i]`` is the deepest vs
          neighbour lows on each side (max symmetric with ``high[i]``).
          This matches the usual fractal definition and is noticeably less
          strict than ``open``.
        * ``"open"``: legacy rule — compare ``open[i]`` to neighbour lows/highs
          (fewer labels; stricter for entry-at-open semantics).

    ``k_atr``
        If negative, the ATR swing filter is disabled.  If zero, swings are
        checked against a zero threshold.  Otherwise require move
        ``≥ k_atr · ATR[i]`` vs. the opposite-side range.

    ``require_candle_direction`` — if True, min only when ``close > open``,
    max only when ``close < open`` (target construction only).
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

        if k_atr < 0:
            min_swing_ok = True
            max_swing_ok = True
        else:
            thr = k_atr * atr[i]
            if pivot == "wick":
                min_swing_ok = (left_hi - lo[i]) >= thr and (
                    right_hi - lo[i]
                ) >= thr
                max_swing_ok = (hi[i] - left_lo) >= thr and (
                    hi[i] - right_lo
                ) >= thr
            else:
                min_swing_ok = (left_hi - op[i]) >= thr and (
                    right_hi - op[i]
                ) >= thr
                max_swing_ok = (op[i] - left_lo) >= thr and (
                    op[i] - right_lo
                ) >= thr

        can_min = (not require_candle_direction) or (cl[i] > op[i])
        if pivot == "wick":
            ok_min = lo[i] <= left_lo and lo[i] <= right_lo
        else:
            ok_min = op[i] <= left_lo and op[i] <= right_lo
        if can_min and ok_min and min_swing_ok:
            is_min[i] = 1

        can_max = (not require_candle_direction) or (cl[i] < op[i])
        if pivot == "wick":
            ok_max = hi[i] >= left_hi and hi[i] >= right_hi
        else:
            ok_max = op[i] >= left_hi and op[i] >= right_hi
        if can_max and ok_max and max_swing_ok:
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
