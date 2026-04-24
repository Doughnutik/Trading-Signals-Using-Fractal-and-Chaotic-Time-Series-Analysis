"""Trading session awareness.

Equity data has two kinds of discontinuities:

* **Overnight gap** – between the last bar of a trading day and the
  first bar of the next session (typically a ~17-18h or ~65h
  "weekend" jump).  Prices can jump arbitrarily during that gap due
  to news flow, earnings, etc.
* **Intraday pauses** – rare, but sometimes TwelveData returns data
  with tiny pauses (feed gaps).

For feature engineering these gaps are a problem:

1. `log(close[i] / close[i-1])` on the first bar of the day is
   not an intraday move, it is an *overnight return* of a different
   statistical nature.
2. Rolling windows that span through the boundary mix up regimes.
3. A "local extremum" on the first bar of a session is usually a
   gap artefact, not a meaningful turning point.

This module provides utilities to *detect* sessions and gaps, and
feature helpers so downstream code can opt-in to session-aware
processing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_modal_bar(df: pd.DataFrame) -> pd.Timedelta:
    """Most common spacing between bars."""
    deltas = df.index.to_series().diff().dropna()
    return deltas.mode().iloc[0]


def session_boundaries(
    df: pd.DataFrame,
    gap_factor: float = 2.0,
    modal_delta: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Return a DataFrame indexed like ``df`` with session columns.

    Columns
    -------
    ``is_session_start`` : ``int8``
        1 if the bar starts a new trading session (first bar, or bar
        whose distance from the previous one is more than
        ``gap_factor × modal_delta``).
    ``is_session_end`` : ``int8``
        1 if the bar is the last bar of a session.
    ``bars_since_start`` : ``int32``
        Count of bars since the beginning of the current session
        (0 for ``is_session_start`` bars).
    ``session_id`` : ``int32``
        Monotonically increasing session identifier.
    """
    modal = modal_delta or detect_modal_bar(df)
    threshold = modal * gap_factor
    deltas = df.index.to_series().diff()
    is_start = (deltas > threshold) | deltas.isna()
    session_id = is_start.cumsum().astype(np.int32)
    bars_since = (
        pd.Series(np.arange(len(df)), index=df.index)
        .groupby(session_id)
        .transform(lambda s: s - s.iloc[0])
        .astype(np.int32)
    )
    is_end = is_start.shift(-1, fill_value=True).astype(np.int8)
    return pd.DataFrame(
        {
            "is_session_start": is_start.astype(np.int8),
            "is_session_end": is_end,
            "bars_since_start": bars_since,
            "session_id": session_id,
        },
        index=df.index,
    )


def overnight_gap(df: pd.DataFrame, sess: pd.DataFrame | None = None) -> pd.Series:
    """Relative overnight gap at each bar.

    The gap is defined as ``(open[i] - close[i-1]) / close[i-1]`` but
    it is only non-zero on ``is_session_start`` bars (otherwise we
    would double-count with regular log returns).
    """
    sess = sess if sess is not None else session_boundaries(df)
    gap = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    return gap.where(sess["is_session_start"].astype(bool), 0.0)


def masked_log_returns(df: pd.DataFrame, sess: pd.DataFrame | None = None) -> pd.Series:
    """Log returns of ``close`` with overnight returns zeroed out.

    Use this instead of ``np.log(close/close.shift(1))`` in feature
    pipelines that should be invariant to overnight gaps.  The first
    bar of each session contributes zero, so rolling variances do
    not get polluted by gaps.
    """
    sess = sess if sess is not None else session_boundaries(df)
    lr = np.log(df["close"] / df["close"].shift(1))
    return lr.where(~sess["is_session_start"].astype(bool), 0.0)


def time_of_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encoding of time of day (sin/cos over 24h)."""
    t = df.index
    seconds = (t.hour * 3600 + t.minute * 60 + t.second).to_numpy(dtype=float)
    day_frac = seconds / 86400.0
    return pd.DataFrame(
        {
            "tod_sin": np.sin(2 * np.pi * day_frac),
            "tod_cos": np.cos(2 * np.pi * day_frac),
        },
        index=df.index,
    )
