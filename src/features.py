"""Feature engineering for extrema prediction.

All features are built for bar ``i`` using information available *before*
the current bar closes, i.e.:

* for bars ``i-1, i-2, ..., i-N`` the full OHLCV is fair game,
* for bar ``i`` **only** ``open[i]`` may be used.

That strict constraint is enforced by always computing rolling quantities
with ``closed="left"`` or by slicing ``[i-N:i]`` (exclusive of ``i``).

Feature groups:

``returns``    log returns, cumulative returns, price-vs-open ratios.
``rolling``    rolling mean/std/min/max/skew/kurt of close and log-returns.
``technical``  RSI, Williams %R, Stochastic K, Bollinger position,
               ATR-normalised range, SMA/EMA ratios.
``fractal``    Hurst (R/S), DFA, Higuchi FD, Katz FD, Petrosian FD.
``entropy``    Sample, approximate, permutation and spectral entropy.
``slopes``     Normalised piecewise-linear slopes of the close series.

The heavy groups (``fractal`` and ``entropy``) are optional.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .sessions import (
    masked_log_returns,
    overnight_gap,
    session_boundaries,
    time_of_day_features,
)

try:
    import antropy as ant
except Exception:  # pragma: no cover - optional
    ant = None

try:
    import nolds
except Exception:  # pragma: no cover
    nolds = None

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# basic rolling helpers (close="left": the window does NOT include bar i)
# ---------------------------------------------------------------------------

def _roll(series: pd.Series, window: int, func: str) -> pd.Series:
    return getattr(series.shift(1).rolling(window, min_periods=window), func)()


# ---------------------------------------------------------------------------
# returns & ratios
# ---------------------------------------------------------------------------

def returns_features(
    df: pd.DataFrame, window: int, session_aware: bool = True
) -> pd.DataFrame:
    """Log returns of previous bars and price ratios vs. ``open[i]``.

    When ``session_aware`` is True the overnight gap is not propagated
    into the log-return series: the first bar of each session has
    ``log_ret = 0``.  A separate ``overnight_gap`` feature captures
    the jump.
    """
    op = df["open"]
    cl = df["close"]
    if session_aware:
        sess = session_boundaries(df)
        log_ret = masked_log_returns(df, sess)
        gap = overnight_gap(df, sess)
    else:
        log_ret = np.log(cl / cl.shift(1))
        gap = pd.Series(0.0, index=df.index)

    feats = {
        "overnight_gap": gap,
    }
    # log returns of last K bars (k=1..min(window,10))
    for k in range(1, min(window, 10) + 1):
        feats[f"logret_lag{k}"] = log_ret.shift(k)

    # cumulative returns over horizons
    for h in (3, 5, 10, 20, window):
        if h <= window:
            feats[f"cumret_{h}"] = (cl.shift(1) / cl.shift(h) - 1.0)

    # price of previous bar relative to current open
    feats["close_prev_over_open"] = cl.shift(1) / op - 1.0
    feats["high_prev_over_open"] = df["high"].shift(1) / op - 1.0
    feats["low_prev_over_open"] = df["low"].shift(1) / op - 1.0

    # open vs window min/max
    roll_min = df["low"].shift(1).rolling(window, min_periods=window).min()
    roll_max = df["high"].shift(1).rolling(window, min_periods=window).max()
    feats["open_pos_in_range"] = (op - roll_min) / (roll_max - roll_min)
    feats["open_vs_window_min"] = op / roll_min - 1.0
    feats["open_vs_window_max"] = op / roll_max - 1.0

    return pd.DataFrame(feats, index=df.index)


# ---------------------------------------------------------------------------
# rolling statistics
# ---------------------------------------------------------------------------

def rolling_features(
    df: pd.DataFrame, window: int, session_aware: bool = True
) -> pd.DataFrame:
    cl = df["close"]
    if session_aware:
        log_ret = masked_log_returns(df)
    else:
        log_ret = np.log(cl / cl.shift(1))
    feats = {}
    for w in (5, 10, 20, window):
        if w > window:
            continue
        feats[f"ret_std_{w}"] = _roll(log_ret, w, "std")
        feats[f"ret_mean_{w}"] = _roll(log_ret, w, "mean")
        feats[f"ret_skew_{w}"] = _roll(log_ret, w, "skew")
        feats[f"ret_kurt_{w}"] = _roll(log_ret, w, "kurt")
    return pd.DataFrame(feats, index=df.index)


# ---------------------------------------------------------------------------
# technical indicators
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def technical_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    op = df["open"]
    hi = df["high"]
    lo = df["low"]
    cl = df["close"]

    feats = {}
    # RSI of the closed history (use shift(1) to only feed past bars)
    feats["rsi_14"] = _rsi(cl, 14).shift(1)
    feats["rsi_7"] = _rsi(cl, 7).shift(1)

    # Williams %R over past 14 bars (closed)
    hh = hi.shift(1).rolling(14, min_periods=14).max()
    ll = lo.shift(1).rolling(14, min_periods=14).min()
    feats["williams_r_14"] = -100.0 * (hh - cl.shift(1)) / (hh - ll)

    # Stochastic K
    feats["stoch_k_14"] = 100.0 * (cl.shift(1) - ll) / (hh - ll)

    # Bollinger z-score on close (20)
    mean_20 = cl.shift(1).rolling(20, min_periods=20).mean()
    std_20 = cl.shift(1).rolling(20, min_periods=20).std()
    feats["bb_z_20"] = (cl.shift(1) - mean_20) / std_20
    feats["open_bb_z_20"] = (op - mean_20) / std_20

    # SMA / EMA ratios (price vs trend)
    ema_fast = cl.shift(1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = cl.shift(1).ewm(span=26, adjust=False, min_periods=26).mean()
    feats["ema_fast_over_slow"] = ema_fast / ema_slow - 1.0
    feats["open_over_ema_slow"] = op / ema_slow - 1.0

    # ATR normalised bar range
    prev_close = cl.shift(1)
    tr = np.maximum.reduce(
        [
            hi.shift(1).to_numpy() - lo.shift(1).to_numpy(),
            np.abs(hi.shift(1).to_numpy() - prev_close.shift(1).to_numpy()),
            np.abs(lo.shift(1).to_numpy() - prev_close.shift(1).to_numpy()),
        ]
    )
    tr = pd.Series(tr, index=df.index)
    atr_14 = tr.rolling(14, min_periods=14).mean()
    feats["atr14_norm"] = atr_14 / op
    feats["range_prev_over_atr"] = (hi.shift(1) - lo.shift(1)) / atr_14

    # MACD-like
    feats["macd"] = (ema_fast - ema_slow) / op

    return pd.DataFrame(feats, index=df.index)


# ---------------------------------------------------------------------------
# fractal & chaotic features (one value per bar, over the past window)
# ---------------------------------------------------------------------------

def _apply_rolling_window(arr: np.ndarray, window: int, func, default=np.nan):
    """Apply ``func`` over a trailing window ending at ``i-1`` (exclusive of i).

    Returns an array of the same length with ``default`` for positions
    where the window is incomplete.
    """
    n = arr.shape[0]
    out = np.full(n, default, dtype=float)
    for i in range(window, n):
        w = arr[i - window : i]
        try:
            out[i] = float(func(w))
        except Exception:
            out[i] = default
    return out


def fractal_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if ant is None or nolds is None:
        raise RuntimeError("antropy and nolds are required for fractal_features")

    cl = df["close"].to_numpy(dtype=float)

    feats = {}
    feats["hurst_rs"] = _apply_rolling_window(
        cl, window, lambda w: nolds.hurst_rs(w, debug_data=False)
    )
    feats["dfa"] = _apply_rolling_window(
        cl, window, lambda w: nolds.dfa(w)
    )
    feats["higuchi_fd"] = _apply_rolling_window(
        cl, window, lambda w: ant.higuchi_fd(w, kmax=10)
    )
    feats["katz_fd"] = _apply_rolling_window(
        cl, window, lambda w: ant.katz_fd(w)
    )
    feats["petrosian_fd"] = _apply_rolling_window(
        cl, window, lambda w: ant.petrosian_fd(w)
    )
    return pd.DataFrame(feats, index=df.index)


def entropy_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if ant is None:
        raise RuntimeError("antropy is required for entropy_features")

    # entropies are more meaningful on returns
    cl = df["close"].to_numpy(dtype=float)
    logret = np.concatenate([[0.0], np.diff(np.log(cl))])

    feats = {}
    feats["sample_entropy"] = _apply_rolling_window(
        logret, window, lambda w: ant.sample_entropy(w, order=2)
    )
    feats["app_entropy"] = _apply_rolling_window(
        logret, window, lambda w: ant.app_entropy(w, order=2)
    )
    feats["perm_entropy"] = _apply_rolling_window(
        logret, window, lambda w: ant.perm_entropy(w, order=3, normalize=True)
    )
    feats["spectral_entropy"] = _apply_rolling_window(
        logret, window, lambda w: ant.spectral_entropy(w, sf=1, method="welch", normalize=True)
    )
    return pd.DataFrame(feats, index=df.index)


# ---------------------------------------------------------------------------
# piecewise-linear slopes (fast custom implementation, not pwlf)
# ---------------------------------------------------------------------------

def _equal_split_slopes(y: np.ndarray, n_segments: int = 4) -> np.ndarray:
    """Fit independent linear regressions on ``n_segments`` equal chunks.

    This replaces ``pwlf``: it is deterministic, fast, and expressive
    enough for short windows.  The slopes are normalised by the starting
    price so that they are scale invariant.
    """
    n = y.shape[0]
    base = y[0] if y[0] != 0 else 1.0
    y_norm = y / base
    size = n // n_segments
    slopes = np.zeros(n_segments)
    for s in range(n_segments):
        start = s * size
        end = n if s == n_segments - 1 else (s + 1) * size
        seg = y_norm[start:end]
        x = np.arange(seg.shape[0], dtype=float)
        # slope of least-squares line
        x_mean = x.mean()
        y_mean = seg.mean()
        num = ((x - x_mean) * (seg - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        slopes[s] = num / den if den > 0 else 0.0
    return slopes


def slopes_features(df: pd.DataFrame, window: int, n_segments: int = 4) -> pd.DataFrame:
    cl = df["close"].to_numpy(dtype=float)
    n = cl.shape[0]
    out = np.full((n, n_segments), np.nan)
    for i in range(window, n):
        w = cl[i - window : i]
        out[i] = _equal_split_slopes(w, n_segments)
    cols = [f"slope_{s}" for s in range(n_segments)]
    return pd.DataFrame(out, index=df.index, columns=cols)


# ---------------------------------------------------------------------------
# session features
# ---------------------------------------------------------------------------

def session_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    sess = session_boundaries(df)
    tod = time_of_day_features(df)
    # normalised bars-since-start (bounded to [0, 1] using a typical
    # session length; values >1 simply get clipped)
    typical_session = max(int(sess.groupby("session_id").size().median()), 1)
    norm_bars = np.clip(sess["bars_since_start"].to_numpy() / typical_session, 0.0, 1.5)
    out = pd.DataFrame(
        {
            "is_session_start": sess["is_session_start"].astype(float),
            "bars_since_start_norm": norm_bars,
            "tod_sin": tod["tod_sin"],
            "tod_cos": tod["tod_cos"],
        },
        index=df.index,
    )
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "returns": returns_features,
    "rolling": rolling_features,
    "technical": technical_features,
    "slopes": slopes_features,
    "fractal": fractal_features,
    "entropy": entropy_features,
    "session": session_features,
}

LIGHT_GROUPS = ("returns", "rolling", "technical", "slopes", "session")
FULL_GROUPS = LIGHT_GROUPS + ("fractal", "entropy")


@dataclass
class FeatureConfig:
    window: int = 64
    groups: tuple = FULL_GROUPS
    n_slope_segments: int = 4
    session_aware: bool = True


def build_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FeatureConfig()
    frames = []
    for g in cfg.groups:
        fn = FEATURE_GROUPS[g]
        if g == "slopes":
            frames.append(fn(df, cfg.window, cfg.n_slope_segments))
        elif g in ("returns", "rolling"):
            frames.append(fn(df, cfg.window, session_aware=cfg.session_aware))
        else:
            frames.append(fn(df, cfg.window))
    feats = pd.concat(frames, axis=1)
    return feats
