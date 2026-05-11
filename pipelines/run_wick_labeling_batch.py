#!/usr/bin/env python3
"""Supervised + cluster experiments: pivot wick, k_atr=0.25, mixed labeling sessions.

Labels: session_aware=True for AAPL_1min / AAPL_5min; False for AAPL_15min / AAPL_1hour.
Trading session-awareness follows each run_experiment default (respecting --no-session-aware).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PY = ROOT / ".venv" / "bin" / "python"
PY = str(_PY) if _PY.is_file() else sys.executable
RUN_EXP = ROOT / "pipelines" / "run_experiment.py"
RUN_BASE = ROOT / "pipelines" / "run_baseline_cluster.py"
RESULTS = ROOT / "results"

LABEL_SUP = ["--labeling", "volatility", "--k-atr", "0.25", "--label-pivot", "wick"]
LABEL_CL = ["--k-atr", "0.25", "--label-pivot", "wick"]


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def exp(
    dataset: str,
    *,
    labeling_session_aware: bool,
    periods: float,
    window: int = 64,
    label_window: int = 4,
    heavy: bool = False,
    no_sess_trade: bool = False,
    blockstart: bool = False,
) -> None:
    cmd = [
        PY,
        str(RUN_EXP),
        "--dataset",
        dataset,
        "--window",
        str(window),
        "--label-window",
        str(label_window),
        "--periods-per-year",
        str(periods),
        *LABEL_SUP,
    ]
    if labeling_session_aware:
        cmd.append("--labels-session-aware")
    if heavy:
        cmd.append("--heavy-features")
    if no_sess_trade:
        cmd.append("--no-session-aware")
    if blockstart:
        cmd.append("--block-session-start-entry")
    run(cmd)


def baseline(
    dataset: str,
    kind: str,
    *,
    labeling_session_aware: bool,
    heavy: bool = False,
    window: int = 64,
    label_window: int = 4,
) -> None:
    cmd = [
        PY,
        str(RUN_BASE),
        "--dataset",
        dataset,
        "--window",
        str(window),
        "--label-window",
        str(label_window),
        "--kind",
        kind,
        *LABEL_CL,
    ]
    if labeling_session_aware:
        cmd.append("--labels-session-aware")
    if heavy:
        cmd.append("--heavy-features")
    run(cmd)


def subdir_supervised(
    ds: str,
    *,
    lw: int,
    heavy: bool,
    labeling_lsa: bool,
    sess_trade: bool,
    blk: bool,
    w_feat: int = 64,
) -> str:
    k, pv = "0p25", "w"
    h = "_heavy" if heavy else ""
    sess = "_nosess" if not sess_trade else "_sess"
    lsa = "_lsa" if labeling_lsa else ""
    blk_s = "_blockstart" if blk else ""
    return (
        f"{ds}_volatility_w{w_feat}_lw{lw}_k{k}_pv{pv}{h}{sess}{lsa}{blk_s}"
    )


def subdir_baseline(
    ds: str, kind: str, *, heavy: bool, lw: int, labeling_lsa: bool
) -> str:
    k, pv = "0p25", "w"
    fea = "_heavy" if heavy else "_light"
    lsa = "_lsa" if labeling_lsa else ""
    return f"baseline_{kind}_{ds}_w64_lw{lw}_k{k}_pv{pv}{fea}{lsa}"


def load_sup(path: Path) -> dict:
    with path.open() as f:
        r = json.load(f)
    bm = r["backtest"]["best_model"]
    st = r["backtest"]["strategy"]
    n_sig = st["n_long_signals"] + st["n_short_signals"]
    ost = st["long_signals_on_session_start"] + st["short_signals_on_session_start"]
    return {
        "pr_min": r["models"][bm]["min"]["overall"]["pr_auc"],
        "pr_max": r["models"][bm]["max"]["overall"]["pr_auc"],
        "sharpe": st["sharpe"],
        "mdd": st["max_drawdown"],
        "tr": st["total_return"],
        "bh": r["backtest"]["buy_and_hold"]["total_return"],
        "on_pct": 100 * ost / n_sig if n_sig else 0.0,
    }


def summary_print(rows: list[tuple[str, Path]]) -> None:
    print("\n=== Summary ===\n")
    for name, p in rows:
        if not p.exists():
            print(f"MISSING {name}: {p}")
            continue
        if p.parts[-2].startswith("baseline_"):
            with p.open() as f:
                r = json.load(f)
            pm, px = r["min"]["mean_pr_auc"], r["max"]["mean_pr_auc"]
            print(f"{name}: PR {pm:.4f} / {px:.4f}")
        else:
            d = load_sup(p)
            print(
                f"{name}: PR {d['pr_min']:.4f}/{d['pr_max']:.4f}  "
                f"S {d['sharpe']:.2f}  MDD {d['mdd']*100:.1f}%  "
                f"TR {d['tr']*100:.1f}%  BH {d['bh']*100:.1f}%  onStart {d['on_pct']:.1f}%"
            )


def main() -> None:
    lw = 4
    # coarse TFs: labeling без привязки к сессии
    lax = False  # labeling session-aware off
    # fine TFs: labeling внутри сессии
    fin = True

    # ----- 1h -----
    ds = "AAPL_1hour"
    pp = 1638.0
    exp(ds, labeling_session_aware=lax, periods=pp, label_window=lw, heavy=True)
    exp(ds, labeling_session_aware=lax, periods=pp, label_window=lw, heavy=False)
    exp(
        ds,
        labeling_session_aware=lax,
        periods=pp,
        label_window=lw,
        heavy=True,
        no_sess_trade=True,
    )
    exp(
        ds,
        labeling_session_aware=lax,
        periods=pp,
        label_window=lw,
        heavy=True,
        blockstart=True,
    )
    baseline(ds, "kmeans", labeling_session_aware=lax, heavy=True, label_window=lw)
    baseline(ds, "dbscan", labeling_session_aware=lax, heavy=False, label_window=lw)

    # ----- 15m -----
    ds = "AAPL_15min"
    pp = 6552.0
    exp(ds, labeling_session_aware=lax, periods=pp, label_window=lw, heavy=True)
    exp(ds, labeling_session_aware=lax, periods=pp, label_window=lw, heavy=False)
    exp(
        ds,
        labeling_session_aware=lax,
        periods=pp,
        label_window=lw,
        heavy=True,
        no_sess_trade=True,
    )
    exp(
        ds,
        labeling_session_aware=lax,
        periods=pp,
        label_window=lw,
        heavy=True,
        blockstart=True,
    )

    # ----- 5m -----
    ds = "AAPL_5min"
    pp = 19656.0
    exp(ds, labeling_session_aware=fin, periods=pp, label_window=lw, heavy=True)
    exp(ds, labeling_session_aware=fin, periods=pp, label_window=lw, heavy=False)
    baseline(ds, "kmeans", labeling_session_aware=fin, heavy=False, label_window=lw)
    baseline(ds, "dbscan", labeling_session_aware=fin, heavy=False, label_window=lw)

    # ----- 1m -----
    ds = "AAPL_1min"
    pp = 98280.0
    exp(ds, labeling_session_aware=fin, periods=pp, label_window=lw, heavy=True)
    exp(ds, labeling_session_aware=fin, periods=pp, label_window=lw, heavy=False)
    exp(
        ds,
        labeling_session_aware=fin,
        periods=pp,
        label_window=lw,
        heavy=True,
        window=128,
    )
    baseline(ds, "kmeans", labeling_session_aware=fin, heavy=False, label_window=lw)
    baseline(ds, "dbscan", labeling_session_aware=fin, heavy=False, label_window=lw)

    rows = [
        ("1h heavy", RESULTS / f"{subdir_supervised('AAPL_1hour', lw=lw, heavy=True, labeling_lsa=False, sess_trade=True, blk=False)}/report.json"),
        ("1h light", RESULTS / f"{subdir_supervised('AAPL_1hour', lw=lw, heavy=False, labeling_lsa=False, sess_trade=True, blk=False)}/report.json"),
        ("1h heavy nosess", RESULTS / f"{subdir_supervised('AAPL_1hour', lw=lw, heavy=True, labeling_lsa=False, sess_trade=False, blk=False)}/report.json"),
        ("1h block", RESULTS / f"{subdir_supervised('AAPL_1hour', lw=lw, heavy=True, labeling_lsa=False, sess_trade=True, blk=True)}/report.json"),
        ("1h km", RESULTS / subdir_baseline("AAPL_1hour", "kmeans", heavy=True, lw=lw, labeling_lsa=False) / "report.json"),
        ("1h db", RESULTS / subdir_baseline("AAPL_1hour", "dbscan", heavy=False, lw=lw, labeling_lsa=False) / "report.json"),
        ("15m heavy", RESULTS / f"{subdir_supervised('AAPL_15min', lw=lw, heavy=True, labeling_lsa=False, sess_trade=True, blk=False)}/report.json"),
        ("15m light", RESULTS / f"{subdir_supervised('AAPL_15min', lw=lw, heavy=False, labeling_lsa=False, sess_trade=True, blk=False)}/report.json"),
        ("15m nosess", RESULTS / f"{subdir_supervised('AAPL_15min', lw=lw, heavy=True, labeling_lsa=False, sess_trade=False, blk=False)}/report.json"),
        ("15m block", RESULTS / f"{subdir_supervised('AAPL_15min', lw=lw, heavy=True, labeling_lsa=False, sess_trade=True, blk=True)}/report.json"),
        ("5m heavy", RESULTS / f"{subdir_supervised('AAPL_5min', lw=lw, heavy=True, labeling_lsa=True, sess_trade=True, blk=False)}/report.json"),
        ("5m light", RESULTS / f"{subdir_supervised('AAPL_5min', lw=lw, heavy=False, labeling_lsa=True, sess_trade=True, blk=False)}/report.json"),
        ("5m km", RESULTS / subdir_baseline("AAPL_5min", "kmeans", heavy=False, lw=lw, labeling_lsa=True) / "report.json"),
        ("5m db", RESULTS / subdir_baseline("AAPL_5min", "dbscan", heavy=False, lw=lw, labeling_lsa=True) / "report.json"),
        ("1m heavy64", RESULTS / f"{subdir_supervised('AAPL_1min', lw=lw, heavy=True, labeling_lsa=True, sess_trade=True, blk=False)}/report.json"),
        ("1m light64", RESULTS / f"{subdir_supervised('AAPL_1min', lw=lw, heavy=False, labeling_lsa=True, sess_trade=True, blk=False)}/report.json"),
        ("1m heavy128", RESULTS / f"{subdir_supervised('AAPL_1min', lw=lw, heavy=True, labeling_lsa=True, sess_trade=True, blk=False, w_feat=128)}/report.json"),
        ("1m km", RESULTS / subdir_baseline("AAPL_1min", "kmeans", heavy=False, lw=lw, labeling_lsa=True) / "report.json"),
        ("1m db", RESULTS / subdir_baseline("AAPL_1min", "dbscan", heavy=False, lw=lw, labeling_lsa=True) / "report.json"),
    ]

    summary_print(rows)


if __name__ == "__main__":
    main()
