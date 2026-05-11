#!/usr/bin/env python3
"""Aggregate ``results/*/report.json`` for table 3.3: pick best window by strategy total return."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WINDOWS_DEFAULT = [64, 96, 128]
sys.path.insert(0, str(ROOT))

from src import evaluation as ev  # noqa: E402
from src.utils import load_ohlcv  # noqa: E402

K_TAG = "k0p25"
PV = "pvw"


@dataclass(frozen=True)
class DS:
    key: str
    latex: str
    labels_lsa: bool
    periods: float


DATASETS = [
    DS("AAPL_1hour", r"AAPL 1h", False, 1638.0),
    DS("AAPL_15min", r"AAPL 15min", False, 6552.0),
    DS("AAPL_5min", r"AAPL 5min", True, 19656.0),
    DS("AAPL_1min", r"AAPL 1min", True, 98280.0),
]


def sup_path(
    ds: str,
    w: int,
    *,
    heavy: bool,
    sess_trade: bool,
    labels_lsa: bool,
    blockstart: bool,
) -> Path:
    h = "_heavy" if heavy else ""
    sess = "_sess" if sess_trade else "_nosess"
    lsa = "_lsa" if labels_lsa else ""
    blk = "_blockstart" if blockstart else ""
    name = (
        f"{ds}_volatility_w{w}_lw4_{K_TAG}_{PV}{h}{sess}{lsa}{blk}"
    )
    return RESULTS / name / "report.json"


def baseline_path(ds: str, w: int, kind: str, labels_lsa: bool) -> Path:
    lsa = "_lsa" if labels_lsa else ""
    name = f"baseline_{kind}_{ds}_w{w}_lw4_{K_TAG}_{PV}_heavy{lsa}_sess"
    return RESULTS / name / "report.json"


def load_sup(p: Path) -> dict | None:
    if not p.exists():
        return None
    with p.open() as f:
        r = json.load(f)
    if "backtest" not in r or not r["backtest"]:
        return None
    bm = r["backtest"]["best_model"]
    st = r["backtest"]["strategy"]
    n_sig = st["n_long_signals"] + st["n_short_signals"]
    ost = st["long_signals_on_session_start"] + st["short_signals_on_session_start"]
    pr_m = r["models"][bm]["min"]["overall"]["pr_auc"]
    pr_x = r["models"][bm]["max"]["overall"]["pr_auc"]
    return {
        "path": str(p),
        "window": int(re.search(r"_w(\d+)_", p.parts[-2]).group(1)),
        "pr_min": pr_m,
        "pr_max": pr_x,
        "sharpe": st["sharpe"],
        "mdd": st["max_drawdown"],
        "tr": st["total_return"],
        "bh": r["backtest"]["buy_and_hold"]["total_return"],
        "on_pct": 100 * ost / n_sig if n_sig else 0.0,
    }


def load_base(p: Path) -> dict | None:
    if not p.exists():
        return None
    with p.open() as f:
        r = json.load(f)
    bt = r.get("backtest") or {}
    if not bt:
        return None
    st = bt["strategy"]
    n_sig = st["n_long_signals"] + st["n_short_signals"]
    ost = st["long_signals_on_session_start"] + st["short_signals_on_session_start"]
    return {
        "path": str(p),
        "window": int(re.search(r"_w(\d+)_", p.parts[-2]).group(1)),
        "pr_min": r["min"]["mean_pr_auc"],
        "pr_max": r["max"]["mean_pr_auc"],
        "sharpe": st["sharpe"],
        "mdd": st["max_drawdown"],
        "tr": st["total_return"],
        "bh": bt["buy_and_hold"]["total_return"],
        "on_pct": 100 * ost / n_sig if n_sig else 0.0,
    }


def best_over_windows(
    paths: list[Path],
    loader,
) -> dict | None:
    cand = [loader(p) for p in paths]
    cand = [c for c in cand if c is not None]
    if not cand:
        return None
    return max(cand, key=lambda d: d["tr"])


def canonical_bh(ds: DS) -> float:
    ohlc = load_ohlcv(ds.key)
    return ev.buy_and_hold_stats(ohlc, periods_per_year=ds.periods)["total_return"]


def row_to_tex2(label: str, d: dict, bh_same: float) -> str:
    w = int(d["window"])
    pm_body = f"{d['pr_min']:.3f}"[2:].replace(".", "{,}")
    px_body = f"{d['pr_max']:.3f}"[2:].replace(".", "{,}")
    sh = f"{d['sharpe']:.2f}".replace(".", "{,}")
    mdd_s = f"{abs(d['mdd']) * 100:.1f}".replace(".", "{,}")
    trs = f"{d['tr'] * 100:+.1f}".replace(".", "{,}")
    bhs = f"{bh_same * 100:+.1f}".replace(".", "{,}")
    on = f"{d['on_pct']:.1f}".replace(".", "{,}")
    return (
        f"{label} & ${w}$ & $0{{,}}{pm_body}$ & $0{{,}}{px_body}$ & "
        f"${sh}$ & $-{mdd_s}\\,\\%$ & ${trs}\\,\\%$ & ${bhs}\\,\\%$ & ${on}\\,\\%$ \\\\"
    )


def collect_block(ds: DS) -> dict[str, dict]:
    dsn = ds.key
    lsa = ds.labels_lsa
    bh0 = canonical_bh(ds)
    out: dict[str, dict] = {}
    specs = [
        ("heavy, session-aware (trade)", lambda w: sup_path(dsn, w, heavy=True, sess_trade=True, labels_lsa=lsa, blockstart=False)),
        ("light, session-aware (trade)", lambda w: sup_path(dsn, w, heavy=False, sess_trade=True, labels_lsa=lsa, blockstart=False)),
        ("heavy, без сессии (trade)", lambda w: sup_path(dsn, w, heavy=True, sess_trade=False, labels_lsa=lsa, blockstart=False)),
        ("light, без сессии (trade)", lambda w: sup_path(dsn, w, heavy=False, sess_trade=False, labels_lsa=lsa, blockstart=False)),
        ("heavy, блок старта сессии", lambda w: sup_path(dsn, w, heavy=True, sess_trade=True, labels_lsa=lsa, blockstart=True)),
        ("light, блок старта сессии", lambda w: sup_path(dsn, w, heavy=False, sess_trade=True, labels_lsa=lsa, blockstart=True)),
        ("baseline KMeans, heavy", lambda w: baseline_path(dsn, w, "kmeans", lsa)),
        ("baseline DBSCAN, heavy", lambda w: baseline_path(dsn, w, "dbscan", lsa)),
    ]
    for name, fn in specs:
        paths = [fn(w) for w in WINDOWS]
        if name.startswith("baseline"):
            b = best_over_windows(paths, load_base)
        else:
            b = best_over_windows(paths, load_sup)
        if b:
            b["label"] = name
            b["bh"] = bh0
        out[name] = b
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--windows",
        default="64,96,128",
        help="Comma-separated windows considered when picking best TR row.",
    )
    args = ap.parse_args()
    global WINDOWS
    WINDOWS = [int(x.strip()) for x in args.windows.split(",") if x.strip()]

    all_out: dict[str, dict] = {}
    tex_blocks: list[str] = []
    for ds in DATASETS:
        block = collect_block(ds)
        all_out[ds.key] = block
        lines = [f"        \\multicolumn{{9}}{{l}}{{\\textbf{{{ds.latex}}}}} \\\\", "        \\midrule"]
        bh_ref = canonical_bh(ds)
        for _k, row in block.items():
            if row is None:
                raise RuntimeError(f"Missing results for {ds.key} / {_k}")
            lines.append("        " + row_to_tex2(_k, row, bh_ref))
        tex_blocks.append("\n".join(lines))
    out_json = ROOT / "results" / "table33_best.json"
    with open(out_json, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"[+] wrote {out_json}")
    print("\n% --- paste below into vkr.tex tab:results ---\n")
    print("\n\\midrule\n".join(tex_blocks))


if __name__ == "__main__":
    main()
