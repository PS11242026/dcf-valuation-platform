"""Shared helpers for the live paper-portfolio tooling.

Lifted verbatim from paper_portfolio.py so the inception/track script and the live dashboard
share ONE ranking + pricing implementation (no drift between them). No behaviour change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> import scorecard
from scorecard import _sector_reliability  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"
PORT_DIR = REPORT_DIR / "portfolio"
STATE = PORT_DIR / "paper_portfolio.json"
SNAP = REPORT_DIR / "_live_universe_snapshot.json"
DISCOUNT_CLIP = 5.0
BENCHMARKS = ["SPY", "QQQ"]
VALID_SECTORS = {
    "software_saas_infra", "internet_platforms_marketplaces", "semiconductors",
    "semiconductor_equipment", "consumer_staples", "consumer_discretionary",
    "industrials_assetlight_services",
}


def _sector_scores() -> dict[str, float]:
    inst = json.load(open(REPORT_DIR / "final_institutional_validation.json")).get("by_sector_family", {})
    return {f: _sector_reliability(b)["score"] for f, b in inst.items()}


def _live_prices(tickers: list[str]) -> dict[str, float]:
    import yfinance as yf

    out: dict[str, float] = {}
    batch = 40
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        try:
            raw = yf.download(chunk, period="5d", interval="1d", auto_adjust=True,
                              progress=False, threads=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  price batch {i // batch} failed: {exc}")
            continue
        if raw is None or len(raw) == 0:
            continue
        import pandas as pd

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(
            columns={"Close": chunk[0]})
        for t in close.columns:
            s = close[t].dropna()
            if len(s):
                out[t] = float(s.iloc[-1])
    return out


def _rank_books() -> dict:
    snap = json.load(open(SNAP))["snapshot"]
    scores = _sector_scores()
    rows = []
    for tk, v in snap.items():
        d = v.get("discount")
        sec = v.get("sector")
        if not isinstance(d, (int, float)) or abs(d) > DISCOUNT_CLIP or sec not in VALID_SECTORS:
            continue
        rows.append({"ticker": tk, "discount": d, "sector": sec,
                     "intrinsic": v.get("intrinsic"), "snap_price": v.get("price"),
                     "conf": scores.get(sec, 0.0),
                     "sig_discount": d, "sig_discount_x_conf": d * scores.get(sec, 0.0)})
    books = {}
    for sig in ("discount", "discount_x_conf"):
        ranked = sorted(rows, key=lambda r: r[f"sig_{sig}"], reverse=True)
        for n in (10, 20):
            books[f"{sig}_top{n}"] = [dict(r) for r in ranked[:n]]
    return books
