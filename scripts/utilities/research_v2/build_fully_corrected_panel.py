"""Tighten the corrected panel: add the DIVIDEND component to the split correction (full price basis).

The PIT harness compared a then-basis intrinsic to a yfinance auto_adjust=True price (split AND
dividend adjusted to today). The split correction undid the split part; this undoes the dividend part.

yfinance auto_adjust=False gives Close (split-adjusted only) and Adj Close (split+dividend). So
cumdiv(d) = AdjClose(d)/Close(d) <= 1 is the cumulative dividend-adjustment factor between d and today,
and the fully price-basis-corrected discount is:

    full = (1 + split_corrected_discount) * cumdiv - 1
         = (1 + original_discount) / cum_split * cumdiv - 1

PIT-valid: the contemporaneous traded price is split/dividend invariant; we only re-express it. The
dividend effect is largest for high-dividend value names (e.g. consumer staples) over long windows.

Outputs (append-only; split-corrected panel + Phase-0a artifacts are NOT overwritten):
    reports/research_v2/final_universe_backtest_fully_corrected.json
    reports/research_v2/correction_validation.json   (+ stdout table for correction_validation.md)

Usage: PYTHONPATH=src python scripts/utilities/research_v2/build_fully_corrected_panel.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PANEL = ROOT / "reports" / "research_v2" / "final_universe_backtest_split_corrected.json"
OUT = ROOT / "reports" / "research_v2" / "final_universe_backtest_fully_corrected.json"
VAL = ROOT / "reports" / "research_v2" / "correction_validation.json"
DIVCACHE = ROOT / "reports" / "research_v2" / "_cache" / "dividend_factors.json"
HORIZONS = (1, 3, 5)


def _cumdiv_for(tickers_dates: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """cumdiv(ticker, date) = AdjClose/Close (auto_adjust=False), as-of each needed date. Cached."""
    cache: dict[str, dict[str, float]] = {}
    if DIVCACHE.exists():
        cache = json.loads(DIVCACHE.read_text(encoding="utf-8"))
    import yfinance as yf
    todo = [t for t in tickers_dates if t not in cache]
    for i, tk in enumerate(todo):
        rec: dict[str, float] = {}
        try:
            df = yf.Ticker(tk).history(start="2008-01-01", auto_adjust=False)
            if df is not None and not df.empty and "Adj Close" in df and "Close" in df:
                idx = pd.to_datetime(df.index).tz_localize(None).normalize()
                close = pd.Series(df["Close"].values, index=idx).sort_index()
                adj = pd.Series(df["Adj Close"].values, index=idx).sort_index()
                ratio = (adj / close).replace([np.inf, -np.inf], np.nan).dropna()
                for ds in tickers_dates[tk]:
                    sub = ratio.loc[: pd.Timestamp(ds)]
                    if len(sub):
                        rec[ds] = float(sub.iloc[-1])
        except Exception:  # noqa: BLE001
            rec = {}
        cache[tk] = rec
        if (i + 1) % 25 == 0:
            DIVCACHE.parent.mkdir(parents=True, exist_ok=True)
            DIVCACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  dividend factors {i+1}/{len(todo)}", flush=True)
        time.sleep(0.12)
    DIVCACHE.parent.mkdir(parents=True, exist_ok=True)
    DIVCACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def _per_date_ic(rows, fwd_key, sig_key):
    byd: dict[str, list] = {}
    for o in rows:
        if isinstance(o.get(fwd_key), (int, float)) and isinstance(o.get(sig_key), (int, float)):
            byd.setdefault(o["date"], []).append((o[sig_key], o[fwd_key]))
    ics = []
    for pairs in byd.values():
        if len(pairs) >= 4:
            ic = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
            if np.isfinite(ic):
                ics.append(float(ic))
    if len(ics) < 4:
        return {"n_dates": len(ics), "ic": None, "t": None}
    a = np.array(ics)
    return {"n_dates": len(a), "ic": float(a.mean()),
            "t": float(a.mean() / a.std() * np.sqrt(len(a))) if a.std() > 0 else None}


def main() -> None:
    payload = json.loads(SPLIT_PANEL.read_text(encoding="utf-8"))
    obs = payload["observations"]
    td: dict[str, list[str]] = {}
    for o in obs:
        td.setdefault(o["ticker"], []).append(o["date"])
    print(f"loaded {len(obs)} obs / {len(td)} tickers; fetching dividend factors ...", flush=True)
    div = _cumdiv_for(td)

    n_div = 0
    for o in obs:
        cumdiv = div.get(o["ticker"], {}).get(o["date"], 1.0)
        split_only = o["discount_to_intrinsic"]            # already split-corrected
        original = o.get("discount_original")
        o["discount_split_only"] = split_only
        o["cumdiv"] = cumdiv
        o["discount_div_only"] = ((1.0 + original) * cumdiv - 1.0) if isinstance(original, (int, float)) else None
        o["discount_to_intrinsic"] = (1.0 + split_only) * cumdiv - 1.0   # FULL correction
        if cumdiv != 1.0:
            n_div += 1
    payload["dividend_corrected"] = True
    payload["n_obs_dividend_adjusted"] = n_div
    payload["note"] = ("discount_to_intrinsic = FULLY price-basis corrected (split + dividend). "
                       "discount_split_only / discount_div_only / discount_original retained.")
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # validation IC table: original vs split-only vs div-only vs full
    variants = {"original": "discount_original", "split_only": "discount_split_only",
                "div_only": "discount_div_only", "full": "discount_to_intrinsic"}
    table = {}
    print(f"\n=== correction validation (IC, mean per-date Spearman) -- {n_div} obs dividend-adjusted ===")
    print(f"{'horizon':>7} {'original':>18} {'split_only':>18} {'div_only':>18} {'full':>18}")
    for y in HORIZONS:
        fwd = f"fwd_{y}y"
        row = {k: _per_date_ic(obs, fwd, col) for k, col in variants.items()}
        table[f"{y}y"] = row
        def c(k):
            s = row[k]
            return f"{s['ic']:+.4f}(t{s['t']:+.1f})" if s["ic"] is not None else "n/a"
        print(f"{y}y".rjust(7) + f" {c('original'):>18} {c('split_only'):>18} {c('div_only'):>18} {c('full'):>18}")

    # by sector family at 3y
    fams: dict[str, list] = {}
    for o in obs:
        fams.setdefault(o.get("sector", "unknown"), []).append(o)
    byfam = {}
    print("\n=== by sector family (3y IC): original -> split_only -> full ===")
    for fam, rows in sorted(fams.items()):
        o3 = _per_date_ic(rows, "fwd_3y", "discount_original")["ic"]
        s3 = _per_date_ic(rows, "fwd_3y", "discount_split_only")["ic"]
        f3 = _per_date_ic(rows, "fwd_3y", "discount_to_intrinsic")["ic"]
        byfam[fam] = {"original": o3, "split_only": s3, "full": f3}
        def f(x):
            return f"{x:+.4f}" if isinstance(x, (int, float)) else "  n/a"
        print(f"  {fam:32} {f(o3)} -> {f(s3)} -> {f(f3)}")

    VAL.write_text(json.dumps({"overall": table, "by_family_3y": byfam,
                               "n_obs": len(obs), "n_dividend_adjusted": n_div}, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)} and {VAL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
