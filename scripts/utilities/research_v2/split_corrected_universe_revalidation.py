"""DECISIVE split-corrected re-validation of the broad-universe DCF discount (Step 3).

The frozen `final_universe_backtest.py` used the split-defective PIT harness (adjusted price x
as-reported pre-split shares), so its cached `discount_to_intrinsic` is inflated for splitters'
pre-split windows -- exactly the artifact that flipped the n=50 growth-tranche discount IC from
+0.12 to -0.08. This script tests whether the BROAD-universe edge (the '10,057-obs PASS') survives
the same correction.

No MCMC / no re-valuation: discount = iv/px - 1, so iv/px = 1+discount and the split-consistent
discount is (1+discount)/cum_split - 1, where cum_split = product of WHITELISTED yfinance split
factors after the shares' availability date (cutoff - 75d filing lag). Recomputes the same per-date
cross-sectional Spearman IC as final_universe_backtest.metrics(), original vs corrected, overall and
by sector family.

Usage: PYTHONPATH=src python scripts/utilities/research_v2/split_corrected_universe_revalidation.py
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "reports" / "final_universe_backtest.json"
OUT = ROOT / "reports" / "research_v2" / "split_corrected_universe_revalidation.json"
SPLIT_CACHE = ROOT / "reports" / "research_v2" / "_cache" / "universe_splits.json"
LAG_DAYS = 75
HORIZONS = (1, 3, 5)
WHITELIST = {2, 3, 4, 5, 6, 7, 8, 10, 15, 20}


def _clean_factor(ratio: float) -> float:
    r = float(ratio)
    f = round(r)
    return float(f) if (f in WHITELIST and abs(r - f) / f <= 0.15) else 1.0


def fetch_splits(tickers: list[str]) -> dict[str, list]:
    """Whitelisted forward-split events per ticker (yfinance), cached + resumable."""
    cache: dict[str, list] = {}
    if SPLIT_CACHE.exists():
        cache = json.loads(SPLIT_CACHE.read_text(encoding="utf-8"))
    import pandas as pd
    import yfinance as yf
    todo = [t for t in tickers if t not in cache]
    for i, tk in enumerate(todo):
        evs = []
        try:
            s = yf.Ticker(tk).splits
            if s is not None and len(s):
                for ts, ratio in s.items():
                    f = _clean_factor(ratio)
                    if f > 1.0:
                        evs.append([pd.Timestamp(ts).strftime("%Y-%m-%d"), f])
        except Exception:  # noqa: BLE001
            evs = []
        cache[tk] = evs
        if (i + 1) % 25 == 0:
            SPLIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            SPLIT_CACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  splits {i+1}/{len(todo)} fetched", flush=True)
        time.sleep(0.15)
    SPLIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def _cum_after(events: list, cutoff: date) -> float:
    """Product of whitelisted split factors dated after the shares' availability (cutoff-75d)."""
    ref = cutoff - timedelta(days=LAG_DAYS)
    f = 1.0
    for ds, fac in events:
        if date.fromisoformat(ds) > ref:
            f *= float(fac)
    return f


def metrics(rows: list[dict], y: int, key: str) -> dict:
    """Per-date cross-sectional Spearman IC (>=4 names/date) + quintile L/S -- mirrors
    final_universe_backtest.metrics()."""
    rk = f"fwd_{y}y"
    rows = [o for o in rows if isinstance(o.get(rk), (int, float)) and isinstance(o.get(key), (int, float))]
    if len(rows) < 12:
        return {"n": len(rows), "note": "insufficient"}
    sig = np.array([o[key] for o in rows])
    fwd = np.array([o[rk] for o in rows])
    sp = stats.spearmanr(sig, fwd)
    bydate: dict[str, list] = {}
    for o in rows:
        bydate.setdefault(o["date"], []).append((o[key], o[rk]))
    ics = [stats.spearmanr([p[0] for p in v], [p[1] for p in v]).statistic for v in bydate.values() if len(v) >= 4]
    ics = [x for x in ics if np.isfinite(x)]
    order = np.argsort(sig)
    qm = [float(np.mean(fwd[idx])) for idx in np.array_split(order, 5)]
    ic = float(np.mean(ics)) if ics else None
    ic_std = float(np.std(ics)) if ics else None
    t = (ic / ic_std * np.sqrt(len(ics))) if (ic is not None and ic_std) else None
    return {"n": len(rows), "IC": ic, "IC_t": t, "ic_periods": len(ics),
            "spearman": float(sp.statistic), "long_short_q5_q1": qm[-1] - qm[0],
            "quintiles": [round(x, 4) for x in qm]}


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    obs = payload["observations"]
    tickers = sorted({o["ticker"] for o in obs})
    print(f"loaded {len(obs)} obs / {len(tickers)} tickers; fetching whitelisted splits ...", flush=True)
    splits = fetch_splits(tickers)
    n_splitters = sum(1 for t in tickers if splits.get(t))
    print(f"  {n_splitters} tickers have >=1 whitelisted split", flush=True)

    affected = 0
    for o in obs:
        cum = _cum_after(splits.get(o["ticker"], []), date.fromisoformat(o["date"]))
        d = o.get("discount_to_intrinsic")
        o["discount_corrected"] = ((1.0 + d) / cum - 1.0) if isinstance(d, (int, float)) else None
        o["cum_split"] = cum
        if cum != 1.0:
            affected += 1
    print(f"  {affected}/{len(obs)} obs received a split correction", flush=True)

    overall = {f"{y}y": {"original": metrics(obs, y, "discount_to_intrinsic"),
                         "corrected": metrics(obs, y, "discount_corrected")} for y in HORIZONS}
    # by sector family
    fams: dict[str, list] = {}
    for o in obs:
        fams.setdefault(o.get("sector", "unknown"), []).append(o)
    by_family = {fam: {f"{y}y": {"original": metrics(rows, y, "discount_to_intrinsic"),
                                 "corrected": metrics(rows, y, "discount_corrected")} for y in HORIZONS}
                 for fam, rows in fams.items()}

    result = {"n_obs": len(obs), "n_tickers": len(tickers), "n_splitters": n_splitters,
              "n_obs_corrected": affected, "overall": overall, "by_sector_family": by_family}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    def _f(v, p=4):
        return f"{v:+.{p}f}" if isinstance(v, (int, float)) else "   n/a"
    print("\n=== BROAD UNIVERSE: discount IC, original vs split-corrected ===")
    print(f"{'horizon':>7} {'n':>6} {'IC_orig':>9} {'t_orig':>7} {'IC_corr':>9} {'t_corr':>7} "
          f"{'LS_orig':>8} {'LS_corr':>8}")
    for y in HORIZONS:
        o_, c_ = overall[f"{y}y"]["original"], overall[f"{y}y"]["corrected"]
        print(f"{y}y".rjust(7) + f" {o_.get('n',0):>6} {_f(o_.get('IC')):>9} {_f(o_.get('IC_t'),2):>7} "
              f"{_f(c_.get('IC')):>9} {_f(c_.get('IC_t'),2):>7} {_f(o_.get('long_short_q5_q1')):>8} "
              f"{_f(c_.get('long_short_q5_q1')):>8}")
    print("\n=== BY SECTOR FAMILY (3y IC: original -> corrected) ===")
    for fam, d in sorted(by_family.items()):
        o3 = d["3y"]["original"]; c3 = d["3y"]["corrected"]
        print(f"  {fam:28} n={o3.get('n',0):>5}  {_f(o3.get('IC')):>9} -> {_f(c3.get('IC')):>9}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
