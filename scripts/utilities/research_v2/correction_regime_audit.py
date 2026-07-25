"""Phase-0a tightening -- corrected-panel REGIME test + split-inventory cross-check (offline, reproducible).

Two questions the user posed after adding the dividend component to the price-basis correction:

  (A) REGIME. Does the discount's predictive power concentrate in 2019+ cutoffs survive the FULL
      (split+dividend) correction, and is the early-negative / late-positive split CORRECTION-INDUCED
      (created by fixing the splitters' pre-split windows, which sit in early cutoffs) or PRE-EXISTING
      (visible in the original, uncorrected discount too)?  -> per-cutoff-year per-date IC for
      discount_original vs discount_to_intrinsic (full), plus a pre-2019 / 2019+ split with t-stats.

  (B) SPLIT CROSS-CHECK. The offline, reproducible companion to the SEC share-jump scan: every
      cum_split>1 actually applied in the panel must be EXACTLY explained by clean, in-window,
      whitelisted (standard-ratio) splits from the cached yfinance .splits inventory -- none spurious,
      none missing. Also lists the rejected (non-whitelisted) backtest-era ratios that were correctly
      excluded (e.g. BKNG 25x, DHR spin-offs), which the SEC scan likewise excluded.

No network. Append-only (prints to stdout; writes one new JSON, no prior artifact overwritten).

Usage: python scripts/utilities/research_v2/correction_regime_audit.py
       [--panel final_universe_backtest_fully_corrected.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RV = ROOT / "reports" / "research_v2"
WHITELIST = {2, 3, 4, 5, 6, 7, 8, 10, 15, 20}

_ap = argparse.ArgumentParser()
_ap.add_argument("--panel", default="final_universe_backtest_fully_corrected.json")
_args, _ = _ap.parse_known_args()


def _clean(ratio: float):
    r = float(ratio)
    f = round(r)
    return float(f) if (f in WHITELIST and abs(r - f) / f <= 0.15) else None


def _per_date_ic(rows, fwd_key, sig_key):
    byd: dict[str, list] = defaultdict(list)
    for o in rows:
        if isinstance(o.get(fwd_key), (int, float)) and isinstance(o.get(sig_key), (int, float)):
            byd[o["date"]].append((o[sig_key], o[fwd_key]))
    out = {}
    for d, pairs in byd.items():
        if len(pairs) >= 4:
            ic = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
            if np.isfinite(ic):
                out[d] = float(ic)
    return out


def _summ(vals):
    a = np.array(list(vals), dtype=float)
    if len(a) == 0:
        return {"n": 0, "ic": None, "t": None}
    if len(a) < 2 or a.std() == 0:
        return {"n": int(len(a)), "ic": float(a.mean()), "t": None}
    return {"n": int(len(a)), "ic": float(a.mean()), "t": float(a.mean() / a.std() * np.sqrt(len(a)))}


def regime(obs, out):
    print("\n" + "=" * 80)
    print("(A) REGIME TEST  --  per-cutoff-year per-date IC:  original  vs  FULL (split+dividend)")
    print("=" * 80)
    out["regime"] = {}
    for y in (3, 5):
        fwd = f"fwd_{y}y"
        ic_o = _per_date_ic(obs, fwd, "discount_original")
        ic_f = _per_date_ic(obs, fwd, "discount_to_intrinsic")
        years = sorted({d[:4] for d in set(ic_o) | set(ic_f)})
        print(f"\n--- {y}y horizon ---")
        print(f"{'cutoff yr':>9} {'n_dates':>8} {'IC original':>14} {'IC full':>14}   (full - orig)")
        yr_rows = {}
        for yr in years:
            so = _summ([v for d, v in ic_o.items() if d[:4] == yr])
            sf = _summ([v for d, v in ic_f.items() if d[:4] == yr])
            io = f"{so['ic']:+.4f}" if so["ic"] is not None else "   n/a"
            iff = f"{sf['ic']:+.4f}" if sf["ic"] is not None else "   n/a"
            delta = (f"{sf['ic'] - so['ic']:+.4f}"
                     if so["ic"] is not None and sf["ic"] is not None else "")
            print(f"{yr:>9} {sf['n']:>8} {io:>14} {iff:>14}   {delta}")
            yr_rows[yr] = {"n_dates": sf["n"], "ic_original": so["ic"], "ic_full": sf["ic"]}
        buckets = {}
        for lbl, cond in (("pre_2019", lambda d: d[:4] < "2019"),
                          ("y2019_plus", lambda d: d[:4] >= "2019")):
            so = _summ([v for d, v in ic_o.items() if cond(d)])
            sf = _summ([v for d, v in ic_f.items() if cond(d)])

            def fmt(s):
                if s["ic"] is None:
                    return "n/a"
                t = f", t{s['t']:+.2f}" if s["t"] is not None else ""
                return f"{s['ic']:+.4f}{t} (n={s['n']})"

            print(f"   {lbl:11}:  original {fmt(so):28}  full {fmt(sf)}")
            buckets[lbl] = {"original": so, "full": sf}
        out["regime"][f"{y}y"] = {"by_year": yr_rows, "buckets": buckets}


def split_audit(obs, out):
    print("\n" + "=" * 80)
    print("(B) SPLIT-INVENTORY CROSS-CHECK  --  applied cum_split vs clean in-window split inventory")
    print("=" * 80)
    splits = json.loads((RV / "_cache" / "universe_splits.json").read_text())

    first_cut: dict[str, str] = {}
    max_cs: dict[str, float] = {}
    n_corr_obs: dict[str, int] = defaultdict(int)
    for o in obs:
        t, d = o["ticker"], o["date"]
        first_cut[t] = min(first_cut.get(t, d), d)
        cs = float(o.get("cum_split", 1.0) or 1.0)
        max_cs[t] = max(max_cs.get(t, 1.0), cs)
        if cs != 1.0:
            n_corr_obs[t] += 1
    corrected = {t for t, c in max_cs.items() if c > 1.0}

    # expected max cum_split = product of clean ratios dated after (earliest cutoff - 100d filing lag)
    expected: dict[str, float] = {}
    for t in first_cut:
        fc = date.fromisoformat(first_cut[t]) - timedelta(days=100)
        prod = 1.0
        for ds, r in splits.get(t, []):
            f = _clean(r)
            if f and date.fromisoformat(ds) > fc:
                prod *= f
        if prod > 1.0:
            expected[t] = prod

    # reconcile
    matches, mismatches = [], []
    for t in sorted(corrected | set(expected)):
        a, b = max_cs.get(t, 1.0), expected.get(t, 1.0)
        (matches if abs(a - b) < 0.01 else mismatches).append((t, a, b, n_corr_obs.get(t, 0)))

    # rejected (non-whitelisted) ratios in the backtest era -- correctly NOT applied
    rejected = []
    for t, evs in splits.items():
        for ds, r in evs:
            if ds >= "2014-01-01" and float(r) != 1.0 and _clean(r) is None:
                rejected.append((t, ds, float(r)))

    print(f"  tickers corrected in panel (cum_split>1): {len(corrected)}")
    print(f"  tickers expected from clean in-window splits: {len(expected)}")
    print(f"  EXACT reconciliation: {len(matches)} match, {len(mismatches)} mismatch")
    only_panel = sorted(corrected - set(expected))
    only_inv = sorted(set(expected) - corrected)
    if only_panel:
        print(f"  ! corrected in panel but no clean in-window split (spurious?): {only_panel}")
    if only_inv:
        print(f"  ! clean in-window split but NOT corrected in panel (missed?): {only_inv}")
    if mismatches:
        print("  ! ratio mismatches (ticker: panel vs inventory):")
        for t, a, b, _ in mismatches:
            print(f"      {t:6} panel {a:g}  inventory {b:g}")
    print("\n  largest applied corrections (panel cum_split, #obs):")
    for t, a, b, n in sorted(matches, key=lambda x: -x[1])[:14]:
        print(f"      {t:6} cum_split {a:>5g}x   obs={n}")
    print(f"\n  rejected non-whitelisted backtest-era ratios (correctly NOT applied): {len(rejected)}")
    for t, ds, r in sorted(rejected):
        print(f"      {t:6} {ds}  ratio {r:g}")

    out["split_audit"] = {
        "n_tickers_corrected_panel": len(corrected),
        "n_tickers_expected_inventory": len(expected),
        "n_exact_match": len(matches),
        "n_mismatch": len(mismatches),
        "corrected_not_explained": only_panel,
        "explained_not_corrected": only_inv,
        "mismatches": [{"ticker": t, "panel": a, "inventory": b} for t, a, b, _ in mismatches],
        "largest": [{"ticker": t, "cum_split": a, "obs": n}
                    for t, a, b, n in sorted(matches, key=lambda x: -x[1])[:20]],
        "rejected_backtest_era": [{"ticker": t, "date": ds, "ratio": r} for t, ds, r in sorted(rejected)],
    }


def main():
    panel = json.loads((RV / _args.panel).read_text())
    obs = panel["observations"]
    print(f"[panel: {_args.panel}]  obs={len(obs)}  "
          f"dividend_adjusted={panel.get('n_obs_dividend_adjusted', 'n/a')}")
    out = {"panel": _args.panel, "n_obs": len(obs)}
    regime(obs, out)
    split_audit(obs, out)
    p = RV / "correction_regime_audit.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
