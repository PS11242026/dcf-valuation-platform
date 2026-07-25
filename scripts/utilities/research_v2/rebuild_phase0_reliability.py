"""Rebuild Phase 0a -- corrected SECTOR RELIABILITY, validated individually (PIT, no lookahead).

The blast radius showed the discount is *conditional*: real in some sector families (staples), absent
in others (semis). This builds sector reliability as a deployable, point-in-time factor and tests
whether conditioning the discount on it beats the raw discount.

PIT reliability(family, t) = mean per-date Spearman IC of the discount vs realized forward return,
over that family's quarterly cutoffs whose horizon has ALREADY realized by t (cutoff + y <= t). No
lookahead: only realized predictions enter. Then we compare, on the corrected panel:
  - raw discount (baseline, same date subset)
  - reliability-GATED discount (keep names whose family reliability > 0 as of t)
  - reliability-WEIGHTED discount (discount * max(0, reliability))

Usage: PYTHONPATH=src python scripts/utilities/research_v2/rebuild_phase0_reliability.py
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
import argparse  # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument("--panel", default="final_universe_backtest_split_corrected.json",
                 help="obs panel under reports/research_v2/ (split-corrected default; "
                      "final_universe_backtest_fully_corrected.json for the tightened panel)")
_args, _ = _ap.parse_known_args()
print(f"[panel: {_args.panel}]")
OBS = json.loads((ROOT / "reports" / "research_v2" / _args.panel).read_text())["observations"]
MIN_REL_PERIODS = 4   # min realized cutoffs before a family reliability is defined


def _per_date_ic(rows, fwd_key, sig_key="discount_to_intrinsic", min_n=4):
    byd = {}
    for o in rows:
        if isinstance(o.get(fwd_key), (int, float)) and isinstance(o.get(sig_key), (int, float)):
            byd.setdefault(o["date"], []).append((o[sig_key], o[fwd_key]))
    out = {}
    for d, pairs in byd.items():
        if len(pairs) >= min_n:
            ic = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
            if np.isfinite(ic):
                out[d] = float(ic)
    return out


def _ic_summary(rows, fwd_key, sig_key, min_n=4):
    series = list(_per_date_ic(rows, fwd_key, sig_key, min_n).values())
    if len(series) < 4:
        return {"n_dates": len(series), "ic": None, "t": None}
    a = np.array(series)
    return {"n_dates": len(a), "ic": float(a.mean()),
            "t": float(a.mean() / a.std() * np.sqrt(len(a))) if a.std() > 0 else None}


def main() -> None:
    for o in OBS:
        o["_d"] = date.fromisoformat(o["date"])
    families = sorted({o["sector"] for o in OBS})

    for y in (3, 5):
        fwd = f"fwd_{y}y"
        # per-(family, cutoff) IC of the discount vs realized fwd
        fam_dates: dict[str, dict[str, float]] = {}
        for fam in families:
            fam_dates[fam] = _per_date_ic([o for o in OBS if o["sector"] == fam], fwd)
        horizon = timedelta(days=round(365.25 * y))

        def reliability(fam, t: date):
            vals = [ic for ds, ic in fam_dates[fam].items()
                    if date.fromisoformat(ds) + horizon <= t]
            return float(np.mean(vals)) if len(vals) >= MIN_REL_PERIODS else None

        # attach PIT reliability + conditional signals
        rows = []
        for o in OBS:
            if not isinstance(o.get(fwd), (int, float)) or not isinstance(o.get("discount_to_intrinsic"), (int, float)):
                continue
            rel = reliability(o["sector"], o["_d"])
            r = dict(o)
            r["_rel"] = rel
            r["disc_gated"] = o["discount_to_intrinsic"] if (rel is not None and rel > 0) else None
            r["disc_weighted"] = (o["discount_to_intrinsic"] * max(0.0, rel)) if rel is not None else None
            rows.append(r)

        defined = [r for r in rows if r["_rel"] is not None]   # fair comparison subset
        print(f"\n=== {y}y === (obs with PIT reliability defined: {len(defined)} / {len(rows)})")
        raw_full = _ic_summary(rows, fwd, "discount_to_intrinsic")
        raw_sub = _ic_summary(defined, fwd, "discount_to_intrinsic")
        gated = _ic_summary(defined, fwd, "disc_gated")
        weighted = _ic_summary(defined, fwd, "disc_weighted")
        for label, s in (("raw discount (all dates)", raw_full),
                         ("raw discount (rel-defined subset)", raw_sub),
                         ("reliability-GATED discount", gated),
                         ("reliability-WEIGHTED discount", weighted)):
            ic = f"{s['ic']:+.4f}" if s["ic"] is not None else "  n/a"
            t = f"{s['t']:+.2f}" if s["t"] is not None else " n/a"
            print(f"  {label:38} IC={ic}  t={t}  (n_dates={s['n_dates']})")

        # final-date family reliabilities (most-informed estimate, for interpretation)
        if y == 3:
            tmax = max(o["_d"] for o in OBS)
            rels = {fam: reliability(fam, tmax) for fam in families}
            print("  family reliability (PIT, as of last cutoff):")
            for fam, rl in sorted(rels.items(), key=lambda x: -(x[1] if x[1] is not None else -9)):
                print(f"    {fam:32} {rl:+.4f}" if rl is not None else f"    {fam:32}   n/a")


if __name__ == "__main__":
    main()
