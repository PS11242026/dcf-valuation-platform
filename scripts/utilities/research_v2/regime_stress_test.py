"""Regime stress-test -- is the corrected discount's 2020+ edge real, or value-factor timing?

The user's decisive gate before any Phase 0b build. On the fully-corrected panel the discount
anti-predicts pre-2020 (3y −0.11) and predicts 2020+ (peaking at the COVID-rebound cutoffs). Three
things could make that a mirage:
  (1) overlapping 3–5y forward windows inflate the naive t-stats (need Newey–West / block bootstrap);
  (2) it could be the value factor (Book/Market) switching on, not DCF skill;
  (3) it could live entirely in one macro episode (2020 meltup) with no other support.

Tests, per horizon, on the FULL-corrected discount joined to the cached PIT factor panel:
  A. Overlap-aware significance -- obs-weighted IC + Newey–West t (lag = horizon quarters) +
     block-bootstrap 95% CI, for the discount AND Book/Market, in pre-2020 vs 2020+ buckets, plus the
     finer regime taxonomy (where in 2020+ does the signal live?).
  B. Value control -- per-date OLS fwd ~ rank_z(discount) [+ rank_z(B/M)]; does the discount's coef
     survive in 2020+ once B/M is controlled, or does B/M absorb it?
  C. Holdout -- pre-2020 (train) vs 2020+ (holdout), side by side, raw IC and B/M-residualized coef.

Eval-only; no MCMC, no src import. Reads the corrected panel + reports/_factor_panel.json.
Writes reports/research_v2/regime_stress_test.json (+ stdout tables).

Usage: python scripts/utilities/research_v2/regime_stress_test.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.validation_common as C  # noqa: E402

RV = ROOT / "reports" / "research_v2"
PANEL = RV / "final_universe_backtest_fully_corrected.json"
FACTORS = ROOT / "reports" / "_factor_panel.json"
CUT = "2020-01-01"   # pre-2020 (train) vs 2020+ (holdout)
BUCKETS = [("pre_2020", "0000-00-00", CUT), ("y2020_plus", CUT, "9999-99-99")]


def _rank_z(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x)
    s = r.std()
    return (r - r.mean()) / s if s > 0 else r * 0.0


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def load_joined() -> list[dict]:
    obs = json.loads(PANEL.read_text(encoding="utf-8"))["observations"]
    pmap = {(p["ticker"], p["date"]): p for p in json.loads(FACTORS.read_text(encoding="utf-8"))}
    out, n_bm = [], 0
    for o in obs:
        f = pmap.get((o["ticker"], o["date"]))
        if f is None:
            continue
        r = dict(o)
        r["book_to_market"] = f.get("book_to_market")
        if isinstance(r["book_to_market"], (int, float)):
            n_bm += 1
        out.append(r)
    print(f"joined {len(out)}/{len(obs)} corrected obs to factor panel; with Book/Market: {n_bm}")
    return out


def _ic_bucket(series, lo, hi, lag):
    sel = [(d, n, ic) for (d, n, ic) in series if lo <= d < hi]
    if len(sel) < 3:
        return {"n_dates": len(sel), "ic_obs": None, "ic_eq": None, "nw_t": None}
    nw = C.newey_west_tstat([ic for _, _, ic in sel], lag)
    return {"n_dates": len(sel), "ic_obs": C.obs_weighted_ic(sel),
            "ic_eq": nw.get("mean"), "nw_t": nw.get("t_stat")}


def _coef_series(joined, y, regressors):
    """Per-date OLS fwd ~ [rank_z(r) for r in regressors]. Returns list of (date, n, coef0, coef1...)."""
    rk = f"fwd_{y}y"
    byd: dict[str, list[dict]] = defaultdict(list)
    for o in joined:
        if isinstance(o.get(rk), (int, float)) and all(isinstance(o.get(k), (int, float)) for k in regressors):
            byd[o["date"]].append(o)
    out = []
    for dt in sorted(byd):
        g = byd[dt]
        if len(g) < 20:
            continue
        yv = _z(np.array([o[rk] for o in g], float))
        X = np.column_stack([np.ones(len(g))] +
                            [_rank_z(np.array([o[k] for o in g], float)) for k in regressors])
        try:
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        except Exception:  # noqa: BLE001
            continue
        out.append((dt, len(g), *[float(b) for b in beta[1:]]))
    return out


def _coef_bucket(coefs, idx, lo, hi, lag):
    vals = [c[2 + idx] for c in coefs if lo <= c[0] < hi]
    if len(vals) < 3:
        return {"n_dates": len(vals), "coef": None, "nw_t": None}
    nw = C.newey_west_tstat(vals, lag)
    return {"n_dates": len(vals), "coef": nw.get("mean"), "nw_t": nw.get("t_stat")}


def main():
    joined = load_joined()
    result = {"cut": CUT}
    for y in (3, 5):
        lag = C.HORIZON_QUARTERS[y]
        print("\n" + "=" * 84)
        print(f"  HORIZON {y}y   (Newey-West lag = {lag} quarters)")
        print("=" * 84)

        # --- A. overlap-aware significance, discount vs Book/Market, by regime bucket ---
        s_disc = C.per_date_ic_series(joined, y, "discount_to_intrinsic")
        s_bm = C.per_date_ic_series(joined, y, "book_to_market")
        print("\n(A) Overlap-aware IC  [obs-wtd IC | equal-wtd | Newey-West t | block-boot 95% CI]")
        print(f"{'bucket':>12} {'signal':>13} {'n_d':>4} {'IC_obs':>9} {'IC_eq':>9} {'NW_t':>7}  bootstrap95")
        A = {}
        for name, lo, hi in BUCKETS:
            rows_b = [o for o in joined if lo <= o["date"] < hi]
            for sig, series in (("discount", s_disc), ("book_to_market", s_bm)):
                b = _ic_bucket(series, lo, hi, lag)
                ci = C.block_bootstrap_ci(
                    rows_b, y,
                    lambda r, sk=("discount_to_intrinsic" if sig == "discount" else "book_to_market"):
                        C.obs_weighted_ic(C.per_date_ic_series(r, y, sk)),
                    lag)
                cis = (f"[{ci['ci95'][0]:+.3f},{ci['ci95'][1]:+.3f}]{'*' if ci.get('excludes_zero') else ''}"
                       if ci.get("ci95") else "n/a")
                io = f"{b['ic_obs']:+.4f}" if b["ic_obs"] is not None else "   n/a"
                ie = f"{b['ic_eq']:+.4f}" if b["ic_eq"] is not None else "   n/a"
                tt = f"{b['nw_t']:+.2f}" if b["nw_t"] is not None else "  n/a"
                print(f"{name:>12} {sig:>13} {b['n_dates']:>4} {io:>9} {ie:>9} {tt:>7}  {cis}")
                A[f"{name}:{sig}"] = {**b, "bootstrap_ci95": ci.get("ci95"),
                                      "bootstrap_excludes_zero": ci.get("excludes_zero")}

        # --- finer regime taxonomy: where does the 2020+ discount signal live? ---
        print("\n  discount IC by fine regime (obs-wtd | NW t):")
        reg = {}
        for rname, lo, hi in C.REGIMES:
            sel = [(d, n, ic) for (d, n, ic) in s_disc if lo <= d <= hi]
            if not sel:
                continue
            nw = C.newey_west_tstat([ic for _, _, ic in sel], lag)
            reg[rname] = {"n_dates": len(sel), "ic_obs": C.obs_weighted_ic(sel), "nw_t": nw.get("t_stat")}
            print(f"      {rname:34} n_d={len(sel):>2}  IC={C.obs_weighted_ic(sel):+.4f}  "
                  f"NW_t={(nw.get('t_stat') or 0):+.2f}")

        # --- B. value control: discount alone vs discount | Book/Market ---
        uni = _coef_series(joined, y, ["discount_to_intrinsic"])
        ctl = _coef_series(joined, y, ["discount_to_intrinsic", "book_to_market"])
        print("\n(B) Value control -- per-date OLS coef (rank-z), Newey-West t:")
        print(f"{'bucket':>12} {'disc|alone':>16} {'disc|B/M ctrl':>16} {'B/M|disc ctrl':>16}")
        B = {}
        for name, lo, hi in BUCKETS:
            du = _coef_bucket(uni, 0, lo, hi, lag)
            dc = _coef_bucket(ctl, 0, lo, hi, lag)
            bc = _coef_bucket(ctl, 1, lo, hi, lag)

            def cell(d):
                return (f"{d['coef']:+.3f}(t{d['nw_t']:+.1f})"
                        if d["coef"] is not None and d["nw_t"] is not None else "n/a")
            print(f"{name:>12} {cell(du):>16} {cell(dc):>16} {cell(bc):>16}")
            B[name] = {"discount_alone": du, "discount_ctrl_bm": dc, "bm_ctrl_disc": bc}

        result[f"{y}y"] = {"bucket_ic": A, "fine_regime": reg, "value_control": B}

    out = RV / "regime_stress_test.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
