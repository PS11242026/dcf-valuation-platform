"""Validation Suites 1, 2, 3, 5 -- computed from the existing 10,057-observation PIT
dataset (reports/final_universe_backtest.json). Evaluation only; no production code touched.

  Suite 1  Sector stratification (+ Payments carve-out; Healthcare noted as excluded)
  Suite 2  Time-period / regime stability (+ horizon-coverage matrix)
  Suite 3  Decile portfolios (HEADLINE: 3Y forward return by valuation decile)
  Suite 5  Freeze-and-test holdout (+ leave-one-family-out)

Headline metric is obs-weighted IC; every IC/spread carries autocorrelation-aware
significance (Newey-West t-stat + contiguous-date block-bootstrap CI). See validation_common.
"""

from __future__ import annotations

import json

import scripts.validation_common as C

REPORT_DIR = C.REPORT_DIR
HORIZONS = (1, 3, 5)

FAMILIES = [
    "software_saas_infra", "internet_platforms_marketplaces", "semiconductors",
    "semiconductor_equipment", "consumer_staples", "consumer_discretionary",
    "industrials_assetlight_services",
]

# Payments carve-out: curated payments/fintech names, intersected with the universe at runtime.
PAYMENTS_TICKERS = {
    "PYPL", "GPN", "FI", "FIS", "WEX", "FLT", "CPAY", "EEFT", "JKHY", "BILL",
    "TOST", "GLBE", "DLO", "PAYO", "FOUR", "NVEI", "RPAY", "PSFE", "MQ", "AFRM",
}

FREEZE_DATE = "2021-06-30"


def _load():
    return C.load_obs()


# ---------------- Suite 1: sector stratification ----------------
def suite1_sector(obs: list[dict]) -> dict:
    out = {
        "note": "obs-weighted IC is the headline; equal_weight_ic_nw_tstat is the (untrustworthy) "
                "equal-weight series t-stat; the obs-weighted IC's significance is the block-bootstrap CI.",
        "healthcare": "EXCLUDED by universe construction (biotech/dev-pharma; DCF shown anti-predictive). "
                      "Not measurable here.",
        "by_family": {}, "payments_carveout": {}, "thin_family_flags": {},
    }
    for fam in FAMILIES:
        frows = [o for o in obs if o.get("sector") == fam]
        out["by_family"][fam] = {f"{y}y": C.core_metrics(frows, y) for y in HORIZONS}
        # thinness: median per-date cross-section at 3y
        series = C.per_date_ic_series(frows, 3)
        import numpy as np
        med_n = float(np.median([n for _, n, _ in series])) if series else 0.0
        out["thin_family_flags"][fam] = {"median_cross_section_3y": med_n,
                                         "thin": med_n < 8}
    pay = [o for o in obs if o.get("ticker") in PAYMENTS_TICKERS]
    present = sorted({o["ticker"] for o in pay})
    out["payments_carveout"] = {
        "tickers_present": present, "n_tickers": len(present),
        "metrics": {f"{y}y": C.core_metrics(pay, y) for y in HORIZONS},
        "note": "Payments names carved by ticker across families; thin cross-section -> interpret cautiously.",
    }
    # ranking by mean obs-weighted IC at 3y/5y
    import numpy as np
    ranking = []
    for fam in FAMILIES:
        i3 = out["by_family"][fam]["3y"].get("ic_obs_weighted")
        i5 = out["by_family"][fam]["5y"].get("ic_obs_weighted")
        score = float(np.nanmean([x for x in (i3, i5) if x is not None])) if (i3 or i5) else None
        ranking.append({"family": fam, "ic_3y": i3, "ic_5y": i5, "mean_3y5y": score})
    ranking.sort(key=lambda d: (d["mean_3y5y"] is not None, d["mean_3y5y"] or -9), reverse=True)
    out["ranking_by_robust_ic"] = ranking
    return out


# ---------------- Suite 2: time-period stability ----------------
def suite2_regimes(obs: list[dict]) -> dict:
    out = {"regimes": [r[0] for r in C.REGIMES], "by_regime": {}, "coverage_matrix": {}}
    for name, lo, hi in C.REGIMES:
        rrows = [o for o in obs if C.regime_of(o["date"]) == name]
        out["by_regime"][name] = {}
        out["coverage_matrix"][name] = {}
        for y in HORIZONS:
            rk = f"fwd_{y}y"
            n_valid = sum(1 for o in rrows if isinstance(o.get(rk), (int, float)))
            out["coverage_matrix"][name][f"{y}y"] = {"n_realized": n_valid,
                                                     "sufficient": n_valid >= 12}
            out["by_regime"][name][f"{y}y"] = (C.core_metrics(rrows, y, with_ci=False)
                                               if n_valid >= 12 else {"n": n_valid, "note": "insufficient"})
    out["horizon_availability"] = {
        "1y": "all regimes", "3y": "regimes 1-4 (cutoffs <= ~2023)",
        "5y": "regimes 1-3 only (cutoffs <= ~2021)",
    }
    return out


# ---------------- Suite 3: decile portfolios (HEADLINE) ----------------
def suite3_deciles(obs: list[dict]) -> dict:
    out = {"construction": "per-date deciles (rank within each cutoff, average across dates); "
                           "time-neutral. n_d>=50 deciles, 30-49 quintiles, <30 dropped. "
                           "Top decile (D10) = most undervalued = expected highest return. "
                           "Equal-weight within decile (no cap weighting; signal-rank test).",
           "overall": {}, "by_regime_spread": {}, "by_sector_3y": {}}
    for y in HORIZONS:
        b = C.per_date_buckets(obs, y)
        sig = C.spread_significance(b["per_date_spread"], y, rows=obs)
        out["overall"][f"{y}y"] = {
            "decile_curve": b["decile_curve"],
            "quintile_fallback": b["quintile_curve_fallback"],
            "spread_top_minus_bottom": sig,
            "pooled_reference": C.pooled_buckets(obs, y),
            "n_dropped_dates": len(b["dropped_dates"]),
        }
        # per-regime breakdown of the per-date spread
        reg = {}
        for name, _, _ in C.REGIMES:
            vals = [s for dt, s in b["per_date_spread"] if C.regime_of(dt) == name]
            if vals:
                import numpy as np
                reg[name] = {"mean_spread": float(np.mean(vals)), "n_dates": len(vals),
                             "pct_positive": float(np.mean([v > 0 for v in vals]))}
        out["by_regime_spread"][f"{y}y"] = reg
    # per-sector 3y decile spread (secondary; thin)
    for fam in FAMILIES:
        frows = [o for o in obs if o.get("sector") == fam]
        b = C.per_date_buckets(frows, 3)
        dc = b["decile_curve"] or b["quintile_curve_fallback"]
        if dc:
            out["by_sector_3y"][fam] = {"curve_obs_weighted": dc.get("obs_weighted"),
                                        "monotonicity": dc.get("monotonicity_spearman"),
                                        "n_dates": dc.get("n_dates"),
                                        "note": "thin per-sector cross-section; interpret cautiously"}
    return out


# ---------------- Suite 5: freeze-and-test holdout ----------------
def suite5_holdout(obs: list[dict]) -> dict:
    visible = [o for o in obs if o["date"] <= FREEZE_DATE]
    holdout = [o for o in obs if o["date"] > FREEZE_DATE]
    out = {
        "freeze_date": FREEZE_DATE,
        "framing": "Model has fixed priors and is NOT fit to data, and the PIT harness strips lookahead -- "
                   "every obs is already out-of-sample in the parameter sense. This holdout guards against "
                   "RESEARCHER overfit. A true post-freeze fully-realized window is primarily a 1Y test "
                   "(partial 3Y); a post-freeze 5Y holdout is impossible (data ends ~2026).",
        "visible_pre_freeze": {}, "holdout_post_freeze": {}, "leave_one_family_out_overall": {},
    }
    for y in HORIZONS:
        out["visible_pre_freeze"][f"{y}y"] = C.core_metrics(visible, y, with_ci=False)
        rk = f"fwd_{y}y"
        n_hold = sum(1 for o in holdout if isinstance(o.get(rk), (int, float)))
        out["holdout_post_freeze"][f"{y}y"] = (C.core_metrics(holdout, y)
                                               if n_hold >= 12 else {"n": n_hold, "note": "insufficient post-freeze realized returns"})
    # leave-one-family-out on overall 3y/5y obs-weighted IC
    for y in (3, 5):
        loo = {}
        for fam in FAMILIES:
            kept = [o for o in obs if o.get("sector") != fam]
            m = C.core_metrics(kept, y, with_ci=False)
            loo[f"ex_{fam}"] = m.get("ic_obs_weighted")
        full = C.core_metrics(obs, y, with_ci=False).get("ic_obs_weighted")
        out["leave_one_family_out_overall"][f"{y}y"] = {"full_universe_ic": full, "leave_one_out": loo}
    return out


def main() -> None:
    obs = _load()
    s1 = suite1_sector(obs)
    s2 = suite2_regimes(obs)
    s3 = suite3_deciles(obs)
    s5 = suite5_holdout(obs)
    (REPORT_DIR / "validation_sector_stratification.json").write_text(json.dumps(s1, indent=2), encoding="utf-8")
    (REPORT_DIR / "validation_time_regimes.json").write_text(json.dumps(s2, indent=2), encoding="utf-8")
    (REPORT_DIR / "validation_decile_portfolios.json").write_text(json.dumps(s3, indent=2), encoding="utf-8")
    (REPORT_DIR / "validation_freeze_holdout.json").write_text(json.dumps(s5, indent=2), encoding="utf-8")

    def f(x):
        return "  None" if x is None else f"{x:6.3f}"
    print("=== SUITE 1: sector obs-weighted IC (1y/3y/5y) ===")
    for fam in FAMILIES:
        m = s1["by_family"][fam]
        thin = " [THIN]" if s1["thin_family_flags"][fam]["thin"] else ""
        print(f"  {fam:34} {f(m['1y'].get('ic_obs_weighted'))} {f(m['3y'].get('ic_obs_weighted'))} {f(m['5y'].get('ic_obs_weighted'))}{thin}")
    pm = s1["payments_carveout"]["metrics"]
    print(f"  {'PAYMENTS (carve-out, n=' + str(s1['payments_carveout']['n_tickers']) + ')':34} {f(pm['1y'].get('ic_obs_weighted'))} {f(pm['3y'].get('ic_obs_weighted'))} {f(pm['5y'].get('ic_obs_weighted'))}")

    print("\n=== SUITE 3: HEADLINE decile portfolios ===")
    for y in HORIZONS:
        o = s3["overall"][f"{y}y"]; dc = o["decile_curve"]; sp = o["spread_top_minus_bottom"]
        if dc:
            print(f"  {y}y: monotonicity={dc['monotonicity_spearman']:.3f}  D1={dc['obs_weighted'][0]:.3f} -> D10={dc['obs_weighted'][-1]:.3f}  "
                  f"D10-D1={sp['mean_spread']:.4f} NW_t={sp['nw_tstat']:.2f} CI={(sp.get('block_bootstrap_ci95') or {}).get('ci95')}")

    print("\n=== SUITE 5: freeze-and-test (visible vs post-freeze holdout) ===")
    for y in HORIZONS:
        v = s5["visible_pre_freeze"][f"{y}y"]; h = s5["holdout_post_freeze"][f"{y}y"]
        print(f"  {y}y: visible IC={f(v.get('ic_obs_weighted'))}  holdout IC={f(h.get('ic_obs_weighted'))} (holdout n={h.get('n')})")

    print("\n=== SUITE 2: regime obs-weighted IC ===")
    for name, _, _ in C.REGIMES:
        r = s2["by_regime"][name]
        print(f"  {name:34} 1y={f(r['1y'].get('ic_obs_weighted'))} 3y={f(r['3y'].get('ic_obs_weighted'))} 5y={f(r['5y'].get('ic_obs_weighted'))}")
    print("\nwrote 4 suite reports to reports/")


if __name__ == "__main__":
    main()
