"""Full point-in-time backtest of the canonical DCF on the FINAL >500-ticker universe.

Universe loaded from reports/_universe_tickers.json (built by scripts/build_universe.py):
506 unique US-listed names across 7 sector families (banks/insurance/REITs/telecom/
utilities/biotech/dev-stage pharma excluded by construction).

Methodology (identical to scripts/extended_universe_backtest.py):
  per ticker x quarterly cutoff from 2015 -> truncate fundamentals to cutoff (75d filing
  lag) -> canonical DCF intrinsic (internal WACC + internal terminal growth +
  forecast_mode=state_space, no consensus/provider/scenario lookahead) ->
  discount_to_intrinsic = iv/px - 1 -> forward 1Y/3Y/5Y annualized total returns.

Tolerant of fetch failures (yfinance/SEC rate limits); checkpoints every CHECKPOINT
tickers so a long run preserves partial progress.

Outputs: reports/final_universe_backtest.json, final_archetype_breakdown.json,
final_predictive_power_summary.json, final_universe_verdict.json
Usage: PYTHONPATH=src python scripts/final_universe_backtest.py
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.data_sources.sec import SECClient  # noqa: E402
import scripts.historical_predictive_power as bt  # noqa: E402

REPORT_DIR = ROOT / "reports"
HORIZONS = (1, 3, 5)
CHECKPOINT = 20

_uni = json.loads((REPORT_DIR / "_universe_tickers.json").read_text(encoding="utf-8"))
ALL_TICKERS: list[str] = _uni["tickers"]
FAMILY_OF: dict[str, str] = _uni["family_of"]


def _settings():
    return replace(
        get_settings(), wacc_source="internal", terminal_growth_source="internal",
        forecast_mode="state_space", consensus_weight=0.0,
        monte_carlo_samples=0, scenario_monte_carlo_samples=0, run_sensitivity=False,
    )


def collect() -> tuple[list[dict[str, Any]], dict[str, int]]:
    settings = _settings()
    cutoffs = bt._quarter_ends(2015)
    obs: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    n = len(ALL_TICKERS)
    for i, ticker in enumerate(ALL_TICKERS):
        fam = FAMILY_OF.get(ticker, "unknown")
        try:
            periods, _ = SECClient(settings).financial_periods(ticker)
        except Exception as exc:
            skips[f"{ticker}:sec"] = 1
            print(f"  [{i+1}/{n}] {ticker} ({fam}): SEC fail {str(exc)[:40]}", flush=True)
            continue
        price = bt._price_history(ticker)
        if price is None:
            skips[f"{ticker}:price"] = 1
            print(f"  [{i+1}/{n}] {ticker} ({fam}): no price", flush=True)
            continue
        try:
            beta = bt._beta(ticker)
        except Exception:
            beta = 1.0
        last_pd = price.index[-1].date()
        n_obs = 0
        for cutoff in cutoffs:
            try:
                bundle, _ = bt._build_pit_bundle(ticker, periods, price, beta, cutoff, settings)
            except Exception:
                continue
            if bundle is None:
                continue
            try:
                r = bt._PITPipeline(settings, bundle).run(ticker)
            except Exception:
                continue
            px = bundle.market.price
            if not px or px <= 0:
                continue
            iv = float(r.intrinsic_value_per_share)
            o = {
                "ticker": ticker,
                "sector": fam,
                "company_type": r.diagnostics.get("resolved_company_type") or r.diagnostics.get("company_type"),
                "date": cutoff.isoformat(),
                "discount_to_intrinsic": iv / px - 1.0,
            }
            for y in HORIZONS:
                tgt = cutoff + timedelta(days=round(365.25 * y))
                if tgt > last_pd:
                    continue
                pf = bt._price_asof(price, tgt)
                if pf and pf > 0:
                    o[f"fwd_{y}y"] = float((pf / px) ** (1.0 / y) - 1.0)
                    if iv > 0:
                        o[f"fwd_{y}y_pred"] = float((iv / px) ** (1.0 / y) - 1.0)
            obs.append(o)
            n_obs += 1
        print(f"  [{i+1}/{n}] {ticker} ({fam}): {n_obs} obs", flush=True)
        if (i + 1) % CHECKPOINT == 0:
            _save(obs, skips, partial=True, done=i + 1)
            print(f"  --- checkpoint @ {i+1}/{n}: {len(obs)} obs so far ---", flush=True)
    return obs, skips


def metrics(rows: list[dict[str, Any]], y: int) -> dict[str, Any]:
    rk = f"fwd_{y}y"
    rows = [o for o in rows if isinstance(o.get(rk), (int, float)) and isinstance(o.get("discount_to_intrinsic"), (int, float))]
    if len(rows) < 12:
        return {"n": len(rows), "note": "insufficient"}
    sig = np.array([o["discount_to_intrinsic"] for o in rows])
    fwd = np.array([o[rk] for o in rows])
    sp = stats.spearmanr(sig, fwd)
    bydate: dict[str, list] = {}
    for o in rows:
        bydate.setdefault(o["date"], []).append((o["discount_to_intrinsic"], o[rk]))
    ics = [stats.spearmanr([p[0] for p in v], [p[1] for p in v]).statistic for v in bydate.values() if len(v) >= 4]
    ics = [x for x in ics if np.isfinite(x)]
    order = np.argsort(sig)
    qs = np.array_split(order, 5)
    qmeans = [float(np.mean(fwd[idx])) for idx in qs]
    med = float(np.median(fwd))
    under = fwd[sig > 0]
    pk = f"fwd_{y}y_pred"
    cal_rows = [o for o in rows if isinstance(o.get(pk), (int, float))]
    if len(cal_rows) >= 10:
        pr = np.array([o[pk] for o in cal_rows]); re_ = np.array([o[rk] for o in cal_rows])
        cal = {"rmse": float(np.sqrt(np.mean((re_ - pr) ** 2))), "bias": float(np.mean(re_ - pr))}
    else:
        cal = {"rmse": None, "bias": None}
    return {
        "n": len(rows),
        "spearman": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "information_coefficient": float(np.mean(ics)) if ics else None,
        "ic_std": float(np.std(ics)) if ics else None,
        "ic_periods": len(ics),
        "hit_rate_beats_median": float(np.mean(under > med)) if len(under) else None,
        "quintile_fwd_returns": [round(x, 4) for x in qmeans],
        "long_short_q5_minus_q1": float(qmeans[-1] - qmeans[0]),
        "top_quintile_return": float(qmeans[-1]),
        "bottom_quintile_return": float(qmeans[0]),
        "calibration_error": cal,
    }


def archetype_breakdown(obs):
    by_type: dict[str, list] = {}
    for o in obs:
        by_type.setdefault(o.get("company_type") or "unknown", []).append(o)
    out = {}
    for ct, rows in by_type.items():
        out[ct] = {"n_obs": len(rows), "n_tickers": len({o["ticker"] for o in rows}),
                   **{f"{y}y": metrics(rows, y) for y in HORIZONS}}
    by_sec: dict[str, list] = {}
    for o in obs:
        by_sec.setdefault(o["sector"], []).append(o)
    sectors = {}
    for sec, rows in by_sec.items():
        sectors[sec] = {"n_obs": len(rows), "n_tickers": len({o["ticker"] for o in rows}),
                        **{f"{y}y": metrics(rows, y) for y in HORIZONS}}
    return {"by_model_company_type": out, "by_sector_family": sectors}


def _winner_derater_split(obs):
    r = [o["fwd_3y"] for o in obs if isinstance(o.get("fwd_3y"), (int, float))]
    if not r:
        return {}
    a = np.array(r)
    return {"n_3y": len(r), "pct_negative_3y_return": float(np.mean(a < 0)),
            "min_3y_ann": float(a.min()), "max_3y_ann": float(a.max()), "median_3y_ann": float(np.median(a))}


def _verdict(overall, arch, n, n_tickers_used):
    ics = {y: (overall[y].get("information_coefficient") if "information_coefficient" in overall[y] else None) for y in ("1y", "3y", "5y")}
    good, poor = [], []
    for sec, d in arch["by_sector_family"].items():
        ic3 = (d.get("3y") or {}).get("information_coefficient")
        if isinstance(ic3, (int, float)):
            (good if ic3 >= 0.05 else poor).append((sec, round(ic3, 3)))
    coherent = (isinstance(ics.get("3y"), (int, float)) and ics["3y"] > 0
                and isinstance(ics.get("5y"), (int, float)) and ics["5y"] > 0)
    return {
        "n_observations": n,
        "n_tickers_with_data": n_tickers_used,
        "n_tickers_universe": len(ALL_TICKERS),
        "overall_information_coefficient": ics,
        "canonical_dcf_coherent_and_predictive": bool(coherent),
        "sector_families_handled_well": sorted(good, key=lambda x: -x[1]),
        "sector_families_handled_poorly": sorted(poor, key=lambda x: x[1]),
        "notes": [
            "Final universe: 506 names, 7 asset-light/FCF-coherent sector families.",
            "Banks/insurance/REITs/telecom/utilities/biotech/dev-stage pharma excluded by construction.",
            "Canonical DCF intrinsic only; internal WACC/terminal; state_space forecast; no lookahead.",
            "Overlapping multi-year windows -> autocorrelated obs; treat significance as indicative.",
        ],
    }


def _save(obs, skips, partial=False, done=0):
    overall = {f"{y}y": metrics(obs, y) for y in HORIZONS}
    arch = archetype_breakdown(obs)
    n_used = len({o["ticker"] for o in obs})
    (REPORT_DIR / "final_universe_backtest.json").write_text(json.dumps({
        "n_tickers_universe": len(ALL_TICKERS), "n_tickers_with_data": n_used,
        "n_observations": len(obs), "partial": partial, "tickers_processed": done or len(ALL_TICKERS),
        "skips": skips, "settings": "internal_wacc_terminal_state_space_canonical_dcf_only",
        "overall_metrics": overall, "observations": obs,
    }, indent=2, default=str), encoding="utf-8")
    (REPORT_DIR / "final_archetype_breakdown.json").write_text(json.dumps(arch, indent=2, default=str), encoding="utf-8")
    summary = {
        "n_tickers_universe": len(ALL_TICKERS), "n_tickers_with_data": n_used, "n_observations": len(obs),
        "partial": partial,
        "headline": {y: {k: overall[y].get(k) for k in ("n", "information_coefficient", "spearman", "hit_rate_beats_median", "long_short_q5_minus_q1", "calibration_error")} for y in ("1y", "3y", "5y")},
        "winners_vs_deraters": _winner_derater_split(obs),
    }
    (REPORT_DIR / "final_predictive_power_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (REPORT_DIR / "final_universe_verdict.json").write_text(json.dumps(_verdict(overall, arch, len(obs), n_used), indent=2, default=str), encoding="utf-8")
    return overall, arch


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    print(f"FINAL universe: {len(ALL_TICKERS)} tickers across {len(set(FAMILY_OF.values()))} families", flush=True)
    obs, skips = collect()
    overall, arch = _save(obs, skips, partial=False)

    print("\n=== OVERALL ===")
    for y in ("1y", "3y", "5y"):
        m = overall[y]
        if "spearman" not in m:
            print(f"  {y}: {m.get('note')}"); continue
        print(f"  {y}: n={m['n']} IC={_f(m['information_coefficient'])} spearman={_f(m['spearman'])} hit={_f(m['hit_rate_beats_median'])} L/S={_f(m['long_short_q5_minus_q1'])} calRMSE={_f((m['calibration_error'] or {}).get('rmse'))}")
    print("\n=== BY SECTOR FAMILY (3y IC / L/S) ===")
    for sec, d in arch["by_sector_family"].items():
        m = d.get("3y", {})
        print(f"  {sec:34} n={d['n_obs']:5} IC={_f(m.get('information_coefficient'))} L/S={_f(m.get('long_short_q5_minus_q1'))}")
    print(f"\nskips={len(skips)} -> {skips}")


def _f(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "None"


if __name__ == "__main__":
    main()
