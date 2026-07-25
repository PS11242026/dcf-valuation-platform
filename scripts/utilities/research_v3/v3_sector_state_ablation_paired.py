"""G5 (corrected, self-paired): sector-state tilt ablation on the CORRECTED v3 engine.

The original v3_sector_state_ablation.py diffs an in-process tilted leg against the
base_discount stored in v3_universe_backtest_n50.json -- but that panel predates the
2026-06-11 margin-seasonality fix, so its base_discount is stale and the diff would
conflate the sector-state tilt with the bug fix. This version values BOTH legs in-process
on the same corrected engine and the same PIT bundle, so the paired per-row difference
isolates the tilt cleanly.

For each (ticker, cutoff) we build ONE PIT bundle (split-consistent, identical machinery
imported from v3_universe_backtest) and value engine_v3 twice -- kappa=0 (production default)
and kappa=K -- injecting the expanding-window sector_state_score from the P4B panel. The
paired per-date IC difference (tilted minus untilted) gets a Newey-West HAC t-stat.

Pre-registered adoption rule (plan G5): adopt kappa>0 only if the paired 3y IC diff > 0
with NW t > 2; otherwise keep kappa=0. Read-only; writes a JSON ledger.

Usage: PYTHONPATH=src python scripts/utilities/research_v3/v3_sector_state_ablation_paired.py \
           [--kappa 0.25] [--n 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for c in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "utilities"):
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))

import validation_common as vc  # noqa: E402
from historical_predictive_power import _quarter_ends  # noqa: E402
from research_v3.v3_universe_backtest import (  # noqa: E402
    PRICE_CACHE,
    REPORT_DIR,
    SPLIT_CACHE,
    SPLIT_SEED_CACHES,
    TRANCHE,
    _beta_cached,
    _build_bundle,
    _PITPipelineV3,
    _sec_periods_with_retry,
)

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.engine_v3.drivers import DriverExtractionError  # noqa: E402
from dcf_project.valuation.terminal import TerminalValueError  # noqa: E402

P4B_PANEL = ROOT / "reports" / "research_v2" / "sector_state_backtest_50.json"
HORIZONS = (3, 5)


def _value_v3(settings, bundle, ticker, score):
    pipe = _PITPipelineV3(settings, bundle)
    pipe.sector_state_score = float(score) if isinstance(score, (int, float)) else None
    try:
        r = pipe.run(ticker)
    except (DriverExtractionError, TerminalValueError):
        return None
    except Exception:  # noqa: BLE001
        return None
    iv = float(r.intrinsic_value_per_share)
    px = bundle.market.price
    return iv / px - 1.0 if px else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kappa", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    p4b = json.loads(P4B_PANEL.read_text(encoding="utf-8"))
    score_of = {(o["ticker"], o["date"]): o.get("sector_state") for o in p4b["observations"]}
    # forward returns + sector come from the P4B panel rows (same split-consistent harness)
    fwd_of = {(o["ticker"], o["date"]): o for o in p4b["observations"]}

    base = replace(
        get_settings(),
        wacc_source="internal", terminal_growth_source="internal",
        forecast_mode="state_space", consensus_weight=0.0,
        monte_carlo_samples=0, scenario_monte_carlo_samples=0, run_sensitivity=False,
        sec_cache_enabled=True, engine_version="v3", v3_mc_draws=0,
    )
    s_untilted = replace(base, v3_sector_state_kappa=0.0)
    s_tilted = replace(base, v3_sector_state_kappa=args.kappa)

    tickers = TRANCHE[: args.n]
    cutoffs = _quarter_ends(2015)
    splits = vc.fetch_whitelisted_splits(tickers, SPLIT_CACHE, seed_caches=SPLIT_SEED_CACHES)

    rows: list[dict] = []
    n_scored = 0
    for i, ticker in enumerate(tickers):
        try:
            periods_all, _ = _sec_periods_with_retry(ticker, base)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i+1}/{len(tickers)}] {ticker}: sec fail {type(exc).__name__}", flush=True)
            continue
        periods_tb = vc.reexpress_shares_today_basis(periods_all, splits.get(ticker, []))
        prices = vc.pit_price_frame(ticker, PRICE_CACHE)
        if prices is None or not len(prices):
            continue
        close_series = prices["close"]
        beta = _beta_cached(ticker)
        kept = 0
        for cutoff in cutoffs:
            key = (ticker, cutoff.isoformat())
            ref = fwd_of.get(key)
            if ref is None:  # only rows the P4B harness produced (gives fwd + score)
                continue
            bundle, _r = _build_bundle(ticker, periods_tb, close_series, beta, cutoff, base)
            if bundle is None:
                continue
            score = score_of.get(key)
            d_un = _value_v3(s_untilted, bundle, ticker, score)
            d_ti = _value_v3(s_tilted, bundle, ticker, score)
            if d_un is None or d_ti is None:
                continue
            if isinstance(score, (int, float)):
                n_scored += 1
            row = {"ticker": ticker, "date": key[1], "sector": ref.get("sector"),
                   "base_discount": d_un, "tilted_discount": d_ti,
                   "sector_state_score": score}
            for y in HORIZONS:
                fv = ref.get(f"fwd_{y}y_ann_return")  # P4B panel's forward-return key naming
                if isinstance(fv, (int, float)):
                    row[f"fwd_{y}y"] = fv
            rows.append(row)
            kept += 1
        print(f"  [{i+1}/{len(tickers)}] {ticker}: ok:{kept}", flush=True)

    scored_rows = [r for r in rows if r.get("sector_state_score") is not None]
    overall = {
        f"{y}y": {
            "tilted": vc.core_metrics(rows, y, signal_key="tilted_discount"),
            "base": vc.core_metrics(rows, y, signal_key="base_discount"),
            "paired_ic_diff_tilted_minus_base":
                vc.paired_ic_diff(rows, y, "tilted_discount", "base_discount"),
        } for y in HORIZONS
    }
    scored = {f"{y}y": vc.paired_ic_diff(scored_rows, y, "tilted_discount", "base_discount")
              for y in HORIZONS}
    diff3 = overall["3y"]["paired_ic_diff_tilted_minus_base"]
    adopt = (isinstance(diff3.get("mean_ic_diff"), (int, float)) and diff3["mean_ic_diff"] > 0
             and isinstance(diff3.get("nw_tstat"), (int, float)) and diff3["nw_tstat"] > 2.0)
    payload = {
        "kappa": args.kappa, "engine": "corrected (post margin-seasonality fix)",
        "self_paired": True,
        "n_rows": len(rows), "n_rows_with_score": len(scored_rows),
        "score_source": str(P4B_PANEL),
        "overall": overall, "paired_diff_scored_rows_only": scored,
        "adoption_rule": "adopt kappa>0 only if paired 3y IC diff > 0 with NW t > 2",
        "verdict": "ADOPT" if adopt else "KEEP_KAPPA_0",
    }
    out_path = REPORT_DIR / f"v3_sector_state_ablation_paired_k{args.kappa}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nG5 self-paired ablation kappa={args.kappa} (corrected engine): "
          f"rows={len(rows)} scored={len(scored_rows)}")
    for y in HORIZONS:
        d = overall[f"{y}y"]["paired_ic_diff_tilted_minus_base"]
        print(f"  {y}y paired IC diff (tilted-base): {d.get('mean_ic_diff')} (NW t {d.get('nw_tstat')}, n_d {d.get('n_dates')})")
    print(f"  verdict: {payload['verdict']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
