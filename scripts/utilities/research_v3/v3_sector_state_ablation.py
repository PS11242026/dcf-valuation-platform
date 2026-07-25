"""G5: sector-state tilt ablation -- kappa=0 (production default) vs kappa=0.25.

Reuses the PIT sector_state scores already computed by the P4B runner
(reports/research_v2/sector_state_backtest_50.json: expanding-window z-scores, no
lookahead) joined on (ticker, date), and re-values engine_v3 ONLY (skeleton, mc_draws=0
-- the headline signal is the deterministic skeleton, so the ablation does not need
bands) with pipeline.sector_state_score injected per cutoff and sector_state_kappa=0.25.

The kappa=0 leg is the existing G2 panel (v3_universe_backtest_n50.json): identical
bundle construction (same checkpoint-validated machinery imported from
v3_universe_backtest), so per-(ticker,date) rows pair exactly.

Pre-registered adoption rule (plan G5): adopt kappa>0 only if the paired per-date IC
difference (tilted minus untilted) improves with NW t > 2; otherwise default 0 stays.

Usage: PYTHONPATH=src python scripts/utilities/research_v3/v3_sector_state_ablation.py \
           [--kappa 0.25] [--n 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "utilities"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

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
G2_PANEL = REPORT_DIR / "v3_universe_backtest_n50.json"
HORIZONS = (3, 5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kappa", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    g2 = json.loads(G2_PANEL.read_text(encoding="utf-8"))
    base_rows = {(o["ticker"], o["date"]): o for o in g2["observations"]}
    p4b = json.loads(P4B_PANEL.read_text(encoding="utf-8"))
    score_of = {(o["ticker"], o["date"]): o.get("sector_state") for o in p4b["observations"]}

    settings = replace(
        get_settings(),
        wacc_source="internal", terminal_growth_source="internal",
        forecast_mode="state_space", consensus_weight=0.0,
        monte_carlo_samples=0, scenario_monte_carlo_samples=0, run_sensitivity=False,
        sec_cache_enabled=True,
        engine_version="v3", v3_mc_draws=0, v3_sector_state_kappa=args.kappa,
    )
    tickers = TRANCHE[: args.n]
    cutoffs = _quarter_ends(2015)
    splits = vc.fetch_whitelisted_splits(tickers, SPLIT_CACHE, seed_caches=SPLIT_SEED_CACHES)

    rows: list[dict] = []
    n_scored = 0
    for i, ticker in enumerate(tickers):
        try:
            periods_all, _ = _sec_periods_with_retry(ticker, settings)
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
            base = base_rows.get(key)
            if base is None:  # only pair rows the kappa=0 panel produced
                continue
            bundle, _reason = _build_bundle(ticker, periods_tb, close_series, beta, cutoff, settings)
            if bundle is None:
                continue
            pipe = _PITPipelineV3(settings, bundle)
            score = score_of.get(key)
            pipe.sector_state_score = float(score) if isinstance(score, (int, float)) else None
            if pipe.sector_state_score is not None:
                n_scored += 1
            try:
                r3 = pipe.run(ticker)
            except (DriverExtractionError, TerminalValueError):
                continue
            except Exception:  # noqa: BLE001
                continue
            iv = float(r3.intrinsic_value_per_share)
            row = {
                "ticker": ticker, "date": key[1], "sector": base.get("sector"),
                "price": bundle.market.price,
                "tilted_discount": iv / bundle.market.price - 1.0,
                "base_discount": base["discount_to_intrinsic"],
                "sector_state_score": pipe.sector_state_score,
            }
            for y in HORIZONS:
                if isinstance(base.get(f"fwd_{y}y"), (int, float)):
                    row[f"fwd_{y}y"] = base[f"fwd_{y}y"]
            rows.append(row)
            kept += 1
        print(f"  [{i+1}/{len(tickers)}] {ticker}: ok:{kept}", flush=True)

    overall = {
        f"{y}y": {
            "tilted": vc.core_metrics(rows, y, signal_key="tilted_discount"),
            "base": vc.core_metrics(rows, y, signal_key="base_discount"),
            "paired_ic_diff_tilted_minus_base":
                vc.paired_ic_diff(rows, y, "tilted_discount", "base_discount"),
        } for y in HORIZONS
    }
    scored_rows = [r for r in rows if r.get("sector_state_score") is not None]
    scored = {
        f"{y}y": vc.paired_ic_diff(scored_rows, y, "tilted_discount", "base_discount")
        for y in HORIZONS
    }
    diff3 = overall["3y"]["paired_ic_diff_tilted_minus_base"]
    adopt = (isinstance(diff3.get("mean_ic_diff"), (int, float)) and diff3["mean_ic_diff"] > 0
             and isinstance(diff3.get("nw_tstat"), (int, float)) and diff3["nw_tstat"] > 2.0)
    payload = {
        "kappa": args.kappa,
        "n_rows": len(rows),
        "n_rows_with_score": len(scored_rows),
        "score_source": str(P4B_PANEL),
        "overall": overall,
        "paired_diff_scored_rows_only": scored,
        "adoption_rule": "adopt kappa>0 only if paired 3y IC diff > 0 with NW t > 2",
        "verdict": "ADOPT" if adopt else "KEEP_KAPPA_0",
    }
    out_path = REPORT_DIR / f"v3_sector_state_ablation_k{args.kappa}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"G5 ablation kappa={args.kappa}: rows={len(rows)} scored={len(scored_rows)}")
    for y in HORIZONS:
        d = overall[f"{y}y"]["paired_ic_diff_tilted_minus_base"]
        print(f"  {y}y paired IC diff (tilted-base): {d.get('mean_ic_diff')} (NW t {d.get('nw_tstat')})")
    print(f"  verdict: {payload['verdict']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
