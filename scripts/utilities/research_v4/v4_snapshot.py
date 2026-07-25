"""Research_V4 Phase A -- current-snapshot diagnostic harness (engine_v3 vs market vs consensus).

For each of the ~78 stratified names (6 families, Healthcare/Financials excluded):
  1. run engine_v3 (internal WACC/terminal for reproducibility) and extract the full
     driver_audit panel (intrinsic, CAP, fade half-lives, growth/margin/ROIC paths, terminal
     value contribution, PV(TV)/EV, a WACC x g sensitivity whisker);
  2. back out the market-implied effective growth / normalized margin / terminal RONIC with the
     V3-native reverse-DCF (v4_reverse), including a round-trip self-check;
  3. read free analyst consensus (target price, recommendation, forward growth) read-only.

ISOLATION: writes only under reports/research_v4/; instantiates the engine but changes no
src/ code, no live-book state, no frozen artifact. Per-ticker checkpoints make it resumable
and re-runs deterministic/offline.

Usage:
    python scripts/utilities/research_v4/v4_snapshot.py --per-family 13
    python scripts/utilities/research_v4/v4_snapshot.py --limit 3        # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "src", str(Path(__file__).resolve().parent)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import universe  # noqa: E402
import v4_reverse  # noqa: E402
from v4_consensus import fetch_consensus  # noqa: E402

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.data_sources.fundamentals import FundamentalsBundle  # noqa: E402
from dcf_project.engine_v3.drivers import DriverExtractionError  # noqa: E402
from dcf_project.engine_v3.pipeline import EconomicDCFPipeline  # noqa: E402
from dcf_project.valuation.report_fade import fade_iv_range  # noqa: E402
from dcf_project.valuation.terminal import TerminalValueError  # noqa: E402

REPORT_DIR = ROOT / "reports" / "research_v4"
CKPT_DIR = REPORT_DIR / "_checkpoints"
TABLE_DIR = REPORT_DIR / "tables"


class _BundlePipeline(EconomicDCFPipeline):
    """Inject a pre-fetched bundle so run() and the reverse context share one fetch."""

    def __init__(self, settings, bundle: FundamentalsBundle):
        super().__init__(settings)
        self._bundle = bundle

    def load_data(self, ticker: str) -> FundamentalsBundle:
        return self._bundle


def _settings():
    # Internal WACC/terminal growth -> reproducible run-to-run diffs (provider-wacc-instability
    # memory). Current market price still comes live from the bundle's market snapshot.
    return replace(
        get_settings(),
        wacc_source="internal",
        terminal_growth_source="internal",
        consensus_weight=0.0,
        sec_cache_enabled=True,
        v3_mc_draws=0,
    )


def _audit_paths(audit: dict[str, Any]) -> dict[str, Any]:
    paths = audit.get("paths", {})
    return {
        "growth_path": paths.get("growth_annual"),
        "margin_path": paths.get("nopat_margin"),
        "ronic_path": paths.get("ronic"),
        "reinvestment_path": paths.get("reinvestment_rate"),
    }


def _row_for(ticker: str, sector: str, settings) -> dict[str, Any]:
    """Run the engine + reverse-DCF + consensus for one ticker; return a flat panel row."""
    base = EconomicDCFPipeline(settings)
    bundle = base.load_data(ticker)  # single live fetch (SEC cached, market live)
    pipe = _BundlePipeline(settings, bundle)
    result = pipe.run(ticker)
    audit = (result.diagnostics or {}).get("driver_audit", {})
    persistence = audit.get("persistence", {})
    half_lives = audit.get("half_lives", {})
    terminal = audit.get("terminal", {})
    drivers = audit.get("drivers", {})

    iv = float(result.intrinsic_value_per_share)
    price = result.market_price
    coherence = (result.diagnostics or {}).get("valuation_coherence", {})
    sens_min, sens_max, _ = fade_iv_range(result, ticker)

    # Reverse-DCF (V3-native) against the live market price.
    ctx = v4_reverse.build_context(settings, bundle, ticker)
    implied_g = implied_m = implied_ronic = roundtrip_g = None
    if price and price > 0:
        implied_g = v4_reverse.implied_growth(ctx, price)
        implied_m = v4_reverse.implied_margin(ctx, price)
        implied_ronic = v4_reverse.implied_terminal_ronic(ctx, price)
    # Round-trip self-check: solving against the engine's OWN IV must recover g_base_effective.
    roundtrip_g = v4_reverse.implied_growth(ctx, iv)

    consensus = fetch_consensus(ticker)

    return {
        "ticker": ticker,
        "sector": sector,
        # --- engine_v3 internal outputs
        "intrinsic": iv,
        "market_price": price,
        "diluted_shares": float(result.diluted_shares_outstanding) if result.diluted_shares_outstanding else None,
        "market_cap": (float(result.diluted_shares_outstanding) * price)
        if (price and result.diluted_shares_outstanding) else None,
        "implied_upside": result.implied_upside,
        # --- output-coherence guard (engine_v3 Mod 1) verdict, now native to the result
        "valuation_coherence": result.valuation_coherence,
        "coherence_flags": coherence.get("flags"),
        "coherence_checks": coherence.get("checks"),
        "wacc": float(result.wacc),
        "terminal_growth": float(result.terminal_growth_rate),
        "cap_years": persistence.get("cap_years"),
        "phi": persistence.get("phi"),
        "excess_terminal": persistence.get("excess_terminal"),
        "excess_roic_0": drivers.get("excess_roic_0"),
        "roic_annual_0": drivers.get("roic_annual_0"),
        "quality": drivers.get("quality"),
        "g_base_effective": audit.get("g_base_effective"),
        "margin_current": audit.get("margin_current"),
        "margin_target_blend": audit.get("margin_target_blend"),
        "half_life_growth_years": half_lives.get("growth_years"),
        "half_life_margin_years": half_lives.get("margin_years"),
        "terminal_value_contribution": terminal.get("terminal_value"),
        "ronic_terminal": terminal.get("ronic_terminal"),
        "pv_tv_share_of_ev": audit.get("pv_terminal_share_of_ev"),
        "n_growth_binds": len(audit.get("growth_binds_quarters") or []),
        "n_fcf_band_breaches": len(audit.get("fcf_margin_breach_quarters") or []),
        "sensitivity_iv_min": sens_min,
        "sensitivity_iv_max": sens_max,
        **_audit_paths(audit),
        # --- market-implied (reverse-DCF, V3-native)
        "implied_growth": implied_g,
        "implied_margin": implied_m,
        "implied_terminal_ronic": implied_ronic,
        "implied_growth_minus_model": (
            implied_g - ctx.g_base_effective if implied_g is not None else None
        ),
        "reverse_roundtrip_g": roundtrip_g,
        "reverse_roundtrip_err": (
            roundtrip_g - ctx.g_base_effective if roundtrip_g is not None else None
        ),
        # --- analyst consensus (free, read-only)
        "consensus_target_mean": consensus.target_mean,
        "consensus_target_median": consensus.target_median,
        "consensus_recommendation_mean": consensus.recommendation_mean,
        "consensus_recommendation_key": consensus.recommendation_key,
        "consensus_n_analysts": consensus.n_analysts,
        "consensus_revenue_growth": consensus.revenue_growth_annual,
        "consensus_price": consensus.price,
        "consensus_error": consensus.error,
        "status": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Research_V4 snapshot diagnostic (engine_v3 vs market vs consensus)")
    ap.add_argument("--per-family", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None, help="cap total names (smoke test)")
    ap.add_argument("--refresh", action="store_true", help="ignore per-ticker checkpoints")
    args = ap.parse_args()

    settings = _settings()
    names = universe.flat_sample(args.per_family)
    if args.limit:
        names = names[: args.limit]
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skips: dict[str, str] = {}
    for i, (ticker, sector) in enumerate(names, 1):
        ckpt = CKPT_DIR / f"{ticker}.json"
        if ckpt.exists() and not args.refresh:
            row = json.loads(ckpt.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"[{i:>3}/{len(names)}] {ticker:<6} cached  ({row.get('status')})", flush=True)
            continue
        t0 = time.time()
        try:
            row = _row_for(ticker, sector, settings)
        except (DriverExtractionError, TerminalValueError) as exc:
            row = {"ticker": ticker, "sector": sector, "status": f"skip:{type(exc).__name__}:{exc}"}
            skips[ticker] = row["status"]
        except Exception as exc:  # noqa: BLE001 -- record and continue the sweep
            row = {"ticker": ticker, "sector": sector, "status": f"error:{type(exc).__name__}:{exc}"}
            skips[ticker] = row["status"]
        ckpt.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        rt = row.get("reverse_roundtrip_err")
        rt_s = f" rt_err={rt:+.4f}" if isinstance(rt, (int, float)) else ""
        print(f"[{i:>3}/{len(names)}] {ticker:<6} {row.get('status'):<28} {time.time()-t0:5.1f}s{rt_s}", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": "current-snapshot engine_v3 vs market vs analyst consensus; 6 families "
                  "(Healthcare & Financials excluded); internal WACC/terminal for reproducibility",
        "per_family": args.per_family,
        "n_names": len(names),
        "n_ok": len(ok),
        "selection": universe.stratified_sample(args.per_family),
        "skips": skips,
        "rows": rows,
    }
    (REPORT_DIR / "v4_snapshot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_panel_csv(ok)
    print(f"\nWrote {REPORT_DIR / 'v4_snapshot.json'} ({len(ok)}/{len(names)} ok)")


def _write_panel_csv(rows: list[dict[str, Any]]) -> None:
    import csv

    scalar_cols = [
        "ticker", "sector", "intrinsic", "market_price", "market_cap", "implied_upside",
        "valuation_coherence", "wacc",
        "terminal_growth", "cap_years", "phi", "excess_terminal", "excess_roic_0",
        "roic_annual_0", "quality", "g_base_effective", "margin_current",
        "margin_target_blend", "half_life_growth_years", "half_life_margin_years",
        "terminal_value_contribution", "ronic_terminal", "pv_tv_share_of_ev",
        "n_growth_binds", "n_fcf_band_breaches", "sensitivity_iv_min", "sensitivity_iv_max",
        "implied_growth", "implied_margin", "implied_terminal_ronic",
        "implied_growth_minus_model", "reverse_roundtrip_err",
        "consensus_target_mean", "consensus_target_median", "consensus_recommendation_mean",
        "consensus_recommendation_key", "consensus_n_analysts", "consensus_revenue_growth",
    ]
    path = TABLE_DIR / "v4_panel.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(scalar_cols)
        for r in rows:
            writer.writerow([r.get(c) for c in scalar_cols])


if __name__ == "__main__":
    main()
