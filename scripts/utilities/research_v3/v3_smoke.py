"""End-to-end smoke + economics-audit printer for engine_v3 (Phase 2 gate; G1 aid).

Runs the EconomicDCFPipeline on live tickers (engine_version="v3", SEC disk cache ON)
and prints the driver audit chain: drivers -> persistence/CAP -> fade endpoints ->
Grinold-Kroner/value-driver terminal -> valuation bridge. Every printed number should
be traceable to a named economic assumption -- that is the G1 review criterion.

Usage: python scripts/utilities/research_v3/v3_smoke.py KO AAPL MSFT
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import replace

from dcf_project.config import get_settings
from dcf_project.engine_v2 import select_pipeline


def pct(x: float | None, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%" if x is not None else "n/a"


def run_one(ticker: str) -> None:
    settings = replace(get_settings(), engine_version="v3", sec_cache_enabled=True)
    pipeline = select_pipeline(settings)
    print(f"\n=== {ticker} [{type(pipeline).__name__}] ===")
    started = time.perf_counter()
    result = pipeline.run(ticker)
    elapsed = time.perf_counter() - started
    audit = result.diagnostics["driver_audit"]
    drv = audit["drivers"]
    per = audit["persistence"]
    gk = audit["grinold_kroner"]
    term = audit["terminal"]

    print(f"engine={result.engine_version} mode={result.forecast_mode} "
          f"sector={result.diagnostics.get('sector')} archetype={result.company_archetype}")
    iv = result.intrinsic_value_per_share
    print(f"price={result.market_price} intrinsic={iv:.2f} upside={pct(result.implied_upside)}")
    print(f"WACC={pct(result.wacc)} g_terminal={pct(result.terminal_growth_rate)} "
          f"(GK raw {pct(gk['gk_raw'])} = GDP {pct(gk['long_run_nominal_gdp'])} - dilution {pct(gk['dilution_haircut'])})")
    g_path = audit["paths"]["growth_annual"]
    print(f"growth: base {pct(audit['g_base'])} -> y1 {pct(g_path[0])} -> end {pct(g_path[-1])} "
          f"(tau {audit['tau_quarters']:.1f}q)")
    print(f"margin (NOPAT): current {pct(audit['margin_current'])} -> target blend {pct(audit['margin_target_blend'])} "
          f"(company {pct(audit['margin_target_company'])}, sector anchor {pct(audit['sector_margin_anchor'])} w={audit['sector_anchor_weight']})")
    phi_c = f"{per['phi_company']:.3f}" if per["phi_company"] is not None else "n/a"
    print(f"persistence: phi={per['phi']:.3f} (company {phi_c}, n={per['n_years']}y, weight {per['shrink_weight']:.2f}) "
          f"CAP={per['cap_years']:.1f}y")
    print(f"excess ROIC: now {pct(drv['excess_roic_0'])} -> terminal {pct(per['excess_terminal'])} "
          f"=> RONIC_T {pct(term['ronic_terminal'])}, terminal reinvestment {pct(term['terminal_reinvestment_rate'])}")
    print(f"terminal: TV {term['terminal_value'] / 1e9:.1f}B, PV(TV)/EV {pct(audit['pv_terminal_share_of_ev'])}")
    band = audit["fcf_margin_band"]
    print(f"FCF margin band p10/p50/p90: {pct(band['p10'])}/{pct(band['p50'])}/{pct(band['p90'])}; "
          f"implied final TTM {pct(audit['implied_fcf_margin_final_ttm'])}")
    print(f"binds: growth {audit['growth_binds_quarters']} divestment {audit['divestment_binds_quarters']} "
          f"band breaches {audit['fcf_margin_breach_quarters']}")
    print(f"quality={drv['quality']:.2f} notes={drv['notes'] + per['notes']}")
    mc = result.monte_carlo_distribution
    if mc:
        ivd = mc["iv"]
        dd = mc["driver_draws"]
        clips = {k: v["clipped"] for k, v in dd.items() if v}
        print(f"MC ({mc['n_effective']}/{mc['n_draws']} draws, {mc['failed_draws']} failed, seed {mc['seed']}): "
              f"p10/p50/p90 = {result.valuation_p10:.2f}/{result.valuation_p50:.2f}/{result.valuation_p90:.2f}")
        print(f"   mean {ivd['expected']:.2f} (jensen gap {pct(mc.get('jensen_gap'))}) std {ivd['std']:.2f} "
              f"P(iv>price) {ivd.get('prob_value_gt_price', float('nan')):.2f} clips {clips}")
    else:
        print("MC: disabled/unavailable (skeleton only)")
    print(f"elapsed {elapsed:.1f}s")
    if result.warnings:
        shown = "; ".join(result.warnings[:4])
        print(f"warnings ({len(result.warnings)}): {shown}")


def main() -> int:
    tickers = [t.upper() for t in sys.argv[1:]] or ["KO"]
    failures: list[str] = []
    for ticker in tickers:
        try:
            run_one(ticker)
        except Exception as exc:  # smoke harness: report and keep going
            failures.append(ticker)
            print(f"\n=== {ticker} FAILED: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tickers) - len(failures)}/{len(tickers)} OK" + (f"; failed: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
