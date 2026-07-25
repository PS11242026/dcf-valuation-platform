"""Canonical DCF -- the frozen production state-space pipeline + validated scoring."""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from dcf_project.core import get_settings
from dcf_project.paths import VALUATION
from dcf_project.portfolio.scoring import Reference, score_ticker
from dcf_project.reporting_kit import save_model, xlsx
from dcf_project.valuation.report import build_canonical_model
from dcf_project.valuation.report_fade import build_fade_model


def canonical_valuation(ticker: str, forecast_mode="consensus_blend", wacc_source="auto",
                        terminal_growth_source="auto", company_type="auto", engine="v1",
                        mc_draws: int | None = None, country: str = "auto") -> dict:
    from dcf_project.data_sources.markets import resolve_country

    overrides: dict = dict(forecast_mode=forecast_mode, wacc_source=wacc_source,
                           terminal_growth_source=terminal_growth_source, company_type=company_type,
                           engine_version=engine)
    if mc_draws is not None:  # engine_v3 optional Monte Carlo sidecar (diagnostic-only)
        overrides["v3_mc_draws"] = int(mc_draws)
    # Country is resolved once here and the whole India macro profile (G-Sec risk-free, ERP,
    # tax, inflation, nominal GDP, terminal band, wacc_source=internal) is folded into the
    # Settings via replace(). The India-flavoured Settings then flows through score_ticker ->
    # the v3 pipeline/terminal/MC unchanged. A US ticker leaves `overrides` untouched.
    if resolve_country(ticker, None if country == "auto" else country) == "IN":
        from dcf_project.data_sources.india.macro import live_india_macro

        overrides.update(live_india_macro().settings_overrides())
    settings = replace(get_settings(), **overrides)
    return score_ticker(ticker, Reference(), settings)


def run_canonical(ticker: str, forecast_mode="consensus_blend", wacc_source="auto",
                  terminal_growth_source="auto", company_type="auto", engine="v1",
                  mc_draws: int | None = None, country: str = "auto") -> Path:
    ticker = ticker.upper()
    folder = VALUATION / ticker
    folder.mkdir(parents=True, exist_ok=True)
    sc = canonical_valuation(ticker, forecast_mode, wacc_source, terminal_growth_source, company_type,
                             engine=engine, mc_draws=mc_draws, country=country)
    res = sc["result"]
    # engine_v3 produces the Fundamentals Fade workbook; v1/v2 keep the canonical workbook.
    if getattr(res, "engine_version", "v1") == "v3":
        model = build_fade_model(sc, folder)
        out = xlsx.render(model, folder / "fade_dcf.xlsx")
    else:
        model = build_canonical_model(sc, folder)
        out = xlsx.render(model, folder / "canonical_dcf.xlsx")
    save_model(model, folder)

    ff = [dict(r) for r in (res.forecast_financials or [])]
    if ff:
        keys = sorted({k for r in ff for k in r.keys()})
        with open(folder / "forecast.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in ff:
                w.writerow(r)
    (folder / "assumptions.json").write_text(json.dumps(sc["assumptions"], indent=2, default=str), encoding="utf-8")
    # engine_v3 traceability: the driver audit maps every projected number to a named
    # economic assumption (drivers -> persistence/CAP -> fade -> terminal decomposition).
    audit = res.diagnostics.get("driver_audit")
    if audit:
        (folder / "driver_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    d = sc.get("discount_to_intrinsic")
    print(f"wrote {out}  ({sc['verdict']}, discount {d:.1%})" if d is not None else f"wrote {out}")
    return out
