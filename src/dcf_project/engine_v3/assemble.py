"""Assemble the engine_v3 DCFResult from the fade forecast (deterministic skeleton).

Valuation conventions are deliberately identical to v1's ``_value_cash_flow_path``
(quarterly discounting at WACC/4, terminal value on annualized flows discounted back
over the explicit horizon, net debt = debt − cash) so that v3-vs-v1 differences are
forecast-driven, never discounting-convention-driven. The terminal itself is the v3
value-driver formula -- this module does NOT route through
``valuation.dcf.build_state_space_dcf_result`` (whose policy/ROIC terminal blending
would contaminate the Grinold-Kroner anchor).

The headline ``intrinsic_value_per_share`` is the deterministic skeleton. The Monte
Carlo layer (``engine_v3.mc``) supplies the distribution payload; valuation is convex
in the drivers, so the MC mean/median is Jensen-inflated and reported as uncertainty,
not as the headline (unless ``v3_headline="mc_median"`` is explicitly chosen).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from dcf_project.contracts import DCFResult, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.data_sources.wacc import wacc_source_label
from dcf_project.engine_v3.coherence import assess_coherence
from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.terminal import value_driver_terminal
from dcf_project.valuation.discounting import present_value, present_value_series


def value_fade_path(
    frame: pd.DataFrame,
    wacc: float,
    terminal_growth: float,
    ronic_terminal: float,
    latest_cash: float | None,
    latest_debt: float | None,
    latest_shares: float | None,
    market: MarketSnapshot,
) -> dict[str, Any]:
    """Deterministic valuation of one fade forecast frame (also reused per MC draw)."""
    projected_fcf = [float(x) for x in frame["fcf"]]
    quarterly_wacc = wacc / 4.0
    explicit_pv = present_value_series(projected_fcf, quarterly_wacc)

    # Terminal-year NOPAT = last 4 forecast quarters, grown one year at g.
    nopat_terminal_annual = float(frame["nopat"].tail(4).sum())
    nopat_t1_annual = nopat_terminal_annual * (1.0 + terminal_growth)
    tv, terminal_detail = value_driver_terminal(nopat_t1_annual, terminal_growth, ronic_terminal, wacc)
    pv_terminal = present_value(tv, quarterly_wacc, len(projected_fcf))

    enterprise_value = explicit_pv + pv_terminal
    cash = latest_cash if latest_cash is not None else (market.cash or 0.0)
    debt = latest_debt if latest_debt is not None else (market.debt or 0.0)
    net_debt = float(debt - cash)
    equity_value = enterprise_value - net_debt
    shares = latest_shares or market.shares_outstanding
    if not shares or shares <= 0:
        raise ValueError("Diluted shares outstanding are required for per-share value")
    return {
        "projected_fcf": projected_fcf,
        "explicit_pv": float(explicit_pv),
        "terminal_value": float(tv),
        "pv_terminal": float(pv_terminal),
        "terminal_detail": terminal_detail,
        "enterprise_value": float(enterprise_value),
        "net_debt": net_debt,
        "equity_value": float(equity_value),
        "shares": float(shares),
        "intrinsic": float(equity_value / shares),
    }


def run_economic_dcf(
    ticker: str,
    frame: pd.DataFrame,
    driver_audit: dict[str, Any],
    wacc_inputs: WACCInputs,
    market: MarketSnapshot,
    latest_cash: float | None,
    latest_debt: float | None,
    latest_shares: float | None,
    terminal_growth: float,
    ronic_terminal: float,
    sources: list[SourceMeta],
    warnings: list[str],
    cfg: V3Config,
    diagnostics: dict[str, Any],
    mc_payload: dict[str, Any] | None = None,
) -> DCFResult:
    valuation = value_fade_path(
        frame,
        wacc_inputs.wacc,
        terminal_growth,
        ronic_terminal,
        latest_cash,
        latest_debt,
        latest_shares,
        market,
    )
    driver_audit = dict(driver_audit)
    driver_audit["terminal"] = valuation["terminal_detail"]
    driver_audit["pv_terminal_share_of_ev"] = (
        valuation["pv_terminal"] / valuation["enterprise_value"] if valuation["enterprise_value"] else None
    )
    diagnostics = dict(diagnostics)
    diagnostics["driver_audit"] = driver_audit

    intrinsic = valuation["intrinsic"]
    headline = intrinsic
    p10 = p50 = p90 = intrinsic
    confidence_interval: dict[str, float] = {}
    mc_distribution: dict[str, Any] = {}
    if mc_payload:
        mc_distribution = dict(mc_payload.get("distribution", {}))
        percentiles = mc_payload.get("percentiles", {})
        p10 = float(percentiles.get("p10", intrinsic))
        p50 = float(percentiles.get("p50", intrinsic))
        p90 = float(percentiles.get("p90", intrinsic))
        confidence_interval = {"p10": p10, "p50": p50, "p90": p90}
        # Convexity audit: how far Jensen pushes the MC mean above the skeleton.
        mc_distribution["skeleton_iv"] = intrinsic
        expected = mc_distribution.get("iv", {}).get("expected")
        if expected is not None and intrinsic:
            mc_distribution["jensen_gap"] = float(expected) / intrinsic - 1.0
        if cfg.headline == "mc_median":
            headline = p50

    growth_annual_y1 = float(driver_audit["paths"]["growth_annual"][0])
    bind_triggered = bool(driver_audit.get("growth_binds_quarters") or driver_audit.get("divestment_binds_quarters"))
    market_price = market.price
    implied_upside = (headline / market_price - 1.0) if market_price else None

    # Output-coherence guard (Research_V4 Mod 1): classify (never alter) the headline.
    # On an incomplete verdict the number is withheld as a usable headline -- flagged in
    # diagnostics, warned, and its implied upside nulled -- but retained for audit.
    coherence_status: str | None = None
    if cfg.coherence_guard:
        verdict = assess_coherence(
            intrinsic=headline,
            equity_value=valuation["equity_value"],
            enterprise_value=valuation["enterprise_value"],
            recon_shares=valuation["shares"],
            market_shares=market.shares_outstanding,
            revenue_ttm=driver_audit.get("drivers", {}).get("base_revenue_ttm"),
            pv_terminal_share_of_ev=driver_audit.get("pv_terminal_share_of_ev"),
            market_price=market_price,
        )
        coherence_status = verdict.status
        diagnostics["valuation_coherence"] = {
            "status": verdict.status,
            "flags": list(verdict.flags),
            "checks": dict(verdict.checks),
        }
        if verdict.is_incomplete:
            implied_upside = None
            warnings = list(warnings) + [
                f"valuation withheld as incomplete (coherence guard: {', '.join(verdict.flags)})"
            ]

    return DCFResult(
        ticker=ticker.upper(),
        explicit_forecast_quarters=len(frame),
        intrinsic_value_per_share=float(headline),
        enterprise_value=valuation["enterprise_value"],
        equity_value=valuation["equity_value"],
        net_debt=valuation["net_debt"],
        projected_fcf=valuation["projected_fcf"],
        explicit_pv=valuation["explicit_pv"],
        pv_projected_fcf=valuation["explicit_pv"],
        terminal_value=valuation["terminal_value"],
        pv_terminal_value=valuation["pv_terminal"],
        pv_terminal=valuation["pv_terminal"],
        fcf_terminal_base=float(valuation["terminal_detail"]["terminal_fcf_t1_annual"]) / 4.0,
        wacc=wacc_inputs.wacc,
        wacc_source=wacc_source_label(wacc_inputs),
        terminal_growth_rate=float(terminal_growth),
        fcf_growth_rate=growth_annual_y1,
        raw_fcf_growth_rate=float(driver_audit["g_base_effective"]),
        smoothed_fcf_growth_rate=growth_annual_y1,
        clamped_fcf_growth_rate=growth_annual_y1,
        growth_clamped=bind_triggered,
        clamp_triggered=bind_triggered,
        valuation_p10=p10,
        valuation_p50=p50,
        valuation_p90=p90,
        diluted_shares_outstanding=valuation["shares"],
        market_price=market_price,
        implied_upside=implied_upside,
        forecast_horizon_quarters=len(frame),
        forecast_mode="economic_fade",
        company_archetype=diagnostics.get("resolved_company_type"),
        valuation_regime=diagnostics.get("valuation_regime"),
        engine_version="v3",
        valuation_coherence=coherence_status,
        monte_carlo_distribution=mc_distribution,
        forecast_financials=[{k: float(v) for k, v in row.items()} for row in frame.to_dict("records")],
        confidence_interval=confidence_interval,
        sensitivity={},
        diagnostics=diagnostics,
        sources=sources,
        warnings=warnings,
    )
