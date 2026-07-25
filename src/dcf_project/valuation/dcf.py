from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from dcf_project.config import Settings
from dcf_project.contracts import DCFResult, GrowthForecast, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.data_sources.growth import resolve_terminal_growth
from dcf_project.data_sources.wacc import wacc_source_label
from dcf_project.valuation.discounting import present_value, present_value_series
from dcf_project.valuation.terminal import terminal_value


def build_state_space_dcf_result(
    ticker: str,
    forecast_financials: pd.DataFrame,
    base_fcf: float,
    fcf_growth_rate: float,
    wacc_inputs: WACCInputs,
    market: MarketSnapshot,
    latest_cash: float | None,
    latest_debt: float | None,
    latest_shares: float | None,
    sources: list[SourceMeta],
    warnings: list[str],
    settings: Settings,
    diagnostics: dict[str, object] | None = None,
) -> DCFResult:
    if len(forecast_financials) != settings.explicit_forecast_quarters:
        raise ValueError(
            f"DCF requires exactly {settings.explicit_forecast_quarters} forecast quarters; "
            f"received {len(forecast_financials)}."
        )
    path_columns = {
        "accounting": "accounting_fcf" if "accounting_fcf" in forecast_financials else "fcf",
        "economic": "economic_fcf" if "economic_fcf" in forecast_financials else "fcf",
        "owner": "owner_earnings" if "owner_earnings" in forecast_financials else "fcf",
    }
    path_values = {
        name: [float(x) for x in forecast_financials[column].to_list()]
        for name, column in path_columns.items()
    }
    weights = _ensemble_weights(settings.company_type, forecast_financials)
    projected_fcf = [
        sum(weights[name] * path_values[name][idx] for name in weights)
        for idx in range(len(forecast_financials))
    ]
    quarterly_wacc = wacc_inputs.wacc / 4.0
    terminal_growth_assumption = resolve_terminal_growth(ticker, wacc_inputs.wacc, settings)
    terminal_growth = terminal_growth_assumption.terminal_growth
    terminal_growth_trace = _terminal_growth_trace(terminal_growth_assumption, wacc_inputs.wacc, settings)
    growth_profile = _fcf_growth_projection_profile(
        projected_fcf,
        forecast_financials,
        base_fcf,
        wacc_inputs.wacc,
        terminal_growth,
        settings,
    )
    terminal_growth = growth_profile["terminal_growth_rate"]
    terminal_growth_trace["fcf_growth_projection_engine"] = growth_profile
    component_values = {
        name: _value_cash_flow_path(
            path,
            forecast_financials,
            wacc_inputs,
            market,
            latest_cash,
            latest_debt,
            latest_shares,
            terminal_growth,
        )
        for name, path in path_values.items()
    }
    weights = _adjust_weights_for_path_dispersion(weights, component_values)
    projected_fcf = [
        sum(weights[name] * path_values[name][idx] for name in weights)
        for idx in range(len(forecast_financials))
    ]
    growth_profile = _fcf_growth_projection_profile(
        projected_fcf,
        forecast_financials,
        base_fcf,
        wacc_inputs.wacc,
        terminal_growth_assumption.terminal_growth,
        settings,
    )
    if abs(growth_profile["terminal_growth_rate"] - terminal_growth) > 1e-9:
        terminal_growth = growth_profile["terminal_growth_rate"]
        terminal_growth_trace = _terminal_growth_trace(terminal_growth_assumption, wacc_inputs.wacc, settings)
        terminal_growth_trace["fcf_growth_projection_engine"] = growth_profile
        component_values = {
            name: _value_cash_flow_path(
                path,
                forecast_financials,
                wacc_inputs,
                market,
                latest_cash,
                latest_debt,
                latest_shares,
                terminal_growth,
            )
            for name, path in path_values.items()
        }
        weights = _adjust_weights_for_path_dispersion(weights, component_values)
        projected_fcf = [
            sum(weights[name] * path_values[name][idx] for name in weights)
            for idx in range(len(forecast_financials))
        ]
        growth_profile = _fcf_growth_projection_profile(
            projected_fcf,
            forecast_financials,
            base_fcf,
            wacc_inputs.wacc,
            terminal_growth_assumption.terminal_growth,
            settings,
        )
        terminal_growth = growth_profile["terminal_growth_rate"]
    terminal_growth_trace = _terminal_growth_trace(terminal_growth_assumption, wacc_inputs.wacc, settings)
    terminal_growth_trace["fcf_growth_projection_engine"] = growth_profile
    component_values = {
        name: _value_cash_flow_path(
            path,
            forecast_financials,
            wacc_inputs,
            market,
            latest_cash,
            latest_debt,
            latest_shares,
            terminal_growth,
        )
        for name, path in path_values.items()
    }
    diagnostic_fcf_growth = growth_profile["explicit_fcf_growth_rate"]
    ensemble_intrinsic = sum(weights[name] * component_values[name]["intrinsic"] for name in weights)
    ensemble_enterprise = sum(weights[name] * component_values[name]["enterprise_value"] for name in weights)
    ensemble_equity = sum(weights[name] * component_values[name]["equity_value"] for name in weights)
    discount_schedule = [
        {
            "quarter": float(idx),
            "fcf": fcf,
            "discount_factor": float((1.0 + quarterly_wacc) ** idx),
            "discount_rate_per_quarter": float(quarterly_wacc),
            "present_value": float(fcf / ((1.0 + quarterly_wacc) ** idx)),
            "pv_fcf": float(fcf / ((1.0 + quarterly_wacc) ** idx)),
        }
        for idx, fcf in enumerate(projected_fcf, start=1)
    ]
    forecast_fcf_trace = []
    forecast_records = _frame_records(forecast_financials)
    for row, discount in zip(forecast_records, discount_schedule):
        traced = dict(row)
        traced["discount_factor"] = discount["discount_factor"]
        traced["pv_fcf"] = discount["present_value"]
        forecast_fcf_trace.append(traced)
    explicit_pv = float(sum(weights[name] * component_values[name]["explicit_pv"] for name in weights))
    fcf_terminal = sum(weights[name] * component_values[name]["terminal_fcf"] for name in weights)
    annual_terminal_fcf = fcf_terminal * 4.0
    tv = sum(weights[name] * component_values[name]["terminal_value"] for name in weights)
    pv_terminal = sum(weights[name] * component_values[name]["pv_terminal"] for name in weights)
    enterprise_value = ensemble_enterprise

    cash = latest_cash if latest_cash is not None else market.cash or 0.0
    debt = latest_debt if latest_debt is not None else market.debt or 0.0
    net_debt = float(debt - cash)
    equity_value = ensemble_equity
    shares = latest_shares or market.shares_outstanding
    if not shares or shares <= 0:
        raise ValueError("Diluted shares outstanding are required for per-share value")
    intrinsic = ensemble_intrinsic
    implied_upside = (intrinsic / market.price - 1.0) if market.price else None
    diag = dict(diagnostics or {})
    share_count_trace = dict(diag.get("share_count_trace") or {})
    share_count_trace.update(
        {
            "final_valuation_denominator": float(shares),
            "market_snapshot_shares": market.shares_outstanding,
            "per_share_equity_value_check": float(equity_value / shares),
        }
    )
    diag.update(
        {
            "valuation_mode": "state_space_dcf",
            "company_type": settings.company_type,
            "share_count_trace": share_count_trace,
            "quarterly_discount_rate": quarterly_wacc,
            "discount_schedule": discount_schedule,
            "fcf_growth_method": "avg_fcf_17_20_over_avg_fcf_1_4_four_year_cagr",
            "fcf_growth_projection_engine": growth_profile,
            "terminal_assumptions": {
                "method": "weighted_average_final_4_quarters",
                "weights": [0.1, 0.2, 0.3, 0.4],
                "terminal_base_quarterly": fcf_terminal,
                "terminal_base_annual": annual_terminal_fcf,
                "fcf_terminal": fcf_terminal,
                "annualized_terminal_fcf": annual_terminal_fcf,
                "terminal_growth": terminal_growth,
                "terminal_growth_policy": terminal_growth_trace,
                "terminal_growth_source": terminal_growth_trace["terminal_growth_source"],
                "terminal_growth_source_type": terminal_growth_trace["terminal_growth_source_type"],
                "terminal_provider": terminal_growth_trace["terminal_provider"],
                "terminal_fallback_used": terminal_growth_trace["terminal_fallback_used"],
                "terminal_parser_strategy": terminal_growth_trace["terminal_parser_strategy"],
                "provider_long_term_growth": terminal_growth_trace["provider_long_term_growth"],
                "provider_terminal_growth": terminal_growth_trace["provider_terminal_growth"],
                "wacc": wacc_inputs.wacc,
                "formula": "TV = terminal_base_annual * (1 + g) / (WACC - g)",
            },
            "forecast_financials": forecast_records,
            "forecast_fcf_trace": forecast_fcf_trace,
            "projected_fcf": projected_fcf,
            "valuation_paths": component_values,
            "ensemble_weights": weights,
            "wacc": wacc_inputs.model_dump(),
            "wacc_decomposition": _wacc_decomposition(wacc_inputs),
            "wacc_trace": _wacc_decomposition(wacc_inputs),
            "net_debt_bridge": {"debt": float(debt), "cash": float(cash), "net_debt": net_debt},
            "primary_valuation_driver": "state_space_projected_quarterly_fcf_path",
            "provider_long_term_growth": terminal_growth_trace["provider_long_term_growth"],
            "provider_terminal_growth": terminal_growth_trace["provider_terminal_growth"],
            "terminal_growth_source_type": terminal_growth_trace["terminal_growth_source_type"],
            "terminal_growth_fallback_used": terminal_growth_trace["terminal_fallback_used"],
            "diagnostic_flags": _diagnostic_flags(
                diagnostic_fcf_growth,
                float(intrinsic),
                market.price,
                projected_fcf,
                fcf_terminal,
                forecast_financials,
            ),
            "terminal_trace": {
                "fcf_17": projected_fcf[16],
                "fcf_18": projected_fcf[17],
                "fcf_19": projected_fcf[18],
                "fcf_20": projected_fcf[19],
                "weighted_terminal_base": fcf_terminal,
                "terminal_base_quarterly": fcf_terminal,
                "terminal_base_annual": annual_terminal_fcf,
                "annualized_terminal_fcf": annual_terminal_fcf,
                "terminal_value": tv,
                "pv_terminal": pv_terminal,
                "terminal_growth": terminal_growth,
                "terminal_growth_policy": terminal_growth_trace,
                "terminal_growth_source": terminal_growth_trace["terminal_growth_source"],
                "terminal_growth_source_type": terminal_growth_trace["terminal_growth_source_type"],
                "terminal_provider": terminal_growth_trace["terminal_provider"],
                "terminal_fallback_used": terminal_growth_trace["terminal_fallback_used"],
                "terminal_parser_strategy": terminal_growth_trace["terminal_parser_strategy"],
                "provider_long_term_growth": terminal_growth_trace["provider_long_term_growth"],
                "provider_terminal_growth": terminal_growth_trace["provider_terminal_growth"],
            },
        }
    )
    mc = _monte_carlo_valuation(
        projected_fcf,
        [0.0] * len(projected_fcf),
        [0.0] * len(projected_fcf),
        quarterly_wacc,
        terminal_growth / 4.0,
        wacc_inputs.wacc,
        terminal_growth,
        net_debt,
        float(shares),
        settings.random_seed,
        max(settings.monte_carlo_samples, 0),
    )
    sensitivity = (
        _sensitivity(projected_fcf, quarterly_wacc, terminal_growth / 4.0, net_debt, float(shares))
        if settings.run_sensitivity
        else {}
    )
    return DCFResult(
        ticker=ticker.upper(),
        explicit_forecast_quarters=len(projected_fcf),
        intrinsic_value_per_share=float(intrinsic),
        accounting_intrinsic=float(component_values["accounting"]["intrinsic"]),
        economic_intrinsic=float(component_values["economic"]["intrinsic"]),
        owner_intrinsic=float(component_values["owner"]["intrinsic"]),
        ensemble_intrinsic=float(ensemble_intrinsic),
        ensemble_weights=weights,
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        net_debt=net_debt,
        projected_fcf=projected_fcf,
        explicit_pv=explicit_pv,
        pv_projected_fcf=explicit_pv,
        terminal_value=tv,
        pv_terminal_value=pv_terminal,
        pv_terminal=pv_terminal,
        fcf_terminal_base=float(fcf_terminal),
        wacc=float(wacc_inputs.wacc),
        wacc_source=wacc_source_label(wacc_inputs),
        terminal_growth_rate=float(terminal_growth),
        fcf_growth_rate=float(diagnostic_fcf_growth),
        raw_fcf_growth_rate=float(fcf_growth_rate),
        smoothed_fcf_growth_rate=float(diagnostic_fcf_growth),
        clamped_fcf_growth_rate=float(diagnostic_fcf_growth),
        growth_clamped=False,
        clamp_triggered=False,
        valuation_p10=mc["valuation_p10"],
        valuation_p50=mc["valuation_p50"],
        valuation_p90=mc["valuation_p90"],
        diluted_shares_outstanding=float(shares),
        market_price=market.price,
        implied_upside=float(implied_upside) if implied_upside is not None else None,
        forecast_horizon_quarters=len(projected_fcf),
        forecast_mode=str(diag.get("forecast_mode", "state_space")),
        company_archetype=str(diag.get("company_archetype")) if diag.get("company_archetype") is not None else None,
        consensus_weight=float(diag.get("consensus_weight", 0.0) or 0.0),
        buyback_rate=float(diag.get("buyback_rate", 0.0) or 0.0),
        r_and_d_capitalized=bool(diag.get("r_and_d_capitalized", False)),
        segment_mode=str(diag.get("segment_mode", "disabled")),
        terminal_growth_source_type=str(terminal_growth_trace.get("terminal_growth_source_type")),
        provider_long_term_growth=terminal_growth_trace.get("provider_long_term_growth"),
        provider_terminal_growth=terminal_growth_trace.get("provider_terminal_growth"),
        valuation_regime=str(diag.get("valuation_regime", "mature_compounder")),
        scenario_values=diag.get("scenario_values", {}),
        probability_weights=diag.get("probability_weights", {}),
        reverse_dcf_expectations=diag.get("reverse_dcf_expectations", {}),
        scenario_generation_method=diag.get("scenario_generation_method"),
        probability_method=diag.get("probability_method"),
        monte_carlo_distribution=diag.get("monte_carlo_distribution", {}),
        forecast_financials=forecast_records,
        confidence_interval={
            "low_intrinsic_value": mc["valuation_p10"],
            "median_intrinsic_value": mc["valuation_p50"],
            "high_intrinsic_value": mc["valuation_p90"],
        },
        sensitivity=sensitivity,
        diagnostics=diag,
        sources=sources,
        warnings=warnings,
    )


def build_dcf_result(
    ticker: str,
    forecast: GrowthForecast,
    wacc_inputs: WACCInputs,
    market: MarketSnapshot,
    latest_cash: float | None,
    latest_debt: float | None,
    latest_shares: float | None,
    sources: list[SourceMeta],
    warnings: list[str],
    settings: Settings,
    latest_operating_lease_liability: float | None = None,
) -> DCFResult:
    timings: dict[str, float] = {}
    projected_fcf = forecast.fcf_path
    if not projected_fcf:
        raise ValueError("Forecast must contain at least one projected FCF")
    if len(projected_fcf) != settings.explicit_forecast_quarters:
        raise ValueError(
            f"DCF requires exactly {settings.explicit_forecast_quarters} explicit forecast quarters; "
            f"received {len(projected_fcf)}."
        )
    quarterly_wacc = wacc_inputs.wacc / 4.0
    terminal_growth_assumption = resolve_terminal_growth(ticker, wacc_inputs.wacc, settings)
    annual_terminal_growth = terminal_growth_assumption.terminal_growth
    quarterly_terminal_growth = annual_terminal_growth / 4.0
    terminal_fcf_path = forecast.diagnostics.get("terminal_fcf_path") or projected_fcf
    fcf_terminal_base = _stabilized_terminal_fcf([float(x) for x in terminal_fcf_path])
    annual_terminal_fcf = fcf_terminal_base * 4.0
    tv = terminal_value(annual_terminal_fcf, wacc_inputs.wacc, annual_terminal_growth)
    pv_fcf = present_value_series(projected_fcf, quarterly_wacc)
    pv_tv = present_value(tv, quarterly_wacc, len(projected_fcf))
    enterprise_value = pv_fcf + pv_tv

    cash = latest_cash if latest_cash is not None else market.cash or 0.0
    debt = latest_debt if latest_debt is not None else market.debt or 0.0
    lease_adjustment = latest_operating_lease_liability or 0.0
    if settings.company_type in {"marketplace_hybrid"} or settings.normalize_leases:
        debt += lease_adjustment
    else:
        lease_adjustment = 0.0
    net_debt = max(debt - cash, -cash)
    equity_value = enterprise_value - net_debt
    shares = latest_shares or market.shares_outstanding
    if not shares or shares <= 0:
        raise ValueError("Diluted shares outstanding are required for per-share value")
    intrinsic = equity_value / shares
    implied_upside = (intrinsic / market.price - 1.0) if market.price else None
    cash_flow_diagnostics = _cash_flow_diagnostics(forecast)
    warning_flags = _valuation_warning_flags(cash_flow_diagnostics, settings)
    margin_warnings = [
        flag["message"]
        for flag in warning_flags
        if flag.get("severity") == "warning"
    ]
    start = perf_counter()
    if settings.run_reverse_dcf:
        reverse_dcf_growth = _reverse_dcf_growth(
            market.price,
            shares,
            net_debt,
            projected_fcf[0],
            _latest_metric(cash_flow_diagnostics, "economic_fcf_margin"),
            _latest_metric(cash_flow_diagnostics, "maintenance_capex"),
            _latest_revenue(cash_flow_diagnostics),
            forecast.single_growth_rate,
            quarterly_wacc,
            quarterly_terminal_growth,
            len(projected_fcf),
        )
    else:
        reverse_dcf_growth = {
            "market_implied_annual_fcf_growth": None,
            "market_implied_terminal_growth": None,
        }
    timings["reverse_dcf"] = perf_counter() - start
    start = perf_counter()
    confidence_score = _valuation_confidence_score(
        warnings + forecast.warnings,
        forecast,
        warning_flags,
        settings,
    )
    timings["confidence"] = perf_counter() - start
    start = perf_counter()
    mc = _monte_carlo_valuation(
        projected_fcf,
        forecast.lower_growth_path,
        forecast.upper_growth_path,
        quarterly_wacc,
        quarterly_terminal_growth,
        wacc_inputs.wacc,
        annual_terminal_growth,
        net_debt,
        shares,
        settings.random_seed,
        max(settings.monte_carlo_samples, 0),
    )
    timings["monte_carlo"] = perf_counter() - start
    confidence = {
        "low_intrinsic_value": mc["valuation_p10"],
        "median_intrinsic_value": mc["valuation_p50"],
        "high_intrinsic_value": mc["valuation_p90"],
    }
    start = perf_counter()
    sensitivity = (
        _sensitivity(
            projected_fcf,
            quarterly_wacc,
            quarterly_terminal_growth,
            net_debt,
            shares,
            forecast,
        )
        if settings.run_sensitivity
        else {}
    )
    timings["sensitivity"] = perf_counter() - start
    return DCFResult(
        ticker=ticker.upper(),
        explicit_forecast_quarters=len(projected_fcf),
        intrinsic_value_per_share=float(intrinsic),
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        net_debt=float(net_debt),
        projected_fcf=[float(x) for x in projected_fcf],
        explicit_pv=float(pv_fcf),
        pv_projected_fcf=float(pv_fcf),
        terminal_value=float(tv),
        pv_terminal_value=float(pv_tv),
        pv_terminal=float(pv_tv),
        fcf_terminal_base=float(fcf_terminal_base),
        wacc=float(wacc_inputs.wacc),
        terminal_growth_rate=float(annual_terminal_growth),
        fcf_growth_rate=float(forecast.single_growth_rate),
        raw_fcf_growth_rate=float(forecast.raw_fcf_growth_rate),
        smoothed_fcf_growth_rate=float(forecast.smoothed_fcf_growth_rate),
        clamped_fcf_growth_rate=float(forecast.clamped_fcf_growth_rate),
        growth_clamped=forecast.growth_clamped,
        clamp_triggered=forecast.clamp_triggered,
        valuation_p10=mc["valuation_p10"],
        valuation_p50=mc["valuation_p50"],
        valuation_p90=mc["valuation_p90"],
        diluted_shares_outstanding=float(shares),
        market_price=market.price,
        implied_upside=float(implied_upside) if implied_upside is not None else None,
        forecast_horizon_quarters=len(projected_fcf),
        forecast_mode=settings.forecast_mode,
        company_archetype=None,
        consensus_weight=settings.consensus_weight,
        buyback_rate=None,
        r_and_d_capitalized=settings.capitalize_rd,
        segment_mode=settings.segment_mode,
        terminal_growth_source_type=None,
        provider_long_term_growth=None,
        provider_terminal_growth=None,
        valuation_regime=None,
        scenario_values={},
        probability_weights={},
        reverse_dcf_expectations={},
        scenario_generation_method=None,
        probability_method=None,
        monte_carlo_distribution={},
        forecast_financials=forecast.diagnostics.get("forecast_components", []),
        confidence_interval=confidence,
        sensitivity=sensitivity,
        diagnostics={
            "forecast": forecast.diagnostics,
            "model_type": forecast.model_type,
            "valuation_mode": forecast.diagnostics.get("valuation_mode", settings.valuation_mode),
            "company_type": settings.company_type,
            "maintenance_capex_method": forecast.diagnostics.get("maintenance_capex_method"),
            "maintenance_capex_estimates": forecast.diagnostics.get("maintenance_capex_estimates"),
            "cash_flow_diagnostics": cash_flow_diagnostics,
            "economic_fcf_margin": _latest_metric(cash_flow_diagnostics, "economic_fcf_margin"),
            "reported_fcf_margin": _latest_metric(cash_flow_diagnostics, "reported_fcf_margin"),
            "maintenance_capex_ratio": _latest_metric(cash_flow_diagnostics, "maintenance_capex_ratio"),
            "growth_capex_ratio": _latest_metric(cash_flow_diagnostics, "growth_capex_ratio"),
            "owner_earnings": _latest_metric(cash_flow_diagnostics, "owner_earnings"),
            "owner_earnings_margin": _latest_metric(cash_flow_diagnostics, "owner_earnings_margin"),
            "sbc_adjustment": _latest_metric(cash_flow_diagnostics, "sbc_adjustment"),
            "lease_adjustment": lease_adjustment,
            "valuation_warning_flags": warning_flags,
            "reverse_dcf_growth": reverse_dcf_growth,
            "market_implied_growth": reverse_dcf_growth.get("market_implied_annual_fcf_growth"),
            "market_implied_terminal_growth": reverse_dcf_growth.get("market_implied_terminal_growth"),
            "valuation_confidence_score": confidence_score,
            "wacc": wacc_inputs.model_dump(),
            "wacc_decomposition": {
                "wacc_source": _wacc_source_note(wacc_inputs, "wacc_source"),
                "provider": _wacc_source_note(wacc_inputs, "provider"),
                "fallback_used": _wacc_source_note(wacc_inputs, "fallback_used") == "true",
                "extracted_selected_wacc": _wacc_float_note(wacc_inputs, "extracted_selected_wacc"),
                "extracted_range_low": _wacc_float_note(wacc_inputs, "extracted_range_low"),
                "extracted_range_high": _wacc_float_note(wacc_inputs, "extracted_range_high"),
                "parser_strategy_used": _wacc_source_note(wacc_inputs, "parser_strategy_used"),
                "risk_free_rate": wacc_inputs.risk_free_rate,
                "equity_risk_premium": wacc_inputs.equity_risk_premium,
                "beta": wacc_inputs.beta,
                "cost_of_equity": wacc_inputs.cost_of_equity,
                "cost_of_debt": wacc_inputs.pre_tax_cost_of_debt,
                "tax_shield": 1.0 - wacc_inputs.tax_rate,
                "after_tax_cost_of_debt": wacc_inputs.after_tax_cost_of_debt,
                "debt_weight": wacc_inputs.debt_weight,
                "equity_weight": wacc_inputs.equity_weight,
                "final_wacc": wacc_inputs.wacc,
                "source_notes": wacc_inputs.source_notes,
            },
            "quarterly_discount_rate": quarterly_wacc,
            "quarterly_terminal_growth_rate": quarterly_terminal_growth,
            "terminal_growth_policy": _terminal_growth_trace(terminal_growth_assumption, wacc_inputs.wacc, settings),
            "terminal_stabilization": {
                "method": "weighted_average_final_4_quarters",
                "cash_flow_basis": "economic_fcf" if forecast.diagnostics.get("terminal_fcf_path") else "selected_fcf",
                "quarters_used": [len(projected_fcf) - 3, len(projected_fcf) - 2, len(projected_fcf) - 1, len(projected_fcf)],
                "quarterly_fcf_terminal_base": fcf_terminal_base,
                "annualized_terminal_fcf": annual_terminal_fcf,
            },
            "monte_carlo": mc,
            "timing": timings,
        },
        sources=sources,
        warnings=warnings + forecast.warnings + margin_warnings,
    )


def _terminal_growth_trace(assumption, annual_wacc: float, settings: Settings) -> dict[str, object]:
    return {
        "method": "provider_resolved_terminal_growth",
        "company_type": settings.company_type,
        "terminal_growth_source": assumption.terminal_growth_source,
        "terminal_growth_source_type": assumption.terminal_growth_source_type,
        "terminal_provider": assumption.provider,
        "terminal_fallback_used": assumption.fallback_used,
        "terminal_parser_strategy": assumption.parser_strategy,
        "raw_parsed_values": assumption.raw_parsed_values,
        "provider_long_term_growth": assumption.provider_long_term_growth,
        "provider_terminal_growth": assumption.provider_terminal_growth,
        "terminal_growth": float(assumption.terminal_growth),
        "long_term_growth": assumption.long_term_growth,
        "retrieved_at": assumption.retrieved_at.isoformat() if assumption.retrieved_at else None,
        "provider_error": assumption.provider_error,
        "spread_guardrail_bps": 150.0,
        "max_allowed_by_wacc": float(max(annual_wacc - 0.015, 0.0)),
        "uses_short_horizon_roic_reinvestment": 0.0,
    }


def _finite_or_default(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else float(default)


def _smoothed_forecast_fcf_growth(projected_fcf: list[float]) -> float:
    if len(projected_fcf) < 8:
        return float("nan")
    start = float(np.nanmean(np.asarray(projected_fcf[:4], dtype=float)))
    end = float(np.nanmean(np.asarray(projected_fcf[-4:], dtype=float)))
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0.0 or end <= 0.0:
        return float("nan")
    return float((end / start) ** 0.25 - 1.0)


def _fcf_growth_projection_profile(
    projected_fcf: list[float],
    forecast_financials: pd.DataFrame,
    base_fcf: float,
    annual_wacc: float,
    policy_terminal_growth: float,
    settings: Settings,
) -> dict[str, float | str | dict[str, float]]:
    path = np.asarray(projected_fcf, dtype=float)
    clean = path[np.isfinite(path)]
    if clean.size < 8:
        explicit_growth = 0.0
        path_cagr = 0.0
        median_annual_growth = 0.0
    else:
        start = _positive_mean(clean[:4], fallback=max(float(base_fcf), 1.0))
        end = _positive_mean(clean[-4:], fallback=start)
        path_cagr = (end / start) ** 0.25 - 1.0 if start > 0.0 and end > 0.0 else 0.0
        quarterly_growth = []
        for previous, current in zip(clean, clean[1:]):
            if previous > 0.0 and current > 0.0:
                quarterly_growth.append(current / previous - 1.0)
        median_quarterly_growth = float(np.nanmedian(quarterly_growth[-8:])) if quarterly_growth else 0.0
        median_annual_growth = (1.0 + median_quarterly_growth) ** 4 - 1.0
        explicit_growth = 0.70 * path_cagr + 0.30 * median_annual_growth
    # The [min_mature_growth, max_mature_growth] band is a guardrail for the
    # terminal-growth *blending input* only. Applying it to the reported explicit
    # FCF growth flattened every above-cap name (GOOGL/NVDA/TSLA) to an identical
    # 0.15, erasing cross-sectional differentiation in the headline metric while
    # not affecting valuation (the discounted EV uses projected_fcf directly and
    # the clamped 0.15 stays above terminal policy, so the terminal branch is
    # unchanged). Report the true blended path growth; clamp only the copy that
    # feeds the terminal-growth logic so intrinsic value is bit-for-bit unchanged.
    reported_growth = float(explicit_growth)
    terminal_blend_growth = float(np.clip(explicit_growth, settings.min_mature_growth, settings.max_mature_growth))
    roic_terminal = _roic_reinvestment_terminal_growth(forecast_financials)
    policy = float(policy_terminal_growth)
    flat_explicit_path = abs(terminal_blend_growth) < 0.002
    if flat_explicit_path:
        terminal_raw = policy
    else:
        terminal_raw = 0.55 * policy + 0.45 * roic_terminal
    if not flat_explicit_path and terminal_blend_growth < 0.0:
        terminal_raw = min(terminal_raw, policy)
    elif not flat_explicit_path and terminal_blend_growth < policy:
        terminal_raw = min(terminal_raw, 0.70 * policy + 0.30 * max(terminal_blend_growth, 0.0))
    terminal_growth = float(
        np.clip(
            terminal_raw,
            settings.terminal_growth_floor,
            max(settings.terminal_growth_floor, min(settings.terminal_growth_cap, annual_wacc - 0.015)),
        )
    )
    return {
        "method": "ensemble_fcf_path_cagr_blended_with_tail_roic_reinvestment",
        "explicit_fcf_growth_rate": reported_growth,
        "terminal_blend_growth_rate": terminal_blend_growth,
        "path_cagr": float(path_cagr),
        "median_annual_growth": float(median_annual_growth),
        "terminal_growth_rate": terminal_growth,
        "policy_terminal_growth": policy,
        "roic_reinvestment_terminal_growth": roic_terminal,
        "guardrails": {
            "growth_floor": float(settings.min_mature_growth),
            "growth_cap": float(settings.max_mature_growth),
            "terminal_floor": float(settings.terminal_growth_floor),
            "terminal_cap": float(max(settings.terminal_growth_floor, min(settings.terminal_growth_cap, annual_wacc - 0.015))),
        },
    }


def _positive_mean(values: np.ndarray, fallback: float) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean) & (clean > 0.0)]
    return float(np.nanmean(clean)) if clean.size else float(fallback)


def _roic_reinvestment_terminal_growth(forecast_financials: pd.DataFrame) -> float:
    if not {"reinvestment_rate", "roic"}.issubset(forecast_financials.columns):
        return 0.025
    tail = forecast_financials.tail(8)
    reinvestment = float(np.nanmedian(tail["reinvestment_rate"].astype(float)))
    roic = float(np.nanmedian(tail["roic"].astype(float)))
    annual_roic = roic * 4.0 if roic < 0.75 else roic
    stable_growth = annual_roic * reinvestment
    return float(np.clip(stable_growth, 0.0, 0.04))


def _wacc_decomposition(wacc_inputs: WACCInputs) -> dict[str, object]:
    return {
        "wacc_source": _wacc_source_note(wacc_inputs, "wacc_source"),
        "provider": _wacc_source_note(wacc_inputs, "provider"),
        "fallback_used": _wacc_source_note(wacc_inputs, "fallback_used") == "true",
        "extracted_selected_wacc": _wacc_float_note(wacc_inputs, "extracted_selected_wacc"),
        "extracted_range_low": _wacc_float_note(wacc_inputs, "extracted_range_low"),
        "extracted_range_high": _wacc_float_note(wacc_inputs, "extracted_range_high"),
        "parser_strategy_used": _wacc_source_note(wacc_inputs, "parser_strategy_used"),
        "risk_free_rate": wacc_inputs.risk_free_rate,
        "equity_risk_premium": wacc_inputs.equity_risk_premium,
        "beta": wacc_inputs.beta,
        "cost_of_equity": wacc_inputs.cost_of_equity,
        "cost_of_debt": wacc_inputs.pre_tax_cost_of_debt,
        "tax_shield": 1.0 - wacc_inputs.tax_rate,
        "after_tax_cost_of_debt": wacc_inputs.after_tax_cost_of_debt,
        "market_value_equity": wacc_inputs.market_value_equity,
        "market_value_debt": wacc_inputs.market_value_debt,
        "equity_weight": wacc_inputs.equity_weight,
        "debt_weight": wacc_inputs.debt_weight,
        "final_wacc": wacc_inputs.wacc,
        "source_notes": wacc_inputs.source_notes,
    }


def _wacc_source_note(wacc_inputs: WACCInputs, key: str) -> str | None:
    prefix = f"{key}="
    for note in wacc_inputs.source_notes:
        if note.startswith(prefix):
            return note[len(prefix) :]
    return None


def _wacc_float_note(wacc_inputs: WACCInputs, key: str) -> float | None:
    value = _wacc_source_note(wacc_inputs, key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _value_cash_flow_path(
    projected_fcf: list[float],
    forecast_financials: pd.DataFrame,
    wacc_inputs: WACCInputs,
    market: MarketSnapshot,
    latest_cash: float | None,
    latest_debt: float | None,
    latest_shares: float | None,
    terminal_growth: float,
) -> dict[str, float]:
    quarterly_wacc = wacc_inputs.wacc / 4.0
    explicit_pv = present_value_series(projected_fcf, quarterly_wacc)
    terminal_fcf = _stabilized_terminal_fcf(projected_fcf)
    annual_terminal_fcf = terminal_fcf * 4.0
    tv = terminal_value(annual_terminal_fcf, wacc_inputs.wacc, terminal_growth)
    pv_terminal = present_value(tv, quarterly_wacc, len(projected_fcf))
    enterprise_value = explicit_pv + pv_terminal
    cash = latest_cash if latest_cash is not None else market.cash or 0.0
    debt = latest_debt if latest_debt is not None else market.debt or 0.0
    equity_value = enterprise_value - float(debt - cash)
    shares = latest_shares or market.shares_outstanding
    if not shares or shares <= 0:
        raise ValueError("Diluted shares outstanding are required for per-share value")
    return {
        "explicit_pv": float(explicit_pv),
        "terminal_fcf": float(terminal_fcf),
        "terminal_value": float(tv),
        "pv_terminal": float(pv_terminal),
        "enterprise_value": float(enterprise_value),
        "equity_value": float(equity_value),
        "intrinsic": float(equity_value / shares),
    }


def _ensemble_weights(company_type: str, forecast_financials: pd.DataFrame) -> dict[str, float]:
    if company_type in {"asset_light_quality", "consumer_compounder", "consumer_staple", "platform"}:
        weights = {"owner": 0.50, "accounting": 0.30, "economic": 0.20}
    elif company_type in {"industrial", "retail", "telecom", "semiconductor"}:
        weights = {"accounting": 0.50, "owner": 0.30, "economic": 0.20}
    elif company_type in {"hyperscaler", "marketplace_hybrid"}:
        weights = {"economic": 0.40, "owner": 0.40, "accounting": 0.20}
    else:
        weights = {"owner": 0.40, "accounting": 0.35, "economic": 0.25}
    if "reinvestment_rate" in forecast_financials and forecast_financials["reinvestment_rate"].std() > 0.18:
        shift = min(weights["economic"] * 0.50, 0.15)
        weights["economic"] -= shift
        weights["owner"] += shift * 0.70
        weights["accounting"] += shift * 0.30
    if {"accounting_fcf", "owner_earnings"}.issubset(forecast_financials.columns):
        accounting = forecast_financials["accounting_fcf"].astype(float)
        owner = forecast_financials["owner_earnings"].astype(float)
        if accounting.std() > max(abs(owner.mean()), 1.0) * 0.50:
            shift = min(weights["accounting"] * 0.40, 0.12)
            weights["accounting"] -= shift
            weights["owner"] += shift
    total = sum(weights.values())
    return {key: float(value / total) for key, value in weights.items()}


def _adjust_weights_for_path_dispersion(
    weights: dict[str, float],
    component_values: dict[str, dict[str, float]],
) -> dict[str, float]:
    adjusted = dict(weights)
    economic = component_values["economic"]["intrinsic"]
    accounting = component_values["accounting"]["intrinsic"]
    owner = component_values["owner"]["intrinsic"]
    if economic > 0 and economic < min(accounting, owner) * 0.75:
        shift = min(adjusted["economic"] * 0.50, 0.20)
        adjusted["economic"] -= shift
        adjusted["owner"] += shift * 0.65
        adjusted["accounting"] += shift * 0.35
    if accounting > 0 and accounting < min(economic, owner) * 0.70:
        shift = min(adjusted["accounting"] * 0.40, 0.12)
        adjusted["accounting"] -= shift
        adjusted["owner"] += shift
    total = sum(adjusted.values())
    return {key: float(value / total) for key, value in adjusted.items()}


def _stable_terminal_fcf(forecast_financials: pd.DataFrame, projected_fcf: list[float]) -> float:
    if {"nopat", "reinvestment_rate"}.issubset(forecast_financials.columns):
        stable_nopat = _weighted_tail(forecast_financials["nopat"].to_numpy(float))
        stable_reinvestment = _weighted_tail(forecast_financials["reinvestment_rate"].to_numpy(float))
        return float(stable_nopat * (1.0 - stable_reinvestment))
    return _stabilized_terminal_fcf(projected_fcf)


def _weighted_tail(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 4:
        return float(clean[-1]) if clean.size else 0.0
    weights = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    return float(np.average(clean[-4:], weights=weights))


def _diagnostic_flags(
    fcf_growth_rate: float,
    intrinsic: float,
    market_price: float | None,
    projected_fcf: list[float],
    terminal_fcf: float,
    forecast_financials: pd.DataFrame,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    if market_price and fcf_growth_rate > 0.20 and intrinsic < market_price * 0.80:
        flags.append(
            {
                "flag": "high_growth_low_value_inconsistency",
                "severity": "warning",
                "message": "FCF growth exceeds 20% while intrinsic value is below 80% of market price.",
                "fcf_growth_rate": float(fcf_growth_rate),
                "intrinsic_value": float(intrinsic),
                "market_price": float(market_price),
            }
        )
    if projected_fcf and terminal_fcf < 0.75 * max(projected_fcf[-1], 1.0):
        flags.append(
            {
                "flag": "terminal_fcf_below_final_fcf",
                "severity": "info",
                "message": "Weighted terminal FCF is materially below final forecast FCF.",
                "terminal_fcf": float(terminal_fcf),
                "fcf_20": float(projected_fcf[-1]),
            }
        )
    for column, threshold in [("reinvestment_rate", 0.20), ("roic", 0.06)]:
        if column in forecast_financials:
            values = forecast_financials[column].astype(float).to_numpy()
            if len(values) >= 4 and np.nanstd(values) > threshold:
                flags.append(
                    {
                        "flag": f"{column}_unstable",
                        "severity": "warning",
                        "message": f"Forecast {column} is unstable.",
                        "std": float(np.nanstd(values)),
                        "threshold": threshold,
                    }
                )
    if "fallback_triggered" in forecast_financials and forecast_financials["fallback_triggered"].sum() > 0:
        flags.append(
            {
                "flag": "forecast_fallback_triggered",
                "severity": "warning",
                "message": "Fallback owner-earnings or marketplace policy was used in forecast.",
                "quarters": int(forecast_financials["fallback_triggered"].sum()),
            }
        )
    for ratio in ["capex_ratio", "nwc_ratio"]:
        if ratio in forecast_financials:
            max_abs = float(forecast_financials[ratio].abs().max())
            threshold = 0.25 if ratio == "nwc_ratio" else 0.30
            if max_abs > threshold:
                flags.append(
                    {
                        "flag": f"{ratio}_unstable",
                        "severity": "warning",
                        "message": f"Forecast {ratio} exceeds stability threshold.",
                        "max_abs": max_abs,
                        "threshold": threshold,
                    }
                )
    return flags


def _confidence_interval(
    lower_growth: list[float],
    upper_growth: list[float],
    first_fcf: float,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
) -> dict[str, float]:
    low_fcf = _path_from_growth(first_fcf, lower_growth)
    high_fcf = _path_from_growth(first_fcf, upper_growth)
    return {
        "low_intrinsic_value": _value_from_path(low_fcf, wacc, terminal_growth, net_debt, shares),
        "high_intrinsic_value": _value_from_path(high_fcf, wacc, terminal_growth, net_debt, shares),
    }


def _path_from_growth(first_fcf: float, growth_path: list[float]) -> list[float]:
    values = [first_fcf]
    for growth in growth_path[1:]:
        values.append(values[-1] * (1.0 + growth))
    return values


def _value_from_path(
    fcf_path: list[float],
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
) -> float:
    annual_wacc = wacc * 4.0
    annual_terminal_growth = terminal_growth * 4.0
    annual_terminal_fcf = _stabilized_terminal_fcf(fcf_path) * 4.0
    tv = terminal_value(
        annual_terminal_fcf,
        annual_wacc,
        min(annual_terminal_growth, annual_wacc - 0.01),
    )
    ev = present_value_series(fcf_path, wacc) + present_value(tv, wacc, len(fcf_path))
    return float((ev - net_debt) / shares)


def _sensitivity(
    projected_fcf: list[float],
    base_wacc: float,
    base_terminal_growth: float,
    net_debt: float,
    shares: float,
    forecast: GrowthForecast | None = None,
) -> dict[str, object]:
    base_annual_wacc = base_wacc * 4.0
    annual_wacc_values = [base_annual_wacc - 0.01, base_annual_wacc, base_annual_wacc + 0.01]
    annual_growth_values = [0.02, 0.025, 0.03, 0.035]
    table = []
    for annual_wacc in annual_wacc_values:
        quarterly_wacc = annual_wacc / 4.0
        row = {"annual_wacc": round(annual_wacc, 5), "values": []}
        for annual_growth in annual_growth_values:
            quarterly_growth = annual_growth / 4.0
            try:
                value = _value_from_path(
                    projected_fcf,
                    quarterly_wacc,
                    quarterly_growth,
                    net_debt,
                    shares,
                )
            except Exception:
                value = None
            row["values"].append(value)
        table.append(row)
    return {
        "annual_terminal_growth_values": annual_growth_values,
        "rows": table,
        "maintenance_capex_surface": _maintenance_capex_surface(
            forecast,
            base_wacc,
            base_terminal_growth,
            net_debt,
            shares,
        )
        if forecast is not None
        else [],
    }


def _monte_carlo_valuation(
    projected_fcf: list[float],
    lower_growth_path: list[float],
    upper_growth_path: list[float],
    quarterly_wacc: float,
    quarterly_terminal_growth: float,
    annual_wacc: float,
    annual_terminal_growth: float,
    net_debt: float,
    shares: float,
    seed: int,
    n_sims: int = 2000,
) -> dict[str, float]:
    if n_sims <= 0:
        value = _value_from_path(
            projected_fcf,
            quarterly_wacc,
            quarterly_terminal_growth,
            net_debt,
            shares,
        )
        return {
            "valuation_p10": float(value),
            "valuation_p50": float(value),
            "valuation_p90": float(value),
            "simulation_count": 0.0,
        }
    rng = np.random.default_rng(seed)
    base_growth = np.asarray(_growth_from_path(projected_fcf), dtype=float)
    lower = np.asarray(lower_growth_path[: len(base_growth)], dtype=float)
    upper = np.asarray(upper_growth_path[: len(base_growth)], dtype=float)
    if len(lower) != len(base_growth) or len(upper) != len(base_growth):
        sigma = np.full(len(base_growth), 0.03)
    else:
        sigma = np.clip((upper - lower) / 3.92, 0.005, 0.08)
    values = []
    first_fcf = projected_fcf[0]
    for _ in range(n_sims):
        sampled_growth = rng.normal(base_growth, sigma)
        sampled_growth = np.clip(sampled_growth, -0.25, 0.25)
        path = [first_fcf]
        for growth in sampled_growth[1:]:
            path.append(path[-1] * (1.0 + float(growth)))
        sampled_annual_wacc = max(rng.normal(annual_wacc, 0.01), annual_terminal_growth + 0.01)
        sampled_annual_terminal = min(
            max(rng.normal(annual_terminal_growth, 0.005), 0.0),
            sampled_annual_wacc - 0.01,
        )
        values.append(
            _value_from_path(
                path,
                sampled_annual_wacc / 4.0,
                sampled_annual_terminal / 4.0,
                net_debt,
                shares,
            )
        )
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {
        "valuation_p10": float(p10),
        "valuation_p50": float(p50),
        "valuation_p90": float(p90),
        "simulation_count": float(n_sims),
    }


def _growth_from_path(projected_fcf: list[float]) -> list[float]:
    if not projected_fcf:
        return []
    growth = [0.0]
    for previous, current in zip(projected_fcf, projected_fcf[1:]):
        growth.append((current - previous) / max(abs(previous), 1.0))
    return growth


def _stabilized_terminal_fcf(projected_fcf: list[float]) -> float:
    if len(projected_fcf) < 4:
        return float(projected_fcf[-1])
    final_four = np.asarray(projected_fcf[-4:], dtype=float)
    weights = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    return float(np.average(final_four, weights=weights))


def _cash_flow_diagnostics(forecast: GrowthForecast) -> list[dict[str, float | str]]:
    rows = forecast.diagnostics.get("forecast_components") or []
    mode = forecast.diagnostics.get("valuation_mode", "reported")
    diagnostics = []
    for row in rows:
        diagnostics.append(
            {
                "quarter": row.get("quarter"),
                "revenue": row.get("revenue"),
                "reported_capex": row.get("reported_capex", row.get("capex")),
                "maintenance_capex": row.get("maintenance_capex"),
                "growth_capex": row.get("growth_capex"),
                "economic_fcf": row.get("economic_fcf"),
                "reported_fcf": row.get("reported_fcf"),
                "owner_earnings": row.get("owner_earnings"),
                "sbc_adjustment": row.get("sbc_adjustment"),
                "sbc_addback": row.get("sbc_addback"),
                "economic_fcf_margin": _safe_ratio(row.get("economic_fcf"), row.get("revenue")),
                "reported_fcf_margin": _safe_ratio(row.get("reported_fcf"), row.get("revenue")),
                "owner_earnings_margin": _safe_ratio(row.get("owner_earnings"), row.get("revenue")),
                "maintenance_capex_ratio": _safe_ratio(row.get("maintenance_capex"), row.get("revenue")),
                "growth_capex_ratio": _safe_ratio(row.get("growth_capex"), row.get("revenue")),
                "valuation_mode": mode,
            }
        )
    return diagnostics


def _valuation_warning_flags(
    cash_flow_diagnostics: list[dict[str, float | str]],
    settings: Settings,
) -> list[dict[str, float | str]]:
    if not cash_flow_diagnostics or settings.allow_high_economic_fcf_margin:
        return []
    latest_margin = _latest_metric(cash_flow_diagnostics, "economic_fcf_margin")
    if latest_margin is None:
        return []
    threshold = 0.35 if settings.company_type == "hyperscaler" else 0.45
    if settings.company_type in {"consumer_staple", "platform"}:
        threshold = 0.45
    if latest_margin <= threshold:
        return []
    return [
        {
            "flag": "economic_fcf_margin_high",
            "severity": "warning",
            "value": float(latest_margin),
            "threshold": float(threshold),
            "message": (
                f"Economic FCF margin {latest_margin:.1%} exceeds "
                f"{settings.company_type} guardrail {threshold:.1%}."
            ),
        }
    ]


def _latest_metric(
    rows: list[dict[str, float | str]],
    key: str,
) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None


def _latest_revenue(rows: list[dict[str, float | str]]) -> float | None:
    return _latest_metric(rows, "revenue")


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return None
    return float(numerator / denominator)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for record in frame.to_dict(orient="records"):
        row = {}
        for key, value in record.items():
            if hasattr(value, "isoformat"):
                row[key] = value.isoformat()
            elif isinstance(value, float) and not np.isfinite(value):
                row[key] = None
            else:
                row[key] = value
        rows.append(row)
    return rows


def _reverse_dcf_growth(
    market_price: float | None,
    shares: float,
    net_debt: float,
    first_fcf: float,
    latest_margin: float | None,
    latest_maintenance_capex: float | None,
    latest_revenue: float | None,
    model_annual_growth: float,
    quarterly_wacc: float,
    quarterly_terminal_growth: float,
    horizon_quarters: int,
) -> dict[str, float | None]:
    if market_price is None or shares <= 0 or first_fcf <= 0:
        return {"market_implied_annual_fcf_growth": None}
    target_enterprise_value = market_price * shares + net_debt

    def value_at_growth(annual_growth: float) -> float:
        quarterly_growth = (1.0 + annual_growth) ** 0.25 - 1.0
        path = []
        fcf = first_fcf
        for _ in range(horizon_quarters):
            fcf *= 1.0 + quarterly_growth
            path.append(fcf)
        terminal_fcf = _stabilized_terminal_fcf(path) * 4.0
        tv = terminal_value(terminal_fcf, quarterly_wacc * 4.0, quarterly_terminal_growth * 4.0)
        return present_value_series(path, quarterly_wacc) + present_value(tv, quarterly_wacc, len(path))

    low, high = -0.30, 0.40
    for _ in range(80):
        mid = (low + high) / 2.0
        if value_at_growth(mid) < target_enterprise_value:
            low = mid
        else:
            high = mid
    maintenance_delta = _solve_implied_maintenance_delta(
        target_enterprise_value,
        first_fcf,
        model_annual_growth,
        quarterly_wacc,
        quarterly_terminal_growth,
        horizon_quarters,
    )
    implied_margin = None
    if latest_revenue and latest_revenue > 0 and maintenance_delta is not None:
        implied_margin = (first_fcf - maintenance_delta) / latest_revenue
    return {
        "market_implied_annual_fcf_growth": float((low + high) / 2.0),
        "target_enterprise_value": float(target_enterprise_value),
        "market_implied_terminal_growth": _solve_implied_terminal_growth(
            target_enterprise_value,
            first_fcf,
            quarterly_wacc,
            horizon_quarters,
        ),
        "market_implied_maintenance_capex_delta": maintenance_delta,
        "market_implied_margin": implied_margin if implied_margin is not None else latest_margin,
        "latest_maintenance_capex": latest_maintenance_capex,
    }


def _solve_implied_maintenance_delta(
    target_enterprise_value: float,
    first_fcf: float,
    model_annual_growth: float,
    quarterly_wacc: float,
    quarterly_terminal_growth: float,
    horizon_quarters: int,
) -> float | None:
    quarterly_growth = (1.0 + float(np.clip(model_annual_growth, -0.50, 0.50))) ** 0.25 - 1.0

    def value_with_delta(delta: float) -> float:
        path = []
        fcf = first_fcf
        for _ in range(horizon_quarters):
            fcf *= 1.0 + quarterly_growth
            path.append(max(fcf - delta, 1.0))
        tv = terminal_value(
            _stabilized_terminal_fcf(path) * 4.0,
            quarterly_wacc * 4.0,
            quarterly_terminal_growth * 4.0,
        )
        return present_value_series(path, quarterly_wacc) + present_value(tv, quarterly_wacc, len(path))

    low, high = -abs(first_fcf) * 5.0, abs(first_fcf) * 5.0
    try:
        for _ in range(80):
            mid = (low + high) / 2.0
            if value_with_delta(mid) < target_enterprise_value:
                high = mid
            else:
                low = mid
        return float((low + high) / 2.0)
    except Exception:
        return None


def _solve_implied_terminal_growth(
    target_enterprise_value: float,
    first_fcf: float,
    quarterly_wacc: float,
    horizon_quarters: int,
) -> float | None:
    path = [first_fcf] * horizon_quarters
    low, high = 0.0, max(quarterly_wacc * 4.0 - 0.01, 0.001)
    try:
        for _ in range(80):
            mid = (low + high) / 2.0
            tv = terminal_value(first_fcf * 4.0, quarterly_wacc * 4.0, mid)
            ev = present_value_series(path, quarterly_wacc) + present_value(tv, quarterly_wacc, horizon_quarters)
            if ev < target_enterprise_value:
                low = mid
            else:
                high = mid
        return float((low + high) / 2.0)
    except Exception:
        return None


def _valuation_confidence_score(
    warnings: list[str],
    forecast: GrowthForecast,
    warning_flags: list[dict[str, float | str]],
    settings: Settings,
) -> dict[str, float | str]:
    score = 100.0
    score -= min(len(warnings) * 2.0, 30.0)
    if forecast.clamp_triggered:
        score -= 15.0
    if forecast.model_type != "multivariate_dlm":
        score -= 10.0
    if warning_flags:
        score -= 15.0
    if abs(forecast.raw_fcf_growth_rate - forecast.smoothed_fcf_growth_rate) > 0.25:
        score -= 10.0
    complexity_penalty = {
        "bank": 35.0,
        "insurance": 35.0,
        "marketplace_hybrid": 25.0,
        "hyperscaler": 15.0,
        "platform": 12.0,
        "semiconductor": 15.0,
    }.get(settings.company_type, 5.0)
    score -= complexity_penalty
    score = float(np.clip(score, 0.0, 100.0))
    if score >= 75:
        bucket = "high"
    elif score >= 50:
        bucket = "medium"
    else:
        bucket = "low"
    return {"score": score, "bucket": bucket}


def _maintenance_capex_surface(
    forecast: GrowthForecast | None,
    base_wacc: float,
    base_terminal_growth: float,
    net_debt: float,
    shares: float,
) -> list[dict[str, object]]:
    if forecast is None:
        return []
    rows = forecast.diagnostics.get("forecast_components") or []
    if not rows:
        return []
    annual_wacc_values = [base_wacc * 4.0 - 0.01, base_wacc * 4.0, base_wacc * 4.0 + 0.01]
    annual_growth_values = [max(base_terminal_growth * 4.0 - 0.005, 0.0), base_terminal_growth * 4.0]
    multipliers = [0.9, 1.0, 1.1, 1.25]
    surface = []
    for multiplier in multipliers:
        path = []
        for row in rows:
            reported_fcf = float(row.get("reported_fcf", 0.0))
            growth_capex = float(row.get("growth_capex", 0.0))
            maintenance_delta = float(row.get("maintenance_capex", 0.0)) * (multiplier - 1.0)
            path.append(reported_fcf + growth_capex - maintenance_delta)
        multiplier_rows = []
        for annual_wacc in annual_wacc_values:
            values = []
            for annual_growth in annual_growth_values:
                try:
                    values.append(
                        _value_from_path(
                            path,
                            annual_wacc / 4.0,
                            annual_growth / 4.0,
                            net_debt,
                            shares,
                        )
                    )
                except Exception:
                    values.append(None)
            multiplier_rows.append({"annual_wacc": annual_wacc, "values": values})
        surface.append(
            {
                "maintenance_capex_multiplier": multiplier,
                "annual_terminal_growth_values": annual_growth_values,
                "rows": multiplier_rows,
            }
        )
    return surface
