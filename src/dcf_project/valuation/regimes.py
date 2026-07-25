from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from dcf_project.config import Settings
from dcf_project.contracts import DCFResult


REGIME_TICKER_DEFAULTS = {
    "AAPL": "mature_compounder",
    "MSFT": "mature_compounder",
    "GOOGL": "platform_growth",
    "GOOG": "platform_growth",
    "META": "platform_growth",
    "AMZN": "optionality_hypergrowth",
    "TSLA": "optionality_hypergrowth",
    "COST": "consumer_defensive",
    "KO": "consumer_defensive",
    "JNJ": "consumer_defensive",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    revenue_cagr: float
    operating_margin: float
    capex_intensity: float
    energy_contribution: float
    software_contribution: float
    terminal_growth: float
    fcf_conversion: float


def classify_valuation_regime(ticker: str, settings: Settings) -> str:
    if settings.valuation_regime != "auto":
        return settings.valuation_regime
    mapping = dict(REGIME_TICKER_DEFAULTS)
    if settings.valuation_regime_mapping_json:
        try:
            parsed = json.loads(settings.valuation_regime_mapping_json)
            if isinstance(parsed, dict):
                mapping.update({str(key).upper(): str(value) for key, value in parsed.items()})
        except json.JSONDecodeError:
            pass
    return mapping.get(ticker.upper(), _regime_from_company_type(settings.company_type))


def apply_valuation_regime(result: DCFResult, settings: Settings) -> DCFResult:
    regime = classify_valuation_regime(result.ticker, settings)
    if regime != "optionality_hypergrowth":
        diagnostics = dict(result.diagnostics)
        diagnostics["valuation_regime"] = regime
        diagnostics["regime_primary_engine"] = _engine_for_regime(regime)
        diagnostics.update(_mature_compounder_audit(result))
        return result.model_copy(update={"valuation_regime": regime, "diagnostics": diagnostics})
    return _optionality_hypergrowth_result(result, regime, settings)


def _optionality_hypergrowth_result(result: DCFResult, regime: str, settings: Settings) -> DCFResult:
    latest_revenue_trace = _latest_revenue_trace(result)
    latest_revenue = float(latest_revenue_trace["annual_revenue"])
    shares = max(result.diluted_shares_outstanding, 1.0)
    scenarios, generation_trace = _generate_scenarios(result, settings)
    decompositions = {
        scenario.name: _scenario_decomposition(result, scenario, latest_revenue, shares)
        for scenario in scenarios
    }
    raw_values = {name: details["intrinsic_value"] for name, details in decompositions.items()}
    weights, probability_method = _scenario_probabilities(result, scenarios, raw_values, settings)
    probability_trace = _probability_trace(result, scenarios, raw_values, weights, probability_method)
    scenario_details = {
        scenario.name: {
            **decompositions[scenario.name],
            "probability": weights[scenario.name],
            "revenue_cagr": scenario.revenue_cagr,
            "operating_margin": scenario.operating_margin,
            "capex_intensity": scenario.capex_intensity,
            "energy_contribution": scenario.energy_contribution,
            "software_contribution": scenario.software_contribution,
            "terminal_growth": scenario.terminal_growth,
            "fcf_conversion": scenario.fcf_conversion,
        }
        for scenario in scenarios
    }
    scenario_intrinsic = sum(raw_values[name] * weights[name] for name in raw_values)
    scenario_enterprise_value = scenario_intrinsic * shares + result.net_debt
    scenario_equity_value = scenario_intrinsic * shares
    scenario_implied_upside = (
        (float(scenario_intrinsic) / result.market_price - 1.0) if result.market_price else None
    )
    reverse = _reverse_dcf_expectations(result, latest_revenue, shares, scenarios)
    distribution = _monte_carlo_scenarios(result, latest_revenue, shares, scenarios, settings)
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "valuation_regime": regime,
            "regime_primary_engine": "dynamic_scenario_valuation_reverse_dcf_expectation_engine",
            "valuation_label": "speculative scenario valuation, not intrinsic DCF",
            # The canonical intrinsic value remains the disciplined DCF. The
            # optionality scenario value is exposed separately so it never
            # silently replaces intrinsic_value_per_share (which corrupted
            # cross-company comparison and predictive power).
            "canonical_intrinsic_source": "state_space_dcf",
            "state_space_dcf_diagnostic_intrinsic": result.intrinsic_value_per_share,
            "scenario_intrinsic_value": float(scenario_intrinsic),
            "scenario_enterprise_value": float(scenario_enterprise_value),
            "scenario_equity_value": float(scenario_equity_value),
            "scenario_implied_upside": scenario_implied_upside,
            "scenario_generation_method": generation_trace["method"],
            "scenario_generation_trace": generation_trace,
            "scenario_revenue_base_trace": latest_revenue_trace,
            "scenario_values": raw_values,
            "scenario_details": scenario_details,
            "probability_weights": weights,
            "probability_method": probability_method,
            "probability_trace": probability_trace,
            "reverse_dcf_expectations": reverse,
            "monte_carlo_distribution": distribution,
        }
    )
    # Canonical valuation fields (intrinsic/EV/equity/upside) are left on the DCF
    # result. Only scenario diagnostics and the regime label are attached.
    return result.model_copy(
        update={
            "valuation_regime": regime,
            "scenario_values": raw_values,
            "probability_weights": weights,
            "reverse_dcf_expectations": reverse,
            "scenario_generation_method": generation_trace["method"],
            "probability_method": probability_method,
            "monte_carlo_distribution": distribution,
            "diagnostics": diagnostics,
        }
    )


def _generate_scenarios(result: DCFResult, settings: Settings) -> tuple[list[Scenario], dict[str, object]]:
    archetype = result.company_archetype or "optionality_hypergrowth"
    consensus_growth = _float_diag(result, "consensus_revenue_growth_annual")
    model_growth = _annualized_forecast_growth(result)
    base_growth = consensus_growth if consensus_growth is not None else model_growth
    volatility = max(_historical_growth_volatility(result), 0.04)
    base_margin = _forecast_margin(result)
    capex_intensity = _forecast_capex_intensity(result)
    terminal_growth = min(max(result.terminal_growth_rate or 0.0275, 0.02), result.wacc - 0.015)
    optionality = _optionality_assumptions(result.ticker, archetype)
    bounds = _scenario_bounds(result.ticker, archetype, result.wacc)
    scenarios = [
        Scenario(
            "bear",
            0.0,
            float(np.clip(base_growth - volatility, bounds["min_revenue_cagr"], bounds["max_revenue_cagr"])),
            float(np.clip(base_margin - volatility * 0.35, bounds["min_margin"], bounds["max_margin"])),
            float(np.clip(capex_intensity + volatility * 0.20, bounds["min_capex_intensity"], bounds["max_capex_intensity"])),
            optionality["energy"] * 0.35,
            optionality["software"] * 0.20,
            float(np.clip(terminal_growth - 0.0075, bounds["min_terminal_growth"], bounds["max_terminal_growth"])),
            0.42,
        ),
        Scenario(
            "base",
            0.0,
            float(np.clip(base_growth, bounds["min_revenue_cagr"], bounds["max_revenue_cagr"])),
            float(np.clip(base_margin, bounds["min_margin"], bounds["max_margin"])),
            float(np.clip(capex_intensity, bounds["min_capex_intensity"], bounds["max_capex_intensity"])),
            optionality["energy"],
            optionality["software"],
            float(np.clip(terminal_growth, bounds["min_terminal_growth"], bounds["max_terminal_growth"])),
            0.55,
        ),
        Scenario(
            "bull",
            0.0,
            float(np.clip(base_growth + volatility * 1.25, bounds["min_revenue_cagr"], bounds["max_revenue_cagr"])),
            float(np.clip(base_margin + volatility * 0.45 + optionality["margin"], bounds["min_margin"], bounds["max_margin"])),
            float(np.clip(capex_intensity - volatility * 0.10, bounds["min_capex_intensity"], bounds["max_capex_intensity"])),
            min(optionality["energy"] * 1.8, bounds["max_optionality_contribution"]),
            min(optionality["software"] * 2.2, bounds["max_optionality_contribution"]),
            float(np.clip(terminal_growth + 0.005, bounds["min_terminal_growth"], bounds["max_terminal_growth"])),
            0.68,
        ),
    ]
    return scenarios, {
        "method": "ticker_aware_dynamic_scenarios_from_consensus_history_macro_archetype",
        "archetype": archetype,
        "consensus_revenue_growth_annual": consensus_growth,
        "model_revenue_growth_annual": model_growth,
        "base_revenue_cagr": base_growth,
        "historical_growth_volatility": volatility,
        "base_margin": base_margin,
        "capex_intensity": capex_intensity,
        "terminal_growth_anchor": terminal_growth,
        "optionality_assumptions": optionality,
        "sanity_bounds": bounds,
    }


def _scenario_probabilities(
    result: DCFResult,
    scenarios: list[Scenario],
    values: dict[str, float],
    settings: Settings,
) -> tuple[dict[str, float], str]:
    manual = _manual_probability_weights(settings)
    if manual:
        return manual, "manual"
    if settings.scenario_probability_method == "market_implied" and result.market_price:
        return _market_distance_weights(values, result.market_price), "market_implied"
    volatility = _historical_growth_volatility(result)
    base_bias = float(np.clip(0.55 - volatility, 0.35, 0.65))
    bull_bias = float(np.clip(0.25 + max(_annualized_forecast_growth(result), 0.0) * 0.50, 0.15, 0.40))
    bear_bias = max(1.0 - base_bias - bull_bias, 0.10)
    prior_weights = _normalize_weights({"bear": bear_bias, "base": base_bias, "bull": bull_bias})
    if result.market_price:
        likelihood_weights = _market_distance_weights(values, result.market_price)
        posterior = {
            name: (prior_weights.get(name, 0.0) * likelihood_weights.get(name, 0.0)) ** 0.5
            for name in ("bear", "base", "bull")
        }
        return _normalize_weights(posterior), "heuristic_market_adjusted_bayesian_style"
    return prior_weights, "heuristic_bayesian_style"


def _monte_carlo_scenarios(
    result: DCFResult,
    latest_revenue: float,
    shares: float,
    scenarios: list[Scenario],
    settings: Settings,
) -> dict[str, object]:
    samples = max(settings.scenario_monte_carlo_samples, 0)
    if samples <= 0:
        return {
            "simulation_count": 0,
            "p10": None,
            "p50": None,
            "p90": None,
            "distribution": [],
            "distribution_assumptions": {},
            "parameter_ranges": {},
            "sample_correlations": {},
        }
    rng = np.random.default_rng(settings.random_seed)
    base = next(s for s in scenarios if s.name == "base")
    growth_sigma = max(_historical_growth_volatility(result), 0.05)
    margin_sigma = max(growth_sigma * 0.35, 0.02)
    values = []
    sampled_growth = []
    sampled_margin = []
    sampled_terminal = []
    sampled_conversion = []
    for _ in range(samples):
        growth = float(np.clip(rng.normal(base.revenue_cagr, growth_sigma), -0.10, 0.60))
        margin = float(np.clip(rng.normal(base.operating_margin, margin_sigma), 0.01, 0.45))
        terminal_growth = float(np.clip(rng.normal(base.terminal_growth, 0.004), 0.015, result.wacc - 0.015))
        fcf_conversion = float(np.clip(rng.normal(base.fcf_conversion, 0.08), 0.25, 0.80))
        scenario = Scenario(
            "sample",
            1.0,
            growth,
            margin,
            float(np.clip(rng.normal(base.capex_intensity, 0.015), 0.01, 0.22)),
            float(max(rng.normal(base.energy_contribution, 0.02), 0.0)),
            float(max(rng.normal(base.software_contribution, 0.03), 0.0)),
            terminal_growth,
            fcf_conversion,
        )
        values.append(_scenario_value_per_share(result, scenario, latest_revenue, shares))
        sampled_growth.append(growth)
        sampled_margin.append(margin)
        sampled_terminal.append(terminal_growth)
        sampled_conversion.append(fcf_conversion)
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {
        "simulation_count": samples,
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "distribution": [float(x) for x in values[:200]],
        "distribution_assumptions": {
            "growth_distribution": "normal_clipped",
            "margin_distribution": "normal_clipped",
            "terminal_growth_distribution": "normal_clipped",
            "fcf_conversion_distribution": "normal_clipped",
            "random_seed": settings.random_seed,
        },
        "parameter_ranges": {
            "revenue_cagr": [float(min(sampled_growth)), float(max(sampled_growth))],
            "operating_margin": [float(min(sampled_margin)), float(max(sampled_margin))],
            "terminal_growth": [float(min(sampled_terminal)), float(max(sampled_terminal))],
            "fcf_conversion": [float(min(sampled_conversion)), float(max(sampled_conversion))],
        },
        "sample_correlations": {
            "value_vs_revenue_cagr": _correlation(values, sampled_growth),
            "value_vs_operating_margin": _correlation(values, sampled_margin),
            "value_vs_terminal_growth": _correlation(values, sampled_terminal),
            "value_vs_fcf_conversion": _correlation(values, sampled_conversion),
        },
    }


def _scenario_value_per_share(result: DCFResult, scenario: Scenario, latest_revenue: float, shares: float) -> float:
    ev = _scenario_enterprise_value(result, scenario, latest_revenue)
    return float((ev - result.net_debt) / shares)


def _scenario_decomposition(
    result: DCFResult,
    scenario: Scenario,
    latest_revenue: float,
    shares: float,
) -> dict[str, object]:
    years = 10
    annual_wacc = result.wacc
    tax_rate = 0.18
    explicit_pv = 0.0
    revenue = latest_revenue
    final_fcf = 0.0
    path = []
    for year in range(1, years + 1):
        revenue *= 1.0 + scenario.revenue_cagr
        operating_income = revenue * scenario.operating_margin
        nopat = operating_income * (1.0 - tax_rate)
        strategic_contribution = revenue * (scenario.energy_contribution + scenario.software_contribution)
        reinvestment = revenue * scenario.capex_intensity * 0.35
        final_fcf = max(nopat * scenario.fcf_conversion + strategic_contribution - reinvestment, 0.0)
        invested_capital = max(revenue * scenario.capex_intensity * 2.5, 1.0)
        roic = nopat / invested_capital
        pv_fcf = final_fcf / ((1.0 + annual_wacc) ** year)
        explicit_pv += pv_fcf
        path.append(
            {
                "year": year,
                "revenue": float(revenue),
                "operating_margin": scenario.operating_margin,
                "fcf_margin": float(final_fcf / max(revenue, 1.0)),
                "capex_intensity": scenario.capex_intensity,
                "reinvestment": float(reinvestment),
                "roic": float(roic),
                "fcf": float(final_fcf),
                "pv_fcf": float(pv_fcf),
            }
        )
    terminal_growth = min(scenario.terminal_growth, annual_wacc - 0.015)
    terminal_multiple = (1.0 + terminal_growth) / max(annual_wacc - terminal_growth, 1e-6)
    terminal_value = final_fcf * terminal_multiple
    pv_terminal = terminal_value / ((1.0 + annual_wacc) ** years)
    enterprise_value = explicit_pv + pv_terminal
    equity_value = enterprise_value - result.net_debt
    return {
        "revenue_cagr": scenario.revenue_cagr,
        "revenue_path": [row["revenue"] for row in path],
        "operating_margin_path": [row["operating_margin"] for row in path],
        "fcf_margin": float(path[-1]["fcf_margin"]),
        "fcf_margin_path": [row["fcf_margin"] for row in path],
        "capex_intensity": scenario.capex_intensity,
        "reinvestment": float(path[-1]["reinvestment"]),
        "reinvestment_path": [row["reinvestment"] for row in path],
        "roic": float(path[-1]["roic"]),
        "roic_path": [row["roic"] for row in path],
        "terminal_growth": terminal_growth,
        "wacc": annual_wacc,
        "terminal_multiple": float(terminal_multiple),
        "enterprise_value": float(enterprise_value),
        "equity_value": float(equity_value),
        "intrinsic_value": float(equity_value / shares),
        "explicit_pv": float(explicit_pv),
        "terminal_value": float(terminal_value),
        "pv_terminal": float(pv_terminal),
        "annual_path": path,
    }


def _scenario_enterprise_value(result: DCFResult, scenario: Scenario, latest_revenue: float) -> float:
    years = 10
    annual_wacc = result.wacc
    tax_rate = 0.18
    explicit_pv = 0.0
    revenue = latest_revenue
    final_fcf = 0.0
    for year in range(1, years + 1):
        revenue *= 1.0 + scenario.revenue_cagr
        operating_income = revenue * scenario.operating_margin
        nopat = operating_income * (1.0 - tax_rate)
        strategic_contribution = revenue * (scenario.energy_contribution + scenario.software_contribution)
        reinvestment_drag = revenue * scenario.capex_intensity * 0.35
        final_fcf = max(nopat * scenario.fcf_conversion + strategic_contribution - reinvestment_drag, 0.0)
        explicit_pv += final_fcf / ((1.0 + annual_wacc) ** year)
    terminal_growth = min(scenario.terminal_growth, annual_wacc - 0.015)
    terminal_value = final_fcf * (1.0 + terminal_growth) / (annual_wacc - terminal_growth)
    return float(explicit_pv + terminal_value / ((1.0 + annual_wacc) ** years))


def _reverse_dcf_expectations(
    result: DCFResult,
    latest_revenue: float,
    shares: float,
    scenarios: list[Scenario],
) -> dict[str, float | None]:
    if not result.market_price or latest_revenue <= 0 or shares <= 0:
        return {"required_revenue_cagr_at_base_margin": None}
    target_ev = result.market_price * shares + result.net_debt
    base = next(s for s in scenarios if s.name == "base")
    required_growth = _solve_required_revenue_growth(result, latest_revenue, target_ev, base)
    required_margin = _solve_required_margin(result, latest_revenue, target_ev, base)
    required_fcf_margin = _solve_required_fcf_margin(result, latest_revenue, target_ev, base)
    implied_terminal_multiple = (1.0 + base.terminal_growth) / max(result.wacc - base.terminal_growth, 1e-6)
    required_roic = required_margin * (1.0 - 0.18) / max(base.capex_intensity * 2.5, 0.05)
    required_reinvestment = required_growth / max(required_roic, 0.01) if required_growth is not None else None
    return {
        "required_revenue_cagr_at_base_margin": required_growth,
        "required_operating_margin_at_base_growth": required_margin,
        "required_fcf_cagr": required_growth,
        "required_fcf_margin": required_fcf_margin,
        "required_roic": required_roic,
        "required_reinvestment": required_reinvestment,
        "implied_terminal_multiple": float(implied_terminal_multiple),
        "implied_bull_probability": _implied_bull_probability(result, scenarios),
        "target_enterprise_value": float(target_ev),
        "market_price": result.market_price,
    }


def _solve_required_revenue_growth(result: DCFResult, latest_revenue: float, target_ev: float, base: Scenario) -> float:
    low, high = -0.10, 0.70
    for _ in range(70):
        mid = (low + high) / 2.0
        scenario = Scenario("reverse", 1.0, mid, base.operating_margin, base.capex_intensity, base.energy_contribution, base.software_contribution, base.terminal_growth, base.fcf_conversion)
        if _scenario_enterprise_value(result, scenario, latest_revenue) < target_ev:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def _solve_required_margin(result: DCFResult, latest_revenue: float, target_ev: float, base: Scenario) -> float:
    low, high = 0.01, 0.50
    for _ in range(70):
        mid = (low + high) / 2.0
        scenario = Scenario("reverse", 1.0, base.revenue_cagr, mid, base.capex_intensity, base.energy_contribution, base.software_contribution, base.terminal_growth, base.fcf_conversion)
        if _scenario_enterprise_value(result, scenario, latest_revenue) < target_ev:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def _solve_required_fcf_margin(result: DCFResult, latest_revenue: float, target_ev: float, base: Scenario) -> float:
    terminal_revenue = latest_revenue * ((1.0 + base.revenue_cagr) ** 10)
    implied_terminal_fcf = target_ev * ((1.0 + result.wacc) ** 10) * (result.wacc - base.terminal_growth) / (1.0 + base.terminal_growth)
    return float(implied_terminal_fcf / max(terminal_revenue, 1.0))


def _latest_revenue(result: DCFResult) -> float:
    return float(_latest_revenue_trace(result)["annual_revenue"])


def _latest_revenue_trace(result: DCFResult) -> dict[str, object]:
    rows = result.diagnostics.get("historical_financials", [])
    quarterly_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("fiscal_period") != "FY"
        and isinstance(row.get("revenue"), (int, float))
        and row["revenue"] > 0
    ]
    if len(quarterly_rows) >= 4:
        trailing_rows = quarterly_rows[-4:]
        return {
            "annual_revenue": float(sum(float(row["revenue"]) for row in trailing_rows)),
            "method": "trailing_four_quarters",
            "periods_used": [str(row.get("date") or row.get("period") or "") for row in trailing_rows],
            "quarterly_revenues_used": [float(row["revenue"]) for row in trailing_rows],
        }
    for row in reversed(rows):
        revenue = row.get("revenue") if isinstance(row, dict) else None
        if isinstance(revenue, (int, float)) and revenue > 0:
            fiscal_period = row.get("fiscal_period") if isinstance(row, dict) else None
            annual_revenue = float(revenue if fiscal_period == "FY" else revenue * 4.0)
            return {
                "annual_revenue": annual_revenue,
                "method": "latest_fy" if fiscal_period == "FY" else "latest_quarter_annualized",
                "periods_used": [str(row.get("date") or row.get("period") or "")],
                "quarterly_revenues_used": [] if fiscal_period == "FY" else [float(revenue)],
            }
    forecast = result.forecast_financials
    if forecast:
        first = forecast[0].get("revenue")
        growth = forecast[0].get("revenue_growth", 0.0)
        if isinstance(first, (int, float)) and isinstance(growth, (int, float)):
            implied_latest_quarter = float(first / (1.0 + growth))
            return {
                "annual_revenue": implied_latest_quarter * 4.0,
                "method": "first_forecast_quarter_backsolved_annualized",
                "periods_used": [],
                "quarterly_revenues_used": [implied_latest_quarter],
            }
    return {"annual_revenue": 1.0, "method": "missing_revenue_fallback", "periods_used": [], "quarterly_revenues_used": []}


def _historical_growth_volatility(result: DCFResult) -> float:
    rows = result.diagnostics.get("historical_financials", [])
    revenues = [row.get("revenue") for row in rows if isinstance(row, dict) and isinstance(row.get("revenue"), (int, float))]
    if len(revenues) < 6:
        return 0.08
    growth = np.diff(np.log(np.maximum(np.asarray(revenues, dtype=float), 1.0)))
    return float(np.clip(np.nanstd(growth) * 2.0, 0.03, 0.25))


def _annualized_forecast_growth(result: DCFResult) -> float:
    rows = result.forecast_financials[:8]
    values = [row.get("revenue_growth") for row in rows if isinstance(row.get("revenue_growth"), (int, float))]
    if not values:
        return 0.08
    return float((1.0 + np.nanmedian(values)) ** 4 - 1.0)


def _forecast_margin(result: DCFResult) -> float:
    rows = result.forecast_financials[:8]
    values = [row.get("operating_margin") for row in rows if isinstance(row.get("operating_margin"), (int, float))]
    return float(np.clip(np.nanmedian(values) if values else 0.10, 0.02, 0.40))


def _forecast_capex_intensity(result: DCFResult) -> float:
    rows = result.forecast_financials[:8]
    values = []
    for row in rows:
        capex = row.get("capex")
        revenue = row.get("revenue")
        if isinstance(capex, (int, float)) and isinstance(revenue, (int, float)) and revenue > 0:
            values.append(capex / revenue)
    return float(np.clip(np.nanmedian(values) if values else 0.075, 0.02, 0.18))


def _float_diag(result: DCFResult, key: str) -> float | None:
    value = result.diagnostics.get(key)
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _optionality_assumptions(ticker: str, archetype: str) -> dict[str, float]:
    if ticker == "TSLA":
        return {"energy": 0.05, "software": 0.04, "margin": 0.025}
    if archetype == "amzn":
        return {"energy": 0.00, "software": 0.06, "margin": 0.020}
    return {"energy": 0.00, "software": 0.03, "margin": 0.015}


def _scenario_bounds(ticker: str, archetype: str, wacc: float) -> dict[str, float]:
    if ticker == "TSLA":
        return {
            "min_revenue_cagr": -0.05,
            "max_revenue_cagr": 0.35,
            "min_margin": 0.02,
            "max_margin": 0.24,
            "min_capex_intensity": 0.035,
            "max_capex_intensity": 0.16,
            "min_terminal_growth": 0.015,
            "max_terminal_growth": min(0.03, wacc - 0.015),
            "max_optionality_contribution": 0.12,
        }
    return {
        "min_revenue_cagr": -0.05,
        "max_revenue_cagr": 0.45,
        "min_margin": 0.02,
        "max_margin": 0.35,
        "min_capex_intensity": 0.015,
        "max_capex_intensity": 0.18,
        "min_terminal_growth": 0.015,
        "max_terminal_growth": min(0.035, wacc - 0.015),
        "max_optionality_contribution": 0.15 if archetype == "amzn" else 0.10,
    }


def _probability_trace(
    result: DCFResult,
    scenarios: list[Scenario],
    values: dict[str, float],
    weights: dict[str, float],
    method: str,
) -> dict[str, object]:
    volatility = _historical_growth_volatility(result)
    forecast_growth = _annualized_forecast_growth(result)
    if method in {"heuristic_bayesian_style", "heuristic_market_adjusted_bayesian_style"}:
        prior = {"bear": 0.25, "base": 0.50, "bull": 0.25}
        likelihood_inputs = {
            "historical_growth_volatility": volatility,
            "annualized_forecast_growth": forecast_growth,
            "market_price": result.market_price,
            "scenario_values": values,
            "market_distance_weights": _market_distance_weights(values, result.market_price)
            if result.market_price
            else None,
        }
        return {
            "label": method,
            "prior": prior,
            "likelihood_inputs": likelihood_inputs,
            "posterior_weights": weights,
        }
    return {
        "label": method,
        "prior": {scenario.name: scenario.probability for scenario in scenarios},
        "likelihood_inputs": {"market_price": result.market_price, "scenario_values": values},
        "posterior_weights": weights,
    }


def _implied_bull_probability(result: DCFResult, scenarios: list[Scenario]) -> float | None:
    if result.market_price is None:
        return None
    latest_revenue = _latest_revenue(result)
    shares = max(result.diluted_shares_outstanding, 1.0)
    values = {
        scenario.name: _scenario_value_per_share(result, scenario, latest_revenue, shares)
        for scenario in scenarios
    }
    bear = values.get("bear")
    base = values.get("base")
    bull = values.get("bull")
    if bear is None or base is None or bull is None or bull <= base:
        return None
    non_bull_value = 0.35 * bear + 0.65 * base
    probability = (result.market_price - non_bull_value) / max(bull - non_bull_value, 1e-9)
    return float(np.clip(probability, 0.0, 1.0))


def _correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(b) < 3:
        return None
    corr = np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1]
    return float(corr) if np.isfinite(corr) else None


def _manual_probability_weights(settings: Settings) -> dict[str, float] | None:
    if not settings.scenario_probability_weights_json:
        return None
    try:
        parsed = json.loads(settings.scenario_probability_weights_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    weights = {name: float(parsed.get(name, 0.0)) for name in ("bear", "base", "bull")}
    return _normalize_weights(weights) if sum(weights.values()) > 0 else None


def _market_distance_weights(values: dict[str, float], market_price: float) -> dict[str, float]:
    scores = {name: 1.0 / (abs(value - market_price) + 1.0) for name, value in values.items()}
    return _normalize_weights(scores)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        return {"bear": 1 / 3, "base": 1 / 3, "bull": 1 / 3}
    return {key: float(max(value, 0.0) / total) for key, value in weights.items()}


def _mature_compounder_audit(result: DCFResult) -> dict[str, object]:
    margins = [row.get("operating_margin") for row in result.forecast_financials if isinstance(row.get("operating_margin"), (int, float))]
    shares = [row.get("shares") for row in result.forecast_financials if isinstance(row.get("shares"), (int, float))]
    return {
        "mature_compounder_calibration_audit": {
            "margin_persistence_proxy": float(np.nanstd(margins)) if margins else None,
            "forecast_margin_median": float(np.nanmedian(margins)) if margins else None,
            "buyback_effect_total": float(1.0 - shares[-1] / shares[0]) if len(shares) >= 2 and shares[0] else None,
            "consensus_weight": result.consensus_weight,
            "consensus_provider": result.diagnostics.get("consensus_provider"),
        }
    }


def _regime_from_company_type(company_type: str) -> str:
    if company_type in {"platform", "hyperscaler"}:
        return "platform_growth"
    if company_type == "capital_intensive_growth":
        return "optionality_hypergrowth"
    if company_type in {"industrial", "semiconductor", "telecom"}:
        return "capital_intensive_cyclical"
    if company_type in {"consumer_staple", "consumer_compounder"}:
        return "consumer_defensive"
    return "mature_compounder"


def _engine_for_regime(regime: str) -> str:
    return {
        "mature_compounder": "dcf_consensus_owner_earnings",
        "platform_growth": "dcf_rd_capitalization_scenario_overlay",
        "capital_intensive_cyclical": "normalized_cycle_dcf",
        "consumer_defensive": "stable_dcf",
    }.get(regime, "dcf")
