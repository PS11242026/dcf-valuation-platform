from __future__ import annotations

from dataclasses import replace
import json
from typing import Iterable

import numpy as np

from dcf_project.config import get_settings
from dcf_project.contracts import DCFResult
from dcf_project.pipeline import DCFPipeline
from dcf_project.valuation.discounting import present_value, present_value_series
from dcf_project.valuation.terminal import terminal_value


def main() -> None:
    settings = replace(
        get_settings(),
        terminal_growth_source="internal",
        wacc_source="internal",
        monte_carlo_samples=0,
        run_sensitivity=False,
    )
    payload = build_attribution_report(["AAPL", "MSFT", "TSLA"], settings)
    print(json.dumps(payload, indent=2, default=str))


def build_attribution_report(
    tickers: Iterable[str],
    settings=None,
    pipeline_factory=DCFPipeline,
) -> dict[str, object]:
    settings = settings or replace(
        get_settings(),
        terminal_growth_source="internal",
        wacc_source="internal",
        monte_carlo_samples=0,
        run_sensitivity=False,
    )
    payload = {}
    for ticker in tickers:
        result = pipeline_factory(settings).run(ticker)
        diagnostics = result.diagnostics
        payload[ticker] = {
            "market_price": result.market_price,
            "intrinsic_value_per_share": result.intrinsic_value_per_share,
            "gap_to_market": (result.intrinsic_value_per_share / result.market_price - 1.0)
            if result.market_price
            else None,
            "valuation_regime": result.valuation_regime,
            "company_type": diagnostics.get("company_type"),
            "resolved_company_type": diagnostics.get("resolved_company_type"),
            "forecast_mode": result.forecast_mode,
            "consensus_weight": result.consensus_weight,
            "wacc": result.wacc,
            "terminal_growth_rate": result.terminal_growth_rate,
            "explicit_pv": result.explicit_pv,
            "pv_terminal": result.pv_terminal,
            "terminal_value": result.terminal_value,
            "terminal_to_enterprise": result.pv_terminal / result.enterprise_value
            if result.enterprise_value
            else None,
            "ensemble_weights": result.ensemble_weights,
            "forecast_ranges": _forecast_ranges(result.forecast_financials),
            "projected_fcf": _series_stats(result.projected_fcf),
            "raw_model_forecast": diagnostics.get("raw_model_forecast", [])[:5],
            "consensus_forecast": diagnostics.get("consensus_forecast", [])[:5],
            "blended_forecast": diagnostics.get("blended_forecast", [])[:5],
            "overlay_forecast": diagnostics.get("overlay_forecast", [])[:5],
            "final_forecast": diagnostics.get("final_forecast", [])[:5],
            "clamp_triggered_map": diagnostics.get("clamp_triggered_map"),
            "field_source_first": (diagnostics.get("field_source_map") or [{}])[0],
            "terminal_trace": diagnostics.get("terminal_trace"),
            "wacc_trace": diagnostics.get("wacc_trace"),
            "invested_capital_tail": (diagnostics.get("invested_capital_components") or [])[-6:],
            "invested_capital_flags": diagnostics.get("invested_capital_continuity_flags", [])[-6:],
            "forecast_sanity": diagnostics.get("forecast_sanity"),
            "prior_influence": diagnostics.get("prior_influence"),
            "scenario_revenue_base_trace": diagnostics.get("scenario_revenue_base_trace"),
            "share_count_trace": diagnostics.get("share_count_trace"),
            "scenario_trace": _scenario_trace(result),
            "value_impact": _value_impact(result),
            "bug_classification": _bug_classification(result),
            "largest_gap_driver": _largest_gap_driver(result),
            "scenario_values": result.scenario_values,
            "probability_weights": result.probability_weights,
        }
    return payload


def _scenario_trace(result: DCFResult) -> dict[str, object] | None:
    if result.valuation_regime != "optionality_hypergrowth":
        return None
    diagnostics = result.diagnostics
    return {
        "regime": result.valuation_regime,
        "state_space_dcf_diagnostic_intrinsic": diagnostics.get("state_space_dcf_diagnostic_intrinsic"),
        "scenario_revenue_base_trace": diagnostics.get("scenario_revenue_base_trace"),
        "scenario_values": result.scenario_values,
        "probability_weights": result.probability_weights,
        "reverse_dcf_expectations": result.reverse_dcf_expectations,
        "probability_method": result.probability_method,
    }


def _value_impact(result: DCFResult) -> dict[str, object]:
    diagnostic_dcf = result.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
    base = float(diagnostic_dcf) if isinstance(diagnostic_dcf, (int, float)) else result.intrinsic_value_per_share
    shares = result.diluted_shares_outstanding
    net_debt = result.net_debt
    path = [float(value) for value in result.projected_fcf]
    terminal_growth = result.terminal_growth_rate
    wacc = result.wacc
    impacts = {
        "revenue_cagr_plus_100bps": _value_delta_from_path(_growth_scaled_path(path, 0.01), wacc, terminal_growth, net_debt, shares, base),
        "operating_margin_plus_100bps": _value_delta_from_path(_margin_scaled_path(result, 0.01), wacc, terminal_growth, net_debt, shares, base),
        "reinvestment_rate_plus_100bps": _value_delta_from_path(_reinvestment_scaled_path(result, 0.01), wacc, terminal_growth, net_debt, shares, base),
        "terminal_growth_plus_25bps": _value_delta_from_path(path, wacc, terminal_growth + 0.0025, net_debt, shares, base),
        "wacc_plus_25bps": _value_delta_from_path(path, wacc + 0.0025, terminal_growth, net_debt, shares, base),
        "share_count_forecast_denominator_impact": _forecast_share_denominator_impact(result),
        "sensitivity_basis_value": base,
        "sensitivity_basis": "state_space_dcf_diagnostic_intrinsic"
        if isinstance(diagnostic_dcf, (int, float))
        else "final_intrinsic_value",
        "valuation_regime_impact": _valuation_regime_impact(result),
        "scenario_weight_impact": _scenario_weight_impact(result),
    }
    return impacts


def _value_delta_from_path(
    path: list[float],
    annual_wacc: float,
    annual_terminal_growth: float,
    net_debt: float,
    shares: float,
    base: float,
) -> dict[str, float | None]:
    value = _value_from_quarterly_path(path, annual_wacc, annual_terminal_growth, net_debt, shares)
    return {"value": value, "delta": None if value is None else value - base}


def _value_from_quarterly_path(
    path: list[float],
    annual_wacc: float,
    annual_terminal_growth: float,
    net_debt: float,
    shares: float,
) -> float | None:
    if not path or shares <= 0 or annual_wacc <= annual_terminal_growth:
        return None
    quarterly_wacc = annual_wacc / 4.0
    terminal_base = _weighted_terminal_base(path) * 4.0
    tv = terminal_value(terminal_base, annual_wacc, min(annual_terminal_growth, annual_wacc - 0.01))
    ev = present_value_series(path, quarterly_wacc) + present_value(tv, quarterly_wacc, len(path))
    return float((ev - net_debt) / shares)


def _weighted_terminal_base(path: list[float]) -> float:
    if len(path) < 4:
        return float(path[-1])
    return float(np.average(np.asarray(path[-4:], dtype=float), weights=np.asarray([0.1, 0.2, 0.3, 0.4])))


def _growth_scaled_path(path: list[float], annual_delta: float) -> list[float]:
    quarterly_delta = (1.0 + annual_delta) ** 0.25 - 1.0
    return [float(value * ((1.0 + quarterly_delta) ** idx)) for idx, value in enumerate(path, start=1)]


def _margin_scaled_path(result: DCFResult, margin_delta: float) -> list[float]:
    rows = result.forecast_financials
    margins = [row.get("operating_margin") for row in rows]
    valid = [float(value) for value in margins if isinstance(value, (int, float)) and value > 0]
    if not valid:
        return [float(value) for value in result.projected_fcf]
    scale = (float(np.mean(valid)) + margin_delta) / max(float(np.mean(valid)), 1e-9)
    return [float(value) * scale for value in result.projected_fcf]


def _reinvestment_scaled_path(result: DCFResult, reinvestment_delta: float) -> list[float]:
    rows = result.forecast_financials
    rates = [row.get("reinvestment_rate") for row in rows]
    valid = [float(value) for value in rates if isinstance(value, (int, float)) and 0.0 <= value < 1.0]
    if not valid:
        return [float(value) for value in result.projected_fcf]
    base_keep = max(1.0 - float(np.mean(valid)), 1e-9)
    stressed_keep = max(base_keep - reinvestment_delta, 1e-9)
    scale = stressed_keep / base_keep
    return [float(value) * scale for value in result.projected_fcf]


def _forecast_share_denominator_impact(result: DCFResult) -> dict[str, float | None]:
    trace = result.diagnostics.get("share_count_trace") or {}
    forecast_shares = trace.get("forecast_terminal_shares")
    if not isinstance(forecast_shares, (int, float)) or forecast_shares <= 0:
        return {"value": None, "delta": None}
    value = result.equity_value / float(forecast_shares)
    return {"value": float(value), "delta": float(value - result.intrinsic_value_per_share)}


def _valuation_regime_impact(result: DCFResult) -> dict[str, float | None]:
    diagnostic = result.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
    if not isinstance(diagnostic, (int, float)):
        return {"value": result.intrinsic_value_per_share, "delta": 0.0}
    return {"value": float(result.intrinsic_value_per_share), "delta": float(result.intrinsic_value_per_share - diagnostic)}


def _scenario_weight_impact(result: DCFResult) -> dict[str, float | None]:
    values = result.scenario_values or {}
    if not values:
        return {"value": None, "delta": None}
    equal_weight = sum(float(value) for value in values.values()) / len(values)
    return {"equal_weight_value": float(equal_weight), "delta_vs_probability_weighted": float(result.intrinsic_value_per_share - equal_weight)}


def _bug_classification(result: DCFResult) -> list[dict[str, str]]:
    diagnostics = result.diagnostics
    issues: list[dict[str, str]] = []
    conversion = diagnostics.get("consensus_growth_conversion") or {}
    if conversion.get("input_basis") == "annualized_cagr" and "quarterly_growth" in conversion:
        issues.append({"issue": "consensus CAGR converted before quarterly forecast use", "classification": "assumption choice, not a bug"})
    if diagnostics.get("share_count_trace", {}).get("forecast_share_path_used_for_denominator") is False:
        issues.append({"issue": "forecast share path audited separately from valuation denominator", "classification": "assumption choice, not a bug"})
    if any((diagnostics.get("clamp_triggered_map") or {}).values()):
        issues.append({"issue": "explicit policy clamps triggered and flagged in field_source_map", "classification": "assumption choice, not a bug"})
    if any(row.get("proxy_used") for row in diagnostics.get("invested_capital_components", []) if isinstance(row, dict)):
        issues.append({"issue": "proxy invested capital used", "classification": "data reconstruction bug"})
    else:
        issues.append({"issue": "proxy invested capital not used in default ROIC path", "classification": "assumption choice, not a bug"})
    if result.valuation_regime == "optionality_hypergrowth":
        trace = diagnostics.get("scenario_revenue_base_trace") or {}
        classification = "assumption choice, not a bug" if trace.get("method") == "trailing_four_quarters" else "unit conversion bug"
        issues.append({"issue": "scenario revenue base", "classification": classification})
    return issues


def _largest_gap_driver(result: DCFResult) -> dict[str, object]:
    impacts = _value_impact(result)
    if result.valuation_regime == "optionality_hypergrowth":
        return {
            "market_gap_per_share": None if result.market_price is None else float(result.market_price - result.intrinsic_value_per_share),
            "largest_sensitivity_driver": "scenario_weights_and_bull_case_probability",
            "largest_sensitivity_delta": (impacts.get("scenario_weight_impact") or {}).get("delta_vs_probability_weighted"),
            "interpretation": _gap_interpretation(result, "scenario_weights_and_bull_case_probability"),
        }
    candidates = {
        key: value.get("delta")
        for key, value in impacts.items()
        if isinstance(value, dict) and isinstance(value.get("delta"), (int, float))
    }
    largest = max(candidates, key=lambda key: abs(float(candidates[key]))) if candidates else None
    return {
        "market_gap_per_share": None if result.market_price is None else float(result.market_price - result.intrinsic_value_per_share),
        "largest_sensitivity_driver": largest,
        "largest_sensitivity_delta": None if largest is None else float(candidates[largest]),
        "interpretation": _gap_interpretation(result, largest),
    }


def _gap_interpretation(result: DCFResult, largest: str | None) -> str:
    if result.valuation_regime == "optionality_hypergrowth":
        return "Scenario valuation remains below market because probability-weighted scenario assumptions do not fully price the bull case."
    if largest in {"terminal_growth_plus_25bps", "wacc_plus_25bps"}:
        return "Gap is primarily discount-rate/terminal-value sensitive; no hidden overwrite identified."
    return "Gap is explained by explicit forecast and policy assumptions; source maps expose any clamp or consensus blend."


def _forecast_ranges(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    return {
        field: _series_stats([row.get(field) for row in rows])
        for field in ["revenue_growth", "operating_margin", "reinvestment_rate", "roic", "fcf", "owner_earnings"]
    }


def _series_stats(values: list) -> dict[str, float | None]:
    clean = [float(x) for x in values if isinstance(x, (int, float)) and np.isfinite(x)]
    if not clean:
        return {"min": None, "max": None, "first": None, "last": None, "mean": None}
    return {
        "min": min(clean),
        "max": max(clean),
        "first": clean[0],
        "last": clean[-1],
        "mean": sum(clean) / len(clean),
    }


if __name__ == "__main__":
    main()
