from __future__ import annotations

import pytest

from dcf_project.config import Settings
from dcf_project.contracts import DCFResult, SourceMeta
from dcf_project.valuation.regimes import (
    Scenario,
    _scenario_probabilities,
    apply_valuation_regime,
    classify_valuation_regime,
)


def test_tsla_routes_to_optionality_hypergrowth() -> None:
    assert classify_valuation_regime("TSLA", Settings()) == "optionality_hypergrowth"


def test_optionality_hypergrowth_keeps_dcf_canonical_and_exposes_scenario_separately() -> None:
    result = _result("TSLA")

    routed = apply_valuation_regime(result, Settings())

    assert routed.valuation_regime == "optionality_hypergrowth"
    # Canonical intrinsic value / EV / equity / upside remain the disciplined DCF.
    assert routed.intrinsic_value_per_share == result.intrinsic_value_per_share
    assert routed.enterprise_value == result.enterprise_value
    assert routed.equity_value == result.equity_value
    assert routed.implied_upside == result.implied_upside
    assert routed.diagnostics["canonical_intrinsic_source"] == "state_space_dcf"
    assert routed.diagnostics["state_space_dcf_diagnostic_intrinsic"] == result.intrinsic_value_per_share
    # The optionality scenario value is exposed separately, never replacing intrinsic.
    assert "scenario_intrinsic_value" in routed.diagnostics
    assert routed.diagnostics["scenario_intrinsic_value"] != result.intrinsic_value_per_share
    assert routed.diagnostics["scenario_enterprise_value"] is not None
    assert routed.diagnostics["scenario_equity_value"] is not None
    # Scenario diagnostics are still fully populated.
    assert routed.scenario_values
    assert set(routed.probability_weights) == {"bear", "base", "bull"}
    assert sum(routed.probability_weights.values()) == pytest.approx(1.0)
    assert routed.reverse_dcf_expectations["required_revenue_cagr_at_base_margin"] is not None
    assert routed.reverse_dcf_expectations["required_fcf_margin"] is not None
    assert routed.reverse_dcf_expectations["implied_terminal_multiple"] is not None
    assert routed.scenario_generation_method == "ticker_aware_dynamic_scenarios_from_consensus_history_macro_archetype"
    assert routed.probability_method == "heuristic_market_adjusted_bayesian_style"
    assert routed.monte_carlo_distribution["p50"] is not None
    assert routed.monte_carlo_distribution["distribution_assumptions"]
    assert routed.monte_carlo_distribution["parameter_ranges"]
    assert routed.monte_carlo_distribution["sample_correlations"]
    assert routed.reverse_dcf_expectations["implied_bull_probability"] is not None
    assert routed.diagnostics["valuation_label"] == "speculative scenario valuation, not intrinsic DCF"
    assert routed.diagnostics["scenario_revenue_base_trace"]["method"] == "trailing_four_quarters"
    assert routed.diagnostics["scenario_revenue_base_trace"]["annual_revenue"] == pytest.approx(356_000_000_000.0)
    assert routed.diagnostics["probability_trace"]["posterior_weights"] == routed.probability_weights
    assert routed.diagnostics["scenario_details"]["base"]["revenue_path"]
    assert routed.diagnostics["scenario_details"]["base"]["revenue_path"][0] > 356_000_000_000.0
    assert routed.diagnostics["scenario_details"]["base"]["terminal_multiple"] > 0


def test_default_scenario_probabilities_use_market_likelihood() -> None:
    result = _result("AMZN")
    values = {"bear": 0.0, "base": 260.0, "bull": 7_500.0}
    scenarios = [
        Scenario("bear", 0.0, 0.0, 0.05, 0.10, 0.0, 0.0, 0.025, 0.40),
        Scenario("base", 0.0, 0.08, 0.10, 0.08, 0.0, 0.02, 0.030, 0.55),
        Scenario("bull", 0.0, 0.20, 0.20, 0.05, 0.0, 0.10, 0.035, 0.68),
    ]

    weights, method = _scenario_probabilities(result, scenarios, values, Settings())

    assert method == "heuristic_market_adjusted_bayesian_style"
    assert weights["base"] > weights["bull"]
    assert weights["bull"] < 0.10


def test_mature_compounder_keeps_dcf_primary() -> None:
    result = _result("AAPL")

    routed = apply_valuation_regime(result, Settings())

    assert routed.valuation_regime == "mature_compounder"
    assert routed.intrinsic_value_per_share == result.intrinsic_value_per_share
    assert routed.scenario_values == {}


def _result(ticker: str) -> DCFResult:
    return DCFResult(
        ticker=ticker,
        explicit_forecast_quarters=20,
        intrinsic_value_per_share=40.0,
        enterprise_value=150_000_000_000.0,
        equity_value=140_000_000_000.0,
        net_debt=10_000_000_000.0,
        projected_fcf=[1_000_000_000.0] * 20,
        explicit_pv=10_000_000_000.0,
        pv_projected_fcf=10_000_000_000.0,
        terminal_value=140_000_000_000.0,
        pv_terminal_value=90_000_000_000.0,
        fcf_terminal_base=1_000_000_000.0,
        wacc=0.08,
        terminal_growth_rate=0.0275,
        fcf_growth_rate=0.10,
        raw_fcf_growth_rate=0.10,
        smoothed_fcf_growth_rate=0.10,
        clamped_fcf_growth_rate=0.10,
        growth_clamped=False,
        clamp_triggered=False,
        valuation_p10=30.0,
        valuation_p50=40.0,
        valuation_p90=60.0,
        diluted_shares_outstanding=3_500_000_000.0,
        market_price=426.0,
        forecast_financials=[
            {
                "quarter": float(idx),
                "revenue": 25_000_000_000.0 * (1.03**idx),
                "revenue_growth": 0.03,
                "operating_margin": 0.10,
                "capex": 2_000_000_000.0,
            }
            for idx in range(1, 21)
        ],
        diagnostics={
            "historical_financials": [
                {"revenue": 80_000_000_000.0},
                {"revenue": 86_000_000_000.0},
                {"revenue": 90_000_000_000.0},
                {"revenue": 100_000_000_000.0},
            ],
            "consensus_revenue_growth_annual": 0.14,
        },
        sources=[SourceMeta(provider="fixture", key=ticker)],
    )
