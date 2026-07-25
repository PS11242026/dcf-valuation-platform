from __future__ import annotations

from dcf_project.config import Settings
from dcf_project.contracts import GrowthForecast, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.valuation.dcf import build_dcf_result


def test_build_dcf_result_returns_intrinsic_value() -> None:
    source = SourceMeta(provider="test", key="fixture")
    forecast = GrowthForecast(
        ticker="TST",
        model_type="univariate_local_linear",
        single_growth_rate=0.02,
        raw_fcf_growth_rate=0.02,
        smoothed_fcf_growth_rate=0.02,
        clamped_fcf_growth_rate=0.02,
        growth_clamped=False,
        clamp_triggered=False,
        horizon_quarters=20,
        growth_path=[0.02] * 20,
        fcf_path=[100.0 * (1.02**i) for i in range(1, 21)],
        lower_growth_path=[0.0] * 20,
        upper_growth_path=[0.04] * 20,
    )
    market = MarketSnapshot(
        ticker="TST",
        price=10.0,
        market_cap=1000.0,
        shares_outstanding=100.0,
        beta=1.0,
        source=source,
    )
    wacc = WACCInputs(
        risk_free_rate=0.04,
        beta=1.0,
        equity_risk_premium=0.05,
        cost_of_equity=0.09,
        pre_tax_cost_of_debt=0.045,
        tax_rate=0.21,
        market_value_equity=1000.0,
        market_value_debt=100.0,
        equity_weight=1000.0 / 1100.0,
        debt_weight=100.0 / 1100.0,
        after_tax_cost_of_debt=0.045 * (1 - 0.21),
        wacc=0.09,
    )
    result = build_dcf_result(
        ticker="TST",
        forecast=forecast,
        wacc_inputs=wacc,
        market=market,
        latest_cash=20.0,
        latest_debt=100.0,
        latest_shares=100.0,
        sources=[source],
        warnings=[],
        settings=Settings(terminal_growth_source="internal"),
    )
    assert result.intrinsic_value_per_share > 0
    assert result.enterprise_value > result.pv_projected_fcf
    assert result.explicit_forecast_quarters == 20


def test_terminal_value_uses_economic_fcf_path_when_available() -> None:
    source = SourceMeta(provider="test", key="fixture")
    forecast = GrowthForecast(
        ticker="TST",
        model_type="univariate_local_linear",
        single_growth_rate=0.02,
        raw_fcf_growth_rate=0.02,
        smoothed_fcf_growth_rate=0.02,
        clamped_fcf_growth_rate=0.02,
        growth_clamped=False,
        clamp_triggered=False,
        horizon_quarters=20,
        growth_path=[0.02] * 20,
        fcf_path=[50.0] * 20,
        lower_growth_path=[0.0] * 20,
        upper_growth_path=[0.04] * 20,
        diagnostics={
            "terminal_fcf_path": [100.0] * 20,
            "valuation_mode": "reported",
            "maintenance_capex_method": "ensemble",
            "maintenance_capex_estimates": {"depreciation_anchor": 0.05},
            "forecast_components": [
                {
                    "quarter": float(i + 1),
                    "revenue": 200.0,
                    "reported_capex": 40.0,
                    "maintenance_capex": 20.0,
                    "growth_capex": 20.0,
                    "reported_fcf": 50.0,
                    "economic_fcf": 70.0,
                }
                for i in range(20)
            ],
        },
    )
    market = MarketSnapshot(
        ticker="TST",
        price=10.0,
        market_cap=1000.0,
        shares_outstanding=100.0,
        beta=1.0,
        source=source,
    )
    wacc = WACCInputs(
        risk_free_rate=0.04,
        beta=1.0,
        equity_risk_premium=0.05,
        cost_of_equity=0.09,
        pre_tax_cost_of_debt=0.045,
        tax_rate=0.21,
        market_value_equity=1000.0,
        market_value_debt=100.0,
        equity_weight=1000.0 / 1100.0,
        debt_weight=100.0 / 1100.0,
        after_tax_cost_of_debt=0.045 * (1 - 0.21),
        wacc=0.09,
    )
    result = build_dcf_result(
        ticker="TST",
        forecast=forecast,
        wacc_inputs=wacc,
        market=market,
        latest_cash=0.0,
        latest_debt=0.0,
        latest_shares=100.0,
        sources=[source],
        warnings=[],
        settings=Settings(terminal_growth_source="internal"),
    )
    assert result.fcf_terminal_base == 100.0
    assert result.diagnostics["terminal_stabilization"]["cash_flow_basis"] == "economic_fcf"
    assert result.diagnostics["maintenance_capex_method"] == "ensemble"
    assert result.diagnostics["economic_fcf_margin"] == 0.35
    assert result.diagnostics["reverse_dcf_growth"]["market_implied_annual_fcf_growth"] is not None
    assert result.sensitivity["maintenance_capex_surface"]
