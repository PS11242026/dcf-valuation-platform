from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.data_sources.fundamentals import FundamentalsBundle
from dcf_project.models.state_space import FCFStateSpaceModel, model_derived_fcf_growth
from dcf_project.pipeline import DCFPipeline
from dcf_project.preprocessing.reconstruction import reconstruct_historical_fcf, state_matrix
from dcf_project.valuation.dcf import build_state_space_dcf_result


def _periods(ticker: str = "TST", scale: float = 1.0) -> list[FinancialPeriod]:
    periods = []
    revenue = 100.0 * scale
    current_assets = 45.0 * scale
    cash = 10.0 * scale
    current_liabilities = 20.0 * scale
    for idx in range(24):
        revenue *= 1.02
        current_assets += 1.0 * scale
        current_liabilities += 0.4 * scale
        periods.append(
            FinancialPeriod(
                ticker=ticker,
                report_date=date(2020 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                revenue=revenue,
                ebit=revenue * 0.25,
                tax_expense=revenue * 0.04,
                da=revenue * 0.03,
                capex=revenue * 0.05,
                current_assets=current_assets,
                cash=cash,
                current_liabilities=current_liabilities,
                debt=30.0 * scale,
                shares_outstanding=10.0 * scale,
            )
        )
    return periods


def _market(ticker: str = "TST") -> tuple[MarketSnapshot, WACCInputs, SourceMeta]:
    source = SourceMeta(provider="fixture", key=ticker)
    market = MarketSnapshot(
        ticker=ticker,
        price=20.0,
        market_cap=200.0,
        shares_outstanding=10.0,
        beta=1.0,
        cash=10.0,
        debt=30.0,
        source=source,
    )
    wacc = WACCInputs(
        risk_free_rate=0.04,
        beta=1.0,
        equity_risk_premium=0.05,
        cost_of_equity=0.09,
        pre_tax_cost_of_debt=0.045,
        tax_rate=0.21,
        market_value_equity=200.0,
        market_value_debt=30.0,
        equity_weight=200.0 / 230.0,
        debt_weight=30.0 / 230.0,
        after_tax_cost_of_debt=0.045 * (1 - 0.21),
        wacc=0.09,
    )
    return market, wacc, source


def test_fcf_reconstruction() -> None:
    result = reconstruct_historical_fcf(_periods())
    row = result.history.iloc[1]
    expected_nwc = (row["current_assets"] - row["cash"]) - row["current_liabilities"]
    expected_fcf = row["nopat"] + row["da"] - row["capex"] - row["delta_nwc"]
    assert row["nwc"] == pytest.approx(expected_nwc)
    assert row["fcf"] == pytest.approx(expected_fcf)
    assert len(result.history) >= 20


def test_state_space_fit() -> None:
    history = reconstruct_historical_fcf(_periods()).history
    model = FCFStateSpaceModel().fit(state_matrix(history))
    assert model.result is not None
    assert model.phi.shape == (4, 4)
    assert model.q.shape == (4, 4)


def test_kalman_stability() -> None:
    history = reconstruct_historical_fcf(_periods()).history
    model = FCFStateSpaceModel().fit(state_matrix(history))
    assert np.max(np.abs(np.linalg.eigvals(model.phi))) < 1.0
    assert np.min(np.linalg.eigvalsh(model.q)) > 0.0
    assert np.min(np.linalg.eigvalsh(model.r)) > 0.0
    assert np.min(np.linalg.eigvalsh(model.result.filtered_cov)) > 0.0


def test_forecast_20_quarters() -> None:
    history = reconstruct_historical_fcf(_periods()).history
    model = FCFStateSpaceModel().fit(state_matrix(history))
    forecast = model.forecast_financials(history.iloc[-1]["revenue"], history.iloc[-1]["fcf"], 20)
    assert len(forecast.financials) == 20
    assert forecast.states.shape == (20, 4)
    assert np.isfinite(forecast.financials["fcf"]).all()


def test_dcf_discounting() -> None:
    market, wacc, source = _market()
    result = build_state_space_dcf_result(
        "TST",
        forecast_financials=_flat_forecast(),
        base_fcf=100.0,
        fcf_growth_rate=0.0,
        wacc_inputs=wacc,
        market=market,
        latest_cash=10.0,
        latest_debt=30.0,
        latest_shares=10.0,
        sources=[source],
        warnings=[],
        settings=Settings(monte_carlo_samples=0, run_sensitivity=False, terminal_growth_source="internal"),
    )
    expected = sum(100.0 / ((1.0 + 0.09 / 4.0) ** quarter) for quarter in range(1, 21))
    assert result.explicit_pv == pytest.approx(expected)


def test_terminal_value() -> None:
    market, wacc, source = _market()
    result = build_state_space_dcf_result(
        "TST",
        forecast_financials=_flat_forecast(),
        base_fcf=100.0,
        fcf_growth_rate=0.0,
        wacc_inputs=wacc,
        market=market,
        latest_cash=10.0,
        latest_debt=30.0,
        latest_shares=10.0,
        sources=[source],
        warnings=[],
        settings=Settings(monte_carlo_samples=0, run_sensitivity=False, terminal_growth_source="internal"),
    )
    assert result.terminal_value == pytest.approx(400.0 * 1.025 / (0.09 - 0.025))
    assert result.diagnostics["terminal_trace"]["terminal_base_quarterly"] == pytest.approx(100.0)
    assert result.diagnostics["terminal_trace"]["terminal_base_annual"] == pytest.approx(400.0)


def test_equity_bridge() -> None:
    market, wacc, source = _market()
    result = build_state_space_dcf_result(
        "TST",
        forecast_financials=_flat_forecast(),
        base_fcf=100.0,
        fcf_growth_rate=0.0,
        wacc_inputs=wacc,
        market=market,
        latest_cash=10.0,
        latest_debt=30.0,
        latest_shares=10.0,
        sources=[source],
        warnings=[],
        settings=Settings(monte_carlo_samples=0, run_sensitivity=False, terminal_growth_source="internal"),
    )
    assert result.net_debt == pytest.approx(20.0)
    assert result.equity_value == pytest.approx(result.enterprise_value - 20.0)
    assert result.intrinsic_value_per_share == pytest.approx(result.equity_value / 10.0)


def test_model_derived_growth_only_from_forecast_path() -> None:
    projected = np.array([100.0] * 4 + [110.0] * 12 + [150.0] * 4)
    assert model_derived_fcf_growth(10.0, projected, 20) == pytest.approx(1.5 ** 0.25 - 1.0)


def test_discount_schedule_uses_quarterly_wacc() -> None:
    market, wacc, source = _market()
    result = build_state_space_dcf_result(
        "TST",
        forecast_financials=_flat_forecast(),
        base_fcf=100.0,
        fcf_growth_rate=0.0,
        wacc_inputs=wacc,
        market=market,
        latest_cash=10.0,
        latest_debt=30.0,
        latest_shares=10.0,
        sources=[source],
        warnings=[],
        settings=Settings(monte_carlo_samples=0, run_sensitivity=False, terminal_growth_source="internal"),
    )
    schedule = result.diagnostics["discount_schedule"]
    assert schedule[0]["discount_rate_per_quarter"] == pytest.approx(0.09 / 4.0)
    assert schedule[19]["discount_factor"] == pytest.approx((1.0 + 0.09 / 4.0) ** 20)
    assert schedule[19]["pv_fcf"] == pytest.approx(100.0 / ((1.0 + 0.09 / 4.0) ** 20))
    assert result.diagnostics["fcf_growth_method"] == "avg_fcf_17_20_over_avg_fcf_1_4_four_year_cagr"


def test_aapl_sanity_state_space_architecture() -> None:
    class FixturePipeline(DCFPipeline):
        def load_data(self, ticker: str) -> FundamentalsBundle:
            market, wacc, source = _market("AAPL")
            market.price = 300.0
            market.market_cap = 4_500_000_000_000.0
            market.shares_outstanding = 15_000_000_000.0
            market.cash = 65_000_000_000.0
            market.debt = 85_000_000_000.0
            wacc.market_value_equity = 4_500_000_000_000.0
            wacc.market_value_debt = 85_000_000_000.0
            periods = _periods("AAPL", scale=900_000_000.0)
            for period in periods:
                period.ebit = period.revenue * 0.30
                period.tax_expense = period.revenue * 0.045
                period.capex = period.revenue * 0.025
                period.da = period.revenue * 0.03
                period.cash = 65_000_000_000.0
                period.debt = 85_000_000_000.0
                period.shares_outstanding = 15_000_000_000.0
            return FundamentalsBundle(periods, market, wacc, [source], [])

    result = FixturePipeline(
        Settings(monte_carlo_samples=0, run_sensitivity=False, terminal_growth_source="internal")
    ).run("AAPL")
    assert result.enterprise_value > 0.0
    assert 10.0 < result.intrinsic_value_per_share < 1_000.0
    assert len(result.forecast_financials) == 20


def _flat_forecast():
    return pd.DataFrame(
        {
            "quarter": [float(i) for i in range(1, 21)],
            "revenue": [500.0] * 20,
            "revenue_growth": [0.0] * 20,
            "operating_margin": [0.25] * 20,
            "ebit_margin": [0.25] * 20,
            "ebit": [125.0] * 20,
            "tax_rate": [0.0] * 20,
            "nopat": [125.0] * 20,
            "reinvestment_rate": [0.20] * 20,
            "reinvestment": [25.0] * 20,
            "invested_capital": [5_000.0] * 20,
            "roic": [0.025] * 20,
            "fcf": [100.0] * 20,
        }
    )
