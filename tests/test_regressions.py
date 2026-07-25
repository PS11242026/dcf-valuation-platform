from __future__ import annotations

from datetime import date

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.data_sources.fundamentals import FundamentalsBundle
from dcf_project.models.forecast import reconstruct_fcf_from_states
from dcf_project.models.state_space import FCFStateSpaceModel
from dcf_project.pipeline import DCFPipeline
from dcf_project.valuation.discounting import present_value
from dcf_project.valuation.terminal import TerminalValueError, terminal_value

import numpy as np
import pytest


class AppleScaleFixturePipeline(DCFPipeline):
    def load_data(self, ticker: str) -> FundamentalsBundle:
        periods = []
        revenue = 90_000_000_000.0
        nwc = -5_000_000_000.0
        for idx in range(24):
            revenue *= 1.01
            nwc += 100_000_000.0
            periods.append(
                FinancialPeriod(
                    ticker=ticker.upper(),
                    report_date=date(2021 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                    revenue=revenue,
                    ebit=revenue * 0.30,
                    tax_expense=revenue * 0.045,
                    capex=revenue * 0.025,
                    da=revenue * 0.03,
                    nwc=nwc,
                    cash=65_000_000_000.0,
                    debt=85_000_000_000.0,
                    shares_outstanding=15_000_000_000.0,
                )
            )
        source = SourceMeta(provider="fixture", key="AAPL")
        market = MarketSnapshot(
            ticker=ticker.upper(),
            price=300.0,
            market_cap=4_500_000_000_000.0,
            shares_outstanding=15_000_000_000.0,
            beta=1.0,
            cash=65_000_000_000.0,
            debt=85_000_000_000.0,
            source=source,
        )
        wacc = WACCInputs(
            risk_free_rate=0.04,
            beta=1.0,
            equity_risk_premium=0.05,
            cost_of_equity=0.09,
            pre_tax_cost_of_debt=0.045,
            tax_rate=0.18,
            market_value_equity=4_500_000_000_000.0,
            market_value_debt=85_000_000_000.0,
            equity_weight=4_500_000_000_000.0 / 4_585_000_000_000.0,
            debt_weight=85_000_000_000.0 / 4_585_000_000_000.0,
            after_tax_cost_of_debt=0.045 * (1 - 0.18),
            wacc=0.09,
        )
        return FundamentalsBundle(periods, market, wacc, [source], [])


def test_aapl_sanity() -> None:
    result = AppleScaleFixturePipeline(Settings(terminal_growth_source="internal")).run("AAPL", horizon_quarters=20)
    assert result.enterprise_value > 0
    assert 25.0 < result.intrinsic_value_per_share < 1_000.0


def test_terminal_guardrails() -> None:
    with pytest.raises(TerminalValueError):
        terminal_value(100.0, 0.02, 0.019)
    assert terminal_value(100.0, 0.02, 0.01) > 0


def test_discount_frequency() -> None:
    annual_wacc = 0.08
    quarterly_rate = annual_wacc / 4.0
    assert present_value(100.0, quarterly_rate, 4) == pytest.approx(100.0 / (1.02**4))


def test_fcf_signs() -> None:
    states = np.array([[0.0, 0.30, 0.05, 0.02, 0.03, 0.20]])
    fcf_path, growth_path, components = reconstruct_fcf_from_states(states, 100.0, 20.0)
    assert fcf_path[0] == pytest.approx(30.0 * 0.8 + 3.0 - 5.0 - 2.0)
    assert components[0]["capex"] > 0
    assert components[0]["delta_nwc"] > 0
    assert growth_path[0] == pytest.approx((fcf_path[0] - 20.0) / 20.0)


def test_state_space_stability() -> None:
    x = np.array(
        [
            [0.01, 0.30, 0.20, 0.08],
            [0.02, 0.31, 0.20, 0.08],
            [0.01, 0.30, 0.22, 0.07],
            [0.03, 0.32, 0.19, 0.08],
            [0.02, 0.31, 0.20, 0.08],
            [0.02, 0.30, 0.21, 0.08],
            [0.01, 0.31, 0.20, 0.08],
            [0.02, 0.32, 0.19, 0.08],
        ]
    )
    model = FCFStateSpaceModel().fit(x)
    assert np.max(np.abs(np.linalg.eigvals(model.phi))) < 1
    assert np.min(np.linalg.eigvalsh(model.q)) > 0
    assert np.min(np.linalg.eigvalsh(model.h)) > 0
    assert np.min(np.linalg.eigvalsh(model.result.filtered_cov)) > 0
