from __future__ import annotations

from datetime import date

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.data_sources.fundamentals import FundamentalsBundle
from dcf_project.pipeline import DCFPipeline


class FixturePipeline(DCFPipeline):
    def load_data(self, ticker: str) -> FundamentalsBundle:
        periods = []
        revenue = 100.0
        nwc = 10.0
        for idx in range(24):
            revenue *= 1.03
            nwc += 0.5
            periods.append(
                FinancialPeriod(
                    ticker=ticker.upper(),
                    report_date=date(2020 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                    revenue=revenue,
                    ebit=revenue * 0.2,
                    tax_expense=revenue * 0.04,
                    capex=revenue * 0.06,
                    da=revenue * 0.03,
                    nwc=nwc,
                    cash=50.0,
                    debt=100.0,
                    shares_outstanding=100.0,
                )
            )
        source = SourceMeta(provider="fixture", key="TST")
        market = MarketSnapshot(
            ticker=ticker.upper(),
            price=10.0,
            market_cap=1000.0,
            shares_outstanding=100.0,
            beta=1.0,
            cash=50.0,
            debt=100.0,
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
        return FundamentalsBundle(periods, market, wacc, [source], [])


def test_pipeline_run_with_fixture_data() -> None:
    result = FixturePipeline(Settings(terminal_growth_source="internal")).run("TST", horizon_quarters=20)
    assert result.ticker == "TST"
    assert result.intrinsic_value_per_share > 0
    assert len(result.projected_fcf) == 20


def test_pipeline_uses_current_shares_not_forecast_buyback_denominator() -> None:
    result = FixturePipeline(Settings(terminal_growth_source="internal")).run("TST", horizon_quarters=20)
    trace = result.diagnostics["share_count_trace"]

    assert result.diluted_shares_outstanding == 100.0
    assert trace["final_valuation_denominator"] == 100.0
    assert trace["forecast_share_path_used_for_denominator"] is False
