from __future__ import annotations

from scripts.valuation_attribution_audit import build_attribution_report

from dcf_project.config import Settings
from dcf_project.contracts import DCFResult, SourceMeta


class FixturePipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, ticker: str) -> DCFResult:
        return _result(ticker)


def test_attribution_report_produces_aapl_msft_tsla_tables() -> None:
    report = build_attribution_report(["AAPL", "MSFT", "TSLA"], Settings(), pipeline_factory=FixturePipeline)

    assert set(report) == {"AAPL", "MSFT", "TSLA"}
    for ticker in report:
        row = report[ticker]
        assert row["raw_model_forecast"]
        assert row["consensus_forecast"]
        assert row["blended_forecast"]
        assert row["overlay_forecast"]
        assert row["final_forecast"]
        assert row["terminal_trace"]
        assert row["wacc_trace"]
        assert row["share_count_trace"]["forecast_share_path_used_for_denominator"] is False
        assert "revenue_cagr_plus_100bps" in row["value_impact"]


def test_attribution_report_traces_scenario_ttm_revenue_base() -> None:
    report = build_attribution_report(["TSLA"], Settings(), pipeline_factory=FixturePipeline)
    trace = report["TSLA"]["scenario_trace"]

    assert report["TSLA"]["valuation_regime"] == "optionality_hypergrowth"
    assert trace["scenario_revenue_base_trace"]["method"] == "trailing_four_quarters"
    assert trace["scenario_revenue_base_trace"]["annual_revenue"] == 1000.0


def test_proxy_and_clamp_status_are_classified_explicitly() -> None:
    report = build_attribution_report(["AAPL"], Settings(), pipeline_factory=FixturePipeline)
    classifications = {item["issue"]: item["classification"] for item in report["AAPL"]["bug_classification"]}

    assert classifications["proxy invested capital not used in default ROIC path"] == "assumption choice, not a bug"
    assert classifications["explicit policy clamps triggered and flagged in field_source_map"] == "assumption choice, not a bug"


def _result(ticker: str) -> DCFResult:
    is_tsla = ticker.upper() == "TSLA"
    scenario_values = {"bear": 50.0, "base": 150.0, "bull": 450.0} if is_tsla else {}
    probability_weights = {"bear": 0.25, "base": 0.50, "bull": 0.25} if is_tsla else {}
    forecast = [
        {
            "quarter": float(idx),
            "revenue": 250.0 * (1.01**idx),
            "revenue_growth": 0.01 + idx * 0.001,
            "operating_margin": 0.30,
            "reinvestment_rate": 0.05,
            "roic": 0.25,
            "fcf": 25.0 * (1.02**idx),
            "shares": 99.0 - idx * 0.1,
        }
        for idx in range(1, 21)
    ]
    diagnostics = {
        "raw_model_forecast": forecast,
        "consensus_forecast": forecast,
        "blended_forecast": forecast,
        "overlay_forecast": forecast,
        "final_forecast": forecast,
        "field_source_map": [
            {
                "quarter": 1.0,
                "revenue_growth": {"source": "consensus", "clamp_triggered": True},
                "operating_margin": {"source": "model", "clamp_triggered": False},
            }
        ],
        "clamp_triggered_map": {"revenue_growth": True, "operating_margin": False, "reinvestment_rate": False, "roic": False},
        "terminal_trace": {"terminal_base_annual": 120.0, "terminal_growth": 0.03},
        "wacc_trace": {"final_wacc": 0.09},
        "consensus_growth_conversion": {
            "input_basis": "annualized_cagr",
            "annual_growth": 0.08,
            "quarterly_growth": (1.08**0.25) - 1.0,
        },
        "invested_capital_components": [{"proxy_used": False}],
        "share_count_trace": {
            "valuation_denominator": 100.0,
            "final_valuation_denominator": 100.0,
            "forecast_terminal_shares": 97.0,
            "forecast_share_path_used_for_denominator": False,
        },
    }
    if is_tsla:
        diagnostics.update(
            {
                "state_space_dcf_diagnostic_intrinsic": 80.0,
                "scenario_revenue_base_trace": {
                    "annual_revenue": 1000.0,
                    "method": "trailing_four_quarters",
                    "quarterly_revenues_used": [240.0, 250.0, 255.0, 255.0],
                },
            }
        )
    return DCFResult(
        ticker=ticker.upper(),
        explicit_forecast_quarters=20,
        intrinsic_value_per_share=200.0 if not is_tsla else 250.0,
        enterprise_value=20_500.0,
        equity_value=20_000.0 if not is_tsla else 25_000.0,
        net_debt=500.0,
        projected_fcf=[row["fcf"] for row in forecast],
        explicit_pv=400.0,
        pv_projected_fcf=400.0,
        terminal_value=20_000.0,
        pv_terminal_value=15_000.0,
        pv_terminal=15_000.0,
        fcf_terminal_base=30.0,
        wacc=0.09,
        terminal_growth_rate=0.03,
        fcf_growth_rate=0.08,
        raw_fcf_growth_rate=0.08,
        smoothed_fcf_growth_rate=0.08,
        clamped_fcf_growth_rate=0.08,
        growth_clamped=False,
        clamp_triggered=False,
        valuation_p10=150.0,
        valuation_p50=200.0,
        valuation_p90=260.0,
        diluted_shares_outstanding=100.0,
        market_price=220.0 if not is_tsla else 400.0,
        forecast_horizon_quarters=20,
        forecast_mode="consensus_blend",
        valuation_regime="optionality_hypergrowth" if is_tsla else "mature_compounder",
        scenario_values=scenario_values,
        probability_weights=probability_weights,
        probability_method="fixture" if is_tsla else None,
        forecast_financials=forecast,
        diagnostics=diagnostics,
        sources=[SourceMeta(provider="fixture", key=ticker.upper())],
    )
