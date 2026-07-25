from __future__ import annotations

from datetime import date

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod, SourceMeta
from dcf_project.data_sources.sec import SECClient, _select_instant_facts
from dcf_project.preprocessing.reconstruction import reconstruct_historical_fcf


def test_invested_capital_components_and_continuity_flag() -> None:
    periods = []
    revenue = 100.0
    for idx in range(24):
        revenue *= 1.01
        goodwill = None if idx < 20 else 120.0
        periods.append(
            FinancialPeriod(
                ticker="TST",
                report_date=date(2020 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                revenue=revenue,
                ebit=revenue * 0.25,
                tax_expense=revenue * 0.04,
                da=revenue * 0.03,
                capex=revenue * 0.04,
                current_assets=80.0,
                cash=20.0,
                current_liabilities=30.0,
                ppe=40.0,
                goodwill=goodwill,
                intangible_assets=10.0,
                operating_lease_liability=5.0,
                debt=30.0,
                shares_outstanding=10.0,
            )
        )

    result = reconstruct_historical_fcf(periods)
    history = result.history

    for column in [
        "operating_assets",
        "operating_liabilities",
        "cash_excluded",
        "goodwill",
        "intangible_assets",
        "lease_assets",
        "raw_invested_capital_reconstructed",
        "invested_capital",
    ]:
        assert column in history
    assert history["invested_capital_continuity_flag"].any()
    assert any("Invested capital" in warning for warning in result.warnings)


def test_proxy_ic_is_not_used_by_default_when_accounting_ic_is_low() -> None:
    periods = []
    revenue = 100.0
    for idx in range(24):
        revenue *= 1.01
        periods.append(
            FinancialPeriod(
                ticker="TST",
                report_date=date(2020 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                revenue=revenue,
                ebit=revenue * 0.25,
                tax_expense=revenue * 0.04,
                da=revenue * 0.03,
                capex=revenue * 0.04,
                current_assets=105.0 if idx < 20 else 130.0,
                cash=20.0,
                current_liabilities=84.0 if idx < 20 else 100.0,
                ppe=0.0,
                goodwill=0.0,
                intangible_assets=0.0,
                operating_lease_liability=0.0,
                debt=30.0,
                shares_outstanding=10.0,
            )
        )

    result = reconstruct_historical_fcf(periods)
    history = result.history

    assert not history["proxy_used"].any()
    assert history.iloc[-1]["ic_used_for_roic"] == history.iloc[-1]["accounting_ic"]
    assert history.iloc[-1]["ic_used_for_roic"] != history.iloc[-1]["proxy_ic"]
    assert any("proxy IC not used" in warning for warning in result.warnings)


def test_duplicate_comparative_instant_facts_collapse_to_earliest_filing() -> None:
    facts = [
        {
            "val": 100.0,
            "end": "2025-09-27",
            "form": "10-K",
            "filed": "2025-10-31",
            "accn": "original",
            "fp": "FY",
        },
        {
            "val": 999.0,
            "end": "2025-09-27",
            "form": "10-Q",
            "filed": "2026-05-01",
            "accn": "comparative",
            "fp": "Q2",
            "frame": "CY2025Q3I",
        },
    ]

    selected = _select_instant_facts(facts)

    assert len(selected) == 1
    assert selected[0]["val"] == 100.0
    assert selected[0]["accn"] == "original"


def test_long_stale_carry_forward_values_raise_warning() -> None:
    periods = []
    revenue = 100.0
    for idx in range(24):
        revenue *= 1.01
        periods.append(
            FinancialPeriod(
                ticker="TST",
                report_date=date(2020 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                revenue=revenue,
                ebit=revenue * 0.25,
                tax_expense=revenue * 0.04,
                da=revenue * 0.03,
                capex=revenue * 0.04,
                current_assets=80.0,
                cash=20.0,
                current_liabilities=30.0,
                ppe=40.0,
                goodwill=5.0 if idx == 0 else None,
                intangible_assets=10.0,
                operating_lease_liability=5.0,
                debt=30.0,
                shares_outstanding=10.0,
            )
        )

    result = reconstruct_historical_fcf(periods)

    assert result.history["goodwill_stale_carry_forward_flag"].any()
    assert any("goodwill carried forward" in warning for warning in result.warnings)


def test_ifrs_annual_20f_facts_expand_to_quarterly_periods(monkeypatch) -> None:
    facts = {"facts": {"ifrs-full": {}, "dei": {}}}
    annual_values = {
        "Revenue": 400.0,
        "ProfitLossFromOperatingActivities": 80.0,
        "IncomeTaxExpenseContinuingOperations": 16.0,
        "DepreciationAndAmortisationExpense": 20.0,
        "PropertyPlantAndEquipment": 120.0,
        "Goodwill": 10.0,
        "IntangibleAssetsOtherThanGoodwill": 5.0,
        "CurrentAssets": 200.0,
        "CurrentLiabilities": 90.0,
        "CashAndCashEquivalents": 50.0,
        "Borrowings": 30.0,
    }
    for concept, value in annual_values.items():
        facts["facts"]["ifrs-full"][concept] = {"units": {"USD": [_annual_20f_fact(year, value) for year in range(2020, 2025)]}}
    facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"] = {
        "units": {"shares": [_annual_20f_fact(year, 100.0) for year in range(2020, 2025)]}
    }
    client = SECClient(Settings())
    monkeypatch.setattr(client, "company_facts", lambda ticker: (facts, SourceMeta(provider="SEC", key="fixture")))

    periods, _ = client.financial_periods("IFRS")

    assert len(periods) == 20
    assert periods[0].source == "SEC:ifrs-full:annual_expanded"
    assert periods[0].revenue == 100.0
    assert periods[-1].shares_outstanding == 100.0


def _annual_20f_fact(year: int, value: float) -> dict[str, object]:
    return {
        "start": f"{year - 1}-04-01",
        "end": f"{year}-03-31",
        "val": value,
        "accn": f"fixture-{year}",
        "fy": year,
        "fp": "FY",
        "form": "20-F",
        "filed": f"{year}-06-01",
    }
