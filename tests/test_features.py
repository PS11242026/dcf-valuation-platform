from __future__ import annotations

from datetime import date

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod
from dcf_project.preprocessing.features import build_features


def test_build_features_computes_ratios_and_fcf() -> None:
    periods = [
        FinancialPeriod(
            ticker="TST",
            report_date=date(2023, 3, 31),
            revenue=100.0,
            ebit=20.0,
            tax_expense=4.0,
            capex=8.0,
            da=3.0,
            nwc=10.0,
        ),
        FinancialPeriod(
            ticker="TST",
            report_date=date(2023, 6, 30),
            revenue=110.0,
            ebit=22.0,
            tax_expense=4.4,
            capex=9.0,
            da=3.2,
            nwc=11.0,
        ),
        FinancialPeriod(
            ticker="TST",
            report_date=date(2023, 9, 30),
            revenue=121.0,
            ebit=24.2,
            tax_expense=4.8,
            capex=9.5,
            da=3.4,
            nwc=12.0,
        ),
        FinancialPeriod(
            ticker="TST",
            report_date=date(2023, 12, 31),
            revenue=133.1,
            ebit=26.6,
            tax_expense=5.3,
            capex=10.0,
            da=3.7,
            nwc=13.0,
        ),
        FinancialPeriod(
            ticker="TST",
            report_date=date(2024, 3, 31),
            revenue=146.4,
            ebit=29.3,
            tax_expense=5.8,
            capex=11.0,
            da=4.0,
            nwc=14.0,
        ),
    ]
    df, frame = build_features(periods)
    assert len(df) == 4
    assert frame.ticker == "TST"
    latest = df.iloc[-1]
    assert round(latest["ebit_margin"], 3) == 0.2
    assert latest["fcf"] > 0


def test_economic_fcf_removes_growth_capex_for_hyperscaler() -> None:
    periods = []
    revenue = 100.0
    nwc = 10.0
    for idx in range(8):
        revenue *= 1.05
        nwc += 0.5
        periods.append(
            FinancialPeriod(
                ticker="TST",
                report_date=date(2023 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                revenue=revenue,
                ebit=revenue * 0.25,
                tax_expense=revenue * 0.05,
                capex=revenue * 0.18,
                da=revenue * 0.04,
                nwc=nwc,
            )
        )
    reported, _ = build_features(
        periods,
        Settings(valuation_mode="reported", company_type="hyperscaler"),
    )
    economic, _ = build_features(
        periods,
        Settings(valuation_mode="economic", company_type="hyperscaler"),
    )
    assert economic.iloc[-1]["maintenance_capex"] < economic.iloc[-1]["reported_capex"]
    assert economic.iloc[-1]["growth_capex"] > 0
    assert economic.iloc[-1]["economic_fcf"] > economic.iloc[-1]["reported_fcf"]
    assert economic.iloc[-1]["fcf"] > reported.iloc[-1]["fcf"]
    for column in [
        "maintenance_capex_depreciation_anchor_ratio",
        "maintenance_capex_growth_adjusted_ratio",
        "maintenance_capex_historical_lower_band_ratio",
        "maintenance_capex_asset_turnover_ratio",
    ]:
        assert column in economic
    assert economic.iloc[-1]["maintenance_capex_method"] == "ensemble"


def test_owner_earnings_adds_partial_sbc_after_haircut() -> None:
    periods = []
    revenue = 100.0
    nwc = 10.0
    for idx in range(8):
        revenue *= 1.02
        nwc += 0.25
        periods.append(
            FinancialPeriod(
                ticker="AMZN",
                report_date=date(2023 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                revenue=revenue,
                ebit=revenue * 0.10,
                tax_expense=revenue * 0.02,
                capex=revenue * 0.08,
                da=revenue * 0.03,
                sbc=revenue * 0.04,
                nwc=nwc,
            )
        )
    df, _ = build_features(
        periods,
        Settings(valuation_mode="owner_earnings", company_type="marketplace_hybrid"),
    )
    latest = df.iloc[-1]
    assert latest["sbc_adjustment"] == latest["sbc"] * latest["sbc_haircut_factor"]
    assert latest["owner_earnings"] > latest["economic_fcf"]
    assert latest["fcf"] == latest["owner_earnings"]


def test_features_include_roic_reinvestment_economics() -> None:
    periods = []
    revenue = 100.0
    nwc = 10.0
    for idx in range(8):
        revenue *= 1.03
        nwc += 0.5
        periods.append(
            FinancialPeriod(
                ticker="TST",
                report_date=date(2023 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                revenue=revenue,
                ebit=revenue * 0.20,
                tax_expense=revenue * 0.04,
                capex=revenue * 0.05,
                da=revenue * 0.03,
                ppe=revenue * 0.30,
                nwc=nwc,
            )
        )
    df, _ = build_features(periods, Settings(valuation_mode="roic_dcf"))
    latest = df.iloc[-1]
    assert latest["invested_capital"] > 0
    assert latest["roic"] > 0
    assert 0 <= latest["reinvestment_rate"] <= 0.95
    assert "growth_from_roic" in df
