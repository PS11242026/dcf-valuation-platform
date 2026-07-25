"""End-to-end engine_v3 tests on the synthetic fixture + the V3 golden lock.

The fixture extends the v1 FixturePipeline data with the balance-sheet stock accounts
(current assets/liabilities, PPE) that invested-capital reconstruction needs -- the v1
fixture's thin balance sheet is itself a useful case: it must produce the TYPED skip,
never garbage values.
"""
from __future__ import annotations

from datetime import date

import pytest

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.data_sources.fundamentals import FundamentalsBundle
from dcf_project.engine_v3.drivers import DriverExtractionError
from dcf_project.engine_v3.pipeline import EconomicDCFPipeline
from test_pipeline import FixturePipeline


class V3FixturePipeline(EconomicDCFPipeline):
    def load_data(self, ticker: str) -> FundamentalsBundle:
        periods = []
        revenue = 100.0
        for idx in range(24):
            revenue *= 1.03
            periods.append(
                FinancialPeriod(
                    ticker=ticker.upper(),
                    report_date=date(2020 + idx // 4, [3, 6, 9, 12][idx % 4], 28),
                    revenue=revenue,
                    ebit=revenue * 0.2,
                    tax_expense=revenue * 0.04,
                    capex=revenue * 0.06,
                    da=revenue * 0.03,
                    current_assets=revenue * 0.40 + 50.0,
                    current_liabilities=revenue * 0.28,
                    ppe=revenue * 4.0,
                    cash=50.0,
                    debt=100.0,
                    shares_outstanding=100.0,
                )
            )
        source = SourceMeta(provider="fixture", key="TST")
        market = MarketSnapshot(
            ticker=ticker.upper(), price=10.0, market_cap=1000.0, shares_outstanding=100.0,
            beta=1.0, cash=50.0, debt=100.0, source=source,
        )
        wacc = WACCInputs(
            risk_free_rate=0.04, beta=1.0, equity_risk_premium=0.05, cost_of_equity=0.09,
            pre_tax_cost_of_debt=0.045, tax_rate=0.21, market_value_equity=1000.0,
            market_value_debt=100.0, equity_weight=1000.0 / 1100.0, debt_weight=100.0 / 1100.0,
            after_tax_cost_of_debt=0.045 * (1 - 0.21), wacc=0.09,
        )
        return FundamentalsBundle(periods, market, wacc, [source], [])


SETTINGS = Settings(terminal_growth_source="internal", v3_mc_draws=200)

# Golden v3 outputs from the deterministic fixture, REGENERATED at the productization
# pivot (2026-06-10): explicit horizon is now 40 quarters (10 years) and growth/margin
# fade at per-driver half-lives (the constant-growth fixture is degenerate, so both fall
# back to the CAP-derived half-life -- the integrated 40q numerics are what change). The
# longer horizon pushes value out of the perpetuity: PV(terminal)/EV falls 0.889 -> 0.717.
# v1 golden remains byte-identical (tests/test_engine_v2_scaffold.py). Seeded MC, 200 draws.
GOLDEN_V3 = {
    "intrinsic_value_per_share": 14.7126637656,
    "enterprise_value": 1521.2663765552,
    "equity_value": 1471.2663765552,
    "terminal_value": 2657.8865144756,
    "terminal_growth_rate": 0.03,
    "sum_projected_fcf": 735.8496756140,
    "valuation_p10": 14.2177504719,
    "valuation_p50": 14.6321477217,
    "valuation_p90": 15.1281370825,
    "phi": 0.8125,
    "cap_years": 11.860603,
    "excess_roic_0": 0.0586839361,
    "g_base": 0.1255088100,
    "pv_terminal_share_of_ev": 0.7174613328,
    "half_life_growth_years": 3.5703973325,
    "half_life_margin_years": 3.5703973325,
}


def _run():
    # No explicit horizon -> the v3 product default (40q), so the golden locks what the
    # flagship `valuation dcf` actually produces.
    return V3FixturePipeline(SETTINGS).run("TST")


def test_v3_golden_numerics() -> None:
    r = _run()
    assert r.intrinsic_value_per_share == pytest.approx(GOLDEN_V3["intrinsic_value_per_share"], rel=1e-9)
    assert r.enterprise_value == pytest.approx(GOLDEN_V3["enterprise_value"], rel=1e-9)
    assert r.equity_value == pytest.approx(GOLDEN_V3["equity_value"], rel=1e-9)
    assert r.terminal_value == pytest.approx(GOLDEN_V3["terminal_value"], rel=1e-9)
    assert r.terminal_growth_rate == pytest.approx(GOLDEN_V3["terminal_growth_rate"], rel=1e-9)
    assert sum(r.projected_fcf) == pytest.approx(GOLDEN_V3["sum_projected_fcf"], rel=1e-9)
    assert r.valuation_p10 == pytest.approx(GOLDEN_V3["valuation_p10"], rel=1e-9)
    assert r.valuation_p50 == pytest.approx(GOLDEN_V3["valuation_p50"], rel=1e-9)
    assert r.valuation_p90 == pytest.approx(GOLDEN_V3["valuation_p90"], rel=1e-9)
    audit = r.diagnostics["driver_audit"]
    assert audit["persistence"]["phi"] == pytest.approx(GOLDEN_V3["phi"], rel=1e-9)
    assert audit["persistence"]["cap_years"] == pytest.approx(GOLDEN_V3["cap_years"], abs=1e-5)
    assert audit["drivers"]["excess_roic_0"] == pytest.approx(GOLDEN_V3["excess_roic_0"], rel=1e-8)
    assert audit["g_base"] == pytest.approx(GOLDEN_V3["g_base"], rel=1e-9)
    assert audit["pv_terminal_share_of_ev"] == pytest.approx(GOLDEN_V3["pv_terminal_share_of_ev"], rel=1e-8)
    assert audit["half_life_growth_years"] == pytest.approx(GOLDEN_V3["half_life_growth_years"], rel=1e-8)
    assert audit["half_life_margin_years"] == pytest.approx(GOLDEN_V3["half_life_margin_years"], rel=1e-8)


def test_result_contract_complete_and_finite() -> None:
    r = _run()
    assert r.engine_version == "v3"
    assert r.forecast_mode == "economic_fade"
    assert r.ticker == "TST"
    assert r.explicit_forecast_quarters == 40 and len(r.projected_fcf) == 40
    assert r.diluted_shares_outstanding == 100.0
    assert r.net_debt == pytest.approx(50.0)
    for name in (
        "intrinsic_value_per_share", "enterprise_value", "equity_value", "explicit_pv",
        "pv_projected_fcf", "terminal_value", "pv_terminal_value", "fcf_terminal_base",
        "wacc", "terminal_growth_rate", "fcf_growth_rate", "raw_fcf_growth_rate",
        "valuation_p10", "valuation_p50", "valuation_p90",
    ):
        value = getattr(r, name)
        assert value is not None and value == value, name  # finite, not NaN
    assert len(r.forecast_financials) == 40
    assert all("revenue" in row and "fcf" in row for row in r.forecast_financials)
    audit = r.diagnostics["driver_audit"]
    assert audit["terminal"]["terminal_fcf_t1_annual"] / 4.0 == pytest.approx(r.fcf_terminal_base, rel=1e-12)


def test_driver_audit_complete() -> None:
    audit = _run().diagnostics["driver_audit"]
    for key in (
        "drivers", "persistence", "grinold_kroner", "terminal", "paths",
        "pv_terminal_share_of_ev", "g_base", "g_base_effective",
        "half_life_growth_years", "half_life_margin_years", "half_lives",
        "margin_target_blend", "fcf_margin_band", "fcf_margin_breach_quarters",
        "growth_binds_quarters", "divestment_binds_quarters",
    ):
        assert key in audit, key
    assert set(audit["paths"]) == {"growth_annual", "nopat_margin", "ronic", "reinvestment_rate"}
    assert all(len(path) == 40 for path in audit["paths"].values())
    assert audit["grinold_kroner"]["terminal_growth"] == pytest.approx(0.03)
    # Per-driver half-life provenance is recorded (here both fall back: constant fixture).
    for leg in ("growth", "margin"):
        assert audit["half_lives"][leg]["source"] in {"company_ar1_shrunk", "fallback_cap"}


def test_mc_payload_attached_and_headline_stays_deterministic() -> None:
    r = _run()
    mc = r.monte_carlo_distribution
    assert mc["engine"] == "v3" and mc["n_effective"] == 200 and mc["failed_draws"] == 0
    assert "jensen_gap" in mc and mc["skeleton_iv"] == pytest.approx(r.intrinsic_value_per_share)
    assert r.confidence_interval == {
        "p10": r.valuation_p10, "p50": r.valuation_p50, "p90": r.valuation_p90,
    }
    # Headline is the deterministic skeleton, NOT the (Jensen-shifted) MC median.
    assert r.intrinsic_value_per_share != r.valuation_p50


def test_mc_median_headline_opt_in() -> None:
    r = V3FixturePipeline(
        Settings(terminal_growth_source="internal", v3_mc_draws=200, v3_headline="mc_median")
    ).run("TST")
    assert r.intrinsic_value_per_share == pytest.approx(r.valuation_p50, rel=1e-12)


def test_thin_balance_sheet_is_typed_skip() -> None:
    class ThinFixture(EconomicDCFPipeline):
        load_data = FixturePipeline.load_data

    with pytest.raises(DriverExtractionError):
        ThinFixture(SETTINGS).run("TST")
