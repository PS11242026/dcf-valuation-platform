"""India market bolt-on: country routing, live-macro fallbacks, Ind-AS parsing, sector priors.

These cover the verifiable core of the bolt-on. The live NSE/BSE fetch + live macro endpoints
are network-dependent and intentionally not exercised here; the offline fallbacks and the pure
XBRL parser are. A US-path invariance test guards that the frozen engine is unperturbed.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dcf_project.config import Settings, get_settings
from dcf_project.contracts import MarketSnapshot, SourceMeta
from dcf_project.data_sources import fundamentals as fund
from dcf_project.data_sources import market as market_mod
from dcf_project.data_sources.india import filings, macro
from dcf_project.data_sources.india.sectors import (
    india_archetype_for_sector,
    india_sector_for_ticker,
    nic_to_sector,
)
from dcf_project.data_sources.markets import is_indian_ticker, resolve_country
from dcf_project.engine_v3.sector_priors import SECTOR_PRIORS, prior_for, prior_for_sector


# ----------------------------- country resolution -----------------------------
@pytest.mark.parametrize("ticker,expected", [
    ("RELIANCE.NS", "IN"), ("tcs.ns", "IN"), ("500325.BO", "IN"),
    ("KO", "US"), ("MSFT", "US"), ("BRK.B", "US"),
])
def test_resolve_country_infers_from_suffix(ticker: str, expected: str) -> None:
    assert resolve_country(ticker) == expected


def test_resolve_country_explicit_wins() -> None:
    assert resolve_country("RELIANCE.NS", "US") == "US"   # explicit overrides the suffix
    assert resolve_country("KO", "IN") == "IN"
    assert resolve_country("KO", "auto") == "US"          # auto falls back to inference
    assert resolve_country("KO", "garbage") == "US"       # unknown flag ignored


def test_is_indian_ticker() -> None:
    assert is_indian_ticker("INFY.NS") and is_indian_ticker("532540.BO")
    assert not is_indian_ticker("AAPL")


# ----------------------------- live macro fallbacks -----------------------------
def test_live_india_macro_offline_uses_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macro, "_fetch_gsec_10y", lambda timeout: None)
    monkeypatch.setattr(macro, "_fetch_damodaran_india_erp", lambda timeout: None)
    monkeypatch.setattr(macro, "_fetch_wb_indicator", lambda code, timeout: None)

    m = macro.live_india_macro()

    assert m.risk_free == pytest.approx(macro.FALLBACK_GSEC_10Y)
    assert m.equity_risk_premium == pytest.approx(macro.FALLBACK_ERP)
    assert m.inflation_anchor == pytest.approx(macro.FALLBACK_CPI)
    assert m.default_tax_rate == pytest.approx(macro.STATUTORY_TAX_RATE)
    # nominal GDP = real + CPI, clamped to the sanity band.
    assert m.long_run_nominal_gdp == pytest.approx(macro.FALLBACK_REAL_GDP + macro.FALLBACK_CPI)
    assert all(src == "fallback" for src in
               (m.sources["risk_free"], m.sources["equity_risk_premium"], m.sources["inflation_anchor"]))


def test_terminal_band_tied_to_live_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terminal-growth band tracks the macro: ceiling = G-Sec (Damodaran g<=rf), floor = CPI."""
    monkeypatch.setattr(macro, "_fetch_gsec_10y", lambda timeout: 0.072)
    monkeypatch.setattr(macro, "_fetch_damodaran_india_erp", lambda timeout: None)
    monkeypatch.setattr(macro, "_fetch_wb_indicator",
                        lambda code, timeout: 0.046 if code == "FP.CPI.TOTL.ZG" else None)

    m = macro.live_india_macro()

    assert m.terminal_growth_cap == pytest.approx(0.072)           # = G-Sec risk-free
    assert m.terminal_growth_floor == pytest.approx(0.046)         # = CPI inflation
    assert m.terminal_growth_floor < m.terminal_growth_cap
    assert m.sources["terminal_growth_band"] == "derived:cpi..gsec"


def test_settings_overrides_shape() -> None:
    m = macro.IndiaMacro(risk_free=0.071, equity_risk_premium=0.084, inflation_anchor=0.052,
                         long_run_nominal_gdp=0.108, default_cost_of_debt=0.091)
    ov = m.settings_overrides()
    assert ov["country"] == "IN"
    assert "wacc_source" not in ov                    # not forced: valueinvesting.io covers India
    assert ov["risk_free_override"] == pytest.approx(0.071)
    assert ov["equity_risk_premium"] == pytest.approx(0.084)
    # Every override key is a real Settings field (replace() would raise otherwise).
    replace(get_settings(), **ov)


# ----------------------------- risk-free override -----------------------------
def test_risk_free_override_returns_verbatim() -> None:
    settings = replace(get_settings(), risk_free_override=0.072)
    rate, source = market_mod.MarketClient(settings).risk_free_rate()
    assert rate == pytest.approx(0.072)
    assert source == "risk_free_override"


def test_default_settings_are_us_safe() -> None:
    s = Settings()
    assert s.country == "US"
    assert s.risk_free_override is None


# ----------------------------- India market-data suffix resolution -----------------------------
def _snap(ticker: str, *, price=None, mcap=None, shares=None) -> MarketSnapshot:
    return MarketSnapshot(ticker=ticker, price=price, market_cap=mcap, shares_outstanding=shares,
                          currency="INR", source=SourceMeta(provider="test", key=ticker))


def test_india_market_snapshot_appends_ns(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(self, t):  # noqa: ANN001
        seen.append(t)
        return _snap(t, price=2160.0, mcap=5e12, shares=2.3e9) if t == "HINDUNILVR.NS" else _snap(t)

    monkeypatch.setattr(fund.MarketClient, "snapshot", fake)
    snap = fund._india_market_snapshot("HINDUNILVR", get_settings(), [])
    assert snap.ticker == "HINDUNILVR.NS" and snap.price == 2160.0
    assert seen == ["HINDUNILVR.NS"]                       # .NS populated -> .BO never tried


def test_india_market_snapshot_falls_back_to_bo(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(self, t):  # noqa: ANN001
        return _snap(t, price=500.0, mcap=1e11, shares=2e8) if t.endswith(".BO") else _snap(t)

    monkeypatch.setattr(fund.MarketClient, "snapshot", fake)
    warnings: list[str] = []
    snap = fund._india_market_snapshot("SOMEBSEONLY", get_settings(), warnings)
    assert snap.ticker.endswith(".BO") and snap.price == 500.0
    assert any(".BO" in w for w in warnings)              # empty .NS -> .BO fallback, warned


def test_india_market_snapshot_keeps_existing_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(self, t):  # noqa: ANN001
        seen.append(t)
        return _snap(t, price=1296.0, mcap=1.7e13, shares=1.3e10)

    monkeypatch.setattr(fund.MarketClient, "snapshot", fake)
    fund._india_market_snapshot("RELIANCE.NS", get_settings(), [])
    assert seen == ["RELIANCE.NS"]                         # already suffixed -> used as-is


# ----------------------------- Ind-AS XBRL parser -----------------------------
# Mirrors the real NSE INDAS results taxonomy (validated against live RELIANCE data):
# reporting columns are distinguished by CONTEXT-ID PREFIX (OneD = current quarter, FourD =
# year-to-date), NOT by period dates (every context is stamped with the current quarter), and
# monetary facts are in FULL RUPEES ("Crores" rounding describes the PDF only -> never scaled).
_XBRL_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<xbrl xmlns="http://www.xbrl.org/2003/instance"
      xmlns:in="http://www.example.com/in-capmkt">
  <context id="OneD"><period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period></context>
  <context id="FourD"><period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period></context>
  <context id="OneI"><period><instant>2024-12-31</instant></period></context>
  <in:LevelOfRoundingUsedInFinancialStatements contextRef="OneD">Crores</in:LevelOfRoundingUsedInFinancialStatements>
  <in:RevenueFromOperations contextRef="OneD">1282600000000</in:RevenueFromOperations>
  <in:RevenueFromOperations contextRef="FourD">3966450000000</in:RevenueFromOperations>
  <in:ProfitBeforeTax contextRef="OneD">115970000000</in:ProfitBeforeTax>
  <in:TaxExpense contextRef="OneD">28760000000</in:TaxExpense>
  <in:CashAndCashEquivalents contextRef="OneI">50000000000</in:CashAndCashEquivalents>
  <in:NumberOfSharesOutstanding contextRef="OneI">13532000000</in:NumberOfSharesOutstanding>
</xbrl>
"""


def test_parser_returns_full_rupees_no_scaling() -> None:
    facts = filings.parse_india_xbrl_instance(_XBRL_FIXTURE)
    rev = [f for f in facts if f["field"] == "revenue"]
    assert {f["ctx"] for f in rev} == {"OneD", "FourD"}          # general parse keeps all contexts
    one_d = next(f for f in rev if f["ctx"] == "OneD")
    assert one_d["val"] == pytest.approx(1_282_600_000_000)      # full rupees, NOT x crore


def test_quarter_values_selects_current_quarter_context() -> None:
    report_date, values = filings.quarter_values_from_instance(_XBRL_FIXTURE)
    assert report_date.isoformat() == "2024-12-31"
    # revenue is the OneD (current quarter) value, never the FourD year-to-date.
    assert values["revenue"] == pytest.approx(1_282_600_000_000)
    assert values["ebit"] == pytest.approx(115_970_000_000)
    assert values["tax_expense"] == pytest.approx(28_760_000_000)
    assert values["cash"] == pytest.approx(50_000_000_000)       # instant from OneI


# Mirrors the 2018-2021 quirk: facts reference contextRef="OneD"/"OneI" but those contexts are
# NEVER DEFINED in the instance (only segment/expense contexts are). Selection must match the
# contextRef STRING on the fact, not the parsed context table.
_XBRL_UNDEFINED_ONED = """<?xml version="1.0" encoding="UTF-8"?>
<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:in="http://example.com/in">
  <context id="OneReportableSegmentRevenue01D"><period><startDate>2020-10-01</startDate><endDate>2020-12-31</endDate></period></context>
  <in:RevenueFromOperations contextRef="OneD">999000000000</in:RevenueFromOperations>
  <in:RevenueFromOperations contextRef="FourD">2500000000000</in:RevenueFromOperations>
  <in:ProfitBeforeTax contextRef="OneD">88000000000</in:ProfitBeforeTax>
</xbrl>
"""


def test_extract_handles_referenced_but_undefined_oned_context() -> None:
    vals = filings.extract_period_values(_XBRL_UNDEFINED_ONED)
    assert vals["revenue"] == pytest.approx(999_000_000_000)     # OneD picked by contextRef string
    assert vals["ebit"] == pytest.approx(88_000_000_000)         # never the FourD year-to-date


def test_carry_balance_sheet_fills_between_annuals() -> None:
    import datetime as dt

    merged = {
        dt.date(2022, 12, 31): {"revenue": 0.9},                 # before first annual -> back-filled
        dt.date(2023, 3, 31): {"revenue": 1.0, "ppe": 600.0, "cash": 90.0},
        dt.date(2023, 6, 30): {"revenue": 1.1},                  # between annuals -> forward-filled
    }
    filings._carry_balance_sheet(merged)
    assert merged[dt.date(2023, 6, 30)]["ppe"] == 600.0
    assert merged[dt.date(2022, 12, 31)]["ppe"] == 600.0


def _qfilings(symbol, period="Quarterly"):  # noqa: ANN001
    if period != "Quarterly":
        return []
    return [
        {"toDate": "31-Dec-2024", "consolidated": "Non-Consolidated", "xbrl": "https://x/std.xml"},
        {"toDate": "31-Dec-2024", "consolidated": "Consolidated", "xbrl": "https://x/con.xml"},
    ]


def test_financial_periods_prefers_consolidated(monkeypatch: pytest.MonkeyPatch) -> None:
    client = filings.IndiaFilingsClient(get_settings())
    monkeypatch.setattr(client, "_list_filings", _qfilings)
    monkeypatch.setattr(client, "_fetch_xbrl", lambda url: _XBRL_FIXTURE.encode())
    monkeypatch.setattr(filings.time, "sleep", lambda *_: None)

    rows, src = client.financial_periods("RELIANCE.NS")

    assert len(rows) == 1 and src.provider == "NSE"
    assert rows[0].source.endswith("consolidated")              # consolidated chosen over standalone
    assert rows[0].revenue == pytest.approx(1_282_600_000_000)
    assert rows[0].report_date.isoformat() == "2024-12-31"      # report date from the NSE record toDate


def test_financial_periods_raises_when_no_filings(monkeypatch: pytest.MonkeyPatch) -> None:
    client = filings.IndiaFilingsClient(get_settings())
    monkeypatch.setattr(client, "_list_filings", lambda symbol, period="Quarterly": [])
    with pytest.raises(filings.IndiaFilingsError):
        client.financial_periods("RELIANCE.NS")


# ----------------------------- sector priors + map -----------------------------
def test_india_sector_map() -> None:
    assert india_sector_for_ticker("TCS.NS") == "technology"
    assert india_sector_for_ticker("HINDUNILVR.NS") == "consumer_staples"
    assert india_sector_for_ticker("UNKNOWNXYZ.NS") is None
    assert nic_to_sector(62) == "technology"          # IT services division
    assert nic_to_sector(64) == "financials"


def test_prior_for_us_is_identical_to_prior_for_sector() -> None:
    for sector in list(SECTOR_PRIORS) + [None, "not_a_sector"]:
        assert prior_for("US", sector) == prior_for_sector(sector)


def test_india_priors_differ_by_country_same_sector() -> None:
    # FMCG (consumer_staples) and IT (technology) run structurally higher ROIC in India.
    assert prior_for("IN", "consumer_staples").roic_anchor > prior_for("US", "consumer_staples").roic_anchor
    assert prior_for("IN", "technology").roic_anchor > prior_for("US", "technology").roic_anchor


def test_prior_for_unknown_sector_falls_back_to_default() -> None:
    from dcf_project.engine_v3.sector_priors import DEFAULT_PRIOR

    assert prior_for("IN", "not_a_real_sector") == DEFAULT_PRIOR
    assert prior_for("IN", None) == DEFAULT_PRIOR


# ----------------------------- archetype mapping -----------------------------
def test_india_archetype_from_sector() -> None:
    # The capex-based US classifier collapses every India name to asset_light (no capex line),
    # so the archetype is taken from the GICS sector instead.
    assert india_archetype_for_sector("consumer_staples") == "consumer_staple"
    assert india_archetype_for_sector("technology") == "it_services"
    assert india_archetype_for_sector("energy") == "energy"
    assert india_archetype_for_sector("health_care") == "pharma_healthcare"
    assert india_archetype_for_sector(None) is None


# ----------------------------- India relative-ranking reference -----------------------------
def test_india_reference_relative_rank(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The verdict ranks an India discount against the INDIA cross-section, so a name that is
    deeply discounted in absolute terms can still be the relatively-CHEAPEST among Indian peers."""
    from dcf_project.portfolio import scoring

    ref = {"names": [
        {"ticker": "A", "sector": "technology", "discount": -0.45},   # least negative -> top
        {"ticker": "B", "sector": "consumer_staples", "discount": -0.55},
        {"ticker": "C", "sector": "energy", "discount": -0.80},
        {"ticker": "D", "sector": "consumer_staples", "discount": -0.85},  # most negative -> bottom
    ]}
    d = tmp_path / "research_v3_india"
    d.mkdir()
    (d / "india_v3_reference.json").write_text(json.dumps(ref), encoding="utf-8")
    monkeypatch.setattr(scoring, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(scoring, "_INDIA_REF_CACHE", None)

    iref = scoring.india_reference()
    assert iref is not None and iref["n"] == 4
    # -0.45 is the least-negative -> ranks at/near the top of the India distribution.
    assert scoring.Reference._pct(iref["overall"], -0.45) >= 75.0
    # -0.85 is the most-negative -> bottom.
    assert scoring.Reference._pct(iref["overall"], -0.85) <= 25.0


def test_india_reference_absent_returns_none(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dcf_project.portfolio import scoring

    monkeypatch.setattr(scoring, "REPORT_DIR", tmp_path)        # no research_v3_india/ here
    monkeypatch.setattr(scoring, "_INDIA_REF_CACHE", None)
    assert scoring.india_reference() is None


def test_india_verdict_rank_absolute_discount_gate() -> None:
    """The verdict honours ABSOLUTE value: a top relative rank (er_rank=72) cannot read
    'Attractive' when the name trades far below its own intrinsic (big negative discount)."""
    from dcf_project.portfolio import scoring

    # INFY-like: relatively the 2nd-cheapest India name (er_rank 72) but -45% to its own
    # intrinsic -> gate caps the verdict rank at 39 -> "Unattractive", not "Attractive".
    assert scoring.india_verdict_rank(72.0, -0.45) == 39.0
    assert scoring._label(scoring.india_verdict_rank(72.0, -0.45)) == "Unattractive"
    # IOC/ONGC-like: -22% discount -> capped at Neutral (<60), relative rank preserved otherwise.
    assert scoring.india_verdict_rank(88.0, -0.22) == 59.0
    assert scoring._label(scoring.india_verdict_rank(88.0, -0.22)) == "Neutral"
    # A genuinely undervalued name (discount above -10%) keeps its full relative rank.
    assert scoring.india_verdict_rank(88.0, +0.15) == 88.0
    assert scoring._label(scoring.india_verdict_rank(88.0, +0.15)) == "Very Attractive"
    # The cap never RAISES a rank: an already-low rank stays put.
    assert scoring.india_verdict_rank(12.0, -0.85) == 12.0
    # None discount is inert (no gate).
    assert scoring.india_verdict_rank(72.0, None) == 72.0
