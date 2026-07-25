from __future__ import annotations

import pytest

from dcf_project.config import get_settings
from dcf_project.contracts import MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.data_sources import wacc
from dcf_project.data_sources.wacc import (
    WACCProviderError,
    parse_valueinvesting_wacc,
    resolve_wacc,
    wacc_source_label,
)


def _wacc_inputs(source_notes: list[str]) -> WACCInputs:
    return WACCInputs(
        risk_free_rate=0.04, beta=1.0, equity_risk_premium=0.045, cost_of_equity=0.085,
        pre_tax_cost_of_debt=0.045, tax_rate=0.21, market_value_equity=1.0, market_value_debt=0.0,
        equity_weight=1.0, debt_weight=0.0, after_tax_cost_of_debt=0.035, wacc=0.085,
        source_notes=source_notes,
    )


def test_wacc_source_label_maps_provenance() -> None:
    assert wacc_source_label(_wacc_inputs(
        ["wacc_source=valueinvesting", "provider=ValueInvesting.io", "fallback_used=false"])) == "ValueInvesting.io"
    assert wacc_source_label(_wacc_inputs(
        ["wacc_source=internal", "provider=internal_capm", "fallback_used=true"])) == "Internal CAPM (provider unavailable)"
    assert wacc_source_label(_wacc_inputs(
        ["wacc_source=internal", "fallback_used=false"])) == "Internal CAPM"
    assert wacc_source_label(_wacc_inputs(
        ["wacc_source=override", "provider=user_override"])) == "User override"


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        ticker="MSFT",
        price=420.0,
        market_cap=3_100_000_000_000.0,
        shares_outstanding=7_400_000_000.0,
        beta=0.95,
        cash=80_000_000_000.0,
        debt=100_000_000_000.0,
        currency="USD",
        source=SourceMeta(provider="test", key="MSFT"),
    )


def test_valueinvesting_wacc_parse() -> None:
    html = """
    <html><body>
      <table>
        <thead><tr><th>Metric</th><th>Range</th><th>Selected</th></tr></thead>
        <tbody>
          <tr><td>WACC</td><td>6.4% - 9.1%</td><td>7.8%</td></tr>
        </tbody>
      </table>
      <section>
        Cost of Equity 7.1%
        Cost of Debt 4.2%
        Beta 0.92
        Market Risk Premium 4.8%
      </section>
    </body></html>
    """

    parsed = parse_valueinvesting_wacc(html)

    assert parsed.provider == "ValueInvesting.io"
    assert parsed.wacc == pytest.approx(0.078)
    assert parsed.extracted_selected_wacc == pytest.approx(0.078)
    assert parsed.extracted_range_low == pytest.approx(0.064)
    assert parsed.extracted_range_high == pytest.approx(0.091)
    assert parsed.parser_strategy_used == "range_selected_table_row"
    assert parsed.cost_of_equity == pytest.approx(0.071)
    assert parsed.cost_of_debt == pytest.approx(0.042)
    assert parsed.beta == pytest.approx(0.92)
    assert parsed.market_risk_premium == pytest.approx(0.048)


def test_valueinvesting_msft_selected_wacc_regression() -> None:
    html = """
    <html><body>
      <table>
        <tr><th>Assumption</th><th>Range</th><th>Selected</th></tr>
        <tr><td>WACC</td><td>7.0% - 10.2%</td><td>8.6%</td></tr>
      </table>
    </body></html>
    """

    parsed = parse_valueinvesting_wacc(html)

    assert parsed.wacc == pytest.approx(0.086)
    assert parsed.wacc != pytest.approx(0.07)
    assert parsed.extracted_range_low == pytest.approx(0.07)
    assert parsed.extracted_range_high == pytest.approx(0.102)
    assert parsed.extracted_selected_wacc == pytest.approx(0.086)


def test_valueinvesting_aapl_selected_wacc_regression() -> None:
    html = """
    <html><body>
      <table>
        <tr><th>Assumption</th><th>Range</th><th>Selected</th></tr>
        <tr><td>WACC</td><td>6.8% - 9.8%</td><td>8.2%</td></tr>
      </table>
    </body></html>
    """

    parsed = parse_valueinvesting_wacc(html)

    assert parsed.wacc == pytest.approx(0.082)
    assert parsed.wacc != pytest.approx(0.068)
    assert parsed.extracted_range_low == pytest.approx(0.068)
    assert parsed.extracted_range_high == pytest.approx(0.098)
    assert parsed.extracted_selected_wacc == pytest.approx(0.082)


def test_valueinvesting_range_low_without_selected_raises() -> None:
    html = """
    <html><body>
      <section>WACC 7.0% - 10.2%</section>
    </body></html>
    """

    with pytest.raises(WACCProviderError, match="Selected WACC"):
        parse_valueinvesting_wacc(html)


def test_provider_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()

    def fail_provider(ticker: str, settings: object) -> wacc.ProviderWACC:
        raise WACCProviderError(f"missing provider data for {ticker}")

    monkeypatch.setattr(wacc, "valueinvesting_wacc", fail_provider)
    monkeypatch.setattr(wacc, "finbox_wacc", fail_provider)
    monkeypatch.setattr(wacc, "gurufocus_wacc", fail_provider)

    resolved = resolve_wacc("MSFT", _market(), 0.18, 100_000_000_000.0, settings)

    assert resolved.wacc > 0
    assert "wacc_source=internal" in resolved.source_notes
    assert "provider=internal_capm" in resolved.source_notes
    assert "fallback_used=true" in resolved.source_notes


def test_internal_fallback_skips_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    settings = settings.__class__(**{**settings.__dict__, "wacc_source": "internal"})

    def unexpected_provider(ticker: str, settings: object) -> wacc.ProviderWACC:
        raise AssertionError("provider should not be called in internal mode")

    monkeypatch.setattr(wacc, "valueinvesting_wacc", unexpected_provider)

    resolved = resolve_wacc("MSFT", _market(), 0.18, 100_000_000_000.0, settings)

    assert resolved.wacc > 0
    assert "wacc_source=internal" in resolved.source_notes
    assert "fallback_used=false" in resolved.source_notes


def test_wacc_override() -> None:
    settings = get_settings()
    settings = settings.__class__(**{**settings.__dict__, "wacc_override": 0.065})

    resolved = resolve_wacc("MSFT", _market(), 0.18, 100_000_000_000.0, settings)

    assert resolved.wacc == pytest.approx(0.065)
    assert "wacc_source=override" in resolved.source_notes
    assert "provider=user_override" in resolved.source_notes
    assert "fallback_used=false" in resolved.source_notes
