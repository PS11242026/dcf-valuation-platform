from __future__ import annotations

import pytest

from dcf_project.config import get_settings
from dcf_project.data_sources import growth
from dcf_project.data_sources.growth import (
    TerminalGrowthAssumption,
    TerminalGrowthProviderError,
    parse_valueinvesting_terminal_growth,
    resolve_terminal_growth,
)


def _settings(**updates):
    settings = get_settings()
    return settings.__class__(**{**settings.__dict__, **updates})


def test_terminal_growth_parse() -> None:
    html = """
    <html><body>
      <table>
        <tr><th>Assumption</th><th>Range</th><th>Selected</th></tr>
        <tr><td>Terminal Growth</td><td>2.0% - 3.5%</td><td>2.8%</td></tr>
        <tr><td>Long-Term Growth</td><td>2.0% - 3.0%</td><td>2.5%</td></tr>
      </table>
    </body></html>
    """

    parsed = parse_valueinvesting_terminal_growth(html)

    assert parsed.provider == "ValueInvesting.io"
    assert parsed.terminal_growth == pytest.approx(0.028)
    assert parsed.long_term_growth == pytest.approx(0.025)
    assert parsed.provider_terminal_growth == pytest.approx(0.028)
    assert parsed.provider_long_term_growth == pytest.approx(0.025)
    assert parsed.terminal_growth_source_type == "provider_terminal"
    assert parsed.parser_strategy == "terminal_growth_range_selected_table_row"
    assert parsed.terminal_growth_source == "valueinvesting"


def test_long_term_growth_is_not_terminal_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = TerminalGrowthAssumption(
        provider="ValueInvesting.io",
        terminal_growth=None,
        long_term_growth=0.04,
        parser_strategy="long_term_growth_range_selected_table_row",
        terminal_growth_source="valueinvesting",
        terminal_growth_source_type="provider_long_term_only",
        provider_long_term_growth=0.04,
        raw_parsed_values={"terminal_growth": None, "long_term_growth": 0.04},
    )

    monkeypatch.setattr(growth, "valueinvesting_terminal_growth", lambda ticker, settings: provider)

    resolved = resolve_terminal_growth("AAPL", 0.09, _settings(terminal_growth_source="auto"))

    assert resolved.terminal_growth == pytest.approx(0.0275)
    assert resolved.terminal_growth != pytest.approx(0.04)
    assert resolved.provider_long_term_growth == pytest.approx(0.04)
    assert resolved.provider_terminal_growth is None
    assert resolved.terminal_growth_source_type == "provider_long_term_only"
    assert resolved.fallback_used is True


def test_provider_long_term_only_does_not_crash_valueinvesting_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = TerminalGrowthAssumption(
        provider="ValueInvesting.io",
        terminal_growth=None,
        long_term_growth=0.04,
        parser_strategy="long_term_growth_only_not_terminal",
        terminal_growth_source="valueinvesting",
        terminal_growth_source_type="provider_long_term_only",
        provider_long_term_growth=0.04,
        raw_parsed_values={"terminal_growth": None, "long_term_growth": 0.04},
    )

    monkeypatch.setattr(growth, "valueinvesting_terminal_growth", lambda ticker, settings: provider)

    resolved = resolve_terminal_growth("AAPL", 0.09, _settings(terminal_growth_source="valueinvesting"))

    assert resolved.terminal_growth == pytest.approx(0.0275)
    assert resolved.provider_long_term_growth == pytest.approx(0.04)
    assert resolved.provider_terminal_growth is None
    assert resolved.fallback_used is True


def test_strict_provider_terminal_growth_raises_on_long_term_only(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = TerminalGrowthAssumption(
        provider="ValueInvesting.io",
        terminal_growth=None,
        long_term_growth=0.04,
        parser_strategy="long_term_growth_only_not_terminal",
        terminal_growth_source="valueinvesting",
        terminal_growth_source_type="provider_long_term_only",
        provider_long_term_growth=0.04,
        raw_parsed_values={"terminal_growth": None, "long_term_growth": 0.04},
    )

    monkeypatch.setattr(growth, "valueinvesting_terminal_growth", lambda ticker, settings: provider)

    with pytest.raises(TerminalGrowthProviderError, match="no true perpetual terminal growth"):
        resolve_terminal_growth(
            "AAPL",
            0.09,
            _settings(terminal_growth_source="valueinvesting", strict_provider_terminal_growth=True),
        )


def test_provider_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_provider(ticker: str, settings: object) -> TerminalGrowthAssumption:
        raise TerminalGrowthProviderError(f"missing provider data for {ticker}")

    monkeypatch.setattr(growth, "valueinvesting_terminal_growth", fail_provider)

    resolved = resolve_terminal_growth("AAPL", 0.09, _settings(terminal_growth_source="auto"))

    assert resolved.provider == "internal_policy"
    assert resolved.terminal_growth_source == "internal"
    assert resolved.fallback_used is True
    assert resolved.provider_error is not None


def test_manual_override() -> None:
    resolved = resolve_terminal_growth(
        "AAPL",
        0.09,
        _settings(terminal_growth_source="auto", terminal_growth_override=0.03),
    )

    assert resolved.provider == "user_override"
    assert resolved.terminal_growth == pytest.approx(0.03)
    assert resolved.terminal_growth_source == "override"
    assert resolved.fallback_used is False


def test_internal_policy_fallback() -> None:
    resolved = resolve_terminal_growth(
        "AAPL",
        0.09,
        _settings(terminal_growth_source="internal", company_type="asset_light"),
    )

    assert resolved.provider == "internal_policy"
    assert resolved.terminal_growth == pytest.approx(0.0275)
    assert resolved.terminal_growth_source == "internal"
    assert resolved.fallback_used is False


def test_forced_provider_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_provider(ticker: str, settings: object) -> TerminalGrowthAssumption:
        raise TerminalGrowthProviderError(f"missing provider data for {ticker}")

    monkeypatch.setattr(growth, "valueinvesting_terminal_growth", fail_provider)

    with pytest.raises(TerminalGrowthProviderError, match="ValueInvesting.io terminal growth failed"):
        resolve_terminal_growth(
            "AAPL",
            0.09,
            _settings(terminal_growth_source="valueinvesting", strict_provider_terminal_growth=True),
        )
