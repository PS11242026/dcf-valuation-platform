from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
import json
import re
from threading import Lock
from typing import Any

import requests

from dcf_project.config import Settings
from dcf_project.models.priors import prior_for


VALUEINVESTING_GROWTH_URLS = (
    "https://valueinvesting.io/{ticker}/valuation/intrinsic-value",
    "https://valueinvesting.io/{ticker}/valuation/fair-value",
    "https://valueinvesting.io/{ticker}/valuation/dcf-growth-exit-5y",
)


class TerminalGrowthProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminalGrowthAssumption:
    provider: str
    terminal_growth: float | None
    long_term_growth: float | None = None
    retrieved_at: datetime | None = None
    parser_strategy: str | None = None
    terminal_growth_source: str = "internal"
    fallback_used: bool = False
    provider_error: str | None = None
    raw_parsed_values: dict[str, float | None] | None = None
    terminal_growth_source_type: str = "internal_policy"
    provider_terminal_growth: float | None = None
    provider_long_term_growth: float | None = None


_GROWTH_CACHE: dict[tuple[str, str], TerminalGrowthAssumption] = {}
_GROWTH_CACHE_LOCK = Lock()


def resolve_terminal_growth(
    ticker: str,
    annual_wacc: float,
    settings: Settings,
) -> TerminalGrowthAssumption:
    if settings.terminal_growth_override is not None:
        return _manual_override_terminal_growth(settings.terminal_growth_override, annual_wacc)
    if settings.terminal_growth_source == "internal":
        return internal_terminal_growth(ticker, annual_wacc, settings, fallback_used=False)

    provider_errors: list[str] = []
    try:
        provider = valueinvesting_terminal_growth(ticker, settings)
        if provider.terminal_growth is None:
            if settings.strict_provider_terminal_growth:
                raise TerminalGrowthProviderError(
                    "Provider returned long-term growth but no true perpetual terminal growth; "
                    f"provider_long_term_growth={provider.provider_long_term_growth}"
                )
            return _provider_long_term_fallback(ticker, annual_wacc, settings, provider)
        return _with_growth_guardrail(provider, annual_wacc)
    except Exception as exc:
        provider_errors.append(f"valueinvesting_terminal_growth: {exc}")
        if settings.terminal_growth_source == "valueinvesting" and settings.strict_provider_terminal_growth:
            raise TerminalGrowthProviderError(
                f"ValueInvesting.io terminal growth failed for {ticker}: {exc}"
            ) from exc

    fallback = internal_terminal_growth(ticker, annual_wacc, settings, fallback_used=True)
    return TerminalGrowthAssumption(
        provider=fallback.provider,
        terminal_growth=fallback.terminal_growth,
        long_term_growth=provider_long_term_from_errors(provider_errors) or fallback.long_term_growth,
        retrieved_at=fallback.retrieved_at,
        parser_strategy=fallback.parser_strategy,
        terminal_growth_source=fallback.terminal_growth_source,
        fallback_used=True,
        provider_error="; ".join(provider_errors)[:180] if provider_errors else None,
        raw_parsed_values=fallback.raw_parsed_values,
        terminal_growth_source_type="internal_policy_provider_long_term_ignored",
        provider_long_term_growth=provider_long_term_from_errors(provider_errors),
        provider_terminal_growth=None,
    )


def _provider_long_term_fallback(
    ticker: str,
    annual_wacc: float,
    settings: Settings,
    provider: TerminalGrowthAssumption,
) -> TerminalGrowthAssumption:
    fallback = internal_terminal_growth(ticker, annual_wacc, settings, fallback_used=True)
    return TerminalGrowthAssumption(
        provider=fallback.provider,
        terminal_growth=fallback.terminal_growth,
        long_term_growth=provider.long_term_growth,
        retrieved_at=fallback.retrieved_at,
        parser_strategy=fallback.parser_strategy,
        terminal_growth_source=fallback.terminal_growth_source,
        fallback_used=True,
        provider_error=None,
        raw_parsed_values={
            **(fallback.raw_parsed_values or {}),
            "provider_terminal_growth": provider.provider_terminal_growth,
            "provider_long_term_growth": provider.provider_long_term_growth,
        },
        terminal_growth_source_type=provider.terminal_growth_source_type,
        provider_long_term_growth=provider.provider_long_term_growth,
        provider_terminal_growth=provider.provider_terminal_growth,
    )


def valueinvesting_terminal_growth(ticker: str, settings: Settings) -> TerminalGrowthAssumption:
    ticker_upper = ticker.upper()
    key = ("valueinvesting", ticker_upper)
    with _GROWTH_CACHE_LOCK:
        if key in _GROWTH_CACHE:
            return _GROWTH_CACHE[key]

    errors: list[str] = []
    for template in VALUEINVESTING_GROWTH_URLS:
        url = template.format(ticker=ticker_upper.lower())
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": settings.sec_user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                timeout=settings.request_timeout_seconds,
            )
            response.raise_for_status()
            parsed = parse_valueinvesting_terminal_growth(response.text)
            with _GROWTH_CACHE_LOCK:
                _GROWTH_CACHE[key] = parsed
            return parsed
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise TerminalGrowthProviderError("; ".join(errors)[:240])


def parse_valueinvesting_terminal_growth(html: str) -> TerminalGrowthAssumption:
    text = _clean_text(html)
    data = _extract_embedded_json(html)
    terminal = _extract_growth_assumption(
        html,
        text,
        data,
        ["terminal growth", "terminal growth rate", "perpetuity growth", "perpetual growth"],
        "terminal_growth",
    )
    long_term = _extract_growth_assumption(
        html,
        text,
        data,
        ["long term growth", "long-term growth", "long term growth rate", "long-term growth rate"],
        "long_term_growth",
    )
    terminal_growth = terminal["selected"]
    if terminal_growth is None and long_term["selected"] is None:
        raise TerminalGrowthProviderError("Terminal growth not found in ValueInvesting.io response")
    strategy = terminal["strategy"] if terminal["selected"] is not None else "long_term_growth_only_not_terminal"
    return TerminalGrowthAssumption(
        provider="ValueInvesting.io",
        terminal_growth=float(terminal_growth) if terminal_growth is not None else None,
        long_term_growth=float(long_term["selected"]) if long_term["selected"] is not None else None,
        retrieved_at=datetime.now(timezone.utc),
        parser_strategy=str(strategy),
        terminal_growth_source="valueinvesting",
        fallback_used=False,
        raw_parsed_values={
            "terminal_growth": float(terminal["selected"]) if terminal["selected"] is not None else None,
            "long_term_growth": float(long_term["selected"]) if long_term["selected"] is not None else None,
        },
        terminal_growth_source_type="provider_terminal" if terminal_growth is not None else "provider_long_term_only",
        provider_terminal_growth=float(terminal_growth) if terminal_growth is not None else None,
        provider_long_term_growth=float(long_term["selected"]) if long_term["selected"] is not None else None,
    )


def internal_terminal_growth(
    ticker: str,
    annual_wacc: float,
    settings: Settings,
    fallback_used: bool,
) -> TerminalGrowthAssumption:
    policy_growth, policy_floor, policy_cap = _institutional_terminal_growth_policy(ticker, settings.company_type)
    archetype = prior_for(ticker, settings.company_type).archetype
    configured = settings.default_terminal_growth if settings.default_terminal_growth != 0.025 else policy_growth
    raw = min(configured, policy_cap, settings.terminal_growth_cap)
    bounded = _bound_terminal_growth(raw, annual_wacc, policy_floor, policy_cap, settings.terminal_growth_cap)
    return TerminalGrowthAssumption(
        provider="internal_policy",
        terminal_growth=bounded,
        long_term_growth=policy_growth,
        retrieved_at=datetime.now(timezone.utc),
        parser_strategy=(
            "institutional_company_type_policy:"
            f"{archetype}:floor={policy_floor}:cap={policy_cap}:spread_guardrail_bps=150"
        ),
        terminal_growth_source="internal",
        fallback_used=fallback_used,
        raw_parsed_values={"internal_policy_growth": policy_growth},
        terminal_growth_source_type="internal_policy",
        provider_terminal_growth=None,
        provider_long_term_growth=None,
    )


def _manual_override_terminal_growth(override: float, annual_wacc: float) -> TerminalGrowthAssumption:
    bounded = min(float(override), max(annual_wacc - 0.015, 0.0))
    return TerminalGrowthAssumption(
        provider="user_override",
        terminal_growth=bounded,
        long_term_growth=float(override),
        retrieved_at=datetime.now(timezone.utc),
        parser_strategy="manual_override_with_wacc_spread_guardrail",
        terminal_growth_source="override",
        fallback_used=False,
        raw_parsed_values={"manual_override": float(override)},
        terminal_growth_source_type="manual_override",
        provider_terminal_growth=None,
        provider_long_term_growth=None,
    )


def _with_growth_guardrail(
    assumption: TerminalGrowthAssumption,
    annual_wacc: float,
) -> TerminalGrowthAssumption:
    if assumption.terminal_growth is None:
        raise TerminalGrowthProviderError("Cannot apply guardrail without a provider terminal growth")
    bounded = min(assumption.terminal_growth, max(annual_wacc - 0.015, 0.0))
    return TerminalGrowthAssumption(
        provider=assumption.provider,
        terminal_growth=bounded,
        long_term_growth=assumption.long_term_growth,
        retrieved_at=assumption.retrieved_at,
        parser_strategy=assumption.parser_strategy,
        terminal_growth_source=assumption.terminal_growth_source,
        fallback_used=assumption.fallback_used,
        provider_error=assumption.provider_error,
        raw_parsed_values=assumption.raw_parsed_values,
        terminal_growth_source_type=assumption.terminal_growth_source_type,
        provider_terminal_growth=assumption.provider_terminal_growth,
        provider_long_term_growth=assumption.provider_long_term_growth,
    )


def provider_long_term_from_errors(errors: list[str]) -> float | None:
    for error in errors:
        match = re.search(r"provider_long_term_growth=([0-9.]+)", error)
        if match:
            return float(match.group(1))
    return None


def _institutional_terminal_growth_policy(ticker: str, company_type: str) -> tuple[float, float, float]:
    prior = prior_for(ticker, company_type)
    if prior.archetype in {"aapl", "msft"}:
        return prior.terminal_growth_policy, prior.terminal_growth_floor, prior.terminal_growth_cap
    if prior.archetype in {"googl", "meta"}:
        return prior.terminal_growth_policy, prior.terminal_growth_floor, prior.terminal_growth_cap
    if prior.archetype == "capital_intensive_growth":
        return prior.terminal_growth_policy, prior.terminal_growth_floor, prior.terminal_growth_cap
    if company_type in {"consumer_staple", "consumer_compounder"}:
        return 0.0225, 0.0200, 0.0250
    if company_type in {"asset_light_quality", "platform"}:
        return 0.0275, 0.0250, 0.0300
    if company_type in {"hyperscaler", "marketplace_hybrid"}:
        return 0.0300, 0.0200, 0.0300
    if company_type in {"megacap", "asset_light"}:
        return 0.0250, 0.0200, 0.0250
    return 0.0250, 0.0150, 0.0300


def _bound_terminal_growth(
    growth: float,
    annual_wacc: float,
    floor: float,
    policy_cap: float,
    configured_cap: float,
) -> float:
    bounded = min(max(float(growth), floor), policy_cap, configured_cap)
    return float(min(bounded, max(annual_wacc - 0.015, 0.0)))


def _extract_growth_assumption(
    html: str,
    text: str,
    data: list[Any],
    labels: list[str],
    field_name: str,
) -> dict[str, float | str | None]:
    from_rows = _extract_selected_growth_from_rows(html, labels, field_name)
    if from_rows["selected"] is not None:
        return from_rows
    from_json = _extract_selected_growth_from_json(data, labels, field_name)
    if from_json["selected"] is not None:
        return from_json
    from_text = _extract_selected_growth_from_text(text, labels, field_name)
    if from_text["selected"] is not None:
        return from_text
    return {"selected": None, "strategy": f"{field_name}_not_found"}


def _extract_selected_growth_from_rows(
    html: str,
    labels: list[str],
    field_name: str,
) -> dict[str, float | str | None]:
    rows = _html_table_rows(html)
    for row in rows:
        row_text = " ".join(row)
        if not _matches_any_label(row_text, labels):
            continue
        values = _percent_values(row_text)
        if len(values) >= 3:
            return {"selected": values[-1], "strategy": f"{field_name}_range_selected_table_row"}
        if len(row) >= 2:
            selected_values = _percent_values(row[-1])
            if selected_values:
                return {"selected": selected_values[-1], "strategy": f"{field_name}_selected_table_cell"}
    return {"selected": None, "strategy": None}


def _extract_selected_growth_from_text(
    text: str,
    labels: list[str],
    field_name: str,
) -> dict[str, float | str | None]:
    for label in labels:
        range_pattern = (
            rf"{re.escape(label)}[^0-9%]{{0,80}}"
            r"([0-9]+(?:\.[0-9]+)?\s*%)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?\s*%)"
            r"[^0-9%]{0,80}([0-9]+(?:\.[0-9]+)?\s*%)"
        )
        match = re.search(range_pattern, text, flags=re.IGNORECASE)
        if match:
            selected = _parse_number(match.group(3))
            if selected is not None:
                return {"selected": selected, "strategy": f"{field_name}_range_selected_text"}
        selected_pattern = rf"{re.escape(label)}[^0-9%]{{0,80}}([0-9]+(?:\.[0-9]+)?\s*%)"
        match = re.search(selected_pattern, text, flags=re.IGNORECASE)
        if match:
            selected = _parse_number(match.group(1))
            if selected is not None:
                return {"selected": selected, "strategy": f"{field_name}_single_value_text"}
    return {"selected": None, "strategy": None}


def _extract_selected_growth_from_json(
    data: list[Any],
    labels: list[str],
    field_name: str,
) -> dict[str, float | str | None]:
    for obj in data:
        found = _find_growth_json(obj, labels, field_name)
        if found["selected"] is not None:
            return found
    return {"selected": None, "strategy": None}


def _find_growth_json(obj: Any, labels: list[str], field_name: str) -> dict[str, float | str | None]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = str(key).lower().replace("_", " ")
            if any(label in normalized for label in labels):
                parsed = _parse_number(value)
                if parsed is not None:
                    return {"selected": parsed, "strategy": f"{field_name}_json"}
            found = _find_growth_json(value, labels, field_name)
            if found["selected"] is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_growth_json(value, labels, field_name)
            if found["selected"] is not None:
                return found
    return {"selected": None, "strategy": None}


def _matches_any_label(text: str, labels: list[str]) -> bool:
    normalized = text.lower().replace("_", " ")
    return any(label in normalized for label in labels)


def _html_table_rows(html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        row_html = row_match.group(1)
        cells = [
            _clean_text(cell_match.group(1))
            for cell_match in re.finditer(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
        ]
        if cells:
            rows.append(cells)
    return rows


def _percent_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"-?[0-9]+(?:\.[0-9]+)?\s*%", text):
        parsed = _parse_number(match.group(0))
        if parsed is not None:
            values.append(parsed)
    return values


def _clean_text(html: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_embedded_json(html: str) -> list[Any]:
    objects: list[Any] = []
    for match in re.finditer(
        r"<script[^>]+type=[\"']application/(?:ld\+)?json[\"'][^>]*>(.*?)</script>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        try:
            objects.append(json.loads(match.group(1)))
        except Exception:
            continue
    for match in re.finditer(r"__NEXT_DATA__[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL):
        try:
            objects.append(json.loads(match.group(1)))
        except Exception:
            continue
    return objects


def _parse_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric / 100.0 if numeric > 1.0 and numeric <= 100.0 else numeric
    text = str(value).strip()
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", text)
    if not match:
        return None
    numeric = float(match.group(0))
    if "%" in text or numeric > 1.0:
        numeric /= 100.0
    return numeric
