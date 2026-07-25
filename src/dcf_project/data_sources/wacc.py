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
from dcf_project.contracts import MarketSnapshot, WACCInputs
from dcf_project.data_sources.market import build_wacc as build_internal_wacc


VALUEINVESTING_URL = "https://valueinvesting.io/{ticker}/valuation/wacc"


class WACCProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderWACC:
    provider: str
    wacc: float
    cost_of_equity: float | None = None
    cost_of_debt: float | None = None
    beta: float | None = None
    market_risk_premium: float | None = None
    extracted_selected_wacc: float | None = None
    extracted_range_low: float | None = None
    extracted_range_high: float | None = None
    parser_strategy_used: str | None = None
    retrieved_at: datetime | None = None


_WACC_CACHE: dict[tuple[str, str], ProviderWACC] = {}
_WACC_CACHE_LOCK = Lock()


def resolve_wacc(
    ticker: str,
    market: MarketSnapshot,
    tax_rate: float | None,
    debt: float | None,
    settings: Settings,
) -> WACCInputs:
    internal = build_internal_wacc(market, tax_rate, debt, settings)
    if settings.wacc_override is not None:
        return _with_override(internal, settings.wacc_override)
    if settings.wacc_source == "internal":
        internal.source_notes.append("wacc_source=internal")
        internal.source_notes.append("fallback_used=false")
        return internal

    provider_errors: list[str] = []
    providers = [valueinvesting_wacc, finbox_wacc, gurufocus_wacc]
    for provider_func in providers:
        try:
            provider = provider_func(ticker, settings)
            return _provider_to_inputs(provider, internal, market, tax_rate, debt, settings, fallback_used=False)
        except Exception as exc:
            provider_errors.append(f"{provider_func.__name__}: {exc}")
            if settings.wacc_source == "valueinvesting":
                raise WACCProviderError(f"ValueInvesting.io WACC failed for {ticker}: {exc}") from exc

    internal.source_notes.append("wacc_source=internal")
    internal.source_notes.append("provider=internal_capm")
    internal.source_notes.append("fallback_used=true")
    if provider_errors:
        internal.source_notes.append(f"provider_error={'; '.join(provider_errors)[:180]}")
    return internal


def wacc_source_label(wacc: WACCInputs) -> str:
    """Human-readable WACC provenance, derived from ``WACCInputs.source_notes``.

    Used by the report front end (US and India share this provider-first path):
    "ValueInvesting.io" when the provider was used, "Internal CAPM" / "Internal CAPM
    (provider unavailable)" for the CAPM build, or "User override".
    """
    notes = wacc.source_notes

    def value_of(key: str) -> str | None:
        for note in notes:
            if note.startswith(f"{key}="):
                return note.split("=", 1)[1]
        return None

    source = value_of("wacc_source")
    provider = value_of("provider")
    fallback = value_of("fallback_used") == "true"
    if source == "override":
        return "User override"
    if source == "valueinvesting" or (provider and "valueinvesting" in provider.lower()):
        return "ValueInvesting.io"
    if source == "internal" or provider == "internal_capm":
        return "Internal CAPM (provider unavailable)" if fallback else "Internal CAPM"
    return provider or source or "Internal CAPM"


def valueinvesting_wacc(ticker: str, settings: Settings) -> ProviderWACC:
    ticker_upper = ticker.upper()
    key = ("valueinvesting", ticker_upper)
    with _WACC_CACHE_LOCK:
        if key in _WACC_CACHE:
            return _WACC_CACHE[key]
    url = VALUEINVESTING_URL.format(ticker=ticker_upper.lower())
    response = requests.get(
        url,
        headers={
            "User-Agent": settings.sec_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    parsed = parse_valueinvesting_wacc(response.text)
    with _WACC_CACHE_LOCK:
        _WACC_CACHE[key] = parsed
    return parsed


def finbox_wacc(ticker: str, settings: Settings) -> ProviderWACC:
    _ = ticker, settings
    raise WACCProviderError("Finbox WACC adapter is not configured")


def gurufocus_wacc(ticker: str, settings: Settings) -> ProviderWACC:
    _ = ticker, settings
    raise WACCProviderError("GuruFocus WACC adapter is not configured")


def parse_valueinvesting_wacc(html: str) -> ProviderWACC:
    text = _clean_text(html)
    data = _extract_embedded_json(html)
    selected_wacc = _extract_selected_wacc(html, text, data)
    fields = {
        "wacc": selected_wacc["selected"],
        "cost_of_equity": _extract_metric(text, data, ["cost of equity", "cost_of_equity"]),
        "cost_of_debt": _extract_metric(text, data, ["cost of debt", "cost_of_debt"]),
        "beta": _extract_metric(text, data, ["beta"], percent=False),
        "market_risk_premium": _extract_metric(
            text,
            data,
            ["market risk premium", "equity risk premium", "market_risk_premium"],
        ),
    }
    if fields["wacc"] is None:
        raise WACCProviderError("Selected WACC not found in ValueInvesting.io response")
    return ProviderWACC(
        provider="ValueInvesting.io",
        wacc=fields["wacc"],
        cost_of_equity=fields["cost_of_equity"],
        cost_of_debt=fields["cost_of_debt"],
        beta=fields["beta"],
        market_risk_premium=fields["market_risk_premium"],
        extracted_selected_wacc=selected_wacc["selected"],
        extracted_range_low=selected_wacc["range_low"],
        extracted_range_high=selected_wacc["range_high"],
        parser_strategy_used=str(selected_wacc["strategy"]),
        retrieved_at=datetime.now(timezone.utc),
    )


def _provider_to_inputs(
    provider: ProviderWACC,
    internal: WACCInputs,
    market: MarketSnapshot,
    tax_rate: float | None,
    debt: float | None,
    settings: Settings,
    fallback_used: bool,
) -> WACCInputs:
    tax = min(max(tax_rate if tax_rate is not None else settings.default_tax_rate, 0.0), 0.4)
    cost_of_debt = provider.cost_of_debt or internal.pre_tax_cost_of_debt
    cost_of_equity = provider.cost_of_equity or internal.cost_of_equity
    beta = provider.beta or internal.beta
    erp = provider.market_risk_premium or internal.equity_risk_premium
    mv_equity = internal.market_value_equity
    mv_debt = max(debt or market.debt or internal.market_value_debt or 0.0, 0.0)
    total = max(mv_equity + mv_debt, 1.0)
    # Same minimum-equity-premium floor as the internal CAPM path, so a provider
    # value near the risk-free rate cannot explode the terminal multiple.
    floored_wacc = max(provider.wacc, internal.risk_free_rate + 0.5 * settings.equity_risk_premium)
    return WACCInputs(
        risk_free_rate=internal.risk_free_rate,
        beta=beta,
        equity_risk_premium=erp,
        cost_of_equity=cost_of_equity,
        pre_tax_cost_of_debt=cost_of_debt,
        tax_rate=tax,
        market_value_equity=mv_equity,
        market_value_debt=mv_debt,
        equity_weight=mv_equity / total,
        debt_weight=mv_debt / total,
        after_tax_cost_of_debt=cost_of_debt * (1.0 - tax),
        wacc=floored_wacc,
        source_notes=[
            "wacc_source=valueinvesting",
            f"provider={provider.provider}",
            f"fallback_used={str(fallback_used).lower()}",
            f"extracted_selected_wacc={provider.extracted_selected_wacc if provider.extracted_selected_wacc is not None else ''}",
            f"extracted_range_low={provider.extracted_range_low if provider.extracted_range_low is not None else ''}",
            f"extracted_range_high={provider.extracted_range_high if provider.extracted_range_high is not None else ''}",
            f"parser_strategy_used={provider.parser_strategy_used or ''}",
            f"retrieved_at={provider.retrieved_at.isoformat() if provider.retrieved_at else ''}",
        ],
    )


def _with_override(internal: WACCInputs, override: float) -> WACCInputs:
    return internal.model_copy(
        update={
            "wacc": float(override),
            "source_notes": internal.source_notes
            + ["wacc_source=override", "provider=user_override", "fallback_used=false"],
        }
    )


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


def _extract_selected_wacc(html: str, text: str, data: list[Any]) -> dict[str, float | str | None]:
    from_rows = _extract_selected_wacc_from_rows(html)
    if from_rows["selected"] is not None:
        return from_rows
    from_text = _extract_selected_wacc_from_text(text)
    if from_text["selected"] is not None:
        return from_text
    from_json = _extract_selected_wacc_from_json(data)
    if from_json["selected"] is not None:
        return from_json
    return {"selected": None, "range_low": None, "range_high": None, "strategy": "selected_wacc_not_found"}


def _extract_selected_wacc_from_rows(html: str) -> dict[str, float | str | None]:
    rows = _html_table_rows(html)
    for row in rows:
        row_text = " ".join(row)
        if not re.search(r"\bwacc\b|weighted average cost of capital", row_text, flags=re.IGNORECASE):
            continue
        values = _percent_values(row_text)
        if len(values) >= 3:
            return {
                "selected": values[-1],
                "range_low": values[0],
                "range_high": values[1],
                "strategy": "range_selected_table_row",
            }
        if len(row) >= 3:
            range_values = _percent_values(row[-2])
            selected_values = _percent_values(row[-1])
            if len(range_values) >= 2 and selected_values:
                return {
                    "selected": selected_values[-1],
                    "range_low": range_values[0],
                    "range_high": range_values[1],
                    "strategy": "range_selected_table_cells",
                }
    return {"selected": None, "range_low": None, "range_high": None, "strategy": None}


def _extract_selected_wacc_from_text(text: str) -> dict[str, float | str | None]:
    patterns = [
        r"(?:range\s+selected|selected\s+wacc|wacc)[^0-9%]{0,80}"
        r"([0-9]+(?:\.[0-9]+)?\s*%)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?\s*%)"
        r"[^0-9%]{0,80}([0-9]+(?:\.[0-9]+)?\s*%)",
        r"\bwacc\b[^0-9%]{0,80}"
        r"([0-9]+(?:\.[0-9]+)?\s*%)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?\s*%)"
        r"[^0-9%]{0,80}([0-9]+(?:\.[0-9]+)?\s*%)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            low = _parse_number(match.group(1))
            high = _parse_number(match.group(2))
            selected = _parse_number(match.group(3))
            if selected is not None:
                return {
                    "selected": selected,
                    "range_low": low,
                    "range_high": high,
                    "strategy": "range_selected_text",
                }
    return {"selected": None, "range_low": None, "range_high": None, "strategy": None}


def _extract_selected_wacc_from_json(data: list[Any]) -> dict[str, float | str | None]:
    for obj in data:
        found = _find_selected_wacc_json(obj)
        if found["selected"] is not None:
            return found
    return {"selected": None, "range_low": None, "range_high": None, "strategy": None}


def _find_selected_wacc_json(obj: Any) -> dict[str, float | str | None]:
    empty: dict[str, float | str | None] = {
        "selected": None,
        "range_low": None,
        "range_high": None,
        "strategy": None,
    }
    if isinstance(obj, dict):
        keys = {str(key).lower().replace("_", " "): key for key in obj}
        selected_key = next(
            (
                original
                for normalized, original in keys.items()
                if "wacc" in normalized and ("selected" in normalized or "default" in normalized)
            ),
            None,
        )
        if selected_key is not None:
            selected = _parse_number(obj[selected_key])
            low = _parse_number(obj[keys["wacc min"]]) if "wacc min" in keys else None
            high = _parse_number(obj[keys["wacc max"]]) if "wacc max" in keys else None
            return {
                "selected": selected,
                "range_low": low,
                "range_high": high,
                "strategy": "selected_wacc_json",
            }
        for value in obj.values():
            found = _find_selected_wacc_json(value)
            if found["selected"] is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_selected_wacc_json(value)
            if found["selected"] is not None:
                return found
    return empty


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
        else:
            cleaned = _clean_text(row_html)
            if cleaned:
                rows.append([cleaned])
    return rows


def _percent_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"-?[0-9]+(?:\.[0-9]+)?\s*%", text):
        parsed = _parse_number(match.group(0))
        if parsed is not None:
            values.append(parsed)
    return values


def _extract_metric(text: str, data: list[Any], labels: list[str], percent: bool = True) -> float | None:
    for obj in data:
        found = _find_in_json(obj, labels, percent)
        if found is not None:
            return found
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?\s*%?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _parse_number(match.group(1), percent)
        loose = rf"{re.escape(label)}[^0-9%]{{0,80}}([0-9]+(?:\.[0-9]+)?\s*%?)"
        match = re.search(loose, text, flags=re.IGNORECASE)
        if match:
            return _parse_number(match.group(1), percent)
    return None


def _find_in_json(obj: Any, labels: list[str], percent: bool) -> float | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = str(key).lower().replace("_", " ")
            if any(label.replace("_", " ") in key_norm for label in labels):
                parsed = _parse_number(value, percent)
                if parsed is not None:
                    return parsed
            found = _find_in_json(value, labels, percent)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_in_json(value, labels, percent)
            if found is not None:
                return found
    return None


def _parse_number(value: object, percent: bool = True) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric / 100.0 if percent and numeric > 1.0 and numeric <= 100.0 else numeric
    text = str(value).strip()
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", text)
    if not match:
        return None
    numeric = float(match.group(0))
    if percent and ("%" in text or numeric > 1.0):
        numeric /= 100.0
    return numeric
