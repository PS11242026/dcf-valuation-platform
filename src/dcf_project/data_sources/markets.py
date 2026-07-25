"""Country/market resolution for the multi-market bolt-on.

The product is US-only by default (SEC filings + US Treasury risk-free + US macro
anchors). India is an additive bolt-on selected per ticker: a Yahoo-style ``.NS`` / ``.BO``
suffix (NSE / BSE) routes to the India data + macro path, and an explicit ``--country``
flag always wins. Anything unrecognised stays ``US`` so existing behaviour is unchanged.

This module is intentionally tiny and dependency-free: it is the single place that decides
"is this an Indian name?", consumed once at the valuation entry point (see
``valuation.canonical.canonical_valuation``).
"""
from __future__ import annotations

from typing import Literal

Country = Literal["US", "IN", "CA"]

#: Yahoo Finance exchange suffixes that denote an Indian listing (NSE / BSE).
_INDIA_SUFFIXES = (".NS", ".BO")
#: Yahoo Finance exchange suffixes that denote a Canadian listing (TSX / TSXV).
_CANADA_SUFFIXES = (".TO", ".V")

_SUPPORTED = {"US", "IN", "CA"}


def is_indian_ticker(ticker: str) -> bool:
    """True if ``ticker`` carries an NSE/BSE (``.NS`` / ``.BO``) suffix."""
    return ticker.upper().endswith(_INDIA_SUFFIXES)


def is_canadian_ticker(ticker: str) -> bool:
    """True if ``ticker`` carries a TSX/TSXV (``.TO`` / ``.V``) suffix."""
    return ticker.upper().endswith(_CANADA_SUFFIXES)


def resolve_country(ticker: str, explicit: str | None = None) -> Country:
    """Resolve the market for ``ticker``.

    Precedence: an explicit ``US``/``IN``/``CA`` wins; ``auto``/``None``/unknown falls back to
    inference from the ticker suffix (``.NS``/``.BO`` -> IN; ``.TO``/``.V`` -> CA), else ``US``.
    """
    if explicit:
        flag = explicit.strip().upper()
        if flag in _SUPPORTED:
            return flag  # type: ignore[return-value]
    if is_indian_ticker(ticker):
        return "IN"
    if is_canadian_ticker(ticker):
        return "CA"
    return "US"
