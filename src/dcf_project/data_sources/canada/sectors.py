"""Canadian (TSX) large-cap universe + GICS sector map for the Canada bolt-on.

Scope: TSX-listed large caps that are **SEC filers** (so EDGAR ``companyfacts`` carries their
point-in-time fundamentals) and for which a discounted-cash-flow model is applicable (banks,
insurers, asset managers and REITs are excluded — financials are not DCF-valued anywhere in
this project). Tickers are the **EDGAR / US-listing** symbols (the SEC keys facts by these).

``CCY_HINT_CA`` records each name's reporting currency from public filings as a *fallback*;
the backtest detects the authoritative currency live from the EDGAR unit tag and only falls
back to this map if detection is unavailable. CAD reporters are priced on the TSX (``.TO``);
USD reporters (global operators that report in USD) are priced on their US listing.
"""
from __future__ import annotations

#: EDGAR/US-listing ticker -> canonical sector label (shared with the India/US sector vocab).
TICKER_SECTOR_CA: dict[str, str] = {
    # Energy (oil sands, pipelines, integrated) -- the core of the TSX, CAD reporters.
    "SU": "energy", "CNQ": "energy", "ENB": "energy", "TRP": "energy", "CVE": "energy",
    "IMO": "energy", "PBA": "energy",
    # Materials (gold/precious, base metals, fertilizer, uranium, chemicals).
    "GOLD": "materials", "AEM": "materials", "NTR": "materials", "TECK": "materials",
    "FNV": "materials", "WPM": "materials", "CCJ": "materials", "KGC": "materials",
    "PAAS": "materials", "AGI": "materials", "BTG": "materials", "HBM": "materials",
    "MEOH": "materials", "NGD": "materials",
    # Industrials (rail, waste, info-services, capital goods, engineering).
    "CNI": "industrials", "CP": "industrials", "WCN": "industrials", "TRI": "industrials",
    "RBA": "industrials", "GIB": "industrials", "STN": "industrials",
    # Technology (e-commerce, enterprise software, logistics SaaS).
    "SHOP": "technology", "OTEX": "technology", "DSGX": "technology",
    # Consumer discretionary (QSR franchising, auto parts, apparel, powersports).
    "QSR": "consumer_discretionary", "MGA": "consumer_discretionary",
    "GIL": "consumer_discretionary", "DOO": "consumer_discretionary",
    # Consumer staples (convenience retail).
    "ATD": "consumer_staples",
    # Utilities / renewables / infrastructure.
    "AQN": "utilities", "BEP": "utilities", "BIP": "utilities", "FTS": "utilities",
    # Communication services (telecom).
    "BCE": "communication_services", "RCI": "communication_services", "TU": "communication_services",
    # Health care.
    "BHC": "health_care",
}

#: Reporting-currency fallback (authoritative source is the live EDGAR unit tag).
CCY_HINT_CA: dict[str, str] = {
    "SU": "CAD", "CNQ": "CAD", "ENB": "CAD", "TRP": "CAD", "CVE": "CAD", "IMO": "CAD", "PBA": "CAD",
    "GOLD": "USD", "AEM": "USD", "NTR": "USD", "TECK": "CAD", "FNV": "USD", "WPM": "USD",
    "CCJ": "CAD", "KGC": "USD", "PAAS": "USD", "AGI": "USD", "BTG": "USD", "HBM": "USD",
    "MEOH": "USD", "NGD": "USD",
    "CNI": "CAD", "CP": "CAD", "WCN": "USD", "TRI": "USD", "RBA": "USD", "GIB": "CAD", "STN": "CAD",
    "SHOP": "USD", "OTEX": "USD", "DSGX": "USD",
    "QSR": "USD", "MGA": "USD", "GIL": "USD", "DOO": "CAD",
    "ATD": "USD",
    "AQN": "USD", "BEP": "USD", "BIP": "USD", "FTS": "CAD",
    "BCE": "CAD", "RCI": "CAD", "TU": "CAD",
    "BHC": "USD",
}


def canada_sector_for_ticker(ticker: str) -> str:
    """Canonical sector label for a Canadian ticker (``unknown`` if unmapped)."""
    return TICKER_SECTOR_CA.get(ticker.upper().removesuffix(".TO").removesuffix(".V"), "unknown")


def currency_hint(ticker: str) -> str:
    """Fallback reporting currency (``CAD``/``USD``) when live detection is unavailable."""
    return CCY_HINT_CA.get(ticker.upper().removesuffix(".TO").removesuffix(".V"), "USD")
