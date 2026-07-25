"""Ticker -> sector -> SPDR sector-ETF mapping for the P4B Sector-State Engine (engine_v2).

Sector membership for large, established issuers is *stable* over our 2015-2026 backtest
window, so a curated static map (mirroring ``models.classification.TICKER_COMPANY_TYPES``) is
both PIT-safe and avoids extra SEC submission calls during a backtest. A SIC-code crosswalk is
provided as a general fallback for tickers outside the curated universe.

Each sector maps to its Select Sector SPDR ETF, used as the free, point-in-time momentum proxy.
Two ETFs post-date 2015 (``XLRE`` 2015-10, ``XLC`` 2018-06); callers must treat momentum as
unavailable (None) before an ETF's inception rather than fabricating it.

This module is *diagnostic / candidate-signal infrastructure only* -- it is never consumed by
valuation, portfolio construction, or the live ``conf_top10`` book.
"""
from __future__ import annotations

from datetime import date

__all__ = [
    "SECTORS",
    "SECTOR_ETF",
    "ETF_INCEPTION",
    "TICKER_SECTOR",
    "sector_for_ticker",
    "etf_for_sector",
    "etf_available_asof",
    "sic_to_sector",
]

# Canonical sector labels (GICS-aligned).
SECTORS = (
    "technology",
    "communication_services",
    "consumer_discretionary",
    "consumer_staples",
    "health_care",
    "industrials",
    "financials",
    "energy",
    "materials",
    "utilities",
    "real_estate",
)

# Sector -> Select Sector SPDR ETF (the free momentum proxy).
SECTOR_ETF = {
    "technology": "XLK",
    "communication_services": "XLC",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "health_care": "XLV",
    "industrials": "XLI",
    "financials": "XLF",
    "energy": "XLE",
    "materials": "XLB",
    "utilities": "XLU",
    "real_estate": "XLRE",
}

# First reliably-tradeable date per ETF (momentum is None before this).
ETF_INCEPTION = {
    "XLB": date(1998, 12, 16), "XLE": date(1998, 12, 16), "XLF": date(1998, 12, 16),
    "XLI": date(1998, 12, 16), "XLK": date(1998, 12, 16), "XLP": date(1998, 12, 16),
    "XLU": date(1998, 12, 16), "XLV": date(1998, 12, 16), "XLY": date(1998, 12, 16),
    "XLRE": date(2015, 10, 8), "XLC": date(2018, 6, 18),
}

# Curated map for the research TRANCHE universe (+ common extras). Static/current sector is
# PIT-safe: none of these names changed broad sector over 2015-2026 in a way that matters.
# (V/MA/PYPL follow the current GICS placement in Financials; ADP/PAYX/VRSK in Industrials.)
TICKER_SECTOR = {
    # technology (hardware, semis, semi-equipment, software, IT services)
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology", "ADBE": "technology",
    "ORCL": "technology", "CRM": "technology", "AMD": "technology", "INTC": "technology",
    "TXN": "technology", "QCOM": "technology", "CSCO": "technology", "ACN": "technology",
    "NOW": "technology", "INTU": "technology", "AMAT": "technology", "LRCX": "technology",
    "KLAC": "technology", "MU": "technology", "AVGO": "technology", "ADI": "technology",
    "MCHP": "technology", "NXPI": "technology", "MRVL": "technology", "ON": "technology",
    "TER": "technology", "ENTG": "technology", "SWKS": "technology", "MPWR": "technology",
    "ADSK": "technology", "ANSS": "technology", "CDNS": "technology", "SNPS": "technology",
    "WDAY": "technology", "PANW": "technology", "FTNT": "technology",
    # communication services
    "GOOGL": "communication_services", "GOOG": "communication_services",
    "META": "communication_services", "EA": "communication_services", "TTWO": "communication_services",
    # consumer discretionary
    "AMZN": "consumer_discretionary", "HD": "consumer_discretionary", "NKE": "consumer_discretionary",
    "MCD": "consumer_discretionary", "SBUX": "consumer_discretionary", "LOW": "consumer_discretionary",
    "TGT": "consumer_discretionary", "BKNG": "consumer_discretionary", "EBAY": "consumer_discretionary",
    "ETSY": "consumer_discretionary", "TJX": "consumer_discretionary", "ROST": "consumer_discretionary",
    "ORLY": "consumer_discretionary", "AZO": "consumer_discretionary", "YUM": "consumer_discretionary",
    "CMG": "consumer_discretionary", "LULU": "consumer_discretionary", "DPZ": "consumer_discretionary",
    "MAR": "consumer_discretionary", "HLT": "consumer_discretionary",
    # consumer staples
    "KO": "consumer_staples", "COST": "consumer_staples", "PEP": "consumer_staples",
    "WMT": "consumer_staples", "PG": "consumer_staples", "CL": "consumer_staples",
    "GIS": "consumer_staples", "EL": "consumer_staples", "MDLZ": "consumer_staples",
    "KMB": "consumer_staples", "KHC": "consumer_staples", "STZ": "consumer_staples",
    "KDP": "consumer_staples", "HSY": "consumer_staples", "MKC": "consumer_staples",
    "CHD": "consumer_staples", "CLX": "consumer_staples", "K": "consumer_staples",
    "SJM": "consumer_staples", "HRL": "consumer_staples",
    # health care
    "DHR": "health_care", "TMO": "health_care", "ABT": "health_care", "SYK": "health_care",
    "ISRG": "health_care", "REGN": "health_care", "VRTX": "health_care",
    # industrials
    "ODFL": "industrials", "FAST": "industrials", "PAYX": "industrials", "ADP": "industrials",
    "VRSK": "industrials", "CTAS": "industrials", "ROL": "industrials", "CPRT": "industrials",
    "EXPD": "industrials",
    # financials (incl. payment networks per current GICS)
    "V": "financials", "MA": "financials", "PYPL": "financials", "JKHY": "financials",
    # materials
    "LIN": "materials",
}


def sector_for_ticker(ticker: str, sic: str | int | None = None) -> str | None:
    """Resolve a ticker to a canonical sector. Curated map first, then SIC crosswalk."""
    if not ticker:
        return None
    s = TICKER_SECTOR.get(ticker.upper())
    if s is not None:
        return s
    return sic_to_sector(sic) if sic is not None else None


def etf_for_sector(sector: str | None) -> str | None:
    return SECTOR_ETF.get(sector) if sector else None


def etf_available_asof(etf: str | None, asof: date) -> bool:
    """True if the ETF was trading on/before ``asof`` (momentum is None otherwise)."""
    if not etf:
        return False
    inc = ETF_INCEPTION.get(etf)
    return inc is not None and asof >= inc


def sic_to_sector(sic: str | int | None) -> str | None:
    """Map a 4-digit SIC code to a canonical sector (broad, approximate v1 crosswalk).

    Curve-outs (tech/health/energy) take precedence over the coarse division ranges so that
    e.g. semiconductors (367x) land in technology and pharma (283x) in health_care.
    """
    if sic is None:
        return None
    try:
        code = int(str(sic).strip()[:4])
    except (TypeError, ValueError):
        return None
    # Specific carve-outs first.
    if code in range(3570, 3580) or code in range(3670, 3680) or code in range(7370, 7380):
        return "technology"  # computers, semiconductors, computer services/software
    if code in range(2833, 2837) or code in range(3840, 3852) or code in range(8000, 8100) \
            or code in range(3826, 3830):
        return "health_care"  # pharma, medical devices/instruments, health services
    if code in range(1300, 1400) or code in range(2900, 3000):
        return "energy"  # oil & gas, petroleum refining
    if code in range(4800, 4900) or code in range(2700, 2742):
        return "communication_services"  # communications, publishing
    # Coarse division ranges.
    if code in range(1000, 1300) or code in range(1400, 1500) or code in range(2800, 2900) \
            or code in range(3300, 3400) or code in range(2400, 2700):
        return "materials"
    if code in range(4900, 5000):
        return "utilities"
    if code in range(6500, 6600):
        return "real_estate"
    if code in range(6000, 6800):
        return "financials"
    if code in range(2000, 2200):
        return "consumer_staples"
    if code in range(5400, 5500):
        return "consumer_staples"  # grocery stores
    if code in range(2200, 2400) or code in range(5000, 6000) or code in range(7000, 7400) \
            or code in range(5800, 5900):
        return "consumer_discretionary"
    if code in range(1500, 1800) or code in range(3400, 3800) or code in range(3000, 3300) \
            or code in range(3800, 4000) or code in range(4000, 4800) or code in range(7300, 8000):
        return "industrials"
    return None
