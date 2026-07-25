"""India ticker -> canonical GICS sector resolution (country bolt-on).

Parallels ``engine_v2.sector_map.sector_for_ticker`` (the US curated-map + SIC-crosswalk
pattern) for Indian listings. Keys are NSE symbols WITHOUT the ``.NS``/``.BO`` Yahoo suffix
(stripped here). Curated map first (Nifty-100-ish large caps), then a coarse NIC-2008
crosswalk fallback, then None (the engine's classifier then resolves an archetype anyway).

Sector labels are the same 11 canonical GICS labels as ``engine_v2.sector_map.SECTORS`` so the
India sector priors (``engine_v3.sector_priors.IN_SECTOR_PRIORS``) key off them directly.
"""
from __future__ import annotations

# Curated NSE-symbol -> canonical sector. Sector membership for these large caps is stable.
TICKER_SECTOR_IN: dict[str, str] = {
    # technology -- IT services / products
    "TCS": "technology", "INFY": "technology", "WIPRO": "technology", "HCLTECH": "technology",
    "TECHM": "technology", "LTIM": "technology", "PERSISTENT": "technology", "COFORGE": "technology",
    "MPHASIS": "technology", "LTTS": "technology", "OFSS": "technology",
    # financials -- banks, NBFCs, insurers, AMCs
    "HDFCBANK": "financials", "ICICIBANK": "financials", "SBIN": "financials", "KOTAKBANK": "financials",
    "AXISBANK": "financials", "INDUSINDBK": "financials", "BANKBARODA": "financials", "PNB": "financials",
    "BAJFINANCE": "financials", "BAJAJFINSV": "financials", "CHOLAFIN": "financials", "SHRIRAMFIN": "financials",
    "HDFCLIFE": "financials", "SBILIFE": "financials", "ICICIPRULI": "financials", "ICICIGI": "financials",
    "SBICARD": "financials", "HDFCAMC": "financials", "MUTHOOTFIN": "financials", "IDFCFIRSTB": "financials",
    # consumer staples -- FMCG
    "HINDUNILVR": "consumer_staples", "ITC": "consumer_staples", "NESTLEIND": "consumer_staples",
    "BRITANNIA": "consumer_staples", "DABUR": "consumer_staples", "MARICO": "consumer_staples",
    "GODREJCP": "consumer_staples", "COLPAL": "consumer_staples", "TATACONSUM": "consumer_staples",
    "VBL": "consumer_staples", "UBL": "consumer_staples", "PGHH": "consumer_staples", "EMAMILTD": "consumer_staples",
    # consumer discretionary -- autos, durables, retail, apparel, QSR, internet retail
    "MARUTI": "consumer_discretionary", "TATAMOTORS": "consumer_discretionary", "M&M": "consumer_discretionary",
    "BAJAJ-AUTO": "consumer_discretionary", "EICHERMOT": "consumer_discretionary", "HEROMOTOCO": "consumer_discretionary",
    "TVSMOTOR": "consumer_discretionary", "TITAN": "consumer_discretionary", "TRENT": "consumer_discretionary",
    "DMART": "consumer_discretionary", "HAVELLS": "consumer_discretionary", "VOLTAS": "consumer_discretionary",
    "JUBLFOOD": "consumer_discretionary", "NYKAA": "consumer_discretionary", "ZOMATO": "consumer_discretionary",
    "PAGEIND": "consumer_discretionary", "BOSCHLTD": "consumer_discretionary",
    # health care -- pharma, hospitals, diagnostics
    "SUNPHARMA": "health_care", "DRREDDY": "health_care", "CIPLA": "health_care", "DIVISLAB": "health_care",
    "APOLLOHOSP": "health_care", "AUROPHARMA": "health_care", "LUPIN": "health_care", "BIOCON": "health_care",
    "TORNTPHARM": "health_care", "ALKEM": "health_care", "MANKIND": "health_care", "ZYDUSLIFE": "health_care",
    "MAXHEALTH": "health_care", "LAURUSLABS": "health_care", "IPCALAB": "health_care",
    # industrials -- capital goods, defense, transport
    "LT": "industrials", "SIEMENS": "industrials", "ABB": "industrials", "BEL": "industrials",
    "HAL": "industrials", "BHEL": "industrials", "CUMMINSIND": "industrials", "ADANIPORTS": "industrials",
    "CONCOR": "industrials", "THERMAX": "industrials", "ABBOTINDIA": "health_care",
    # energy -- oil & gas, refining, coal
    "RELIANCE": "energy", "ONGC": "energy", "IOC": "energy", "BPCL": "energy", "HINDPETRO": "energy",
    "GAIL": "energy", "COALINDIA": "energy", "OIL": "energy",
    # materials -- cement, metals, chemicals, paints
    "ULTRACEMCO": "materials", "SHREECEM": "materials", "GRASIM": "materials", "AMBUJACEM": "materials",
    "ACC": "materials", "JSWSTEEL": "materials", "TATASTEEL": "materials", "HINDALCO": "materials",
    "VEDL": "materials", "JINDALSTEL": "materials", "HINDZINC": "materials", "NATIONALUM": "materials",
    "ASIANPAINT": "materials", "BERGEPAINT": "materials", "PIDILITIND": "materials", "SRF": "materials",
    "UPL": "materials", "COROMANDEL": "materials", "PIIND": "materials",
    # communication services -- telecom, towers
    "BHARTIARTL": "communication_services", "IDEA": "communication_services", "INDUSTOWER": "communication_services",
    # utilities -- power generation & transmission
    "NTPC": "utilities", "POWERGRID": "utilities", "ADANIPOWER": "utilities", "ADANIGREEN": "utilities",
    "TATAPOWER": "utilities", "JSWENERGY": "utilities", "NHPC": "utilities", "ADANIENSOL": "utilities",
    # real estate
    "DLF": "real_estate", "GODREJPROP": "real_estate", "OBEROIRLTY": "real_estate", "PRESTIGE": "real_estate",
    "PHOENIXLTD": "real_estate", "LODHA": "real_estate",
}


#: GICS sector -> a readable company "archetype" for Indian listings. The US engine derives
#: the archetype from asset-intensity metrics (capex/PPE/R&D), but the NSE results XBRL carries
#: no capex line, so that classifier collapses every Indian name to ``asset_light``. For India
#: we instead label the archetype from the (curated, reliable) GICS sector. This is a
#: DIAGNOSTIC/stratification label only -- the valuation drivers read the GICS ``sector`` and its
#: ``IN_SECTOR_PRIORS`` prior directly, never this archetype -- so it changes presentation, not value.
_SECTOR_ARCHETYPE_IN: dict[str, str] = {
    "technology": "it_services",
    "communication_services": "telecom",
    "consumer_discretionary": "consumer_cyclical",
    "consumer_staples": "consumer_staple",
    "health_care": "pharma_healthcare",
    "industrials": "industrial",
    "financials": "financials",
    "energy": "energy",
    "materials": "materials",
    "utilities": "utilities",
    "real_estate": "real_estate",
}


def india_archetype_for_sector(sector: str | None) -> str | None:
    """Map a resolved GICS sector to a readable India company archetype (diagnostic only)."""
    if not sector:
        return None
    return _SECTOR_ARCHETYPE_IN.get(sector, sector)


def _strip_suffix(ticker: str) -> str:
    upper = ticker.upper()
    for suffix in (".NS", ".BO"):
        if upper.endswith(suffix):
            return upper[: -len(suffix)]
    return upper


def india_sector_for_ticker(ticker: str, nic: str | int | None = None) -> str | None:
    """Resolve an Indian ticker to a canonical sector. Curated map first, then NIC crosswalk."""
    if not ticker:
        return None
    sector = TICKER_SECTOR_IN.get(_strip_suffix(ticker))
    if sector is not None:
        return sector
    return nic_to_sector(nic) if nic is not None else None


def nic_to_sector(nic: str | int | None) -> str | None:
    """Coarse NIC-2008 (2-digit division) -> canonical sector crosswalk (best-effort fallback)."""
    if nic is None:
        return None
    try:
        division = int(str(nic).strip()[:2])
    except (TypeError, ValueError):
        return None
    if division in range(5, 10) or division == 19:          # mining, coke & refined petroleum
        return "energy"
    if division in (20, 21):                                 # chemicals & pharma
        return "health_care" if division == 21 else "materials"
    if division in range(10, 13):                            # food/beverage/tobacco manufacturing
        return "consumer_staples"
    if division in range(13, 19) or division in range(22, 26) or division == 31:
        return "materials"                                   # textiles, rubber/plastics, metals
    if division in range(26, 31) or division in (32, 33):    # machinery, electricals, transport eq.
        return "industrials"
    if division in range(35, 40):                            # electricity, gas, water, waste
        return "utilities"
    if division in range(41, 44):                            # construction
        return "industrials"
    if division in range(45, 48):                            # wholesale & retail trade
        return "consumer_discretionary"
    if division in range(49, 54):                            # transport & logistics
        return "industrials"
    if division in (61, 62, 63):                             # telecom, IT services, info services
        return "technology" if division in (62, 63) else "communication_services"
    if division in range(64, 67):                            # financial services & insurance
        return "financials"
    if division == 68:                                       # real estate
        return "real_estate"
    if division in (86, 87, 88):                             # human health & social work
        return "health_care"
    return None
