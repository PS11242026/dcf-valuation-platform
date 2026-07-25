"""Deterministic stratified 6-family sample for the Research_V4 diagnostic.

Healthcare and Financials are deliberately EXCLUDED: fade-DCF is anti-predictive for
healthcare and structurally inapplicable to balance-sheet-driven financials, so they are
out of scope for a study whose whole point is to find *fixable* model defects (see the
research_v4 plan + the dcf-predictive-power-archetype-conditional memory).

Selection is reproducible: each family is seeded from the frozen v3 backtest TRANCHE
(the established large caps the v2/v3 gates already ran on) and topped up, in the frozen
universe's own order, until ``per_family`` names are reached. The chosen names and the
rule are written verbatim into the snapshot output so the sample is never cherry-picked.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_UNIVERSE = ROOT / "reports" / "_universe_tickers.json"

# The six families the engine is designed for. "Semiconductors" is the fabless/IDM
# `semiconductors` family only (semiconductor_equipment is a distinct capital-goods
# family and is left out to keep the bucket economically homogeneous).
FAMILIES: dict[str, str] = {
    "Consumer Staples": "consumer_staples",
    "Consumer Discretionary": "consumer_discretionary",
    "Industrials": "industrials_assetlight_services",
    "Software": "software_saas_infra",
    "Internet": "internet_platforms_marketplaces",
    "Semiconductors": "semiconductors",
}

# Frozen v3 backtest TRANCHE (large, liquid, established names) used as the seed so the
# V4 snapshot overlaps the universe the prior gates ran on. Copied from
# scripts/utilities/research_v3/v3_universe_backtest.py::TRANCHE.
_TRANCHE = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "KO", "COST", "V", "ADBE",
    "ORCL", "CRM", "PEP", "WMT", "HD", "MA", "AMD", "INTC", "TXN", "NKE",
    "MCD", "SBUX", "QCOM", "CSCO", "ACN", "LIN", "LOW", "TGT", "DHR", "TMO",
    "ABT", "NOW", "INTU", "AMAT", "LRCX", "KLAC", "MU", "PG", "CL", "GIS",
    "EL", "MDLZ", "KMB", "SYK", "ISRG", "REGN", "VRTX", "BKNG", "EBAY", "ETSY",
    "AVGO", "ADI", "MCHP", "NXPI", "MRVL", "ON", "TER", "ENTG", "SWKS", "MPWR",
    "ADSK", "ANSS", "CDNS", "SNPS", "WDAY", "PANW", "FTNT", "PYPL", "EA", "TTWO",
    "TJX", "ROST", "ORLY", "AZO", "YUM", "CMG", "LULU", "DPZ", "MAR", "HLT",
    "KHC", "STZ", "KDP", "HSY", "MKC", "CHD", "CLX", "K", "SJM", "HRL",
    "ODFL", "FAST", "PAYX", "ADP", "VRSK", "CTAS", "ROL", "CPRT", "JKHY", "EXPD",
]


def load_family_of() -> dict[str, str]:
    meta = json.loads(_UNIVERSE.read_text(encoding="utf-8"))
    return {t.upper(): fam for t, fam in meta["family_of"].items()}


def stratified_sample(per_family: int = 13) -> dict[str, list[str]]:
    """Return {V4 sector label -> [tickers]}, ``per_family`` each, deterministically."""
    family_of = load_family_of()
    # Universe members per family key, in the universe file's native order.
    members: dict[str, list[str]] = {}
    for ticker, fam in family_of.items():
        members.setdefault(fam, []).append(ticker)

    sample: dict[str, list[str]] = {}
    for label, key in FAMILIES.items():
        seed = [t for t in _TRANCHE if family_of.get(t) == key]
        chosen: list[str] = list(dict.fromkeys(seed))[:per_family]
        if len(chosen) < per_family:
            for ticker in members.get(key, []):
                if ticker not in chosen:
                    chosen.append(ticker)
                if len(chosen) >= per_family:
                    break
        sample[label] = chosen[:per_family]
    return sample


def flat_sample(per_family: int = 13) -> list[tuple[str, str]]:
    """Flatten to [(ticker, V4 sector label), ...]."""
    out: list[tuple[str, str]] = []
    for label, tickers in stratified_sample(per_family).items():
        out.extend((t, label) for t in tickers)
    return out
