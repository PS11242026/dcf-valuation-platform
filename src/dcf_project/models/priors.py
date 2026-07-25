from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyPrior:
    archetype: str
    growth_persistence: float
    margin_persistence: float
    roic_persistence: float
    reinvestment_persistence: float
    reinvestment_burden: float
    revenue_growth_floor: float
    revenue_growth_cap: float
    margin_floor: float
    margin_cap: float
    roic_floor: float
    roic_cap: float
    default_rd_ratio: float
    default_buyback_rate: float
    terminal_growth_policy: float
    terminal_growth_floor: float
    terminal_growth_cap: float


TICKER_ARCHETYPES = {
    "AAPL": "aapl",
    "MSFT": "msft",
    "GOOGL": "googl",
    "GOOG": "googl",
    "META": "meta",
    "AMZN": "amzn",
    "TSLA": "capital_intensive_growth",
}


ARCHETYPE_PRIORS = {
    "aapl": CompanyPrior("aapl", 0.90, 0.995, 0.995, 0.78, 0.07, 0.006, 0.055, 0.25, 0.38, 0.12, 0.32, 0.07, 0.025, 0.0275, 0.025, 0.030),
    "msft": CompanyPrior("msft", 0.93, 0.997, 0.996, 0.82, 0.09, 0.010, 0.075, 0.32, 0.48, 0.12, 0.34, 0.13, 0.010, 0.0300, 0.025, 0.035),
    "googl": CompanyPrior("googl", 0.92, 0.996, 0.995, 0.80, 0.10, 0.010, 0.080, 0.25, 0.40, 0.10, 0.30, 0.15, 0.020, 0.0300, 0.025, 0.035),
    "meta": CompanyPrior("meta", 0.91, 0.995, 0.995, 0.80, 0.10, 0.008, 0.085, 0.28, 0.45, 0.10, 0.32, 0.18, 0.015, 0.0300, 0.025, 0.035),
    "amzn": CompanyPrior("amzn", 0.94, 0.970, 0.940, 0.92, 0.22, 0.012, 0.110, 0.05, 0.18, 0.03, 0.16, 0.10, 0.000, 0.0300, 0.025, 0.030),
    "capital_intensive_growth": CompanyPrior("capital_intensive_growth", 0.88, 0.900, 0.965, 0.92, 0.30, 0.005, 0.110, 0.04, 0.22, 0.04, 0.18, 0.045, 0.000, 0.0275, 0.025, 0.030),
    "consumer": CompanyPrior("consumer", 0.76, 0.960, 0.940, 0.72, 0.07, 0.003, 0.040, 0.14, 0.30, 0.05, 0.16, 0.02, 0.010, 0.0225, 0.020, 0.025),
    "industrial": CompanyPrior("industrial", 0.70, 0.900, 0.880, 0.70, 0.15, 0.000, 0.045, 0.08, 0.22, 0.04, 0.14, 0.02, 0.005, 0.0250, 0.020, 0.030),
}


def archetype_for(ticker: str | None, company_type: str) -> str:
    ticker_upper = (ticker or "").upper()
    if ticker_upper in TICKER_ARCHETYPES:
        return TICKER_ARCHETYPES[ticker_upper]
    if company_type in {"consumer_staple", "consumer_compounder"}:
        return "consumer"
    if company_type in {"industrial", "retail", "telecom", "semiconductor"}:
        return "industrial"
    if company_type == "capital_intensive_growth":
        return "capital_intensive_growth"
    if company_type in {"hyperscaler", "platform"}:
        return "googl"
    return "consumer" if company_type.startswith("consumer") else "industrial"


def prior_for(ticker: str | None, company_type: str) -> CompanyPrior:
    return ARCHETYPE_PRIORS[archetype_for(ticker, company_type)]
