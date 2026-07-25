from __future__ import annotations

import numpy as np
import pandas as pd


TICKER_COMPANY_TYPES = {
    "AAPL": "asset_light_quality",
    "MSFT": "hyperscaler",
    "GOOGL": "hyperscaler",
    "GOOG": "hyperscaler",
    "META": "platform",
    "AMZN": "marketplace_hybrid",
    "TSLA": "capital_intensive_growth",
    "COST": "consumer_staple",
    "KO": "consumer_staple",
    "JNJ": "consumer_staple",
    "PG": "consumer_staple",
    "PEP": "consumer_staple",
    "WMT": "retail",
    "T": "telecom",
    "VZ": "telecom",
    "INTC": "semiconductor",
    "NVDA": "semiconductor",
    "AMD": "semiconductor",
}


def resolve_company_type(ticker: str, configured: str, history: pd.DataFrame) -> tuple[str, dict[str, object]]:
    configured = str(configured or "asset_light")
    ticker_upper = ticker.upper()
    if configured != "auto":
        return configured, {"method": "explicit_user_company_type", "configured_company_type": configured}
    if ticker_upper in TICKER_COMPANY_TYPES:
        resolved = TICKER_COMPANY_TYPES[ticker_upper]
        return resolved, {
            "method": "explicit_ticker_mapping",
            "configured_company_type": configured,
            "resolved_company_type": resolved,
        }
    metrics = _asset_metrics(history)
    resolved = _classify_from_assets(metrics)
    return resolved, {
        "method": "asset_intensity_and_unit_economics",
        "configured_company_type": configured,
        "resolved_company_type": resolved,
        "metrics": metrics,
    }


def _asset_metrics(history: pd.DataFrame) -> dict[str, float | None]:
    tail = history.tail(12)
    revenue = _median_positive(tail, "revenue")
    capex_ratio = _median_ratio(tail, "capex", "revenue")
    # PPE and invested capital are balance-sheet stocks; revenue here is a single
    # quarter (a flow). Dividing a stock by quarterly revenue inflates the ratio ~4x
    # (e.g. INTC ppe/rev 7.7 instead of ~1.9, LLY 1.33 instead of 0.33), which the
    # annual-calibrated telecom/industrial thresholds then mis-fire on. Annualize the
    # denominator so stock-to-revenue ratios are on an annual basis.
    ppe_ratio = _annualize_stock_ratio(_median_ratio(tail, "ppe", "revenue"))
    invested_capital_ratio = _annualize_stock_ratio(_median_ratio(tail, "invested_capital", "revenue"))
    margin = _median(tail, "operating_margin")
    roic = _median(tail, "roic")
    rd_ratio = _median_ratio(tail, "r_and_d", "revenue")
    revenue_growth = _median(tail, "revenue_growth")
    return {
        "revenue": revenue,
        "capex_to_revenue": capex_ratio,
        "ppe_to_revenue": ppe_ratio,
        "invested_capital_to_revenue": invested_capital_ratio,
        "operating_margin": margin,
        "roic": roic,
        "r_and_d_to_revenue": rd_ratio,
        "quarterly_revenue_growth": revenue_growth,
    }


def _classify_from_assets(metrics: dict[str, float | None]) -> str:
    capex = metrics.get("capex_to_revenue") or 0.0
    ppe = metrics.get("ppe_to_revenue") or 0.0
    invested = metrics.get("invested_capital_to_revenue") or 0.0
    margin = metrics.get("operating_margin") or 0.0
    roic = metrics.get("roic") or 0.0
    rd = metrics.get("r_and_d_to_revenue") or 0.0
    growth = metrics.get("quarterly_revenue_growth") or 0.0

    if capex >= 0.12 or ppe >= 1.10:
        return "telecom"
    if capex >= 0.075 or invested >= 1.60:
        return "industrial"
    if rd >= 0.08 and capex >= 0.045:
        return "hyperscaler"
    if rd >= 0.07 and margin >= 0.22:
        return "platform"
    if margin >= 0.24 and roic >= 0.08 and capex <= 0.045:
        return "asset_light_quality"
    if growth <= 0.015 and margin >= 0.12 and capex <= 0.055:
        return "consumer_staple"
    return "asset_light"


def _annualize_stock_ratio(quarterly_ratio: float | None) -> float | None:
    """Convert a stock / quarterly-revenue ratio to a stock / annual-revenue ratio."""
    return quarterly_ratio / 4.0 if quarterly_ratio is not None else None


def _median_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> float | None:
    if numerator not in frame or denominator not in frame:
        return None
    denom = frame[denominator].astype(float).replace(0.0, np.nan)
    values = (frame[numerator].astype(float).abs() / denom).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else None


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = frame[column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else None


def _median_positive(frame: pd.DataFrame, column: str) -> float | None:
    value = _median(frame, column)
    return value if value is not None and value > 0.0 else None
