from __future__ import annotations

import numpy as np
import pandas as pd

from dcf_project.config import Settings
from dcf_project.contracts import MarketSnapshot, WACCInputs
from dcf_project.valuation.discounting import present_value, present_value_series
from dcf_project.valuation.terminal import terminal_value


def deterministic_baseline(
    features: pd.DataFrame,
    wacc_inputs: WACCInputs,
    market: MarketSnapshot,
    latest_cash: float | None,
    latest_debt: float | None,
    latest_shares: float | None,
    settings: Settings,
    horizon_quarters: int = 20,
) -> dict[str, object]:
    latest = features.iloc[-1]
    history = features.tail(min(len(features), 20))
    revenue_cagr = _annual_revenue_cagr(features)
    quarterly_revenue_growth = (1.0 + revenue_cagr) ** 0.25 - 1.0
    ebit_margin = float(history["ebit_margin"].median())
    tax_rate = float(history["tax_rate"].median())
    da_ratio = float(history["da_ratio"].median())
    capex_ratio = float(min(history["capex_ratio"].tail(12).mean(), features["capex_ratio"].median() * 1.25))
    maintenance_capex_ratio = float(
        history["maintenance_capex_ratio"].median()
        if "maintenance_capex_ratio" in history
        else features["capex_ratio"].median()
    )
    nwc_ratio = float(history["nwc_ratio"].median())
    revenue = float(latest["revenue"])
    projected_fcf = []
    for _ in range(horizon_quarters):
        revenue *= 1.0 + quarterly_revenue_growth
        ebit = revenue * ebit_margin
        nopat = ebit * (1.0 - tax_rate)
        da = revenue * da_ratio
        capex = revenue * capex_ratio
        maintenance_capex = revenue * maintenance_capex_ratio
        delta_nwc = revenue * nwc_ratio
        reported_fcf = nopat + da - capex - delta_nwc
        economic_fcf = nopat + da - maintenance_capex - delta_nwc
        sbc = revenue * _latest_ratio(features, "sbc")
        sbc_haircut = float(features.iloc[-1].get("sbc_haircut_factor", 1.0))
        owner_earnings = economic_fcf + sbc * (1.0 - sbc_haircut)
        selected_fcf = _select_cash_flow(settings, reported_fcf, economic_fcf, owner_earnings)
        projected_fcf.append(float(selected_fcf))

    quarterly_wacc = wacc_inputs.wacc / 4.0
    terminal_base = _stabilized_terminal_fcf(projected_fcf)
    tv = terminal_value(terminal_base * 4.0, wacc_inputs.wacc, settings.default_terminal_growth)
    ev = present_value_series(projected_fcf, quarterly_wacc) + present_value(tv, quarterly_wacc, len(projected_fcf))
    cash = latest_cash if latest_cash is not None else market.cash or 0.0
    debt = latest_debt if latest_debt is not None else market.debt or 0.0
    shares = latest_shares or market.shares_outstanding or float(latest["shares_outstanding"])
    equity_value = ev - (debt - cash)
    intrinsic = equity_value / shares
    return {
        "intrinsic_value_per_share": float(intrinsic),
        "enterprise_value": float(ev),
        "revenue_cagr_5y": float(revenue_cagr),
        "normalized_ebit_margin": ebit_margin,
        "normalized_tax_rate": tax_rate,
        "normalized_da_ratio": da_ratio,
        "normalized_capex_ratio": capex_ratio,
        "normalized_maintenance_capex_ratio": maintenance_capex_ratio,
        "normalized_nwc_ratio": nwc_ratio,
        "valuation_mode": _effective_valuation_mode(settings),
        "terminal_growth_rate": settings.default_terminal_growth,
        "fcf_terminal_base": terminal_base,
    }


def _annual_revenue_cagr(features: pd.DataFrame) -> float:
    clean = features.dropna(subset=["revenue"])
    if len(clean) >= 21:
        start = float(clean.iloc[-21]["revenue"])
        end = float(clean.iloc[-1]["revenue"])
        years = 5.0
    else:
        start = float(clean.iloc[0]["revenue"])
        end = float(clean.iloc[-1]["revenue"])
        years = max((len(clean) - 1) / 4.0, 0.25)
    if start <= 0 or end <= 0:
        return 0.03
    return float(np.clip((end / start) ** (1.0 / years) - 1.0, -0.05, 0.15))


def _stabilized_terminal_fcf(projected_fcf: list[float]) -> float:
    if len(projected_fcf) < 4:
        return float(projected_fcf[-1])
    weights = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    return float(np.average(np.asarray(projected_fcf[-4:], dtype=float), weights=weights))


def _select_cash_flow(
    settings: Settings,
    reported_fcf: float,
    economic_fcf: float,
    owner_earnings: float,
) -> float:
    mode = _effective_valuation_mode(settings)
    if mode == "owner_earnings":
        return owner_earnings
    if mode == "economic":
        return economic_fcf
    return reported_fcf


def _effective_valuation_mode(settings: Settings) -> str:
    if settings.valuation_mode in {"owner_earnings", "owner_earnings_dcf"} or (
        settings.valuation_mode == "auto" and settings.company_type == "marketplace_hybrid"
    ):
        return "owner_earnings"
    if settings.valuation_mode in {"economic", "economic_dcf"} or (
        settings.valuation_mode == "auto" and settings.company_type in {"hyperscaler", "platform"}
    ):
        return "economic"
    return "reported"


def _latest_ratio(features: pd.DataFrame, column: str) -> float:
    if column not in features:
        return 0.0
    latest = features.iloc[-1]
    revenue = float(latest.get("revenue", 0.0))
    value = float(latest.get(column, 0.0) or 0.0)
    if revenue <= 0:
        return 0.0
    return value / revenue
