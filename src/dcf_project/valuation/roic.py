from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from dcf_project.config import Settings
from dcf_project.contracts import DCFResult, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.valuation.discounting import present_value, present_value_series
from dcf_project.valuation.terminal import terminal_value


def build_roic_dcf_result(
    ticker: str,
    features: pd.DataFrame,
    wacc_inputs: WACCInputs,
    market: MarketSnapshot,
    latest_cash: float | None,
    latest_debt: float | None,
    latest_shares: float | None,
    sources: list[SourceMeta],
    warnings: list[str],
    settings: Settings,
    latest_operating_lease_liability: float | None = None,
) -> DCFResult:
    timings: dict[str, float] = {}
    start = perf_counter()
    economics = _forecast_roic_economics(features, settings)
    timings["forecast_fit"] = perf_counter() - start
    projected_fcf = [row["fcf"] for row in economics["explicit_forecast"]]
    quarterly_wacc = wacc_inputs.wacc / 4.0
    pv_fcf = present_value_series(projected_fcf, quarterly_wacc)
    fade_fcfs = [row["fcf"] for row in economics["fade_schedule"]]
    pv_fade = sum(
        present_value(fcf, quarterly_wacc, settings.explicit_forecast_quarters + idx)
        for idx, fcf in enumerate(fade_fcfs, start=1)
    )
    terminal_fcf = economics["terminal_fcf"]
    terminal_growth = min(economics["terminal_growth"], wacc_inputs.wacc - 0.01)
    tv = terminal_value(terminal_fcf * 4.0, wacc_inputs.wacc, terminal_growth)
    pv_tv = present_value(
        tv,
        quarterly_wacc,
        settings.explicit_forecast_quarters + len(fade_fcfs),
    )
    enterprise_value = pv_fcf + pv_fade + pv_tv
    cash = latest_cash if latest_cash is not None else market.cash or 0.0
    debt = latest_debt if latest_debt is not None else market.debt or 0.0
    lease_adjustment = latest_operating_lease_liability or 0.0
    if settings.normalize_leases:
        debt += lease_adjustment
    else:
        lease_adjustment = 0.0
    net_debt = max(debt - cash, -cash)
    equity_value = enterprise_value - net_debt
    shares = latest_shares or market.shares_outstanding
    if not shares or shares <= 0:
        raise ValueError("Diluted shares outstanding are required for per-share value")
    intrinsic = equity_value / shares
    implied_upside = (intrinsic / market.price - 1.0) if market.price else None
    growth_path = _growth_from_path(projected_fcf)
    start = perf_counter()
    reverse = (
        _reverse_roic_cross_check(market.price, shares, net_debt, economics, wacc_inputs.wacc)
        if settings.run_reverse_dcf
        else {"market_implied_growth": None, "market_implied_terminal_growth": None}
    )
    timings["reverse_dcf"] = perf_counter() - start
    start = perf_counter()
    confidence = _confidence_score(warnings, economics, settings)
    timings["confidence"] = perf_counter() - start
    start = perf_counter()
    sensitivity = (
        _roic_sensitivity(economics, wacc_inputs.wacc, net_debt, shares)
        if settings.run_sensitivity
        else {}
    )
    timings["sensitivity"] = perf_counter() - start
    diagnostics = {
        "valuation_mode": "roic_dcf",
        "company_type": settings.company_type,
        "roic": economics["initial_roic"],
        "reinvestment_rate": economics["initial_reinvestment_rate"],
        "growth_from_roic": economics["initial_growth_from_roic"],
        "fade_schedule": economics["fade_schedule"],
        "terminal_roic": economics["terminal_roic"],
        "terminal_reinvestment": economics["terminal_reinvestment"],
        "terminal_growth": economics["terminal_growth"],
        "invested_capital": economics["initial_invested_capital"],
        "roic_forecast": economics["explicit_forecast"],
        "reverse_dcf_growth": reverse,
        "market_implied_growth": reverse.get("market_implied_growth"),
        "market_implied_terminal_growth": reverse.get("market_implied_terminal_growth"),
        "valuation_confidence_score": confidence,
        "lease_adjustment": lease_adjustment,
        "wacc": wacc_inputs.model_dump(),
        "timing": timings,
    }
    return DCFResult(
        ticker=ticker.upper(),
        explicit_forecast_quarters=settings.explicit_forecast_quarters,
        intrinsic_value_per_share=float(intrinsic),
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        net_debt=float(net_debt),
        projected_fcf=[float(x) for x in projected_fcf],
        explicit_pv=float(pv_fcf),
        pv_projected_fcf=float(pv_fcf),
        terminal_value=float(tv),
        pv_terminal_value=float(pv_tv),
        fcf_terminal_base=float(terminal_fcf),
        wacc=float(wacc_inputs.wacc),
        terminal_growth_rate=float(terminal_growth),
        fcf_growth_rate=float(np.median(growth_path) * 4.0 if growth_path else 0.0),
        raw_fcf_growth_rate=float(growth_path[-1] * 4.0 if growth_path else 0.0),
        smoothed_fcf_growth_rate=float(np.median(growth_path) * 4.0 if growth_path else 0.0),
        clamped_fcf_growth_rate=float(np.median(growth_path) * 4.0 if growth_path else 0.0),
        growth_clamped=False,
        clamp_triggered=False,
        valuation_p10=float(intrinsic * 0.75),
        valuation_p50=float(intrinsic),
        valuation_p90=float(intrinsic * 1.25),
        diluted_shares_outstanding=float(shares),
        market_price=market.price,
        implied_upside=float(implied_upside) if implied_upside is not None else None,
        confidence_interval={
            "low_intrinsic_value": float(intrinsic * 0.75),
            "median_intrinsic_value": float(intrinsic),
            "high_intrinsic_value": float(intrinsic * 1.25),
        },
        sensitivity=sensitivity,
        diagnostics=diagnostics,
        sources=sources,
        warnings=warnings,
    )


def _forecast_roic_economics(features: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    latest = features.iloc[-1]
    history = features.tail(min(len(features), 20))
    revenue = float(latest["revenue"])
    margin = float(history["ebit_margin"].median())
    tax_rate = float(history["tax_rate"].median())
    invested_capital = float(latest["invested_capital"])
    roic = _clean_roic(float(history["roic"].tail(8).median()), settings)
    reinvestment_rate = _clean_reinvestment(float(history["reinvestment_rate"].tail(8).median()), settings)
    terminal_roic = _terminal_roic(settings, roic)
    terminal_reinvestment = _terminal_reinvestment(settings, reinvestment_rate)
    explicit = []
    for quarter in range(1, settings.explicit_forecast_quarters + 1):
        fade = quarter / settings.explicit_forecast_quarters
        roic_t = _blend(roic, _mid_fade_roic(settings, roic), fade * 0.35)
        reinvestment_t = _blend(reinvestment_rate, terminal_reinvestment, fade * 0.25)
        annual_growth = float(np.clip(roic_t * reinvestment_t, -0.05, _explicit_growth_cap(settings)))
        quarterly_growth = (1.0 + annual_growth) ** 0.25 - 1.0
        revenue *= 1.0 + quarterly_growth
        margin_t = _blend(margin, _terminal_margin(settings, margin), fade * 0.25)
        nopat = revenue * margin_t * (1.0 - tax_rate)
        fcf = nopat * (1.0 - reinvestment_t)
        invested_capital = max(invested_capital + nopat * reinvestment_t, revenue * 0.03)
        explicit.append(
            {
                "quarter": float(quarter),
                "revenue": float(revenue),
                "operating_margin": float(margin_t),
                "nopat": float(nopat),
                "roic": float(roic_t),
                "reinvestment_rate": float(reinvestment_t),
                "growth_from_roic": float(annual_growth),
                "fcf": float(fcf),
                "invested_capital": float(invested_capital),
            }
        )
    fade_schedule = []
    nopat = explicit[-1]["nopat"]
    for year in range(6, 16):
        fade = (year - 5) / 10.0
        roic_t = _blend(explicit[-1]["roic"], terminal_roic, fade)
        reinvestment_t = _blend(explicit[-1]["reinvestment_rate"], terminal_reinvestment, fade)
        annual_growth = float(np.clip(roic_t * reinvestment_t, 0.0, _fade_growth_cap(settings)))
        for quarter in range(1, 5):
            nopat *= 1.0 + ((1.0 + annual_growth) ** 0.25 - 1.0)
            fcf = nopat * (1.0 - reinvestment_t)
            fade_schedule.append(
                {
                    "year": float(year),
                    "quarter": float(quarter),
                    "fade_weight": float(fade),
                    "roic": float(roic_t),
                    "reinvestment_rate": float(reinvestment_t),
                    "growth_from_roic": float(annual_growth),
                    "nopat": float(nopat),
                    "fcf": float(fcf),
                }
            )
    stable_growth = float(np.clip(terminal_roic * terminal_reinvestment, 0.0, _terminal_growth_cap(settings)))
    terminal_nopat = fade_schedule[-1]["nopat"] * (1.0 + stable_growth / 4.0)
    return {
        "initial_roic": roic,
        "initial_reinvestment_rate": reinvestment_rate,
        "initial_growth_from_roic": roic * reinvestment_rate,
        "initial_invested_capital": float(latest["invested_capital"]),
        "explicit_forecast": explicit,
        "fade_schedule": fade_schedule,
        "terminal_roic": terminal_roic,
        "terminal_reinvestment": terminal_reinvestment,
        "terminal_growth": stable_growth,
        "terminal_fcf": terminal_nopat * (1.0 - terminal_reinvestment),
    }


def _clean_roic(value: float, settings: Settings) -> float:
    high_caps = {"consumer_staple": 0.45, "platform": 0.55, "hyperscaler": 0.45, "marketplace_hybrid": 0.35}
    return float(np.clip(value if np.isfinite(value) else 0.15, 0.04, high_caps.get(settings.company_type, 0.30)))


def _clean_reinvestment(value: float, settings: Settings) -> float:
    defaults = {"consumer_staple": 0.18, "platform": 0.25, "hyperscaler": 0.35, "marketplace_hybrid": 0.30}
    if not np.isfinite(value) or value <= 0:
        value = defaults.get(settings.company_type, 0.25)
    return float(np.clip(value, 0.02, _reinvestment_cap(settings)))


def _terminal_roic(settings: Settings, initial: float) -> float:
    terminal = {
        "consumer_staple": 0.20,
        "platform": 0.22,
        "hyperscaler": 0.18,
        "marketplace_hybrid": 0.16,
        "industrial": 0.11,
        "retail": 0.13,
        "telecom": 0.08,
        "asset_light": 0.16,
    }.get(settings.company_type, 0.12)
    return float(min(initial, max(terminal, initial * _terminal_roic_retention(settings))))


def _terminal_roic_retention(settings: Settings) -> float:
    return {
        "consumer_staple": 0.75,
        "platform": 0.72,
        "hyperscaler": 0.62,
        "marketplace_hybrid": 0.58,
        "industrial": 0.45,
        "telecom": 0.40,
    }.get(settings.company_type, 0.55)


def _mid_fade_roic(settings: Settings, initial: float) -> float:
    return _blend(initial, _terminal_roic(settings, initial), 0.35)


def _terminal_reinvestment(settings: Settings, initial: float) -> float:
    floor = {"consumer_staple": 0.10, "platform": 0.14, "hyperscaler": 0.18, "marketplace_hybrid": 0.16}.get(
        settings.company_type,
        0.12,
    )
    return float(np.clip(max(floor, initial * 0.45), 0.04, min(0.35, _reinvestment_cap(settings))))


def _reinvestment_cap(settings: Settings) -> float:
    return {
        "consumer_staple": 0.35,
        "platform": 0.55,
        "hyperscaler": 0.60,
        "marketplace_hybrid": 0.65,
        "asset_light": 0.45,
        "industrial": 0.50,
    }.get(settings.company_type, 0.55)


def _explicit_growth_cap(settings: Settings) -> float:
    return {
        "consumer_staple": 0.08,
        "platform": 0.18,
        "hyperscaler": 0.16,
        "marketplace_hybrid": 0.18,
        "asset_light": 0.12,
        "industrial": 0.10,
    }.get(settings.company_type, 0.12)


def _fade_growth_cap(settings: Settings) -> float:
    return {
        "consumer_staple": 0.055,
        "platform": 0.10,
        "hyperscaler": 0.09,
        "marketplace_hybrid": 0.10,
        "asset_light": 0.07,
    }.get(settings.company_type, 0.07)


def _terminal_growth_cap(settings: Settings) -> float:
    return {
        "consumer_staple": 0.025,
        "platform": 0.035,
        "hyperscaler": 0.035,
        "marketplace_hybrid": 0.035,
        "asset_light": 0.030,
        "industrial": 0.025,
        "telecom": 0.020,
    }.get(settings.company_type, settings.terminal_growth_cap)


def _terminal_margin(settings: Settings, initial: float) -> float:
    cap = {"consumer_staple": 0.35, "platform": 0.35, "hyperscaler": 0.32, "marketplace_hybrid": 0.14}.get(
        settings.company_type,
        0.25,
    )
    return float(np.clip(initial, -0.05, cap))


def _blend(start: float, end: float, weight: float) -> float:
    weight = float(np.clip(weight, 0.0, 1.0))
    return start * (1.0 - weight) + end * weight


def _growth_from_path(path: list[float]) -> list[float]:
    growth = [0.0]
    for prev, cur in zip(path, path[1:]):
        growth.append((cur - prev) / max(abs(prev), 1.0))
    return growth


def _reverse_roic_cross_check(
    market_price: float | None,
    shares: float,
    net_debt: float,
    economics: dict[str, Any],
    wacc: float,
) -> dict[str, float | None]:
    if market_price is None:
        return {"market_implied_growth": None, "market_implied_terminal_growth": None}
    ev = market_price * shares + net_debt
    base_fcf = economics["explicit_forecast"][0]["fcf"]
    implied_growth = (ev / max(base_fcf * 4.0 / max(wacc - 0.025, 0.01), 1.0)) ** (1.0 / 10.0) - 1.0
    implied_terminal = min(max(wacc - (base_fcf * 4.0 / max(ev, 1.0)), 0.0), wacc - 0.01)
    return {
        "market_implied_growth": float(np.clip(implied_growth, -0.30, 0.40)),
        "market_implied_terminal_growth": float(implied_terminal),
    }


def _confidence_score(warnings: list[str], economics: dict[str, Any], settings: Settings) -> dict[str, float | str]:
    score = 100.0 - min(len(warnings) * 2.0, 30.0)
    roic = economics["initial_roic"]
    terminal_roic = economics["terminal_roic"]
    if roic <= 0 or roic > 0.60:
        score -= 20.0
    if terminal_roic > roic:
        score -= 15.0
    if economics["terminal_growth"] > 0.04:
        score -= 10.0
    score -= {"marketplace_hybrid": 20.0, "platform": 10.0, "hyperscaler": 12.0}.get(settings.company_type, 5.0)
    score = float(np.clip(score, 0.0, 100.0))
    return {"score": score, "bucket": "high" if score >= 75 else "medium" if score >= 50 else "low"}


def _roic_sensitivity(
    economics: dict[str, Any],
    wacc: float,
    net_debt: float,
    shares: float,
) -> dict[str, object]:
    terminal_growth_values = [0.015, economics["terminal_growth"], min(economics["terminal_growth"] + 0.01, wacc - 0.01)]
    rows = []
    explicit = [row["fcf"] for row in economics["explicit_forecast"]]
    fade = [row["fcf"] for row in economics["fade_schedule"]]
    for wacc_case in [wacc - 0.01, wacc, wacc + 0.01]:
        values = []
        q = wacc_case / 4.0
        for g in terminal_growth_values:
            try:
                pv = present_value_series(explicit, q)
                pv += sum(present_value(fcf, q, len(explicit) + idx) for idx, fcf in enumerate(fade, start=1))
                tv = terminal_value(economics["terminal_fcf"] * 4.0, wacc_case, min(g, wacc_case - 0.01))
                pv += present_value(tv, q, len(explicit) + len(fade))
                values.append((pv - net_debt) / shares)
            except Exception:
                values.append(None)
        rows.append({"annual_wacc": wacc_case, "values": values})
    return {"annual_terminal_growth_values": terminal_growth_values, "rows": rows}
