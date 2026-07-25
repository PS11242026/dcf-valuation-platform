"""Economic-driver extraction from the reconstructed SEC history (engine_v3).

Input is the frozen v1 reconstruction frame (``preprocessing.reconstruction``):
quarterly rows with revenue, nopat, fcf, invested_capital, tax_rate, and data-quality
flags. Output is a small set of ANNUALIZED driver estimates, each carrying a point
value and a standard error (the parameter uncertainty the Monte Carlo layer samples).

Unit convention (engine_v3 rule): everything here is annual. Quarterly history is
annualized via trailing-twelve-month (TTM) aggregates -- never via the ``×4`` shortcut --
and conversion back to quarterly happens only when the forecast frame is emitted
(``fade.build_fade_forecast``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


#: Sanity bands on annualized drivers (wide; these catch data errors, not opinions).
YOY_GROWTH_BAND = (-0.50, 1.00)
ROIC_ANNUAL_BAND = (-0.25, 0.60)
MIN_SE_GROWTH = 0.010
MIN_SE_MARGIN = 0.005


@dataclass(frozen=True)
class DriverEstimate:
    value: float
    se: float
    window: str


@dataclass(frozen=True)
class DriverEstimates:
    """Annualized economic drivers + uncertainty, extracted from one company's history."""

    g_base: DriverEstimate                  # base annual revenue growth (TTM-over-TTM)
    margin_current: float                   # latest TTM NOPAT margin
    margin_target_company: DriverEstimate   # normalized NOPAT margin (trailing-12 TTM median)
    excess_roic_annual: pd.Series           # TTM NOPAT / IC − WACC, quarterly readings
    roic_annual_0: float                    # current annual ROIC (median of last 4 readings)
    excess_0: float                         # roic_annual_0 − WACC
    fcf_margin_band: dict[str, float]       # p10/p50/p90 of TTM FCF / TTM revenue
    fcf_margin_current: float | None
    base_revenue_ttm: float
    base_nopat_ttm: float
    base_invested_capital: float
    tax_rate: float
    quality: float                          # 0 = clean history … 1 = heavily flagged
    notes: list[str] = field(default_factory=list)


class DriverExtractionError(ValueError):
    """History cannot support an economically meaningful driver set."""


def extract_drivers(history: pd.DataFrame, wacc: float) -> DriverEstimates:
    notes: list[str] = []
    if len(history) < 20:
        raise DriverExtractionError("engine_v3 requires at least 20 complete quarters")

    revenue = history["revenue"].astype(float)
    nopat = history["nopat"].astype(float)
    fcf = history["fcf"].astype(float)
    ic = history["invested_capital"].astype(float)

    ttm_revenue = revenue.rolling(4).sum()
    ttm_nopat = nopat.rolling(4).sum()
    ttm_fcf = fcf.rolling(4).sum()

    base_revenue_ttm = float(ttm_revenue.iloc[-1])
    base_nopat_ttm = float(ttm_nopat.iloc[-1])
    if not np.isfinite(base_revenue_ttm) or base_revenue_ttm <= 0:
        raise DriverExtractionError("Latest TTM revenue is non-positive")
    if not np.isfinite(base_nopat_ttm) or base_nopat_ttm <= 0:
        # Typed skip condition: a fade/value-driver model needs positive normalized
        # operating profit to anchor on. Negative-NOPAT names are out of scope for v3.
        raise DriverExtractionError("v3_not_applicable: non-positive TTM NOPAT")

    # --- Revenue growth: TTM-over-TTM (seasonality-proof), annual by construction.
    yoy = (ttm_revenue / ttm_revenue.shift(4) - 1.0).replace([np.inf, -np.inf], np.nan)
    yoy = yoy.clip(*YOY_GROWTH_BAND).dropna()
    if len(yoy) < 4:
        raise DriverExtractionError("Insufficient history for TTM revenue growth")
    g_recent = yoy.tail(4)
    g_base = float(g_recent.mean())
    g_se = float(yoy.tail(8).std(ddof=1) / np.sqrt(min(len(yoy), 8)))
    g_se = max(g_se if np.isfinite(g_se) else MIN_SE_GROWTH, MIN_SE_GROWTH)

    # --- NOPAT margin: current TTM level + the normalization target it reverts toward.
    # Both are TTM-based (the module's unit rule). A raw per-quarter NOPAT margin is
    # heavily seasonal -- KO swings ~10pp quarter to quarter (e.g. 0.10 next to 0.30) --
    # so its trailing median sits well below the revenue-weighted TTM level the fade
    # reverts FROM, imposing a spurious ~6pp margin compression. The TTM margin series is
    # seasonality-proof and unit-consistent with margin_current.
    ttm_margin = (ttm_nopat / ttm_revenue).replace([np.inf, -np.inf], np.nan)
    margin_current = base_nopat_ttm / base_revenue_ttm
    margin_window = ttm_margin.tail(12).dropna()
    margin_target = float(margin_window.median())
    margin_se = float(margin_window.std(ddof=1) / np.sqrt(max(len(margin_window), 1)))
    margin_se = max(margin_se if np.isfinite(margin_se) else MIN_SE_MARGIN, MIN_SE_MARGIN)

    # --- Annual ROIC: TTM NOPAT over current invested capital (no ×4 shortcut).
    ic_valid = ic.where(ic > 0)
    roic_annual = (ttm_nopat / ic_valid).replace([np.inf, -np.inf], np.nan)
    roic_annual = roic_annual.clip(*ROIC_ANNUAL_BAND)
    roic_readings = roic_annual.dropna()
    if len(roic_readings) < 4:
        raise DriverExtractionError("Insufficient invested-capital history for annual ROIC")
    roic_annual_0 = float(roic_readings.tail(4).median())
    excess_series = (roic_annual - wacc).dropna()
    excess_0 = roic_annual_0 - wacc

    # --- Historical FCF-margin band: the coherence tripwire the forecast is checked against.
    fcf_margin = (ttm_fcf / ttm_revenue).replace([np.inf, -np.inf], np.nan).dropna()
    if len(fcf_margin) >= 8:
        band = {
            "p10": float(fcf_margin.quantile(0.10)),
            "p50": float(fcf_margin.quantile(0.50)),
            "p90": float(fcf_margin.quantile(0.90)),
        }
    else:
        band = {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        notes.append("FCF-margin band unavailable (<8 TTM readings)")
    fcf_margin_current = float(fcf_margin.iloc[-1]) if len(fcf_margin) else None

    # --- Data-quality score: flagged invested-capital discontinuities + stale stocks
    # lower trust in the company's own series (raising the sector-prior weight downstream).
    flag_share = 0.0
    if "invested_capital_continuity_flag" in history:
        flag_share = float(history["invested_capital_continuity_flag"].fillna(False).mean())
    stale_share = 0.0
    stale_cols = [c for c in history.columns if c.endswith("_stale_carry_forward_flag")]
    if stale_cols:
        stale_share = float(history[stale_cols].fillna(False).any(axis=1).mean())
    quality = float(np.clip(2.0 * flag_share + 0.5 * stale_share, 0.0, 1.0))
    if quality > 0.25:
        notes.append(f"History quality penalty {quality:.2f} (IC flags {flag_share:.2f}, stale {stale_share:.2f})")

    return DriverEstimates(
        g_base=DriverEstimate(g_base, g_se, "mean of last 4 TTM-over-TTM growth readings"),
        margin_current=float(margin_current),
        margin_target_company=DriverEstimate(margin_target, margin_se, "median of last 12 TTM NOPAT margins"),
        excess_roic_annual=excess_series,
        roic_annual_0=roic_annual_0,
        excess_0=float(excess_0),
        fcf_margin_band=band,
        fcf_margin_current=fcf_margin_current,
        base_revenue_ttm=base_revenue_ttm,
        base_nopat_ttm=base_nopat_ttm,
        base_invested_capital=float(ic.iloc[-1]),
        tax_rate=float(history["tax_rate"].iloc[-1]),
        quality=quality,
        notes=notes,
    )
