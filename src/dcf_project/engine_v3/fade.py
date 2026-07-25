"""The deterministic economic skeleton: fade-based forecast (engine_v3).

Primitives (the over-determination rule): the forecast is fully specified by the
revenue-growth path g(t) and the NOPAT-margin path m(t), both mean-reverting, plus the
persistence-faded RONIC path that prices growth. Everything else is derived:

    reinvestment_rate(t) = g(t) / RONIC(t)        (sustainable-growth identity)
    FCF(t)               = NOPAT(t) · (1 − g(t)/RONIC(t))
    IC(t+1)              = IC(t) + reinvestment(t)

Book ROIC is *emergent* from the accumulated capital and reported as a coherence
diagnostic -- it is never imposed. The implied TTM FCF margin is checked against the
company's historical p10–p90 band (the tripwire for an incoherent driver set).

Fade speed is tied to the Competitive Advantage Period: the growth/margin excess decays
to ~10% of its initial value by the end of the CAP (tau = CAP / ln 10). Unit rule:
internals are annual; quarterly conversion happens only at frame emission.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.drivers import DriverEstimates
from dcf_project.engine_v3.halflives import HalfLifeEstimates, fallback_half_life_years
from dcf_project.engine_v3.persistence import PersistenceFit
from dcf_project.engine_v3.sector_priors import SectorPrior

TAU_BOUNDS_QUARTERS = (4.0, 40.0)
RONIC_FLOOR = 0.02
REINVESTMENT_FLOOR = -0.25       # divestment bound: asset sales can fund FCF, within reason
FCF_MARGIN_BAND_TOLERANCE = 0.03


def fade_time_constant(cap_quarters: float) -> float:
    """Quarters for the growth/margin excess to decay by e; ~CAP/ln(10) so the excess
    is ~10% of initial at the CAP endpoint."""
    return float(np.clip(cap_quarters / math.log(10.0), *TAU_BOUNDS_QUARTERS))


def build_fade_forecast(
    drivers: DriverEstimates,
    persistence: PersistenceFit,
    sector_prior: SectorPrior,
    cfg: V3Config,
    wacc: float,
    g_terminal: float,
    half_lives: HalfLifeEstimates | None = None,
    horizon: int = 40,
    sector_tilt: float = 0.0,
    g_base_override: float | None = None,
    margin_target_override: float | None = None,
    phi_override: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the explicit-horizon forecast frame + the driver-audit fragment.

    Growth and margin each fade toward their anchors at their OWN company-specific
    half-life (``half_lives``); when not supplied they share the CAP-derived fallback
    half-life (the engine's pre-half-life behaviour). The ``*_override`` arguments let
    the Monte Carlo layer re-run the identical deterministic skeleton under sampled
    drivers (zero-variance draws must reproduce the skeleton exactly -- a tested
    invariant), so the half-lives are held fixed across draws.
    """
    g_base = drivers.g_base.value if g_base_override is None else float(g_base_override)
    g_base = g_base + float(sector_tilt)
    phi = persistence.phi if phi_override is None else float(np.clip(phi_override, 0.0, 0.99))

    # Margin target: company normalization blended toward the sector anchor.
    company_target = (
        drivers.margin_target_company.value if margin_target_override is None else float(margin_target_override)
    )
    w = float(sector_prior.anchor_weight)
    margin_target = (1.0 - w) * company_target + w * sector_prior.margin_anchor

    # Per-driver fade speeds: each primitive reverts at its own half-life (years for half
    # the gap to close). Absent an estimate, both fall back to the CAP-derived half-life.
    if half_lives is None:
        h_growth = h_margin = fallback_half_life_years(persistence.cap_quarters, cfg)
        growth_hl_audit: dict[str, Any] | None = None
        margin_hl_audit: dict[str, Any] | None = None
    else:
        h_growth = float(half_lives.h_growth_years)
        h_margin = float(half_lives.h_margin_years)
        growth_hl_audit = half_lives.growth_audit
        margin_hl_audit = half_lives.margin_audit

    quarters = np.arange(1, horizon + 1, dtype=float)
    t_years = quarters / 4.0
    g_annual = g_terminal + (g_base - g_terminal) * 0.5 ** (t_years / h_growth)
    margin = margin_target + (drivers.margin_current - margin_target) * 0.5 ** (t_years / h_margin)
    ronic = np.maximum(wacc + persistence.excess_0 * phi ** (quarters / 4.0), RONIC_FLOOR)

    # Sustainable-growth bind: growth is paid for with capital at RONIC. Reinvestment
    # above the cap means the implied growth is unfundable -> reduce growth, flag it.
    growth_binds: list[int] = []
    divestment_binds: list[int] = []
    reinvestment_rate = g_annual / ronic
    for i in range(horizon):
        if reinvestment_rate[i] > cfg.reinvestment_cap:
            g_annual[i] = cfg.reinvestment_cap * ronic[i]
            reinvestment_rate[i] = cfg.reinvestment_cap
            growth_binds.append(i + 1)
        elif reinvestment_rate[i] < REINVESTMENT_FLOOR:
            reinvestment_rate[i] = REINVESTMENT_FLOOR
            g_annual[i] = REINVESTMENT_FLOOR * ronic[i]
            divestment_binds.append(i + 1)

    # --- Quarterly emission (the only annual->quarterly conversion in the engine).
    g_quarterly = (1.0 + g_annual) ** 0.25 - 1.0
    base_revenue_q = drivers.base_revenue_ttm / 4.0  # seasonal-neutral run-rate
    rows: list[dict[str, float]] = []
    revenue_prev = base_revenue_q
    ic = drivers.base_invested_capital
    nopat_trail: list[float] = []
    for i in range(horizon):
        revenue_q = revenue_prev * (1.0 + g_quarterly[i])
        nopat_q = revenue_q * margin[i]
        reinvestment_q = nopat_q * reinvestment_rate[i]
        fcf_q = nopat_q - reinvestment_q
        ic = ic + reinvestment_q
        nopat_trail.append(nopat_q)
        trailing = nopat_trail[-4:]
        emergent_roic = (sum(trailing) * (4.0 / len(trailing))) / ic if ic > 0 else float("nan")
        rows.append(
            {
                "quarter": i + 1,
                "revenue": revenue_q,
                "revenue_growth": g_quarterly[i],
                "growth_annual": g_annual[i],
                "ebit": nopat_q / (1.0 - drivers.tax_rate) if drivers.tax_rate < 1.0 else nopat_q,
                "operating_margin": margin[i] / (1.0 - drivers.tax_rate) if drivers.tax_rate < 1.0 else margin[i],
                "nopat": nopat_q,
                "nopat_margin": margin[i],
                "ronic": ronic[i],
                "reinvestment_rate": reinvestment_rate[i],
                "reinvestment": reinvestment_q,
                "fcf": fcf_q,
                "fcf_margin": fcf_q / revenue_q if revenue_q else float("nan"),
                "invested_capital": ic,
                "roic": emergent_roic,
                "tax_rate": drivers.tax_rate,
            }
        )
        revenue_prev = revenue_q
    frame = pd.DataFrame(rows)

    # --- Coherence tripwire: implied TTM FCF margin vs the historical p10-p90 band.
    ttm_fcf_margin = (
        frame["fcf"].rolling(4).sum() / frame["revenue"].rolling(4).sum()
    ).dropna()
    band = drivers.fcf_margin_band
    breach_quarters: list[int] = []
    if np.isfinite(band.get("p10", float("nan"))) and np.isfinite(band.get("p90", float("nan"))):
        low = band["p10"] - FCF_MARGIN_BAND_TOLERANCE
        high = band["p90"] + FCF_MARGIN_BAND_TOLERANCE
        for idx, value in ttm_fcf_margin.items():
            if not (low <= value <= high):
                breach_quarters.append(int(frame.loc[idx, "quarter"]))

    audit: dict[str, Any] = {
        "g_base": drivers.g_base.value,
        "g_base_effective": float(g_base),
        "sector_tilt": float(sector_tilt),
        "g_terminal": float(g_terminal),
        "half_life_growth_years": float(h_growth),
        "half_life_margin_years": float(h_margin),
        "half_life_growth_audit": growth_hl_audit,
        "half_life_margin_audit": margin_hl_audit,
        "margin_current": drivers.margin_current,
        "margin_target_company": company_target,
        "margin_target_blend": float(margin_target),
        "sector_anchor_weight": w,
        "sector_margin_anchor": sector_prior.margin_anchor,
        "phi": float(phi),
        "excess_roic_0": persistence.excess_0,
        "cap_years": persistence.cap_years,
        "ronic_terminal_explicit_end": float(ronic[-1]),
        "growth_binds_quarters": growth_binds,
        "divestment_binds_quarters": divestment_binds,
        "fcf_margin_band": band,
        "implied_fcf_margin_final_ttm": float(ttm_fcf_margin.iloc[-1]) if len(ttm_fcf_margin) else None,
        "fcf_margin_breach_quarters": breach_quarters,
        "paths": {
            "growth_annual": [float(x) for x in g_annual],
            "nopat_margin": [float(x) for x in margin],
            "ronic": [float(x) for x in ronic],
            "reinvestment_rate": [float(x) for x in reinvestment_rate],
        },
    }
    return frame, audit
