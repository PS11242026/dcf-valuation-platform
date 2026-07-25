"""Per-driver fade half-lives (engine_v3).

The fade model reverts each economic primitive toward its long-run anchor at its OWN
speed. Rather than a single CAP-derived time constant shared by growth and margin, this
module estimates a company-specific half-life for revenue growth and for operating
margin from the company's own history, then shrinks each toward a sector prior
(empirical Bayes -- the same machinery as the ROIC-persistence fit).

A half-life ``h`` is, literally, the number of YEARS for half of the current gap to the
anchor to close::

    x(t) = anchor + (x0 - anchor) * 0.5 ** (t / h)     # t in years

Estimation is a through-origin AR(1) on deviations from the anchor:

- growth: annual non-overlapping TTM-over-TTM readings, deviations from ``g_terminal``;
- margin: quarterly NOPAT-margin readings, deviations from the blended margin target,
  with the quarterly persistence annualized (``rho_annual = rho_quarter ** 4``).

``rho`` in (0, 1) ⇒ ``h = ln(0.5) / ln(rho)``. Anything else -- too few points, ``rho<=0``
(no clean mean reversion) or ``rho>=1`` (a trend, not reversion) -- falls back to the
CAP-derived half-life, flagged in the audit so the assumption is never silent. The
CAP fallback equals the engine's pre-half-life shared time constant, so a name with no
identifiable own-speed fades exactly as before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.drivers import YOY_GROWTH_BAND
from dcf_project.engine_v3.persistence import PersistenceFit
from dcf_project.engine_v3.sector_priors import SectorPrior

#: Minimum AR(1) pairs (one less than readings) for an own-company half-life estimate.
MIN_AR1_PAIRS = 4


@dataclass(frozen=True)
class HalfLifeEstimates:
    """The fade half-lives (years) actually used, plus a per-driver estimation audit."""

    h_growth_years: float
    h_margin_years: float
    growth_audit: dict[str, Any] = field(default_factory=dict)
    margin_audit: dict[str, Any] = field(default_factory=dict)


def fallback_half_life_years(cap_quarters: float, cfg: V3Config) -> float:
    """CAP-derived half-life used when a driver's own reversion is not identifiable.

    Equals the legacy shared fade time constant ``clip(CAP/ln10, [4, 40] quarters)``
    re-expressed as a half-life (``h = tau · ln2 / 4``), so a no-estimate driver fades
    exactly as the pre-half-life engine did. Clipped to the configured half-life bounds
    (a no-op over the [4, 40]q range; present for safety).
    """
    tau_q = float(np.clip(cap_quarters / math.log(10.0), 4.0, 40.0))
    h = tau_q * math.log(2.0) / 4.0
    return float(np.clip(h, cfg.half_life_min_years, cfg.half_life_max_years))


def _prior_rho_from_half_life(h_years: float) -> float:
    """Annual persistence implied by a half-life: rho such that rho**h = 0.5."""
    return float(0.5 ** (1.0 / h_years)) if h_years > 0 else 0.0


def _through_origin_ar1(dev: np.ndarray) -> tuple[float | None, int]:
    """rho_hat for dev_{t+1} = rho·dev_t (no intercept); None if unidentifiable."""
    n_pairs = max(dev.size - 1, 0)
    if dev.size < MIN_AR1_PAIRS + 1:
        return None, n_pairs
    x, y = dev[:-1], dev[1:]
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return None, n_pairs
    return float(np.dot(x, y) / denom), n_pairs


def _resolve_half_life(
    rho_company: float | None,
    n_pairs: int,
    sector_half_life: float,
    cap_quarters: float,
    cfg: V3Config,
    *,
    annualize_power: int = 1,
) -> tuple[float, dict[str, Any]]:
    """EB-shrink the company rho toward the sector prior; convert to a clipped half-life.

    ``annualize_power`` raises a quarterly rho to annual (4); 1 for an already-annual rho.
    The empirical-Bayes weight is ``n / (n + n0)`` (same form as the persistence fit).
    """
    sector_rho = _prior_rho_from_half_life(sector_half_life)
    fallback_h = fallback_half_life_years(cap_quarters, cfg)
    rho_company_annual: float | None = None
    if rho_company is not None and 0.0 < rho_company < 1.0:
        rho_company_annual = float(rho_company ** annualize_power)

    if rho_company_annual is not None and 0.0 < rho_company_annual < 1.0:
        n0 = max(cfg.half_life_prior_strength, 1e-9)
        weight = n_pairs / (n_pairs + n0)
        rho = weight * rho_company_annual + (1.0 - weight) * sector_rho
        rho = float(np.clip(rho, 1e-6, 0.999999))
        h = float(np.clip(math.log(0.5) / math.log(rho),
                          cfg.half_life_min_years, cfg.half_life_max_years))
        source = "company_ar1_shrunk"
        shrink_weight = float(weight)
    else:
        h = fallback_h
        source = "fallback_cap"
        shrink_weight = 0.0

    audit = {
        "h_used_years": float(h),
        "source": source,
        "rho_company_annual": rho_company_annual,
        "rho_sector_prior": float(sector_rho),
        "sector_half_life_years": float(sector_half_life),
        "n_pairs": int(n_pairs),
        "shrink_weight": shrink_weight,
        "fallback_half_life_years": float(fallback_h),
    }
    return h, audit


def estimate_half_lives(
    history: pd.DataFrame,
    g_terminal: float,
    margin_target_blend: float,
    sector_prior: SectorPrior,
    persistence: PersistenceFit,
    cfg: V3Config,
) -> HalfLifeEstimates:
    """Estimate the growth and margin fade half-lives from one company's history."""
    revenue = history["revenue"].astype(float)
    nopat = history["nopat"].astype(float)
    ttm_revenue = revenue.rolling(4).sum()

    # --- growth: annual non-overlapping TTM-over-TTM, deviations from g_terminal.
    yoy = (ttm_revenue / ttm_revenue.shift(4) - 1.0).replace([np.inf, -np.inf], np.nan)
    yoy_vals = yoy.clip(*YOY_GROWTH_BAND).dropna().to_numpy(dtype=float)
    annual_growth = yoy_vals[::-1][::4][::-1] if yoy_vals.size else np.empty(0)
    rho_g, n_g = _through_origin_ar1(annual_growth - float(g_terminal))
    h_growth, growth_audit = _resolve_half_life(
        rho_g, n_g, sector_prior.growth_half_life_years, persistence.cap_quarters, cfg,
        annualize_power=1,
    )

    # --- margin: TTM NOPAT margin sampled annually (non-overlapping), deviations from the
    # blended target. Mirrors the growth treatment exactly. A raw per-quarter margin is
    # seasonal -- its lag-1 autocorrelation is dominated by the quarterly cost/tax cycle,
    # not economic reversion (KO's quarterly margin AR(1) is *negative*) -- which collapsed
    # the estimated half-life to the floor and reverted the margin almost instantly. The
    # TTM margin is annual by construction, so rho is already annual (annualize_power=1).
    ttm_nopat = nopat.rolling(4).sum()
    ttm_margin = (ttm_nopat / ttm_revenue).replace([np.inf, -np.inf], np.nan)
    margin_vals = ttm_margin.dropna().to_numpy(dtype=float)
    annual_margin = margin_vals[::-1][::4][::-1] if margin_vals.size else np.empty(0)
    rho_m, n_m = _through_origin_ar1(annual_margin - float(margin_target_blend))
    h_margin, margin_audit = _resolve_half_life(
        rho_m, n_m, sector_prior.margin_half_life_years, persistence.cap_quarters, cfg,
        annualize_power=1,
    )

    return HalfLifeEstimates(
        h_growth_years=h_growth,
        h_margin_years=h_margin,
        growth_audit=growth_audit,
        margin_audit=margin_audit,
    )
