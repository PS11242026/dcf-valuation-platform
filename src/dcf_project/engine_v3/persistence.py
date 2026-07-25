"""ROIC persistence and the Competitive Advantage Period (engine_v3).

Economic mechanism: competition erodes excess returns. Excess ROIC (ROIC − WACC)
follows a mean-zero AR(1) in annual time,

    excess_{y+1} = phi · excess_y + eps,

so phi IS the durability of the moat. The company's own phi is estimated from
non-overlapping annual readings and shrunk (empirical Bayes) toward its sector prior --
more history and cleaner data earn more weight on the company estimate.

Derived quantities (no extra free parameters):
- CAP -- the Competitive Advantage Period -- is the horizon at which the current excess
  has decayed to an economically negligible floor: cap_years = ln(floor/excess_0)/ln(phi).
- The terminal excess that survives the explicit horizon is excess_0 · phi^T, capped at
  the moat ceiling. Terminal RONIC = WACC + that excess. There is NO separate moat knob.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dcf_project.engine_v3.config import V3Config

#: Excess returns below this are treated as competitive equilibrium (CAP endpoint).
EXCESS_FLOOR = 0.005
PHI_BOUNDS = (0.0, 0.99)
MIN_ANNUAL_PAIRS = 4


@dataclass(frozen=True)
class PersistenceFit:
    phi: float                  # shrunk annual persistence of excess ROIC
    phi_se: float
    phi_company: float | None   # raw company OLS estimate (None if insufficient data)
    n_years: int                # non-overlapping annual pairs used
    shrink_weight: float        # weight on the company estimate vs the sector prior
    excess_0: float
    excess_terminal: float      # excess_0·phi^horizon, capped; 0 if excess_0 <= 0
    cap_years: float
    cap_quarters: float
    sector: str | None
    notes: list[str] = field(default_factory=list)


def fit_persistence(
    excess_series: pd.Series,
    excess_0: float,
    sector: str | None,
    sector_phi: float,
    quality: float,
    cfg: V3Config,
    horizon_years: float = 5.0,
) -> PersistenceFit:
    notes: list[str] = []

    # Non-overlapping annual readings: TTM excess sampled every 4th quarter, anchored
    # at the most recent reading (overlapping TTM points would fake an inflated phi).
    values = excess_series.dropna().to_numpy(dtype=float)
    annual = values[::-1][::4][::-1] if values.size else np.empty(0)

    phi_company: float | None = None
    phi_se = 0.15  # prior-level uncertainty when the company fit is unavailable
    n_pairs = max(len(annual) - 1, 0)
    if n_pairs >= MIN_ANNUAL_PAIRS:
        x, y = annual[:-1], annual[1:]
        denom = float(np.dot(x, x))
        if denom > 1e-12:
            # Through-origin AR(1): theory fixes the long-run mean of excess at zero.
            phi_company = float(np.dot(x, y) / denom)
            resid = y - phi_company * x
            dof = max(n_pairs - 1, 1)
            phi_se = float(np.sqrt(np.dot(resid, resid) / dof / denom))
        else:
            notes.append("Excess-ROIC series ~0 throughout; company phi unidentifiable")
    else:
        notes.append(f"Only {n_pairs} non-overlapping annual pairs; sector prior carries the fit")

    # Empirical-Bayes shrinkage: weight = n / (n + n0·(1 + 2·quality_penalty)).
    n0_effective = cfg.phi_prior_strength * (1.0 + 2.0 * float(np.clip(quality, 0.0, 1.0)))
    if phi_company is None:
        weight = 0.0
        phi = float(sector_phi)
    else:
        weight = n_pairs / (n_pairs + n0_effective)
        phi = weight * phi_company + (1.0 - weight) * float(sector_phi)
    phi = float(np.clip(phi, *PHI_BOUNDS))

    # CAP: years for the current excess to decay to the floor. Zero/negative excess
    # means no competitive advantage today; near-unit phi clamps at the ceiling.
    if excess_0 <= EXCESS_FLOOR:
        cap_years = 0.0
    elif phi <= 0.0:
        cap_years = 0.0
    elif phi >= 0.995:
        cap_years = cfg.cap_max_years
    else:
        cap_years = float(np.log(EXCESS_FLOOR / excess_0) / np.log(phi))
        cap_years = float(np.clip(cap_years, 0.0, cfg.cap_max_years))

    # Terminal excess: what the persistence model says survives the explicit horizon.
    # Negative current excess does not project permanent value destruction into the
    # perpetuity -- competition works both ways; terminal RONIC floors at WACC.
    excess_terminal = float(np.clip(max(excess_0, 0.0) * phi**horizon_years, 0.0, cfg.moat_premium_cap))

    return PersistenceFit(
        phi=phi,
        phi_se=float(max(phi_se, 0.02)),
        phi_company=phi_company,
        n_years=n_pairs,
        shrink_weight=float(weight),
        excess_0=float(excess_0),
        excess_terminal=excess_terminal,
        cap_years=cap_years,
        cap_quarters=cap_years * 4.0,
        sector=sector,
        notes=notes,
    )
