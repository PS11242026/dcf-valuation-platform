"""ROIC persistence / CAP tests (engine_v3): exact recovery, shrinkage limits, caps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.persistence import EXCESS_FLOOR, fit_persistence

CFG = V3Config()


def geometric_series(phi: float, excess_0: float = 0.08, quarters: int = 40) -> pd.Series:
    """Quarterly TTM excess readings whose every-4th sampling is an exact AR(1) decay."""
    t = np.arange(quarters, dtype=float)
    return pd.Series(excess_0 * phi ** (t / 4.0))


def test_recovers_phi_exactly_from_noiseless_decay() -> None:
    phi_true = 0.85
    fit = fit_persistence(geometric_series(phi_true), 0.08, "technology", phi_true, 0.0, CFG)
    # Sector prior set equal to truth so shrinkage cannot mask a wrong company fit.
    assert fit.phi_company == pytest.approx(phi_true, rel=1e-12)
    assert fit.phi == pytest.approx(phi_true, rel=1e-12)
    assert fit.n_years == 9  # 10 non-overlapping annual readings -> 9 pairs


def test_shrinkage_weight_limits() -> None:
    phi_true = 0.90
    sector_phi = 0.60

    # Too little history: company fit unavailable -> prior carries everything.
    short = geometric_series(phi_true, quarters=8)
    fit = fit_persistence(short, 0.08, "technology", sector_phi, 0.0, CFG)
    assert fit.phi_company is None
    assert fit.shrink_weight == 0.0
    assert fit.phi == pytest.approx(sector_phi)
    assert any("sector prior" in n for n in fit.notes)

    # Moderate history: weight = n/(n + n0) between the two anchors.
    fit_mid = fit_persistence(geometric_series(phi_true, quarters=40), 0.08, "technology", sector_phi, 0.0, CFG)
    assert fit_mid.shrink_weight == pytest.approx(9 / (9 + CFG.phi_prior_strength))
    assert sector_phi < fit_mid.phi < phi_true

    # Long history: company estimate dominates.
    fit_long = fit_persistence(geometric_series(phi_true, quarters=400), 0.08, "technology", sector_phi, 0.0, CFG)
    assert fit_long.shrink_weight > fit_mid.shrink_weight
    assert abs(fit_long.phi - phi_true) < abs(fit_mid.phi - phi_true)


def test_quality_penalty_raises_prior_weight() -> None:
    series = geometric_series(0.90, quarters=40)
    clean = fit_persistence(series, 0.08, "technology", 0.60, 0.0, CFG)
    dirty = fit_persistence(series, 0.08, "technology", 0.60, 1.0, CFG)
    # quality=1 -> n0 tripled -> less weight on the company estimate.
    assert dirty.shrink_weight == pytest.approx(9 / (9 + 3 * CFG.phi_prior_strength))
    assert dirty.shrink_weight < clean.shrink_weight
    assert abs(dirty.phi - 0.60) < abs(clean.phi - 0.60)


def test_phi_clipped_to_unit_band() -> None:
    # Explosive series (excess grows 20%/yr) with an (artificially) explosive prior:
    # the fit must still be clipped below 1.
    t = np.arange(40, dtype=float)
    explosive = pd.Series(0.02 * 1.2 ** (t / 4.0))
    fit = fit_persistence(explosive, 0.10, "technology", 0.99, 0.0, CFG)
    assert fit.phi_company == pytest.approx(1.2, rel=1e-12)
    assert fit.phi <= 0.99


def test_cap_zero_when_no_advantage() -> None:
    fit = fit_persistence(geometric_series(0.85), EXCESS_FLOOR, "energy", 0.55, 0.0, CFG)
    assert fit.cap_years == 0.0
    neg = fit_persistence(geometric_series(0.85, excess_0=0.08), -0.03, "energy", 0.55, 0.0, CFG)
    assert neg.cap_years == 0.0
    assert neg.excess_terminal == 0.0  # losses are not projected into the perpetuity


def test_cap_formula_and_ceiling() -> None:
    phi, excess_0 = 0.80, 0.06
    fit = fit_persistence(geometric_series(phi, excess_0), excess_0, "industrials", phi, 0.0, CFG)
    assert fit.cap_years == pytest.approx(np.log(EXCESS_FLOOR / excess_0) / np.log(phi), rel=1e-9)
    assert fit.cap_quarters == pytest.approx(fit.cap_years * 4.0)

    # Near-unit persistence with a big advantage hits the ceiling.
    sticky = fit_persistence(geometric_series(0.99, 0.30), 0.30, "technology", 0.99, 0.0, CFG)
    assert sticky.cap_years == CFG.cap_max_years


def test_terminal_excess_capped_at_moat_premium() -> None:
    fit = fit_persistence(geometric_series(0.99, 0.30), 0.30, "technology", 0.99, 0.0, CFG, horizon_years=5.0)
    assert fit.excess_terminal == CFG.moat_premium_cap  # 0.30 * 0.99^5 >> cap

    modest = fit_persistence(geometric_series(0.60, 0.04), 0.04, "materials", 0.60, 0.0, CFG, horizon_years=5.0)
    expected = max(0.04, 0.0) * modest.phi**5.0
    assert modest.excess_terminal == pytest.approx(min(expected, CFG.moat_premium_cap), rel=1e-9)
