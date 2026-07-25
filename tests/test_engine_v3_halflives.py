"""Per-driver fade half-life tests (engine_v3.halflives): AR(1) estimation, empirical-
Bayes shrinkage, bound clipping, the CAP fallback, and the legacy-tau equivalence."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.fade import fade_time_constant
from dcf_project.engine_v3.halflives import (
    _prior_rho_from_half_life,
    _resolve_half_life,
    _through_origin_ar1,
    estimate_half_lives,
    fallback_half_life_years,
)
from dcf_project.engine_v3.persistence import PersistenceFit
from dcf_project.engine_v3.sector_priors import SectorPrior

CFG = V3Config()
PRIOR = SectorPrior(
    phi=0.85, margin_anchor=0.15, roic_anchor=0.15,
    growth_half_life_years=3.0, margin_half_life_years=4.0,
)


def make_persistence(cap_years: float = 8.0, phi: float = 0.85) -> PersistenceFit:
    return PersistenceFit(
        phi=phi, phi_se=0.05, phi_company=phi, n_years=6, shrink_weight=0.5,
        excess_0=0.06, excess_terminal=0.01, cap_years=cap_years,
        cap_quarters=cap_years * 4.0, sector="technology", notes=[],
    )


# ----------------------------- helper math ----------------------------- #

def test_prior_rho_half_life_roundtrip() -> None:
    for h in (1.0, 3.0, 6.0):
        rho = _prior_rho_from_half_life(h)
        assert math.log(0.5) / math.log(rho) == pytest.approx(h, rel=1e-12)


def test_through_origin_ar1_recovers_known_rho() -> None:
    rho = 0.7
    dev = np.array([0.1 * rho**k for k in range(8)])  # exact geometric decay
    est, n_pairs = _through_origin_ar1(dev)
    assert est == pytest.approx(rho, rel=1e-12)
    assert n_pairs == 7


def test_through_origin_ar1_insufficient_or_degenerate() -> None:
    assert _through_origin_ar1(np.array([0.1, 0.2, 0.3])) == (None, 2)  # < 5 readings
    est, _ = _through_origin_ar1(np.zeros(8))                            # no variance
    assert est is None


def test_resolve_half_life_shrinks_toward_prior() -> None:
    rho_company = _prior_rho_from_half_life(6.0)  # a 6y own half-life
    h, audit = _resolve_half_life(
        rho_company, 4, sector_half_life=3.0, cap_quarters=32.0, cfg=CFG, annualize_power=1
    )
    assert audit["source"] == "company_ar1_shrunk"
    assert audit["shrink_weight"] == pytest.approx(4 / (4 + CFG.half_life_prior_strength))
    assert 3.0 < h < 6.0  # shrunk between the 3y prior and the 6y company estimate


def test_resolve_half_life_clips_and_falls_back() -> None:
    h_long, _ = _resolve_half_life(0.999999, 50, 6.0, 32.0, CFG, annualize_power=1)
    assert h_long == pytest.approx(CFG.half_life_max_years)  # near-unit rho -> clipped
    h_fb, audit_fb = _resolve_half_life(1.5, 6, 4.0, 32.0, CFG, annualize_power=1)
    assert audit_fb["source"] == "fallback_cap"  # rho outside (0,1)
    assert h_fb == pytest.approx(fallback_half_life_years(32.0, CFG))


def test_faster_reversion_shorter_half_life() -> None:
    h_fast, _ = _resolve_half_life(_prior_rho_from_half_life(1.0), 50, 1.0, 32.0, CFG)
    h_slow, _ = _resolve_half_life(_prior_rho_from_half_life(7.0), 50, 7.0, 32.0, CFG)
    assert h_fast < h_slow


def test_fallback_matches_legacy_tau() -> None:
    """The CAP fallback equals the legacy shared fade time constant as a half-life."""
    for cap_q in (8.0, 32.0, 200.0, 0.0):
        tau = fade_time_constant(cap_q)
        expected = float(np.clip(tau * math.log(2.0) / 4.0,
                                 CFG.half_life_min_years, CFG.half_life_max_years))
        assert fallback_half_life_years(cap_q, CFG) == pytest.approx(expected, rel=1e-12)


# ----------------------------- integration ----------------------------- #

def _history(margin_of, q_growth: float = 0.03, n: int = 32) -> pd.DataFrame:
    rev = [100.0]
    for _ in range(n - 1):
        rev.append(rev[-1] * (1.0 + q_growth))
    revenue = np.array(rev)
    nopat = revenue * np.array([margin_of(i) for i in range(n)])
    return pd.DataFrame({"revenue": revenue, "nopat": nopat})


def test_constant_history_both_fall_back() -> None:
    hist = _history(lambda i: 0.20)  # constant margin + geometric revenue = degenerate
    est = estimate_half_lives(hist, g_terminal=0.03, margin_target_blend=0.15,
                              sector_prior=PRIOR, persistence=make_persistence(), cfg=CFG)
    assert est.growth_audit["source"] == "fallback_cap"
    assert est.margin_audit["source"] == "fallback_cap"
    fb = fallback_half_life_years(make_persistence().cap_quarters, CFG)
    assert est.h_growth_years == pytest.approx(fb)
    assert est.h_margin_years == pytest.approx(fb)


def test_mean_reverting_margin_gets_company_estimate() -> None:
    # Margin decays 0.27 -> 0.15 geometrically (rho_q=0.85); revenue geometric so growth
    # has no own reversion signal and must fall back.
    hist = _history(lambda i: 0.15 + 0.12 * 0.85**i)
    est = estimate_half_lives(hist, g_terminal=0.03, margin_target_blend=0.15,
                              sector_prior=PRIOR, persistence=make_persistence(), cfg=CFG)
    assert est.margin_audit["source"] == "company_ar1_shrunk"
    assert CFG.half_life_min_years <= est.h_margin_years <= CFG.half_life_max_years
    assert est.margin_audit["rho_company_annual"] == pytest.approx(0.85**4, rel=1e-6)
    assert est.growth_audit["source"] == "fallback_cap"
