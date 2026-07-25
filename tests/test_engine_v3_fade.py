"""Fade forecast tests (engine_v3): identities to machine precision, binds, tripwire.

Also exports the synthetic driver/persistence builders the MC tests reuse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.drivers import DriverEstimate, DriverEstimates
from dcf_project.engine_v3.fade import TAU_BOUNDS_QUARTERS, build_fade_forecast, fade_time_constant
from dcf_project.engine_v3.persistence import PersistenceFit
from dcf_project.engine_v3.sector_priors import SectorPrior

CFG = V3Config()
WACC = 0.09
G_TERMINAL = 0.03


def make_drivers(
    g: float = 0.10,
    g_se: float = 0.0,
    margin_current: float = 0.20,
    margin_target: float = 0.18,
    margin_se: float = 0.0,
    excess_0: float = 0.06,
    fcf_band: tuple[float, float, float] = (0.05, 0.15, 0.25),
) -> DriverEstimates:
    return DriverEstimates(
        g_base=DriverEstimate(g, g_se, "test"),
        margin_current=margin_current,
        margin_target_company=DriverEstimate(margin_target, margin_se, "test"),
        excess_roic_annual=pd.Series([excess_0] * 20),
        roic_annual_0=WACC + excess_0,
        excess_0=excess_0,
        fcf_margin_band={"p10": fcf_band[0], "p50": fcf_band[1], "p90": fcf_band[2]},
        fcf_margin_current=fcf_band[1],
        base_revenue_ttm=400.0,
        base_nopat_ttm=400.0 * margin_current,
        base_invested_capital=1000.0,
        tax_rate=0.21,
        quality=0.0,
    )


def make_persistence(phi: float = 0.85, excess_0: float = 0.06, cap_years: float = 8.0) -> PersistenceFit:
    return PersistenceFit(
        phi=phi,
        phi_se=0.05,
        phi_company=phi,
        n_years=6,
        shrink_weight=0.5,
        excess_0=excess_0,
        excess_terminal=float(np.clip(max(excess_0, 0.0) * phi**5.0, 0.0, CFG.moat_premium_cap)),
        cap_years=cap_years,
        cap_quarters=cap_years * 4.0,
        sector="technology",
        notes=[],
    )


# anchor_weight=0 -> margin target is purely the company's (clean identities).
PRIOR = SectorPrior(phi=0.85, margin_anchor=0.18, roic_anchor=0.15, anchor_weight=0.0)


def build(drivers=None, persistence=None, prior=PRIOR, **kwargs):
    return build_fade_forecast(
        drivers if drivers is not None else make_drivers(),
        persistence if persistence is not None else make_persistence(),
        prior,
        CFG,
        kwargs.pop("wacc", WACC),
        kwargs.pop("g_terminal", G_TERMINAL),
        **kwargs,
    )


def test_sustainable_growth_identities_machine_precision() -> None:
    frame, audit = build()
    assert len(frame) == 40  # v3 default explicit horizon (10 years)
    assert not audit["growth_binds_quarters"] and not audit["divestment_binds_quarters"]
    np.testing.assert_allclose(frame["reinvestment_rate"], frame["growth_annual"] / frame["ronic"], rtol=1e-12)
    np.testing.assert_allclose(frame["reinvestment"], frame["nopat"] * frame["reinvestment_rate"], rtol=1e-12)
    np.testing.assert_allclose(frame["fcf"], frame["nopat"] * (1.0 - frame["reinvestment_rate"]), rtol=1e-12)
    np.testing.assert_allclose(frame["nopat"], frame["revenue"] * frame["nopat_margin"], rtol=1e-12)
    np.testing.assert_allclose(frame["ebit"], frame["nopat"] / (1.0 - 0.21), rtol=1e-12)


def test_revenue_recurrence_and_ic_accumulation() -> None:
    frame, _ = build()
    base_q = 400.0 / 4.0
    expected_rev = base_q * np.cumprod(1.0 + frame["revenue_growth"].to_numpy())
    np.testing.assert_allclose(frame["revenue"], expected_rev, rtol=1e-12)
    expected_ic = 1000.0 + frame["reinvestment"].cumsum()
    np.testing.assert_allclose(frame["invested_capital"], expected_ic, rtol=1e-12)
    # Quarterly emission of an annual rate: (1+g_a)^(1/4)-1.
    np.testing.assert_allclose(
        frame["revenue_growth"], (1.0 + frame["growth_annual"]) ** 0.25 - 1.0, rtol=1e-12
    )


def test_fade_monotone_both_directions() -> None:
    down, _ = build(make_drivers(g=0.20, margin_current=0.25, margin_target=0.18))
    assert np.all(np.diff(down["growth_annual"]) < 0)
    assert np.all(np.diff(down["nopat_margin"]) < 0)
    assert down["growth_annual"].iloc[0] < 0.20 and down["growth_annual"].iloc[-1] > G_TERMINAL

    up, _ = build(make_drivers(g=0.00, margin_current=0.10, margin_target=0.18))
    assert np.all(np.diff(up["growth_annual"]) > 0)
    assert np.all(np.diff(up["nopat_margin"]) > 0)


def test_growth_bind_growth_is_paid_for() -> None:
    # Huge growth on a no-advantage capital base: RONIC ~ floors, reinvestment would
    # exceed the cap -> growth is reduced to what capital can fund, and flagged.
    drivers = make_drivers(g=0.60, excess_0=-0.08)
    persistence = make_persistence(phi=0.85, excess_0=-0.08)
    frame, audit = build(drivers, persistence)
    assert audit["growth_binds_quarters"]
    assert np.all(frame["reinvestment_rate"] <= CFG.reinvestment_cap + 1e-12)
    bound = frame.iloc[0]
    assert bound["growth_annual"] == pytest.approx(CFG.reinvestment_cap * bound["ronic"], rel=1e-12)


def test_divestment_bind_floor() -> None:
    drivers = make_drivers(g=-0.45, excess_0=0.20)
    persistence = make_persistence(phi=0.95, excess_0=0.20)
    frame, audit = build(drivers, persistence)
    assert audit["divestment_binds_quarters"]
    assert np.all(frame["reinvestment_rate"] >= -0.25 - 1e-12)


def test_override_equivalence_for_mc() -> None:
    """Overrides equal to the point estimates must reproduce the skeleton exactly."""
    drivers, persistence = make_drivers(), make_persistence()
    base_frame, base_audit = build(drivers, persistence)
    over_frame, over_audit = build(
        drivers,
        persistence,
        g_base_override=drivers.g_base.value,
        margin_target_override=drivers.margin_target_company.value,
        phi_override=persistence.phi,
    )
    pd.testing.assert_frame_equal(base_frame, over_frame)
    assert base_audit["paths"] == over_audit["paths"]


def test_sector_tilt_shifts_near_term_growth() -> None:
    base, base_audit = build()
    tilted, tilt_audit = build(sector_tilt=0.01)
    assert tilt_audit["g_base_effective"] == pytest.approx(base_audit["g_base_effective"] + 0.01)
    assert tilted["growth_annual"].iloc[0] > base["growth_annual"].iloc[0]


def test_tau_tied_to_cap_with_bounds() -> None:
    assert fade_time_constant(200.0) == TAU_BOUNDS_QUARTERS[1]
    assert fade_time_constant(0.0) == TAU_BOUNDS_QUARTERS[0]
    cap_q = 32.0
    assert fade_time_constant(cap_q) == pytest.approx(cap_q / np.log(10.0))


def test_fcf_margin_band_tripwire() -> None:
    # Forecast FCF margins (~10-17% here) sit far outside an absurdly tight band.
    tight, audit_tight = build(make_drivers(fcf_band=(0.40, 0.41, 0.42)))
    assert audit_tight["fcf_margin_breach_quarters"]
    _, audit_wide = build(make_drivers(fcf_band=(-0.50, 0.15, 0.90)))
    assert audit_wide["fcf_margin_breach_quarters"] == []


def test_margin_target_blends_sector_anchor() -> None:
    prior = SectorPrior(phi=0.85, margin_anchor=0.10, roic_anchor=0.15, anchor_weight=0.25)
    _, audit = build(prior=prior)
    assert audit["margin_target_blend"] == pytest.approx(0.75 * 0.18 + 0.25 * 0.10, rel=1e-12)
