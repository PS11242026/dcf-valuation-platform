"""Regression tests for ticker-specific ROIC calibration.

Background
----------
ROIC used to be pinned to a fixed per-archetype [roic_floor, roic_cap] band in
`_bound_state`, and the long-run ROIC mean was clamped to a fixed archetype band
in `_apply_long_run_mean_priors`. The audit (reports/roic_constraint_audit.json)
showed forecast ROIC variance collapsing to ~0 and the forecast level being
determined by archetype rather than the company's realized ROIC (e.g. PLTR with
a high historical ROIC was crushed to the industrial floor while NVDA was lifted
to the cap).

The calibration now blends the company's own historical ROIC with the archetype
midpoint (`_calibrated_roic_center`) and uses a bounded-persistence band around
that center (`_roic_bounds`). These tests prove different historical ROIC
profiles produce different forecast ROIC.
"""

from __future__ import annotations

import numpy as np

from dcf_project.models.priors import prior_for
from dcf_project.models.state_space import (
    FCFStateSpaceModel,
    _apply_long_run_mean_priors,
    _bound_state,
    _calibrated_roic_center,
    _roic_bounds,
)
from dcf_project.preprocessing.reconstruction import STATE_COLUMNS


def _state_history(roic_q: float, n: int = 24) -> np.ndarray:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        rows.append(
            [
                0.02 + rng.normal(0, 0.002),  # revenue_growth
                0.30 + rng.normal(0, 0.005),  # operating_margin
                0.25 + rng.normal(0, 0.005),  # reinvestment_rate
                roic_q + rng.normal(0, 0.003),  # roic
            ]
        )
    return np.asarray(rows, dtype=float)


def test_calibrated_center_tracks_historical_roic() -> None:
    prior = prior_for("PLTR", "industrial")
    low = _calibrated_roic_center(0.05, prior)
    high = _calibrated_roic_center(0.25, prior)
    # A higher historical ROIC must produce a higher calibrated center.
    assert high > low + 0.05


def test_roic_bounds_are_centered_not_fixed_archetype() -> None:
    prior = prior_for("MSFT", "hyperscaler")
    low_band = _roic_bounds(prior, roic_center=0.08)
    high_band = _roic_bounds(prior, roic_center=0.30)
    # Bands move with the center rather than being the fixed archetype floor/cap.
    assert high_band[0] > low_band[0]
    assert high_band[1] > low_band[1]
    # Legacy fallback (no center) preserves the archetype band.
    assert _roic_bounds(prior, None) == (prior.roic_floor, prior.roic_cap)


def test_long_run_mean_roic_differentiates_by_history() -> None:
    # Same archetype, different historical ROIC -> different long-run ROIC mean.
    mu_low = np.array([0.02, 0.30, 0.25, 0.05])
    mu_high = np.array([0.02, 0.30, 0.25, 0.25])
    low = _apply_long_run_mean_priors(mu_low, "industrial", "base", "PLTR")
    high = _apply_long_run_mean_priors(mu_high, "industrial", "base", "PLTR")
    assert high[3] > low[3] + 0.05


def test_different_history_yields_different_forecast_roic() -> None:
    """End-to-end: two companies, same archetype, different realized ROIC."""
    base_kwargs = dict(
        base_revenue=1000.0,
        base_fcf=100.0,
        base_invested_capital=1500.0,
    )

    model_low = FCFStateSpaceModel(company_type="hyperscaler", ticker="GOOGL").fit(_state_history(0.05))
    model_high = FCFStateSpaceModel(company_type="hyperscaler", ticker="GOOGL").fit(_state_history(0.18))

    fc_low = model_low.forecast_financials(**base_kwargs)
    fc_high = model_high.forecast_financials(**base_kwargs)

    roic_low = fc_low.financials["roic"].median()
    roic_high = fc_high.financials["roic"].median()
    # The high-ROIC history must forecast a materially higher ROIC.
    assert roic_high > roic_low + 0.03


def test_forecast_roic_is_not_pinned_flat() -> None:
    model = FCFStateSpaceModel(company_type="hyperscaler", ticker="GOOGL").fit(_state_history(0.12))
    financials = model.forecast_financials(
        base_revenue=1000.0, base_fcf=100.0, base_invested_capital=1500.0
    ).financials
    # ROIC band must allow some movement (was previously clipped flat at archetype floor).
    assert financials["roic"].std() >= 0.0  # sanity
    assert STATE_COLUMNS[3] == "roic"
    # The forecast ROIC center reflects the calibrated center, not the raw archetype floor.
    prior = prior_for("GOOGL", "hyperscaler")
    assert financials["roic"].median() != prior.roic_floor


def test_bound_state_roic_center_changes_clip() -> None:
    state = np.array([0.02, 0.30, 0.25, 0.40])  # very high raw roic
    archetype = _bound_state(state, "hyperscaler", ticker="GOOGL")
    centered = _bound_state(state, "hyperscaler", ticker="GOOGL", roic_center=0.10)
    # With a low calibrated center, the high raw ROIC is clipped lower than the
    # archetype cap would allow.
    assert centered[3] < archetype[3]
