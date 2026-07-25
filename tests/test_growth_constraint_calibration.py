"""Regression tests for the reported-FCF-growth clamp calibration defect.

Background
----------
`_fcf_growth_projection_profile` blends the explicit FCF path CAGR with the
median annual growth, then used to clamp that value to
``[min_mature_growth, max_mature_growth]`` (default [-0.05, 0.15]) before
*reporting* it as the headline ``fcf_growth_rate``. Every name whose true
blended growth exceeded 0.15 (GOOGL 0.177, NVDA 0.200, TSLA 0.167) was therefore
flattened to an identical 0.15, erasing cross-sectional differentiation in the
reported metric.

The band is only meant to guardrail the terminal-growth *blending input*. The
fix reports the true blended growth and clamps only the copy fed into the
terminal-growth logic, so intrinsic value is unchanged while the headline metric
once again differentiates companies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcf_project.config import Settings
from dcf_project.valuation.dcf import _fcf_growth_projection_profile


def _geometric_path(annual_growth: float, base: float = 100.0, quarters: int = 20) -> list[float]:
    q = (1.0 + annual_growth) ** 0.25 - 1.0
    return [base * (1.0 + q) ** i for i in range(quarters)]


def _forecast_financials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reinvestment_rate": [0.25] * 20,
            "roic": [0.10] * 20,
        }
    )


def _profile(annual_growth: float, settings: Settings) -> dict:
    path = _geometric_path(annual_growth)
    return _fcf_growth_projection_profile(
        path,
        _forecast_financials(),
        base_fcf=path[0],
        annual_wacc=0.09,
        policy_terminal_growth=0.025,
        settings=settings,
    )


def test_reported_growth_is_not_flattened_to_cap() -> None:
    settings = Settings()
    profile = _profile(0.30, settings)
    # The reported headline growth reflects the true ~30% path, not the 0.15 cap.
    assert profile["explicit_fcf_growth_rate"] > 0.15
    assert profile["explicit_fcf_growth_rate"] == np.float64(np.clip(profile["explicit_fcf_growth_rate"], 0.0, 1.0))
    # The terminal-blend copy retains the guardrail clamp.
    assert profile["terminal_blend_growth_rate"] == settings.max_mature_growth


def test_distinct_high_growth_names_remain_distinct() -> None:
    settings = Settings()
    low = _profile(0.18, settings)
    high = _profile(0.30, settings)
    # Previously both clamped to exactly 0.15; now they differentiate.
    assert high["explicit_fcf_growth_rate"] > low["explicit_fcf_growth_rate"] + 0.05
    assert low["explicit_fcf_growth_rate"] > 0.15


def test_unclamping_reported_growth_is_value_neutral_for_terminal() -> None:
    settings = Settings()
    low = _profile(0.18, settings)
    high = _profile(0.30, settings)
    # Both above-cap paths clamp to the same terminal-blend input...
    assert low["terminal_blend_growth_rate"] == high["terminal_blend_growth_rate"]
    # ...so the terminal growth rate (the value-relevant output) is identical
    # regardless of the now-differentiated reported growth.
    assert low["terminal_growth_rate"] == high["terminal_growth_rate"]


def test_below_cap_growth_is_reported_unchanged() -> None:
    settings = Settings()
    profile = _profile(0.08, settings)
    # Below-cap growth was never clamped; reported == terminal-blend here.
    assert abs(profile["explicit_fcf_growth_rate"] - 0.08) < 0.01
    assert profile["terminal_blend_growth_rate"] == profile["explicit_fcf_growth_rate"]
