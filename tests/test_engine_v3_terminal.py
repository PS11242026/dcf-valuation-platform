"""Terminal tests (engine_v3): Grinold-Kroner anchor + value-driver limit identities."""
from __future__ import annotations

import pytest

from dcf_project.config import Settings
from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.terminal import gk_terminal_growth, value_driver_terminal
from dcf_project.valuation.terminal import TerminalValueError

SETTINGS = Settings()  # long_run_nominal_gdp=0.04, floor=0.015, cap=0.04
CFG = V3Config()  # dilution haircut 0.01


def test_gk_anchor_interior() -> None:
    g, detail = gk_terminal_growth(SETTINGS, CFG, wacc=0.09)
    assert g == pytest.approx(0.03)  # 4% GDP - 1% dilution, inside the band
    assert detail["gk_raw"] == pytest.approx(0.03)
    assert detail["implied_real_growth"] == pytest.approx(0.015)
    assert detail["band_ceiling"] == pytest.approx(min(0.04, 0.09 - 0.015))


def test_gk_anchor_wacc_ceiling_binds() -> None:
    g, detail = gk_terminal_growth(SETTINGS, CFG, wacc=0.032)
    assert g == pytest.approx(0.032 - 0.015)  # ceiling = wacc - 150bp
    assert detail["band_ceiling"] == pytest.approx(0.017)


def test_gk_anchor_floor_binds() -> None:
    heavy_dilution = V3Config(terminal_dilution_haircut=0.035)
    g, _ = gk_terminal_growth(SETTINGS, heavy_dilution, wacc=0.09)
    assert g == pytest.approx(SETTINGS.terminal_growth_floor)


def test_value_driver_algebra_hand_computed() -> None:
    tv, detail = value_driver_terminal(100.0, terminal_growth=0.03, ronic_terminal=0.12, wacc=0.09)
    assert detail["terminal_reinvestment_rate"] == pytest.approx(0.25)  # 0.03/0.12
    assert detail["terminal_fcf_t1_annual"] == pytest.approx(75.0)
    assert tv == pytest.approx(75.0 / 0.06)  # 1250


def test_limit_ronic_equals_wacc_growth_adds_nothing() -> None:
    # With RONIC_T == WACC, TV == NOPAT/WACC for ANY g: growth at the cost of capital
    # creates no value. This is the core identity of the value-driver formula.
    for g in (0.0, 0.02, 0.035):
        tv, _ = value_driver_terminal(100.0, terminal_growth=g, ronic_terminal=0.09, wacc=0.09)
        assert tv == pytest.approx(100.0 / 0.09, rel=1e-12)


def test_limit_zero_growth() -> None:
    tv, detail = value_driver_terminal(100.0, terminal_growth=0.0, ronic_terminal=0.15, wacc=0.09)
    assert detail["terminal_reinvestment_rate"] == 0.0
    assert tv == pytest.approx(100.0 / 0.09, rel=1e-12)


def test_ronic_floors_at_wacc() -> None:
    # Persistence never emits RONIC_T < WACC, but the formula guards anyway: a
    # below-WACC RONIC would make perpetual growth value-DESTROYING, which the
    # persistence economics (excess floors at 0) deliberately excludes.
    _, detail = value_driver_terminal(100.0, terminal_growth=0.02, ronic_terminal=0.05, wacc=0.09)
    assert detail["ronic_terminal"] == 0.09


def test_spread_guard() -> None:
    with pytest.raises(TerminalValueError):
        value_driver_terminal(100.0, terminal_growth=0.0575, ronic_terminal=0.09, wacc=0.06)
    tv, _ = value_driver_terminal(100.0, terminal_growth=0.055, ronic_terminal=0.09, wacc=0.06)
    assert tv > 0
