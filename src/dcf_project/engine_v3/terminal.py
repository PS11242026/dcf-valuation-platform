"""Terminal value for engine_v3: Grinold-Kroner growth anchor + value-driver formula.

Growth anchor (Grinold-Kroner): a single firm cannot perpetually out-grow nominal
output, and per-share claims are diluted by net issuance. So the terminal growth
anchor is

    g = long-run nominal GDP (expected inflation + real growth) − dilution haircut,

clamped to the same band the v1 engine uses ([floor, min(cap, WACC − 150bp)]) so v3
and v1 terminal assumptions stay directly comparable.

Terminal value (value-driver formula, McKinsey/Koller):

    TV = NOPAT_{T+1} · (1 − g/RONIC_T) / (WACC − g)

Growth must be paid for: sustaining g forever requires reinvesting g/RONIC_T of NOPAT
every year. At RONIC_T = WACC the firm earns its cost of capital on new projects and
growth adds nothing: as additionally g→0, TV → NOPAT/WACC. RONIC_T comes from the
persistence model (WACC + the capped surviving excess), never from a separate knob.
"""
from __future__ import annotations

from typing import Any

from dcf_project.config import Settings
from dcf_project.engine_v3.config import V3Config
from dcf_project.valuation.terminal import TerminalValueError

#: Same guard semantics as valuation.terminal.terminal_value (v1).
MIN_TERMINAL_SPREAD = 0.0025
WACC_GUARD_GAP = 0.015


def gk_terminal_growth(settings: Settings, cfg: V3Config, wacc: float) -> tuple[float, dict[str, Any]]:
    raw = settings.long_run_nominal_gdp - cfg.terminal_dilution_haircut
    ceiling = min(settings.terminal_growth_cap, wacc - WACC_GUARD_GAP)
    g = min(max(raw, settings.terminal_growth_floor), ceiling)
    decomposition = {
        "long_run_nominal_gdp": settings.long_run_nominal_gdp,
        "inflation_anchor": settings.inflation_anchor,
        "implied_real_growth": settings.long_run_nominal_gdp - settings.inflation_anchor,
        "dilution_haircut": cfg.terminal_dilution_haircut,
        "gk_raw": raw,
        "band_floor": settings.terminal_growth_floor,
        "band_ceiling": ceiling,
        "terminal_growth": g,
    }
    return float(g), decomposition


def value_driver_terminal(
    nopat_t1_annual: float,
    terminal_growth: float,
    ronic_terminal: float,
    wacc: float,
) -> tuple[float, dict[str, Any]]:
    """TV at the end of the explicit horizon from next-year NOPAT, g, RONIC_T, WACC."""
    spread = wacc - terminal_growth
    if spread <= MIN_TERMINAL_SPREAD:
        raise TerminalValueError(
            f"Invalid terminal spread: WACC ({wacc:.4f}) must exceed terminal growth "
            f"({terminal_growth:.4f}) by at least {MIN_TERMINAL_SPREAD:.2%}."
        )
    # RONIC_T >= WACC by construction (persistence floors the surviving excess at 0),
    # which also guarantees g/RONIC_T < 1 given the spread guard above.
    ronic = max(ronic_terminal, wacc)
    reinvestment_rate = terminal_growth / ronic
    fcf_t1 = nopat_t1_annual * (1.0 - reinvestment_rate)
    tv = fcf_t1 / spread
    detail = {
        "nopat_t1_annual": float(nopat_t1_annual),
        "terminal_growth": float(terminal_growth),
        "ronic_terminal": float(ronic),
        "terminal_reinvestment_rate": float(reinvestment_rate),
        "terminal_fcf_t1_annual": float(fcf_t1),
        "spread": float(spread),
        "terminal_value": float(tv),
    }
    return float(tv), detail
