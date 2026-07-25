"""Equilibrium (endogenous) terminal value for engine_v2 (U3).

v1 picks terminal growth from an exogenous policy blend. v2 derives it from the steady
state of the mean-reverting transition: the fixed point of ``x_{t+1}=mu+Phi(x_t-mu)`` is
``x*=mu`` (unique because |eig(Phi)|<1). Terminal growth emerges from the equilibrium ROIC
and reinvestment via the same identity the engine already uses
(:func:`dcf._roic_reinvestment_terminal_growth`), then is clipped to the *identical* v1
terminal band so only the source of the number changes, not the guardrails.
"""
from __future__ import annotations

import numpy as np

from dcf_project.config import Settings


def equilibrium_growth(reinvestment: float, roic: float, annual_wacc: float, settings: Settings) -> float:
    """Annual steady-state growth g* = annual_roic * reinvestment, v1-banded.

    Mirrors ``_roic_reinvestment_terminal_growth`` (roic is quarterly when <0.75) and the
    terminal band from ``_fcf_growth_projection_profile``.
    """
    if not np.isfinite(reinvestment) or not np.isfinite(roic):
        g = settings.default_terminal_growth
    else:
        annual_roic = roic * 4.0 if roic < 0.75 else roic
        g = float(np.clip(annual_roic * reinvestment, 0.0, 0.04))
    cap = max(settings.terminal_growth_floor, min(settings.terminal_growth_cap, annual_wacc - 0.015))
    return float(np.clip(g, settings.terminal_growth_floor, cap))


def equilibrium_state(mu: np.ndarray, phi: np.ndarray) -> dict:
    """Steady-state diagnostics: x*=mu, transition eigenvalues, per-state half-life."""
    mu = np.asarray(mu, dtype=float)
    phi = np.asarray(phi, dtype=float)
    diag = np.clip(np.abs(np.diag(phi)), 1e-6, 0.999999)
    half_life = np.log(0.5) / np.log(diag)
    eig = np.linalg.eigvals(phi).real
    return {
        "x_star": mu.tolist(),
        "phi_eigenvalues": eig.tolist(),
        "spectral_radius": float(np.max(np.abs(eig))) if eig.size else 0.0,
        "half_life_quarters": half_life.tolist(),
    }
