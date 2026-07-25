"""Posterior-coherent Monte Carlo simulator (engine_v2, U1).

For each path: draw (Phi, Q) from the parameter posterior + an initial latent state from
the state posterior, propagate the state dynamics with process noise ``w~N(0,Q)``, map the
raw state path through the frozen :func:`financials_from_states`, blend the three FCF paths
with the *fixed* v1 ensemble weights, and discount via the frozen :func:`_value_from_path`
with a per-path equilibrium terminal growth. This replaces v1's ad-hoc perturbation of the
mean FCF path -- the distribution now comes from the actual state-space model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dcf_project.config import Settings
from dcf_project.contracts import WACCInputs
from dcf_project.engine_v2.posterior import PosteriorDraws
from dcf_project.engine_v2.terminal import equilibrium_growth
from dcf_project.models.state_space import financials_from_states
from dcf_project.valuation.dcf import _value_from_path

_PATHS = ("accounting", "economic", "owner")
_COLS = {"accounting": "accounting_fcf", "economic": "economic_fcf", "owner": "owner_earnings"}


@dataclass
class SimResult:
    iv: np.ndarray            # (n_paths,) per-share intrinsic value
    equilibrium_g: np.ndarray  # (n_paths,) per-path annual terminal growth
    revenue: np.ndarray       # (n_paths, H)
    ebit: np.ndarray          # (n_paths, H)
    fcf: np.ndarray           # (n_paths, H) ensemble-blended
    states: np.ndarray        # (n_paths, H, k) raw state paths
    n_paths: int
    seed: int


def simulate(
    *,
    model,
    posterior: PosteriorDraws,
    base_revenue: float,
    base_fcf: float,
    base_invested_capital: float | None,
    tax_rate: float,
    company_type: str,
    calibration_mode: str,
    ticker: str | None,
    weights: dict[str, float],
    wacc_inputs: WACCInputs,
    net_debt: float,
    shares: float,
    settings: Settings,
    n_paths: int,
    seed: int,
    horizon: int = 20,
    terminal_growth_override: float | None = None,
) -> SimResult:
    rng = np.random.default_rng(seed)
    mu = np.asarray(model.mu, dtype=float)
    scale = np.asarray(model.standardization_scale, dtype=float)
    k = mu.shape[0]
    annual_wacc = float(wacc_inputs.wacc)
    quarterly_wacc = annual_wacc / 4.0
    roic_center = float(mu[3]) if k >= 4 else None

    # Draw parameter+state posterior indices (with replacement if n_paths > n_draws).
    n_draws = posterior.n_draws
    idx = rng.integers(0, n_draws, size=n_paths) if n_paths > n_draws else posterior.sample_indices(n_paths, rng)
    phi_draws = posterior.phi[idx]        # (n_paths, k) std-space persistence
    q_draws = posterior.q[idx]            # (n_paths, k) std-space process var
    x0 = posterior.last_state[idx]        # (n_paths, k) std-space initial deviation

    iv = np.empty(n_paths)
    g_arr = np.empty(n_paths)
    rev = np.empty((n_paths, horizon))
    ebit = np.empty((n_paths, horizon))
    fcf = np.empty((n_paths, horizon))
    states_out = np.empty((n_paths, horizon, k))

    for s in range(n_paths):
        phi_s = phi_draws[s]
        sd_s = np.sqrt(np.maximum(q_draws[s], 0.0))
        d = x0[s].copy()
        raw_path = np.empty((horizon, k))
        for t in range(horizon):
            d = phi_s * d + sd_s * rng.standard_normal(k)
            raw_path[t] = mu + d * scale
        fin = financials_from_states(
            states=raw_path,
            base_revenue=base_revenue,
            base_fcf=base_fcf,
            base_invested_capital=base_invested_capital,
            tax_rate=tax_rate,
            company_type=company_type,
            calibration_mode=calibration_mode,
            ticker=ticker,
            roic_center=roic_center,
        )
        path_values = {name: fin[_COLS[name]].to_numpy(float) for name in _PATHS}
        blended = sum(weights[name] * path_values[name] for name in weights)
        # Terminal growth: equilibrium (U3) from the path's terminal state, or a fixed v1
        # override (P2.5 diagnostic isolating the distribution effect from the terminal change).
        if terminal_growth_override is not None:
            g_s = float(terminal_growth_override)
        else:
            reinv_term = float(fin["reinvestment_rate"].to_numpy(float)[-1])
            roic_term = float(fin["roic"].to_numpy(float)[-1])
            g_s = equilibrium_growth(reinv_term, roic_term, annual_wacc, settings)
        iv[s] = _value_from_path(list(blended), quarterly_wacc, g_s / 4.0, net_debt, shares)
        g_arr[s] = g_s
        rev[s] = fin["revenue"].to_numpy(float)
        ebit[s] = fin["ebit"].to_numpy(float)
        fcf[s] = blended
        states_out[s] = raw_path

    return SimResult(
        iv=iv, equilibrium_g=g_arr, revenue=rev, ebit=ebit, fcf=fcf,
        states=states_out, n_paths=n_paths, seed=seed,
    )
