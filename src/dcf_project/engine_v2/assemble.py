"""run_bayesian_dcf -- assemble a v2 DCFResult from the v1 baseline + posterior MC.

Strategy: reuse the frozen ``build_state_space_dcf_result`` for the v1 baseline (so ensemble
weights, net debt, shares, wacc and the v1 mean-path intrinsic are byte-identical to v1), then
fit the Bayesian parameter posterior and run the posterior-coherent Monte Carlo on top. The
v2 ``intrinsic_value_per_share`` is the posterior median; the full distribution lands in
``monte_carlo_distribution``. The v1 baseline intrinsic is retained in diagnostics for the
V1-vs-V2 comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dcf_project.config import Settings
from dcf_project.contracts import DCFResult, WACCInputs
from dcf_project.engine_v2.mcmc import GibbsPriors, run_gibbs
from dcf_project.engine_v2.posterior import PosteriorDraws
from dcf_project.engine_v2.posterior_predictive import simulate
from dcf_project.engine_v2.result import build_distribution_payload
from dcf_project.preprocessing.reconstruction import STATE_COLUMNS, state_matrix


def _working_observations(history: pd.DataFrame, model) -> np.ndarray:
    """Standardized, mean-centered observations y = (x - mu)/scale for the DLM."""
    x = np.asarray(state_matrix(history), dtype=float)
    mu = np.asarray(model.mu, dtype=float)
    scale = np.asarray(model.standardization_scale, dtype=float)
    y = (x - mu) / scale
    frame = pd.DataFrame(y).replace([np.inf, -np.inf], np.nan).interpolate().ffill().bfill()
    return frame.to_numpy(dtype=float)


def fit_posterior(history: pd.DataFrame, model, settings: Settings) -> PosteriorDraws:
    y = _working_observations(history, model)
    scale = np.asarray(model.standardization_scale, dtype=float)
    phi_diag = np.clip(np.diag(np.asarray(model.phi, dtype=float)), 1e-4, 0.999)
    q_std = np.diag(np.asarray(model.q, dtype=float)) / (scale ** 2)
    r_std = np.diag(np.asarray(model.r, dtype=float)) / (scale ** 2)
    priors = GibbsPriors.from_v1(phi_diag, q_std, r_std)
    return run_gibbs(
        y,
        priors,
        state_columns=tuple(STATE_COLUMNS),
        draws=settings.v2_mcmc_draws,
        chains=settings.v2_mcmc_chains,
        warmup=settings.v2_mcmc_warmup,
        thin=settings.v2_mcmc_thin,
        seed=settings.random_seed,
        meta={"working_space": "standardized_centered", "mu": list(map(float, model.mu)),
              "scale": list(map(float, scale))},
    )


def run_bayesian_dcf(
    *,
    baseline: DCFResult,
    model,
    history: pd.DataFrame,
    base_fcf: float,
    wacc_inputs: WACCInputs,
    market,
    settings: Settings,
    posterior_path: Path | None = None,
) -> DCFResult:
    posterior = fit_posterior(history, model, settings)
    if posterior_path is not None:
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        posterior.save(posterior_path)

    net_debt = float(baseline.net_debt)
    shares = float(baseline.diluted_shares_outstanding)
    weights = dict(baseline.ensemble_weights) or {"accounting": 1 / 3, "economic": 1 / 3, "owner": 1 / 3}
    last = history.iloc[-1]

    terminal_override = (
        float(baseline.terminal_growth_rate)
        if getattr(settings, "v2_terminal_mode", "equilibrium") == "v1"
        else None
    )
    sim = simulate(
        model=model,
        posterior=posterior,
        base_revenue=float(last["revenue"]),
        base_fcf=float(base_fcf),
        base_invested_capital=float(last["invested_capital"]) if "invested_capital" in history else None,
        tax_rate=float(last["tax_rate"]) if "tax_rate" in history else 0.21,
        company_type=settings.company_type,
        calibration_mode=settings.valuation_mode,
        ticker=baseline.ticker.upper(),
        weights=weights,
        wacc_inputs=wacc_inputs,
        net_debt=net_debt,
        shares=shares,
        settings=settings,
        n_paths=int(settings.v2_paths),
        seed=int(settings.random_seed),
        horizon=int(settings.explicit_forecast_quarters),
        terminal_growth_override=terminal_override,
    )

    price = baseline.market_price
    payload = build_distribution_payload(sim, posterior, model, price)
    iv = sim.iv[np.isfinite(sim.iv)]
    median = float(np.median(iv))

    diagnostics = dict(baseline.diagnostics)
    diagnostics["v1_baseline_intrinsic"] = float(baseline.intrinsic_value_per_share)
    diagnostics["engine"] = "v2_bayesian_state_space_mc"

    return baseline.model_copy(
        update={
            "engine_version": "v2",
            "intrinsic_value_per_share": median,
            "valuation_p10": float(np.percentile(iv, 10)),
            "valuation_p50": median,
            "valuation_p90": float(np.percentile(iv, 90)),
            "implied_upside": (median / price - 1.0) if price else None,
            "confidence_interval": {
                "low_intrinsic_value": float(np.percentile(iv, 5)),
                "median_intrinsic_value": median,
                "high_intrinsic_value": float(np.percentile(iv, 95)),
            },
            "monte_carlo_distribution": payload,
            "diagnostics": diagnostics,
        }
    )
