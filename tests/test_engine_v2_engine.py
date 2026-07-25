"""P2 tests for the posterior-coherent engine (Gates 3, 3.5, 4).

Gate 3   posterior coherence: no parameter/state/process uncertainty => zero valuation variance.
Gate 3.5 calibration: the predictive percentiles cover independent realizations at ~90%.
Gate 4   terminal stability: equilibrium state = mu, half-lives valid, growth banded, bounded.
Plus an end-to-end assemble test (v2 DCFResult shape + mean ~ V1 baseline).
"""
from __future__ import annotations

import numpy as np
import pytest

from dcf_project.config import Settings
from dcf_project.contracts import WACCInputs
from dcf_project.engine_v2.posterior import PosteriorDraws
from dcf_project.engine_v2.posterior_predictive import simulate
from dcf_project.engine_v2.terminal import equilibrium_growth, equilibrium_state
from dcf_project.preprocessing.reconstruction import STATE_COLUMNS
from test_pipeline import FixturePipeline


class _StubModel:
    mu = np.array([0.03, 0.20, 0.10, 0.05])
    standardization_scale = np.array([0.01, 0.02, 0.02, 0.01])
    phi = np.diag([0.80, 0.90, 0.70, 0.95])


def _wacc(w: float = 0.09) -> WACCInputs:
    return WACCInputs(
        risk_free_rate=0.04, beta=1.0, equity_risk_premium=0.05, cost_of_equity=0.09,
        pre_tax_cost_of_debt=0.045, tax_rate=0.21, market_value_equity=1000.0,
        market_value_debt=100.0, equity_weight=0.9, debt_weight=0.1,
        after_tax_cost_of_debt=0.0355, wacc=w,
    )


def _posterior(n: int, q_var: float, x0_sd: float, seed: int = 0) -> PosteriorDraws:
    rng = np.random.default_rng(seed)
    k = 4
    phi = np.tile(np.array([0.80, 0.90, 0.70, 0.95]), (n, 1))
    q = np.full((n, k), q_var)
    last = x0_sd * rng.standard_normal((n, k))
    return PosteriorDraws(
        phi=phi, q=q, r=np.zeros((n, k)), last_state=last,
        state_columns=tuple(STATE_COLUMNS), diagnostics={}, meta={},
    )


def _simulate(post: PosteriorDraws, n_paths: int, seed: int = 1):
    return simulate(
        model=_StubModel(), posterior=post, base_revenue=100.0, base_fcf=5.0,
        base_invested_capital=125.0, tax_rate=0.21, company_type="asset_light_quality",
        calibration_mode="roic_dcf", ticker="TST",
        weights={"accounting": 0.30, "economic": 0.20, "owner": 0.50},
        wacc_inputs=_wacc(), net_debt=0.0, shares=100.0, settings=Settings(),
        n_paths=n_paths, seed=seed, horizon=20,
    )


def test_gate3_no_uncertainty_zero_variance() -> None:
    # q=0 and identical initial states => every path is the mu-constant path => zero variance.
    post = _posterior(n=200, q_var=0.0, x0_sd=0.0)
    sim = _simulate(post, n_paths=200)
    assert np.std(sim.iv) < 1e-9
    assert np.all(np.isfinite(sim.iv))


def test_gate3_variance_grows_with_process_noise() -> None:
    lo = _simulate(_posterior(200, q_var=0.001, x0_sd=0.0), n_paths=400)
    hi = _simulate(_posterior(200, q_var=0.05, x0_sd=0.0), n_paths=400)
    assert np.std(hi.iv) > np.std(lo.iv)


def test_gate3_5_calibration_coverage() -> None:
    post = _posterior(n=400, q_var=0.02, x0_sd=0.3, seed=5)
    ref = _simulate(post, n_paths=1500, seed=10).iv
    real = _simulate(post, n_paths=1500, seed=999).iv
    lo, hi = np.percentile(ref, 5), np.percentile(ref, 95)
    coverage = float(np.mean((real >= lo) & (real <= hi)))
    assert 0.82 <= coverage <= 0.98, coverage


def test_gate4_equilibrium_state_is_mu() -> None:
    eq = equilibrium_state(_StubModel.mu, _StubModel.phi)
    assert eq["x_star"] == _StubModel.mu.tolist()
    # half-life of phi_ii=0.8 is ln0.5/ln0.8
    assert eq["half_life_quarters"][0] == pytest.approx(np.log(0.5) / np.log(0.8), rel=1e-9)
    assert eq["spectral_radius"] < 1.0


def test_gate4_equilibrium_growth_banded() -> None:
    s = Settings()
    # high reinvestment*roic saturates to the band cap (<= wacc-0.015)
    g_high = equilibrium_growth(0.5, 0.05, 0.09, s)
    assert s.terminal_growth_floor <= g_high <= min(s.terminal_growth_cap, 0.09 - 0.015) + 1e-12
    # zero reinvestment floors at terminal_growth_floor
    assert equilibrium_growth(0.0, 0.05, 0.09, s) == pytest.approx(s.terminal_growth_floor)
    # non-finite inputs never raise
    assert np.isfinite(equilibrium_growth(np.nan, np.nan, 0.09, s))


def test_assemble_end_to_end(tmp_path) -> None:
    # Build a real v1 baseline + fitted model via a BayesianDCFPipeline on fixture data.
    from dcf_project.engine_v2.pipeline import BayesianDCFPipeline

    class FixtureV2(BayesianDCFPipeline):
        load_data = FixturePipeline.load_data

    pipe = FixtureV2(Settings(terminal_growth_source="internal", forecast_mode="state_space",
                              v2_paths=300, v2_mcmc_draws=150, v2_mcmc_chains=2, v2_mcmc_warmup=120))
    pipe.posterior_dir = tmp_path
    r = pipe.run("TST", horizon_quarters=20)
    assert r.engine_version == "v2"
    mc = r.monte_carlo_distribution
    assert mc["engine"] == "v2" and mc["n_paths"] == 300
    for key in ("expected", "median", "mode", "p5", "p25", "p75", "p95", "std"):
        assert key in mc["iv"]
    assert set(mc["signals"]) >= {"mc_expected_discount", "mc_probability_undervalued",
                                  "mc_margin_of_safety_probability", "mc_distribution_confidence"}
    assert len(mc["terminal_state"]["x_star"]) == 4  # x* = mu (4 latent states)
    # v2 median is economically close to the v1 baseline (terminal methodology differs).
    base = r.diagnostics["v1_baseline_intrinsic"]
    assert abs(r.intrinsic_value_per_share - base) / base < 0.30
    assert (tmp_path / "TST.npz").exists()
