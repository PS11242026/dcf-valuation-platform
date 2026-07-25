"""P1 tests for the Gibbs+FFBS MCMC parameter layer (Gate 2: synthetic recovery).

Asserts the sampler recovers known Phi/Q/R from synthetic AR(1)+noise data, that the
posterior contracts as the sample grows, that chains mix (R-hat), and that draws are
reproducible and round-trip through .npz storage.
"""
from __future__ import annotations

import numpy as np
import pytest

from dcf_project.engine_v2.mcmc import GibbsPriors, run_gibbs
from dcf_project.engine_v2.posterior import PosteriorDraws

PHI = np.array([0.85, 0.60, 0.92, 0.70])
Q = np.array([0.02, 0.04, 0.01, 0.03])
R = np.array([0.01, 0.02, 0.008, 0.015])


def _synth(T: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = len(PHI)
    d = np.zeros((T, k))
    for t in range(1, T):
        d[t] = PHI * d[t - 1] + np.sqrt(Q) * rng.standard_normal(k)
    return d + np.sqrt(R) * rng.standard_normal((T, k))


def test_recovers_known_parameters() -> None:
    post = run_gibbs(_synth(300), draws=500, chains=2, warmup=300, thin=2, seed=7)
    phi_err = np.abs(post.phi.mean(0) - PHI)
    q_rel = np.abs(post.q.mean(0) - Q) / Q
    r_rel = np.abs(post.r.mean(0) - R) / R
    assert phi_err.max() < 0.20, phi_err
    assert q_rel.max() < 0.6, q_rel
    assert r_rel.max() < 0.8, r_rel


def test_chains_mix_rhat() -> None:
    post = run_gibbs(_synth(300), draws=500, chains=2, warmup=300, thin=2, seed=7)
    for group in ("phi", "q", "r"):
        rhat = np.asarray(post.diagnostics["rhat"][group], dtype=float)
        assert np.nanmax(rhat) < 1.2, (group, rhat)
        ess = np.asarray(post.diagnostics["ess"][group], dtype=float)
        assert np.nanmin(ess) > 10.0, (group, ess)


def test_posterior_contracts_with_n() -> None:
    small = run_gibbs(_synth(80, seed=2), draws=400, chains=2, warmup=300, thin=2, seed=7)
    large = run_gibbs(_synth(600, seed=2), draws=400, chains=2, warmup=300, thin=2, seed=7)
    assert np.all(large.phi.std(0) < small.phi.std(0))


def test_reproducible_seed() -> None:
    a = run_gibbs(_synth(150), draws=150, chains=2, warmup=100, thin=2, seed=11)
    b = run_gibbs(_synth(150), draws=150, chains=2, warmup=100, thin=2, seed=11)
    assert np.array_equal(a.phi, b.phi)
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.last_state, b.last_state)


def test_save_load_roundtrip(tmp_path) -> None:
    post = run_gibbs(
        _synth(120), draws=100, chains=2, warmup=80, thin=2, seed=3,
        state_columns=("revenue_growth", "operating_margin", "reinvestment_rate", "roic"),
        meta={"working_space": "test"},
    )
    p = tmp_path / "post.npz"
    post.save(p)
    loaded = PosteriorDraws.load(p)
    assert np.array_equal(loaded.phi, post.phi)
    assert loaded.state_columns == post.state_columns
    assert loaded.meta["working_space"] == "test"
    assert loaded.diagnostics["n_chains"] == 2


def test_priors_from_v1_centering() -> None:
    pr = GibbsPriors.from_v1(np.array([0.9, 0.95, 0.8, 0.99]), np.array([0.01, 0.02, 0.03, 0.005]),
                             np.array([0.01, 0.01, 0.01, 0.01]))
    # InvGamma(a,b) mode = b/(a+1) should equal the supplied v1 variance
    assert pr.q_b[0] / (pr.q_a[0] + 1.0) == pytest.approx(0.01, rel=1e-9)
    assert np.all(pr.phi_mean <= 0.999)
