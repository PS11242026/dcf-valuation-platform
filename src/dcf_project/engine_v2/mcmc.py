"""Gibbs + FFBS sampler for the linear-Gaussian state-space DLM (engine_v2, U2).

Model (per state i, in mean-centered working space; observation matrix C = I):

    latent:   d_{t+1,i} = phi_i * d_{t,i} + w_{t,i},   w ~ N(0, q_i)
    observed: y_{t,i}   = d_{t,i}        + v_{t,i},     v ~ N(0, r_i)

Because v1 uses a diagonal transition, diagonal noise, and C = I, the k-dimensional filter
decouples into k independent scalar AR(1)+noise DLMs. The Gibbs sampler is therefore fully
conjugate and vectorized across states:

  1. d_{0:T} | phi, q, r, y   via FFBS (Carter-Kohn), k scalar filters in parallel
  2. phi_i   | d, q           Normal posterior (Normal prior centered on v1 persistence)
  3. q_i     | d, phi         Inverse-Gamma posterior
  4. r_i     | d, y           Inverse-Gamma posterior

No ad-hoc parameter fixing: Phi, Q, R all carry full posteriors. Seeded and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dcf_project.engine_v2.posterior import PosteriorDraws

_PHI_LO, _PHI_HI = 1e-4, 0.9990  # stationarity band (matches v1's stabilized transition)


@dataclass(frozen=True)
class GibbsPriors:
    """Conjugate priors (length-k vectors). Defaults are weakly informative."""

    phi_mean: np.ndarray
    phi_sd: np.ndarray
    q_a: np.ndarray
    q_b: np.ndarray
    r_a: np.ndarray
    r_b: np.ndarray

    @classmethod
    def weakly_informative(cls, k: int) -> "GibbsPriors":
        ones = np.ones(k)
        return cls(
            phi_mean=0.7 * ones,
            phi_sd=0.3 * ones,
            q_a=2.0 * ones,
            q_b=0.02 * ones,
            r_a=2.0 * ones,
            r_b=0.02 * ones,
        )

    @classmethod
    def from_v1(
        cls,
        phi_diag: np.ndarray,
        q_diag: np.ndarray,
        r_diag: np.ndarray,
        phi_sd: float = 0.15,
        noise_strength: float = 6.0,
    ) -> "GibbsPriors":
        """Center priors on the fitted v1 parameters (company-type persistence priors).

        ``phi_sd`` controls how tightly Phi is pulled toward v1; ``noise_strength`` is the
        Inverse-Gamma prior pseudo-count for Q/R (mode ~ the v1 variance estimate).
        """
        phi_diag = np.clip(np.asarray(phi_diag, dtype=float), _PHI_LO, _PHI_HI)
        q_diag = np.maximum(np.asarray(q_diag, dtype=float), 1e-8)
        r_diag = np.maximum(np.asarray(r_diag, dtype=float), 1e-8)
        a = noise_strength * np.ones_like(q_diag)
        # InvGamma(a, b) mode = b/(a+1); set b so mode == v1 variance estimate.
        return cls(
            phi_mean=phi_diag,
            phi_sd=phi_sd * np.ones_like(phi_diag),
            q_a=a,
            q_b=q_diag * (a + 1.0),
            r_a=a,
            r_b=r_diag * (a + 1.0),
        )


def _invgamma(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # If g ~ Gamma(shape=a, rate=b) then 1/g ~ InvGamma(a, b).
    g = rng.gamma(shape=a, scale=1.0 / b)
    return 1.0 / np.maximum(g, 1e-300)


def _ffbs(
    y: np.ndarray, phi: np.ndarray, q: np.ndarray, r: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Forward-filter backward-sample. y:(T,k); phi/q/r:(k,). Returns sampled d:(T,k)."""
    T, k = y.shape
    a = np.empty((T, k))   # predicted mean
    Rp = np.empty((T, k))  # predicted variance
    m = np.empty((T, k))   # filtered mean
    P = np.empty((T, k))   # filtered variance
    m_prev = np.zeros(k)
    P_prev = np.var(y, axis=0) + r  # diffuse-ish start
    for t in range(T):
        at = phi * m_prev
        Rpt = phi * phi * P_prev + q
        S = Rpt + r
        K = Rpt / S
        a[t] = at
        Rp[t] = Rpt
        m[t] = at + K * (y[t] - at)
        P[t] = (1.0 - K) * Rpt
        m_prev, P_prev = m[t], P[t]
    d = np.empty((T, k))
    d[T - 1] = m[T - 1] + np.sqrt(np.maximum(P[T - 1], 0.0)) * rng.standard_normal(k)
    for t in range(T - 2, -1, -1):
        J = P[t] * phi / Rp[t + 1]
        m_bar = m[t] + J * (d[t + 1] - a[t + 1])
        P_bar = P[t] - J * J * Rp[t + 1]
        d[t] = m_bar + np.sqrt(np.maximum(P_bar, 0.0)) * rng.standard_normal(k)
    return d


def _sample_phi(d: np.ndarray, q: np.ndarray, pr: GibbsPriors, rng: np.random.Generator) -> np.ndarray:
    x = d[:-1]
    y = d[1:]
    sxx = np.sum(x * x, axis=0)
    sxy = np.sum(x * y, axis=0)
    prior_prec = 1.0 / (pr.phi_sd ** 2)
    prec = prior_prec + sxx / q
    mean = (pr.phi_mean * prior_prec + sxy / q) / prec
    sd = 1.0 / np.sqrt(prec)
    phi = mean + sd * rng.standard_normal(d.shape[1])
    return np.clip(phi, _PHI_LO, _PHI_HI)


def _sample_q(d: np.ndarray, phi: np.ndarray, pr: GibbsPriors, rng: np.random.Generator) -> np.ndarray:
    w = d[1:] - phi * d[:-1]
    a_n = pr.q_a + 0.5 * (d.shape[0] - 1)
    b_n = pr.q_b + 0.5 * np.sum(w * w, axis=0)
    return _invgamma(a_n, b_n, rng)


def _sample_r(d: np.ndarray, y: np.ndarray, pr: GibbsPriors, rng: np.random.Generator) -> np.ndarray:
    v = y - d
    a_n = pr.r_a + 0.5 * y.shape[0]
    b_n = pr.r_b + 0.5 * np.sum(v * v, axis=0)
    return _invgamma(a_n, b_n, rng)


def _run_chain(
    y: np.ndarray, priors: GibbsPriors, draws: int, warmup: int, thin: int, seed: int, fix_r: bool
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = y.shape[1]
    phi = np.clip(priors.phi_mean.copy(), _PHI_LO, _PHI_HI)
    q = priors.q_b / (priors.q_a + 1.0)
    r = priors.r_b / (priors.r_a + 1.0)
    kept_phi, kept_q, kept_r, kept_last = [], [], [], []
    total = warmup + draws
    for it in range(total):
        d = _ffbs(y, phi, q, r, rng)
        phi = _sample_phi(d, q, priors, rng)
        q = _sample_q(d, phi, priors, rng)
        if not fix_r:
            r = _sample_r(d, y, priors, rng)
        if it >= warmup and ((it - warmup) % thin == 0):
            kept_phi.append(phi.copy())
            kept_q.append(q.copy())
            kept_r.append(r.copy())
            kept_last.append(d[-1].copy())
    return {
        "phi": np.asarray(kept_phi),
        "q": np.asarray(kept_q),
        "r": np.asarray(kept_r),
        "last_state": np.asarray(kept_last),
    }


def _split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split-Rhat per parameter. chains: (n_chains, n_draws, k) -> (k,)."""
    n_chains, n, k = chains.shape
    if n < 4 or n_chains < 1:
        return np.full(k, np.nan)
    half = n // 2
    splits = np.concatenate([chains[:, :half, :], chains[:, half:2 * half, :]], axis=0)
    m = splits.shape[0]
    nn = half
    means = splits.mean(axis=1)            # (m, k)
    variances = splits.var(axis=1, ddof=1)  # (m, k)
    B = nn * means.var(axis=0, ddof=1)     # between
    W = variances.mean(axis=0)             # within
    var_hat = (nn - 1) / nn * W + B / nn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(W > 0, var_hat / W, np.nan))


def _ess(chains: np.ndarray) -> np.ndarray:
    """Crude multi-chain ESS per parameter. chains: (n_chains, n_draws, k) -> (k,)."""
    n_chains, n, k = chains.shape
    if n < 4:
        return np.full(k, float(n_chains * n))
    out = np.empty(k)
    for j in range(k):
        x = chains[:, :, j] - chains[:, :, j].mean()
        # mean autocorrelation across chains
        acf = np.zeros(n)
        for c in range(n_chains):
            xc = chains[c, :, j] - chains[c, :, j].mean()
            var = np.sum(xc * xc)
            if var <= 0:
                continue
            full = np.correlate(xc, xc, mode="full")[n - 1:] / var
            acf += full / n_chains
        s = 1.0
        for t in range(1, n):
            if acf[t] <= 0:
                break
            s += 2.0 * acf[t]
        out[j] = n_chains * n / max(s, 1e-6)
    return out


def run_gibbs(
    y: np.ndarray,
    priors: GibbsPriors | None = None,
    *,
    state_columns: tuple[str, ...] | None = None,
    draws: int = 1000,
    chains: int = 4,
    warmup: int = 1000,
    thin: int = 2,
    seed: int = 7,
    fix_r: bool = False,
    meta: dict | None = None,
) -> PosteriorDraws:
    """Run the Gibbs+FFBS sampler. ``y`` is the (T,k) mean-centered observation matrix."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must be a (T, k) matrix of centered observations")
    T, k = y.shape
    if T < 8:
        raise ValueError("at least eight observations are required")
    priors = priors or GibbsPriors.weakly_informative(k)
    per_chain = []
    for c in range(chains):
        per_chain.append(_run_chain(y, priors, draws, warmup, thin, seed + 1000 * c, fix_r))
    phi_c = np.stack([pc["phi"] for pc in per_chain])  # (chains, draws_kept, k)
    q_c = np.stack([pc["q"] for pc in per_chain])
    r_c = np.stack([pc["r"] for pc in per_chain])
    last_c = np.stack([pc["last_state"] for pc in per_chain])
    diagnostics = {
        "rhat": {
            "phi": _split_rhat(phi_c).tolist(),
            "q": _split_rhat(q_c).tolist(),
            "r": _split_rhat(r_c).tolist(),
        },
        "ess": {
            "phi": _ess(phi_c).tolist(),
            "q": _ess(q_c).tolist(),
            "r": _ess(r_c).tolist(),
        },
        "n_chains": chains,
        "draws_per_chain": phi_c.shape[1],
        "warmup": warmup,
        "thin": thin,
        "seed": seed,
    }
    cols = state_columns or tuple(f"state_{i}" for i in range(k))
    return PosteriorDraws(
        phi=phi_c.reshape(-1, k),
        q=q_c.reshape(-1, k),
        r=r_c.reshape(-1, k),
        last_state=last_c.reshape(-1, k),
        state_columns=cols,
        diagnostics=diagnostics,
        meta=meta or {},
    )
