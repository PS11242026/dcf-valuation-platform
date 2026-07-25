from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanResult:
    predicted_states: np.ndarray
    predicted_covariances: np.ndarray
    filtered_states: np.ndarray
    filtered_covariances: np.ndarray
    smoothed_states: np.ndarray
    smoothed_covariances: np.ndarray
    filtered_state: np.ndarray
    filtered_cov: np.ndarray
    log_likelihood: float
    innovations: np.ndarray


def kalman_filter(
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    q: np.ndarray,
    h: np.ndarray,
    z: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
    initial_cov: np.ndarray | None = None,
) -> KalmanResult:
    y = np.asarray(y, dtype=float)
    n_obs, n_state = y.shape
    z = np.eye(n_state) if z is None else np.asarray(z, dtype=float)
    state = initial_state.copy() if initial_state is not None else np.nanmean(y, axis=0)
    cov = initial_cov.copy() if initial_cov is not None else np.eye(n_state) * 0.25
    innovations = []
    log_likelihood = 0.0
    predicted_states = np.zeros((n_obs, n_state), dtype=float)
    filtered_states = np.zeros((n_obs, n_state), dtype=float)
    predicted_covariances = np.zeros((n_obs, n_state, n_state), dtype=float)
    filtered_covariances = np.zeros((n_obs, n_state, n_state), dtype=float)

    for t in range(n_obs):
        pred_state = mu + phi @ (state - mu)
        pred_cov = phi @ cov @ phi.T + q
        pred_cov = _stabilize(pred_cov)
        predicted_states[t] = pred_state
        predicted_covariances[t] = pred_cov
        obs = y[t]
        mask = np.isfinite(obs)
        if not mask.any():
            state, cov = pred_state, pred_cov
            filtered_states[t] = state
            filtered_covariances[t] = cov
            continue

        z_t = z[mask]
        obs_t = obs[mask]
        innovation = obs_t - z_t @ pred_state
        s = z_t @ pred_cov @ z_t.T + h[np.ix_(mask, mask)]
        s = _stabilize(s)
        k = pred_cov @ z_t.T @ np.linalg.pinv(s)
        state = pred_state + k @ innovation
        eye = np.eye(n_state)
        cov = (eye - k @ z_t) @ pred_cov @ (eye - k @ z_t).T + k @ h[np.ix_(mask, mask)] @ k.T
        cov = _stabilize(cov)
        filtered_states[t] = state
        filtered_covariances[t] = cov
        innovations.append(float(np.mean(innovation)))

        sign, logdet = np.linalg.slogdet(s)
        if sign > 0:
            quad = innovation.T @ np.linalg.pinv(s) @ innovation
            log_likelihood += float(-0.5 * (len(obs_t) * np.log(2 * np.pi) + logdet + quad))

    smoothed_states, smoothed_covariances = rts_smoother(
        predicted_states,
        predicted_covariances,
        filtered_states,
        filtered_covariances,
        mu,
        phi,
    )
    return KalmanResult(
        predicted_states=predicted_states,
        predicted_covariances=predicted_covariances,
        filtered_states=filtered_states,
        filtered_covariances=filtered_covariances,
        smoothed_states=smoothed_states,
        smoothed_covariances=smoothed_covariances,
        filtered_state=state,
        filtered_cov=cov,
        log_likelihood=log_likelihood,
        innovations=np.asarray(innovations, dtype=float),
    )


def rts_smoother(
    predicted_states: np.ndarray,
    predicted_covariances: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_obs = filtered_states.shape[0]
    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for t in range(n_obs - 2, -1, -1):
        pred_next = mu + phi @ (filtered_states[t] - mu)
        pred_cov_next = predicted_covariances[t + 1]
        gain = filtered_covariances[t] @ phi.T @ np.linalg.pinv(pred_cov_next)
        smoothed_states[t] = filtered_states[t] + gain @ (smoothed_states[t + 1] - pred_next)
        smoothed_covariances[t] = _stabilize(
            filtered_covariances[t]
            + gain @ (smoothed_covariances[t + 1] - pred_cov_next) @ gain.T
        )
    return smoothed_states, smoothed_covariances


def forecast_states(
    last_state: np.ndarray,
    last_cov: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    q: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    covs = []
    state = last_state.copy()
    cov = last_cov.copy()
    for _ in range(horizon):
        state = mu + phi @ (state - mu)
        cov = _stabilize(phi @ cov @ phi.T + q)
        states.append(state.copy())
        covs.append(cov.copy())
    return np.asarray(states), np.asarray(covs)


def _stabilize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    eigvals = np.linalg.eigvalsh(matrix)
    jitter = max(1e-9, -float(np.min(eigvals)) + 1e-9)
    return matrix + np.eye(matrix.shape[0]) * jitter
