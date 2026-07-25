"""Posterior draw storage for engine_v2.

Holds MCMC draws of the (diagonal) transition Phi, process noise Q, observation noise R,
and the sampled terminal latent state (the initial condition for posterior-predictive
simulation). Draws live in the *working space* the sampler ran in (standardized, mean-centered
deviations); ``center``/``scale``/``mu`` in ``meta`` let the simulator map back to raw units.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PosteriorDraws:
    phi: np.ndarray  # (n_draws, k) diagonal transition persistence
    q: np.ndarray    # (n_draws, k) diagonal process-noise variances
    r: np.ndarray    # (n_draws, k) diagonal observation-noise variances
    last_state: np.ndarray  # (n_draws, k) sampled latent deviation at final time (working space)
    state_columns: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return int(self.phi.shape[0])

    @property
    def k(self) -> int:
        return int(self.phi.shape[1])

    def summary(self) -> dict:
        def stats(a: np.ndarray) -> dict:
            return {
                "mean": np.mean(a, axis=0).tolist(),
                "std": np.std(a, axis=0, ddof=1).tolist() if a.shape[0] > 1 else np.zeros(a.shape[1]).tolist(),
            }

        return {
            "state_columns": list(self.state_columns),
            "n_draws": self.n_draws,
            "phi": stats(self.phi),
            "q": stats(self.q),
            "r": stats(self.r),
            "diagnostics": self.diagnostics,
        }

    def sample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n >= self.n_draws:
            return np.arange(self.n_draws)
        return rng.choice(self.n_draws, size=n, replace=False)

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            phi=self.phi,
            q=self.q,
            r=self.r,
            last_state=self.last_state,
            state_columns=np.asarray(self.state_columns, dtype=object),
            diagnostics=np.asarray(json.dumps(self.diagnostics), dtype=object),
            meta=np.asarray(json.dumps(self.meta, default=float), dtype=object),
        )

    @classmethod
    def load(cls, path) -> "PosteriorDraws":
        with np.load(path, allow_pickle=True) as data:
            return cls(
                phi=data["phi"],
                q=data["q"],
                r=data["r"],
                last_state=data["last_state"],
                state_columns=tuple(str(c) for c in data["state_columns"].tolist()),
                diagnostics=json.loads(str(data["diagnostics"].item())),
                meta=json.loads(str(data["meta"].item())),
            )
