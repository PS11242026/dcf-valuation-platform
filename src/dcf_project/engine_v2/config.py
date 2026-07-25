"""Configuration for the Bayesian state-space Monte Carlo engine (engine_v2).

A thin view over the additive ``v2_*`` knobs on the frozen :class:`Settings`. Keeping
this separate makes the v2 surface explicit without touching the v1 config.
"""
from __future__ import annotations

from dataclasses import dataclass

from dcf_project.config import Settings


@dataclass(frozen=True)
class BayesianV2Config:
    """Resolved engine_v2 knobs (sampling + MCMC budget)."""

    paths: int = 2000
    mcmc_draws: int = 1000
    mcmc_chains: int = 4
    mcmc_warmup: int = 1000
    mcmc_thin: int = 2
    sample_wacc: bool = False
    seed: int = 7

    @classmethod
    def from_settings(cls, settings: Settings) -> "BayesianV2Config":
        return cls(
            paths=int(getattr(settings, "v2_paths", 2000)),
            mcmc_draws=int(getattr(settings, "v2_mcmc_draws", 1000)),
            mcmc_chains=int(getattr(settings, "v2_mcmc_chains", 4)),
            mcmc_warmup=int(getattr(settings, "v2_mcmc_warmup", 1000)),
            mcmc_thin=int(getattr(settings, "v2_mcmc_thin", 2)),
            sample_wacc=bool(getattr(settings, "v2_sample_wacc", False)),
            seed=int(getattr(settings, "random_seed", 7)),
        )
