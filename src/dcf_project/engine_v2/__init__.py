"""engine_v2 -- Bayesian state-space Monte Carlo DCF.

A *parallel* valuation engine built behind the ``engine_version`` flag (default "v1").
The frozen canonical engine (v1) is preserved as the scientific control; v2 must pass the
validation gates before it is ever made default. See the approved plan for the gate sequence.

Public surface:
- :func:`select_pipeline` -- return the v1 or v2 pipeline for a given Settings.
- :class:`BayesianV2Config` -- resolved v2 sampling/MCMC knobs.
- ``run_bayesian_dcf`` -- the v2 assembler (implemented in P2).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from dcf_project.engine_v2.config import BayesianV2Config

if TYPE_CHECKING:  # avoid importing heavy/frozen modules at package import time
    from dcf_project.config import Settings
    from dcf_project.pipeline import DCFPipeline

ENGINE_VERSION = "v2"

__all__ = ["BayesianV2Config", "ENGINE_VERSION", "select_pipeline"]


def select_pipeline(settings: "Settings | None" = None) -> "DCFPipeline":
    """Return the pipeline matching ``settings.engine_version`` (default v1).

    This is the single dispatch point: callers that want to honor the engine flag
    construct their pipeline through here instead of instantiating ``DCFPipeline``
    directly. v1 remains the default everywhere until v2 is validated.
    """
    from dcf_project.config import get_settings
    from dcf_project.pipeline import DCFPipeline

    settings = settings or get_settings()
    engine_version = getattr(settings, "engine_version", "v1")
    if engine_version == "v2":
        from dcf_project.engine_v2.pipeline import BayesianDCFPipeline

        return BayesianDCFPipeline(settings)
    if engine_version == "v3":
        from dcf_project.engine_v3.pipeline import EconomicDCFPipeline

        return EconomicDCFPipeline(settings)
    return DCFPipeline(settings)
