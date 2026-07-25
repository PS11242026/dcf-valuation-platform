"""engine_v3 -- the economic-drivers DCF.

A *parallel* valuation engine behind ``engine_version="v3"`` (default stays "v1").
Philosophy: economically grounded valuation science -- every assumption maps to a
named economic mechanism rather than a statistical abstraction:

- Competitive Advantage Period + ROIC persistence (competition erodes excess returns)
- fade-based forecast with mean-reverting revenue growth and NOPAT margins
- the sustainable-growth identity (reinvestment = g/RONIC: growth is paid for)
- Grinold-Kroner terminal growth anchor + the value-driver terminal formula
- Monte Carlo over the ECONOMIC DRIVERS (not latent states) for the distribution

Dispatch goes through ``dcf_project.engine_v2.select_pipeline`` (the single
engine_version dispatch point). v1 stays byte-identical; engine_v2 stays the archived
Bayesian research engine.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from dcf_project.engine_v3.config import V3Config

if TYPE_CHECKING:  # avoid importing pandas-heavy modules at package import time
    from dcf_project.engine_v3.pipeline import EconomicDCFPipeline  # noqa: F401

ENGINE_VERSION = "v3"

__all__ = ["ENGINE_VERSION", "V3Config"]
