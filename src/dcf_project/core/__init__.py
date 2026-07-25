"""Facade for the frozen DCF engine.

New product code (valuation/, portfolio/, research/) imports the engine through
``dcf_project.core`` so there is one stable public surface. The engine modules remain
physically and byte-identically in ``data_sources/``, ``models/``, ``preprocessing/``,
``valuation/``, ``analytics/``, ``reporting/`` + ``pipeline``/``config``/``contracts`` --
importing this facade neither moves nor modifies any frozen code. This is the only thing
``core`` is: a re-export layer, deliberately added (not a relocation) to keep the 18 tests
green and the canonical math unchanged.
"""
from __future__ import annotations

from dcf_project.config import Settings, get_settings
from dcf_project.contracts import (
    DCFResult,
    FinancialPeriod,
    GrowthForecast,
    MarketSnapshot,
    SourceMeta,
    WACCInputs,
)
from dcf_project.pipeline import DCFPipeline
from dcf_project.valuation.baseline import deterministic_baseline
from dcf_project.valuation.dcf import build_dcf_result, build_state_space_dcf_result

__all__ = [
    "Settings",
    "get_settings",
    "DCFPipeline",
    "deterministic_baseline",
    "build_dcf_result",
    "build_state_space_dcf_result",
    "DCFResult",
    "FinancialPeriod",
    "GrowthForecast",
    "MarketSnapshot",
    "SourceMeta",
    "WACCInputs",
]
