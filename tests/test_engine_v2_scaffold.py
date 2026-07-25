"""P0 scaffold tests for engine_v2 (Gate 1: V1 integrity).

Locks the frozen v1 numeric outputs against future v2 work and verifies the additive
``engine_version`` stamp + the pipeline dispatch. v1 must stay byte-identical: the only
permitted change is additive metadata (``engine_version="v1"``).
"""
from __future__ import annotations

import pytest

from dcf_project.config import Settings
from dcf_project.contracts import DCFResult
from dcf_project.engine_v2 import BayesianV2Config, select_pipeline
from dcf_project.engine_v2.pipeline import BayesianDCFPipeline
from dcf_project.pipeline import DCFPipeline
from test_pipeline import FixturePipeline


# Golden v1 outputs from the deterministic FixturePipeline (captured at P0).
# Any drift here means v1 numerics changed -- Gate 1 fails.
GOLDEN = {
    "intrinsic_value_per_share": 25.0359997967,
    "enterprise_value": 2553.5999796666,
    "equity_value": 2503.5999796666,
    "terminal_value": 3011.6137189959,
    "wacc": 0.0900000000,
    "terminal_growth_rate": 0.0238505040,
    "sum_projected_fcf": 795.8377475040,
}


def _v1_result() -> DCFResult:
    return FixturePipeline(Settings(terminal_growth_source="internal")).run("TST", horizon_quarters=20)


def test_v1_numerics_byte_identical() -> None:
    r = _v1_result()
    assert r.intrinsic_value_per_share == pytest.approx(GOLDEN["intrinsic_value_per_share"], rel=1e-9)
    assert r.enterprise_value == pytest.approx(GOLDEN["enterprise_value"], rel=1e-9)
    assert r.equity_value == pytest.approx(GOLDEN["equity_value"], rel=1e-9)
    assert r.terminal_value == pytest.approx(GOLDEN["terminal_value"], rel=1e-9)
    assert r.wacc == pytest.approx(GOLDEN["wacc"], rel=1e-9)
    assert r.terminal_growth_rate == pytest.approx(GOLDEN["terminal_growth_rate"], rel=1e-9)
    assert sum(r.projected_fcf) == pytest.approx(GOLDEN["sum_projected_fcf"], rel=1e-9)
    assert len(r.projected_fcf) == 20


def test_v1_stamps_engine_version() -> None:
    assert _v1_result().engine_version == "v1"
    # default on the contract itself is also "v1"
    assert DCFResult.model_fields["engine_version"].default == "v1"


def test_select_pipeline_defaults_to_v1() -> None:
    assert isinstance(select_pipeline(Settings()), DCFPipeline)
    assert not isinstance(select_pipeline(Settings()), BayesianDCFPipeline)


def test_select_pipeline_v2_is_bayesian() -> None:
    p = select_pipeline(Settings(engine_version="v2"))
    assert isinstance(p, BayesianDCFPipeline)


def test_v2_config_from_settings() -> None:
    cfg = BayesianV2Config.from_settings(Settings())
    assert cfg.paths == 2000 and cfg.seed == 7 and cfg.sample_wacc is False
