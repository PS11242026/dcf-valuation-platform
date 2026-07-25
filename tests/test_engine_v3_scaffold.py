"""Scaffold tests for engine_v3 (G0: v1 integrity + dispatch + config).

v1 must stay byte-identical under every additive change Phase 4 made (the v3_*
settings block, the select_pipeline v3 branch, the scoring.py dispatch rewire, the
SEC-cache flag). The golden numbers are imported from the engine_v2 scaffold so there
is exactly ONE set of frozen v1 numerics in the test suite.
"""
from __future__ import annotations

import pytest

from dcf_project.config import Settings
from dcf_project.engine_v2 import select_pipeline
from dcf_project.engine_v2.pipeline import BayesianDCFPipeline
from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.pipeline import EconomicDCFPipeline
from dcf_project.pipeline import DCFPipeline
from test_engine_v2_scaffold import GOLDEN
from test_pipeline import FixturePipeline


def test_v1_numerics_unchanged_with_v3_block_and_cache_flag() -> None:
    """The v3_* Settings block + sec_cache flag must not perturb v1 numerics."""
    r = FixturePipeline(
        Settings(terminal_growth_source="internal", sec_cache_enabled=True)
    ).run("TST", horizon_quarters=20)
    assert r.intrinsic_value_per_share == pytest.approx(GOLDEN["intrinsic_value_per_share"], rel=1e-9)
    assert r.enterprise_value == pytest.approx(GOLDEN["enterprise_value"], rel=1e-9)
    assert r.terminal_value == pytest.approx(GOLDEN["terminal_value"], rel=1e-9)
    assert sum(r.projected_fcf) == pytest.approx(GOLDEN["sum_projected_fcf"], rel=1e-9)
    assert r.engine_version == "v1"


def test_select_pipeline_dispatch_matrix() -> None:
    # Default stays the exact frozen class (what portfolio.scoring now routes through).
    assert type(select_pipeline(Settings())) is DCFPipeline
    assert isinstance(select_pipeline(Settings(engine_version="v2")), BayesianDCFPipeline)
    assert isinstance(select_pipeline(Settings(engine_version="v3")), EconomicDCFPipeline)
    # v3 pipeline is still a DCFPipeline (inherits load_data -> PIT bundle injection works).
    assert isinstance(select_pipeline(Settings(engine_version="v3")), DCFPipeline)


def test_v3_config_from_settings_defaults() -> None:
    cfg = V3Config.from_settings(Settings())
    # Monte Carlo is diagnostic-only and OFF by default (failed calibration gate G6).
    assert cfg.mc_draws == 0
    # Explicit fade horizon is 10 years, decoupled from v1's 20-quarter setting.
    assert cfg.horizon_quarters == 40
    assert cfg.headline == "deterministic"
    assert cfg.phi_prior_strength == 12.0
    assert cfg.half_life_prior_strength == 8.0
    assert cfg.half_life_min_years == 0.5 and cfg.half_life_max_years == 8.0
    assert cfg.cap_max_years == 15.0
    assert cfg.moat_premium_cap == 0.02
    assert cfg.reinvestment_cap == 0.95
    assert cfg.terminal_dilution_haircut == 0.01
    assert cfg.sample_wacc is False
    assert cfg.sector_state_kappa == 0.0
    assert cfg.seed == 7


def test_v3_config_reads_settings_overrides() -> None:
    cfg = V3Config.from_settings(
        Settings(v3_mc_draws=500, v3_headline="mc_median", v3_sector_state_kappa=0.25, v3_horizon_quarters=20)
    )
    assert cfg.mc_draws == 500
    assert cfg.headline == "mc_median"
    assert cfg.sector_state_kappa == 0.25
    assert cfg.horizon_quarters == 20


def test_sector_tilt_bounded_and_off_by_default() -> None:
    p = EconomicDCFPipeline(Settings())  # kappa default 0
    p.sector_state_score = 2.0
    assert p._sector_tilt(V3Config.from_settings(p.settings)) == 0.0

    p = EconomicDCFPipeline(Settings(v3_sector_state_kappa=0.25))
    assert p._sector_tilt(V3Config.from_settings(p.settings)) == 0.0  # score None
    p.sector_state_score = 1.5
    assert p._sector_tilt(V3Config.from_settings(p.settings)) == pytest.approx(0.25 * 1.5 * 0.01)
    p.sector_state_score = 1000.0
    assert p._sector_tilt(V3Config.from_settings(p.settings)) == 0.02  # hard bound


def test_v3_horizon_config_driven_and_decoupled_from_v1() -> None:
    """v3's explicit horizon comes from its own config (default 40q) and can be
    overridden per call or via settings -- it is NOT tied to v1's fixed 20-quarter
    ``explicit_forecast_quarters`` (which stays 20, keeping v1 byte-identical)."""
    from test_engine_v3_assemble import V3FixturePipeline

    base = Settings(terminal_growth_source="internal")
    assert base.explicit_forecast_quarters == 20  # v1 setting untouched

    default = V3FixturePipeline(base).run("TST")
    assert default.explicit_forecast_quarters == 40 and len(default.projected_fcf) == 40

    via_settings = V3FixturePipeline(
        Settings(terminal_growth_source="internal", v3_horizon_quarters=24)
    ).run("TST")
    assert via_settings.explicit_forecast_quarters == 24 and len(via_settings.projected_fcf) == 24

    via_call = V3FixturePipeline(base).run("TST", horizon_quarters=12)
    assert via_call.explicit_forecast_quarters == 12 and len(via_call.projected_fcf) == 12
