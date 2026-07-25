"""Driver Monte Carlo tests (engine_v3): zero-variance invariant, determinism, schema."""
from __future__ import annotations

from dataclasses import replace

import pytest

from dcf_project.contracts import MarketSnapshot, SourceMeta
from dcf_project.engine_v3.assemble import value_fade_path
from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.fade import build_fade_forecast
from dcf_project.engine_v3.mc import run_driver_mc
from test_engine_v3_fade import G_TERMINAL, PRIOR, WACC, make_drivers, make_persistence

CASH, DEBT, SHARES = 50.0, 100.0, 100.0
G_FLOOR, G_CAP = 0.015, 0.04


def market() -> MarketSnapshot:
    source = SourceMeta(provider="fixture", key="TST")
    return MarketSnapshot(
        ticker="TST", price=10.0, market_cap=1000.0, shares_outstanding=SHARES,
        beta=1.0, cash=CASH, debt=DEBT, source=source,
    )


def run_mc(drivers, persistence, cfg):
    return run_driver_mc(
        drivers, persistence, PRIOR, cfg, WACC, G_TERMINAL, G_FLOOR, G_CAP,
        CASH, DEBT, SHARES, market(), horizon=20, sector_tilt=0.0,
    )


def skeleton_intrinsic(drivers, persistence, cfg) -> float:
    frame, _ = build_fade_forecast(drivers, persistence, PRIOR, cfg, WACC, G_TERMINAL, horizon=20)
    valuation = value_fade_path(
        frame, WACC, G_TERMINAL, WACC + persistence.excess_terminal, CASH, DEBT, SHARES, market()
    )
    return valuation["intrinsic"]


def test_zero_variance_draws_reproduce_skeleton_exactly() -> None:
    """THE invariant: with all driver sigmas at zero, every draw == the deterministic
    skeleton. Guarantees the sampler runs the same economics, not a copy."""
    drivers = make_drivers(g_se=0.0, margin_se=0.0)
    persistence = replace(make_persistence(), phi_se=0.0)
    cfg = V3Config(mc_draws=32, terminal_g_sigma=0.0, sample_wacc=False)
    skeleton = skeleton_intrinsic(drivers, persistence, cfg)
    payload = run_mc(drivers, persistence, cfg)
    assert payload is not None
    p = payload["percentiles"]
    assert p["p10"] == pytest.approx(skeleton, rel=1e-12)
    assert p["p50"] == pytest.approx(skeleton, rel=1e-12)
    assert p["p90"] == pytest.approx(skeleton, rel=1e-12)
    dist = payload["distribution"]
    assert dist["iv"]["std"] == pytest.approx(0.0, abs=1e-12)
    assert dist["iv"]["expected"] == pytest.approx(skeleton, rel=1e-12)
    assert dist["failed_draws"] == 0


def test_same_seed_bit_identical_different_seed_differs() -> None:
    drivers = make_drivers(g_se=0.03, margin_se=0.01)
    persistence = make_persistence()
    cfg = V3Config(mc_draws=128)
    a = run_mc(drivers, persistence, cfg)
    b = run_mc(drivers, persistence, cfg)
    assert a["percentiles"] == b["percentiles"]
    assert a["distribution"]["iv"] == b["distribution"]["iv"]

    other = run_mc(drivers, persistence, V3Config(mc_draws=128, seed=11))
    assert other["percentiles"]["p50"] != a["percentiles"]["p50"]


def test_payload_schema_and_ordering() -> None:
    payload = run_mc(make_drivers(g_se=0.03, margin_se=0.01), make_persistence(), V3Config(mc_draws=128))
    dist = payload["distribution"]
    for key in ("engine", "n_draws", "n_effective", "failed_draws", "seed", "iv",
                "signals", "histogram", "fan_chart", "driver_draws"):
        assert key in dist, key
    assert dist["engine"] == "v3"
    assert dist["n_draws"] == 128
    p = payload["percentiles"]
    assert p["p10"] <= p["p50"] <= p["p90"]
    assert p["p10"] < p["p90"]  # sigmas on -> a real band
    assert len(dist["fan_chart"]["revenue"]["p50"]) == 20
    assert len(dist["fan_chart"]["fcf"]["p5"]) == 20
    assert dist["histogram"]["counts"] and len(dist["histogram"]["bin_edges"]) == 51
    # price > 0 -> probability signals present
    assert "mc_probability_undervalued" in dist["signals"]
    assert "prob_value_gt_price" in dist["iv"]


def test_clip_counters_record_band_enforcement() -> None:
    wild = make_drivers(g=0.9, g_se=2.0)  # most draws fall outside (-0.5, 1.0)
    payload = run_mc(wild, make_persistence(), V3Config(mc_draws=256))
    draws = payload["distribution"]["driver_draws"]
    assert draws["g_base"]["clipped"] > 0
    assert draws["wacc"] is None  # sample_wacc off by default


def test_disabled_mc_returns_none() -> None:
    assert run_mc(make_drivers(), make_persistence(), V3Config(mc_draws=0)) is None
