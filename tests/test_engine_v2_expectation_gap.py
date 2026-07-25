"""Gate 5 (P3) unit tests for engine_v2.expectation_gap.

Pins the four Expectation Gaps' arithmetic, polarity (higher == cheaper), and None-handling.
Uses a light stub carrying only the attributes the module reads, so the test is independent of
the full valuation pipeline -- the gaps are a pure, additive transform of a DCFResult.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dcf_project.engine_v2 import expectation_gap as eg


def _result(**over):
    """A minimal DCFResult-shaped stub with sensible defaults."""
    diag = {
        "economic_fcf_margin": 0.30,
        "reverse_dcf_growth": {
            "market_implied_annual_fcf_growth": 0.08,
            "market_implied_terminal_growth": 0.025,
            "market_implied_margin": 0.22,
            "market_implied_maintenance_capex_delta": 50.0,
        },
    }
    diag.update(over.pop("diagnostics", {}))
    base = dict(
        diagnostics=diag,
        fcf_growth_rate=0.12,
        economic=None,
        market_price=100.0,
        intrinsic_value_per_share=130.0,
        projected_fcf=[200.0, 210.0, 220.0],
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_reverse_dcf_growth_is_promoted() -> None:
    # The function is re-exported (promoted) into engine_v2.expectation_gap.
    assert callable(eg._reverse_dcf_growth)
    assert "_reverse_dcf_growth" in eg.__all__


def test_all_four_gaps_present() -> None:
    gaps = eg.expectation_gaps(_result())
    assert set(gaps) == set(eg.GAP_KEYS)


def test_gap_arithmetic_and_polarity() -> None:
    gaps = eg.expectation_gaps(_result())
    # growth: model 0.12 - market 0.08 = +0.04  (model expects more growth -> cheap)
    assert gaps["growth_gap"] == pytest.approx(0.04)
    # margin: model 0.30 - market 0.22 = +0.08
    assert gaps["margin_gap"] == pytest.approx(0.08)
    # reinvestment: market maint delta 50 / |first_fcf 200| = +0.25
    assert gaps["reinvestment_gap"] == pytest.approx(0.25)
    # value: intrinsic 130 / price 100 - 1 = +0.30  (== the discount)
    assert gaps["value_gap"] == pytest.approx(0.30)


def test_value_gap_equals_discount() -> None:
    r = _result(intrinsic_value_per_share=80.0, market_price=100.0)
    assert eg.expectation_gaps(r)["value_gap"] == pytest.approx(-0.20)


def test_negative_gaps_when_market_optimistic() -> None:
    # Market prices in MORE than the model -> gaps go negative (expensive).
    r = _result(
        fcf_growth_rate=0.05,
        diagnostics={"economic_fcf_margin": 0.15,
                     "reverse_dcf_growth": {
                         "market_implied_annual_fcf_growth": 0.20,
                         "market_implied_margin": 0.35,
                         "market_implied_maintenance_capex_delta": -40.0,
                     }},
    )
    gaps = eg.expectation_gaps(r)
    assert gaps["growth_gap"] == pytest.approx(-0.15)
    assert gaps["margin_gap"] == pytest.approx(-0.20)
    assert gaps["reinvestment_gap"] == pytest.approx(-0.20)  # -40 / 200


def test_missing_reverse_dcf_yields_none() -> None:
    r = _result(diagnostics={"reverse_dcf_growth": {}})
    gaps = eg.expectation_gaps(r)
    assert gaps["growth_gap"] is None
    assert gaps["margin_gap"] is None
    assert gaps["reinvestment_gap"] is None
    # value_gap only needs intrinsic + price, still computable
    assert gaps["value_gap"] == pytest.approx(0.30)


def test_no_diagnostics_block_is_safe() -> None:
    r = SimpleNamespace(diagnostics=None, fcf_growth_rate=0.1,
                        market_price=None, intrinsic_value_per_share=None, projected_fcf=[])
    gaps = eg.expectation_gaps(r)
    assert all(v is None for v in gaps.values())


def test_nan_and_inf_inputs_rejected() -> None:
    r = _result(fcf_growth_rate=float("nan"),
                projected_fcf=[float("inf")],
                market_price=0.0)
    gaps = eg.expectation_gaps(r)
    assert gaps["growth_gap"] is None        # nan model growth
    assert gaps["reinvestment_gap"] is None   # inf first_fcf
    assert gaps["value_gap"] is None          # zero price


def test_zero_first_fcf_safe() -> None:
    r = _result(projected_fcf=[0.0, 1.0])
    assert eg.expectation_gaps(r)["reinvestment_gap"] is None


def test_candidate_signals_prefixed() -> None:
    sig = eg.candidate_signals(_result())
    assert set(sig) == {f"eg_{k}" for k in eg.GAP_KEYS}
    assert sig["eg_growth_gap"] == pytest.approx(0.04)


# --- compute-from-fields path (lean state-space / PIT result: no precomputed reverse-DCF) ---

def _lean_result(**over):
    """A lean DCFResult-shaped stub WITHOUT a precomputed reverse_dcf_growth block --
    forces market_implied_expectations to solve the reverse DCF itself."""
    diag = {"expectation_gap_inputs": {
        "latest_revenue": 1000.0, "latest_margin": 0.30, "latest_maintenance_capex": 20.0}}
    diag.update(over.pop("diagnostics", {}))
    base = dict(
        diagnostics=diag,
        fcf_growth_rate=0.12,
        market_price=100.0,
        diluted_shares_outstanding=10.0,
        net_debt=0.0,
        wacc=0.09,
        terminal_growth_rate=0.025,
        intrinsic_value_per_share=130.0,
        projected_fcf=[50.0] + [None] * 19,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_computes_reverse_dcf_when_no_precomputed_block() -> None:
    mkt = eg.market_implied_expectations(_lean_result())
    g = mkt["market_implied_annual_fcf_growth"]
    # market cap 1000 + net debt 0 vs first quarterly FCF 50 -> a solvable implied growth
    assert g is not None and -0.30 <= g <= 0.40
    assert mkt["market_implied_terminal_growth"] is not None
    assert mkt["market_implied_maintenance_capex_delta"] is not None


def test_lean_path_yields_all_four_gaps() -> None:
    gaps = eg.expectation_gaps(_lean_result())
    assert all(gaps[k] is not None for k in eg.GAP_KEYS), gaps
    assert gaps["value_gap"] == pytest.approx(0.30)  # 130/100 - 1


def test_lean_path_missing_inputs_degrade_gracefully() -> None:
    # No expectation_gap_inputs and no revenue -> growth/reinvestment/value still solve,
    # margin_gap falls to None (no model margin available).
    r = _lean_result()
    r.diagnostics = {}  # strip the surfaced inputs entirely
    gaps = eg.expectation_gaps(r)
    assert gaps["growth_gap"] is not None
    assert gaps["reinvestment_gap"] is not None
    assert gaps["value_gap"] is not None
    assert gaps["margin_gap"] is None


def test_unsolvable_reverse_dcf_is_none() -> None:
    # shares <= 0 trips the _reverse_dcf_growth guard -> market-implied all None.
    r = _lean_result(diluted_shares_outstanding=0.0)
    mkt = eg.market_implied_expectations(r)
    assert all(v is None for v in mkt.values())
    gaps = eg.expectation_gaps(r)
    assert gaps["growth_gap"] is None and gaps["value_gap"] is not None
