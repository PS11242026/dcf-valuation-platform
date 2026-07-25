"""Output-coherence guard (Research_V4 Modification 1).

Two layers: pure-function unit tests over ``assess_coherence`` (each of the four
checks fires on its true cause and on nothing else, including the SNPS-type
positive-but-tiny case that must stay healthy), and integration tests proving the
guard is wired into the engine_v3 result -- flagging a broken fixture ``incomplete``
while leaving every healthy intrinsic bit-identical to the guard-off run.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from dcf_project.engine_v3.coherence import assess_coherence

from test_engine_v3_assemble import GOLDEN_V3, SETTINGS, V3FixturePipeline


# --- A healthy baseline kwargs set (KO-like) that every check should pass ------
HEALTHY = dict(
    intrinsic=45.0,
    equity_value=2.0e11,
    enterprise_value=2.2e11,
    recon_shares=4.31e9,
    market_shares=4.30e9,
    revenue_ttm=4.9e10,
    pv_terminal_share_of_ev=0.66,
    market_price=82.0,
)


def test_healthy_is_ok() -> None:
    v = assess_coherence(**HEALTHY)
    assert v.status == "ok" and v.flags == [] and not v.is_incomplete


def test_positive_but_tiny_is_not_flagged() -> None:
    # SNPS-type SBC/low-GAAP conservatism: tiny positive intrinsic is a genuine
    # valuation, NOT a defect -- the guard must leave it healthy.
    v = assess_coherence(**{**HEALTHY, "intrinsic": 3.25, "equity_value": 6.2e8, "market_price": 456.0})
    assert v.status == "ok" and v.flags == []


def test_corrupt_shares_flagged() -> None:
    # MCD: reconstructed 713 vs ~713 million market shares.
    v = assess_coherence(**{**HEALTHY, "recon_shares": 713.5, "market_shares": 7.1e8})
    assert v.is_incomplete and "corrupt_shares" in v.flags


def test_negative_equity_flagged() -> None:
    # EL/MRVL: EV below net debt -> per-share value <= 0.
    v = assess_coherence(**{**HEALTHY, "intrinsic": -1.0, "equity_value": -3.6e8})
    assert v.is_incomplete and "negative_equity" in v.flags


def test_terminal_blowup_flagged() -> None:
    # GIS: terminal dominates EV (pvtv 0.97) AND EV/sales is absurd (~21x).
    v = assess_coherence(
        **{**HEALTHY, "enterprise_value": 3.9e11, "revenue_ttm": 1.8e10, "pv_terminal_share_of_ev": 0.97}
    )
    assert v.is_incomplete and "terminal_blowup" in v.flags


def test_high_pvtv_alone_is_not_a_blowup() -> None:
    # Terminal-weighted but sane EV/sales (software-like): must NOT fire.
    v = assess_coherence(
        **{**HEALTHY, "enterprise_value": 1.3e11, "revenue_ttm": 2.4e10, "pv_terminal_share_of_ev": 0.97}
    )
    assert "terminal_blowup" not in v.flags


def test_implausible_vs_price_backstop() -> None:
    v = assess_coherence(**{**HEALTHY, "intrinsic": 2.342e8, "market_price": 285.0})
    assert v.is_incomplete and "implausible_vs_price" in v.flags


def test_checks_recorded() -> None:
    v = assess_coherence(**HEALTHY)
    assert set(v.checks) == {"share_ratio", "ev_to_sales", "pv_terminal_share_of_ev", "iv_to_price"}


# --- Integration: wired into the engine_v3 result --------------------------------
def test_healthy_fixture_status_ok_and_field_set() -> None:
    r = V3FixturePipeline(SETTINGS).run("TST")
    assert r.valuation_coherence == "ok"
    assert r.diagnostics["valuation_coherence"]["status"] == "ok"
    # Guard never moves a healthy headline.
    assert r.intrinsic_value_per_share == pytest.approx(GOLDEN_V3["intrinsic_value_per_share"], rel=1e-9)


def test_guard_off_leaves_field_none_and_intrinsic_identical() -> None:
    on = V3FixturePipeline(SETTINGS).run("TST")
    off = V3FixturePipeline(replace(SETTINGS, v3_coherence_guard=False)).run("TST")
    assert off.valuation_coherence is None
    assert "valuation_coherence" not in off.diagnostics
    assert off.intrinsic_value_per_share == on.intrinsic_value_per_share


class _CorruptSharesFixture(V3FixturePipeline):
    def load_data(self, ticker: str):  # type: ignore[override]
        bundle = super().load_data(ticker)
        # Market provider reports 100M shares; reconstruction keeps the history's 100.
        bundle.market.shares_outstanding = 1.0e8
        return bundle


def test_corrupt_shares_fixture_is_incomplete() -> None:
    r = _CorruptSharesFixture(SETTINGS).run("TST")
    assert r.valuation_coherence == "incomplete"
    assert "corrupt_shares" in r.diagnostics["valuation_coherence"]["flags"]
    assert r.implied_upside is None
    assert any("incomplete" in w for w in r.warnings)
