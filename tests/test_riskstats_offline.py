"""Offline tests for the book-agnostic risk-stats core + the dashboard/pitch slot separation.

Synthetic weekly series with KNOWN structure pin the math: a holding built as 1.5x the benchmark's
returns must regress to beta ~1.5 / corr ~1; an independent flat series to beta ~0. The persistence
tests pin the collision fix: the dashboard slot and the default (pitch) slot never overwrite each
other, and the merge-write preserves keys another tool owns (India's analyst block).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dcf_project.portfolio.riskstats import WEEKS, _weights, stats_from_closes
from dcf_project.reporting_kit.persist import (
    DASHBOARD_MODEL_FILE,
    MODEL_FILE,
    load_model,
    save_model,
)


def _series(rets, name, start="2023-07-01"):
    idx = pd.date_range(start, periods=len(rets) + 1, freq="W")
    return pd.Series(100.0 * np.cumprod([1.0, *(1 + np.asarray(rets))]), index=idx, name=name)


def _bench_rets(n=156, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0.002, 0.02, n)


def test_beta_recovers_known_loading():
    br = _bench_rets()
    closes = {"HI": _series(1.5 * br, "HI"), "LO": _series(np.full(len(br), 0.001), "LO")}
    out = stats_from_closes([{"ticker": "HI", "entry_price": 100.0},
                             {"ticker": "LO", "entry_price": 100.0}],
                            closes, _series(br, "B"))
    assert out["per_name"]["HI"]["beta"] == pytest.approx(1.5, abs=0.01)
    assert out["per_name"]["HI"]["corr_bench"] == pytest.approx(1.0, abs=0.001)
    assert out["per_name"]["LO"]["beta"] == pytest.approx(0.0, abs=0.01)
    # portfolio beta = weight-blend of name betas (weights ~drifted-equal here)
    w_hi = out["per_name"]["HI"]["weight"]
    assert out["portfolio"]["beta"] == pytest.approx(w_hi * 1.5, abs=0.02)
    assert out["portfolio"]["tracking_error"] > 0
    assert out["portfolio"]["diversification_ratio"] >= 1.0


def test_vol_annualisation_and_hhi():
    br = _bench_rets(120, seed=3)
    closes = {"A": _series(br, "A")}
    out = stats_from_closes([{"ticker": "A", "entry_price": 100.0}], closes, _series(br, "B"))
    expected_vol = float(np.std(pd.Series(br), ddof=1) * np.sqrt(WEEKS))
    assert out["per_name"]["A"]["vol"] == pytest.approx(expected_vol, abs=0.01)
    assert out["portfolio"]["hhi"] == pytest.approx(1.0)  # single name
    assert out["portfolio"]["effective_n_names"] == pytest.approx(1.0)


def test_weights_shares_vs_equal_weight():
    br = _bench_rets(60, seed=1)
    closes = {"X": _series(br, "X"), "Y": _series(br, "Y")}
    lastx = float(closes["X"].iloc[-1])
    shares = [{"ticker": "X", "shares": 300, "entry_price": 100.0},
              {"ticker": "Y", "shares": 100, "entry_price": 100.0}]
    w = _weights(shares, closes)
    assert w["X"] == pytest.approx(300 * lastx / (300 * lastx + 100 * float(closes["Y"].iloc[-1])))
    ew = _weights([{"ticker": "X", "entry_price": 100.0}, {"ticker": "Y", "entry_price": 100.0}], closes)
    assert ew["X"] == pytest.approx(0.5, abs=1e-9)  # identical paths -> equal drifted weights


def test_realized_alpha_zero_for_benchmark_clone():
    br = _bench_rets(156, seed=11)
    closes = {"CLONE": _series(br, "CLONE")}
    out = stats_from_closes([{"ticker": "CLONE", "entry_price": 100.0}], closes, _series(br, "B"),
                            rf=0.05)
    assert out["portfolio"]["beta"] == pytest.approx(1.0, abs=0.001)
    assert out["portfolio"]["realized_alpha"] == pytest.approx(0.0, abs=1e-6)
    assert out["portfolio"]["tracking_error"] == pytest.approx(0.0, abs=1e-9)


def test_model_slots_do_not_collide(tmp_path):
    save_model({"kind": "pitch"}, tmp_path)
    save_model({"kind": "dashboard"}, tmp_path, stem=DASHBOARD_MODEL_FILE)
    assert load_model(tmp_path) == {"kind": "pitch"}
    assert load_model(tmp_path, stem=DASHBOARD_MODEL_FILE) == {"kind": "dashboard"}
    assert (tmp_path / MODEL_FILE).exists() and (tmp_path / DASHBOARD_MODEL_FILE).exists()


def test_dashboard_writes_dashboard_slot_for_pitch_owned_book(tmp_path, monkeypatch):
    """A pitch-owned book's `manage update` must write <name>_dashboard.xlsx + the dashboard
    slot, leaving the default slot (the pitch package) untouched."""
    from dcf_project.portfolio import manager

    folder = tmp_path / "india_v3_aggressive"
    folder.mkdir()
    save_model({"kind": "pitch"}, folder)  # pre-existing pitch package model
    state = {"capital": 1_000_000.0, "currency": "INR", "inception": "2026-06-15", "books": {}}
    rows = [{"ticker": "IOC", "sector": "energy", "entry_price": 140.9, "price": 139.0,
             "weight": 1.0, "market_value": 1_000_000.0, "return": -0.013, "contribution": -0.013,
             "current_discount": -0.22, "current_intrinsic": 112.0, "conviction": 88.0}]
    eq = [{"date": "2026-06-15", "port_nav": 1.0, "bench": {"^NSEI": 1.0}},
          {"date": "2026-07-11", "port_nav": 0.99, "bench": {"^NSEI": 1.02}}]
    metrics = {"excess": {"^NSEI": -0.03}, "primary_benchmark": "^NSEI",
               "alpha_basis": "simple_excess_return", "sharpe": None, "max_drawdown": -0.01,
               "volatility": None, "beta": None, "turnover": None, "hhi": 1.0}
    out = manager._build_dashboard("india_v3_aggressive", state, "holdings", rows,
                                   {"energy": 1.0}, {"holdings": -0.013}, {"^NSEI": 0.02},
                                   metrics, eq, 26, "2026-07-11", folder)
    assert out.name == "india_v3_aggressive_dashboard.xlsx"
    assert load_model(folder) == {"kind": "pitch"}  # default slot untouched
    assert load_model(folder, stem=DASHBOARD_MODEL_FILE) is not None
