"""Offline tests for the book-agnostic manager math (US equal-weight AND India fixed-shares).

The +6.5% vs +7.3% class of bug (first-close basis vs booked-entry basis) is what these pin down:
every surface must compute the SAME return from state.json's booked entries. The India fixtures
mirror reports/managed_portfolios/india_v3_aggressive/state.json semantics (shares, cash, scalar
^NSEI benchmark, INR); the US fixtures mirror conf_top10 (equal-weight, SPY/QQQ dict). No network.
"""
from __future__ import annotations

import pytest

from dcf_project.portfolio.manager import (
    _bench_returns,
    _benchmarks_of,
    _book_return,
    _build_dashboard,
    _holding_rows,
)

IN_BOOK = [  # shares-based, entry marks -> deployed 999,430.15 + 569.85 cash = 1,000,000 capital
    {"ticker": "IOC", "symbol": "IOC.NS", "sector": "energy", "entry_price": 140.94, "shares": 1445,
     "discount_to_intrinsic": -0.2248},
    {"ticker": "BPCL", "symbol": "BPCL.NS", "sector": "energy", "entry_price": 302.35, "shares": 673,
     "discount_to_intrinsic": -0.2272},
]
IN_CASH = 1_000_000.0 - (140.94 * 1445 + 302.35 * 673)
US_BOOK = [
    {"ticker": "KR", "sector": "consumer_staples", "entry_price": 62.38, "discount": 4.12, "conf": 0.84},
    {"ticker": "WHR", "sector": "consumer_discretionary", "entry_price": 38.74, "discount": 4.22, "conf": 0.505},
]


def test_benchmarks_of_both_shapes():
    us = _benchmarks_of({"benchmark_entry": {"SPY": 757.26, "QQQ": 741.86}})
    assert us == {"SPY": 757.26, "QQQ": 741.86}
    ind = _benchmarks_of({"benchmark": "^NSEI", "benchmark_entry": 23622.9})
    assert ind == {"^NSEI": 23622.9}
    assert _benchmarks_of({}) == {}


def test_equal_weight_book_return_is_mean_of_name_returns():
    cur = {"KR": 62.38 * 1.10, "WHR": 38.74 * 0.90}
    r = _book_return(US_BOOK, cur)
    assert r == pytest.approx(0.0, abs=1e-12)  # +10% and -10% average to zero


def test_shares_book_return_is_value_weighted_with_cash():
    cap = 1_000_000.0
    cur = {"IOC": 140.94 * 1.10, "BPCL": 302.35 * 1.10}  # both +10%
    r = _book_return(IN_BOOK, cur, cash=IN_CASH, capital=cap)
    deployed = 140.94 * 1445 + 302.35 * 673
    expected = (deployed * 1.10 + IN_CASH) / cap - 1.0  # cash drags the +10% slightly
    assert r == pytest.approx(expected, rel=1e-12)
    # unpriced name is held at entry, not dropped (NAV stays defined, no fake move)
    r2 = _book_return(IN_BOOK, {"IOC": 140.94 * 1.10}, cash=IN_CASH, capital=cap)
    expected2 = (140.94 * 1445 * 1.10 + 302.35 * 673 + IN_CASH) / cap - 1.0
    assert r2 == pytest.approx(expected2, rel=1e-12)


def test_holding_rows_contributions_sum_to_book_return_shares():
    cap = 1_000_000.0
    cur = {"IOC": 155.0, "BPCL": 290.0}
    rows = _holding_rows(IN_BOOK, cur, None, cap, cash=IN_CASH)
    book = _book_return(IN_BOOK, cur, cash=IN_CASH, capital=cap)
    assert sum(r["contribution"] for r in rows) == pytest.approx(book, rel=1e-9)
    by = {r["ticker"]: r for r in rows}
    assert by["IOC"]["entry_price"] == 140.94 and by["IOC"]["price"] == 155.0
    assert by["IOC"]["market_value"] == pytest.approx(1445 * 155.0)
    assert 0.99 < sum(r["weight"] for r in rows) <= 1.0 + 1e-9


def test_holding_rows_contributions_sum_to_book_return_equal_weight():
    cur = {"KR": 68.0, "WHR": 36.0}
    rows = _holding_rows(US_BOOK, cur, None, None)
    book = _book_return(US_BOOK, cur)
    assert sum(r["contribution"] for r in rows) == pytest.approx(book, rel=1e-9)


def test_bench_returns_from_entry_marks():
    out = _bench_returns({"^NSEI": 23622.9}, {"^NSEI": 23622.9 * 1.02})
    assert out["^NSEI"] == pytest.approx(0.02, rel=1e-9)
    assert _bench_returns({"SPY": 757.0}, {})["SPY"] is None


def test_inr_dashboard_builds_offline(tmp_path):
    """India-style (INR, single ^NSEI benchmark, manual book) renders the workbook end-to-end."""
    folder = tmp_path / "india_v3_aggressive"
    folder.mkdir(parents=True)
    state = {"capital": 1_000_000.0, "cash": IN_CASH, "currency": "INR",
             "inception": "2026-06-15", "books": {}}
    cur = {"IOC": 150.0, "BPCL": 310.0}
    rows = _holding_rows(IN_BOOK, cur, None, state["capital"], cash=IN_CASH)
    port = _book_return(IN_BOOK, cur, cash=IN_CASH, capital=state["capital"])
    bench = {"^NSEI": 0.015}
    eq = [{"date": "2026-06-15", "port_nav": 1.0, "bench": {"^NSEI": 1.0}},
          {"date": "2026-07-11", "port_nav": 1.0 + port, "bench": {"^NSEI": 1.015}}]
    metrics = {"excess": {"^NSEI": port - 0.015}, "primary_benchmark": "^NSEI",
               "alpha_basis": "simple_excess_return", "sharpe": None, "max_drawdown": None,
               "volatility": None, "beta": None, "turnover": None, "hhi": 1.0}
    out = _build_dashboard("india_v3_aggressive", state, "holdings", rows, {"energy": 1.0},
                           {"holdings": port}, bench, metrics, eq, 26, "2026-07-11", folder)
    assert out.exists()
    from openpyxl import load_workbook
    wb = load_workbook(out)
    assert "Executive Summary" in wb.sheetnames and "Current Holdings" in wb.sheetnames
