"""P4B / Sector-State Engine v1 unit tests (engine_v2.sector_state, sector_map).

Pins the pure math: sector aggregation, point-in-time (as-of) z-scoring with NO lookahead,
weight renormalization when components are missing, the blended sector_state, and the
cross-sectional combined signal. Network-free -- the runner supplies real inputs.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dcf_project.engine_v2 import sector_map as sm
from dcf_project.engine_v2 import sector_state as ss


# --- sector_map ---------------------------------------------------------------------------------

def test_curated_ticker_sector_and_etf() -> None:
    assert sm.sector_for_ticker("AAPL") == "technology"
    assert sm.sector_for_ticker("GOOGL") == "communication_services"
    assert sm.sector_for_ticker("KO") == "consumer_staples"
    assert sm.sector_for_ticker("V") == "financials"  # payment networks -> current GICS Financials
    assert sm.etf_for_sector("technology") == "XLK"
    assert sm.etf_for_sector(sm.sector_for_ticker("AMZN")) == "XLY"


def test_sic_crosswalk_fallback() -> None:
    assert sm.sector_for_ticker("ZZZZ", sic=3674) == "technology"      # semiconductors
    assert sm.sector_for_ticker("ZZZZ", sic="2834") == "health_care"   # pharma
    assert sm.sector_for_ticker("ZZZZ", sic=1311) == "energy"          # crude petroleum
    assert sm.sector_for_ticker("UNKNOWN") is None                      # no curated, no SIC


def test_etf_availability_respects_inception() -> None:
    assert sm.etf_available_asof("XLK", date(2015, 3, 31)) is True
    assert sm.etf_available_asof("XLC", date(2017, 1, 1)) is False  # XLC launched 2018-06
    assert sm.etf_available_asof("XLC", date(2019, 1, 1)) is True


# --- raw aggregation ----------------------------------------------------------------------------

def test_aggregate_earnings_yield() -> None:
    # Sum earnings 30 / Sum cap 600 = 0.05.
    assert ss.aggregate_earnings_yield([(100, 5), (200, 10), (300, 15)]) == pytest.approx(0.05)
    # Fewer than MIN_MEMBERS -> None.
    assert ss.aggregate_earnings_yield([(100, 5), (200, 10)]) is None
    # Non-positive / non-numeric caps are dropped before the count check.
    assert ss.aggregate_earnings_yield([(0, 5), (-1, 5), (100, 5), (200, 10)]) is None


def test_aggregate_revisions_median() -> None:
    assert ss.aggregate_revisions([0.1, 0.2, 0.3]) == pytest.approx(0.2)
    assert ss.aggregate_revisions([0.1, 0.2]) is None                       # < MIN_MEMBERS
    assert ss.aggregate_revisions([float("nan"), 0.2, 0.3, 0.4]) == pytest.approx(0.3)  # NaN dropped


def test_sector_momentum_asof_sign_and_none() -> None:
    idx = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    etf = pd.Series([100.0 * (1.0 + 0.001 * i) for i in range(len(idx))], index=idx)  # rising
    spy = pd.Series([100.0] * len(idx), index=idx)                                     # flat
    mom = ss.sector_momentum_asof(etf, spy, date(2020, 6, 30))
    assert mom is not None and mom > 0.0                  # ETF outperformed flat SPY
    # Before the series begins -> no as-of price -> None (mirrors pre-inception ETFs).
    assert ss.sector_momentum_asof(etf, spy, date(2018, 1, 1)) is None
    assert ss.sector_momentum_asof(None, spy, date(2020, 6, 30)) is None


# --- as-of z-scoring (no lookahead) + weighted combination --------------------------------------

_DATES = [date(2018, 3, 31), date(2018, 6, 30), date(2018, 9, 30),
          date(2018, 12, 31), date(2019, 3, 31), date(2019, 6, 30), date(2019, 9, 30)]
_Z6 = 1.463850109  # z of value 6 vs population history [1..6]: (6-3.5)/popstd([1..6])


def _panel_with(valuation=None, revisions=None, momentum=None, flows=None) -> ss.SectorPanel:
    panel = ss.SectorPanel()
    for i, d in enumerate(_DATES):
        panel.set_raw("technology", d,
                      valuation=valuation[i] if valuation else None,
                      revisions=revisions[i] if revisions else None,
                      momentum=momentum[i] if momentum else None,
                      flows=flows[i] if flows else None)
    return panel


def test_zscore_is_as_of_and_needs_min_history() -> None:
    # Ramp [1..6] over the first six dates, then a wild 100 on the seventh.
    panel = _panel_with(valuation=[1, 2, 3, 4, 5, 6, 100])
    # At the 6th date exactly MIN_HISTORY=6 points exist; z uses ONLY [1..6] (the 100 is later).
    assert panel.state("technology", _DATES[5])["sector_valuation"] == pytest.approx(_Z6, rel=1e-6)
    # At the 5th date only 5 points <= it -> below MIN_HISTORY -> None.
    assert panel.state("technology", _DATES[4])["sector_valuation"] is None
    # The 100 only enters once we are AT/after the 7th date -> proves no lookahead at date 6.
    assert panel.state("technology", _DATES[6])["sector_valuation"] != pytest.approx(_Z6, rel=1e-6)


def test_degenerate_std_yields_none() -> None:
    panel = _panel_with(valuation=[5, 5, 5, 5, 5, 5, 5])  # zero variance
    assert panel.state("technology", _DATES[5])["sector_valuation"] is None


def test_weight_renormalization_and_coverage() -> None:
    # valuation z = +Z6, revisions z = -Z6 (reversed ramp), momentum/flows absent.
    panel = _panel_with(valuation=[1, 2, 3, 4, 5, 6, 6], revisions=[6, 5, 4, 3, 2, 1, 1])
    st = panel.state("technology", _DATES[5])
    assert st["coverage"] == pytest.approx(0.65)        # 0.35 + 0.30, flows+momentum missing
    # (0.35*Z6 + 0.30*(-Z6)) / 0.65
    assert st["sector_state"] == pytest.approx((0.35 - 0.30) * _Z6 / 0.65, rel=1e-6)
    assert st["sector_momentum"] is None and st["sector_flows"] is None


def test_below_min_coverage_yields_no_state() -> None:
    # Only momentum present -> coverage 0.20 < MIN_WEIGHT_COVERAGE 0.5 -> sector_state None.
    panel = _panel_with(momentum=[1, 2, 3, 4, 5, 6, 7])
    st = panel.state("technology", _DATES[5])
    assert st["sector_momentum"] == pytest.approx(_Z6, rel=1e-6)
    assert st["sector_state"] is None
    assert panel.score("technology", _DATES[5]) is None


def test_score_matches_state() -> None:
    panel = _panel_with(valuation=[1, 2, 3, 4, 5, 6, 6], revisions=[1, 2, 3, 4, 5, 6, 6],
                        momentum=[1, 2, 3, 4, 5, 6, 6])
    st = panel.state("technology", _DATES[5])
    # All three components equal +Z6 -> blended state equals Z6 (renormalized over coverage 0.85).
    assert st["sector_state"] == pytest.approx(_Z6, rel=1e-6)
    assert panel.score("technology", _DATES[5]) == pytest.approx(st["sector_state"])


# --- cross-sectional combined signal ------------------------------------------------------------

def test_combined_cross_sectional_polarity_and_gating() -> None:
    rows = [
        {"date": "A", "v2_discount": 3.0, "sector_state": 3.0},
        {"date": "A", "v2_discount": 1.0, "sector_state": 1.0},
        {"date": "A", "v2_discount": -1.0, "sector_state": -1.0},
        {"date": "A", "v2_discount": -3.0, "sector_state": -3.0},
        # date B has only one usable row -> < 3 -> all None
        {"date": "B", "v2_discount": 2.0, "sector_state": None},
        {"date": "B", "v2_discount": 1.0, "sector_state": 1.0},
        # date C has 3 usable + 1 missing -> missing row None, others finite
        {"date": "C", "v2_discount": 1.0, "sector_state": 1.0},
        {"date": "C", "v2_discount": 2.0, "sector_state": 2.0},
        {"date": "C", "v2_discount": 3.0, "sector_state": 3.0},
        {"date": "C", "v2_discount": 9.0, "sector_state": None},
    ]
    ss.combined_cross_sectional(rows, "v2_discount", "sector_state", "combined")
    a = [r["combined"] for r in rows if r["date"] == "A"]
    assert a[0] == max(a) and a[3] == min(a)                  # higher in both -> higher combined
    assert sum(a) == pytest.approx(0.0, abs=1e-9)             # z-sum is mean-zero in-group
    assert all(r["combined"] is None for r in rows if r["date"] == "B")   # < 3 usable
    c_missing = [r for r in rows if r["date"] == "C" and r["sector_state"] is None][0]
    assert c_missing["combined"] is None                     # missing input -> None
    assert all(r["combined"] is not None for r in rows
               if r["date"] == "C" and r["sector_state"] is not None)
