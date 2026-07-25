"""Regression tests for the phantom-period data-generation bug.

Background
----------
`SECClient.financial_periods` keys every fact by its XBRL ``end`` date. Instant
facts whose ``end`` does not align to a fiscal period end -- most notably the
``dei:EntityCommonStockSharesOutstanding`` cover-page share count (dated ~3 weeks
after the period end) and stale comparative balance-sheet instants -- used to be
emitted as standalone periods carrying *only* shares/cash and no income-statement
data.

Because those phantom rows are interleaved chronologically between the real
quarters, ``revenue.pct_change()`` (i.e. ``revenue_growth``) collapsed to NaN
almost everywhere: AAPL produced 1 usable quarter and MSFT produced 0, which
raised ``Insufficient complete quarterly history for state-space DCF`` before
valuation could run.

These tests build SEC-shaped ``companyfacts`` payloads (including the dei
cover-page phantom facts) for AAPL- and MSFT-like companies, drive them through
the *real* assembly + reconstruction path, and assert the phantom rows are
filtered so >= 20 complete quarters survive with non-degenerate ROIC and
reinvestment histories.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np

from dcf_project.config import Settings
from dcf_project.contracts import SourceMeta
from dcf_project.data_sources.sec import SECClient
from dcf_project.preprocessing.reconstruction import reconstruct_historical_fcf


def _add_fact(
    facts: dict[str, Any],
    taxonomy: str,
    concept: str,
    unit: str,
    fact: dict[str, Any],
) -> None:
    concept_node = facts["facts"].setdefault(taxonomy, {}).setdefault(concept, {"units": {}})
    concept_node["units"].setdefault(unit, []).append(fact)


def _quarter_end(idx: int) -> date:
    # Four quarter ends per year, anchored on a March fiscal cadence.
    base_year = 2016 + idx // 4
    month = [3, 6, 9, 12][idx % 4]
    return date(base_year, month, 28)


def _companyfacts(num_quarters: int, scale: float, growth: float) -> dict[str, Any]:
    """Build a SEC companyfacts payload with real quarters plus dei phantom rows."""

    facts: dict[str, Any] = {"facts": {"us-gaap": {}, "dei": {}}}
    revenue = 50_000_000_000.0 * scale
    for idx in range(num_quarters):
        revenue *= 1.0 + growth
        end = _quarter_end(idx)
        start = end - timedelta(days=89)
        fp = ["Q1", "Q2", "Q3", "FY"][idx % 4]
        fy = end.year
        form = "10-K" if fp == "FY" else "10-Q"
        accn = f"acc-{idx:04d}"
        flow_meta = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "fy": fy,
            "fp": fp,
            "form": form,
            "filed": (end + timedelta(days=21)).isoformat(),
            "accn": accn,
        }
        instant_meta = {
            "end": end.isoformat(),
            "fy": fy,
            "fp": fp,
            "form": form,
            "filed": (end + timedelta(days=21)).isoformat(),
            "accn": accn,
        }

        # Income-statement flow facts (these define a real reporting period).
        _add_fact(facts, "us-gaap", "Revenues", "USD", {**flow_meta, "val": revenue})
        _add_fact(facts, "us-gaap", "OperatingIncomeLoss", "USD", {**flow_meta, "val": revenue * 0.30})
        _add_fact(facts, "us-gaap", "IncomeTaxExpenseBenefit", "USD", {**flow_meta, "val": revenue * 0.05})
        _add_fact(
            facts,
            "us-gaap",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "USD",
            {**flow_meta, "val": revenue * 0.04},
        )
        _add_fact(
            facts,
            "us-gaap",
            "DepreciationDepletionAndAmortization",
            "USD",
            {**flow_meta, "val": revenue * 0.03},
        )

        # Balance-sheet instant facts.
        _add_fact(facts, "us-gaap", "AssetsCurrent", "USD", {**instant_meta, "val": revenue * 1.1})
        _add_fact(facts, "us-gaap", "LiabilitiesCurrent", "USD", {**instant_meta, "val": revenue * 0.9})
        _add_fact(
            facts,
            "us-gaap",
            "CashAndCashEquivalentsAtCarryingValue",
            "USD",
            {**instant_meta, "val": revenue * 0.4},
        )
        _add_fact(facts, "us-gaap", "PropertyPlantAndEquipmentNet", "USD", {**instant_meta, "val": revenue * 0.5})
        _add_fact(facts, "us-gaap", "Goodwill", "USD", {**instant_meta, "val": revenue * 0.2})
        _add_fact(
            facts,
            "us-gaap",
            "FiniteLivedIntangibleAssetsNet",
            "USD",
            {**instant_meta, "val": revenue * 0.1},
        )
        _add_fact(facts, "us-gaap", "OperatingLeaseLiability", "USD", {**instant_meta, "val": revenue * 0.05})
        _add_fact(facts, "us-gaap", "LongTermDebtNoncurrent", "USD", {**instant_meta, "val": revenue * 0.3})
        _add_fact(
            facts,
            "us-gaap",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "shares",
            {**flow_meta, "val": 15_000_000_000.0 * scale},
        )

        # Phantom trigger: dei cover-page share count dated AFTER the period end,
        # so it lands on its own standalone `end` date with no financial data.
        cover_date = end + timedelta(days=21)
        _add_fact(
            facts,
            "dei",
            "EntityCommonStockSharesOutstanding",
            "shares",
            {
                "end": cover_date.isoformat(),
                "fy": fy,
                "fp": fp,
                "form": form,
                "filed": cover_date.isoformat(),
                "accn": accn,
                "val": 15_000_000_000.0 * scale,
            },
        )

    return facts


def _periods_for(monkeypatch, ticker: str, scale: float, growth: float, num_quarters: int = 30):
    facts = _companyfacts(num_quarters, scale=scale, growth=growth)
    client = SECClient(Settings())
    monkeypatch.setattr(
        client,
        "company_facts",
        lambda _ticker: (facts, SourceMeta(provider="SEC", key="fixture")),
    )
    periods, _ = client.financial_periods(ticker)
    return periods


def test_phantom_dei_cover_rows_are_filtered(monkeypatch) -> None:
    periods = _periods_for(monkeypatch, "AAPL", scale=2.0, growth=0.02)
    # Every emitted period must carry income-statement data: no phantom rows.
    assert all(p.revenue is not None for p in periods)
    assert all(p.ebit is not None for p in periods)
    # Real quarters retain their share counts (us-gaap weighted diluted).
    assert all(p.shares_outstanding is not None for p in periods)


def test_aapl_shaped_facts_yield_at_least_20_usable_quarters(monkeypatch) -> None:
    periods = _periods_for(monkeypatch, "AAPL", scale=2.0, growth=0.02)
    result = reconstruct_historical_fcf(periods)
    assert len(result.history) >= 20


def test_msft_shaped_facts_yield_at_least_20_usable_quarters(monkeypatch) -> None:
    periods = _periods_for(monkeypatch, "MSFT", scale=1.5, growth=0.015)
    result = reconstruct_historical_fcf(periods)
    assert len(result.history) >= 20


def test_reconstruction_does_not_produce_all_nan_roic_or_reinvestment(monkeypatch) -> None:
    periods = _periods_for(monkeypatch, "MSFT", scale=1.5, growth=0.015)
    history = reconstruct_historical_fcf(periods).history
    assert history["roic"].notna().any()
    assert history["reinvestment_rate"].notna().any()
    # Guard against a degenerate constant-fill masquerading as a real history.
    assert np.isfinite(history["roic"].to_numpy()).all()
    assert np.isfinite(history["reinvestment_rate"].to_numpy()).all()
    assert history["roic"].nunique() > 1
