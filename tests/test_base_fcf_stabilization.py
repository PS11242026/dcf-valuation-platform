"""Regression test: base_fcf stabilization (value-neutral diagnostic fix).

A single quarter's FCF is volatile (AMZN's latest quarter was -$78.6B vs a
positive TTM). base_fcf does NOT drive intrinsic value, but it seeds the reported
FCF-growth diagnostic; using one anomalous quarter yields an economically invalid
headline growth. The pipeline now uses a trailing-4-quarter median.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcf_project.pipeline import _stabilized_base_fcf
from dcf_project.valuation.dcf import build_state_space_dcf_result


def test_stabilized_base_fcf_is_trailing_median_not_last_quarter() -> None:
    history = pd.DataFrame({"fcf": [10e9, 12e9, 20e9, -78e9]})  # last quarter anomalous
    # last-quarter would be -78e9; trailing-4 median is robust and positive.
    assert _stabilized_base_fcf(history) == np.median([10e9, 12e9, 20e9, -78e9])
    assert _stabilized_base_fcf(history) > 0


def test_stabilized_base_fcf_handles_empty_and_nan() -> None:
    assert _stabilized_base_fcf(pd.DataFrame({"fcf": []})) == 0.0
    assert _stabilized_base_fcf(pd.DataFrame({"other": [1, 2]})) == 0.0


def test_base_fcf_does_not_change_intrinsic_value() -> None:
    """Prove the stabilization is value-neutral: intrinsic is invariant to base_fcf."""
    ff = pd.DataFrame(
        {
            "quarter": [float(i) for i in range(1, 21)],
            "revenue": [500.0] * 20,
            "revenue_growth": [0.0] * 20,
            "operating_margin": [0.25] * 20,
            "ebit_margin": [0.25] * 20,
            "ebit": [125.0] * 20,
            "tax_rate": [0.0] * 20,
            "nopat": [125.0] * 20,
            "reinvestment_rate": [0.20] * 20,
            "reinvestment": [25.0] * 20,
            "invested_capital": [5_000.0] * 20,
            "roic": [0.025] * 20,
            "fcf": [100.0] * 20,
            "economic_fcf": [100.0] * 20,
            "accounting_fcf": [100.0] * 20,
            "owner_earnings": [100.0] * 20,
        }
    )
    from dcf_project.config import Settings
    from dcf_project.contracts import MarketSnapshot, SourceMeta, WACCInputs

    source = SourceMeta(provider="fixture", key="T")
    market = MarketSnapshot(ticker="T", price=20.0, market_cap=200.0, shares_outstanding=10.0,
                            beta=1.0, cash=10.0, debt=30.0, source=source)
    wacc = WACCInputs(risk_free_rate=0.04, beta=1.0, equity_risk_premium=0.05, cost_of_equity=0.09,
                      pre_tax_cost_of_debt=0.045, tax_rate=0.21, market_value_equity=200.0,
                      market_value_debt=30.0, equity_weight=200.0 / 230.0, debt_weight=30.0 / 230.0,
                      after_tax_cost_of_debt=0.045 * 0.79, wacc=0.09)
    settings = Settings(monte_carlo_samples=0, run_sensitivity=False, terminal_growth_source="internal")

    def iv(base_fcf):
        return build_state_space_dcf_result(
            "T", forecast_financials=ff, base_fcf=base_fcf, fcf_growth_rate=0.0,
            wacc_inputs=wacc, market=market, latest_cash=10.0, latest_debt=30.0,
            latest_shares=10.0, sources=[source], warnings=[], settings=settings,
        ).intrinsic_value_per_share

    assert iv(-78e9) == iv(100.0) == iv(5e9)  # intrinsic invariant to base_fcf
