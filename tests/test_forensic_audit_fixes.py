"""Regression tests for the universe forensic-audit fixes.

Four confirmed defects (with evidence from the 15-name universe):
  1. Classifier divided balance-sheet stocks (PPE, invested capital) by quarterly
     revenue, inflating ratios ~4x -> pharma (LLY) and media (DIS) misclassified as
     telecom.
  2. Total-debt SEC concepts reported as a single combined line (e.g. Verizon's
     DebtLongtermAndShorttermCombinedAmount) were unmapped -> debt under-captured.
  3. Debt taken from the latest quarter alone understated leverage when total debt
     is disclosed only annually -> use trailing-4-quarter max.
  4. No WACC floor -> low-beta/high-leverage names got a ~4.25% all-in discount
     rate, exploding the terminal multiple.
"""

from __future__ import annotations


import pandas as pd

from dcf_project.config import Settings
from dcf_project.contracts import MarketSnapshot, SourceMeta
from dcf_project.data_sources.market import build_wacc
from dcf_project.data_sources.sec import CONCEPT_MAP, _merge_fact_value
from dcf_project.models.classification import _asset_metrics, resolve_company_type
from dcf_project.pipeline import _latest_debt_stock


# ---- Fix 1: classifier annualizes stock ratios ----

def _history_with(ppe_to_annual_rev: float, quarters: int = 12) -> pd.DataFrame:
    # quarterly revenue 25, so annual = 100; ppe set to hit the target annual ratio
    rev_q = 25.0
    ppe = ppe_to_annual_rev * (rev_q * 4.0)
    rows = []
    for i in range(quarters):
        rows.append({
            "revenue": rev_q,
            "ppe": ppe,
            "invested_capital": ppe,
            "capex": 0.5,
            "operating_margin": 0.30,
            "roic": 0.05,
            "r_and_d": 6.0,  # 24% of quarterly revenue -> pharma/tech signal
            "revenue_growth": 0.03,
        })
    return pd.DataFrame(rows)


def test_classifier_uses_annual_stock_ratios() -> None:
    # Annual ppe/revenue = 0.9 (below telecom's 1.10). Quarterly ratio is 3.6.
    metrics = _asset_metrics(_history_with(0.9))
    assert metrics["ppe_to_revenue"] < 1.10
    assert abs(metrics["ppe_to_revenue"] - 0.9) < 0.05
    assert metrics["invested_capital_to_revenue"] < 1.60


def test_high_ppe_pharma_not_misclassified_as_telecom() -> None:
    # A high-margin, high-R&D firm with annual ppe/rev 0.9 must NOT be telecom.
    resolved, _ = resolve_company_type("UNKNOWNPHARMA", "auto", _history_with(0.9))
    assert resolved != "telecom"


def test_genuine_telecom_still_classified_when_annual_ratio_high() -> None:
    # Annual ppe/rev = 1.5 -> still telecom.
    resolved, _ = resolve_company_type("UNKNOWNTELCO", "auto", _history_with(1.5))
    assert resolved == "telecom"


# ---- Fix 2: combined-debt concept is mapped ----

def test_combined_debt_concepts_are_mapped() -> None:
    assert "DebtLongtermAndShorttermCombinedAmount" in CONCEPT_MAP["debt_total"]
    assert "DebtAndCapitalLeaseObligations" in CONCEPT_MAP["debt_total"]


def test_merge_fact_value_takes_max_total_debt() -> None:
    period: dict = {}
    _merge_fact_value(period, "debt_total", 18.6e9)   # current-portion-only quarter
    _merge_fact_value(period, "debt_total", 158.2e9)  # annual combined total
    assert period["debt_total"] == 158.2e9


# ---- Fix 3: trailing-max debt carry-forward ----

def test_latest_debt_stock_uses_trailing_max() -> None:
    # Verizon-like: total debt disclosed only on the annual (Dec) period.
    frame = pd.DataFrame({"debt": [22.6e9, 22.1e9, 20.1e9, 158.2e9, 28.2e9]})
    assert _latest_debt_stock(frame) == 158.2e9  # max of trailing 4 (incl. annual)


def test_latest_debt_stock_stable_for_consistent_reporters() -> None:
    frame = pd.DataFrame({"debt": [100.0, 101.0, 102.0, 103.0]})
    assert _latest_debt_stock(frame) == 103.0


# ---- Fix 4: WACC floor ----

def _market() -> MarketSnapshot:
    return MarketSnapshot(
        ticker="VZ", price=40.0, market_cap=170e9, shares_outstanding=4.2e9,
        beta=0.40, cash=5e9, debt=150e9,
        source=SourceMeta(provider="fixture", key="VZ"),
    )


def test_wacc_floor_prevents_subfloor_discount_rate() -> None:
    settings = Settings()
    wacc = build_wacc(_market(), tax_rate=0.21, debt=150e9, settings=settings)
    floor = wacc.risk_free_rate + 0.5 * settings.equity_risk_premium
    assert wacc.wacc >= floor - 1e-9
    assert wacc.wacc >= 0.055  # must not collapse toward the risk-free rate


def test_wacc_floor_does_not_lower_high_beta_names() -> None:
    settings = Settings()
    high_beta = replace_market_beta(_market(), 1.4)
    wacc = build_wacc(high_beta, tax_rate=0.21, debt=20e9, settings=settings)
    # A normal-beta, low-leverage name is well above the floor -> unchanged by it.
    assert wacc.wacc > wacc.risk_free_rate + 0.5 * settings.equity_risk_premium


def replace_market_beta(m: MarketSnapshot, beta: float) -> MarketSnapshot:
    return m.model_copy(update={"beta": beta, "debt": 20e9})
