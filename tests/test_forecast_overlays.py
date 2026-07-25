from __future__ import annotations

import pandas as pd
import pytest

from dcf_project.config import Settings
from dcf_project.data_sources.consensus import ForwardConsensus
from dcf_project.models.overlays import apply_forecast_overlays
from dcf_project.models.priors import archetype_for, prior_for
from dcf_project.models.state_space import financials_from_states


def test_ticker_archetype_priors() -> None:
    assert archetype_for("AAPL", "asset_light") == "aapl"
    assert archetype_for("MSFT", "asset_light") == "msft"
    assert archetype_for("TSLA", "asset_light") == "capital_intensive_growth"
    assert prior_for("GOOGL", "asset_light").margin_persistence > 0.99
    assert prior_for("TSLA", "asset_light").reinvestment_burden > 0.20


def test_consensus_blend_and_buyback_outputs() -> None:
    financials = _forecast_frame()
    history = _history_frame()
    consensus = ForwardConsensus(
        provider="fixture",
        revenue_growth_annual=0.12,
        ebit_margin=0.35,
        fcf_margin=0.28,
        retrieved_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        fallback_used=False,
        source_notes=[],
    )

    adjusted, diagnostics = apply_forecast_overlays(
        "AAPL",
        financials,
        history,
        consensus,
        Settings(
            forecast_mode="consensus_blend",
            consensus_weight=0.70,
            terminal_growth_source="internal",
        ),
    )

    assert diagnostics["forecast_mode"] == "consensus_blend"
    assert diagnostics["requested_consensus_weight"] == pytest.approx(0.70)
    assert 0.0 < diagnostics["consensus_weight"] < 0.70
    assert diagnostics["dynamic_consensus_weight_trace"]["quarterly_weights"]
    assert diagnostics["historical_forecast_calibration"]["summary"]
    assert diagnostics["fade_trace"]["method"] == "classification_volatility_scaled_economic_fade"
    assert diagnostics["buyback_rate"] > 0.0
    assert diagnostics["r_and_d_capitalized"] is True
    assert diagnostics["raw_model_forecast"] != diagnostics["blended_forecast"]
    assert diagnostics["consensus_forecast"]
    assert adjusted["revenue_growth"].iloc[0] > financials["revenue_growth"].iloc[0]
    assert adjusted["operating_margin"].iloc[0] > financials["operating_margin"].iloc[0]
    assert adjusted["shares"].iloc[-1] < history["shares"].iloc[-1]
    assert adjusted["r_and_d_capitalized"].iloc[0] > 0.0
    assert diagnostics["overlay_forecast"]
    assert diagnostics["final_forecast"]
    assert diagnostics["field_source_map"][0]["revenue_growth"]["source"] == "consensus"
    assert "reinvestment_rate" in diagnostics["clamp_triggered_map"]


def test_manual_segment_assumptions_override_consensus() -> None:
    financials = _forecast_frame()
    history = _history_frame()
    consensus = ForwardConsensus(
        provider="fixture",
        revenue_growth_annual=0.02,
        ebit_margin=0.20,
        fcf_margin=None,
        retrieved_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        fallback_used=False,
        source_notes=[],
    )

    adjusted, diagnostics = apply_forecast_overlays(
        "MSFT",
        financials,
        history,
        consensus,
        Settings(
            forecast_mode="consensus_blend",
            consensus_weight=1.0,
            segment_mode="manual",
            segment_assumptions_json=(
                '{"segments":[{"name":"Azure","weight":0.5,"revenue_growth_annual":0.20,'
                '"operating_margin":0.45},{"name":"Office","weight":0.5,'
                '"revenue_growth_annual":0.08,"operating_margin":0.40}]}'
            ),
            terminal_growth_source="internal",
        ),
    )

    expected_quarterly_growth = 0.5 * ((1.0 + 0.20) ** 0.25 - 1.0) + 0.5 * ((1.0 + 0.08) ** 0.25 - 1.0)
    assert diagnostics["segment_mode"] == "manual"
    assert adjusted["revenue_growth"].iloc[0] == pytest.approx(expected_quarterly_growth)
    assert adjusted["operating_margin"].iloc[0] == pytest.approx(0.425)


def test_state_space_mode_preserves_raw_model_path_for_core_fields() -> None:
    financials = _forecast_frame()
    financials["reinvestment_rate"] = [0.03 + idx * 0.002 for idx in range(len(financials))]
    financials["roic"] = [0.12 + idx * 0.001 for idx in range(len(financials))]
    history = _history_frame()
    consensus = ForwardConsensus(
        provider="fixture",
        revenue_growth_annual=0.20,
        ebit_margin=0.40,
        fcf_margin=0.30,
        retrieved_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        fallback_used=False,
        source_notes=[],
    )

    adjusted, diagnostics = apply_forecast_overlays(
        "AAPL",
        financials,
        history,
        consensus,
        Settings(forecast_mode="state_space", terminal_growth_source="internal"),
    )

    assert adjusted["revenue_growth"].tolist() == pytest.approx(financials["revenue_growth"].tolist())
    assert adjusted["operating_margin"].tolist() == pytest.approx(financials["operating_margin"].tolist())
    assert adjusted["reinvestment_rate"].nunique() > 1
    assert adjusted["roic"].nunique() > 1
    assert diagnostics["field_source_map"][0]["reinvestment_rate"]["source"] == "model"
    assert diagnostics["raw_model_forecast"][0]["reinvestment_rate"] == pytest.approx(0.03)


def test_consensus_weight_changes_final_forecast_path() -> None:
    financials = _forecast_frame()
    history = _history_frame()
    consensus = ForwardConsensus(
        provider="fixture",
        revenue_growth_annual=0.20,
        ebit_margin=0.40,
        fcf_margin=0.30,
        retrieved_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        fallback_used=False,
        source_notes=[],
    )

    low_weight, _ = apply_forecast_overlays(
        "AAPL",
        financials,
        history,
        consensus,
        Settings(forecast_mode="consensus_blend", consensus_weight=0.0, terminal_growth_source="internal"),
    )
    high_weight, _ = apply_forecast_overlays(
        "AAPL",
        financials,
        history,
        consensus,
        Settings(forecast_mode="consensus_blend", consensus_weight=1.0, terminal_growth_source="internal"),
    )

    assert high_weight["revenue_growth"].iloc[0] != pytest.approx(low_weight["revenue_growth"].iloc[0])
    assert high_weight["operating_margin"].iloc[0] != pytest.approx(low_weight["operating_margin"].iloc[0])


def test_dynamic_consensus_weight_preserves_more_resolution_than_static_weight() -> None:
    financials = _forecast_frame()
    financials["revenue_growth"] = [0.00 + idx * 0.002 for idx in range(len(financials))]
    history = _history_frame()
    consensus = ForwardConsensus(
        provider="fixture",
        revenue_growth_annual=0.12,
        ebit_margin=0.30,
        fcf_margin=0.25,
        retrieved_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        fallback_used=False,
        source_notes=[],
    )

    adjusted, diagnostics = apply_forecast_overlays(
        "TSLA",
        financials,
        history,
        consensus,
        Settings(
            forecast_mode="consensus_blend",
            consensus_weight=0.70,
            company_type="capital_intensive_growth",
            terminal_growth_source="internal",
        ),
    )

    raw_var = financials["revenue_growth"].var(ddof=0)
    final_var = adjusted["revenue_growth"].var(ddof=0)
    static_var = ((1.0 - 0.70) * financials["revenue_growth"] + 0.70 * ((1.0 + 0.12) ** 0.25 - 1.0)).var(ddof=0)
    assert diagnostics["consensus_weight"] < 0.70
    assert final_var / raw_var > static_var / raw_var
    assert 0.95 <= final_var / raw_var <= 1.05


def test_asset_light_quality_state_path_not_flattened_to_policy_defaults() -> None:
    states = pd.DataFrame(
        {
            "revenue_growth": [-0.02 + idx * 0.002 for idx in range(20)],
            "operating_margin": [0.30 - idx * 0.001 for idx in range(20)],
            "reinvestment_rate": [0.015 + idx * 0.002 for idx in range(20)],
            "roic": [0.20 + idx * 0.001 for idx in range(20)],
        }
    ).to_numpy(float)

    financials = financials_from_states(
        states,
        base_revenue=100.0,
        base_fcf=20.0,
        base_invested_capital=200.0,
        tax_rate=0.20,
        company_type="asset_light_quality",
        ticker="AAPL",
    )

    assert financials["reinvestment_rate"].nunique() > 1
    assert financials["roic"].nunique() > 1
    assert financials["reinvestment_rate"].iloc[0] != pytest.approx(0.08)
    assert financials["roic"].iloc[0] != pytest.approx(0.08)
    assert "raw_model_reinvestment_rate" in financials
    assert "roic_clamped" in financials


def _forecast_frame() -> pd.DataFrame:
    rows = []
    revenue = 100.0
    for quarter in range(1, 21):
        revenue *= 1.01
        rows.append(
            {
                "quarter": float(quarter),
                "revenue_growth": 0.01,
                "revenue": revenue,
                "operating_margin": 0.25,
                "ebit_margin": 0.25,
                "tax_rate": 0.20,
                "reinvestment_rate": 0.10,
                "invested_capital": 500.0,
                "roic": 0.05,
                "da": revenue * 0.03,
                "capex": revenue * 0.04,
                "delta_nwc": revenue * 0.01,
                "maintenance_capex": revenue * 0.02,
                "sbc_adjustment": revenue * 0.005,
                "fallback_triggered": 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["ebit"] = frame["revenue"] * frame["operating_margin"]
    frame["nopat"] = frame["ebit"] * (1.0 - frame["tax_rate"])
    frame["economic_fcf"] = frame["nopat"] * (1.0 - frame["reinvestment_rate"])
    frame["accounting_fcf"] = frame["nopat"] + frame["da"] - frame["capex"] - frame["delta_nwc"]
    frame["owner_earnings"] = frame["nopat"] + frame["da"] - frame["maintenance_capex"] - frame["delta_nwc"]
    frame["fcf"] = frame["economic_fcf"]
    return frame


def _history_frame() -> pd.DataFrame:
    rows = []
    shares = 110.0
    for quarter in range(20):
        shares -= 0.5
        rows.append(
            {
                "revenue_growth": 0.02,
                "operating_margin": 0.30,
                "fcf": 25.0,
                "revenue": 100.0,
                "shares": shares,
            }
        )
    return pd.DataFrame(rows)
