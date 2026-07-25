from __future__ import annotations

import pandas as pd

from dcf_project.models.classification import resolve_company_type


def test_auto_uses_explicit_ticker_mapping() -> None:
    resolved, trace = resolve_company_type("MSFT", "auto", pd.DataFrame())
    assert resolved == "hyperscaler"
    assert trace["method"] == "explicit_ticker_mapping"


def test_tsla_auto_does_not_resolve_to_asset_light_quality() -> None:
    resolved, trace = resolve_company_type("TSLA", "auto", pd.DataFrame())
    assert resolved == "capital_intensive_growth"
    assert resolved != "asset_light_quality"
    assert trace["method"] == "explicit_ticker_mapping"


def test_asset_intensive_history_routes_to_industrial() -> None:
    history = pd.DataFrame(
        {
            "revenue": [100.0] * 12,
            "capex": [9.0] * 12,
            "ppe": [75.0] * 12,
            "invested_capital": [180.0] * 12,
            "operating_margin": [0.14] * 12,
            "roic": [0.05] * 12,
            "r_and_d": [1.0] * 12,
            "revenue_growth": [0.01] * 12,
        }
    )
    resolved, trace = resolve_company_type("TST", "auto", history)
    assert resolved == "industrial"
    assert trace["method"] == "asset_intensity_and_unit_economics"
