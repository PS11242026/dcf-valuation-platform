"""Driver extraction tests (engine_v3): TTM annualization, bands, quality, typed skips."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dcf_project.engine_v3.drivers import (
    DriverExtractionError,
    extract_drivers,
)

WACC = 0.09


def make_history(
    n: int = 28,
    growth_q: float = 0.02,
    nopat_margin: float = 0.15,
    fcf_margin: float = 0.12,
    ic_multiple: float = 4.0,
    tax_rate: float = 0.21,
) -> pd.DataFrame:
    revenue = 100.0 * (1.0 + growth_q) ** np.arange(n)
    return pd.DataFrame(
        {
            "revenue": revenue,
            "nopat": revenue * nopat_margin,
            "fcf": revenue * fcf_margin,
            "invested_capital": revenue * ic_multiple,
            "tax_rate": tax_rate,
        }
    )


def test_growth_annualized_via_ttm_exactly() -> None:
    d = extract_drivers(make_history(growth_q=0.02), WACC)
    # Constant 2%/quarter growth -> TTM-over-TTM = 1.02^4 - 1, never "x4".
    assert d.g_base.value == pytest.approx(1.02**4 - 1.0, rel=1e-12)
    # Constant readings -> dispersion floored at the minimum standard error.
    assert d.g_base.se == pytest.approx(0.01)


def test_margins_and_band_from_constant_history() -> None:
    d = extract_drivers(make_history(), WACC)
    assert d.margin_current == pytest.approx(0.15, rel=1e-12)
    assert d.margin_target_company.value == pytest.approx(0.15, rel=1e-12)
    assert d.fcf_margin_current == pytest.approx(0.12, rel=1e-12)
    assert d.fcf_margin_band["p10"] == pytest.approx(0.12, rel=1e-12)
    assert d.fcf_margin_band["p50"] == pytest.approx(0.12, rel=1e-12)
    assert d.fcf_margin_band["p90"] == pytest.approx(0.12, rel=1e-12)


def test_roic_is_ttm_nopat_over_invested_capital() -> None:
    h = make_history()
    d = extract_drivers(h, WACC)
    expected = h["nopat"].tail(4).sum() / h["invested_capital"].iloc[-1]
    assert d.roic_annual_0 == pytest.approx(expected, rel=1e-12)
    assert d.excess_0 == pytest.approx(expected - WACC, rel=1e-12)
    assert d.base_revenue_ttm == pytest.approx(h["revenue"].tail(4).sum(), rel=1e-12)
    assert d.base_nopat_ttm == pytest.approx(h["nopat"].tail(4).sum(), rel=1e-12)


def test_non_positive_nopat_is_typed_skip() -> None:
    h = make_history()
    h["nopat"] = -1.0
    with pytest.raises(DriverExtractionError, match="v3_not_applicable"):
        extract_drivers(h, WACC)


def test_short_history_rejected() -> None:
    with pytest.raises(DriverExtractionError, match="at least 20"):
        extract_drivers(make_history(n=16), WACC)


def test_unusable_invested_capital_rejected() -> None:
    h = make_history()
    h["invested_capital"] = np.nan
    with pytest.raises(DriverExtractionError, match="invested-capital"):
        extract_drivers(h, WACC)


def test_quality_score_responds_to_flags() -> None:
    h = make_history(n=28)
    assert extract_drivers(h, WACC).quality == 0.0

    h["invested_capital_continuity_flag"] = [True] * 7 + [False] * 21  # 25% flagged
    d = extract_drivers(h, WACC)
    assert d.quality == pytest.approx(0.5)  # 2 * 0.25

    h["goodwill_stale_carry_forward_flag"] = [True] * 14 + [False] * 14  # 50% stale
    d = extract_drivers(h, WACC)
    assert d.quality == pytest.approx(0.75)  # 2*0.25 + 0.5*0.5
    assert any("quality penalty" in n for n in d.notes)


def test_growth_winsorized_at_band() -> None:
    h = make_history(n=28, growth_q=0.0)
    # One absurd revenue spike: the YoY readings touched by it must be winsorized,
    # so g_base stays inside the sanity band.
    h.loc[h.index[-1], "revenue"] *= 50.0
    h.loc[h.index[-1], "nopat"] = h.loc[h.index[-1], "revenue"] * 0.15
    d = extract_drivers(h, WACC)
    assert d.g_base.value <= 1.0 + 1e-12
