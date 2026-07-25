"""Builder tests for the flagship Fundamentals Fade workbook (engine_v3 -> fade_dcf.xlsx).

Drives ``build_fade_model`` off the deterministic v3 fixture pipeline (Monte Carlo on,
200 draws), asserting: the section set; the mandatory G6 uncertainty caption; that the
sensitivity grid's centre cell reproduces the headline intrinsic (the re-pricing
invariant); and that the model renders to a real workbook without an openpyxl error.
"""
from __future__ import annotations

import pytest
from openpyxl import load_workbook

from dcf_project.reporting_kit import xlsx
from dcf_project.valuation.report_fade import G6_CAPTION, build_fade_model
from test_engine_v3_assemble import SETTINGS, V3FixturePipeline  # fixture has v3_mc_draws=200


def _sc() -> dict:
    res = V3FixturePipeline(SETTINGS).run("TST")
    return {
        "ticker": "TST",
        "result": res,
        "assumptions": {"wacc": res.wacc, "terminal_growth": res.terminal_growth_rate},
        "discount_to_intrinsic": res.implied_upside,
        "intrinsic_value": res.intrinsic_value_per_share,
        "price": res.market_price,
        "verdict": "FAIR",
        "timestamp_utc": "2026-06-11T00:00:00Z",
        "why": ["Synthetic fixture run."],
        "qualifiers": ["Fixture data; not investable."],
    }


def _section(model, name):
    return next(s for s in model.sections if s.name == name)


def test_section_set_and_headline_basis(tmp_path) -> None:
    model = build_fade_model(_sc(), tmp_path)
    names = {s.name for s in model.sections}
    assert {
        "Executive Summary", "Valuation Summary", "Fade Drivers", "CAP & Persistence",
        "Terminal Value", "Sensitivity", "Driver Attribution", "Forecast Audit Trail",
        "Uncertainty", "Historical", "Investment Thesis",
    } <= names
    exec_narr = " ".join(_section(model, "Executive Summary").narrative).lower()
    assert "deterministic" in exec_narr and "skeleton" in exec_narr
    assert "reinv" in exec_narr  # identity-derived reinvestment is explained


def test_g6_caption_present_when_mc_ran(tmp_path) -> None:
    model = build_fade_model(_sc(), tmp_path)
    un = _section(model, "Uncertainty")
    assert any(G6_CAPTION in line for line in un.narrative)
    # No probability/coverage claim leaks into the headline metrics.
    labels = {mtr.label.lower() for mtr in _section(model, "Executive Summary").metrics}
    assert not any("probab" in lab or "coverage" in lab for lab in labels)


def test_sensitivity_centre_equals_headline(tmp_path) -> None:
    sc = _sc()
    model = build_fade_model(sc, tmp_path)
    grid = _section(model, "Sensitivity").tables[0]  # WACC (rows) x terminal growth (cols)
    centre = grid.rows[2][3]  # 5x5: centre data cell (col 0 is the WACC row label)
    assert centre == pytest.approx(round(sc["result"].intrinsic_value_per_share, 2))


def test_bridge_and_terminal_decomposition(tmp_path) -> None:
    model = build_fade_model(_sc(), tmp_path)
    bridge = _section(model, "Valuation Summary").tables[0]
    assert bridge.total_row and bridge.rows[-1][0] == "Upside / (downside)"
    # Grinold-Kroner decomposition resolves to the terminal growth row.
    gk = _section(model, "Terminal Value").tables[0]
    assert any("Terminal growth" in str(r[0]) for r in gk.rows)


def test_renders_to_xlsx(tmp_path) -> None:
    model = build_fade_model(_sc(), tmp_path)
    out = xlsx.render(model, tmp_path / "fade_dcf.xlsx")
    wb = load_workbook(out)
    assert "Fade Drivers" in wb.sheetnames and "Terminal Value" in wb.sheetnames
    fd = wb["Fade Drivers"]
    assert "TST" in (fd.oddFooter.left.text or "")
    assert fd.page_setup.orientation == "landscape"
