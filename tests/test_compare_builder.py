"""Builder tests for the triangulation workbook (`valuation compare` -> comparison.xlsx).

Drives ``build_compare_model`` off the v1 + v3 fixture pipelines (v3 with MC on) plus a
synthetic generic leg, asserting: the section set; the assumption-diff table carrying the
fade-only economics (phi / CAP / half-lives) with em-dashes for the other engines; the
Monte-Carlo G6 caption on the triangulation summary; the v1-vs-fade FCF overlay spanning
both horizons; and a clean xlsx render.
"""
from __future__ import annotations

from openpyxl import load_workbook

from dcf_project.config import Settings
from dcf_project.reporting_kit import xlsx
from dcf_project.valuation.compare import build_compare_model
from dcf_project.valuation.report_fade import G6_CAPTION
from test_engine_v3_assemble import SETTINGS as SETTINGS_V3, V3FixturePipeline
from test_pipeline import FixturePipeline

GENERIC = {"intrinsic_value_per_share": 9.5, "terminal_growth_rate": 0.025,
           "revenue_cagr_5y": 0.05, "normalized_ebit_margin": 0.20, "enterprise_value": 950.0, "wacc": 0.09}


def _inputs():
    res1 = FixturePipeline(Settings(terminal_growth_source="internal")).run("TST", horizon_quarters=20)
    res3 = V3FixturePipeline(SETTINGS_V3).run("TST")
    sc_v1 = {"result": res1, "intrinsic_value": res1.intrinsic_value_per_share, "price": res1.market_price,
             "discount_to_intrinsic": res1.implied_upside, "timestamp_utc": "2026-06-11",
             "assumptions": {"wacc": res1.wacc, "terminal_growth": res1.terminal_growth_rate,
                             "implied_annual_revenue_growth": res1.fcf_growth_rate,
                             "terminal_operating_margin": 0.20}}
    sc_v3 = {"result": res3, "intrinsic_value": res3.intrinsic_value_per_share, "price": res3.market_price,
             "discount_to_intrinsic": res3.implied_upside, "timestamp_utc": "2026-06-11",
             "assumptions": {"wacc": res3.wacc, "terminal_growth": res3.terminal_growth_rate}}
    return sc_v1, sc_v3


def _section(model, name):
    return next(s for s in model.sections if s.name == name)


def _diff_row(table, metric):
    return next(r for r in table.rows if r[0] == metric)


def test_sections_and_assumption_diff(tmp_path):
    sc_v1, sc_v3 = _inputs()
    model = build_compare_model("TST", GENERIC, sc_v1, sc_v3, tmp_path)
    names = {s.name for s in model.sections}
    assert {"Triangulation Summary", "Assumption Diff", "Forecast Overlay", "Why They Differ"} <= names
    diff = _section(model, "Assumption Diff").tables[0]
    assert diff.columns == ["Metric", "Generic", "Canonical v1", "Fade v3"]
    # Fade-only economics: present for v3, em-dash for the other two engines.
    for metric in ("Persistence phi", "CAP", "Growth half-life", "Margin half-life"):
        g, v1, v3 = _diff_row(diff, metric)[1:]
        assert g == "--" and v1 == "--" and v3 != "--", metric
    # Shared rows are populated for every engine.
    assert all(cell != "--" for cell in _diff_row(diff, "Intrinsic / share")[1:])


def test_mc_band_g6_caption(tmp_path):
    sc_v1, sc_v3 = _inputs()
    model = build_compare_model("TST", GENERIC, sc_v1, sc_v3, tmp_path)
    narr = _section(model, "Triangulation Summary").narrative
    assert any(G6_CAPTION in line for line in narr)  # MC ran (fixture draws=200)


def test_forecast_overlay_spans_both_horizons(tmp_path):
    sc_v1, sc_v3 = _inputs()
    model = build_compare_model("TST", GENERIC, sc_v1, sc_v3, tmp_path)
    tbl = _section(model, "Forecast Overlay").charts[0].table
    assert len(tbl.rows) == 40                       # fade horizon
    assert tbl.rows[30][1] is None                   # v1 ended at quarter 20
    assert tbl.rows[30][2] is not None               # fade still forecasting


def test_generic_leg_optional_and_render(tmp_path):
    sc_v1, sc_v3 = _inputs()
    # No generic leg: builder must still produce a valid model (em-dash generic column).
    model = build_compare_model("TST", None, sc_v1, sc_v3, tmp_path)
    diff = _section(model, "Assumption Diff").tables[0]
    assert _diff_row(diff, "Intrinsic / share")[1] == "--"  # generic intrinsic absent
    out = xlsx.render(model, tmp_path / "comparison.xlsx")
    wb = load_workbook(out)
    assert {"Triangulation Summary", "Assumption Diff", "Forecast Overlay", "Why They Differ"} <= set(wb.sheetnames)
