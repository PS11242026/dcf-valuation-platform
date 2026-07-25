"""Smoke tests for the reporting kit: synthetic ReportModel -> xlsx / pdf / pptx + figures.

Offline and fast. Exercises the institutional workbook features (cover TOC + hyperlinks,
tab colors, zebra / heatmap / total tables, native charts with axis titles, KPI notes,
landscape page setup + footer), the multi-page PDF flow renderer and the PPTX deck
(pagination + speaker notes), and the football-field / tornado figure generators.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import load_workbook

from dcf_project.reporting_kit import Image, Metric, NativeChart, ReportModel, Table, pdf, viz3d, xlsx


def _model(figdir: Path) -> ReportModel:
    m = ReportModel(
        title="TST -- Smoke DCF",
        subtitle="synthetic",
        persona="initiation",
        meta={"Ticker": "TST", "As of": "2026-06-10", "Engine": "v3"},
    )
    s = m.section("Executive Summary", "Executive Summary")
    s.metrics = [
        Metric("Intrinsic / sh", 12.34, "usd", note="hover note"),
        Metric("Enterprise value", 5.0e9, "usdB"),
        Metric("CAP", 11.8, "yrs"),
        Metric("Upside", 0.18, "pct", good=True),
    ]
    s.narrative = ["Line one.", "Line two."]
    s.tables.append(Table(
        "Bridge", ["Item", "Value"],
        [["PV FCF", 1.0e9], ["PV TV", 4.0e9], ["EV", 5.0e9], ["Equity", 4.5e9]],
        formats=["text", "usdM"], total_row=True, heatmap_cols=[1],
    ))
    fsec = m.section("Forecast", "Forecast")
    chart_tbl = Table("Paths", ["Q", "Growth", "Margin"],
                      [[i, 0.12 - i * 0.002, 0.20 + i * 0.001] for i in range(1, 11)],
                      formats=["num0", "pct", "pct"], col_widths=[8, 12, 12])
    fsec.charts.append(NativeChart("Driver fade", "line", chart_tbl, x_col=0, y_cols=[1, 2],
                                   x_title="Quarter", y_title="Rate"))
    ff = viz3d.football_field(
        [{"label": "Generic", "low": 9, "high": 14, "point": 11},
         {"label": "Fade", "low": 10, "high": 18, "point": 13}],
        price=12.0, out=figdir / "ff.png")
    if ff:
        fsec.images.append(Image("Football field", ff))
    return m


def test_xlsx_render_and_reopen(tmp_path: Path) -> None:
    m = _model(tmp_path / "figures")
    out = xlsx.render(m, tmp_path / "wb.xlsx")
    assert out.exists()
    wb = load_workbook(out)
    assert wb.sheetnames[0] == "Cover"
    assert "Executive Summary" in wb.sheetnames and "Forecast" in wb.sheetnames
    cover = wb["Cover"]
    # TOC: at least one hyperlinked cell, and section titles present as cover text
    assert any(c.hyperlink for row in cover.iter_rows() for c in row)
    cover_text = {c.value for row in cover.iter_rows() for c in row if isinstance(c.value, str)}
    assert "Forecast" in cover_text and "Executive Summary" in cover_text
    # section sheet: footer carries the ticker, landscape page setup
    es = wb["Executive Summary"]
    assert "TST" in (es.oddFooter.left.text or "")
    assert es.page_setup.orientation == "landscape"


def test_football_field_and_tornado(tmp_path: Path) -> None:
    ff = viz3d.football_field([{"label": "v1", "low": 8, "high": 12, "point": 10}],
                              price=9.5, out=tmp_path / "ff.png")
    assert ff and Path(ff).exists()
    tor = viz3d.tornado([{"label": "WACC", "low": 9, "high": 14},
                         {"label": "g", "low": 11, "high": 12.5}],
                        base=11.5, out=tmp_path / "tor.png")
    assert tor and Path(tor).exists()


def _paginating_model(figdir: Path) -> ReportModel:
    """A model whose long table + narrative force PDF/PPTX continuation pages."""
    m = _model(figdir)
    big = m.section("Long Detail", "Long Detail")
    big.narrative = [f"Narrative line {i} -- coherence note." for i in range(14)]
    big.tables.append(Table(
        "40-quarter paths", ["Q", "Growth", "Margin", "RONIC", "Reinvestment"],
        [[i, 0.12 - i * 0.001, 0.2, 0.15, 0.3] for i in range(1, 41)],
        formats=["num0", "pct", "pct", "pct", "pct"], total_row=False))
    return m


def test_pdf_render_multipage(tmp_path: Path) -> None:
    out = pdf.render(_paginating_model(tmp_path / "figures"), tmp_path / "deck.pdf")
    assert out.exists()
    data = out.read_bytes()
    assert data[:4] == b"%PDF" and len(data) > 8000
    # cover + >=3 section pages + at least one table-continuation page
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") >= 5


def test_pptx_render_with_pagination(tmp_path: Path) -> None:
    pptx = pytest.importorskip("pptx")
    from dcf_project.reporting_kit import pptx as pptx_renderer
    out = pptx_renderer.render(_paginating_model(tmp_path / "figures"), tmp_path / "deck.pptx")
    assert out.exists()
    prs = pptx.Presentation(str(out))
    # title + agenda + >=3 section slides + table-continuation + image slides
    assert len(prs.slides) >= 6
    # speaker notes carry the full narrative for a section that has one
    noted = [sl for sl in prs.slides if sl.has_notes_slide and sl.notes_slide.notes_text_frame.text.strip()]
    assert noted
