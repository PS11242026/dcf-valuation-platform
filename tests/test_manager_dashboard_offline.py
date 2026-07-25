"""Offline render of the live PM dashboard from a synthetic book (no network).

``manager._build_dashboard`` is the pure model builder (prices/returns are fetched
upstream in ``run_manage``); here we feed it synthetic holdings / NAV / metrics in a
tmp_path book and assert it produces the workbook + saved model, and that the saved
model re-renders to PDF and PPTX through the rewritten exporters.
"""
from __future__ import annotations

import pytest
from openpyxl import load_workbook

from dcf_project.portfolio.manager import _build_dashboard
from dcf_project.reporting_kit import pdf
from dcf_project.reporting_kit.persist import load_model


def _eq():
    navs = [1.0, 1.012, 1.02, 1.018, 1.035, 1.05]
    return [{"date": f"2026-06-0{i + 1}", "port_nav": n,
             "bench": {"SPY": 1.0 + 0.5 * (n - 1.0), "QQQ": 1.0 + 0.7 * (n - 1.0)}}
            for i, n in enumerate(navs)]


def _build(folder):
    folder.mkdir(parents=True, exist_ok=True)
    state = {"capital": 1_000_000.0, "inception": "2026-06-04", "rebalance_count": 1, "books": {}}
    rows = [
        {"ticker": "KO", "sector": "staples", "entry_price": 29.0, "price": 29.9, "weight": 0.22,
         "market_value": 220000.0, "return": 0.03, "contribution": 0.0066, "current_discount": -0.10,
         "current_intrinsic": 30.0, "conviction": 70.0},
        {"ticker": "PEP", "sector": "staples", "entry_price": 118.5, "price": 119.9, "weight": 0.18,
         "market_value": 180000.0, "return": 0.012, "contribution": 0.0022, "current_discount": -0.20,
         "current_intrinsic": 120.0, "conviction": 62.0},
        {"ticker": "MSFT", "sector": "technology", "entry_price": 390.0, "price": 409.5, "weight": 0.30,
         "market_value": 300000.0, "return": 0.05, "contribution": 0.015, "current_discount": 0.05,
         "current_intrinsic": 410.0, "conviction": 55.0},
    ]
    sectors = {"staples": 0.40, "technology": 0.60}
    book_rets = {"conf_top10": 0.04}
    bench = {"SPY": 0.025, "QQQ": 0.03}
    metrics = {"excess": {"SPY": 0.015, "QQQ": 0.01}, "primary_benchmark": "SPY", "sharpe": 1.0,
               "max_drawdown": -0.05, "turnover": 0.10, "volatility": 0.12, "beta": 0.9, "hhi": 0.22,
               "alpha_basis": "simple_excess_return"}
    return _build_dashboard("conf_top10", state, "conf_top10", rows, sectors, book_rets, bench,
                            metrics, _eq(), 7, "2026-06-11", folder)


def test_dashboard_workbook_and_model(tmp_path):
    folder = tmp_path / "conf_top10"
    out = _build(folder)
    assert out.exists()
    wb = load_workbook(out)
    for sheet in ("Executive Summary", "Current Holdings", "Performance", "Risk Dashboard"):
        assert sheet in wb.sheetnames
    assert (folder / ".report_model.pkl").exists()


def test_dashboard_exports_pdf_and_pptx(tmp_path):
    folder = tmp_path / "conf_top10"
    _build(folder)
    model = load_model(folder)
    p = pdf.render(model, folder / "dash.pdf")
    assert p.exists() and p.read_bytes()[:4] == b"%PDF"
    pptx = pytest.importorskip("pptx")
    from dcf_project.reporting_kit import pptx as pptx_renderer
    deck = pptx_renderer.render(model, folder / "dash.pptx")
    assert deck.exists()
    assert len(pptx.Presentation(str(deck)).slides) >= 6
