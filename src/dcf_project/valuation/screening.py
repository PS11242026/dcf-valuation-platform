"""Basket screening -- canonical DCF across a list of tickers -> ranked workbook."""
from __future__ import annotations

import csv
from pathlib import Path

from dcf_project.paths import VALUATION
from dcf_project.portfolio.scoring import Reference, score_ticker
from dcf_project.reporting_kit import Metric, ReportModel, Table, save_model, xlsx


def run_screen(basket_csv) -> Path:
    path = Path(basket_csv)
    tickers = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("ticker") or row.get("Ticker") or "").strip().upper()
            if t:
                tickers.append(t)
    ref = Reference()
    rows = []
    for t in tickers:
        try:
            sc = score_ticker(t, ref)
            rows.append([t, sc["sector"], sc.get("price"), sc.get("intrinsic_value"),
                         sc.get("discount_to_intrinsic"), (sc.get("conviction") or 0) / 100.0, sc.get("verdict")])
        except Exception as exc:  # noqa: BLE001
            rows.append([t, "ERROR", None, None, None, None, f"{type(exc).__name__}"])
    rows.sort(key=lambda r: (r[4] is not None, r[4] if r[4] is not None else -1e9), reverse=True)

    folder = VALUATION / f"screen_{path.stem}"
    folder.mkdir(parents=True, exist_ok=True)
    m = ReportModel(title=f"Basket screen -- {path.stem}", subtitle=f"{len(tickers)} names ranked by discount to intrinsic",
                    persona="ic", meta={"Basket": path.name, "Names": len(tickers)})
    s = m.section("Screen", "Ranked Screen")
    valid = [r for r in rows if r[4] is not None]
    s.metrics = [Metric("Names", len(tickers), "num0"),
                 Metric("Attractive (disc>15%)", sum(1 for r in valid if r[4] >= 0.15), "num0"),
                 Metric("Median discount", (sorted(r[4] for r in valid)[len(valid) // 2] if valid else None), "pct")]
    s.tables.append(Table("Ranked by discount to intrinsic", ["Ticker", "Sector", "Price", "Intrinsic", "Discount", "Conviction", "Verdict"],
                          rows, formats=["text", "text", "usd", "usd", "pct", "pct", "text"], heatmap_cols=[4]))
    out = xlsx.render(m, folder / "screen.xlsx")
    save_model(m, folder)
    print(f"wrote {out}  ({len(valid)}/{len(tickers)} valued)")
    return out
