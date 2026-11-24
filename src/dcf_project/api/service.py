"""On-demand single-ticker analysis: ticker -> canonical Kalman/state-space DCF -> web report.

This is the SAME path as ``python -m dcf_project valuation canonical <TICKER>`` (score_ticker ->
build_canonical_model -> xlsx/pdf renderers); the API only adds a JSON serialization of the
format-agnostic ReportModel so a browser can render every section, plus the downloadable
workbook and PDF. No valuation math lives here.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,9}([.\-][A-Z0-9]{1,3})?$")


class AnalysisError(Exception):
    """A user-facing failure (bad ticker, no SEC filings, model cannot value the company)."""


def normalize_ticker(raw: str) -> str:
    t = (raw or "").strip().upper()
    if not TICKER_RE.match(t):
        raise AnalysisError(f"'{raw}' is not a valid ticker symbol.")
    return t


def _clean(x: Any) -> Any:
    """JSON-safe: numpy scalars -> python, NaN/inf -> None, Paths -> str, recurse containers."""
    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "item"):  # numpy scalar
        try:
            return _clean(x.item())
        except (ValueError, TypeError):
            pass
    if isinstance(x, (dt.date, dt.datetime)):
        return x.isoformat()
    return str(x)


def _img_data_uri(p: Path) -> str | None:
    try:
        return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode("ascii")
    except OSError:
        return None


def model_to_json(model) -> dict:
    """ReportModel -> plain dict (images inlined as data URIs) for the web renderer."""
    out = {"title": model.title, "subtitle": model.subtitle, "meta": _clean(model.meta), "sections": []}
    for s in model.sections:
        out["sections"].append({
            "name": s.name, "title": s.title,
            "metrics": [_clean({"label": m.label, "value": m.value, "fmt": m.fmt, "note": m.note,
                                "good": m.good}) for m in s.metrics],
            "tables": [_clean({"title": t.title, "columns": t.columns, "rows": t.rows,
                               "formats": t.formats, "heatmap_cols": t.heatmap_cols,
                               "total_row": t.total_row}) for t in s.tables],
            "charts": [_clean({"title": c.title, "kind": c.kind, "x_col": c.x_col, "y_cols": c.y_cols,
                               "columns": c.table.columns, "rows": c.table.rows,
                               "formats": c.table.formats}) for c in s.charts],
            "images": [{"title": i.title, "src": _img_data_uri(i.path)} for i in s.images
                       if Path(i.path).exists()],
            "narrative": [str(n) for n in s.narrative],
        })
    return out


def _series(rows: list[dict], key: str) -> list:
    return [_clean(r.get(key)) for r in rows]


def analyze(ticker: str, out_root: Path | None = None) -> dict:
    """Run the canonical valuation for one ticker and return the web report dict.

    Side effects: writes canonical_dcf.xlsx and <TICKER>.pdf under ``out_root/<TICKER>/`` so
    the API can serve them as downloads."""
    import matplotlib

    matplotlib.use("Agg")
    from dcf_project.paths import VALUATION
    from dcf_project.reporting_kit import pdf, xlsx
    from dcf_project.valuation.canonical import canonical_valuation
    from dcf_project.valuation.report import build_canonical_model, company_display_name

    t = normalize_ticker(ticker)
    folder = Path(out_root or VALUATION) / t
    folder.mkdir(parents=True, exist_ok=True)
    try:
        sc = canonical_valuation(t)
    except Exception as e:
        msg = str(e).strip() or e.__class__.__name__
        if "No SEC CIK" in msg:
            raise AnalysisError(f"{t} wasn't found among SEC-registered US companies. Check the symbol "
                                "(the engine values US filers from their EDGAR financials).") from e
        raise AnalysisError(f"The DCF engine could not value {t}: {msg[:300]}") from e
    res = sc["result"]
    if not res.intrinsic_value_per_share or not math.isfinite(float(res.intrinsic_value_per_share)):
        raise AnalysisError(f"The DCF engine returned no intrinsic value for {t}.")
    name = company_display_name(t)
    sc["company_name"] = name
    model = build_canonical_model(sc, folder)
    files = {}
    try:
        xlsx.render(model, folder / "canonical_dcf.xlsx")
        files["xlsx"] = "canonical_dcf.xlsx"
    except Exception:
        pass
    try:
        pdf.render(model, folder / f"{t}.pdf")
        files["pdf"] = f"{t}.pdf"
    except Exception:
        pass

    ff = [dict(r) for r in (res.forecast_financials or [])]
    hist = list((res.diagnostics or {}).get("historical_financials", []) or [])
    period_key = next((k for k in ("period", "date", "quarter", "fiscal_period") if ff and k in ff[0]), None)
    hist_key = next((k for k in ("period", "date", "end", "quarter") if hist and k in hist[0]), None)
    report = {
        "ticker": t, "company": name,
        "as_of": sc["timestamp_utc"],
        "price": sc.get("price"), "intrinsic_value": sc.get("intrinsic_value"),
        "discount": sc.get("discount_to_intrinsic"),
        "verdict": sc.get("verdict"), "rank_basis": sc.get("rank_basis"),
        "confidence": sc.get("prediction_confidence"),
        "forecast_confidence": sc.get("forecast_confidence"),
        "range_p10_p90": sc.get("intrinsic_range_p10_p90"),
        "p50": _clean(getattr(res, "valuation_p50", None)),
        "regime": sc.get("regime"), "sector": sc.get("sector"),
        "market_cap": sc.get("market_cap"),
        "assumptions": sc.get("assumptions"),
        "why": sc.get("why"), "qualifiers": sc.get("qualifiers"),
        "engine": getattr(res, "engine_version", "v1"),
        "forecast": {"period": _series(ff, period_key) if period_key else list(range(1, len(ff) + 1)),
                     "revenue": _series(ff, "revenue"), "fcf": _series(ff, "fcf"),
                     "ebit": _series(ff, "ebit")},
        "history": {"period": _series(hist, hist_key) if hist_key else [],
                    "revenue": _series(hist, "revenue"), "fcf": _series(hist, "fcf")},
        "model": model_to_json(model),
        "files": files,
    }
    report = _clean(report)
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return report
