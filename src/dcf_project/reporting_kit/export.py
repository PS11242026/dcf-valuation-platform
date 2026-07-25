"""On-demand export dispatcher: re-render a product's saved ReportModel to PDF / PPTX.

Excel is the primary artifact written when the product runs; this re-renders the SAME model so
there is no duplicated workbook logic and no recomputation. `variant="dashboard"` targets the
live-PM-dashboard model slot (used by books whose DEFAULT slot is a pitch package, e.g. the
India book) and suffixes the output `_dashboard` so the two never collide on disk.
"""
from __future__ import annotations

from pathlib import Path

from dcf_project.paths import MANAGED, PORTFOLIO_BUILDS, VALUATION

from .persist import DASHBOARD_MODEL_FILE, MODEL_FILE, load_model

_SEARCH = [VALUATION, MANAGED, PORTFOLIO_BUILDS]


def _resolve(target: str, stem: str) -> Path | None:
    for base in _SEARCH:
        cand = base / target
        if (cand / stem).exists():
            return cand
    return None


def export_target(target: str, fmt: str = "pdf", variant: str | None = None) -> Path:
    stem = DASHBOARD_MODEL_FILE if variant == "dashboard" else MODEL_FILE
    folder = _resolve(target, stem)
    if folder is None:
        searched = ", ".join(str(b / target) for b in _SEARCH)
        hint = " (dashboard slot; run `portfolio manage update` first)" if variant == "dashboard" else ""
        raise SystemExit(
            f"No saved report for '{target}'{hint}. Build it first "
            f"(valuation/construct/manage), then export. Looked in: {searched}"
        )
    model = load_model(folder, stem=stem)
    ext = "pdf" if fmt == "pdf" else "pptx"
    suffix = "_dashboard" if variant == "dashboard" else ""
    # Use only the leaf name so a nested target ("KO/comparison") writes
    # comparison.pdf INSIDE its folder, not a doubled KO/comparison/KO/... path.
    out = folder / f"{Path(target).name}{suffix}.{ext}"
    if fmt == "pdf":
        from .pdf import render
    else:
        from .pptx import render
    render(model, out)
    print(f"wrote {out}")
    return out
