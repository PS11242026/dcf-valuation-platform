"""Format-agnostic report model.

A product (valuation / construction / management) assembles a ``ReportModel``; the xlsx / pdf /
pptx renderers consume it. Writing content once here is what lets `export pdf|ppt` reuse workbook
logic instead of re-implementing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Persona = Literal["initiation", "ic", "pm"]
# numeric display hints understood by every renderer
Fmt = Literal[
    "usd", "usd0", "usdM", "usdB",
    "inr", "inr0", "inrM", "inrB",
    "pct", "pct1", "pct2",
    "num", "num0", "x", "yrs", "bp", "text",
]


@dataclass
class Metric:
    """A headline KPI shown in summary blocks / cover."""
    label: str
    value: Any
    fmt: Fmt = "text"
    note: str | None = None
    good: bool | None = None  # True=green, False=red, None=neutral (sign-based if numeric)


@dataclass
class Table:
    title: str | None
    columns: list[str]
    rows: list[list[Any]]
    formats: list[Fmt] | None = None           # per-column display hint
    heatmap_cols: list[int] = field(default_factory=list)  # 0-based cols to conditionally colour
    total_row: bool = False                      # bold the last row
    col_widths: list[float] | None = None        # optional per-column width override (Excel units)


@dataclass
class NativeChart:
    """An editable Excel chart (drawn natively by openpyxl from a Table on the sheet)."""
    title: str
    kind: Literal["line", "bar", "barh", "bubble", "area", "scatter"]
    table: Table                                 # the data backing the chart
    x_col: int = 0
    y_cols: list[int] = field(default_factory=lambda: [1])
    size_col: int | None = None                  # bubble size
    anchor: str | None = None                    # optional cell anchor override
    x_title: str | None = None                   # axis titles (optional; renderers getattr-guard)
    y_title: str | None = None


@dataclass
class Image:
    """A pre-rendered figure (matplotlib PNG: waterfalls, 3D, heatmaps...)."""
    title: str
    path: Path
    width_px: int = 760


@dataclass
class Section:
    """One sheet / slide group."""
    name: str                                    # sheet name (<=31 chars)
    title: str
    metrics: list[Metric] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    charts: list[NativeChart] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    narrative: list[str] = field(default_factory=list)
    prose: bool = False                          # render narrative as flowing essay paragraphs
    #                                              (no bullet markers) instead of bullet points


@dataclass
class ReportModel:
    title: str
    subtitle: str
    persona: Persona
    sections: list[Section] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)   # asof, ticker/book, rating, etc.
    cover_image: Path | None = None   # optional pre-rendered dashboard cover; when set the renderers
    #                                   use it as the cover page instead of the default meta cover.

    def section(self, name: str, title: str | None = None) -> Section:
        s = Section(name=name[:31], title=title or name)
        self.sections.append(s)
        return s
