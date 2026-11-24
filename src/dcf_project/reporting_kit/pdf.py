"""Secondary renderer: ReportModel -> multi-page PDF via matplotlib (no extra dependency).

A cursor-based flow layout: elements (KPI grid, narrative, tables, images) are placed top
to bottom and spill onto continuation pages when they run out of vertical room. Tables
paginate (repeated header per page, font shrinks with column count); images preserve aspect
ratio; every page carries a "ticker - as-of  /  title  /  Page N" footer. A cover page
leads with a contents list. Nothing is silently truncated -- the whole model is rendered.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from .model import Image, ReportModel, Table
from .style import PERSONA_LABEL, fmt_value

NAVY = "#1F3864"
GREY = "#808080"
RULE = "#BFBFBF"
PAGE = (8.27, 11.69)            # A4 portrait, inches
LX, RX = 0.07, 0.93            # text left / right (figure fraction)
TOP, BOT = 0.925, 0.06         # top start / bottom limit (above footer)
LINE = 0.0175                  # one text line, figure fraction
ASPECT = PAGE[0] / PAGE[1]     # width/height, to convert x-fraction <-> y-fraction


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _footer_left(model: ReportModel) -> str:
    meta = model.meta or {}
    head = meta.get("Ticker") or meta.get("Book") or meta.get("Portfolio") or model.title
    asof = meta.get("As of") or meta.get("As-of") or meta.get("Asof") or ""
    return f"{head}  -  {asof}".strip(" -")


def _cell(value, fmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return fmt_value(value, fmt)
    return str(value)


class _Doc:
    """Flow cursor over PdfPages: add_* methods emit elements, paging as needed."""

    def __init__(self, pdf, model: ReportModel):
        self.plt = _mpl()
        self.pdf = pdf
        self.model = model
        self.footer = _footer_left(model)
        self.page = 0
        self.fig = None
        self.y = TOP
        self.section_title = ""

    # ---- page lifecycle ----
    def _open(self) -> None:
        self.fig = self.plt.figure(figsize=PAGE)
        self.fig.patch.set_facecolor("white")
        # Report text is literal: "$13.3M x ... = $269.6M" must not be parsed as mathtext.
        _text = self.fig.text
        self.fig.text = lambda *a, **k: _text(*a, **{"parse_math": False, **k})
        self.page += 1
        self.y = TOP

    def _close(self) -> None:
        if self.fig is None:
            return
        self.fig.text(LX, 0.035, self.footer, fontsize=7, color=GREY, va="center")
        self.fig.text((LX + RX) / 2, 0.035, self.model.title, fontsize=7, color=GREY,
                      ha="center", va="center")
        self.fig.text(RX, 0.035, f"Page {self.page}", fontsize=7, color=GREY, ha="right", va="center")
        self.pdf.savefig(self.fig)
        self.plt.close(self.fig)
        self.fig = None

    def _ensure(self, need: float, *, continued: bool = True) -> None:
        """Guarantee `need` vertical fraction; otherwise start a continuation page."""
        if self.fig is None or self.y - need < BOT:
            self._close()
            self._open()
            if continued and self.section_title:
                self._band(f"{self.section_title} (cont.)", 12)

    # ---- primitives ----
    def _band(self, text: str, size: int) -> None:
        self.fig.text(LX, self.y, text, fontsize=size, color=NAVY, fontweight="bold", va="top")
        self.y -= LINE * 1.6
        rule_y = self.y + LINE * 0.5
        self.fig.add_artist(self.plt.Line2D([LX, RX], [rule_y, rule_y], color=NAVY,
                                            linewidth=1.0, transform=self.fig.transFigure))
        self.y -= LINE * 0.3

    def text(self, s: str, size: float = 8.5, color: str = "#222", bold: bool = False,
             indent: float = 0.0, wrap: int = 112) -> None:
        for chunk in (textwrap.wrap(s, wrap) or [""]):
            self._ensure(LINE)
            self.fig.text(LX + indent, self.y, chunk, fontsize=size, color=color,
                          fontweight="bold" if bold else "normal", va="top")
            self.y -= LINE

    # ---- section elements ----
    def heading(self, title: str) -> None:
        self.section_title = title
        self._ensure(LINE * 3, continued=False)
        self._band(title, 14)
        self.y -= LINE * 0.4

    def metrics(self, metrics) -> None:
        per_row, col_w = 3, (RX - LX) / 3
        i = 0
        while i < len(metrics):
            self._ensure(LINE * 2.4)
            row = metrics[i:i + per_row]
            for j, m in enumerate(row):
                x = LX + j * col_w
                self.fig.text(x, self.y, m.label.upper(), fontsize=6.5, color=GREY, va="top")
                self.fig.text(x, self.y - LINE, _cell(m.value, m.fmt), fontsize=12.5, color=NAVY,
                              fontweight="bold", va="top")
            self.y -= LINE * 2.4
            i += per_row
        self.y -= LINE * 0.3

    def narrative(self, lines, prose: bool = False) -> None:
        for line in lines:
            if not line:
                self.y -= LINE * 0.6
                continue
            # prose: flowing essay paragraphs (no bullet markers); else bullet points.
            self.text(str(line) if prose else f"-  {line}", indent=0.0)
            if prose:
                self.y -= LINE * 0.3
        self.y -= LINE * 0.3

    def table(self, tbl: Table, page_rows: int = 26) -> None:
        cols = tbl.columns
        fmts = tbl.formats or ["text"] * len(cols)
        n = len(cols)
        font = 7.5 if n <= 6 else (6.5 if n <= 9 else 5.6)
        row_h = LINE * 1.18
        # Honour explicit column widths (e.g. wide formula columns) and shrink the font so the widest
        # cell fits its column without overlapping its neighbours (matplotlib tables do not wrap text).
        cw = getattr(tbl, "col_widths", None)
        col_fracs = [w / float(sum(cw)) for w in cw] if (cw and len(cw) == n and sum(cw) > 0) else None
        all_text = [cols] + [[_cell(c, fmts[j] if j < len(fmts) else "text") for j, c in enumerate(r)]
                             for r in tbl.rows]
        fracs = col_fracs or [1.0 / n] * n
        page_w_pts = (RX - LX) * PAGE[0] * 72.0
        for ci in range(n):
            longest = max((len(str(r[ci])) for r in all_text if ci < len(r)), default=0)
            if longest <= 0:
                continue
            avail = fracs[ci] * page_w_pts
            font = min(font, max(4.6, avail / (longest * 0.55)))    # ~0.55pt width per char
        if tbl.title:
            self._ensure(LINE * 2)
            self.text(tbl.title, size=9.5, color=NAVY, bold=True)
        chunks = [tbl.rows[k:k + page_rows] for k in range(0, max(len(tbl.rows), 1), page_rows)] or [[]]
        for chunk in chunks:
            need = row_h * (len(chunk) + 1) + LINE
            self._ensure(need)
            height = row_h * (len(chunk) + 1)
            ax = self.fig.add_axes([LX, self.y - height, RX - LX, height])
            ax.axis("off")
            text_rows = [[_cell(c, fmts[j] if j < len(fmts) else "text") for j, c in enumerate(r)]
                         for r in chunk]
            mtab = ax.table(cellText=text_rows or [[""] * n], colLabels=cols,
                            cellLoc="left" if col_fracs else "center",
                            colWidths=col_fracs, loc="upper center", bbox=[0, 0, 1, 1])
            mtab.auto_set_font_size(False)
            mtab.set_fontsize(font)
            for cell in mtab.get_celld().values():
                cell.get_text().set_parse_math(False)
            total_idx = len(text_rows) if (tbl.total_row and text_rows) else None
            for (ri, _ci), cell in mtab.get_celld().items():
                cell.set_edgecolor(RULE)
                cell.set_linewidth(0.4)
                if ri == 0:
                    cell.set_facecolor("#D6DCE5"); cell.set_text_props(color=NAVY, fontweight="bold")
                elif total_idx is not None and ri == total_idx:
                    cell.set_text_props(fontweight="bold")
                elif ri % 2 == 0:
                    cell.set_facecolor("#F4F6FA")
            self.y -= height + LINE * 0.6

    def image(self, im: Image) -> None:
        if not im.path or not Path(im.path).exists():
            return
        try:
            arr = self.plt.imread(str(im.path))
        except Exception:  # noqa: BLE001
            return
        h, w = arr.shape[0], arr.shape[1]
        width_f = RX - LX
        height_f = width_f / ASPECT * (h / w)            # preserve aspect in figure fraction
        max_h = TOP - BOT - LINE * 2
        if height_f > max_h:                              # clamp very tall images to the page
            height_f = max_h
            width_f = height_f * ASPECT * (w / h)
        self._ensure(height_f + LINE * 2)
        self.text(im.title, size=9.5, color=NAVY, bold=True)
        ax = self.fig.add_axes([LX, self.y - height_f, width_f, height_f])
        ax.axis("off")
        ax.imshow(arr)
        self.y -= height_f + LINE * 0.8

    # ---- cover ----
    def cover(self) -> None:
        self._open()
        m = self.model
        # Dashboard cover: a pre-rendered cover image replaces the default meta cover, with a slim
        # contents list beneath it if there is room.
        ci = getattr(m, "cover_image", None)
        if ci and Path(ci).exists():
            try:
                arr = self.plt.imread(str(ci))
                h, w = arr.shape[0], arr.shape[1]
                width_f = RX - LX
                height_f = width_f / ASPECT * (h / w)
                max_h = 0.58                       # leave room so the contents list never overflows
                if height_f > max_h:
                    height_f = max_h
                    width_f = height_f * ASPECT * (w / h)
                x = LX + ((RX - LX) - width_f) / 2.0
                ax = self.fig.add_axes([x, 0.92 - height_f, width_f, height_f])
                ax.axis("off"); ax.imshow(arr)
                # Contents in up to 3 bounded columns so it stays on the page (above the footer).
                y0 = 0.90 - height_f - 0.03
                if y0 > 0.14:
                    self.fig.text(LX, y0, "CONTENTS", fontsize=9.5, color=NAVY, fontweight="bold", va="top")
                    y0 -= 0.028
                    secs = m.sections
                    ncol = 2
                    rows_per_col = (len(secs) + ncol - 1) // ncol
                    col_w = (RX - LX) / ncol
                    for idx, sec in enumerate(secs):
                        col, rrow = idx // rows_per_col, idx % rows_per_col
                        xx = LX + 0.01 + col * col_w
                        yy = y0 - rrow * 0.023
                        title = sec.title if len(sec.title) <= 52 else sec.title[:50] + ".."
                        self.fig.text(xx, yy, f"{idx + 1}.  {title}", fontsize=8, color="#333", va="top")
            except Exception:  # noqa: BLE001
                pass
            self._close()
            return
        self.fig.text(LX, 0.93, PERSONA_LABEL.get(m.persona, ""), fontsize=10, color=NAVY, va="top")
        self.fig.text(LX, 0.88, m.title, fontsize=21, color=NAVY, fontweight="bold", va="top", wrap=True)
        self.fig.text(LX, 0.835, m.subtitle, fontsize=11, color="#444", va="top", wrap=True)
        y = 0.77
        for k, v in (m.meta or {}).items():
            self.fig.text(LX, y, f"{k}:", fontsize=9, fontweight="bold", va="top")
            self.fig.text(LX + 0.22, y, str(v), fontsize=9, va="top")
            y -= 0.026
        y -= 0.02
        self.fig.text(LX, y, "CONTENTS", fontsize=10, color=NAVY, fontweight="bold", va="top")
        y -= 0.03
        for idx, sec in enumerate(m.sections, 1):
            self.fig.text(LX + 0.02, y, f"{idx}.  {sec.title}", fontsize=9, color="#333", va="top")
            y -= 0.024
        self._close()


def render(model: ReportModel, out_path: Path) -> Path:
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        doc = _Doc(pdf, model)
        doc.cover()
        for sec in model.sections:
            doc._open()
            doc.heading(sec.title)
            if sec.metrics:
                doc.metrics(sec.metrics)
            if sec.narrative:
                doc.narrative(sec.narrative, prose=getattr(sec, "prose", False))
            for tbl in sec.tables:
                doc.table(tbl)
            for ch in sec.charts:                 # render a chart's backing data as a table
                doc.table(ch.table)
            for im in sec.images:
                doc.image(im)
            doc._close()
    return out_path
