"""Primary renderer: ReportModel -> institutional Excel workbook.

Cover sheet (persona-branded, with a hyperlinked table of contents) + one sheet per
section: KPI cards, styled tables (number formats, totals, zebra banding, conditional-
format heatmaps), native editable charts, and embedded matplotlib images. Every section
sheet prints landscape, fit-to-width, with the title row repeated and a
ticker / as-of / page footer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

from . import charts2d
from .model import NativeChart, ReportModel, Table
from .style import (
    BORDER, CENTER, GREEN, GREY, HEADER_FILL, HEADER_FONT, KPI_FILL, KPI_FONT,
    KPI_LABEL_FONT, LABEL_FONT, LEFT, NUMFMT, PERSONA_ACCENT, PERSONA_LABEL, RED, RIGHT,
    SMALL, SUBTITLE_FONT, TITLE_FONT, VALUE_FONT, fill, title_font,
)

ZEBRA_FILL = PatternFill("solid", fgColor="F4F6FA")
LINK_FONT = Font(size=10, color="2E5496", underline="single")
MIN_COL_W, MAX_COL_W = 10.0, 40.0


def _band(ws, row, text, last_col, color, font):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=last_col)
    c = ws.cell(row=row, column=1, value=text)
    c.font = font
    c.fill = fill(color)
    c.alignment = LEFT if font is not TITLE_FONT else CENTER
    return row + 1


def _num(ws, r, c, value, fmt):
    cell = ws.cell(row=r, column=c, value=value)
    if isinstance(value, (int, float)) and NUMFMT.get(fmt):
        cell.number_format = NUMFMT[fmt]
        cell.alignment = RIGHT
    elif isinstance(value, str) and len(value) > 45:
        # long text (e.g. formula cells) wraps within its column instead of being clipped
        cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    else:
        cell.alignment = LEFT
    cell.font = SMALL
    cell.border = BORDER
    return cell


def _kpi_strip(ws, row, metrics, last_col):
    """Lay KPI cards 4 across; each spans 3 columns (label small + big value). A metric
    ``note`` is attached as a cell comment on the value (hover tooltip)."""
    per_row, span = 4, 3
    i = 0
    while i < len(metrics):
        for j in range(per_row):
            if i >= len(metrics):
                break
            m = metrics[i]
            c0 = 1 + j * span
            ws.merge_cells(start_row=row, start_column=c0, end_row=row, end_column=c0 + span - 1)
            ws.merge_cells(start_row=row + 1, start_column=c0, end_row=row + 1, end_column=c0 + span - 1)
            lc = ws.cell(row=row, column=c0, value=m.label.upper())
            lc.font = KPI_LABEL_FONT; lc.alignment = LEFT; lc.fill = KPI_FILL
            vc = ws.cell(row=row + 1, column=c0)
            if isinstance(m.value, (int, float)) and NUMFMT.get(m.fmt):
                vc.value = m.value; vc.number_format = NUMFMT[m.fmt]
            else:
                vc.value = "N/A" if m.value is None else m.value
            color = None
            if m.good is True:
                color = GREEN
            elif m.good is False:
                color = RED
            elif isinstance(m.value, (int, float)) and m.fmt in ("pct", "pct1", "pct2"):
                color = GREEN if m.value >= 0 else RED
            vc.font = title_font(color, 16) if color else KPI_FONT
            vc.alignment = LEFT; vc.fill = KPI_FILL
            if m.note:
                vc.comment = Comment(str(m.note), "DCF")
            i += 1
        row += 2
    return row + 1


def _write_table(ws, row, tbl: Table, last_col, pinned: set) -> tuple[int, dict]:
    if tbl.title:
        row = _band(ws, row, tbl.title, last_col, "EEF1F7", LABEL_FONT)
    header_row = row
    for j, h in enumerate(tbl.columns, start=1):
        c = ws.cell(row=row, column=j, value=h)
        c.font = HEADER_FONT; c.fill = HEADER_FILL; c.alignment = CENTER; c.border = BORDER
    row += 1
    first_data = row
    fmts = tbl.formats or ["text"] * len(tbl.columns)
    for ri, r in enumerate(tbl.rows):
        is_total = tbl.total_row and ri == len(tbl.rows) - 1
        zebra = (ri % 2 == 1) and not is_total
        for j, val in enumerate(r, start=1):
            cell = _num(ws, row, j, val, fmts[j - 1] if j - 1 < len(fmts) else "text")
            if is_total:
                cell.font = LABEL_FONT
            elif zebra:
                cell.fill = ZEBRA_FILL
        row += 1
    last_data = row - 1
    # conditional-format heatmap columns
    for col0 in tbl.heatmap_cols:
        col = col0 + 1
        if last_data >= first_data:
            rng = f"{get_column_letter(col)}{first_data}:{get_column_letter(col)}{last_data}"
            charts2d.heatmap(ws, rng)
    # optional explicit column widths (pin them so autosize leaves them alone)
    for j, w in enumerate(tbl.col_widths or [], start=1):
        letter = get_column_letter(j)
        ws.column_dimensions[letter].width = float(w)
        pinned.add(letter)
    meta = {"header_row": header_row, "first_data": first_data, "last_data": last_data}
    return row + 1, meta


def _add_native_chart(ws, ch: NativeChart, meta, anchor):
    fd, ld = meta["first_data"], meta["last_data"]
    if ld < fd:
        return
    x_title = getattr(ch, "x_title", None)
    y_title = getattr(ch, "y_title", None)
    if ch.kind in ("line", "area"):
        charts2d.line_chart(ws, anchor, ch.title, ch.y_cols[0] + 1, ch.y_cols[-1] + 1, fd, ld,
                            ch.x_col + 1, x_title=x_title, y_title=y_title)
    elif ch.kind in ("bar", "barh"):
        charts2d.bar_chart(ws, anchor, ch.title, ch.y_cols[0] + 1, ch.y_cols[-1] + 1, fd, ld,
                           ch.x_col + 1, horizontal=(ch.kind == "barh"),
                           x_title=x_title, y_title=y_title)
    elif ch.kind == "bubble" and ch.size_col is not None:
        charts2d.bubble_chart(ws, anchor, ch.title, ch.x_col + 1, ch.y_cols[0] + 1, ch.size_col + 1, fd, ld)


def _embed_images(ws, row, images, last_col):
    from openpyxl.drawing.image import Image as XLImage
    r = row
    for im in images:
        if not im.path or not Path(im.path).exists():
            continue
        ws.cell(row=r, column=1, value=im.title).font = LABEL_FONT
        r += 1
        try:
            img = XLImage(str(im.path))
            scale = im.width_px / float(img.width or im.width_px)
            img.width = im.width_px
            img.height = int((img.height or 360) * scale)
            ws.add_image(img, f"A{r}")
            r += max(14, int((img.height) / 18) + 2)
        except Exception as exc:  # noqa: BLE001
            ws.cell(row=r, column=1, value=f"[image unavailable: {exc}]").font = SMALL
            r += 1
    return r


def _autosize(ws, pinned: set) -> None:
    """Content-aware column widths (clamped), skipping any columns pinned by col_widths."""
    widths: dict[str, int] = {}
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            col = cell.column_letter
            if col in pinned:
                continue
            widths[col] = max(widths.get(col, 0), len(str(cell.value)))
    for col, ln in widths.items():
        ws.column_dimensions[col].width = max(MIN_COL_W, min(MAX_COL_W, ln + 2))


def _footer_label(model: ReportModel) -> str:
    meta = model.meta or {}
    head = meta.get("Ticker") or meta.get("Book") or meta.get("Portfolio") or model.title
    asof = meta.get("As of") or meta.get("As-of") or meta.get("Asof") or ""
    return f"{head}  ·  {asof}".strip(" ·")


def _page_setup(ws, footer_label: str) -> None:
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.print_title_rows = "1:1"  # repeat the section title band on every printed page
    ws.oddFooter.left.text = footer_label
    ws.oddFooter.right.text = "Page &P of &N"


def _cover(ws, model: ReportModel):
    accent = PERSONA_ACCENT.get(model.persona, "1F3864")
    # Dashboard cover: when the model carries a pre-rendered cover image, the Cover sheet IS that
    # dashboard (the default meta cover is replaced), followed by a slim hyperlinked contents list.
    ci = getattr(model, "cover_image", None)
    if ci and Path(ci).exists():
        from openpyxl.drawing.image import Image as XLImage
        for col, w in zip("ABCDEFGHIJKL", (16,) * 12):
            ws.column_dimensions[col].width = w
        start = 4
        try:
            img = XLImage(str(ci))
            target_w = 1500           # high-resolution cover
            scale = target_w / float(img.width or target_w)
            img.width = target_w
            img.height = int((img.height or 660) * scale)
            ws.add_image(img, "A1")
            start = max(46, int(img.height / 18) + 3)
        except Exception:  # noqa: BLE001
            pass
        row = start
        ws.cell(row=row, column=1, value="CONTENTS").font = LABEL_FONT
        row += 1
        for sec in model.sections:
            cell = ws.cell(row=row, column=1, value=sec.title)
            cell.hyperlink = f"#'{sec.name}'!A1"
            cell.font = LINK_FONT
            row += 1
        ws.sheet_view.showGridLines = False
        return
    for col, w in zip("ABCDEFGHIJKL", (22, 16, 16, 16, 14, 14, 14, 14, 14, 14, 14, 14)):
        ws.column_dimensions[col].width = w
    ws.row_dimensions[1].height = 30
    _band(ws, 1, PERSONA_LABEL.get(model.persona, ""), 12, accent, SUBTITLE_FONT)
    ws.row_dimensions[2].height = 34
    _band(ws, 2, model.title, 12, accent, TITLE_FONT)
    _band(ws, 3, model.subtitle, 12, accent, SUBTITLE_FONT)
    row = 5
    for k, v in model.meta.items():
        lc = ws.cell(row=row, column=1, value=str(k)); lc.font = LABEL_FONT
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
        ws.cell(row=row, column=2, value=str(v)).font = VALUE_FONT
        row += 1
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    engine = (model.meta or {}).get("Engine") or (model.meta or {}).get("engine")
    gen_line = f"Generated {gen}" + (f"  ·  engine {engine}" if engine else "")
    ws.cell(row=row, column=1, value=gen_line).font = Font(size=8, italic=True, color=GREY)
    row += 2
    # Table of contents with internal hyperlinks to each section sheet.
    ws.cell(row=row, column=1, value="CONTENTS").font = LABEL_FONT
    row += 1
    for sec in model.sections:
        cell = ws.cell(row=row, column=1, value=sec.title)
        cell.hyperlink = f"#'{sec.name}'!A1"
        cell.font = LINK_FONT
        row += 1
    ws.freeze_panes = "A4"


def render(model: ReportModel, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    cover = wb.active
    cover.title = "Cover"
    _cover(cover, model)
    footer_label = _footer_label(model)
    accent = PERSONA_ACCENT.get(model.persona, "1F3864")

    for sec in model.sections:
        ws = wb.create_sheet(sec.name)
        ws.sheet_properties.tabColor = accent
        pinned: set = set()
        last_col = 12
        ws.row_dimensions[1].height = 24
        row = _band(ws, 1, sec.title, last_col, accent, TITLE_FONT)
        row += 1
        if sec.metrics:
            row = _kpi_strip(ws, row, sec.metrics, last_col)
        for line in sec.narrative:
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=last_col)
            # prose sections read as flowing essay paragraphs (no bullet markers); the row
            # is sized to the text length so a full paragraph stays readable in one cell.
            text = str(line) if sec.prose else f"- {line}"
            c = ws.cell(row=row, column=1, value=text); c.font = VALUE_FONT; c.alignment = LEFT
            ws.row_dimensions[row].height = max(26, 15 * (len(text) // (last_col * 11) + 1)) if sec.prose else 26
            row += 1
        if sec.narrative:
            row += 1
        for tbl in sec.tables:
            row, _ = _write_table(ws, row, tbl, last_col, pinned)
        for ch in sec.charts:
            row, meta = _write_table(ws, row, ch.table, last_col, pinned)
            _add_native_chart(ws, ch, meta, f"A{row}")
            row += 16
        if sec.images:
            row = _embed_images(ws, row, sec.images, last_col)
        _autosize(ws, pinned)
        _page_setup(ws, footer_label)
        ws.freeze_panes = "A2"

    wb.save(out_path)
    return out_path
