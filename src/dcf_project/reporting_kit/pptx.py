"""Secondary renderer: ReportModel -> PowerPoint deck (python-pptx).

Title + agenda, then per section: a KPI strip, narrative bullets, paginated tables
(10 rows/slide, "(cont.)" overflow), captioned aspect-preserved images on their own
slides, a footer on every slide, and the full section narrative in the speaker notes.
"""
from __future__ import annotations

from pathlib import Path

from .model import ReportModel, Section, Table
from .style import PERSONA_LABEL, fmt_value

_NAVY = (0x1F, 0x38, 0x64)
_GREY = (0x80, 0x80, 0x80)
_HEADER = (0xD6, 0xDC, 0xE5)
_ZEBRA = (0xF4, 0xF6, 0xFA)


def _cell(value, fmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return fmt_value(value, fmt)
    return str(value)


def render(model: ReportModel, out_path: Path) -> Path:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    BLANK = prs.slide_layouts[6]
    navy, grey = RGBColor(*_NAVY), RGBColor(*_GREY)
    footer_label = _footer_text(model)

    def _footer(slide) -> None:
        tf = slide.shapes.add_textbox(Inches(0.3), Inches(7.06), Inches(12.7), Inches(0.32)).text_frame
        tf.text = f"{footer_label}    |    {model.title}"
        p = tf.paragraphs[0]
        p.font.size = Pt(8)
        p.font.color.rgb = grey

    # ---- title slide ----
    s = prs.slides.add_slide(BLANK)
    ci = getattr(model, "cover_image", None)
    if ci and Path(ci).exists():
        # Dashboard cover: the image fills the whole title slide (replaces the default meta cover).
        try:
            s.shapes.add_picture(str(ci), Inches(0.0), Inches(0.0), width=Inches(13.33))
        except Exception:  # noqa: BLE001
            pass
    else:
        box = s.shapes.add_textbox(Inches(0.6), Inches(0.7), Inches(12), Inches(2.6)).text_frame
        box.word_wrap = True
        box.text = PERSONA_LABEL.get(model.persona, "")
        box.paragraphs[0].font.size = Pt(14)
        box.paragraphs[0].font.color.rgb = navy
        p = box.add_paragraph(); p.text = model.title; p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = navy
        p = box.add_paragraph(); p.text = model.subtitle; p.font.size = Pt(16)
        meta = s.shapes.add_textbox(Inches(0.6), Inches(3.4), Inches(12), Inches(3.4)).text_frame
        meta.word_wrap = True
        for i, (k, v) in enumerate((model.meta or {}).items()):
            para = meta.paragraphs[0] if i == 0 else meta.add_paragraph()
            para.text = f"{k}:  {v}"; para.font.size = Pt(12)

    # ---- agenda slide ----
    if model.sections:
        a = prs.slides.add_slide(BLANK)
        _title(a, "Agenda", navy, Inches, Pt)
        body = a.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(12), Inches(5.4)).text_frame
        body.word_wrap = True
        for i, sec in enumerate(model.sections, 1):
            para = body.paragraphs[0] if i == 1 else body.add_paragraph()
            para.text = f"{i}.   {sec.title}"; para.font.size = Pt(16)
        _footer(a)

    # ---- per-section slides ----
    for sec in model.sections:
        slide = prs.slides.add_slide(BLANK)
        _title(slide, sec.title, navy, Inches, Pt)
        top = 1.25
        if sec.metrics:
            _kpi_strip(slide, sec.metrics, top, navy, grey, Inches, Pt)
            top += 1.15
        if sec.narrative:
            tb = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(2.0)).text_frame
            tb.word_wrap = True
            prose = getattr(sec, "prose", False)
            for i, line in enumerate(sec.narrative[:6]):
                para = tb.paragraphs[0] if i == 0 else tb.add_paragraph()
                # prose: flowing essay paragraphs (no bullet markers); else bullet points.
                para.text = str(line) if prose else f"-  {line}"; para.font.size = Pt(11)
            top += 2.1
        # full narrative -> speaker notes
        if sec.narrative:
            slide.notes_slide.notes_text_frame.text = "\n".join(sec.narrative)
        _footer(slide)

        # tables (paginated, 10 rows/slide) -- include chart backing data as a table
        first = True
        for tbl in list(sec.tables) + [ch.table for ch in sec.charts]:
            first = _emit_table(prs, BLANK, sec, tbl, slide if first else None, top,
                                navy, footer_label, model.title, _footer,
                                Inches, Pt, PP_ALIGN, RGBColor)
        # images each on their own captioned slide
        for im in sec.images:
            if not im.path or not Path(im.path).exists():
                continue
            isl = prs.slides.add_slide(BLANK)
            _title(isl, im.title, navy, Inches, Pt)
            try:
                isl.shapes.add_picture(str(im.path), Inches(1.2), Inches(1.3), height=Inches(5.4))
            except Exception:  # noqa: BLE001
                pass
            _footer(isl)

    prs.save(out_path)
    return out_path


def _footer_text(model: ReportModel) -> str:
    meta = model.meta or {}
    head = meta.get("Ticker") or meta.get("Book") or meta.get("Portfolio") or model.title
    asof = meta.get("As of") or meta.get("As-of") or meta.get("Asof") or ""
    return f"{head}  -  {asof}".strip(" -")


def _title(slide, text, navy, Inches, Pt) -> None:
    t = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8)).text_frame
    t.word_wrap = True
    t.text = text
    t.paragraphs[0].font.size = Pt(24)
    t.paragraphs[0].font.bold = True
    t.paragraphs[0].font.color.rgb = navy


def _kpi_strip(slide, metrics, top, navy, grey, Inches, Pt) -> None:
    for i, m in enumerate(metrics[:6]):
        x = Inches(0.5 + i * 2.05)
        tf = slide.shapes.add_textbox(x, Inches(top), Inches(2.0), Inches(1.05)).text_frame
        tf.word_wrap = True
        tf.text = m.label.upper()
        tf.paragraphs[0].font.size = Pt(8)
        tf.paragraphs[0].font.color.rgb = grey
        p = tf.add_paragraph()
        p.text = _cell(m.value, m.fmt)
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = navy


def _emit_table(prs, BLANK, sec: Section, tbl: Table, target, top, navy, footer_label, deck_title,
                footer_fn, Inches, Pt, PP_ALIGN, RGBColor) -> bool:
    """Render a table, paginating 10 rows/slide. Returns False (the in-flow flag is consumed)."""
    cols = tbl.columns
    fmts = tbl.formats or ["text"] * len(cols)
    rows = tbl.rows or []
    page_rows = 10
    chunks = [rows[k:k + page_rows] for k in range(0, max(len(rows), 1), page_rows)] or [[]]
    for ci, chunk in enumerate(chunks):
        if target is not None and ci == 0:
            slide, y = target, top
        else:
            slide = prs.slides.add_slide(BLANK)
            _title(slide, f"{sec.title} (cont.)", navy, Inches, Pt)
            footer_fn(slide)
            y = 1.25
        if tbl.title:
            cap = slide.shapes.add_textbox(Inches(0.5), Inches(y), Inches(12.3), Inches(0.3)).text_frame
            cap.text = tbl.title + ("" if len(chunks) == 1 else f"  (part {ci + 1}/{len(chunks)})")
            cap.paragraphs[0].font.size = Pt(11); cap.paragraphs[0].font.bold = True
            cap.paragraphs[0].font.color.rgb = navy
            y += 0.4
        n_rows = len(chunk) + 1
        height = min(5.4, 0.32 * n_rows)
        gt = slide.shapes.add_table(n_rows, len(cols), Inches(0.5), Inches(y),
                                    Inches(12.3), Inches(height)).table
        total_idx = len(chunk) if (tbl.total_row and ci == len(chunks) - 1 and chunk) else None
        for j, col in enumerate(cols):
            c = gt.cell(0, j); c.text = str(col)
            para = c.text_frame.paragraphs[0]; para.font.size = Pt(9); para.font.bold = True
            para.font.color.rgb = navy
            c.fill.solid(); c.fill.fore_color.rgb = RGBColor(*_HEADER)
        for i, r in enumerate(chunk, start=1):
            for j in range(len(cols)):
                val = r[j] if j < len(r) else None
                c = gt.cell(i, j)
                c.text = _cell(val, fmts[j] if j < len(fmts) else "text")
                para = c.text_frame.paragraphs[0]; para.font.size = Pt(8.5)
                para.font.bold = bool(total_idx is not None and i == total_idx)
                c.fill.solid()
                c.fill.fore_color.rgb = RGBColor(*_ZEBRA) if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
    return False
