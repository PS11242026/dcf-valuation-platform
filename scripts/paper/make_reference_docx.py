"""Generate a professional pandoc reference document -> scripts/paper/reference.docx.

pandoc's default DOCX output is plain (Calibri, ragged, no title styling). This script takes
pandoc's own default reference doc (so every style pandoc expects exists) and restyles it to a
clean journal look -- serif body, justified text, navy headings, Letter page with 1" margins,
and a centred page-number footer -- which ``build_paper_v3.py`` then passes via ``--reference-doc``.

Run once (or whenever the look should change):
    python -m scripts.paper.make_reference_docx
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
OUT = HERE / "reference.docx"
NAVY = RGBColor(0x1F, 0x38, 0x64)
GREY = RGBColor(0x55, 0x55, 0x55)
BODY_FONT = "Cambria"


def _set_font(style, *, name=None, size=None, bold=None, italic=None, color=None):
    f = style.font
    if name:
        f.name = name
        # also set the east-asian / h-ansi slots so Word honours it everywhere
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts")) or OxmlElement("w:rFonts")
        for slot in ("w:ascii", "w:hAnsi", "w:cs"):
            rfonts.set(qn(slot), name)
        if rfonts.getparent() is None:
            rpr.append(rfonts)
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = color


def _add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    run._r.append(begin); run._r.append(instr); run._r.append(end)


def main() -> int:
    default = HERE / "_default_reference.docx"
    with open(default, "wb") as fh:
        subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                       check=True, stdout=fh)
    doc = Document(str(default))
    styles = doc.styles

    def style(name):
        try:
            return styles[name]
        except KeyError:
            return None

    # Body text: serif, justified, comfortable spacing.
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Footnote Text"):
        st = style(name)
        if st is None:
            continue
        _set_font(st, name=BODY_FONT, size=11)
        pf = st.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.line_spacing = 1.15
        pf.space_after = Pt(6)

    # Title block.
    if style("Title"):
        _set_font(styles["Title"], name=BODY_FONT, size=24, bold=True, color=NAVY)
        styles["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        styles["Title"].paragraph_format.space_after = Pt(4)
    for sub in ("Subtitle",):
        if style(sub):
            _set_font(styles[sub], name=BODY_FONT, size=13.5, italic=True, color=GREY)
            styles[sub].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            styles[sub].paragraph_format.space_after = Pt(10)
    for meta in ("Author", "Date"):
        if style(meta):
            _set_font(styles[meta], name=BODY_FONT, size=12)
            styles[meta].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Headings.
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
        st = style(name)
        if st is None:
            continue
        _set_font(st, name=BODY_FONT, size=size, bold=True,
                  italic=(name == "Heading 3"), color=NAVY)
        st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.keep_with_next = True

    # Block quote (pandoc uses "Block Text" / "Quote") -> indented grey italic.
    for q in ("Block Text", "Quote", "Intense Quote"):
        if style(q):
            _set_font(styles[q], name=BODY_FONT, size=10.5, italic=True, color=GREY)
            styles[q].paragraph_format.left_indent = Inches(0.3)

    # Captions (figure/table captions land in Image Caption / Caption).
    for cap in ("Caption", "Image Caption", "Table Caption"):
        if style(cap):
            _set_font(styles[cap], name=BODY_FONT, size=9.5, italic=True, color=GREY)

    # Page geometry + footer page numbers.
    for sec in doc.sections:
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11)
        sec.top_margin = sec.bottom_margin = Inches(1.0)
        sec.left_margin = sec.right_margin = Inches(1.0)
        footer = sec.footer
        footer.is_linked_to_previous = False
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        for r in list(p.runs):
            r.text = ""
        _add_page_number(p)

    doc.save(str(OUT))
    default.unlink(missing_ok=True)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
