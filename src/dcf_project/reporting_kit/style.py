"""Shared institutional styling for the workbook renderer (ported from scripts/excel_export.py).

Palette + fonts + number formats + persona accenting. Presentation only.
"""
from __future__ import annotations

from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# ----- palette -----
NAVY = "1F3864"
BLUE = "2E5496"
LIGHT = "D6DCE5"
GREEN = "375623"
RED = "843C0C"
AMBER = "BF8F00"
WHITE = "FFFFFF"
GREY = "808080"

# persona accent (cover/title band colour)
PERSONA_ACCENT = {"initiation": NAVY, "ic": "1F4E2C", "pm": "31356E"}
PERSONA_LABEL = {
    "initiation": "EQUITY RESEARCH  -  INITIATION",
    "ic": "INVESTMENT COMMITTEE  -  PORTFOLIO PROPOSAL",
    "pm": "PORTFOLIO MANAGEMENT  -  OPERATIONAL DASHBOARD",
}

# ----- fonts -----
TITLE_FONT = Font(bold=True, size=16, color=WHITE)
SUBTITLE_FONT = Font(bold=False, size=10, color=WHITE)
SECTION_FONT = Font(bold=True, size=11, color=WHITE)
LABEL_FONT = Font(bold=True, size=10)
VALUE_FONT = Font(size=10)
HEADER_FONT = Font(bold=True, size=9, color=NAVY)
SMALL = Font(size=9)
KPI_FONT = Font(bold=True, size=18, color=NAVY)
KPI_LABEL_FONT = Font(bold=True, size=8, color=GREY)

# ----- fills / borders / alignment -----
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEADER_FILL = PatternFill("solid", fgColor=LIGHT)
KPI_FILL = PatternFill("solid", fgColor="EEF1F7")
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
CENTER = Alignment(horizontal="center", vertical="center")
RIGHT = Alignment(horizontal="right", vertical="center")

# ----- number formats (keyed by model.Fmt) -----
# $-scaling formats take RAW DOLLARS; the trailing commas scale the displayed value
# (usdM ÷1e6, usdB ÷1e9). Do NOT pre-divide values passed with these formats.
NUMFMT = {
    "usd": '$#,##0.00',
    "usd0": '$#,##0',
    "usdM": '$#,##0.0,,"M"',
    "usdB": '$#,##0.00,,,"B"',
    # INR mirrors of the $-scaling formats (country bolt-on). Same scaling rules.
    "inr": '"₹"#,##0.00',
    "inr0": '"₹"#,##0',
    "inrM": '"₹"#,##0.0,,"M"',
    "inrB": '"₹"#,##0.00,,,"B"',
    "pct": '0.0%',
    "pct1": '0.0%',
    "pct2": '0.00%',
    "num": '#,##0.00',
    "num0": '#,##0',
    "x": '0.00"x"',
    "yrs": '0.0" yr"',
    "bp": '#,##0" bp"',
    "text": None,
}


def fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def title_font(color: str = WHITE, size: int = 16) -> Font:
    return Font(bold=True, size=size, color=color)


#: Currency symbol by the 3-char money-format prefix ("usd" -> $, "inr" -> ₹).
_CCY_SYM = {"usd": "$", "inr": "₹"}


def money_fmt(base: str, currency: str | None) -> str:
    """Map a USD money token to its currency-specific variant.

    ``money_fmt("usdM", "INR") -> "inrM"``; any non-INR currency (incl. None) is left as USD,
    so the frozen US path is byte-identical.
    """
    if (currency or "USD").upper() == "INR" and base.startswith("usd"):
        return "inr" + base[3:]
    return base


def fmt_value(value, fmt: str) -> str:
    """Human string for PDF/PPTX renderers (xlsx uses native number formats instead)."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if fmt in ("pct", "pct1"):
        return f"{v * 100:.1f}%"
    if fmt == "pct2":
        return f"{v * 100:.2f}%"
    if fmt in ("usd", "inr"):
        return f"{_CCY_SYM[fmt[:3]]}{v:,.2f}"
    if fmt in ("usd0", "inr0"):
        return f"{_CCY_SYM[fmt[:3]]}{v:,.0f}"
    if fmt in ("usdM", "inrM"):
        return f"{_CCY_SYM[fmt[:3]]}{v / 1e6:,.1f}M"
    if fmt in ("usdB", "inrB"):
        return f"{_CCY_SYM[fmt[:3]]}{v / 1e9:,.2f}B"
    if fmt == "x":
        return f"{v:.2f}x"
    if fmt == "yrs":
        return f"{v:,.1f} yr"
    if fmt == "bp":
        return f"{v:,.0f} bp"
    if fmt == "num0":
        return f"{v:,.0f}"
    if fmt == "num":
        return f"{v:,.2f}"
    return str(value)
