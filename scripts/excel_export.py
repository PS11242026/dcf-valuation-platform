"""Excel export for the valuation scorecard (PRESENTATION ONLY).

Completely separate from valuation/forecasting/ranking/validation logic: this module reads
ONLY the scorecard object produced by scripts.scorecard.score_ticker (which already carries the
reused DCFResult fields under sc["model"]) and writes reports/{ticker}_valuation_report.xlsx.
It performs NO valuation calculations -- the only arithmetic is a presentation-layer WACC x
terminal-growth sensitivity grid built from already-computed projected FCF + terminal base, so
the visualization can "react" to those drivers.

Two sheets:
  1. FCF Valuation Model   -- full, formatted financial model + thesis + assumptions + limitations.
  2. Valuation Visualization -- historical price vs model-implied forward value + a WACC/terminal-
     growth sensitivity grid (basic chart for now; 3D surface is a planned iteration).

If anything here raises, the caller catches it so the terminal scorecard still renders.
"""

from __future__ import annotations

import functools
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


@functools.lru_cache(maxsize=256)
def _legal_name(ticker: str, fallback: str = "") -> str:
    """Best-effort full legal/company name via yfinance (cached). Falls back to the ticker.
    Presentation only -- no valuation use."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or (fallback or ticker)
    except Exception:
        return fallback or ticker

# ----------------------------- styling -----------------------------
NAVY = "1F3864"
BLUE = "2E5496"
LIGHT = "D6DCE5"
GREEN = "375623"
RED = "843C0C"
WHITE = "FFFFFF"

TITLE_FONT = Font(bold=True, size=16, color=WHITE)
SECTION_FONT = Font(bold=True, size=11, color=WHITE)
LABEL_FONT = Font(bold=True, size=10)
VALUE_FONT = Font(size=10)
HEADER_FONT = Font(bold=True, size=9, color=NAVY)
SMALL = Font(size=9)

TITLE_FILL = PatternFill("solid", fgColor=NAVY)
SECTION_FILL = PatternFill("solid", fgColor=BLUE)
HEADER_FILL = PatternFill("solid", fgColor=LIGHT)

THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
CENTER = Alignment(horizontal="center", vertical="center")
RIGHT = Alignment(horizontal="right", vertical="center")

CUR = '#,##0.00'          # per-share dollars
CUR_M = '#,##0.0'         # $ millions
PCT0 = '0%'
PCT1 = '0.0%'
NUM = '#,##0'


def _title(ws, text, last_col=8, row=1):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=last_col)
    c = ws.cell(row=row, column=1, value=text)
    c.font = TITLE_FONT; c.fill = TITLE_FILL; c.alignment = CENTER
    ws.row_dimensions[row].height = 26
    return row + 1


def _section(ws, text, row, last_col=8):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=last_col)
    c = ws.cell(row=row, column=1, value=text)
    c.font = SECTION_FONT; c.fill = SECTION_FILL; c.alignment = LEFT
    ws.row_dimensions[row].height = 18
    return row + 1


def _kv(ws, row, label, value, value_fmt=None, value_col=3):
    lc = ws.cell(row=row, column=1, value=label); lc.font = LABEL_FONT; lc.alignment = LEFT
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=value_col - 1)
    vc = ws.cell(row=row, column=value_col, value=value); vc.font = VALUE_FONT; vc.alignment = LEFT
    if value_fmt and isinstance(value, (int, float)):
        vc.number_format = value_fmt
    return row + 1


def _table(ws, row, headers, rows, fmts):
    for j, h in enumerate(headers, start=1):
        c = ws.cell(row=row, column=j, value=h)
        c.font = HEADER_FONT; c.fill = HEADER_FILL; c.alignment = CENTER; c.border = BORDER
    row += 1
    for r in rows:
        for j, (val, fmt) in enumerate(zip(r, fmts), start=1):
            c = ws.cell(row=row, column=j, value=val)
            c.font = SMALL; c.border = BORDER
            c.alignment = RIGHT if isinstance(val, (int, float)) else LEFT
            if fmt and isinstance(val, (int, float)):
                c.number_format = fmt
        row += 1
    return row + 1


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _thesis_essay_text(reasons: list[str]) -> list[str]:
    """Join the why-reasons into flowing essay paragraph(s) instead of bullet points."""
    sentences = []
    for reason in reasons:
        s = str(reason).strip()
        if s and s[-1] not in ".!?:":
            s += "."
        if s:
            sentences.append(s)
    return [" ".join(sentences)] if sentences else ["No thesis narrative is available for this name."]


def _terminal_value_section(ws, r, sc):
    """TERMINAL VALUE ASSUMPTIONS: terminal growth rate, the formula, and the assumption values only."""
    steps = sc.get("model", {}).get("dcf_steps", {})
    prov = sc.get("model", {}).get("terminal_growth_provenance") or {}
    eng = prov.get("fcf_growth_projection_engine") or {}
    guard = eng.get("guardrails") or {}
    ff = sc.get("model", {}).get("forecast_financials") or []
    last = ff[-1] if ff else {}
    rate = steps.get("terminal_growth_rate")
    wacc = _num(steps.get("wacc"))
    tb_annual = _num(steps.get("terminal_base_annual"))

    r = _section(ws, "TERMINAL VALUE ASSUMPTIONS", r)
    r = _kv(ws, r, "Terminal Growth Rate", rate, PCT1)

    # formula block
    formulas = [
        "Formula:  g_terminal = min( g_WACC-limit , max( g_floor , Blend(g_FCF, g_forecast, g_ROIC, g_policy) ) )",
        "          g_ROIC = ROIC_terminal x ReinvestmentRate_terminal",
        "          g_FCF  = CAGR(FCF_path)",
        "          Terminal Value:   TV = FCF_(N+1) / (WACC - g_terminal)",
    ]
    for line in formulas:
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
        c = ws.cell(row=r, column=1, value=line); c.font = Font(name="Consolas", size=9); c.alignment = LEFT
        r += 1

    # assumptions
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
    ws.cell(row=r, column=1, value="Assumptions:").font = Font(italic=True, size=9, color="808080"); r += 1
    r = _kv(ws, r, "   g_FCF = CAGR(FCF_path)", _num(eng.get("path_cagr")), PCT1)
    r = _kv(ws, r, "   g_forecast (median annual growth)", _num(eng.get("median_annual_growth")), PCT1)
    r = _kv(ws, r, "   g_ROIC = ROIC_terminal x ReinvestmentRate_terminal", _num(eng.get("roic_reinvestment_terminal_growth")), PCT1)
    r = _kv(ws, r, "      ROIC_terminal (annualized)", _num(last.get("annualized_roic")), PCT1)
    r = _kv(ws, r, "      ReinvestmentRate_terminal", _num(last.get("reinvestment_rate")), PCT1)
    r = _kv(ws, r, "   g_policy (company-type)", _num(eng.get("policy_terminal_growth")), PCT1)
    r = _kv(ws, r, "   g_floor", _num(guard.get("terminal_floor")), PCT1)
    r = _kv(ws, r, "   g_WACC-limit (max allowed by WACC)", _num(prov.get("max_allowed_by_wacc")), PCT1)
    r = _kv(ws, r, "   WACC", wacc, PCT1)
    r = _kv(ws, r, "   FCF_(N+1) (terminal FCF, annual, $M)", _scale(tb_annual), CUR_M)
    r += 1
    return r


# ----------------------------- Sheet 1 -----------------------------
def _build_model_sheet(ws, sc: dict) -> None:
    m = sc.get("model", {})
    steps = m.get("dcf_steps", {})
    a = sc.get("assumptions", {})
    d = sc.get("discount_to_intrinsic")
    anchors: dict = {"sheet": ws.title}   # sheet-1 cell rows (col C) that sheet 2 links back to
    for col, w in zip("ABCDEFGHIJKL", (22, 13, 13, 13, 13, 13, 13, 13, 16, 13, 13, 13)):
        ws.column_dimensions[col].width = w

    r = _title(ws, f"{_legal_name(sc.get('ticker'), sc.get('company_name'))}  -  DCF PROJECTION")
    r += 0

    # ---- summary block ----
    r = _section(ws, "VALUATION SUMMARY", r)
    mc = sc.get("market_cap")
    r = _kv(ws, r, "Ticker", sc.get("ticker"))
    r = _kv(ws, r, "Company Name", _legal_name(sc.get("ticker"), sc.get("company_name")))
    r = _kv(ws, r, "Company Type", _company_type_label(sc))
    r = _kv(ws, r, "Company Stage", _company_stage_label(sc))
    r = _kv(ws, r, "Market Capitalization", (mc / 1e6) if mc else None, CUR_M)
    ws.cell(row=r - 1, column=4, value="$ millions").font = SMALL
    anchors["price"] = r; r = _kv(ws, r, "Current Price", sc.get("price"), CUR)
    anchors["intrinsic"] = r; r = _kv(ws, r, "Intrinsic Value (per share)", sc.get("intrinsic_value"), CUR)
    direction = "Upside" if (d is not None and d >= 0) else "Downside"
    anchors["downside"] = r
    rr = _kv(ws, r, f"{direction} %", d, PCT0)
    if d is not None:
        ws.cell(row=r, column=3).font = Font(size=10, bold=True, color=GREEN if d >= 0 else RED)
    r = rr
    r = _kv(ws, r, "Investment Suggestion (Verdict)", sc.get("verdict"))
    r = _kv(ws, r, "Prediction Confidence", _confidence_band(sc))
    anchors["timestamp"] = r; r = _kv(ws, r, "Timestamp Generated", sc.get("timestamp_utc"))
    r += 1

    # ---- investment thesis: flowing essay, not bullets (drop 'forecast inputs'; 'ranked'->'priced') ----
    r = _section(ws, "INVESTMENT THESIS", r)
    reasons = [reason.replace("ranked", "priced") for reason in (sc.get("why") or [])
               if "forecast input" not in reason.lower()]
    essay = _thesis_essay_text(reasons)
    for para in essay:
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
        c = ws.cell(row=r, column=1, value=para); c.font = VALUE_FONT; c.alignment = LEFT
        ws.row_dimensions[r].height = max(28, 15 * (len(para) // 90 + 1))
        r += 1
    r += 1

    # ---- key assumptions (WACC labelled with its provider) ----
    wacc_provider = m.get("wacc_provider") or "internal CAPM"
    r = _section(ws, "KEY ASSUMPTIONS", r)
    r = _kv(ws, r, "Revenue Growth (per year)", a.get("implied_annual_revenue_growth"), PCT1)
    r = _kv(ws, r, "Terminal Operating Margin", a.get("terminal_operating_margin"), PCT1)
    anchors["termg"] = r; r = _kv(ws, r, "Terminal Growth", a.get("terminal_growth"), PCT1)
    anchors["wacc"] = r; r = _kv(ws, r, f"WACC ({wacc_provider})", a.get("wacc"), PCT1)
    r += 1

    # ---- terminal value assumptions (source + derivation) ----
    r = _terminal_value_section(ws, r, sc)

    # ---- DCF calculation summary ----
    r = _section(ws, "DCF CALCULATION STEPS ($ millions unless per-share)", r)
    wacc = _num(steps.get("wacc"))
    sched = m.get("discount_schedule", [])
    q_rate = _num(sched[0].get("discount_rate_per_quarter")) if sched else (wacc / 4 if wacc else None)
    tv_factor = _num(sched[-1].get("discount_factor")) if sched else None  # (1+wacc/4)^N applied to TV
    n_q = len(sched) or 20
    dcf_rows = [
        ("Annual WACC discount rate", wacc, PCT1),
        ("Quarterly discount rate (FCF,TV)", q_rate, '0.00%'),
        (f"TV discount factor((1+WACC/4)^{n_q})", tv_factor, '0.0000'),
        ("Base Terminal FCF", _scale(steps.get("terminal_base_annual")), CUR_M),
        ("TV(undiscounted)", _scale(steps.get("terminal_value")), CUR_M),
        ("PV(TV)", _scale(steps.get("pv_terminal_value")), CUR_M),
        ("PV(FCF)", _scale(steps.get("pv_projected_fcf")), CUR_M),
        ("EV=PV(TV)+PV(FCF)", _scale(steps.get("enterprise_value")), CUR_M),
        ("(-)Net Debt", _scale(steps.get("net_debt")), CUR_M),
        ("Equity Value", _scale(steps.get("equity_value")), CUR_M),
        ("outstanding shares(millions)", _scale(steps.get("shares_outstanding")), CUR_M),
        ("intrinsic value (per share)", steps.get("intrinsic_value_per_share"), CUR),
    ]
    for label, val, fmt in dcf_rows:
        r = _kv(ws, r, label, val, fmt)
    r += 1

    def _hdr_row(row, headers):
        for j, h in enumerate(headers, start=1):
            c = ws.cell(row=row, column=j, value=h)
            c.font = HEADER_FONT; c.fill = HEADER_FILL; c.border = BORDER
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[row].height = 40

    def _data_cell(row, col, val, fmt):
        c = ws.cell(row=row, column=col); c.font = SMALL; c.border = BORDER
        if val is not None:
            c.value = val
            c.alignment = RIGHT if isinstance(val, (int, float)) else LEFT
            if fmt and isinstance(val, (int, float)):
                c.number_format = fmt
        return c

    def _formula_cell(row, col, formula, fmt=CUR_M):
        c = ws.cell(row=row, column=col, value=formula)
        c.number_format = fmt; c.font = SMALL; c.border = BORDER; c.alignment = RIGHT
        return c

    # ---- forecast financials: FCF is a LIVE formula linking the ensemble path cells ----
    r = _section(ws, "FORECAST FINANCIALS ($ millions)", r)
    ff = m.get("forecast_financials", [])
    ds = {int(round(_num(x.get("quarter")) or i + 1)): x for i, x in enumerate(m.get("discount_schedule", []))}
    w = m.get("ensemble_weights") or {"owner": 0.4, "accounting": 0.35, "economic": 0.25}
    wo, wa, we = _num(w.get("owner")) or 0.0, _num(w.get("accounting")) or 0.0, _num(w.get("economic")) or 0.0
    E, F, G, H, I = (get_column_letter(k) for k in (5, 6, 7, 8, 9))
    _hdr_row(r, ["Quarter", "Revenue", "Op Margin", "EBIT", "Owner FCF", "Accounting FCF",
                 "Economic FCF", "FCF", "Discount Factor", "PV of FCF"])
    f_fmts = [None, CUR_M, PCT0, CUR_M, CUR_M, CUR_M, CUR_M, None, '0.000', None]
    r += 1
    for i, row in enumerate(ff, start=1):
        sch = ds.get(i, {})
        rev = _num(row.get("revenue")); ebit = _num(row.get("ebit"))
        statics = [i, _scale(rev), (ebit / rev) if (rev and ebit is not None) else None, _scale(ebit),
                   _scale(row.get("owner_earnings")), _scale(row.get("accounting_fcf")),
                   _scale(row.get("economic_fcf")), None, _num(sch.get("discount_factor")), None]
        for j, (val, fmt) in enumerate(zip(statics, f_fmts), start=1):
            _data_cell(r, j, val, fmt)
        _formula_cell(r, 8, f"={wo}*{E}{r}+{wa}*{F}{r}+{we}*{G}{r}")   # FCF = ensemble blend of the path cells
        _formula_cell(r, 10, f"={H}{r}/{I}{r}")                       # PV = FCF / discount factor
        r += 1
    r += 1

    # ---- historical financials: FCF is a LIVE formula; per-row SEC filing in the Date column ----
    _hist_name = _legal_name(sc.get("ticker"), sc.get("company_name"))
    r = _section(ws, f"HISTORICAL FINANCIALS ({_hist_name}, in $ millions, pulled via EDGAR)", r)
    hf = m.get("historical_financials", [])
    _hdr_row(r, ["Date (SEC filing)", "Revenue", "EBIT", "Op Margin", "NOPAT", "D&A", "Capex",
                 "dNWC", "FCF", "SBC", "R&D", "Shares"])
    h_fmts = [None, CUR_M, CUR_M, PCT0, CUR_M, CUR_M, CUR_M, CUR_M, None, CUR_M, CUR_M, CUR_M]
    r += 1
    hE, hF, hG, hH = (get_column_letter(k) for k in (5, 6, 7, 8))
    for row in hf:
        rev = _num(row.get("revenue")); ebit = _num(row.get("ebit"))
        form = "10-K" if str(row.get("fiscal_period")) == "FY" else "10-Q"
        statics = [f"{row.get('date')}  ({form})", _scale(rev), _scale(ebit),
                   (ebit / rev) if (rev and ebit is not None) else None,
                   _scale(row.get("nopat")), _scale(row.get("da")), _scale(row.get("capex")),
                   _scale(row.get("delta_nwc")), None, _scale(row.get("sbc")),
                   _scale(row.get("r_and_d")), _scale(row.get("shares"))]
        for j, (val, fmt) in enumerate(zip(statics, h_fmts), start=1):
            _data_cell(r, j, val, fmt)
        _formula_cell(r, 9, f"={hE}{r}+{hF}{r}-{hG}{r}-{hH}{r}")      # FCF = NOPAT + D&A - Capex - dNWC
        r += 1

    # ---- limitations ----
    r = _section(ws, "LIMITATIONS", r)
    lims = sc.get("limitations") or ["None flagged for this name."]
    for lim in lims:
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
        c = ws.cell(row=r, column=1, value=f"- {lim}"); c.font = VALUE_FONT; c.alignment = LEFT
        ws.row_dimensions[r].height = 28
        r += 1
    ws.freeze_panes = "A3"
    return anchors


# ----------------------------- Sheet 2: dated price-path visualization -----------------------------
def _build_visual_sheet(ws, sc: dict, anchors: dict | None = None) -> None:
    import datetime as _dt
    m = sc.get("model", {})
    steps = m.get("dcf_steps", {})
    for col, w in zip("ABCDEF", (16, 16, 18, 3, 20, 16)):
        ws.column_dimensions[col].width = w

    ticker = sc.get("ticker")
    legal = _legal_name(ticker, sc.get("company_name"))
    mc = sc.get("market_cap")
    price = _num(sc.get("price")) or 0.0
    intrinsic = _num(sc.get("intrinsic_value")) or 0.0
    term_g = _num(steps.get("terminal_growth_rate"))
    d = _num(sc.get("discount_to_intrinsic"))

    _title(ws, legal, last_col=12)   # sheet header = full company name

    # top metrics -- every value LINKS BACK to sheet 1 (single source of truth; edit sheet 1, sheet 2 follows)
    a = anchors or {}
    sref = a.get("sheet", "FCF projections")
    def _link(key, fallback):
        row = a.get(key)
        return f"='{sref}'!C{row}" if row else fallback     # formula link; falls back to a literal if no anchor

    def _h(row, label, value, fmt=None):
        lc = ws.cell(row=row, column=1, value=label); lc.font = LABEL_FONT; lc.alignment = LEFT
        vc = ws.cell(row=row, column=2, value=value); vc.font = VALUE_FONT; vc.alignment = LEFT
        if fmt:                                             # apply to linked formulas too, not just literals
            vc.number_format = fmt
        return vc
    _h(2, "Current Price", _link("price", price), CUR)
    iv = _h(3, "Intrinsic Value (per share)", _link("intrinsic", intrinsic), CUR)
    iv.font = Font(size=10, bold=True)                      # intrinsic value stands out as the headline number
    _h(4, "Terminal Growth Rate", _link("termg", term_g), PCT1)
    _h(5, "WACC", _link("wacc", _num(steps.get("wacc"))), PCT1)
    direction = "Upside" if (d is not None and d >= 0) else "Downside"
    vc = _h(6, f"Implied {direction} %", _link("downside", d), PCT0)
    if d is not None:                                       # mirror sheet 1: green = upside, red = downside
        vc.font = Font(size=10, bold=True, color=GREEN if d >= 0 else RED)
    _h(7, "Time Recorded (UTC)", _link("timestamp", sc.get("timestamp_utc")))

    # valuation (transition) date -- parsed from the scorecard timestamp; no recomputation
    val_date = _parse_date(sc.get("timestamp_utc")) or _dt.datetime.utcnow()
    hist = _fetch_price_history(ticker)                # [(datetime, price)]; best-effort, [] on failure
    proj = steps.get("projected_fcf") or []
    n_fwd = len(proj) or 20

    # ---- data table + chart start at row 11 ----
    start = 11
    for j, h in enumerate(["Date", "Historical Price", "DCF value"], start=1):
        c = ws.cell(row=start, column=j, value=h); c.font = HEADER_FONT; c.fill = HEADER_FILL; c.alignment = CENTER; c.border = BORDER
    rr = start + 1
    first_data = rr
    for dt, px in hist:                                 # historical: actual dates + actual prices
        if dt >= val_date:
            continue
        _date_cell(ws, rr, 1, dt); _price_cell(ws, rr, 2, px)
        rr += 1
    _date_cell(ws, rr, 1, val_date); _price_cell(ws, rr, 2, price); _price_cell(ws, rr, 3, price)  # seamless join
    rr += 1
    for q in range(1, n_fwd + 1):                       # forecast: one point per quarter, real dates
        qd = _add_months(val_date, 3 * q)
        implied = price + (intrinsic - price) * (q / n_fwd)
        _date_cell(ws, rr, 1, qd); _price_cell(ws, rr, 3, implied)
        rr += 1
    last_data = rr - 1

    # ---- line chart: historical (solid) vs forecast (dashed + markers + forced labels) ----
    try:
        chart = LineChart()
        chart.title = f"{legal} ({ticker})"             # chart header = company name (ticker)
        try:
            chart.title.overlay = False                 # sit the title above the plot, not over the grid
        except Exception:
            pass
        chart.height = 13.5; chart.width = 30
        chart.y_axis.title = "Price ($ per share)"
        try:
            chart.y_axis.title.overlay = False          # keep the y-axis label clear of the data points
        except Exception:
            pass
        chart.x_axis.title = None                       # no date-axis label
        chart.x_axis.number_format = "dd/mm/yyyy"
        chart.x_axis.delete = False
        chart.y_axis.delete = False
        chart.x_axis.majorGridlines = None
        for col in (2, 3):                              # 0 = Historical Price, 1 = Model-Implied Value
            data = Reference(ws, min_col=col, min_row=start, max_row=last_data)
            chart.add_data(data, titles_from_data=True)
        chart.set_categories(Reference(ws, min_col=1, min_row=first_data, max_row=last_data))

        # historical series: solid navy line, NO data labels / leader lines
        s_hist = chart.series[0]
        s_hist.smooth = False
        s_hist.graphicalProperties = GraphicalProperties()
        s_hist.graphicalProperties.line = LineProperties(solidFill=BLUE, w=22000)
        s_hist.marker = Marker(symbol="none")
        s_hist.dLbls = None

        # forecast series: dashed orange line + circle markers + FORCED value labels (no leader lines)
        s_fc = chart.series[1]
        s_fc.smooth = False
        s_fc.graphicalProperties = GraphicalProperties()
        fc_line = LineProperties(solidFill="C55A11", w=22000); fc_line.prstDash = "dash"
        s_fc.graphicalProperties.line = fc_line
        s_fc.marker = Marker(symbol="circle", size=5)
        dl = DataLabelList()
        dl.showVal = True
        dl.showSerName = dl.showCatName = dl.showLegendKey = dl.showPercent = dl.showBubbleSize = False
        dl.showLeaderLines = False
        dl.numFmt = "$0"
        dl.position = "t"
        s_fc.dLbls = dl

        chart.legend = None                             # no legend
        ws.add_chart(chart, "E11")
    except Exception:
        pass

    note_row = last_data + 2
    ws.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=3)
    ws.cell(row=note_row, column=1,
            value="Historical = actual market closes (DD/MM/YYYY). Forward = model-implied value path "
                  "(current price converging to intrinsic value over the forecast horizon, one point per "
                  "quarter).").font = Font(italic=True, size=9, color="808080")
    ws.freeze_panes = "A11"


# ----------------------------- helpers -----------------------------
def _scale(x):
    v = _num(x)
    return v / 1e6 if v is not None else None


def _date_cell(ws, row, col, dt):
    c = ws.cell(row=row, column=col, value=dt)
    c.number_format = "DD/MM/YYYY"
    c.font = SMALL; c.border = BORDER; c.alignment = CENTER
    return c


def _price_cell(ws, row, col, val):
    c = ws.cell(row=row, column=col, value=round(float(val), 2))
    c.number_format = CUR; c.font = SMALL; c.border = BORDER; c.alignment = RIGHT
    return c


def _add_months(dt, n):
    import calendar
    mi = dt.month - 1 + n
    y = dt.year + mi // 12
    mo = mi % 12 + 1
    d = min(dt.day, calendar.monthrange(y, mo)[1])
    return dt.replace(year=y, month=mo, day=d)


def _parse_date(ts):
    import datetime as _dt
    if not ts:
        return None
    try:
        return _dt.datetime.strptime(str(ts)[:10], "%Y-%m-%d")
    except Exception:
        return None


def _fetch_price_history(ticker, years=3):
    """Best-effort monthly close history -> [(datetime, price), ...] with REAL dates. [] on any failure.
    Reuses the same yfinance source already used elsewhere; performs no valuation calculation."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=f"{years}y", interval="1mo", auto_adjust=True)
        if df is None or df.empty or "Close" not in df:
            return []
        out = []
        for ts, px in df["Close"].dropna().items():
            d = ts.to_pydatetime().replace(tzinfo=None)
            out.append((d, float(px)))
        return out
    except Exception:
        return []


# labels mirrored from scorecard (presentation only; avoids importing to prevent cycles)
_SECTOR_LABEL = {
    "software_saas_infra": "Software / SaaS", "internet_platforms_marketplaces": "Internet Platform / Marketplace",
    "semiconductors": "Semiconductors", "semiconductor_equipment": "Semiconductor Equipment",
    "consumer_staples": "Consumer Staples", "consumer_discretionary": "Consumer Discretionary",
    "industrials_assetlight_services": "Industrials / Asset-Light Services",
}
_STAGE_BY_REGIME = {
    "optionality_hypergrowth": "Optionality / Hypergrowth", "mature_compounder": "Mature Compounder",
    "platform_growth": "Platform Growth", "consumer_defensive": "Defensive Compounder",
}


def _company_type_label(sc):
    s = sc.get("sector") or "unknown"
    return _SECTOR_LABEL.get(s, s.replace("_", " ").title())


def _company_stage_label(sc):
    stage = _STAGE_BY_REGIME.get(sc.get("regime") or "")
    if not stage:
        g = (sc.get("assumptions", {}).get("implied_annual_revenue_growth") or 0.0)
        stage = "High Growth" if g >= 0.15 else "Growth" if g >= 0.08 else "Mature"
    return stage


def _confidence_band(sc):
    c = sc.get("conviction") or 0
    return "High" if c >= 70 else "Medium" if c >= 45 else "Low"


# ----------------------------- entry point -----------------------------
def export_valuation_report(sc: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "FCF projections"
    anchors = _build_model_sheet(ws1, sc)
    ws2 = wb.create_sheet((str(sc.get("ticker")) or "Chart")[:31])
    _build_visual_sheet(ws2, sc, anchors)
    path = out_dir / f"{sc['ticker']}_valuation_report.xlsx"
    wb.save(path)
    return path
