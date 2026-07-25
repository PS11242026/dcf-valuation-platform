"""Assemble the 7-section valuation ReportModel (sell-side initiation persona) for generic & canonical.

Presentation only -- reads already-computed engine outputs (DCFResult / deterministic_baseline) and
re-evaluates the existing DCF formula at a WACC x terminal-growth grid for the sensitivity sheet.
No valuation/forecast math is changed.
"""
from __future__ import annotations

from pathlib import Path

from dcf_project.reporting_kit import Image, Metric, NativeChart, ReportModel, Table, viz3d
from dcf_project.reporting_kit.style import fmt_value, money_fmt
from dcf_project.valuation.discounting import present_value, present_value_series
from dcf_project.valuation.terminal import terminal_value


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sensitivity_grid(projected_fcf, terminal_base_annual, net_debt, shares, base_wacc, base_g):
    """Intrinsic value/share across a WACC x terminal-growth grid (re-uses the engine DCF formula)."""
    if not projected_fcf or not shares:
        return None, [], []
    waccs = [round(base_wacc + d, 4) for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    gs = [round(base_g + d, 4) for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
    matrix = []
    for w in waccs:
        row = []
        qw = w / 4.0
        for g in gs:
            try:
                if w <= g:
                    row.append(None); continue
                tv = terminal_value(terminal_base_annual, w, g)
                ev = present_value_series(projected_fcf, qw) + present_value(tv, qw, len(projected_fcf))
                iv = (ev - (net_debt or 0.0)) / shares
                row.append(round(float(iv), 2))
            except Exception:  # noqa: BLE001
                row.append(None)
        matrix.append(row)
    return matrix, [f"{w:.1%}" for w in waccs], [f"{g:.1%}" for g in gs]


_MONTH_TO_FP = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "FY"}
#: India fiscal year ends 31-Mar, so the quarter-by-period-end map differs from the US calendar.
_MONTH_TO_FP_IN = {"06": "Q1", "09": "Q2", "12": "Q3", "03": "Q4"}

#: Currency symbol + unit word used in the Historical Financials header, keyed by currency.
_CCY_LABEL = {"USD": ("$", "EDGAR"), "INR": ("₹", "NSE")}


def _filing_label(row: dict, source: str = "EDGAR") -> str:
    """The filing each historical row is pulled from: form + fiscal period + period end.

    US (EDGAR): annual periods (FY) come from the 10-K, interim quarters from the 10-Q.
    India (NSE): SEBI LODR Reg-33 results, labelled by the India fiscal calendar (FY ends
    31-Mar; Q4 is the annual results filing). The fiscal period uses the EDGAR/NSE metadata
    when present, else is inferred from the period-end month so every row names its filing.
    """
    raw_date = str(row.get("date") or "")
    period_end = raw_date[:10]
    fp = row.get("fiscal_period")
    if source == "NSE":
        if fp not in {"Q1", "Q2", "Q3", "Q4", "FY"}:
            fp = _MONTH_TO_FP_IN.get(period_end[5:7], None)
        year = period_end[:4]
        # India FY label: a period ending Apr-Dec falls in FY(year+1); Jan-Mar in FY(year).
        fy = ""
        if len(year) == 4 and year.isdigit():
            fy = f" FY{int(year) + (0 if period_end[5:7] in ('01', '02', '03') else 1)}"
        form = "NSE annual" if fp == "Q4" else "NSE results"
        period = (fp or "period") + fy
        return f"{form}  {period} (period end {period_end})" if period_end else f"{form}  {period}"
    if fp not in {"Q1", "Q2", "Q3", "FY"}:
        fp = _MONTH_TO_FP.get(period_end[5:7], None)
    form = "10-K" if fp == "FY" else "10-Q"
    period = fp or "period"
    return f"{form}  {period} (period end {period_end})" if period_end else f"{form}  {period}"


def company_display_name(ticker: str, fallback: str | None = None) -> str:
    """Best-effort full company name (yfinance, guarded). Presentation only; falls back to
    the ticker so the builders stay deterministic and offline-safe."""
    fb = fallback or (ticker or "")
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or fb
    except Exception:  # noqa: BLE001  -- offline / lookup failure: use the ticker
        return fb


def _hist_table(rows: list[dict], company_name: str | None = None,
                currency: str = "USD", source: str = "EDGAR") -> Table | None:
    """Historical Financials sheet: one row per filing, labelled with the filing it is pulled from.

    Header carries the company name and states the values are in <currency> millions, pulled via
    the named source (EDGAR for US, NSE for India). Money columns render in the right symbol.
    """
    if not rows:
        return None
    cols = ["revenue", "ebit", "op_margin", "nopat", "da", "capex", "delta_nwc", "fcf"]
    labels = {"revenue": "Revenue", "ebit": "EBIT", "op_margin": "Op margin", "nopat": "NOPAT",
              "da": "D&A", "capex": "Capex", "delta_nwc": "Chg NWC", "fcf": "FCF"}
    have = [c for c in cols if c == "op_margin" or any(c in r for r in rows)]
    data = []
    for r in rows[-12:]:
        rev, ebit = _num(r.get("revenue")), _num(r.get("ebit"))
        line = [_filing_label(r, source)]
        for c in have:
            if c == "op_margin":
                line.append((ebit / rev) if (rev and ebit is not None) else None)
            else:
                line.append(r.get(c))
        data.append(line)
    money = money_fmt("usdM", currency)
    filing_col = "NSE filing" if source == "NSE" else "SEC filing"
    columns = [filing_col] + [labels[c] for c in have]
    fmts = ["text"] + ["pct" if c == "op_margin" else money for c in have]
    sym, _ = _CCY_LABEL.get((currency or "USD").upper(), ("$", source))
    name = (company_name or "").strip()
    head = f"Historical Financials ({name + ', ' if name else ''}in {sym} millions, pulled via {source})"
    return Table(head, columns, data, formats=fmts)


def _currency_of(res) -> str:
    """Reporting currency for a DCFResult (diagnostics-stamped by engine_v3; defaults USD)."""
    diag = getattr(res, "diagnostics", None) or {}
    return str(diag.get("currency") or "USD").upper()


def _source_of(res) -> str:
    """Filing source label for a DCFResult (diagnostics-stamped by engine_v3; defaults EDGAR)."""
    diag = getattr(res, "diagnostics", None) or {}
    return str(diag.get("filing_source") or "EDGAR")


def _localize_currency(model: ReportModel, currency: str) -> ReportModel:
    """Swap every USD money-format TOKEN (Metric.fmt / Table.formats) to its currency variant.

    A no-op for USD (and any currency without a mapping), so the frozen US path is byte-identical.
    Pre-formatted string cells (built via ``fmt_value`` at assembly time) are localized at their
    call site instead, by passing the currency into ``money_fmt``.
    """
    if (currency or "USD").upper() != "INR":
        return model
    for sec in model.sections:
        for m in sec.metrics:
            if isinstance(m.fmt, str) and m.fmt.startswith("usd"):
                m.fmt = money_fmt(m.fmt, currency)
        for t in (*sec.tables, *(c.table for c in sec.charts)):
            if t.formats:
                t.formats = [money_fmt(f, currency) if isinstance(f, str) and f.startswith("usd") else f
                             for f in t.formats]
    return model


def _forecast_table(ff: list[dict]) -> Table | None:
    if not ff:
        return None
    rows, fcf_series = [], []
    for i, r in enumerate(ff, 1):
        rev = _num(r.get("revenue")); ebit = _num(r.get("ebit"))
        fcf = _num(r.get("fcf")) or _num(r.get("reported_fcf")) or _num(r.get("economic_fcf"))
        rows.append([f"Q{i}", rev, (ebit / rev if rev and ebit is not None else None), fcf])
        fcf_series.append(fcf)
    return Table("Forecast (20 quarters)", ["QTR", "Revenue", "Op margin", "FCF"], rows,
                 formats=["text", "usdM", "pct", "usdM"])


def _figdir(folder: Path) -> Path:
    d = Path(folder) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_canonical_model(sc: dict, folder: Path) -> ReportModel:
    res = sc["result"]
    a = sc["assumptions"]
    d = sc.get("discount_to_intrinsic")
    cur = _currency_of(res)             # USD (US) or INR (India); drives money symbols/scaling
    fsrc = _source_of(res)              # EDGAR (US) or NSE (India)
    csym = _CCY_LABEL.get(cur, ("$", fsrc))[0]
    figs = _figdir(folder)
    m = ReportModel(
        title=f"{sc['ticker']} -- Canonical DCF",
        subtitle="State-space FCF forecast + ensemble DCF (frozen production engine)",
        persona="initiation",
        meta={"Ticker": sc["ticker"], "As of": sc["timestamp_utc"], "Verdict": sc["verdict"],
              "Prediction confidence": sc.get("prediction_confidence"), "Rank basis": sc.get("rank_basis")},
    )
    # 1) Executive summary
    s = m.section("Executive Summary", "Executive Summary")
    s.metrics = [
        Metric("Intrinsic value / sh", sc.get("intrinsic_value"), "usd"),
        Metric("Market price", sc.get("price"), "usd"),
        Metric("Upside" if (d or 0) >= 0 else "Downside", d, "pct", good=(d is not None and d >= 0)),
        Metric("Verdict", sc.get("verdict"), "text"),
        Metric("Conviction", (sc.get("conviction") or 0) / 100.0, "pct"),
        Metric("WACC", a.get("wacc"), "pct",
               note=f"Source: {getattr(sc.get('result'), 'wacc_source', None) or 'Internal CAPM'}"),
        Metric("Terminal growth", a.get("terminal_growth"), "pct"),
        Metric("Regime", sc.get("regime"), "text"),
    ]
    s.narrative = sc.get("why", [])
    for q in sc.get("qualifiers", []):
        s.narrative.append(q)
    # 2) Historical financials
    hist = res.diagnostics.get("historical_financials", []) if hasattr(res, "diagnostics") else []
    sec = m.section("Historical", "Historical Financials")
    t = _hist_table(hist, company_display_name(sc["ticker"], sc.get("company_name")), cur, fsrc)
    if t:
        sec.tables.append(t)
    # 3) Forecast model + charts
    ff = [dict(r) for r in (res.forecast_financials or [])]
    fsec = m.section("Forecast Model", "Forecast Model")
    ft = _forecast_table(ff)
    if ft:
        fsec.charts.append(NativeChart("Revenue & FCF forecast", "line", ft, x_col=0, y_cols=[1, 3]))
    # 4) DCF model + EV->IV bridge + waterfall
    dsec = m.section("DCF Model", "DCF Model")
    pv_fcf = _num(getattr(res, "pv_projected_fcf", None))
    pv_tv = _num(getattr(res, "pv_terminal_value", None))
    ev = _num(getattr(res, "enterprise_value", None)) or 0.0
    equity = _num(getattr(res, "equity_value", None)) or 0.0
    net_debt = _num(getattr(res, "net_debt", None)) or 0.0
    shares = _num(getattr(res, "diluted_shares_outstanding", None))
    iv = _num(sc.get("intrinsic_value"))
    price = _num(sc.get("price"))
    # $-scaling formats take RAW dollars (usdM scales ÷1e6 internally) -- do not pre-divide.
    dsec.metrics = [
        Metric("PV of forecast FCF", pv_fcf or 0, "usdM"),
        Metric("PV of terminal value", pv_tv or 0, "usdM"),
        Metric("Enterprise value", ev, "usdM"),
        Metric("Equity value", equity, "usdM"),
        Metric("PV(TV) / EV", (pv_tv / ev) if (pv_tv and ev) else None, "pct",
               note="Terminal value's share of enterprise value -- lower is more defensible."),
    ]
    mny, mny0 = money_fmt("usdM", cur), money_fmt("usd", cur)
    bridge = [
        ["PV of explicit FCF", fmt_value(pv_fcf, mny)],
        ["PV of terminal value", fmt_value(pv_tv, mny)],
        ["Enterprise value", fmt_value(ev, mny)],
        ["(-) Net debt", fmt_value(-net_debt, mny)],
        ["Equity value", fmt_value(equity, mny)],
        ["(/) Diluted shares (M)", fmt_value((shares or 0) / 1e6, "num")],
        ["Intrinsic value / share", fmt_value(iv, mny0)],
        ["Market price", fmt_value(price, mny0)],
        ["Upside / (downside)", fmt_value(d, "pct")],
    ]
    dsec.tables.append(Table("Enterprise value -> intrinsic value bridge",
                             ["Item", "Value"], bridge, formats=["text", "text"], total_row=True))
    wf = viz3d.waterfall(["PV(FCF)", "PV(TV)", "-Net debt", "Equity"],
                         [(pv_fcf or 0) / 1e6, (pv_tv or 0) / 1e6, -net_debt / 1e6, None],
                         figs / "dcf_waterfall.png", title=f"{sc['ticker']} DCF bridge ({csym}M)")
    if wf:
        dsec.images.append(Image("DCF value bridge", wf))
    # 5) Sensitivity
    ssec = m.section("Sensitivity", "Sensitivity Analysis")
    tb_annual = _num((res.diagnostics.get("terminal_assumptions") or {}).get("terminal_base_annual")) \
        if hasattr(res, "diagnostics") else None
    tb_annual = tb_annual or ((_num(getattr(res, "fcf_terminal_base", None)) or 0.0) * 4.0)
    matrix, wl, gl = sensitivity_grid(list(getattr(res, "projected_fcf", []) or []), tb_annual, net_debt,
                                      _num(getattr(res, "diluted_shares_outstanding", None)),
                                      a.get("wacc") or 0.1, a.get("terminal_growth") or 0.025)
    if matrix:
        rows = [[wl[i]] + matrix[i] for i in range(len(wl))]
        ssec.tables.append(Table("Intrinsic / share: WACC (rows) x terminal growth (cols)",
                                 ["WACC \\ g"] + gl, rows, formats=["text"] + ["usd"] * len(gl),
                                 heatmap_cols=list(range(1, len(gl) + 1))))
        hm = viz3d.sensitivity_heatmap(matrix, gl, wl, figs / "sensitivity.png",
                                       title=f"{sc['ticker']} intrinsic value sensitivity")
        if hm:
            ssec.images.append(Image("Sensitivity heatmap", hm))
    # 6) Valuation drivers + assumption audit
    vsec = m.section("Valuation Drivers", "Valuation Drivers")
    vsec.tables.append(Table("Key assumptions", ["Driver", "Value"], [
        ["Implied revenue growth (/yr)", a.get("implied_annual_revenue_growth")],
        ["Terminal operating margin", a.get("terminal_operating_margin")],
        ["Terminal growth", a.get("terminal_growth")],
        ["WACC", a.get("wacc")],
        ["Forecast confidence", sc.get("forecast_confidence")],
    ], formats=["text", "pct"]))
    src = "; ".join(sorted({getattr(s, "provider", str(s)) for s in (getattr(res, "sources", []) or [])})) or "n/a"
    warns = "; ".join(getattr(res, "warnings", []) or []) or "none"
    vsec.tables.append(Table("Assumption audit", ["Field", "Value"], [
        ["Engine version", getattr(res, "engine_version", "v1")],
        ["Forecast mode", getattr(res, "forecast_mode", None)],
        ["WACC", fmt_value(a.get("wacc"), "pct")],
        ["Terminal growth", fmt_value(a.get("terminal_growth"), "pct")],
        ["Data providers", src],
        ["Warnings", warns],
        ["Conviction (frozen, pre-correction)", sc.get("conviction")],
        ["Sector 3Y IC (frozen, pre-correction)", sc.get("sector_validation_3y_ic")],
    ], formats=["text", "text"]))
    vsec.narrative = [
        "Conviction and the sector-reliability IC are FROZEN pre-correction constructs "
        "(see docs/RESEARCH_POLICY.md): the split-corrected broad finding is ~0 IC, with only "
        "consumer staples surviving (3Y IC +0.211). They are shown for provenance, not as a "
        "live predictive claim.",
    ]
    # 7) Investment thesis
    tsec = m.section("Investment Thesis", "Investment Thesis")
    tsec.narrative = sc.get("why", []) + (["", "Risks / qualifiers:"] + sc["qualifiers"] if sc.get("qualifiers") else [])
    return _localize_currency(m, cur)


def build_generic_model(ticker: str, baseline: dict, price, folder: Path) -> ReportModel:
    iv = _num(baseline.get("intrinsic_value_per_share"))
    d = (iv / price - 1.0) if (iv and price) else None
    figs = _figdir(folder)
    m = ReportModel(
        title=f"{ticker} -- Generic DCF",
        subtitle="Textbook single-stage normalized-margin DCF baseline",
        persona="initiation",
        meta={"Ticker": ticker, "Mode": baseline.get("valuation_mode"),
              "Terminal growth": baseline.get("terminal_growth_rate")},
    )
    s = m.section("Executive Summary", "Executive Summary")
    s.metrics = [
        Metric("Intrinsic value / sh", iv, "usd"),
        Metric("Market price", price, "usd"),
        Metric("Upside" if (d or 0) >= 0 else "Downside", d, "pct", good=(d is not None and d >= 0)),
        Metric("Revenue CAGR (5y)", baseline.get("revenue_cagr_5y"), "pct"),
        Metric("EBIT margin", baseline.get("normalized_ebit_margin"), "pct"),
        Metric("Terminal growth", baseline.get("terminal_growth_rate"), "pct"),
    ]
    s.narrative = [
        "Generic baseline: constant normalized margins + single terminal growth (no state-space forecast).",
        "Use the canonical DCF for the production valuation; this is the transparent textbook comparison.",
    ]
    vsec = m.section("Valuation Drivers", "Valuation Drivers")
    vsec.tables.append(Table("Normalized assumptions", ["Driver", "Value"], [
        ["Revenue CAGR (5y)", baseline.get("revenue_cagr_5y")],
        ["EBIT margin", baseline.get("normalized_ebit_margin")],
        ["Tax rate", baseline.get("normalized_tax_rate")],
        ["D&A / revenue", baseline.get("normalized_da_ratio")],
        ["Capex / revenue", baseline.get("normalized_capex_ratio")],
        ["Maintenance capex / revenue", baseline.get("normalized_maintenance_capex_ratio")],
        ["NWC / revenue", baseline.get("normalized_nwc_ratio")],
        ["Terminal growth", baseline.get("terminal_growth_rate")],
    ], formats=["text", "pct"]))
    dsec = m.section("DCF Model", "DCF Model")
    ev = _num(baseline.get("enterprise_value")) or 0.0
    dsec.metrics = [Metric("Enterprise value", ev, "usdM"),  # raw $ (usdM scales internally)
                    Metric("Intrinsic value / sh", iv, "usd")]
    return m
