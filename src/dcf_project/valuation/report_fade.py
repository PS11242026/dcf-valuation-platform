"""Flagship Fundamentals Fade workbook builder (engine_v3) -> fade_dcf.xlsx.

Presentation only. Reads the already-computed v3 ``DCFResult`` + its ``driver_audit``
(every projected number traced to a named economic assumption) and re-prices the
EXISTING forecast frame at perturbed discount/terminal inputs for the sensitivity and
tornado sheets via the engine's own pure ``assemble.value_fade_path`` -- the forecast
frame is never re-faded here, so no valuation/forecast math is duplicated and the grid
centre reproduces the headline intrinsic exactly (unit-tested).

Sections (plan Section 8): Executive Summary, Valuation Summary (EV->IV bridge +
waterfall), Fade Drivers (40q paths + per-driver half-lives), CAP & Persistence,
Terminal Value (Grinold-Kroner decomposition + value-driver TV), Sensitivity (two
re-priced grids), Driver Attribution (tornado), Forecast Audit Trail, Uncertainty
(rendered ONLY when Monte Carlo ran, with the mandatory G6 caption), Historical +
Investment Thesis (shared with the canonical builder).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dcf_project.contracts import MarketSnapshot, SourceMeta
from dcf_project.engine_v3.assemble import value_fade_path
from dcf_project.reporting_kit import Image, Metric, NativeChart, ReportModel, Table, viz3d
from dcf_project.reporting_kit.style import fmt_value, money_fmt
from dcf_project.research import validation_evidence as ve
from dcf_project.valuation.report import (
    _CCY_LABEL,
    _currency_of,
    _figdir,
    _hist_table,
    _localize_currency,
    _num,
    _source_of,
    company_display_name,
)

#: The mandatory caption for ANY p10/p50/p90 rendered by the product (G6 KILL discipline).
#: Asserted verbatim-substring by the report unit test; do not soften.
G6_CAPTION = (
    "Driver-sensitivity range -- NOT a calibrated forecast of returns (failed "
    "pre-registered calibration gate G6: p10-p90 coverage 5.2% vs 80% nominal)."
)


def _get(d: dict | None, *keys: str, default: Any = None) -> Any:
    """Nested dict getter that tolerates missing keys / None at any level."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _placeholder_market(ticker: str) -> MarketSnapshot:
    """Minimal valid MarketSnapshot for re-pricing; cash/debt/shares are passed
    explicitly to ``value_fade_path`` so this is never dereferenced for them."""
    return MarketSnapshot(ticker=ticker, source=SourceMeta(provider="report_fade", key="reprice"))


def _repricer(res, ticker: str):
    """Build a closure that re-prices the EXISTING forecast frame at (wacc, g, RONIC_T).

    Returns ``(reprice, ok)`` where ``ok`` is False when the frame/shares are unusable.
    ``reprice(w, g, ronic_t)`` -> intrinsic/share (float) or None on a guard breach
    (e.g. WACC <= g). At the base inputs it reproduces the headline intrinsic exactly.
    """
    ff = list(res.forecast_financials or [])
    shares = _num(getattr(res, "diluted_shares_outstanding", None))
    if not ff or not shares:
        return (lambda w, g, rt: None), False
    frame = pd.DataFrame(ff)
    if "fcf" not in frame or "nopat" not in frame:
        return (lambda w, g, rt: None), False
    net_debt = _num(getattr(res, "net_debt", None)) or 0.0
    market = _placeholder_market(ticker)

    def reprice(w: float, g: float, ronic_t: float):
        try:
            # net_debt = debt - cash; pass cash=0, debt=net_debt to reproduce the bridge.
            v = value_fade_path(frame, w, g, ronic_t, 0.0, net_debt, shares, market)
            return float(v["intrinsic"])
        except Exception:  # noqa: BLE001  -- a guard breach is a blank cell, not a crash
            return None

    return reprice, True


def _grid(reprice, row_vals, col_vals, ronic_of):
    """Re-price across a row x col grid. ``ronic_of(w, g)`` supplies RONIC_T per cell.
    Cells rounded to 2dp so the centre cell equals the rounded headline intrinsic."""
    matrix = []
    for w in row_vals:
        line = []
        for g in col_vals:
            iv = reprice(w, g, ronic_of(w, g))
            line.append(round(iv, 2) if iv is not None else None)
        matrix.append(line)
    return matrix


def fade_iv_range(res, ticker: str) -> tuple[float | None, float | None, float | None]:
    """(min, max, headline) intrinsic across the WACC x g8 re-pricing grid.

    The fade 'whiskers' for the triangulation football field (``valuation compare``);
    reuses the section-6 re-pricer so the range is consistent with the workbook grid.
    """
    reprice, ok = _repricer(res, ticker)
    headline = _num(getattr(res, "intrinsic_value_per_share", None))
    da = (res.diagnostics or {}).get("driver_audit", {}) if hasattr(res, "diagnostics") else {}
    wacc = _num(getattr(res, "wacc", None)) or 0.0
    g_term = _num(getattr(res, "terminal_growth_rate", None)) or 0.0
    excess_terminal = _num(_get(da, "persistence", "excess_terminal")) or 0.0
    if not ok or not wacc:
        return None, None, headline
    waccs = [wacc + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    gs = [g_term + d for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
    vals = [reprice(w, g, w + excess_terminal) for w in waccs for g in gs]
    vals = [v for v in vals if v is not None]
    return (min(vals) if vals else None, max(vals) if vals else None, headline)


def build_fade_model(sc: dict, folder: Path) -> ReportModel:
    res = sc["result"]
    a = sc.get("assumptions") or {}
    da = (res.diagnostics or {}).get("driver_audit", {}) if hasattr(res, "diagnostics") else {}
    gk = _get(da, "grinold_kroner", default={})
    per = _get(da, "persistence", default={})
    drv = _get(da, "drivers", default={})
    hl = _get(da, "half_lives", default={})
    term = _get(da, "terminal", default={})
    paths = _get(da, "paths", default={})
    figs = _figdir(folder)
    cur = _currency_of(res)             # USD (US) or INR (India): money symbols, scaling, header
    fsrc = _source_of(res)              # EDGAR (US) or NSE (India)
    csym = _CCY_LABEL.get(cur, ("$", fsrc))[0]
    mny, mny0 = money_fmt("usdM", cur), money_fmt("usd", cur)  # baked-string money formats

    wacc = _num(getattr(res, "wacc", None)) or _num(a.get("wacc")) or 0.0
    g_term = _num(getattr(res, "terminal_growth_rate", None)) or _num(gk.get("terminal_growth")) or 0.0
    iv = _num(getattr(res, "intrinsic_value_per_share", None))
    price = _num(getattr(res, "market_price", None)) or _num(sc.get("price"))
    upside = _num(getattr(res, "implied_upside", None))
    if upside is None and iv and price:
        upside = iv / price - 1.0
    excess_terminal = _num(per.get("excess_terminal")) or 0.0
    ronic_terminal = _num(term.get("ronic_terminal")) or (wacc + excess_terminal)

    m = ReportModel(
        title=f"{sc['ticker']} -- Fundamentals Fade DCF",
        subtitle="Economic-driver fade model (engine v3): identity-derived reinvestment, "
                 "per-driver half-lives, CAP-faded RONIC, Grinold-Kroner terminal",
        persona="initiation",
        meta={
            "Ticker": sc["ticker"], "As of": sc.get("timestamp_utc"),
            "Engine": "v3 (Fundamentals Fade)", "Verdict": sc.get("verdict"),
            "Sector": _get(res.diagnostics, "sector") if hasattr(res, "diagnostics") else None,
            "Archetype": getattr(res, "company_archetype", None),
            "WACC source": getattr(res, "wacc_source", None) or "Internal CAPM",
        },
    )

    # ---- 1) Executive Summary -------------------------------------------------
    s = m.section("Executive Summary", "Executive Summary")
    s.metrics = [
        Metric("Intrinsic value / sh", iv, "usd"),
        Metric("Market price", price, "usd"),
        Metric("Upside" if (upside or 0) >= 0 else "Downside", upside, "pct",
               good=(upside is not None and upside >= 0)),
        Metric("Data quality score", _num(drv.get("quality")), "num",
               note="0-1 composite of history length / coverage; scales the empirical-Bayes "
                    "shrinkage toward sector priors."),
        Metric("WACC", wacc, "pct", note=f"Source: {getattr(res, 'wacc_source', None) or 'Internal CAPM'}"),
        Metric("Terminal growth g8", g_term, "pct"),
        Metric("CAP (competitive advantage period)", _num(per.get("cap_years")), "yrs"),
        Metric("PV(TV) / EV", _num(da.get("pv_terminal_share_of_ev")), "pct",
               note="Terminal value's share of enterprise value -- lower is more defensible."),
    ]
    s.narrative = [
        "Headline basis: the deterministic fade skeleton (never a Monte Carlo statistic).",
        "Forecast primitives are revenue growth, operating margin and RONIC -- each mean-"
        "reverts at its own pace. Reinvestment is DERIVED every period from the identity "
        "g = RONIC x reinvestment (reinv_t = g_t / RONIC_t), so growth is always paid for "
        "and independent driver fades cannot manufacture value.",
    ]
    s.narrative += list(sc.get("why", []))
    s.narrative += list(sc.get("qualifiers", []))

    # ---- 2) Valuation Summary (EV -> IV bridge + waterfall) -------------------
    vs = m.section("Valuation Summary", "Valuation Summary")
    pv_fcf = _num(getattr(res, "pv_projected_fcf", None)) or 0.0
    pv_tv = _num(getattr(res, "pv_terminal_value", None)) or 0.0
    ev = _num(getattr(res, "enterprise_value", None)) or 0.0
    equity = _num(getattr(res, "equity_value", None)) or 0.0
    net_debt = _num(getattr(res, "net_debt", None)) or 0.0
    shares = _num(getattr(res, "diluted_shares_outstanding", None))
    vs.metrics = [
        Metric("PV of explicit FCF (10y)", pv_fcf, "usdM"),
        Metric("PV of terminal value", pv_tv, "usdM"),
        Metric("Enterprise value", ev, "usdM"),
        Metric("PV(explicit) / EV", (pv_fcf / ev) if ev else None, "pct"),
    ]
    bridge = [
        ["PV of explicit FCF (10y)", fmt_value(pv_fcf, mny)],
        ["PV of terminal value", fmt_value(pv_tv, mny)],
        ["Enterprise value", fmt_value(ev, mny)],
        ["(-) Net debt", fmt_value(-net_debt, mny)],
        ["Equity value", fmt_value(equity, mny)],
        ["(/) Diluted shares (M)", fmt_value((shares or 0) / 1e6, "num")],
        ["Intrinsic value / share", fmt_value(iv, mny0)],
        ["Market price", fmt_value(price, mny0)],
        ["Upside / (downside)", fmt_value(upside, "pct")],
    ]
    vs.tables.append(Table("Enterprise value -> intrinsic value bridge", ["Item", "Value"],
                           bridge, formats=["text", "text"], total_row=True))
    wf = viz3d.waterfall(["PV(FCF)", "PV(TV)", "-Net debt", "Equity"],
                         [pv_fcf / 1e6, pv_tv / 1e6, -net_debt / 1e6, None],
                         figs / "fade_waterfall.png", title=f"{sc['ticker']} fade DCF bridge ({csym}M)")
    if wf:
        vs.images.append(Image("Value bridge", wf))

    # ---- 3) Fade Drivers (flagship) -------------------------------------------
    fd = m.section("Fade Drivers", "Fade Drivers")
    fd.narrative = [
        "Each primitive fades from its current reading to its long-run anchor on a half-life "
        "h (years for half the remaining gap to close): x(t) = anchor + (x0 - anchor) * 0.5^(t/h). "
        "Reinvestment below is DERIVED from g(t)/RONIC(t) -- a diagnostic output, not a forecast input.",
    ]
    ff = [dict(r) for r in (res.forecast_financials or [])]
    if ff:
        rows = [[int(r.get("quarter", i + 1)), _num(r.get("growth_annual")), _num(r.get("nopat_margin")),
                 _num(r.get("ronic")), _num(r.get("reinvestment_rate")), _num(r.get("fcf_margin"))]
                for i, r in enumerate(ff)]
        fd.tables.append(Table(
            "40-quarter driver paths", ["Q", "Rev growth", "NOPAT margin", "RONIC",
                                        "Reinvestment (derived)", "Implied FCF margin"],
            rows, formats=["num0", "pct", "pct", "pct", "pct", "pct"],
            col_widths=[6, 12, 13, 11, 18, 15]))
    # Charts driven off the audit paths, with the equilibrium asymptotes as flat series.
    g_path = paths.get("growth_annual") or []
    m_path = paths.get("nopat_margin") or []
    r_path = paths.get("ronic") or []
    margin_anchor = _num(da.get("margin_target_blend"))
    if g_path and m_path:
        ctab = Table("Driver fade vs anchors", ["Q", "Growth", "g8", "NOPAT margin", "Margin target"],
                     [[i + 1, _num(g_path[i]), g_term, _num(m_path[i]), margin_anchor]
                      for i in range(len(g_path))],
                     formats=["num0", "pct", "pct", "pct", "pct"], col_widths=[6, 11, 11, 13, 13])
        fd.charts.append(NativeChart("Growth & margin fade to anchors", "line", ctab,
                                     x_col=0, y_cols=[1, 2, 3, 4], x_title="Quarter", y_title="Annual rate"))
    if r_path:
        rtab = Table("Return on new capital decay", ["Q", "RONIC", "WACC"],
                     [[i + 1, _num(r_path[i]), wacc] for i in range(len(r_path))],
                     formats=["num0", "pct", "pct"], col_widths=[6, 11, 11])
        fd.charts.append(NativeChart("RONIC fades toward WACC (moat decay)", "line", rtab,
                                     x_col=0, y_cols=[1, 2], x_title="Quarter", y_title="Annual rate"))
    # Per-driver half-life estimation table.
    g_hl, m_hl = _get(hl, "growth", default={}), _get(hl, "margin", default={})
    fd.tables.append(Table(
        "Per-driver fade half-lives (company AR(1), EB-shrunk to sector)",
        ["Driver", "h used (yr)", "Source", "rho company (ann.)", "rho sector prior",
         "Sector h (yr)", "AR(1) pairs", "Shrink weight"],
        [["Revenue growth", _num(g_hl.get("h_used_years")), g_hl.get("source"),
          _num(g_hl.get("rho_company_annual")), _num(g_hl.get("rho_sector_prior")),
          _num(g_hl.get("sector_half_life_years")), _num(g_hl.get("n_pairs")), _num(g_hl.get("shrink_weight"))],
         ["Operating margin", _num(m_hl.get("h_used_years")), m_hl.get("source"),
          _num(m_hl.get("rho_company_annual")), _num(m_hl.get("rho_sector_prior")),
          _num(m_hl.get("sector_half_life_years")), _num(m_hl.get("n_pairs")), _num(m_hl.get("shrink_weight"))]],
        formats=["text", "yrs", "text", "num", "num", "yrs", "num0", "num"]))
    fd.narrative.append(
        "ROIC fade is governed by the single persistence parameter phi (h_roic = ln0.5 / ln phi); "
        "see CAP & Persistence. A 'fallback_cap' source means the company's own reversion was not "
        "identifiable (too few points / no clean mean reversion) and the CAP-derived half-life is used.")

    # ---- 4) CAP & Persistence -------------------------------------------------
    cp = m.section("CAP & Persistence", "CAP & Persistence")
    phi = _num(per.get("phi")) or 0.0
    excess_0 = _num(per.get("excess_0"))
    if excess_0 is None:
        excess_0 = _num(drv.get("excess_roic_0")) or 0.0
    cp.metrics = [
        Metric("Persistence phi", phi, "num", note="Annual decay of excess ROIC; the moat-decay rate."),
        Metric("CAP", _num(per.get("cap_years")), "yrs"),
        Metric("Excess ROIC0 (ROIC0 - WACC)", excess_0, "pct"),
        Metric("Surviving excess (yr 10)", excess_terminal, "pct"),
    ]
    cp.tables.append(Table("Persistence fit", ["Field", "Value"], [
        ["phi (shrunk)", fmt_value(phi, "num")],
        ["phi (company AR(1))", fmt_value(_num(per.get("phi_company")), "num")],
        ["phi standard error", fmt_value(_num(per.get("phi_se")), "num")],
        ["Annual excess-ROIC pairs", fmt_value(_num(per.get("n_years")), "num0")],
        ["EB shrink weight (company)", fmt_value(_num(per.get("shrink_weight")), "num")],
        ["Sector", str(per.get("sector"))],
        ["Excess ROIC0", fmt_value(excess_0, "pct")],
        ["Surviving excess (yr 10)", fmt_value(excess_terminal, "pct")],
        ["RONIC_T = WACC + surviving excess", fmt_value(ronic_terminal, "pct")],
        ["CAP (excess -> 50bp floor)", fmt_value(_num(per.get("cap_years")), "yrs")],
    ], formats=["text", "text"]))
    # Annual excess-ROIC decay (excess_0 * phi^y) with the 50bp floor line + CAP marker note.
    decay_rows = []
    for y in range(0, 11):
        ex = excess_0 * (phi ** y)
        decay_rows.append([y, ex, max(wacc + ex, 0.02), 0.005])
    dtab = Table("Excess-ROIC decay (annual)", ["Year", "Excess ROIC", "RONIC", "50bp floor"],
                 decay_rows, formats=["num0", "pct", "pct", "pct"], col_widths=[8, 13, 11, 11])
    cp.tables.append(dtab)
    cp.charts.append(NativeChart("Excess ROIC decays to the floor over the CAP", "line", dtab,
                                 x_col=0, y_cols=[1, 3], x_title="Year", y_title="Excess ROIC"))
    cp.narrative = [
        f"phi is fit by a through-origin AR(1) on annual excess-ROIC readings, empirical-Bayes "
        f"shrunk toward the sector prior (weight = n/(n+n0)). The Competitive Advantage Period is "
        f"the number of years for the excess return to decay to the 50bp floor under phi: "
        f"CAP = ln(0.005/excess0)/ln(phi) ~= {fmt_value(_num(per.get('cap_years')), 'yrs')}.",
    ]

    # ---- 5) Terminal Value (Grinold-Kroner + value-driver) --------------------
    tv = m.section("Terminal Value", "Terminal Value")
    tv.metrics = [
        Metric("Terminal growth g8", g_term, "pct"),
        Metric("RONIC_T", ronic_terminal, "pct"),
        Metric("Terminal reinvestment", _num(term.get("terminal_reinvestment_rate")), "pct"),
        Metric("PV(TV) / EV", _num(da.get("pv_terminal_share_of_ev")), "pct"),
    ]
    tv.tables.append(Table("Grinold-Kroner terminal-growth decomposition", ["Component", "Rate"], [
        ["Expected inflation (anchor)", fmt_value(_num(gk.get("inflation_anchor")), "pct")],
        ["(+) Implied real growth", fmt_value(_num(gk.get("implied_real_growth")), "pct")],
        ["(=) Long-run nominal GDP", fmt_value(_num(gk.get("long_run_nominal_gdp")), "pct")],
        ["(-) Dilution haircut", fmt_value(_num(gk.get("dilution_haircut")), "pct")],
        ["(=) Grinold-Kroner raw", fmt_value(_num(gk.get("gk_raw")), "pct")],
        ["Band floor", fmt_value(_num(gk.get("band_floor")), "pct")],
        ["Band ceiling = min(cap, WACC - 1.5%)", fmt_value(_num(gk.get("band_ceiling")), "pct")],
        ["(=) Terminal growth g8 (clamped)", fmt_value(g_term, "pct")],
    ], formats=["text", "text"], total_row=True))
    tv.tables.append(Table("Value-driver terminal value", ["Component", "Value"], [
        ["Terminal-year NOPAT_T+1 (annual)", fmt_value(_num(term.get("nopat_t1_annual")), mny)],
        ["Terminal growth g8", fmt_value(g_term, "pct")],
        ["RONIC_T", fmt_value(ronic_terminal, "pct")],
        ["Terminal reinvestment = g8 / RONIC_T", fmt_value(_num(term.get("terminal_reinvestment_rate")), "pct")],
        ["Terminal FCF_T+1 = NOPAT x (1 - reinv)", fmt_value(_num(term.get("terminal_fcf_t1_annual")), mny)],
        ["Spread = WACC - g8", fmt_value(_num(term.get("spread")), "pct")],
        ["Terminal value = FCF / spread", fmt_value(_num(term.get("terminal_value")), mny)],
    ], formats=["text", "text"], total_row=True))
    reinv_t = _num(term.get("terminal_reinvestment_rate"))
    spread = _num(term.get("spread"))
    tv.narrative = [
        f"TV = NOPAT_T+1 x (1 - g8/RONIC_T) / (WACC - g8) = "
        f"{fmt_value(_num(term.get('nopat_t1_annual')), mny)} x (1 - {fmt_value(reinv_t, 'pct')}) / "
        f"{fmt_value(spread, 'pct')} = {fmt_value(_num(term.get('terminal_value')), mny)}.",
        "RONIC_T >= WACC by construction (persistence floors the surviving excess at 0), so terminal "
        "reinvestment <= g8/WACC and growth priced at RONIC->WACC is value-neutral.",
        "Sector context enters the EXPLICIT window only (margin anchor, phi prior, half-life priors), "
        "never the perpetuity -- no firm out-grows the economy forever.",
    ]

    # ---- 6) Sensitivity (two re-priced grids; centre == headline) -------------
    ss = m.section("Sensitivity", "Sensitivity Analysis")
    reprice, ok = _repricer(res, sc["ticker"])
    if ok and wacc and shares:
        # Axis values are EXACT (the d=0 centre is exactly the base input), so re-pricing the
        # centre reproduces the headline intrinsic precisely; only the labels are rounded.
        waccs = [wacc + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
        gs = [g_term + d for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
        ronics = [ronic_terminal + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
        # Grid 1: WACC x g8. The moat premium is held fixed, so RONIC_T moves with WACC.
        m1 = _grid(reprice, waccs, gs, ronic_of=lambda w, g: w + excess_terminal)
        rows1 = [[f"{waccs[i]:.1%}"] + m1[i] for i in range(len(waccs))]
        ss.tables.append(Table("Intrinsic / share: WACC (rows) x terminal growth (cols)",
                               ["WACC \\ g8"] + [f"{g:.1%}" for g in gs], rows1,
                               formats=["text"] + ["usd"] * len(gs),
                               heatmap_cols=list(range(1, len(gs) + 1))))
        hm1 = viz3d.sensitivity_heatmap(m1, [f"{g:.1%}" for g in gs], [f"{w:.1%}" for w in waccs],
                                        figs / "fade_sens_wacc_g.png",
                                        title=f"{sc['ticker']} IV: WACC x terminal growth")
        if hm1:
            ss.images.append(Image("WACC x terminal growth", hm1))
        # Grid 2: g8 x RONIC_T (WACC held).
        m2 = _grid(reprice, gs, ronics, ronic_of=lambda g, rt: rt)
        rows2 = [[f"{gs[i]:.1%}"] + m2[i] for i in range(len(gs))]
        ss.tables.append(Table("Intrinsic / share: terminal growth (rows) x RONIC_T (cols)",
                               ["g8 \\ RONIC_T"] + [f"{r:.1%}" for r in ronics], rows2,
                               formats=["text"] + ["usd"] * len(ronics),
                               heatmap_cols=list(range(1, len(ronics) + 1))))
        ss.narrative = [
            "Both grids re-price the EXISTING fade forecast at perturbed discount/terminal inputs "
            "(the explicit FCF path is held fixed); the centre cell reproduces the headline intrinsic.",
        ]
    else:
        ss.narrative = ["Sensitivity unavailable: forecast frame or share count missing."]

    # ---- 7) Driver Attribution (tornado over the valuation inputs) ------------
    da_sec = m.section("Driver Attribution", "Driver Attribution")
    if ok and iv and wacc and shares:
        base = iv
        specs = [
            ("WACC -1pp / +1pp", reprice(wacc - 0.01, g_term, (wacc - 0.01) + excess_terminal),
             reprice(wacc + 0.01, g_term, (wacc + 0.01) + excess_terminal)),
            ("Terminal growth g8 -50bp / +50bp", reprice(wacc, g_term - 0.005, ronic_terminal),
             reprice(wacc, g_term + 0.005, ronic_terminal)),
            ("RONIC_T -2pp / +2pp", reprice(wacc, g_term, ronic_terminal - 0.02),
             reprice(wacc, g_term, ronic_terminal + 0.02)),
        ]
        items, trows = [], []
        for label, lo, hi in specs:
            if lo is None or hi is None:
                continue
            items.append({"label": label, "low": lo, "high": hi})
            trows.append([label, round(lo, 2), round(base, 2), round(hi, 2), round(abs(hi - lo), 2)])
        items.sort(key=lambda it: abs(it["high"] - it["low"]), reverse=True)
        trows.sort(key=lambda r: r[4], reverse=True)
        if trows:
            da_sec.tables.append(Table("One-at-a-time intrinsic-value swings",
                                       ["Driver", "Low", "Base", "High", "Swing"], trows,
                                       formats=["text", "usd", "usd", "usd", "usd"]))
            tor = viz3d.tornado(items, base=base, out=figs / "fade_tornado.png",
                                title=f"{sc['ticker']} driver attribution ({csym}/sh)")
            if tor:
                da_sec.images.append(Image("Tornado", tor))
    da_sec.narrative = [
        "Valuation sensitivity holding the fade forecast fixed (discount + terminal inputs), via "
        "pure re-pricing of the existing frame. The forecast-shape drivers -- growth/margin "
        "half-lives and phi -- reshape the projection itself and are shown on the Fade Drivers and "
        "CAP & Persistence sheets; cross-engine attribution is in `valuation compare`.",
    ]

    # ---- 8) Forecast Audit Trail ----------------------------------------------
    au = m.section("Forecast Audit Trail", "Forecast Audit Trail")
    g_base = _get(drv, "g_base", default={})
    mtc = _get(drv, "margin_target_company", default={})
    au.tables.append(Table("Driver estimates (value / dispersion / window)",
                           ["Driver", "Value", "Std err", "Window"], [
        ["Base revenue growth g0", fmt_value(_num(g_base.get("value")), "pct"),
         fmt_value(_num(g_base.get("se")), "pct"), str(g_base.get("window"))],
        ["Current NOPAT margin", fmt_value(_num(drv.get("margin_current")), "pct"), "", ""],
        ["Company margin target", fmt_value(_num(mtc.get("value")), "pct"),
         fmt_value(_num(mtc.get("se")), "pct"), str(mtc.get("window"))],
        ["Blended margin target", fmt_value(_num(da.get("margin_target_blend")), "pct"), "", ""],
        ["ROIC0 (annual)", fmt_value(_num(drv.get("roic_annual_0")), "pct"), "", ""],
        ["Excess ROIC0", fmt_value(excess_0, "pct"), "", ""],
        ["Effective tax rate", fmt_value(_num(drv.get("tax_rate")), "pct"), "", ""],
        ["Base revenue (TTM)", fmt_value(_num(drv.get("base_revenue_ttm")), mny), "", ""],
        ["Base NOPAT (TTM)", fmt_value(_num(drv.get("base_nopat_ttm")), mny), "", ""],
        ["Base invested capital", fmt_value(_num(drv.get("base_invested_capital")), mny), "", ""],
    ], formats=["text", "text", "text", "text"]))
    band = _get(da, "fcf_margin_band", default={})
    gb = da.get("growth_binds_quarters") or []
    db = da.get("divestment_binds_quarters") or []
    breach = da.get("fcf_margin_breach_quarters") or []
    au.tables.append(Table("Binds & coherence tripwires", ["Check", "Result"], [
        ["Reinvestment-cap binds (quarters)", f"{len(gb)}" + (f"  {gb}" if gb else "")],
        ["Divestment-floor binds (quarters)", f"{len(db)}" + (f"  {db}" if db else "")],
        ["Historical FCF-margin band p10/p50/p90",
         f"{fmt_value(_num(band.get('p10')), 'pct')} / {fmt_value(_num(band.get('p50')), 'pct')} / "
         f"{fmt_value(_num(band.get('p90')), 'pct')}"],
        ["Implied final-TTM FCF margin", fmt_value(_num(da.get("implied_fcf_margin_final_ttm")), "pct")],
        ["FCF-margin band breaches (quarters)", f"{len(breach)}" + (f"  {breach}" if breach else "")],
    ], formats=["text", "text"]))
    au.tables.append(Table("Sector context (display-only; tilt kappa = 0.0)", ["Field", "Value"], [
        ["Sector", str(_get(res.diagnostics, "sector") if hasattr(res, "diagnostics") else None)],
        ["Sector margin anchor", fmt_value(_num(da.get("sector_margin_anchor")), "pct")],
        ["Sector anchor weight", fmt_value(_num(da.get("sector_anchor_weight")), "num")],
        ["Applied growth tilt", fmt_value(_num(da.get("sector_tilt")), "pct")],
    ], formats=["text", "text"]))
    src = "; ".join(sorted({getattr(x, "provider", str(x)) for x in (getattr(res, "sources", []) or [])})) or "n/a"
    warns = "; ".join(getattr(res, "warnings", []) or []) or "none"
    au.tables.append(Table("Provenance", ["Field", "Value"], [
        ["Engine version", getattr(res, "engine_version", "v3")],
        ["Forecast mode", getattr(res, "forecast_mode", None)],
        ["Explicit horizon (quarters)", getattr(res, "forecast_horizon_quarters", None)],
        ["Data providers", src],
        ["Warnings", warns],
    ], formats=["text", "text"]))
    au.narrative = [
        "Every projected number traces to a named economic assumption above; the full machine-"
        "readable audit is written alongside this workbook as driver_audit.json.",
    ]

    # ---- 9) Uncertainty (ONLY when Monte Carlo ran; G6 caption mandatory) ------
    mcd = getattr(res, "monte_carlo_distribution", None) or {}
    if mcd:
        un = m.section("Uncertainty", "Uncertainty (Monte Carlo)")
        ci = getattr(res, "confidence_interval", None) or {}
        un.metrics = [
            Metric("p10 / share", _num(ci.get("p10")), "usd"),
            Metric("p50 / share", _num(ci.get("p50")), "usd"),
            Metric("p90 / share", _num(ci.get("p90")), "usd"),
            Metric("Jensen gap (MC mean vs skeleton)", _num(mcd.get("jensen_gap")), "pct"),
        ]
        un.tables.append(Table("Driver Monte Carlo", ["Field", "Value"], [
            ["Draws", fmt_value(_num(mcd.get("n_draws")), "num0")],
            ["Effective (finite) draws", fmt_value(_num(mcd.get("n_effective")), "num0")],
            ["Skeleton intrinsic (headline)", fmt_value(_num(mcd.get("skeleton_iv")) or iv, mny0)],
            ["p10 / p50 / p90", f"{fmt_value(_num(ci.get('p10')), mny0)} / "
             f"{fmt_value(_num(ci.get('p50')), mny0)} / {fmt_value(_num(ci.get('p90')), mny0)}"],
        ], formats=["text", "text"]))
        un.narrative = [
            G6_CAPTION,
            "The headline intrinsic remains the deterministic skeleton; valuation is convex in the "
            "drivers, so the MC mean is Jensen-inflated and shown as dispersion, not as the point estimate.",
        ]

    # ---- 10) Historical + Investment Thesis (shared) --------------------------
    hist = (res.diagnostics.get("historical_financials", []) if hasattr(res, "diagnostics") else []) or []
    hsec = m.section("Historical", "Historical Financials")
    ht = _hist_table(hist, company_display_name(sc["ticker"], sc.get("company_name")), cur, fsrc)
    if ht:
        hsec.tables.append(ht)
    _add_ranking_context(m, sc["ticker"])

    # Investment thesis reads as a flowing essay (prose), not a bullet list.
    tsec = m.section("Investment Thesis", "Investment Thesis")
    tsec.prose = True
    tsec.narrative = _thesis_essay(sc, csym)
    return _localize_currency(m, cur)


def _as_sentence(text: str) -> str:
    s = str(text).strip()
    if s and s[-1] not in ".!?:":
        s += "."
    return s


def _thesis_essay(sc: dict, csym: str = "$") -> list[str]:
    """Compose the why-reasons and qualifiers into flowing essay paragraphs (no bullets)."""
    why = [_as_sentence(w) for w in (sc.get("why") or []) if str(w).strip()]
    quals = [_as_sentence(q) for q in (sc.get("qualifiers") or []) if str(q).strip()]
    paras: list[str] = []
    iv, price = _num(sc.get("intrinsic_value")), _num(sc.get("price"))
    upside = _num(sc.get("discount_to_intrinsic"))
    verdict = sc.get("verdict")
    if iv is not None and price is not None and upside is not None:
        lead = (f"At a market price of {csym}{price:,.2f}, the Fundamentals Fade model puts intrinsic value "
                f"at {csym}{iv:,.2f} per share, a "
                f"{abs(upside):.0%} {'discount' if upside >= 0 else 'premium'} to fair value")
        lead += f", which frames the {verdict} reading." if verdict else "."
        paras.append(lead)
    if why:
        paras.append(" ".join(why))
    if quals:
        paras.append("On the risks and qualifiers that temper this read: " + " ".join(quals))
    if not paras:
        paras.append("No thesis narrative is available for this name.")
    return paras


def _add_ranking_context(m: ReportModel, ticker: str) -> None:
    """How much weight this fade valuation's RANKING deserves, from the corrected-panel
    ladder (validation_evidence). The economic coherence of the model is always valid; its
    ranking edge over the frozen v1 baseline is family-scoped. Omitted if ledgers absent."""
    if not ve.available():
        return
    rc = m.section("Ranking Context", "Ranking Context")
    verdict = ve.verdict_for_ticker(ticker)
    if verdict is None:
        rc.narrative = [
            f"{ticker} is outside the corrected-panel research universe, so no family-level "
            "ranking verdict applies. Use this workbook as an auditable, economically coherent "
            "valuation -- not as a calibrated return-ranking signal.",
            ve.headline_claim(),
        ]
        return
    badge_good = verdict.tier == "value_orthogonal"
    rc.metrics = [
        Metric("Family", verdict.label, "text"),
        Metric("Ranking verdict", verdict.badge, "text", good=badge_good),
        Metric("Fade (v3) - v1, 3y rank IC", _icn(verdict.paired_v3_minus_v1_3y), "text"),
        Metric("Fade discount net of Book/Market, 3y IC", _icn(verdict.bm_resid_v3_3y), "text"),
    ]
    rc.narrative = [
        f"For {verdict.label}, {verdict.guidance}",
        ("On the corrected point-in-time panel the fade engine measurably out-ranks the frozen "
         "v1 baseline in this family." if verdict.v3_beats_v1 else
         "On the corrected panel the fade engine does not measurably out-rank v1 in this family "
         "(the engines tie)."),
        "This workbook's value is economic traceability -- every projected number maps to a named "
        "mechanism. The frozen v1 engine remains the default ranking and live-book model; the fade "
        "engine is the opt-in, most-explainable one. The ranking caveat above is family-scoped; the "
        "coherence of the valuation is not.",
        ve.headline_claim(),
    ]


def _icn(x) -> str:
    v = _num(x)
    return f"{v:+.3f}" if v is not None else "--"
