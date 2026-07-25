"""valuation compare TICKER -- triangulate the three valuation engines.

Runs (fresh, pure) the generic textbook baseline, the canonical v1 state-space DCF and the
flagship Fundamentals Fade (v3) engine, then assembles ONE comparison workbook:

  1 Triangulation Summary -- football field (generic point, v1 grid whiskers, fade skeleton
    + grid whiskers, and a Monte-Carlo p10-p90 band ONLY when MC ran, G6-captioned) vs price
  2 Assumption Diff -- WACC, g8, growth, terminal-margin construct, plus the fade-only
    economics (phi, CAP, half-lives), PV(TV)/EV, intrinsic, upside, side by side
  3 Forecast Overlay -- v1 vs fade quarterly FCF
  4 Why They Differ -- rule-generated bullets from the actual numbers

Presentation only: reads already-computed engine outputs and reuses each engine's own
re-pricing (``report.sensitivity_grid`` for v1, ``report_fade.fade_iv_range`` for v3).
"""
from __future__ import annotations

from pathlib import Path

from dcf_project.paths import VALUATION
from dcf_project.reporting_kit import Image, Metric, NativeChart, ReportModel, Table, save_model, viz3d, xlsx
from dcf_project.reporting_kit.style import fmt_value, money_fmt
from dcf_project.research import validation_evidence as ve
from dcf_project.valuation.canonical import canonical_valuation
from dcf_project.valuation.generic import generic_valuation
from dcf_project.valuation.report import _CCY_LABEL, _currency_of, _figdir, _localize_currency, _num, sensitivity_grid
from dcf_project.valuation.report_fade import G6_CAPTION, _get, fade_iv_range

DASH = "--"


def _pct(x) -> str:
    v = _num(x)
    return fmt_value(v, "pct") if v is not None else DASH


def _usd(x, cur: str = "USD") -> str:
    v = _num(x)
    return fmt_value(v, money_fmt("usd", cur)) if v is not None else DASH


def _yrs(x) -> str:
    v = _num(x)
    return fmt_value(v, "yrs") if v is not None else DASH


def _upside(iv, price) -> float | None:
    iv, price = _num(iv), _num(price)
    return (iv / price - 1.0) if (iv and price) else None


def _v1_iv_range(res) -> tuple[float | None, float | None]:
    """Min/max intrinsic across v1's own (correct, Gordon-terminal) sensitivity grid."""
    fcf = list(getattr(res, "projected_fcf", []) or [])
    shares = _num(getattr(res, "diluted_shares_outstanding", None))
    net_debt = _num(getattr(res, "net_debt", None)) or 0.0
    wacc = _num(getattr(res, "wacc", None)) or 0.1
    g = _num(getattr(res, "terminal_growth_rate", None)) or 0.025
    tb_annual = (_num(getattr(res, "fcf_terminal_base", None)) or 0.0) * 4.0
    matrix, _, _ = sensitivity_grid(fcf, tb_annual, net_debt, shares, wacc, g)
    vals = [c for row in (matrix or []) for c in row if c is not None]
    return (min(vals) if vals else None, max(vals) if vals else None)


def _pv_tv_share(res) -> float | None:
    pv_tv = _num(getattr(res, "pv_terminal_value", None))
    ev = _num(getattr(res, "enterprise_value", None))
    return (pv_tv / ev) if (pv_tv and ev) else None


def _why_bullets(ticker, price, g_iv, v1_iv, f_iv, a1, da3, per3, cur: str = "USD") -> list[str]:
    def usd(x):
        return _usd(x, cur)
    bullets = [
        f"Intrinsic estimates: generic {usd(g_iv)}, canonical v1 {usd(v1_iv)}, fade v3 {usd(f_iv)} "
        f"(price {usd(price)}).",
        "Generic is a single-stage constant-margin DCF; canonical v1 is the frozen state-space "
        "ensemble; fade v3 fades growth / margin / RONIC over a 10-year window with identity-derived "
        "reinvestment (g = RONIC x reinvestment).",
    ]
    ivs = [v for v in (g_iv, v1_iv, f_iv) if _num(v)]
    if len(ivs) >= 2 and min(ivs) > 0 and max(ivs) / min(ivs) > 1.3:
        bullets.append(f"The engines disagree materially: the range {usd(min(ivs))}-{usd(max(ivs))} "
                       f"spans {max(ivs) / min(ivs) - 1.0:.0%}.")
    phi, cap = _num(per3.get("phi")), _num(per3.get("cap_years"))
    pvtv3 = _num(da3.get("pv_terminal_share_of_ev"))
    if v1_iv and f_iv and _num(f_iv) < _num(v1_iv) and phi is not None:
        bullets.append(
            f"Fade is the more conservative read here: it prices growth at a RONIC that decays toward "
            f"WACC (phi={phi:.2f}, CAP~{cap:.0f}y) and keeps {pvtv3:.0%} of enterprise value inside the "
            f"explicit window rather than the perpetuity." if pvtv3 is not None else
            f"Fade is the more conservative read here (phi={phi:.2f}, CAP~{cap:.0f}y).")
    w1 = _num(a1.get("wacc"))
    if w1 is not None:
        bullets.append(f"Canonical v1 discounts at WACC {_pct(w1)} with terminal growth "
                       f"{_pct(a1.get('terminal_growth'))}.")
    gb = da3.get("growth_binds_quarters") or []
    if gb:
        bullets.append(f"Fade reinvestment hit the sustainable-growth cap in {len(gb)} quarter(s) -- "
                       f"modeled growth was throttled to what RONIC can fund.")
    breach = da3.get("fcf_margin_breach_quarters") or []
    if breach:
        bullets.append(f"Fade's implied FCF margin left the company's historical p10-p90 band in "
                       f"{len(breach)} quarter(s) (coherence tripwire flagged in the audit).")
    return bullets


def build_compare_model(ticker: str, generic_baseline: dict | None, sc_v1: dict, sc_v3: dict,
                        folder: Path) -> ReportModel:
    figs = _figdir(folder)
    res1, res3 = sc_v1["result"], sc_v3["result"]
    cur = _currency_of(res3)            # USD (US) or INR (India); v3 stamps the reporting currency
    csym = _CCY_LABEL.get(cur, ("$", "EDGAR"))[0]

    def usd(x):
        return _usd(x, cur)

    a1 = sc_v1.get("assumptions") or {}
    a3 = sc_v3.get("assumptions") or {}
    da3 = (res3.diagnostics or {}).get("driver_audit", {}) if hasattr(res3, "diagnostics") else {}
    per3 = _get(da3, "persistence", default={})
    hl3 = _get(da3, "half_lives", default={})

    price = _num(sc_v1.get("price")) or _num(sc_v3.get("price")) or _num(getattr(res1, "market_price", None))
    g_iv = _num((generic_baseline or {}).get("intrinsic_value_per_share"))
    v1_iv = _num(sc_v1.get("intrinsic_value")) or _num(getattr(res1, "intrinsic_value_per_share", None))
    f_iv = _num(sc_v3.get("intrinsic_value")) or _num(getattr(res3, "intrinsic_value_per_share", None))

    m = ReportModel(
        title=f"{ticker} -- Valuation Triangulation",
        subtitle="Generic textbook vs canonical v1 state-space vs Fundamentals Fade (v3)",
        persona="initiation",
        meta={"Ticker": ticker, "As of": sc_v3.get("timestamp_utc") or sc_v1.get("timestamp_utc"),
              "Engines": "generic / v1 / v3"},
    )

    # ---- 1) Triangulation Summary --------------------------------------------
    s = m.section("Triangulation Summary", "Triangulation Summary")
    s.metrics = [
        Metric("Generic / sh", g_iv, "usd"),
        Metric("Canonical v1 / sh", v1_iv, "usd"),
        Metric("Fade v3 / sh", f_iv, "usd"),
        Metric("Market price", price, "usd"),
    ]
    rows = []
    if g_iv:
        rows.append({"label": "Generic", "low": g_iv, "high": g_iv, "point": g_iv})
    v1_lo, v1_hi = _v1_iv_range(res1)
    rows.append({"label": "Canonical v1", "low": v1_lo or v1_iv, "high": v1_hi or v1_iv, "point": v1_iv})
    f_lo, f_hi, f_pt = fade_iv_range(res3, ticker)
    rows.append({"label": "Fade v3 (skeleton)", "low": f_lo or f_pt, "high": f_hi or f_pt, "point": f_pt})
    mc = getattr(res3, "monte_carlo_distribution", None) or {}
    ci = getattr(res3, "confidence_interval", None) or {}
    has_mc = bool(mc) and _num(ci.get("p10")) is not None
    if has_mc:
        rows.append({"label": "Fade v3 (MC p10-p90)", "low": _num(ci.get("p10")),
                     "high": _num(ci.get("p90")), "point": _num(ci.get("p50"))})
    ff = viz3d.football_field(rows, price=price, out=figs / "triangulation.png",
                              title=f"{ticker} valuation triangulation ({csym}/sh)")
    if ff:
        s.images.append(Image("Valuation triangulation", ff))
    s.narrative = [
        "Each bar is one engine's intrinsic-value range (whiskers = its own sensitivity grid); the "
        "diamond is the point estimate and the dashed line is the market price.",
    ]
    if has_mc:
        s.narrative.append(G6_CAPTION)

    # ---- 2) Assumption Diff ---------------------------------------------------
    ad = m.section("Assumption Diff", "Assumption Diff")
    gb = generic_baseline or {}
    diff_rows = [
        ["WACC", _pct(gb.get("wacc")), _pct(a1.get("wacc")), _pct(a3.get("wacc"))],
        ["Terminal growth g8", _pct(gb.get("terminal_growth_rate")), _pct(a1.get("terminal_growth")),
         _pct(_num(a3.get("terminal_growth")) or _num(getattr(res3, "terminal_growth_rate", None)))],
        ["Revenue growth (base)", _pct(gb.get("revenue_cagr_5y")),
         _pct(a1.get("implied_annual_revenue_growth")), _pct(da3.get("g_base"))],
        ["Terminal / normalized op margin", _pct(gb.get("normalized_ebit_margin")),
         _pct(a1.get("terminal_operating_margin")), _pct(da3.get("margin_target_blend"))],
        ["Persistence phi", DASH, DASH, fmt_value(_num(per3.get("phi")), "num") if per3.get("phi") is not None else DASH],
        ["CAP", DASH, DASH, _yrs(per3.get("cap_years"))],
        ["Growth half-life", DASH, DASH, _yrs(hl3.get("growth_years"))],
        ["Margin half-life", DASH, DASH, _yrs(hl3.get("margin_years"))],
        ["PV(TV) / EV", DASH, _pct(_pv_tv_share(res1)), _pct(da3.get("pv_terminal_share_of_ev"))],
        ["Intrinsic / share", usd(g_iv), usd(v1_iv), usd(f_iv)],
        ["Upside / (downside)", _pct(_upside(g_iv, price)), _pct(_upside(v1_iv, price)),
         _pct(_upside(f_iv, price))],
    ]
    ad.tables.append(Table("Assumptions & outputs by engine",
                           ["Metric", "Generic", "Canonical v1", "Fade v3"], diff_rows,
                           formats=["text", "text", "text", "text"]))
    ad.narrative = [
        "Constructs differ by design: generic uses one constant margin + single-stage growth; v1 "
        "forecasts a state-space FCF path; v3 fades the economic primitives and DERIVES reinvestment. "
        "phi / CAP / half-lives are fade-only economics (no analogue in the other engines).",
    ]

    # ---- 3) Forecast Overlay (v1 vs fade quarterly FCF) -----------------------
    fo = m.section("Forecast Overlay", "Forecast Overlay")
    fcf1 = [_num(x) for x in (getattr(res1, "projected_fcf", []) or [])]
    fcf3 = [_num(x) for x in (getattr(res3, "projected_fcf", []) or [])]
    n = max(len(fcf1), len(fcf3))
    if n:
        overlay = [[i + 1,
                    (fcf1[i] / 1e6) if i < len(fcf1) and fcf1[i] is not None else None,
                    (fcf3[i] / 1e6) if i < len(fcf3) and fcf3[i] is not None else None]
                   for i in range(n)]
        otab = Table(f"Quarterly FCF ({csym}M): v1 vs fade", ["Q", "Canonical v1", "Fade v3"], overlay,
                     formats=["num0", "num", "num"], col_widths=[6, 14, 14])
        fo.charts.append(NativeChart("Quarterly FCF: v1 vs fade", "line", otab, x_col=0, y_cols=[1, 2],
                                     x_title="Quarter", y_title=f"FCF ({csym}M)"))
    fo.narrative = ["Canonical v1 forecasts 20 quarters; the fade engine forecasts 40 (10 years), so the "
                    "v1 series ends mid-chart -- the comparison is over the overlapping horizon."]

    # ---- 4) Why They Differ ---------------------------------------------------
    w = m.section("Why They Differ", "Why They Differ")
    w.narrative = _why_bullets(ticker, price, g_iv, v1_iv, f_iv, a1, da3, per3, cur)

    # ---- 5) Validation Context (corrected-panel evidence, scoped) -------------
    _add_validation_context(m, ticker)
    return _localize_currency(m, cur)


def _ic3(x) -> str:
    v = _num(x)
    return f"{v:+.3f}" if v is not None else DASH


def _add_validation_context(m: ReportModel, ticker: str) -> None:
    """Tell the reader how much weight the triangulation's RANKING deserves for this name,
    using the frozen corrected-panel ladder (validation_evidence). Coherence is always valid;
    ranking edge is family-scoped. Section is omitted only if the ledgers are absent."""
    if not ve.available():
        return
    vc = m.section("Validation Context", "Validation Context")
    verdict = ve.verdict_for_ticker(ticker)
    if verdict is None:
        vc.narrative = [
            f"{ticker} is outside the corrected-panel research universe, so no family-level "
            "ranking verdict applies. Read the triangulation as an economic-coherence comparison, "
            "not a return-ranking signal.",
            ve.headline_claim(),
        ]
        return
    vc.metrics = [
        Metric("Family", verdict.label, "text"),
        Metric("Ranking verdict", verdict.badge, "text"),
        Metric("v3 - v1 (3y IC)", _ic3(verdict.paired_v3_minus_v1_3y), "text"),
        Metric("v3 net of Book/Market (3y IC)", _ic3(verdict.bm_resid_v3_3y), "text"),
    ]
    vc.tables.append(Table(
        "Corrected-panel evidence for this family (3-year rank IC)",
        ["Metric", "Value", "NW t"],
        [["v3 minus paired v1 (out-ranks v1?)", _ic3(verdict.paired_v3_minus_v1_3y),
          fmt_value(_num(verdict.paired_nw_t_3y), "num") if verdict.paired_nw_t_3y is not None else DASH],
         ["v3 discount, net of Book/Market value factor", _ic3(verdict.bm_resid_v3_3y),
          fmt_value(_num(verdict.bm_resid_v3_t_3y), "num") if verdict.bm_resid_v3_t_3y is not None else DASH],
         ["v1 discount, net of Book/Market value factor", _ic3(verdict.bm_resid_v1_3y), DASH]],
        formats=["text", "text", "text"]))
    beats = verdict.v3_beats_v1
    vc.narrative = [
        f"For {verdict.label}, {verdict.guidance}",
        ("v3 measurably out-ranks the frozen v1 baseline in this family on the corrected panel."
         if beats else
         "v3 does not measurably out-rank v1 in this family on the corrected panel (the engines tie)."),
        "The frozen v1 engine remains the default ranking and live-book model; v3 is the opt-in, "
        "most-explainable engine. The economic-coherence comparison above is valid regardless of "
        "family; only the ranking interpretation is scoped.",
        ve.headline_claim(),
    ]


def run_compare(ticker: str, mc_draws: int = 0, country: str = "auto") -> Path:
    from dcf_project.data_sources.markets import resolve_country

    ticker = ticker.upper()
    folder = VALUATION / ticker / "comparison"
    folder.mkdir(parents=True, exist_ok=True)
    resolved = resolve_country(ticker, None if country == "auto" else country)
    generic_baseline = None
    # The generic textbook leg is a US-only path (USD presentation); skip it for India and
    # triangulate canonical v1 vs fade v3 only. (The v1/v3 legs render in the native currency.)
    if resolved == "US":
        try:
            generic_baseline = generic_valuation(ticker).get("baseline")
        except Exception as exc:  # noqa: BLE001  -- generic is a nice-to-have leg, never block the compare
            print(f"  [compare] generic leg unavailable: {type(exc).__name__}: {exc}")
            generic_baseline = None
    sc_v1 = canonical_valuation(ticker, engine="v1", country=country)
    sc_v3 = canonical_valuation(ticker, engine="v3", mc_draws=mc_draws, country=country)
    model = build_compare_model(ticker, generic_baseline, sc_v1, sc_v3, folder)
    out = xlsx.render(model, folder / "comparison.xlsx")
    save_model(model, folder)
    print(f"wrote {out}")
    return out
