"""Research scorecard generator -- a ranking-first decision-support layer over the DCF.

Aligned with what the validation demonstrates: RANKING skill, not binary buy/sell. The
Expected Return Rank answers the actionable question -- "how attractive is this name vs the
other stocks I can buy TODAY" -- by ranking discount_to_intrinsic against a LIVE current
cross-section of the investable universe (reports/_live_universe_snapshot.json, built by
build_live_snapshot.py). The historical decile mapping is preserved separately as a
backward-looking analog ("stocks that have historically scored in this band").

FOUR PRIMARY NUMBERS (institutional-style):
  1. Expected Return Rank (0-100)   -- live cross-sectional percentile of discount
  2. Conviction (0-100)             -- trust in the rank (sector predictiveness + forecast stability)
  3. Discount to Intrinsic (%)      -- the DCF valuation gap
  4. Historical Validation Outcome  -- realized fwd return of the matching validation decile
Everything else (growth/margin/WACC/terminal assumptions, p10-p90 range) is supporting drill-down.

Reference = frozen validation output (no look-ahead) + the live snapshot. Live per-name
valuation comes from DCFPipeline. No production valuation code is modified.
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.pipeline import DCFPipeline  # noqa: E402

try:  # Excel export is optional/decoupled; absence must never break the scorecard
    from scripts.excel_export import export_valuation_report
except Exception:  # pragma: no cover
    export_valuation_report = None

REPORT_DIR = ROOT / "reports"


# ----------------------------- reference layer -----------------------------
def _sector_reliability(b: dict) -> dict:
    """Evidence-based sector reliability from one `final_institutional_validation.by_sector_family` entry.

    Blends the FULL validated evidence the prior framework ignored: 3Y & 5Y IC (both equal- and
    obs-weighted, averaged -- robust to the equal/obs artifact), a significant (p<0.05) positive 3Y
    Spearman, the 3Y long-short Q5-Q1 spread, and the 3Y hit rate -> score in [0,1] + a tier. A sector is
    'weak' ONLY if it lacks a significant positive Spearman AND has a non-positive long-short spread
    (genuinely non-predictive); positive-but-moderate sectors are 'solid'/'emerging', not 'weak'.
    """
    a3, a5 = b.get("3y", {}) or {}, b.get("5y", {}) or {}

    def f(d, k):
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    ics = [x for x in (f(a3, "ic_equal_weight"), f(a3, "ic_obs_weighted"),
                       f(a5, "ic_equal_weight"), f(a5, "ic_obs_weighted")) if x is not None]
    ic_blend = sum(ics) / len(ics) if ics else 0.0
    spear = f(a3, "spearman_pooled") or 0.0
    spear_p = f(a3, "spearman_p")
    sig = spear_p is not None and spear_p < 0.05 and spear > 0
    ls = f(a3, "long_short_q5_minus_q1") or 0.0
    hit = f(a3, "hit_rate")
    hit = hit if hit is not None else 0.5

    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    score = (0.35 * clip(ic_blend / 0.20, 0, 1)
             + 0.30 * (clip(spear / 0.35, 0, 1) if sig else 0.0)
             + 0.25 * clip(ls / 0.25, 0, 1)
             + 0.10 * clip((hit - 0.5) / 0.12, 0, 1))
    if sig and score >= 0.70:
        tier, band = "strong", "High"
    elif sig and ls > 0 and score >= 0.40:
        tier, band = "solid", "Medium"
    elif (sig or ls > 0) and score >= 0.22:
        tier, band = "emerging", "Medium"
    else:
        tier, band = "weak", "Low"
    return {"score": round(score, 3), "tier": tier, "band": band, "ic_blend": round(ic_blend, 3),
            "spearman_3y": round(spear, 3), "spearman_significant": bool(sig),
            "long_short_3y": round(ls, 3), "hit_rate_3y": round(hit, 3)}


class Reference:
    def __init__(self) -> None:
        self.family_of = json.load(open(REPORT_DIR / "_universe_tickers.json"))["family_of"]
        # LIVE current cross-section of discounts (preferred ranking basis)
        snap_path = REPORT_DIR / "_live_universe_snapshot.json"
        self.rank_basis = "historical_fallback"
        self.overall: list[float] = []
        self.by_sector: dict[str, list[float]] = {}
        if snap_path.exists():
            snap = json.load(open(snap_path)).get("snapshot", {})
            if snap:
                self.rank_basis = "live_current_universe"
                for t, v in snap.items():
                    d = v.get("discount")
                    if isinstance(d, (int, float)):
                        self.overall.append(d)
                        self.by_sector.setdefault(v.get("sector", "unknown"), []).append(d)
        if self.rank_basis == "historical_fallback":
            # fallback to the historical PIT distribution (clearly flagged in output)
            obs = json.load(open(REPORT_DIR / "final_universe_backtest.json"))["observations"]
            for o in obs:
                d = o.get("discount_to_intrinsic")
                if isinstance(d, (int, float)):
                    self.overall.append(d)
                    self.by_sector.setdefault(o["sector"], []).append(d)
        self.overall.sort()
        for s in self.by_sector:
            self.by_sector[s].sort()
        # historical decile outcomes (validation)
        dec = json.load(open(REPORT_DIR / "validation_decile_portfolios.json"))
        self.decile_return = {y: dec["overall"][f"{y}y"]["decile_curve"]["obs_weighted"] for y in (3, 5)}
        # EVIDENCE-BASED sector reliability: derived from the FULL validated per-sector evidence
        # (3Y/5Y IC, significant Spearman, long-short Q5-Q1, hit rate) -- not a single 3Y obs-IC point.
        inst = json.load(open(REPORT_DIR / "final_institutional_validation.json")).get("by_sector_family", {})
        self.sector_reliability = {f: _sector_reliability(b) for f, b in inst.items()}
        self.sector_ic_3y = {f: (b.get("3y", {}) or {}).get("ic_obs_weighted") for f, b in inst.items()}

    @staticmethod
    def _pct(sorted_vals: list[float], x: float) -> float:
        """Percentile of x including itself in the cross-section."""
        if not sorted_vals:
            return 50.0
        return 100.0 * bisect_left(sorted_vals, x) / len(sorted_vals)


# ----------------------------- scoring -----------------------------
LABELS = [(80, "Very Attractive"), (60, "Attractive"), (40, "Neutral"), (20, "Unattractive"), (0, "Very Unattractive")]


def _label(rank: float) -> str:
    for thr, name in LABELS:
        if rank >= thr:
            return name
    return "Very Unattractive"


def score_ticker(ticker: str, ref: Reference, settings=None) -> dict:
    res = DCFPipeline(settings or get_settings()).run(ticker)
    sector = ref.family_of.get(ticker.upper(),
                               res.diagnostics.get("company_classification", {}).get("resolved_company_type", "unknown"))
    diag = res.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
    iv = float(res.intrinsic_value_per_share)
    iv_dcf = float(diag) if isinstance(diag, (int, float)) else iv
    price = float(res.market_price) if res.market_price else None
    discount = (iv_dcf / price - 1.0) if price else None

    ff = res.forecast_financials or []
    last = ff[-1] if ff else {}
    rev = [float(x.get("revenue") or 0) for x in ff]
    growth = ((rev[-1] / rev[0]) ** (4.0 / (len(rev) - 1)) - 1.0) if len(rev) > 4 and rev[0] > 0 else None
    margin = (float(last.get("ebit") or 0) / float(last.get("revenue") or 1)) if last else None

    # live cross-sectional ranks (insert this name's discount into the distribution)
    er_rank = ref._pct(ref.overall, discount) if discount is not None else 50.0
    sec_rank = ref._pct(ref.by_sector.get(sector, ref.overall), discount) if discount is not None else 50.0

    # historical analog: rank band -> validation decile -> realized fwd return
    decile_idx = min(9, int(er_rank // 10))
    hist_3y = ref.decile_return[3][decile_idx]
    hist_5y = ref.decile_return[5][decile_idx]

    # conviction / Prediction Confidence -- driven ENTIRELY by the empirical validated sector track record
    # (blended IC + significant Spearman + long-short spread + hit rate). No per-name forecast-dispersion
    # term and no optionality penalty fold into the headline confidence anymore.
    rel = ref.sector_reliability.get(sector)
    sec_ic = ref.sector_ic_3y.get(sector)
    tier = rel["tier"] if rel else None
    confidence_band = rel["band"] if rel else "Insufficient"
    conviction = round(100.0 * rel["score"], 0) if rel else 0.0

    # forecast dispersion -> reported separately as a per-name value caveat (NOT part of confidence)
    p10, p50, p90 = res.valuation_p10, res.valuation_p50, res.valuation_p90
    rel_width = abs((p90 - p10) / p50) if (p50 and p50 != 0) else 2.0
    forecast_confidence = float(np.clip(1.0 - rel_width / 1.5, 0.0, 1.0))

    # optionality -> a separate VALUE-COMPLETENESS flag, NOT a ranking-confidence penalty
    optionality = res.valuation_regime == "optionality_hypergrowth"
    value_completeness = "incomplete_optionality" if optionality else "complete"

    verdict = _label(er_rank)
    qualifiers = []
    if optionality:
        qualifiers.append("DCF INCOMPLETE (value completeness): market value is dominated by optionality the DCF "
                          "does not model -- treat the intrinsic value as a conservative FLOOR. This is a value "
                          "caveat only; it does NOT lower the ranking confidence, which reflects the sector's "
                          "validated track record.")
    if rel is None:
        # ticker resolves to a type outside the 7 validated sector families (e.g. AAPL/MSFT,
        # which are not in the 506 universe) -> no validation evidence exists for it.
        qualifiers.append(f"OUTSIDE VALIDATED UNIVERSE: '{sector}' is not one of the validated sector "
                          f"families, so no historical reliability is available; treat the rank cautiously.")
    elif tier == "weak":
        qualifiers.append(f"WEAK SECTOR TRACK RECORD: the framework is not reliably predictive in {sector} "
                          f"(no significant rank correlation and non-positive long-short spread); down-weight this rank.")
    if forecast_confidence < 0.35:
        qualifiers.append("WIDE FORECAST DISPERSION: the per-name intrinsic estimate is uncertain (broad p10-p90 "
                          "band) -- a value caveat, separate from the sector's ranking reliability.")

    import datetime as _dt
    shares = float(res.diluted_shares_outstanding) if res.diluted_shares_outstanding else None
    market_cap = (price * shares) if (price and shares) else None
    sc = {
        "ticker": ticker.upper(), "sector": sector, "rank_basis": ref.rank_basis,
        "company_name": ticker.upper(),  # no name source in the pipeline; ticker used as label
        # --- four primary numbers ---
        "expected_return_rank": round(er_rank, 0),
        "conviction": conviction,
        "prediction_confidence": confidence_band,        # High/Medium/Low from validated sector tier
        "confidence_tier": tier,                          # strong/solid/emerging/weak (or None outside universe)
        "value_completeness": value_completeness,         # complete | incomplete_optionality (separate axis)
        "sector_reliability_evidence": rel,               # the IC/Spearman/long-short/hit the band is derived from
        "discount_to_intrinsic": round(discount, 4) if discount is not None else None,
        "historical_validation_outcome": {
            "analog_decile": decile_idx + 1,
            "avg_subsequent_3y_annualized": round(hist_3y, 4),
            "avg_subsequent_5y_annualized": round(hist_5y, 4),
        },
        "verdict": verdict,
        # --- supporting detail ---
        "sector_adjusted_rank": round(sec_rank, 0),
        "price": round(price, 2) if price else None,
        "intrinsic_value": round(iv, 2),
        "market_cap": market_cap,
        "shares_outstanding": shares,
        "forecast_confidence": round(forecast_confidence, 2),
        "intrinsic_range_p10_p90": [round(float(p10), 2), round(float(p90), 2)] if (p10 and p90) else None,
        "sector_validation_3y_ic": round(sec_ic, 3) if sec_ic is not None else None,
        "assumptions": {
            "implied_annual_revenue_growth": round(growth, 4) if growth is not None else None,
            "terminal_operating_margin": round(margin, 4) if margin is not None else None,
            "terminal_growth": round(float(res.terminal_growth_rate), 4),
            "wacc": round(float(res.wacc), 4),
        },
        "qualifiers": qualifiers,
        "regime": res.valuation_regime,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        # --- raw model payload for Excel export (reused DCFResult fields, no recomputation) ---
        "model": {
            "historical_financials": res.diagnostics.get("historical_financials", []),
            "forecast_financials": [dict(r) for r in (res.forecast_financials or [])],
            "discount_schedule": res.diagnostics.get("discount_schedule", []),
            # data passthrough: ensemble path weights so the Excel forecast FCF can be a linked formula.
            "ensemble_weights": dict(res.ensemble_weights or {}),
            # data passthrough (no logic/calculation change): terminal-growth provenance for the
            # Excel TERMINAL VALUE ASSUMPTIONS section -- copied verbatim from the DCFResult diagnostics.
            "terminal_growth_provenance": (res.diagnostics.get("terminal_assumptions") or {}).get("terminal_growth_policy") or {},
            # data passthrough: WACC provider name for the Excel WACC label.
            "wacc_provider": ((res.diagnostics.get("wacc_decomposition") or {}).get("provider")
                              if not (res.diagnostics.get("wacc_decomposition") or {}).get("fallback_used")
                              else "internal CAPM") or "internal CAPM",
            "dcf_steps": {
                "explicit_pv": float(res.explicit_pv),
                "pv_projected_fcf": float(res.pv_projected_fcf),
                "terminal_value": float(res.terminal_value),
                "pv_terminal_value": float(res.pv_terminal_value),
                "fcf_terminal_base_quarterly": float(res.fcf_terminal_base),
                "terminal_base_annual": float((res.diagnostics.get("terminal_assumptions") or {}).get("terminal_base_annual") or res.fcf_terminal_base * 4),
                "enterprise_value": float(res.enterprise_value),
                "net_debt": float(res.net_debt),
                "equity_value": float(res.equity_value),
                "shares_outstanding": shares,
                "wacc": float(res.wacc),
                "terminal_growth_rate": float(res.terminal_growth_rate),
                "projected_fcf": [float(x) for x in (res.projected_fcf or [])],
                "intrinsic_value_per_share": float(iv),
            },
        },
    }
    # generate the WHY (investment thesis) once -> reused by both the terminal card and Excel
    sc["why"] = _why_reasons(sc)
    # limitations = qualifiers + optionality warning text (reused by Excel LIMITATIONS section)
    limitations = list(qualifiers)
    if optionality:
        limitations.append("A large share of this company's market value rests on growth optionality this "
                           "cash-flow model does not capture. Treat the intrinsic value as a conservative floor.")
    sc["limitations"] = limitations
    return sc


# ----------------------------- display layer (presentation only) -----------------------------
# These helpers derive human-readable strings from the existing scorecard object. They add NO new
# calculations -- they only re-present fields that score_ticker already produced. All internal fields
# (expected_return_rank, sector_adjusted_rank, historical_validation_outcome, forecast_confidence,
# intrinsic_range_p10_p90, conviction, ...) remain in the object for model use / Excel export.

_SECTOR_LABEL = {
    "software_saas_infra": "Software / SaaS",
    "internet_platforms_marketplaces": "Internet Platform / Marketplace",
    "semiconductors": "Semiconductors",
    "semiconductor_equipment": "Semiconductor Equipment",
    "consumer_staples": "Consumer Staples",
    "consumer_discretionary": "Consumer Discretionary",
    "industrials_assetlight_services": "Industrials / Asset-Light Services",
}

_STAGE_BY_REGIME = {
    "optionality_hypergrowth": "Optionality / Hypergrowth",
    "mature_compounder": "Mature Compounder",
    "platform_growth": "Platform Growth",
    "consumer_defensive": "Defensive Compounder",
}


# --- defensive formatting helpers: render missing values as "N/A" instead of crashing ---
def _money(x) -> str:
    return "N/A" if x is None else f"${x:,.2f}"


def _pctf(x, signed: bool = False, dp: int = 0) -> str:
    if x is None:
        return "N/A"
    return f"{x:+.{dp}%}" if signed else f"{x:.{dp}%}"


def _company_type(sc: dict) -> str:
    sector = sc.get("sector") or "unknown"
    return _SECTOR_LABEL.get(sector, sector.replace("_", " ").title())


def _company_stage(sc: dict) -> str:
    stage = _STAGE_BY_REGIME.get(sc.get("regime") or "")
    if not stage:
        g = (sc.get("assumptions", {}).get("implied_annual_revenue_growth") or 0.0)
        stage = "High Growth" if g >= 0.15 else "Growth" if g >= 0.08 else "Mature"
    return stage


def _classification(sc: dict) -> str:
    return f"{_company_type(sc)}  |  {_company_stage(sc)}"


def _confidence_band(sc: dict) -> str:
    # band is the evidence-based tier mapping computed in score_ticker; fall back to the numeric
    # conviction only for legacy cards that lack the field.
    band = sc.get("prediction_confidence")
    if band:
        return band
    c = sc.get("conviction") or 0
    return "High" if c >= 70 else "Medium" if c >= 45 else "Low"


def _why_reasons(sc: dict) -> list[str]:
    d = sc["discount_to_intrinsic"]
    ic = sc.get("sector_validation_3y_ic")
    fc = sc.get("forecast_confidence") or 0.0
    a = sc["assumptions"]
    reasons: list[str] = []

    # 1) valuation gap
    if d is None:
        reasons.append("Intrinsic value could not be compared to price.")
    elif d >= 0.5:
        reasons.append(f"Trades roughly {d:.0%} below its cash-flow-based intrinsic value -- a wide margin of safety (large discount).")
    elif d >= 0.15:
        reasons.append(f"Trades about {d:.0%} below estimated intrinsic value.")
    elif d > -0.15:
        reasons.append("Priced close to its cash-flow-based intrinsic value.")
    elif d > -0.5:
        reasons.append(f"Priced about {-d:.0%} above estimated intrinsic value.")
    else:
        reasons.append(f"Market price is roughly {-d:.0%} above the cash-flow-based intrinsic estimate.")

    # 2) sector reliability of the framework -- evidence-based tier (significant Spearman + long-short, not one IC)
    sector_name = _company_type(sc)
    tier = sc.get("confidence_tier")
    if tier is None:
        reasons.append("Sits outside the validated universe, so the framework's reliability for this "
                       "company is unproven -- read with caution.")
    elif tier == "strong":
        reasons.append(f"Sits in {sector_name}, one of the sectors where this framework has ranked stocks "
                       f"most reliably in validation.")
    elif tier in ("solid", "emerging"):
        reasons.append(f"Sits in {sector_name}, a sector with a positive, validated ranking track record "
                       f"(statistically significant rank correlation and a positive long-short spread).")
    else:
        reasons.append(f"Sits in {sector_name}, where the framework's track record is weak -- read with caution.")

    # 3) optionality OR forecast stability
    if sc.get("regime") == "optionality_hypergrowth":
        reasons.append("Much of the market's price reflects growth optionality the cash-flow model does not capture.")
    elif fc >= 0.55:
        reasons.append("Forecast inputs are relatively stable, so the intrinsic estimate is firmer.")
    elif fc < 0.35:
        reasons.append("Forecast inputs are uncertain, so the intrinsic estimate is less firm.")

    # 4) a business-quality note (only if room remains)
    if len(reasons) < 4:
        m = a.get("terminal_operating_margin")
        g = a.get("implied_annual_revenue_growth")
        if m is not None and m >= 0.20:
            reasons.append(f"Durable economics: ~{m:.0%} long-run operating margin supports steady cash generation.")
        elif g is not None and g >= 0.15:
            reasons.append(f"Growth-led story: forecast assumes ~{g:.0%}/yr revenue growth.")
    return reasons[:4]


def render(sc: dict) -> str:
    d = sc["discount_to_intrinsic"]
    a = sc["assumptions"]
    optionality = sc.get("regime") == "optionality_hypergrowth"
    L = []
    L.append("============================================================")
    L.append(f"  {sc['ticker']}")
    L.append(f"  {_classification(sc)}")
    L.append("============================================================")
    L.append(f"  VERDICT:                {sc['verdict']}")
    L.append(f"  PREDICTION CONFIDENCE:  {_confidence_band(sc)}")
    L.append("")
    L.append("  VALUATION")
    L.append(f"    Market Price      {_money(sc.get('price'))}")
    L.append(f"    Intrinsic Value   {_money(sc.get('intrinsic_value'))}")
    direction = "Upside" if (d is not None and d >= 0) else "Downside"
    L.append(f"    {direction:<9}         {_pctf(d, signed=True)}")
    L.append("")
    L.append("  WHY")
    for r in (sc.get("why") or _why_reasons(sc)):
        L.append(f"    - {r}")
    L.append("")
    L.append("  KEY ASSUMPTIONS")
    L.append(f"    Revenue Growth    {('~' + _pctf(a.get('implied_annual_revenue_growth')) + '/yr') if a.get('implied_annual_revenue_growth') is not None else 'N/A'}")
    L.append(f"    Terminal Margin   {('~' + _pctf(a.get('terminal_operating_margin'))) if a.get('terminal_operating_margin') is not None else 'N/A'}")
    L.append(f"    Terminal Growth   {_pctf(a.get('terminal_growth'), dp=1)}")
    L.append(f"    WACC              {_pctf(a.get('wacc'), dp=1)}")
    if optionality:
        L.append("")
        L.append("  OPTIONALITY WARNING")
        L.append("    A large share of this company's market value rests on growth")
        L.append("    optionality this cash-flow model does not capture. Treat the")
        L.append("    intrinsic value as a conservative floor, not a price target.")
    L.append("")
    L.append("  NOTES")
    L.append(f"    - Prediction confidence ({_confidence_band(sc)}) reflects how reliable this framework")
    L.append("      has been for this type of company, not certainty about the outcome.")
    L.append("    - Designed for long-term (3-5 year) investing, not short-term trading.")
    L.append("    - This is research output, not investment advice.")
    return "\n".join(L)


def diagnostic_report(sc: dict) -> str:
    """Pre-render field dump -- surfaces every value (incl. None) before formatting, so missing
    fields are visible rather than crashing the card."""
    a = sc.get("assumptions", {})
    d = sc.get("discount_to_intrinsic")
    rows = [
        ("company_type", _company_type(sc)),
        ("company_stage", _company_stage(sc)),
        ("verdict", sc.get("verdict")),
        ("confidence", _confidence_band(sc)),
        ("revenue_growth", a.get("implied_annual_revenue_growth")),
        ("terminal_margin", a.get("terminal_operating_margin")),
        ("terminal_growth", a.get("terminal_growth")),
        ("wacc", a.get("wacc")),
        ("intrinsic_value", sc.get("intrinsic_value")),
        ("market_price", sc.get("price")),
        ("upside_pct", _pctf(d, signed=True) if d is not None else None),
    ]
    out = [f"--- DIAGNOSTIC: {sc.get('ticker')} (sector='{sc.get('sector')}', "
           f"sector_validation_3y_ic={sc.get('sector_validation_3y_ic')}) ---"]
    for k, v in rows:
        flag = "  <- None" if v is None else ""
        out.append(f"    {k:18}= {v}{flag}")
    return "\n".join(out)


def main() -> None:
    import argparse
    from dataclasses import replace as _replace

    p = argparse.ArgumentParser(
        prog="scorecard",
        description="DCF research scorecard (ranking-first decision support).")
    p.add_argument("tickers", nargs="*", help="tickers to score (default: a demo set)")
    p.add_argument("--diag", action="store_true", help="also print the diagnostic field dump")
    p.add_argument("--forecast-mode", choices=["state_space", "consensus_blend"],
                   help="override the model's forecast_mode")
    p.add_argument("--wacc-source", choices=["auto", "valueinvesting", "internal"],
                   help="override the WACC provider source")
    args = p.parse_args()

    diag = args.diag
    demo = [t.upper() for t in args.tickers] or ["PYPL", "WEX", "GPN", "KO", "NOW", "AMZN"]
    # only construct an override Settings when a flag was actually supplied (else use production defaults)
    overrides = {}
    if args.forecast_mode:
        overrides["forecast_mode"] = args.forecast_mode
    if args.wacc_source:
        overrides["wacc_source"] = args.wacc_source
    settings = _replace(get_settings(), **overrides) if overrides else None
    if overrides:
        print(f"[settings override] {overrides}\n")

    ref = Reference()
    if ref.rank_basis != "live_current_universe":
        print("[warning] live universe snapshot not found -- ranking against HISTORICAL distribution. "
              "Run scripts/build_live_snapshot.py first for live cross-sectional ranks.\n")
    cards = []
    for t in demo:
        try:
            sc = score_ticker(t, ref, settings)
            cards.append(sc)
            if diag:
                print(diagnostic_report(sc) + "\n")
            print(render(sc) + "\n")
        except Exception as exc:
            import traceback
            print(f"{t}: scorecard failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print()
            continue
        # Excel export is fully decoupled: a failure here must NOT break the terminal scorecard.
        if export_valuation_report is None:
            print("  [excel] export unavailable (openpyxl not installed) -- scorecard still valid\n")
        else:
            try:
                path = export_valuation_report(sc, REPORT_DIR)
                print(f"  [excel] wrote {path.relative_to(ROOT)}\n")
            except Exception as exc:
                print(f"  [excel] export skipped ({type(exc).__name__}: {exc}) -- scorecard still valid\n")
    # the scorecards_demo.json keeps the full record incl. the model payload (drop nothing)
    (REPORT_DIR / "scorecards_demo.json").write_text(json.dumps(cards, indent=2, default=str), encoding="utf-8")
    print(f"wrote reports/scorecards_demo.json (rank_basis={ref.rank_basis})")


if __name__ == "__main__":
    main()
