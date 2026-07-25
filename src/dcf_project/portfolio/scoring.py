"""Per-name scoring -- ported verbatim from scripts/scorecard.py (validated ranking logic).

Reference = frozen validation output (no look-ahead) + the live snapshot; live per-name valuation
comes from the canonical DCFPipeline. NO scoring/ranking math is changed here -- only imports
(engine via dcf_project.core, paths via dcf_project.paths) and the removal of the terminal/Excel
display layer, which now lives in the reporting kit.
"""
from __future__ import annotations

import datetime as _dt
import json
from bisect import bisect_left

import numpy as np

from dcf_project.core import get_settings
from dcf_project.engine_v2 import select_pipeline
from dcf_project.paths import REPORTS as REPORT_DIR


# ----------------------------- reference layer -----------------------------
def _sector_reliability(b: dict) -> dict:
    """Evidence-based sector reliability from one final_institutional_validation.by_sector_family entry."""
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
            obs = json.load(open(REPORT_DIR / "final_universe_backtest.json"))["observations"]
            for o in obs:
                d = o.get("discount_to_intrinsic")
                if isinstance(d, (int, float)):
                    self.overall.append(d)
                    self.by_sector.setdefault(o["sector"], []).append(d)
        self.overall.sort()
        for s in self.by_sector:
            self.by_sector[s].sort()
        dec = json.load(open(REPORT_DIR / "validation_decile_portfolios.json"))
        self.decile_return = {y: dec["overall"][f"{y}y"]["decile_curve"]["obs_weighted"] for y in (3, 5)}
        inst = json.load(open(REPORT_DIR / "final_institutional_validation.json")).get("by_sector_family", {})
        self.sector_reliability = {f: _sector_reliability(b) for f, b in inst.items()}
        self.sector_ic_3y = {f: (b.get("3y", {}) or {}).get("ic_obs_weighted") for f, b in inst.items()}

    @staticmethod
    def _pct(sorted_vals: list[float], x: float) -> float:
        if not sorted_vals:
            return 50.0
        return 100.0 * bisect_left(sorted_vals, x) / len(sorted_vals)


_INDIA_REF_CACHE: dict | None = None


def india_reference() -> dict | None:
    """India relative-ranking reference: sorted v3 discounts (overall + per sector) from the
    Phase-B cross-section (``reports/research_v3_india/india_v3_reference.json``).

    Returns ``None`` when the Phase-B run has not produced the file yet, in which case the
    India verdict falls back to a neutral mid-rank (honest: 'no India reference yet'). Cached
    after first load. The US path never calls this, so US scoring is untouched.
    """
    global _INDIA_REF_CACHE
    if _INDIA_REF_CACHE is not None:
        return _INDIA_REF_CACHE or None
    path = REPORT_DIR / "research_v3_india" / "india_v3_reference.json"
    if not path.exists():
        _INDIA_REF_CACHE = {}
        return None
    try:
        names = json.load(open(path)).get("names", [])
    except (json.JSONDecodeError, OSError):
        _INDIA_REF_CACHE = {}
        return None
    overall: list[float] = []
    by_sector: dict[str, list[float]] = {}
    for n in names:
        d = n.get("discount")
        if isinstance(d, (int, float)):
            overall.append(d)
            by_sector.setdefault(n.get("sector", "unknown"), []).append(d)
    overall.sort()
    for s in by_sector:
        by_sector[s].sort()
    _INDIA_REF_CACHE = {"overall": overall, "by_sector": by_sector, "n": len(overall)}
    return _INDIA_REF_CACHE


LABELS = [(80, "Very Attractive"), (60, "Attractive"), (40, "Neutral"), (20, "Unattractive"), (0, "Very Unattractive")]
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


def _label(rank: float) -> str:
    for thr, name in LABELS:
        if rank >= thr:
            return name
    return "Very Unattractive"


def india_verdict_rank(er_rank: float, discount: float | None) -> float:
    """Cap an India name's RELATIVE rank by its ABSOLUTE discount-to-intrinsic so the verdict
    honours absolute value. A name trading above its own DCF intrinsic (negative discount) cannot
    read "Attractive" no matter how cheap it is relative to an all-expensive India universe:
    discount > -10% lets the relative rank stand; -30%..-10% caps the label at Neutral (<60);
    < -30% caps at Unattractive (<40). `expected_return_rank` keeps the uncapped relative rank."""
    if discount is None:
        return er_rank
    if discount < -0.30:
        return min(er_rank, 39.0)
    if discount < -0.10:
        return min(er_rank, 59.0)
    return er_rank


def sector_label(sector: str) -> str:
    return _SECTOR_LABEL.get(sector, (sector or "unknown").replace("_", " ").title())


def score_ticker(ticker: str, ref: Reference, settings=None) -> dict:
    # Engine selection honors settings.engine_version (default "v1" -> the frozen
    # canonical DCFPipeline, byte-identical to instantiating it directly).
    settings = settings or get_settings()
    res = select_pipeline(settings).run(ticker)
    country = str(getattr(settings, "country", "US") or "US").upper()
    is_india = country == "IN"
    # India uses its own GICS sector (diagnostics-stamped); the US family map does not cover it.
    sector = (res.diagnostics.get("sector") if is_india else None) or ref.family_of.get(
        ticker.upper(),
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

    india_ranking = False
    if is_india:
        # Rank an Indian name against the INDIA v3 cross-section (Phase B), NOT the US
        # distribution -- a US-relative percentile is meaningless for a market that trades at
        # structurally higher multiples. US-derived decile returns / sector-reliability do not
        # apply, so they are neutralized and the rank is flagged preliminary.
        iref = india_reference()
        if iref and discount is not None:
            er_rank = Reference._pct(iref["overall"], discount)
            sec_rank = Reference._pct(iref["by_sector"].get(sector, iref["overall"]), discount)
            india_ranking = True
        else:
            er_rank = sec_rank = 50.0
        decile_idx = min(9, int(er_rank // 10))
        hist_3y = hist_5y = None          # no India decile-return history (underpowered window)
        rel = None
        sec_ic = None
        tier = None
        confidence_band = "Preliminary (India)"
        conviction = 0.0
    else:
        er_rank = ref._pct(ref.overall, discount) if discount is not None else 50.0
        sec_rank = ref._pct(ref.by_sector.get(sector, ref.overall), discount) if discount is not None else 50.0
        decile_idx = min(9, int(er_rank // 10))
        hist_3y = ref.decile_return[3][decile_idx]
        hist_5y = ref.decile_return[5][decile_idx]

        rel = ref.sector_reliability.get(sector)
        sec_ic = ref.sector_ic_3y.get(sector)
        tier = rel["tier"] if rel else None
        confidence_band = rel["band"] if rel else "Insufficient"
        conviction = round(100.0 * rel["score"], 0) if rel else 0.0

    p10, p50, p90 = res.valuation_p10, res.valuation_p50, res.valuation_p90
    rel_width = abs((p90 - p10) / p50) if (p50 and p50 != 0) else 2.0
    forecast_confidence = float(np.clip(1.0 - rel_width / 1.5, 0.0, 1.0))

    optionality = res.valuation_regime == "optionality_hypergrowth"
    value_completeness = "incomplete_optionality" if optionality else "complete"
    # India: the VERDICT must honour absolute value -- a name trading above its own DCF intrinsic
    # cannot read "Attractive" however cheap it is RELATIVE to an all-expensive India universe.
    # `expected_return_rank` keeps the (uncapped) relative selection rank; see india_verdict_rank.
    verdict_rank = india_verdict_rank(er_rank, discount) if is_india else er_rank
    verdict = _label(verdict_rank)
    qualifiers = []
    if optionality:
        qualifiers.append("DCF INCOMPLETE (value completeness): market value is dominated by optionality the DCF "
                          "does not model -- treat the intrinsic value as a conservative FLOOR.")
    if is_india:
        if india_ranking:
            qualifiers.append("INDIA: the verdict reflects ABSOLUTE value (discount to this name's own DCF "
                              "intrinsic); `expected_return_rank` is the RELATIVE rank vs the India v3 "
                              "cross-section (Phase B, ~2023+ window, survivorship-optimistic) -- the "
                              "stock-selection signal. A name can rank well relatively yet still trade above "
                              "its intrinsic (capped below Attractive).")
        else:
            qualifiers.append("INDIA REFERENCE PENDING: no India v3 cross-section yet (run Phase B); the relative "
                              "rank is a neutral placeholder. The intrinsic value + absolute verdict are engine-honest.")
    elif rel is None:
        qualifiers.append(f"OUTSIDE VALIDATED UNIVERSE: '{sector}' is not a validated sector family; treat the rank cautiously.")
    elif tier == "weak":
        qualifiers.append(f"WEAK SECTOR TRACK RECORD: framework not reliably predictive in {sector}; down-weight this rank.")
    if forecast_confidence < 0.35:
        qualifiers.append("WIDE FORECAST DISPERSION: the per-name intrinsic estimate is uncertain (broad p10-p90 band).")

    shares = float(res.diluted_shares_outstanding) if res.diluted_shares_outstanding else None
    market_cap = (price * shares) if (price and shares) else None
    rank_basis = ("india_v3_cross_section" if india_ranking
                  else "india_reference_pending") if is_india else ref.rank_basis
    sc = {
        "ticker": ticker.upper(), "sector": sector, "rank_basis": rank_basis,
        "company_name": ticker.upper(),
        "expected_return_rank": round(er_rank, 0), "conviction": conviction,
        "prediction_confidence": confidence_band, "confidence_tier": tier,
        "value_completeness": value_completeness, "sector_reliability_evidence": rel,
        "discount_to_intrinsic": round(discount, 4) if discount is not None else None,
        "historical_validation_outcome": {
            "analog_decile": decile_idx + 1,
            "avg_subsequent_3y_annualized": round(hist_3y, 4) if hist_3y is not None else None,
            "avg_subsequent_5y_annualized": round(hist_5y, 4) if hist_5y is not None else None,
        },
        "verdict": verdict, "sector_adjusted_rank": round(sec_rank, 0),
        "price": round(price, 2) if price else None, "intrinsic_value": round(iv, 2),
        "market_cap": market_cap, "shares_outstanding": shares,
        "forecast_confidence": round(forecast_confidence, 2),
        "intrinsic_range_p10_p90": [round(float(p10), 2), round(float(p90), 2)] if (p10 and p90) else None,
        "sector_validation_3y_ic": round(sec_ic, 3) if sec_ic is not None else None,
        "assumptions": {
            "implied_annual_revenue_growth": round(growth, 4) if growth is not None else None,
            "terminal_operating_margin": round(margin, 4) if margin is not None else None,
            "terminal_growth": round(float(res.terminal_growth_rate), 4),
            "wacc": round(float(res.wacc), 4),
        },
        "qualifiers": qualifiers, "regime": res.valuation_regime,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "result": res,  # full DCFResult for the workbook builder (no recomputation)
    }
    sc["why"] = _why_reasons(sc)
    return sc


def _why_reasons(sc: dict) -> list[str]:
    d = sc["discount_to_intrinsic"]
    fc = sc.get("forecast_confidence") or 0.0
    a = sc["assumptions"]
    reasons: list[str] = []
    if d is None:
        reasons.append("Intrinsic value could not be compared to price.")
    elif d >= 0.5:
        reasons.append(f"Trades roughly {d:.0%} below its cash-flow-based intrinsic value -- a wide margin of safety.")
    elif d >= 0.15:
        reasons.append(f"Trades about {d:.0%} below estimated intrinsic value.")
    elif d > -0.15:
        reasons.append("Priced close to its cash-flow-based intrinsic value.")
    elif d > -0.5:
        reasons.append(f"Priced about {-d:.0%} above estimated intrinsic value.")
    else:
        reasons.append(f"Market price is roughly {-d:.0%} above the cash-flow-based intrinsic estimate.")
    sector_name = sector_label(sc.get("sector"))
    tier = sc.get("confidence_tier")
    if tier is None:
        reasons.append("Sits outside the validated universe, so the framework's reliability here is unproven.")
    elif tier == "strong":
        reasons.append(f"Sits in {sector_name}, one of the sectors where this framework has ranked stocks most reliably.")
    elif tier in ("solid", "emerging"):
        reasons.append(f"Sits in {sector_name}, a sector with a positive, validated ranking track record.")
    else:
        reasons.append(f"Sits in {sector_name}, where the framework's track record is weak -- read with caution.")
    if sc.get("regime") == "optionality_hypergrowth":
        reasons.append("Much of the market price reflects growth optionality the cash-flow model does not capture.")
    elif fc >= 0.55:
        reasons.append("Forecast inputs are relatively stable, so the intrinsic estimate is firmer.")
    elif fc < 0.35:
        reasons.append("Forecast inputs are uncertain, so the intrinsic estimate is less firm.")
    if len(reasons) < 4:
        m = a.get("terminal_operating_margin")
        g = a.get("implied_annual_revenue_growth")
        if m is not None and m >= 0.20:
            reasons.append(f"Durable economics: ~{m:.0%} long-run operating margin supports steady cash generation.")
        elif g is not None and g >= 0.15:
            reasons.append(f"Growth-led story: forecast assumes ~{g:.0%}/yr revenue growth.")
    return reasons[:4]
