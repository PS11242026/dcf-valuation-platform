"""Expectation Gap candidate signals (U5 / Gate 5) -- engine_v2, diagnostic only.

Expectations-investing framing (Mauboussin): the market price implies a set of operating
expectations (the *reverse* DCF). The model carries its own forecast expectations. The
**Expectation Gap** is model-expectation MINUS market-implied-expectation, decomposed into
the three value drivers plus the aggregate value gap:

    growth_gap        model FCF growth        - market-implied FCF growth
    margin_gap        model economic margin   - market-implied margin
    reinvestment_gap  market-implied reinvestment drag, scale-normalized (model is the zero point)
    value_gap         model intrinsic / price - 1            (the aggregate; == the discount)

Every gap is oriented so that **higher == the market underestimates fundamentals == cheaper**,
i.e. the same polarity as the discount signal, so their Rank-ICs are directly comparable.

This module is *additive and diagnostic*: it reads quantities the frozen v1 baseline already
computed (``diagnostics["reverse_dcf_growth"]`` via ``_reverse_dcf_growth``) and is never wired
into valuation, portfolio construction, or the live conf_top10 book. ``_reverse_dcf_growth`` is
re-exported here so the market-implied expectations can also be solved directly from raw inputs.
"""
from __future__ import annotations

from typing import Any

from dcf_project.valuation.dcf import _reverse_dcf_growth

__all__ = [
    "_reverse_dcf_growth",
    "market_implied_expectations",
    "expectation_gaps",
    "candidate_signals",
    "GAP_KEYS",
]

GAP_KEYS = ("growth_gap", "margin_gap", "reinvestment_gap", "value_gap")


def _num(v: Any) -> float | None:
    """Coerce to a finite float or None (rejects NaN/inf/non-numeric/bool)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


_MKT_KEYS = (
    "market_implied_annual_fcf_growth",
    "market_implied_terminal_growth",
    "market_implied_margin",
    "market_implied_maintenance_capex_delta",
)
_EMPTY_MKT = {k: None for k in _MKT_KEYS}


def market_implied_expectations(result) -> dict[str, float | None]:
    """Market-implied operating expectations for one valuation (the reverse DCF).

    The heavy production path (``build_dcf_result``) stores a solved ``reverse_dcf_growth``
    block in diagnostics; if present that is used directly. The lean state-space path the
    engine_v2 / PIT backtest uses does NOT, so the reverse DCF is **solved here** via the
    promoted ``_reverse_dcf_growth`` from quantities the result already carries (price, shares,
    net debt, first projected FCF, model growth, quarterly WACC + terminal growth, horizon),
    plus the trailing revenue/margin surfaced in ``diagnostics["expectation_gap_inputs"]``.
    """
    diag = getattr(result, "diagnostics", {}) or {}
    pre = diag.get("reverse_dcf_growth") or {}
    if _num(pre.get("market_implied_annual_fcf_growth")) is not None:
        return {k: _num(pre.get(k)) for k in _MKT_KEYS}

    price = _num(getattr(result, "market_price", None))
    shares = _num(getattr(result, "diluted_shares_outstanding", None))
    net_debt = _num(getattr(result, "net_debt", None))
    projected = getattr(result, "projected_fcf", None) or []
    first_fcf = _num(projected[0]) if projected else None
    wacc = _num(getattr(result, "wacc", None))
    tg = _num(getattr(result, "terminal_growth_rate", None))
    model_growth = _num(getattr(result, "fcf_growth_rate", None))
    if (None in (price, shares, first_fcf, wacc, tg, model_growth)
            or shares <= 0 or first_fcf <= 0):
        return dict(_EMPTY_MKT)

    inp = diag.get("expectation_gap_inputs") or {}
    rev = _reverse_dcf_growth(
        price, shares, 0.0 if net_debt is None else net_debt, first_fcf,
        _num(inp.get("latest_margin")), _num(inp.get("latest_maintenance_capex")),
        _num(inp.get("latest_revenue")), model_growth,
        wacc / 4.0, tg / 4.0, len(projected),
    )
    return {k: _num(rev.get(k)) for k in _MKT_KEYS}


def expectation_gaps(result) -> dict[str, float | None]:
    """The four Expectation Gaps for one ``DCFResult`` (model expectation - market-implied).

    All gaps are oriented so higher == cheaper (market underestimates fundamentals), matching
    the discount's polarity. Any gap whose inputs are missing/degenerate is returned as None
    so it is simply dropped from the cross-section rather than poisoning the rank.
    """
    diag = getattr(result, "diagnostics", {}) or {}
    mkt = market_implied_expectations(result)
    inp = diag.get("expectation_gap_inputs") or {}

    model_growth = _num(getattr(result, "fcf_growth_rate", None))
    # Trailing FCF margin (matches the reverse-DCF's latest_margin); production diagnostics
    # name it economic_fcf_margin -- fall back to that when present.
    model_margin = _num(inp.get("latest_margin"))
    if model_margin is None:
        model_margin = _num(diag.get("economic_fcf_margin"))
    price = _num(getattr(result, "market_price", None))
    intrinsic = _num(getattr(result, "intrinsic_value_per_share", None))
    projected = getattr(result, "projected_fcf", None) or []
    first_fcf = _num(projected[0]) if projected else None

    growth_gap = (
        model_growth - mkt["market_implied_annual_fcf_growth"]
        if model_growth is not None and mkt["market_implied_annual_fcf_growth"] is not None
        else None
    )
    margin_gap = (
        model_margin - mkt["market_implied_margin"]
        if model_margin is not None and mkt["market_implied_margin"] is not None
        else None
    )
    # Positive maintenance-capex delta == market prices in MORE reinvestment drag than the
    # model assumes (model relatively optimistic) == cheap. Scale-normalize by first FCF.
    maint = mkt["market_implied_maintenance_capex_delta"]
    reinvestment_gap = (
        maint / abs(first_fcf)
        if maint is not None and first_fcf not in (None, 0.0)
        else None
    )
    value_gap = (
        intrinsic / price - 1.0
        if intrinsic is not None and price not in (None, 0.0) and price > 0
        else None
    )
    return {
        "growth_gap": growth_gap,
        "margin_gap": margin_gap,
        "reinvestment_gap": reinvestment_gap,
        "value_gap": value_gap,
    }


def candidate_signals(result, prefix: str = "eg_") -> dict[str, float | None]:
    """Expectation Gaps as namespaced candidate signals (e.g. ``eg_growth_gap``).

    Candidate signals only -- registered for backtesting, never consumed by construction.
    """
    return {f"{prefix}{k}": v for k, v in expectation_gaps(result).items()}
