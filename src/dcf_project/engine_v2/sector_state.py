"""Sector-State Engine v1 (P4B) -- engine_v2, diagnostic / candidate-signal only.

A per-sector ``sector_state`` score (companies inherit their sector's score), composed of four
z-scored components, oriented so **higher == more favorable sector**:

    sector_state = 0.35*valuation + 0.30*revisions + 0.20*momentum + 0.15*flows  (weights renormalized
                                                                                  over available components)

    valuation   home-grown sector aggregate earnings yield  (Sum EBIT / Sum market cap; higher == cheaper)
    revisions   home-grown sector aggregate earnings trend   (median YoY EBIT growth; proxy for revisions)
    momentum    sector SPDR-ETF excess return vs SPY          (3/6/12m blend)
    flows       ETF flow proxy                                (DEFERRED in v1 -> None; weight renormalized)

Every component is z-scored against the sector's **own expanding history up to the cutoff**
(point-in-time; no full-sample lookahead). The valuation/revisions inputs are computed by the
backtest runner from the *same* SEC fundamentals the engine already pulls -- so v1 needs no paid
vendor. True sell-side earnings revisions and ETF fund-flow data are a paid phase-2 upgrade.

This module is the math only (pure, network-free, unit-testable). The runner assembles the raw
per-(sector, cutoff) inputs and queries :class:`SectorPanel`. Nothing here is wired into valuation,
portfolio construction, or the live ``conf_top10`` book.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "WEIGHTS",
    "MIN_MEMBERS",
    "MIN_HISTORY",
    "MIN_WEIGHT_COVERAGE",
    "COMPONENT_KEYS",
    "aggregate_earnings_yield",
    "aggregate_revisions",
    "sector_momentum_asof",
    "SectorPanel",
    "combined_cross_sectional",
]

WEIGHTS = {"valuation": 0.35, "revisions": 0.30, "momentum": 0.20, "flows": 0.15}
COMPONENT_KEYS = ("valuation", "revisions", "momentum", "flows")
MIN_MEMBERS = 3            # min sector members for a home-grown valuation / revisions aggregate
MIN_HISTORY = 6            # min as-of history points before a component z-score is emitted
MIN_WEIGHT_COVERAGE = 0.5  # min summed weight of available components for a valid sector_state


def _num(v: Any) -> float | None:
    """Coerce to a finite float or None (rejects NaN/inf/non-numeric/bool)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _price_asof(series: pd.Series | None, d: date) -> float | None:
    """Last price on or before ``d`` (point-in-time)."""
    if series is None or len(series) == 0:
        return None
    s = series.loc[: pd.Timestamp(d)]
    return float(s.iloc[-1]) if len(s) else None


# --- raw component aggregation (cross-sectional, per sector x cutoff) ----------------------------

def aggregate_earnings_yield(members: list[tuple[Any, Any]]) -> float | None:
    """Sector aggregate earnings yield = Sum(earnings) / Sum(market_cap) over members.

    ``members`` is a list of ``(market_cap, ttm_earnings)``. Higher yield == cheaper sector.
    Requires >= MIN_MEMBERS members with positive market cap; returns None otherwise (thin sector).
    """
    caps, earns = [], []
    for cap, earn in members:
        c, e = _num(cap), _num(earn)
        if c is not None and c > 0 and e is not None:
            caps.append(c)
            earns.append(e)
    if len(caps) < MIN_MEMBERS:
        return None
    total_cap = sum(caps)
    return (sum(earns) / total_cap) if total_cap > 0 else None


def aggregate_revisions(growths: list[Any]) -> float | None:
    """Sector earnings-trend proxy = median YoY earnings growth across members.

    A free, point-in-time stand-in for sell-side earnings revisions. Requires >= MIN_MEMBERS
    finite member growths; returns None otherwise.
    """
    vals = [g for g in (_num(x) for x in growths) if g is not None]
    if len(vals) < MIN_MEMBERS:
        return None
    return float(median(vals))


def sector_momentum_asof(etf_series: pd.Series | None, spy_series: pd.Series | None, asof: date,
                         windows_days: tuple[int, ...] = (63, 126, 252)) -> float | None:
    """Sector ETF excess return vs SPY, blended over 3/6/12-month windows, as of ``asof``.

    ``etf_series`` / ``spy_series`` are date-indexed pandas price Series. Uses only prices on/before
    ``asof`` (point-in-time). Returns the mean available-window excess return, or None if the ETF
    has insufficient history as of ``asof`` (e.g. before its inception).
    """
    if etf_series is None or spy_series is None:
        return None
    excess: list[float] = []
    for w in windows_days:
        e_now = _price_asof(etf_series, asof)
        e_then = _price_asof(etf_series, asof - timedelta(days=w))
        s_now = _price_asof(spy_series, asof)
        s_then = _price_asof(spy_series, asof - timedelta(days=w))
        if None in (e_now, e_then, s_now, s_then) or e_then <= 0 or s_then <= 0:
            continue
        excess.append((e_now / e_then - 1.0) - (s_now / s_then - 1.0))
    return float(np.mean(excess)) if excess else None


# --- as-of z-scoring + weighted combination ------------------------------------------------------

def _zscore_asof(raw_by_date: dict[tuple[str, date], Any], sector: str, asof: date,
                 min_history: int = MIN_HISTORY) -> float | None:
    """Z-score the sector's raw value at ``asof`` against its own expanding history (dates <= asof).

    Point-in-time: only observations on/before ``asof`` enter the mean/std. Returns None if the
    value at ``asof`` is missing, history is too short, or the historical std is degenerate.
    """
    cur = _num(raw_by_date.get((sector, asof)))
    if cur is None:
        return None
    hist = [v for v in (_num(raw_by_date.get((s, d)))
                        for (s, d) in raw_by_date if s == sector and d <= asof) if v is not None]
    if len(hist) < min_history:
        return None
    arr = np.asarray(hist, dtype=float)
    sd = float(arr.std())
    if not np.isfinite(sd) or sd <= 0.0:
        return None
    return float((cur - float(arr.mean())) / sd)


class SectorPanel:
    """Holds raw per-(sector, cutoff) component inputs and computes as-of z-scored sector states.

    The runner populates the full panel (all sectors x all cutoffs) once, then queries
    :meth:`state` / :meth:`score` per row -- the expanding-window z-score for any cutoff only ever
    reads inputs dated on/before that cutoff, so a fully-populated panel introduces no lookahead.
    """

    def __init__(self) -> None:
        self.raw: dict[str, dict[tuple[str, date], float | None]] = {k: {} for k in COMPONENT_KEYS}

    def set_raw(self, sector: str, asof: date, *, valuation: Any = None, revisions: Any = None,
                momentum: Any = None, flows: Any = None) -> None:
        for key, val in (("valuation", valuation), ("revisions", revisions),
                         ("momentum", momentum), ("flows", flows)):
            if val is not None:
                self.raw[key][(sector, asof)] = _num(val)

    def state(self, sector: str, asof: date) -> dict[str, Any]:
        """Full as-of sector state: the blended ``sector_state`` plus each available component z."""
        comps: dict[str, float] = {}
        for key in COMPONENT_KEYS:
            z = _zscore_asof(self.raw[key], sector, asof)
            if z is not None:
                comps[key] = z
        coverage = sum(WEIGHTS[k] for k in comps)
        if coverage < MIN_WEIGHT_COVERAGE:
            sector_state = None
        else:
            sector_state = sum(WEIGHTS[k] * z for k, z in comps.items()) / coverage
        out: dict[str, Any] = {"sector_state": sector_state, "coverage": round(coverage, 3)}
        for key in COMPONENT_KEYS:
            out[f"sector_{key}"] = comps.get(key)
        return out

    def score(self, sector: str, asof: date) -> float | None:
        return self.state(sector, asof)["sector_state"]


def combined_cross_sectional(rows: list[dict[str, Any]], key_a: str, key_b: str, out_key: str,
                             group_key: str = "date") -> list[dict[str, Any]]:
    """Write ``out_key`` = z(key_a) + z(key_b), standardized **within each cross-section** (group).

    The combined signal that actually tests "does V2 + Sector beat V2 alone?": at each rebalance
    date, both inputs are z-scored across that date's names and summed (equal weight). Rows missing
    either input in a group with < 3 usable names get ``out_key = None`` (dropped downstream).
    """
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r.get(group_key)].append(r)

    def _z(vals: np.ndarray) -> np.ndarray:
        sd = float(vals.std())
        return (vals - float(vals.mean())) / sd if sd > 0 else np.zeros_like(vals)

    for grp in groups.values():
        usable = [r for r in grp if _num(r.get(key_a)) is not None and _num(r.get(key_b)) is not None]
        if len(usable) < 3:
            for r in grp:
                r[out_key] = None
            continue
        za = _z(np.asarray([_num(r[key_a]) for r in usable], dtype=float))
        zb = _z(np.asarray([_num(r[key_b]) for r in usable], dtype=float))
        combined = {id(r): float(za[i] + zb[i]) for i, r in enumerate(usable)}
        for r in grp:
            r[out_key] = combined.get(id(r))
    return rows
