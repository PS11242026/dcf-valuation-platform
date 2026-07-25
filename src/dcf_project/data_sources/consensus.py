from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dcf_project.config import Settings
from dcf_project.models.priors import prior_for


@dataclass(frozen=True)
class ForwardConsensus:
    provider: str
    revenue_growth_annual: float
    ebit_margin: float | None
    fcf_margin: float | None
    retrieved_at: datetime
    fallback_used: bool
    source_notes: list[str]


def resolve_forward_consensus(
    ticker: str,
    history: pd.DataFrame,
    settings: Settings,
) -> ForwardConsensus:
    if settings.forecast_mode == "state_space":
        return _internal_consensus(ticker, history, settings, fallback_used=True, note="forecast_mode=state_space")
    try:
        yahoo = yahoo_forward_consensus(ticker, history, settings)
        if yahoo is not None:
            return yahoo
    except Exception:
        pass
    return _internal_consensus(ticker, history, settings, fallback_used=True, note="provider_fallback=archetype_internal")


def yahoo_forward_consensus(
    ticker: str,
    history: pd.DataFrame,
    settings: Settings,
) -> ForwardConsensus | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    yf_ticker = yf.Ticker(ticker)
    growth = _extract_yahoo_growth(yf_ticker)
    if growth is None:
        return None
    margin = _latest_margin(history)
    fcf_margin = _latest_fcf_margin(history)
    return ForwardConsensus(
        provider="Yahoo analyst data",
        revenue_growth_annual=float(growth),
        ebit_margin=margin,
        fcf_margin=fcf_margin,
        retrieved_at=datetime.now(timezone.utc),
        fallback_used=False,
        source_notes=["provider=yahoo", "metric=revenue_growth_estimate"],
    )


def _extract_yahoo_growth(yf_ticker) -> float | None:
    for attr in ("growth_estimates", "revenue_estimate"):
        try:
            frame = getattr(yf_ticker, attr)
        except Exception:
            continue
        if callable(frame):
            try:
                frame = frame()
            except Exception:
                continue
        growth = _growth_from_yahoo_frame(frame)
        if growth is not None:
            return growth
    return None


def _growth_from_yahoo_frame(frame) -> float | None:
    if frame is None or not hasattr(frame, "empty") or frame.empty:
        return None
    text = frame.astype(str)
    numeric_values: list[float] = []
    for value in text.to_numpy().ravel():
        parsed = _parse_percent(value)
        if parsed is not None and -0.30 <= parsed <= 0.60:
            numeric_values.append(parsed)
    if not numeric_values:
        return None
    return float(np.median(numeric_values[:4]))


def _internal_consensus(
    ticker: str,
    history: pd.DataFrame,
    settings: Settings,
    fallback_used: bool,
    note: str,
) -> ForwardConsensus:
    prior = prior_for(ticker, settings.company_type)
    recent_quarterly_growth = history["revenue_growth"].dropna().tail(8)
    if recent_quarterly_growth.empty:
        revenue_growth_annual = prior.revenue_growth_cap * 2.0
    else:
        annualized_recent = float((1.0 + recent_quarterly_growth.median()) ** 4 - 1.0)
        revenue_growth_annual = float(np.clip(annualized_recent, prior.revenue_growth_floor * 4, prior.revenue_growth_cap * 4))
    return ForwardConsensus(
        provider="internal_archetype_consensus",
        revenue_growth_annual=revenue_growth_annual,
        ebit_margin=_latest_margin(history),
        fcf_margin=_latest_fcf_margin(history),
        retrieved_at=datetime.now(timezone.utc),
        fallback_used=fallback_used,
        source_notes=[note, f"archetype={prior.archetype}"],
    )


def _latest_margin(history: pd.DataFrame) -> float | None:
    if "operating_margin" not in history:
        return None
    values = history["operating_margin"].dropna().tail(8)
    return float(values.median()) if not values.empty else None


def _latest_fcf_margin(history: pd.DataFrame) -> float | None:
    if not {"fcf", "revenue"}.issubset(history.columns):
        return None
    margins = (history["fcf"] / history["revenue"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    values = margins.dropna().tail(8)
    return float(values.median()) if not values.empty else None


def _parse_percent(value: object) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        numeric = float(text.rstrip("%"))
    except ValueError:
        return None
    return numeric / 100.0 if "%" in text or abs(numeric) > 1.0 else numeric
