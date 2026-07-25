"""Read-only analyst-consensus snapshot from free Yahoo data (Research_V4).

ISOLATION: this is a research-only reader. It does NOT modify
``dcf_project.data_sources.consensus`` and is never wired into any pipeline. It reuses
that module's growth-extraction logic (``_extract_yahoo_growth``) so the forward-growth
number is computed the same way the engine would, and additionally pulls the analyst
*price target* and *recommendation* fields the engine does not consume.

yfinance coverage is uneven (many fields come back ``None``); every field is recorded
with the value or ``null`` plus a ``coverage`` flag, and responses are cached to disk so
the study is reproducible and re-runs are offline/instant.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "src"):
    import sys
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dcf_project.data_sources.consensus import _extract_yahoo_growth  # noqa: E402

CACHE_DIR = ROOT / "reports" / "research_v4" / "_cache" / "consensus"


@dataclass(frozen=True)
class ConsensusSnapshot:
    ticker: str
    price: float | None              # Yahoo current price (consensus context only)
    target_mean: float | None        # analyst mean price target
    target_median: float | None
    target_high: float | None
    target_low: float | None
    recommendation_mean: float | None  # 1=Strong Buy ... 5=Sell
    recommendation_key: str | None
    n_analysts: int | None
    revenue_growth_annual: float | None  # forward growth, same extractor as the engine
    retrieved_at: str
    coverage: list[str]              # which fields came back populated
    error: str | None = None


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if out == out else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int | None:
    f = _f(value)
    return int(f) if f is not None else None


def fetch_consensus(ticker: str, *, use_cache: bool = True, refresh: bool = False) -> ConsensusSnapshot:
    ticker = ticker.upper()
    cache_path = CACHE_DIR / f"{ticker}.json"
    if use_cache and not refresh and cache_path.exists():
        return ConsensusSnapshot(**json.loads(cache_path.read_text(encoding="utf-8")))

    snap = _fetch_live(ticker)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(asdict(snap), indent=2), encoding="utf-8")
    return snap


def _fetch_live(ticker: str) -> ConsensusSnapshot:
    now = datetime.now(timezone.utc).isoformat()
    try:
        import yfinance as yf
    except ImportError:
        return ConsensusSnapshot(ticker, *( [None] * 9 ), now, [], error="yfinance_unavailable")

    try:
        yf_ticker = yf.Ticker(ticker)
        info: dict[str, Any] = yf_ticker.get_info()
    except Exception as exc:  # noqa: BLE001 -- network/transport; record and move on
        return ConsensusSnapshot(ticker, *( [None] * 9 ), now, [], error=f"info_error:{type(exc).__name__}")

    try:
        growth = _extract_yahoo_growth(yf_ticker)
    except Exception:  # noqa: BLE001
        growth = None

    fields = {
        "price": _f(info.get("currentPrice") or info.get("regularMarketPrice")),
        "target_mean": _f(info.get("targetMeanPrice")),
        "target_median": _f(info.get("targetMedianPrice")),
        "target_high": _f(info.get("targetHighPrice")),
        "target_low": _f(info.get("targetLowPrice")),
        "recommendation_mean": _f(info.get("recommendationMean")),
        "recommendation_key": (info.get("recommendationKey") or None),
        "n_analysts": _i(info.get("numberOfAnalystOpinions")),
        "revenue_growth_annual": _f(growth),
    }
    coverage = [k for k, v in fields.items() if v is not None]
    return ConsensusSnapshot(ticker=ticker, retrieved_at=now, coverage=coverage, error=None, **fields)
