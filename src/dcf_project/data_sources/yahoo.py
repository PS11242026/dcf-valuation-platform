from __future__ import annotations

from dcf_project.config import Settings
from dcf_project.contracts import MarketSnapshot, SourceMeta


class YahooQueryClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def snapshot(self, ticker: str) -> MarketSnapshot:
        try:
            from yahooquery import Ticker
        except ImportError as exc:
            raise RuntimeError("yahooquery is required for YahooQuery fallback") from exc
        data = Ticker(ticker).summary_detail
        price_data = Ticker(ticker).price
        details = data.get(ticker) if isinstance(data, dict) else {}
        price = price_data.get(ticker, {}) if isinstance(price_data, dict) else {}
        market_price = _float_or_none(price.get("regularMarketPrice") or details.get("regularMarketPrice"))
        shares = _float_or_none(price.get("sharesOutstanding") or details.get("sharesOutstanding"))
        market_cap = _float_or_none(price.get("marketCap") or details.get("marketCap"))
        if market_cap is None and market_price is not None and shares is not None:
            market_cap = market_price * shares
        return MarketSnapshot(
            ticker=ticker.upper(),
            price=market_price,
            market_cap=market_cap,
            shares_outstanding=shares,
            beta=_float_or_none(details.get("beta")),
            cash=_float_or_none(details.get("totalCash")),
            debt=_float_or_none(details.get("totalDebt")),
            currency=price.get("currency"),
            source=SourceMeta(provider="YahooQuery", key=ticker.upper(), url=f"https://finance.yahoo.com/quote/{ticker}"),
        )


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
