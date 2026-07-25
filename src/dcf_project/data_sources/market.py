from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

import requests

from dcf_project.config import Settings
from dcf_project.contracts import MarketSnapshot, SourceMeta, WACCInputs


TREASURY_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value_month={year}{month:02d}"


class MarketDataError(RuntimeError):
    pass


class MarketClient:
    _risk_free_cache: tuple[float, str] | None = None
    _risk_free_lock = Lock()

    def __init__(self, settings: Settings):
        self.settings = settings

    def snapshot(self, ticker: str) -> MarketSnapshot:
        try:
            import yfinance as yf
        except ImportError:
            try:
                from dcf_project.data_sources.yahoo import YahooQueryClient

                return YahooQueryClient(self.settings).snapshot(ticker)
            except Exception as fallback_exc:
                raise MarketDataError("Yahoo market data dependency is required") from fallback_exc

        try:
            info = yf.Ticker(ticker).get_info()
        except Exception as exc:
            try:
                from dcf_project.data_sources.yahoo import YahooQueryClient

                return YahooQueryClient(self.settings).snapshot(ticker)
            except Exception:
                raise MarketDataError(f"Yahoo Finance query failed for {ticker}: {exc}") from exc
        if not info:
            raise MarketDataError(f"No Yahoo Finance data returned for {ticker}")
        price = _float_or_none(info.get("currentPrice") or info.get("regularMarketPrice"))
        shares = _float_or_none(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
        market_cap = _float_or_none(info.get("marketCap"))
        if market_cap is None and price is not None and shares is not None:
            market_cap = price * shares
        return MarketSnapshot(
            ticker=ticker.upper(),
            price=price,
            market_cap=market_cap,
            shares_outstanding=shares,
            beta=_float_or_none(info.get("beta")),
            cash=_float_or_none(info.get("totalCash")),
            debt=_float_or_none(info.get("totalDebt")),
            currency=info.get("currency"),
            source=SourceMeta(provider="Yahoo Finance", key=ticker.upper(), url=f"https://finance.yahoo.com/quote/{ticker}"),
        )

    def risk_free_rate(self) -> tuple[float, str]:
        # An explicit override (set by the India bolt-on to the live G-Sec yield) wins and
        # bypasses the class-level cache, so a US Treasury reading from an earlier call in
        # the same process never leaks into an Indian valuation (and vice versa).
        override = getattr(self.settings, "risk_free_override", None)
        if override is not None:
            return float(override), "risk_free_override"
        if MarketClient._risk_free_cache is not None:
            return MarketClient._risk_free_cache
        with MarketClient._risk_free_lock:
            if MarketClient._risk_free_cache is not None:
                return MarketClient._risk_free_cache
            now = datetime.now(timezone.utc)
            url = TREASURY_URL.format(year=now.year, month=now.month)
            try:
                response = requests.get(url, timeout=self.settings.request_timeout_seconds)
                response.raise_for_status()
                text = response.text
                marker = "<d:BC_10YEAR>"
                idx = text.rfind(marker)
                if idx >= 0:
                    end = text.find("</d:BC_10YEAR>", idx)
                    rate = float(text[idx + len(marker) : end]) / 100.0
                    MarketClient._risk_free_cache = (rate, url)
                    return MarketClient._risk_free_cache
            except Exception:
                pass
            MarketClient._risk_free_cache = (0.0375, "default_3_75pct")
            return MarketClient._risk_free_cache


def build_wacc(
    market: MarketSnapshot,
    tax_rate: float | None,
    debt: float | None,
    settings: Settings,
) -> WACCInputs:
    risk_free, source = MarketClient(settings).risk_free_rate()
    beta = market.beta if market.beta and market.beta > 0 else 1.0
    cost_of_equity = risk_free + beta * settings.equity_risk_premium
    tax = _bound(tax_rate if tax_rate is not None else settings.default_tax_rate, 0.0, 0.4)
    mv_equity = market.market_cap or (
        market.price * market.shares_outstanding
        if market.price is not None and market.shares_outstanding is not None
        else 0.0
    )
    mv_debt = max(debt or market.debt or 0.0, 0.0)
    total_capital = mv_equity + mv_debt
    if total_capital <= 0:
        raise MarketDataError("Cannot compute WACC without market value of equity or debt")
    equity_weight = mv_equity / total_capital
    debt_weight = mv_debt / total_capital
    after_tax_debt = settings.default_cost_of_debt * (1.0 - tax)
    wacc = equity_weight * cost_of_equity + debt_weight * after_tax_debt
    # Sanity floor: a blended WACC must retain a minimum equity-risk premium over
    # the risk-free rate. Without it, low-beta + high-leverage names (e.g. Verizon)
    # produce an all-in discount rate near the risk-free rate (~4.25%), which
    # explodes the terminal multiple and yields economically implausible values.
    wacc_floor = risk_free + 0.5 * settings.equity_risk_premium
    wacc = max(wacc, wacc_floor)
    return WACCInputs(
        risk_free_rate=risk_free,
        beta=beta,
        equity_risk_premium=settings.equity_risk_premium,
        cost_of_equity=cost_of_equity,
        pre_tax_cost_of_debt=settings.default_cost_of_debt,
        tax_rate=tax,
        market_value_equity=mv_equity,
        market_value_debt=mv_debt,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        after_tax_cost_of_debt=after_tax_debt,
        wacc=wacc,
        source_notes=[f"risk_free_source={source}", "cost_of_debt=default"],
    )


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bound(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
