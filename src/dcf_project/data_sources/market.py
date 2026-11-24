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
        """Price / shares / market cap / beta for one ticker.

        Primary source is yfinance ``get_info``. Cloud hosts are often rate-limited or handed a
        near-empty quoteSummary, so every MISSING field is then filled, in order, from: yfinance
        fast_info + last close, Yahoo's public chart endpoint, Finnhub (when DCF_FINNHUB_KEY /
        FINNHUB_KEY is set) and yahooquery. Values already present are never overwritten."""
        info: dict = {}
        err: Exception | None = None
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).get_info() or {}
        except Exception as exc:  # noqa: BLE001 -- ImportError, rate limit, network
            err = exc
        q = {
            "price": _float_or_none(info.get("currentPrice") or info.get("regularMarketPrice")),
            "shares": _float_or_none(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")),
            "market_cap": _float_or_none(info.get("marketCap")),
            "currency": info.get("currency"),
            "beta": _float_or_none(info.get("beta")),
        }
        providers = ["Yahoo Finance"] if q["price"] else []
        for name, fill in (("Yahoo Finance", _fill_yf_fast),
                           ("Yahoo Finance chart", _fill_yahoo_chart),
                           ("Finnhub", _fill_finnhub),
                           ("yahooquery", lambda t, d: _fill_yahooquery(t, d, self.settings))):
            if q["price"] is not None and (q["shares"] is not None or q["market_cap"] is not None):
                break
            before = dict(q)
            try:
                fill(ticker, q)
            except Exception:  # noqa: BLE001 -- best-effort fallback chain
                pass
            if q != before and name not in providers:
                providers.append(name)
        if q["market_cap"] is None and q["price"] is not None and q["shares"] is not None:
            q["market_cap"] = q["price"] * q["shares"]
        if q["shares"] is None and q["market_cap"] is not None and q["price"]:
            q["shares"] = q["market_cap"] / q["price"]
        if not info and q["price"] is None and q["market_cap"] is None:
            raise MarketDataError(f"No market data for {ticker} from Yahoo, Finnhub or yahooquery"
                                  + (f" ({err})" if err else ""))
        return MarketSnapshot(
            ticker=ticker.upper(),
            price=q["price"],
            market_cap=q["market_cap"],
            shares_outstanding=q["shares"],
            beta=q["beta"],
            cash=_float_or_none(info.get("totalCash")),
            debt=_float_or_none(info.get("totalDebt")),
            currency=q["currency"],
            source=SourceMeta(provider=" + ".join(providers) or "Yahoo Finance", key=ticker.upper(),
                              url=f"https://finance.yahoo.com/quote/{ticker}"),
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



def _set(q: dict, key: str, value) -> None:
    if q.get(key) is None and value is not None:
        q[key] = value


def _fill_yf_fast(ticker: str, q: dict) -> None:
    """yfinance fast_info (chart-backed) + last daily close."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    fi = t.fast_info

    def get(*keys):
        for k in keys:
            try:
                v = _float_or_none(fi[k])
            except Exception:  # noqa: BLE001
                v = None
            if v:
                return v
        return None

    _set(q, "price", get("lastPrice", "last_price", "regularMarketPrice"))
    _set(q, "shares", get("shares"))
    _set(q, "market_cap", get("marketCap", "market_cap"))
    try:
        _set(q, "currency", fi["currency"])
    except Exception:  # noqa: BLE001
        pass
    if q["price"] is None:
        hist = t.history(period="5d", auto_adjust=False)
        if hist is not None and not hist.empty:
            _set(q, "price", _float_or_none(hist["Close"].dropna().iloc[-1]))


def _fill_yahoo_chart(ticker: str, q: dict) -> None:
    """Yahoo's public chart endpoint (no cookie/crumb session needed)."""
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"range": "5d", "interval": "1d"}, timeout=10,
                     headers={"User-Agent": "Mozilla/5.0 (compatible; dcf-valuation-platform)"})
    r.raise_for_status()
    meta = ((r.json().get("chart") or {}).get("result") or [{}])[0].get("meta") or {}
    _set(q, "price", _float_or_none(meta.get("regularMarketPrice")))
    _set(q, "currency", meta.get("currency"))


def _fill_finnhub(ticker: str, q: dict) -> None:
    """Finnhub quote + profile (needs DCF_FINNHUB_KEY or FINNHUB_KEY)."""
    import os

    key = os.environ.get("DCF_FINNHUB_KEY") or os.environ.get("FINNHUB_KEY")
    if not key:
        return
    base = "https://finnhub.io/api/v1"
    if q["price"] is None:
        qt = requests.get(f"{base}/quote", params={"symbol": ticker, "token": key}, timeout=10).json()
        _set(q, "price", _float_or_none(qt.get("c")) or None)
    if q["shares"] is None and q["market_cap"] is None:
        pr = requests.get(f"{base}/stock/profile2", params={"symbol": ticker, "token": key}, timeout=10).json()
        so, mc = _float_or_none(pr.get("shareOutstanding")), _float_or_none(pr.get("marketCapitalization"))
        _set(q, "shares", so * 1e6 if so else None)          # Finnhub reports millions
        _set(q, "market_cap", mc * 1e6 if mc else None)
        _set(q, "currency", pr.get("currency"))


def _fill_yahooquery(ticker: str, q: dict, settings: Settings) -> None:
    from dcf_project.data_sources.yahoo import YahooQueryClient

    yq = YahooQueryClient(settings).snapshot(ticker)
    _set(q, "price", yq.price)
    _set(q, "shares", yq.shares_outstanding)
    _set(q, "market_cap", yq.market_cap)
    _set(q, "currency", yq.currency)
