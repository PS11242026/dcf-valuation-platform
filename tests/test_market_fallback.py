"""MarketClient.snapshot fills price/shares when Yahoo's quoteSummary comes back near-empty
(typical on cloud hosts) instead of failing WACC with 'no market value of equity'."""
import sys
import types

from dcf_project.core import get_settings
from dcf_project.data_sources.market import MarketClient


class _FastInfo(dict):
    pass


def _fake_yf(info, fast):
    class T:
        def __init__(self, ticker):
            self.ticker = ticker
            self.fast_info = _FastInfo(fast)

        def get_info(self):
            return info

        def history(self, **_):
            import pandas as pd
            return pd.DataFrame({"Close": [50.0]})

    return types.SimpleNamespace(Ticker=T)


def test_empty_info_falls_back_to_fast_info(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance",
                        _fake_yf({"trailingPegRatio": None}, {"lastPrice": 95.0, "shares": 4.3e9, "currency": "USD"}))
    s = MarketClient(get_settings()).snapshot("KO")
    assert s.price == 95.0 and s.shares_outstanding == 4.3e9
    assert abs(s.market_cap - 95.0 * 4.3e9) < 1 and s.currency == "USD"


def test_full_info_is_not_overridden(monkeypatch):
    info = {"currentPrice": 10.0, "sharesOutstanding": 100.0, "marketCap": 1000.0, "currency": "USD"}
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf(info, {"lastPrice": 999.0, "shares": 1.0}))
    s = MarketClient(get_settings()).snapshot("XYZ")
    assert (s.price, s.shares_outstanding, s.market_cap) == (10.0, 100.0, 1000.0)


def test_price_from_history_when_fast_info_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf({"marketCap": 5e9}, {}))
    s = MarketClient(get_settings()).snapshot("ABC")
    assert s.price == 50.0 and s.market_cap == 5e9 and abs(s.shares_outstanding - 1e8) < 1
