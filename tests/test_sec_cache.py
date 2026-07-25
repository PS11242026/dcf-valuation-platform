"""SEC disk cache: roundtrip, expiry, and the cache-on/off SECClient guard.

The guard contract is the load-bearing property: with ``sec_cache_enabled=False``
(the default) SECClient must never touch the cache layer, and with it enabled a
warm cache must serve companyfacts without any network call.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from dcf_project.config import get_settings
from dcf_project.data_sources import sec_cache
from dcf_project.data_sources.sec import SECClient


CIK = "0000320193"
FACTS = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": []}}}}}


def test_roundtrip(tmp_path):
    sec_cache.store(CIK, FACTS, "http://example/cik", cache_dir=tmp_path)
    envelope = sec_cache.load_cached(CIK, max_age_days=30, cache_dir=tmp_path)
    assert envelope is not None
    assert envelope["payload"] == FACTS
    assert envelope["url"] == "http://example/cik"
    datetime.fromisoformat(envelope["retrieved_at"])  # parseable timestamp


def test_miss_and_expiry(tmp_path):
    assert sec_cache.load_cached("0000000000", max_age_days=30, cache_dir=tmp_path) is None
    sec_cache.store(CIK, FACTS, None, cache_dir=tmp_path)
    assert sec_cache.load_cached(CIK, max_age_days=0, cache_dir=tmp_path) is None  # max_age 0 == always stale


def test_corrupt_entry_is_a_miss(tmp_path):
    sec_cache.store(CIK, FACTS, None, cache_dir=tmp_path)
    path = tmp_path / f"CIK{CIK}.json.gz"
    path.write_bytes(b"not gzip")
    assert sec_cache.load_cached(CIK, max_age_days=30, cache_dir=tmp_path) is None


def test_ticker_map_roundtrip(tmp_path):
    mapping = {"AAPL": CIK, "MSFT": "0000789019"}
    sec_cache.store_ticker_map(mapping, "http://example/tickers", cache_dir=tmp_path)
    assert sec_cache.load_ticker_map(max_age_days=30, cache_dir=tmp_path) == mapping


class _NoNetworkSession:
    """Stands in for requests.Session; any HTTP attempt fails the test."""

    headers: dict = {}

    def update(self, *_args, **_kwargs):  # pragma: no cover - headers.update stub
        pass

    def get(self, *_args, **_kwargs):
        raise AssertionError("network call attempted")


def test_client_serves_from_warm_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(sec_cache, "SEC_CACHE_DIR", tmp_path)
    sec_cache.store(CIK, FACTS, None, cache_dir=tmp_path)
    settings = replace(get_settings(), sec_cache_enabled=True)
    client = SECClient(settings)
    client.session = _NoNetworkSession()
    monkeypatch.setattr(SECClient, "cik_for_ticker", lambda self, ticker: CIK)
    facts, meta = client.company_facts("AAPL")
    assert facts == FACTS
    assert meta.provider == "SEC" and meta.key == f"CIK{CIK}"
    assert meta.retrieved_at <= datetime.now(timezone.utc) + timedelta(seconds=5)


def test_client_default_ignores_cache(tmp_path, monkeypatch):
    """sec_cache_enabled=False must go to the network even with a warm cache present."""
    monkeypatch.setattr(sec_cache, "SEC_CACHE_DIR", tmp_path)
    sec_cache.store(CIK, FACTS, None, cache_dir=tmp_path)
    settings = get_settings()
    assert settings.sec_cache_enabled is False
    client = SECClient(settings)
    client.session = _NoNetworkSession()
    monkeypatch.setattr(SECClient, "cik_for_ticker", lambda self, ticker: CIK)
    with pytest.raises(AssertionError, match="network call attempted"):
        client.company_facts("AAPL")
