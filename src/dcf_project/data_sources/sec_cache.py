"""Opt-in disk cache for SEC EDGAR payloads.

Additive layer behind ``Settings.sec_cache_enabled`` (default False -- when off, the
network code path in :mod:`dcf_project.data_sources.sec` is untouched). Cached bytes
are the verbatim ``companyfacts`` JSON wrapped in a small envelope so consumers can
see the snapshot age. A frozen snapshot also makes a backtest campaign internally
consistent: every run prices the same filing set instead of whatever EDGAR serves
that hour.

Layout: ``<repo>/.cache/sec/CIK{cik}.json.gz`` plus ``company_tickers.json.gz`` for
the ticker->CIK map. The directory is git-ignored.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dcf_project.paths import CACHE

SEC_CACHE_DIR = CACHE / "sec"
_TICKER_MAP_NAME = "company_tickers.json.gz"


def _facts_path(cik: str, cache_dir: Path) -> Path:
    return cache_dir / f"CIK{cik}.json.gz"


def _read_envelope(path: Path, max_age_days: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            envelope = json.load(fh)
        retrieved_at = datetime.fromisoformat(envelope["retrieved_at"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None  # unreadable/corrupt cache entries behave as misses
    if datetime.now(timezone.utc) - retrieved_at > timedelta(days=max_age_days):
        return None
    return envelope


def _write_envelope(path: Path, payload: dict[str, Any], url: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "payload": payload,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(envelope, fh, separators=(",", ":"))
    tmp.replace(path)  # atomic on the same filesystem; readers never see partial files


def load_cached(cik: str, *, max_age_days: int, cache_dir: Path | None = None) -> dict[str, Any] | None:
    """Return the cached companyfacts envelope for ``cik`` or None on miss/expiry.

    Envelope keys: ``retrieved_at`` (ISO), ``url``, ``payload`` (verbatim companyfacts).
    ``cache_dir`` defaults to the module-level SEC_CACHE_DIR at call time (late-bound
    so tests can redirect the directory).
    """
    return _read_envelope(_facts_path(cik, cache_dir or SEC_CACHE_DIR), max_age_days)


def store(cik: str, payload: dict[str, Any], url: str | None, *, cache_dir: Path | None = None) -> None:
    _write_envelope(_facts_path(cik, cache_dir or SEC_CACHE_DIR), payload, url)


def load_ticker_map(*, max_age_days: int, cache_dir: Path | None = None) -> dict[str, str] | None:
    envelope = _read_envelope((cache_dir or SEC_CACHE_DIR) / _TICKER_MAP_NAME, max_age_days)
    if envelope is None:
        return None
    payload = envelope.get("payload")
    return dict(payload) if isinstance(payload, dict) else None


def store_ticker_map(mapping: dict[str, str], url: str | None, *, cache_dir: Path | None = None) -> None:
    _write_envelope((cache_dir or SEC_CACHE_DIR) / _TICKER_MAP_NAME, mapping, url)
