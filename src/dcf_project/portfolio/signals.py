"""Signal ranking + live pricing -- ported from scripts/portfolio/_live_common.py (logic unchanged)."""
from __future__ import annotations

import json

from dcf_project.paths import REPORTS
from dcf_project.portfolio.scoring import _sector_reliability

SNAP = REPORTS / "_live_universe_snapshot.json"
INST = REPORTS / "final_institutional_validation.json"
DISCOUNT_CLIP = 5.0
BENCHMARKS = ["SPY", "QQQ"]
VALID_SECTORS = {
    "software_saas_infra", "internet_platforms_marketplaces", "semiconductors",
    "semiconductor_equipment", "consumer_staples", "consumer_discretionary",
    "industrials_assetlight_services",
}


def sector_scores() -> dict[str, float]:
    inst = json.load(open(INST)).get("by_sector_family", {})
    return {f: _sector_reliability(b)["score"] for f, b in inst.items()}


def live_prices(tickers: list[str]) -> dict[str, float]:
    import yfinance as yf

    out: dict[str, float] = {}
    batch = 40
    for i in range(0, len(tickers), batch):
        chunk = tickers[i: i + batch]
        try:
            raw = yf.download(chunk, period="5d", interval="1d", auto_adjust=True, progress=False, threads=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  price batch {i // batch} failed: {exc}")
            continue
        if raw is None or len(raw) == 0:
            continue
        import pandas as pd

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": chunk[0]})
        for t in close.columns:
            s = close[t].dropna()
            if len(s):
                out[t] = float(s.iloc[-1])
    return out


def live_last_prev(tickers: list[str]) -> dict[str, tuple[float, float | None]]:
    """{symbol: (last_close, prev_close)} from one 5d batch -- powers 1-day-change fields.
    prev is None when only a single session is available."""
    import yfinance as yf

    out: dict[str, tuple[float, float | None]] = {}
    batch = 40
    for i in range(0, len(tickers), batch):
        chunk = tickers[i: i + batch]
        try:
            raw = yf.download(chunk, period="5d", interval="1d", auto_adjust=True, progress=False, threads=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  price batch {i // batch} failed: {exc}")
            continue
        if raw is None or len(raw) == 0:
            continue
        import pandas as pd

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": chunk[0]})
        for t in close.columns:
            s = close[t].dropna()
            if len(s):
                out[t] = (float(s.iloc[-1]), float(s.iloc[-2]) if len(s) >= 2 else None)
    return out


def window_prices(symbol: str, start: str) -> tuple[float | None, float | None]:
    """(first, last) auto-adjusted close on/after `start` -- comparator window returns."""
    import yfinance as yf

    raw = yf.download(symbol, start=start, interval="1d", auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return None, None
    close = raw["Close"].dropna()
    if hasattr(close, "columns"):  # multiindex -> single column
        close = close.iloc[:, 0]
    if not len(close):
        return None, None
    return float(close.iloc[0]), float(close.iloc[-1])


def weekly_close(symbol: str, years: float = 3.0):
    """Weekly total-return close series (auto_adjust folds splits+dividends), date-indexed so
    holdings and benchmark align on common weeks. Returns a pandas Series or None (min 30 obs)."""
    import yfinance as yf

    raw = yf.download(symbol, period=f"{int(years * 12)}mo", interval="1wk",
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return None
    close = raw["Close"].dropna()
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close.name = symbol
    return close if len(close) >= 30 else None


def trailing_div_yield(symbol: str, price: float) -> float | None:
    """Trailing-12m dividends / price; best-effort, None on any failure."""
    import datetime as dt

    import yfinance as yf

    try:
        divs = yf.Ticker(symbol).dividends
    except Exception:  # noqa: BLE001
        return None
    if divs is None or len(divs) == 0 or not price:
        return None
    cutoff = (dt.datetime.now(divs.index.tz) if divs.index.tz is not None
              else dt.datetime.now()) - dt.timedelta(days=365)
    last12 = divs[divs.index >= cutoff]
    return float(last12.sum()) / float(price) if len(last12) else None


LOGO_SOURCES = (
    "https://assets.parqet.com/logos/symbol/{sym}?format=png&size=128",
    "https://financialmodelingprep.com/image-stock/{sym}.png",
)


def fetch_logos(symbols: list[str], out_dir) -> int:
    """Best-effort company logo PNGs -> out_dir/<SYM>.png, cached (skip if present) so only the
    first run touches the network. Self-hosted afterwards: the site never hotlinks."""
    import urllib.request
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    got = 0
    for sym in symbols:
        dest = out_dir / f"{sym}.png"
        if dest.exists():
            got += 1
            continue
        for url in LOGO_SOURCES:
            try:
                req = urllib.request.Request(url.format(sym=sym), headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=15).read()
                if data and len(data) > 200 and data[:4] != b"<htm":
                    dest.write_bytes(data)
                    got += 1
                    break
            except Exception:  # noqa: BLE001
                continue
    return got


def fetch_names(symbols: list[str], cache_path) -> dict[str, str]:
    """{symbol: long company name}, cached to JSON so the slow yfinance info call runs once
    per symbol ever. Missing names simply stay absent (hover falls back to the ticker)."""
    import json
    from pathlib import Path

    cache_path = Path(cache_path)
    names = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    missing = [s for s in symbols if s not in names]
    if missing:
        import yfinance as yf

        for s in missing:
            try:
                info = yf.Ticker(s).info or {}
                nm = info.get("longName") or info.get("shortName")
                if nm:
                    names[s] = nm
            except Exception:  # noqa: BLE001
                continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    return names


def fetch_news(symbols: list[str], per_name: int = 3) -> list[dict]:
    """Best-effort recent headlines per symbol (website feed). Handles both the flat and the
    nested `content` schema yfinance has shipped; failures skip the symbol, never raise."""
    import yfinance as yf

    items, seen = [], set()
    for s in symbols:
        try:
            raw = yf.Ticker(s).news or []
        except Exception:  # noqa: BLE001
            continue
        for n in raw[:per_name]:
            c = n.get("content") if isinstance(n.get("content"), dict) else n
            title = c.get("title")
            if not title or title in seen:
                continue
            seen.add(title)
            link = c.get("link") or ((c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else None)
            pub = c.get("publisher") or ((c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else None)
            when = c.get("providerPublishTime") or c.get("pubDate")
            items.append({"symbol": s, "title": title, "publisher": pub, "link": link, "published": when})
    return items


def rank_books(snapshot_path=SNAP) -> dict:
    snap = json.load(open(snapshot_path))["snapshot"]
    scores = sector_scores()
    rows = []
    for tk, v in snap.items():
        d = v.get("discount")
        sec = v.get("sector")
        if not isinstance(d, (int, float)) or abs(d) > DISCOUNT_CLIP or sec not in VALID_SECTORS:
            continue
        rows.append({"ticker": tk, "discount": d, "sector": sec, "intrinsic": v.get("intrinsic"),
                     "snap_price": v.get("price"), "conf": scores.get(sec, 0.0),
                     "sig_discount": d, "sig_discount_x_conf": d * scores.get(sec, 0.0)})
    books = {}
    for sig in ("discount", "discount_x_conf"):
        ranked = sorted(rows, key=lambda r: r[f"sig_{sig}"], reverse=True)
        for n in (10, 20):
            books[f"{sig}_top{n}"] = [dict(r) for r in ranked[:n]]
    return books


def select(signal_id: str, snapshot_path=SNAP) -> list[dict]:
    books = rank_books(snapshot_path)
    if signal_id not in books:
        raise KeyError(f"unknown signal '{signal_id}'; choices: {sorted(books)}")
    return books[signal_id]
