"""Build a quarter-end total-return price panel for the portfolio backtest.

Cheap, one-shot fetch: monthly adjusted close for the 365 panel names + SPY + QQQ via a few
batched ``yfinance.download`` calls, resampled to the 45 panel quarter-end dates. Cached to
``reports/portfolio/_prices.csv`` (resumable -- re-runs skip names already cached unless --refresh).

This is a DATA layer only; it does not touch the DCF model.

Usage:
    python -m scripts.portfolio.build_price_panel            # fetch missing names, cache
    python -m scripts.portfolio.build_price_panel --refresh  # refetch everything
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"
PORT_DIR = REPORT_DIR / "portfolio"
PRICE_CSV = PORT_DIR / "_prices.csv"
BENCHMARKS = ["SPY", "QQQ"]


def panel_dates_and_tickers() -> tuple[list[str], list[str]]:
    obs = json.load(open(REPORT_DIR / "final_universe_backtest.json"))["observations"]
    dates = sorted({o["date"] for o in obs})
    tickers = sorted({o["ticker"] for o in obs})
    return dates, tickers


def _download_monthly(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Batched monthly adjusted-close download; returns wide DataFrame (cols=tickers)."""
    import yfinance as yf

    frames: dict[str, pd.Series] = {}
    batch = 40
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        for attempt in range(3):
            try:
                raw = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    interval="1mo",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  batch {i // batch} attempt {attempt} failed: {exc}")
                time.sleep(2.0)
        else:
            print(f"  batch {i // batch} permanently failed; skipping {chunk}")
            continue
        if raw is None or len(raw) == 0:
            continue
        # yfinance returns a column MultiIndex (field, ticker) for multi-name requests,
        # or a flat frame for a single name.
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": chunk[0]})
        for t in close.columns:
            s = close[t].dropna()
            if len(s):
                frames[t] = s
        print(f"  batch {i // batch + 1}/{(len(tickers) + batch - 1) // batch}: "
              f"{sum(1 for t in chunk if t in frames)}/{len(chunk)} names")
        time.sleep(0.5)
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def build(refresh: bool = False) -> pd.DataFrame:
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    dates, tickers = panel_dates_and_tickers()
    want = tickers + BENCHMARKS
    q_index = pd.to_datetime(dates)

    cached = pd.DataFrame()
    if PRICE_CSV.exists() and not refresh:
        cached = pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
        have = [t for t in want if t in cached.columns and cached[t].notna().any()]
        missing = [t for t in want if t not in have]
        print(f"cache: {len(have)} names present, {len(missing)} missing")
        if not missing:
            return cached
        to_fetch = missing
    else:
        to_fetch = want

    # Fetch monthly a bit before the first panel date so the first quarter-end aligns.
    start = (q_index[0] - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end = (q_index[-1] + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    print(f"fetching {len(to_fetch)} names monthly {start}..{end}")
    monthly = _download_monthly(to_fetch, start, end)

    if monthly.empty and cached.empty:
        raise SystemExit("no price data fetched")

    # Resample monthly -> the exact panel quarter-end dates via as-of (last price <= date).
    def to_quarter_end(df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in df.columns:
            s = df[col].dropna()
            if s.empty:
                continue
            out[col] = s.reindex(s.index.union(q_index)).sort_index().ffill().reindex(q_index)
        return pd.DataFrame(out, index=q_index)

    q_new = to_quarter_end(monthly) if not monthly.empty else pd.DataFrame(index=q_index)

    if not cached.empty:
        # Align cached to the quarter index and overlay newly fetched columns.
        cached = cached.reindex(q_index)
        for c in q_new.columns:
            cached[c] = q_new[c]
        panel = cached
    else:
        panel = q_new

    panel = panel.reindex(q_index)
    panel.to_csv(PRICE_CSV)
    got = [t for t in want if t in panel.columns and panel[t].notna().any()]
    print(f"saved {PRICE_CSV} -- {len(got)}/{len(want)} names with data "
          f"({len(panel)} quarter-end rows)")
    miss = [t for t in want if t not in got]
    if miss:
        print(f"no data for {len(miss)}: {miss[:20]}{' ...' if len(miss) > 20 else ''}")
    return panel


def load_price_panel() -> pd.DataFrame:
    if not PRICE_CSV.exists():
        raise SystemExit(f"{PRICE_CSV} missing -- run build_price_panel first")
    df = pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
    return df.sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build quarter-end price panel for portfolio backtest")
    ap.add_argument("--refresh", action="store_true", help="refetch all names")
    args = ap.parse_args()
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
