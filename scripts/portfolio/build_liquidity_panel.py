"""Build a quarter-end dollar-volume (liquidity) panel for the risk-control ablation.

Mirrors build_price_panel.py but fetches monthly **volume**; dollar-volume = close x volume, resampled
to the 45 panel quarter-end dates as a trailing median (a stable liquidity proxy). Cached to
reports/portfolio/_dollar_volume.csv (resumable). The liquidity screen in portfolio_sim drops names whose
quarter-end trailing dollar-volume is below a threshold (default $10M/day-equivalent on a monthly basis).

Usage:
    python -m scripts.portfolio.build_liquidity_panel
    python -m scripts.portfolio.build_liquidity_panel --refresh
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from scripts.portfolio.build_price_panel import panel_dates_and_tickers

PORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "portfolio"
DOLLAR_VOL_CSV = PORT_DIR / "_dollar_volume.csv"


def _download_dollar_volume(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    frames: dict[str, pd.Series] = {}
    batch = 40
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=start, end=end, interval="1mo",
                                  auto_adjust=True, progress=False, threads=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  batch {i // batch} attempt {attempt} failed: {exc}")
                time.sleep(2.0)
        else:
            continue
        if raw is None or len(raw) == 0:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            close, vol = raw["Close"], raw["Volume"]
        else:
            close = raw[["Close"]].rename(columns={"Close": chunk[0]})
            vol = raw[["Volume"]].rename(columns={"Volume": chunk[0]})
        for t in close.columns:
            dv = (close[t] * vol[t]).dropna()
            if len(dv):
                frames[t] = dv
        print(f"  batch {i // batch + 1}/{(len(tickers) + batch - 1) // batch}: "
              f"{sum(1 for t in chunk if t in frames)}/{len(chunk)}")
        time.sleep(0.5)
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def build(refresh: bool = False) -> pd.DataFrame:
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    dates, tickers = panel_dates_and_tickers()
    q_index = pd.to_datetime(dates)

    if DOLLAR_VOL_CSV.exists() and not refresh:
        cached = pd.read_csv(DOLLAR_VOL_CSV, index_col=0, parse_dates=True)
        have = [t for t in tickers if t in cached.columns and cached[t].notna().any()]
        if len(have) >= len(tickers) - 5:
            print(f"cache: {len(have)}/{len(tickers)} names present")
            return cached

    start = (q_index[0] - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end = (q_index[-1] + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    print(f"fetching monthly volume for {len(tickers)} names {start}..{end}")
    monthly_dv = _download_dollar_volume(tickers, start, end)
    if monthly_dv.empty:
        raise SystemExit("no volume data fetched")

    # quarter-end = trailing 3-month median dollar-volume (robust to a single spiky month)
    out = {}
    for col in monthly_dv.columns:
        s = monthly_dv[col].dropna()
        roll = s.rolling(3, min_periods=1).median()
        out[col] = roll.reindex(roll.index.union(q_index)).sort_index().ffill().reindex(q_index)
    panel = pd.DataFrame(out, index=q_index)
    panel.to_csv(DOLLAR_VOL_CSV)
    got = [t for t in tickers if t in panel.columns and panel[t].notna().any()]
    print(f"saved {DOLLAR_VOL_CSV} -- {len(got)}/{len(tickers)} names ({len(panel)} rows)")
    return panel


def load_liquidity_panel() -> pd.DataFrame:
    if not DOLLAR_VOL_CSV.exists():
        raise SystemExit(f"{DOLLAR_VOL_CSV} missing -- run build_liquidity_panel first")
    return pd.read_csv(DOLLAR_VOL_CSV, index_col=0, parse_dates=True).sort_index()


def screen_counts(panel: pd.DataFrame, thresholds=(5e6, 1e7, 2e7)) -> None:
    print("\nNames passing a per-name median dollar-volume screen (share of panel cells with data):")
    for thr in thresholds:
        med = panel.median(axis=0)
        n_pass = int((med >= thr).sum())
        n_tot = int(med.notna().sum())
        print(f"  >= ${thr/1e6:.0f}M : {n_pass}/{n_tot} names pass "
              f"({100*n_pass/max(1,n_tot):.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build quarter-end dollar-volume panel")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    panel = build(refresh=args.refresh)
    screen_counts(panel)


if __name__ == "__main__":
    main()
