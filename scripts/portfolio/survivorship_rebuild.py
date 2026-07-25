"""Stage B -- empirical rebuild: replace the synthetic delisting shock with REAL recovered returns.

The gold-standard fix (re-run the frozen DCF on every dead name at each PIT date and fold its discount
into the panel) is blocked on the price side: free sources lack accurate delisting *returns*. So this
stage does the feasible, honest version:

  1. Take the EDGAR-identified holes (dead in-family 10-K filers) that still carry a ticker.
  2. Try to recover ANY free price history (yfinance) -> measure the RECOVERY RATE (how identifiable /
     priceable the dead names are at all -- itself the size of the data wall).
  3. For recoverable names, compute the realized terminal return from the base-window era to the last
     available price -> an EMPIRICAL loss-given-delisting distribution (replaces the assumed -70%),
     and split acquired-ish (flat/up) vs failed (down hard).
  4. Emit the empirical L (e.g. median / mean loss) so the Stage A.2 stress test can be re-run on a
     data-grounded shock instead of an assumption.

Usage:
    python -m scripts.portfolio.survivorship_rebuild
    python -m scripts.portfolio.survivorship_rebuild --max-names 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "portfolio"


def _recover_prices(tickers: list[str]):
    import pandas as pd
    import yfinance as yf

    out = {}
    batch = 40
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        try:
            raw = yf.download(chunk, period="max", interval="1mo", auto_adjust=True,
                              progress=False, threads=True)
        except Exception:  # noqa: BLE001
            continue
        if raw is None or len(raw) == 0:
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(
            columns={"Close": chunk[0]})
        for t in close.columns:
            s = close[t].dropna()
            if len(s) >= 6:
                out[t] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B: empirical dead-name recovery + loss distribution")
    ap.add_argument("--max-names", type=int, default=0, help="cap dead names to probe")
    ap.add_argument("--base-year", type=int, default=2016, help="anchor year for realized return")
    args = ap.parse_args()

    holes = json.load(open(PORT_DIR / "survivorship_holes.json"))
    detail = holes.get("holes_detail", [])
    # dead names that at least carry a ticker we can search for
    with_ticker = [(h, h["tickers"][0]) for h in detail if h.get("tickers")]
    n_dead = len(detail)
    n_ticketed = len(with_ticker)
    probe = with_ticker[: args.max_names] if args.max_names else with_ticker
    tickers = [t for _, t in probe]
    print(f"dead in-family names: {n_dead}; with a ticker: {n_ticketed}; probing {len(tickers)}")

    recovered = _recover_prices(tickers) if tickers else {}
    print(f"free price data recovered for {len(recovered)}/{len(tickers)} "
          f"({100 * len(recovered) / max(1, len(tickers)):.0f}%)")

    import pandas as pd

    anchor = pd.Timestamp(f"{args.base_year}-06-30")
    losses = []  # realized terminal return (price_last/price_anchor - 1) for held-to-death
    rows = []
    for h, tk in probe:
        s = recovered.get(tk)
        if s is None:
            continue
        s = s.sort_index()
        near = s[s.index >= anchor]
        if near.empty:
            continue
        p0 = float(near.iloc[0])
        p1 = float(s.iloc[-1])
        if p0 <= 0:
            continue
        ret = p1 / p0 - 1.0
        losses.append(ret)
        rows.append({"ticker": tk, "name": h["name"], "family": h["family"],
                     "last_10k": h["last_10k"], "anchor_price": round(p0, 2),
                     "last_price": round(p1, 2), "realized_return": round(ret, 3)})

    summary = {
        "n_dead_in_family": n_dead,
        "n_with_ticker": n_ticketed,
        "n_probed": len(tickers),
        "n_price_recovered": len(recovered),
        "recovery_rate": round(len(recovered) / max(1, len(tickers)), 3),
        "n_with_realized_return": len(losses),
    }
    if losses:
        arr = np.array(losses)
        summary.update({
            "realized_return_median": round(float(np.median(arr)), 3),
            "realized_return_mean": round(float(np.mean(arr)), 3),
            "frac_down_50pct": round(float(np.mean(arr <= -0.5)), 3),
            "frac_total_loss_90pct": round(float(np.mean(arr <= -0.9)), 3),
            "frac_positive": round(float(np.mean(arr > 0)), 3),
            "empirical_loss_given_delisting": round(float(-np.median(arr[arr < 0])) if (arr < 0).any()
                                                    else 0.0, 3),
        })
    out = {"summary": summary, "recovered_detail": sorted(rows, key=lambda r: r["realized_return"])}
    (PORT_DIR / "survivorship_rebuild.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Stage B summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nNOTE: a full survivorship-free *return* backtest (re-pricing dead names via the frozen "
          "DCF and folding their PIT discounts into the panel) remains blocked without paid "
          "delisting-return data. The recovery rate above quantifies how blocked. The Stage A.2 "
          "stress curve -- now re-runnable with the empirical loss above via "
          "`survivorship_stress --loss <L>` -- is the defensible bound.")
    print("wrote survivorship_rebuild.json")


if __name__ == "__main__":
    main()
