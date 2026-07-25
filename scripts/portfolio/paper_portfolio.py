"""Paper portfolio: freeze today's top-N book and track it forward.

The backtest is survivorship-limited and in-sample; the cleanest validation is genuinely
out-of-sample -- pick the names TODAY and watch them. This freezes the current top-10 and top-20
books (both signal variants) from the live discount snapshot, locks fresh entry prices, and tracks
return vs SPY/QQQ from inception. Forward data accrues from the day you run ``open``.

Commands:
    python -m scripts.portfolio.paper_portfolio open          # create books (refuses to overwrite)
    python -m scripts.portfolio.paper_portfolio open --force  # reset inception (discards history)
    python -m scripts.portfolio.paper_portfolio track         # print live performance vs benchmarks

State: reports/portfolio/paper_portfolio.json (inception date, holdings, locked entry prices).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # scripts/portfolio -> _live_common
from _live_common import (  # noqa: E402
    BENCHMARKS,
    PORT_DIR,
    SNAP,
    STATE,
    _live_prices,
    _rank_books,
)


def cmd_open(force: bool) -> None:
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    if STATE.exists() and not force:
        raise SystemExit(f"{STATE} already exists; use --force to reset inception (discards history)")
    books = _rank_books()
    all_names = sorted({r["ticker"] for b in books.values() for r in b} | set(BENCHMARKS))
    print(f"locking entry prices for {len(all_names)} names...")
    prices = _live_prices(all_names)
    today = dt.date.today().isoformat()
    for b in books.values():
        for r in b:
            r["entry_price"] = prices.get(r["ticker"])
    state = {
        "inception": today,
        "snapshot_asof": dt.datetime.fromtimestamp(SNAP.stat().st_mtime).date().isoformat(),
        "benchmark_entry": {b: prices.get(b) for b in BENCHMARKS},
        "books": books,
        "note": "Out-of-sample forward paper test of the frozen DCF discount signal. "
                "Equal-weight, quarterly rebalance intended; this file is the inception book.",
    }
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"opened paper portfolio @ {today} (snapshot {state['snapshot_asof']})")
    for name, b in books.items():
        priced = sum(1 for r in b if r.get("entry_price"))
        print(f"  {name}: {priced}/{len(b)} priced -> {[r['ticker'] for r in b]}")


def cmd_track() -> None:
    if not STATE.exists():
        raise SystemExit(f"{STATE} missing; run `open` first")
    state = json.load(open(STATE))
    books = state["books"]
    all_names = sorted({r["ticker"] for b in books.values() for r in b} | set(BENCHMARKS))
    print(f"fetching current prices for {len(all_names)} names...")
    cur = _live_prices(all_names)
    today = dt.date.today().isoformat()
    inception = state["inception"]
    days = (dt.date.fromisoformat(today) - dt.date.fromisoformat(inception)).days

    def book_return(b) -> float | None:
        rets = []
        for r in b:
            e, c = r.get("entry_price"), cur.get(r["ticker"])
            if e and c and e > 0:
                rets.append(c / e - 1.0)
        return sum(rets) / len(rets) if rets else None

    bench_ret = {}
    for bm in BENCHMARKS:
        e, c = state["benchmark_entry"].get(bm), cur.get(bm)
        bench_ret[bm] = (c / e - 1.0) if (e and c and e > 0) else None

    print(f"\nPaper portfolio -- inception {inception} ({days} days), as of {today}")
    print("  benchmarks: " + "  ".join(
        f"{bm} {bench_ret[bm] * 100:+.1f}%" if bench_ret[bm] is not None else f"{bm} N/A"
        for bm in BENCHMARKS))
    print(f"  {'book':<24} {'return':>9}  {'vs SPY':>8}  {'vs QQQ':>8}")
    rows_out = {}
    for name, b in books.items():
        br = book_return(b)
        rows_out[name] = br
        if br is None:
            print(f"  {name:<24} {'N/A':>9}")
            continue
        vs_spy = (br - bench_ret["SPY"]) * 100 if bench_ret["SPY"] is not None else None
        vs_qqq = (br - bench_ret["QQQ"]) * 100 if bench_ret["QQQ"] is not None else None
        print(f"  {name:<24} {br * 100:>+8.1f}%  "
              f"{vs_spy:>+7.1f}%  {vs_qqq:>+7.1f}%" if vs_spy is not None else
              f"  {name:<24} {br * 100:>+8.1f}%")
    # persist a tracking snapshot row
    track_path = PORT_DIR / "paper_portfolio_track.json"
    hist = json.load(open(track_path)) if track_path.exists() else {"inception": inception, "rows": []}
    hist["rows"].append({"date": today, "days": days, "bench": bench_ret, "books": rows_out})
    track_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"\nappended tracking row to {track_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward paper portfolio over the DCF discount signal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    po = sub.add_parser("open", help="freeze today's top-N books with locked entry prices")
    po.add_argument("--force", action="store_true", help="reset inception (discards history)")
    sub.add_parser("track", help="print live performance vs SPY/QQQ")
    args = ap.parse_args()
    if args.cmd == "open":
        cmd_open(force=args.force)
    else:
        cmd_track()


if __name__ == "__main__":
    main()
