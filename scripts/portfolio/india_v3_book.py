"""India v3 book -- AGGRESSIVE CONCENTRATED construction (standalone; does NOT touch the US registry).

User-commissioned (2026-06-15): a high-conviction, concentrated India book sized for maximum upside
on a Rs 10,00,000 real-money allocation, picked on the engine_v3 discount signal.

HONESTY CONTRACT (read the report it writes):
  * The v3 discount is a ~1-YEAR signal (India Phase B: 1y IC +0.22 overall, energy +0.48). It has NO
    demonstrated edge over a 10-day horizon. A +10-12% / 10-day target is a POSSIBLE outcome of a
    high-variance book, NOT an expected return -- the expected 10-day return is ~market drift.
  * Concentration + leverage multiply VARIANCE, not the mean. This book is built for torque, and is
    documented as such: a big down move is exactly as likely as a big up move over 10 days.
  * The signal is PRELIMINARY and survivorship-optimistic (Phase B caveats); not validated alpha.

WHAT IT BUILDS: values a candidate universe live (engine_v3, country=IN), ranks by the India v3
relative selection rank, takes the most concentrated top-N (which lands on the Indian oil/gas PSU
complex -- the only sector the model finds cheap AND where the discount predicts returns), conviction-
weights, and allocates the capital into whole shares. Writes a book state + an honest report under
reports/managed_portfolios/india_v3_aggressive/. Benchmark: NIFTY 50 (^NSEI).
"""
from __future__ import annotations

import datetime as dt
import json

from dcf_project.paths import MANAGED
from dcf_project.valuation.canonical import canonical_valuation

CAPITAL_INR = 1_000_000.0
BOOK_NAME = "india_v3_aggressive"
BENCHMARK = "^NSEI"          # NIFTY 50
TOP_N = 5

# Phase B finding: the v3 discount PREDICTS 1y returns only in cyclical / capital-intensive sectors
# (1y IC: materials +0.49, energy +0.48, utilities +0.46, industrials +0.36). In technology (IC ~0),
# consumer_staples (~0) and health_care (negative) a high RELATIVE rank is a "cheap but the signal
# doesn't work here" artifact -- so we restrict the book to the validated-predictive sectors. This is
# what drops HCLTECH/TCS/INFY (tech ranks high on cheapness but the discount is non-predictive there).
VALIDATED_SECTORS = {"energy", "materials", "utilities", "industrials"}

# Candidate universe = the energy PSU complex (the signal's clear call) + the best-ranked names from
# the other India sectors, so the selection is demonstrably the top of the cross-section, not assumed.
CANDIDATES = [
    "IOC", "BPCL", "ONGC", "COALINDIA", "HINDPETRO", "OIL", "GAIL",   # energy
    "TCS", "INFY", "HCLTECH",                                           # technology (best non-energy ranks)
    "ACC", "GRASIM",                                                    # materials
    "MARUTI",                                                           # consumer_discretionary
]


def _live_price(symbol: str) -> float | None:
    import yfinance as yf

    raw = yf.download(symbol, period="5d", interval="1d", auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return None
    ser = raw["Close"].dropna()
    return float(ser.iloc[-1].iloc[0] if hasattr(ser.iloc[-1], "iloc") else ser.iloc[-1])


def value_universe() -> list[dict]:
    rows = []
    for t in CANDIDATES:
        try:
            sc = canonical_valuation(t, engine="v3", country="IN")
        except Exception as exc:  # noqa: BLE001
            print(f"  {t}: skip ({type(exc).__name__})")
            continue
        rows.append({
            "ticker": t, "sector": sc.get("sector"), "price": sc.get("price"),
            "intrinsic": sc.get("intrinsic_value"), "discount": sc.get("discount_to_intrinsic"),
            "er_rank": sc.get("expected_return_rank"), "verdict": sc.get("verdict"),
        })
        d = sc.get("discount_to_intrinsic")
        print(f"  {t:11s} {str(sc.get('sector')):14s} rank={sc.get('expected_return_rank')}"
              f" disc={(f'{d:+.1%}' if d is not None else 'NA')}")
    return rows


def select_book(rows: list[dict], n: int = TOP_N) -> list[dict]:
    eligible = [r for r in rows if r["sector"] in VALIDATED_SECTORS]
    ranked = sorted(eligible, key=lambda r: (r["er_rank"] if r["er_rank"] is not None else -1), reverse=True)
    book = ranked[:n]
    rsum = sum((r["er_rank"] or 0) for r in book) or 1.0
    for r in book:
        r["weight"] = (r["er_rank"] or 0) / rsum          # conviction weight ~ relative selection rank
    return book


def allocate(book: list[dict], capital: float) -> list[dict]:
    import yfinance as yf  # noqa: F401  (ensures yfinance import error surfaces early)

    for r in book:
        sym = f"{r['ticker']}.NS"
        px = _live_price(sym) or r["price"]
        r["entry_symbol"] = sym
        r["entry_price"] = px
        target = capital * r["weight"]
        shares = int(target // px) if px else 0
        r["shares"] = shares
        r["market_value"] = round(shares * px, 2) if px else None
    return book


def main() -> None:
    print(f"valuing {len(CANDIDATES)} India candidates (engine_v3, live)...")
    rows = value_universe()
    book = select_book(rows)
    book = allocate(book, CAPITAL_INR)
    bench_px = _live_price(BENCHMARK)
    deployed = sum(r["market_value"] or 0 for r in book)

    folder = MANAGED / BOOK_NAME
    folder.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    state = {
        "book": BOOK_NAME, "inception": today, "currency": "INR", "capital": CAPITAL_INR,
        "signal": "engine_v3_discount@india_phase_b_2026-06-15",
        "strategy": "aggressive-concentrated-topN-convictionweight",
        "engine_version": "v3", "country": "IN", "benchmark": BENCHMARK,
        "benchmark_entry": bench_px, "deployed": round(deployed, 2),
        "cash": round(CAPITAL_INR - deployed, 2),
        "holdings": [{
            "ticker": r["ticker"], "symbol": r["entry_symbol"], "sector": r["sector"],
            "weight": round(r["weight"], 4), "entry_price": r["entry_price"], "shares": r["shares"],
            "market_value": r["market_value"], "intrinsic": r["intrinsic"],
            "discount_to_intrinsic": r["discount"], "selection_rank": r["er_rank"], "verdict": r["verdict"],
        } for r in book],
        "honesty": {
            "horizon_of_edge": "~1 year (India Phase B 1y IC +0.22; energy +0.48). No 10-day edge.",
            "ten_day_expectation": "~market drift; +/-12% equally likely. Built for variance, not a +EV 10-day bet.",
            "validation_status": "PRELIMINARY / survivorship-optimistic; NOT validated alpha.",
            "concentration_risk": "single-sector (Indian oil/gas PSU) bet; crude/rupee/policy-driven short term.",
        },
    }
    (folder / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (folder / "track.json").write_text(json.dumps({"inception": today, "rows": []}, indent=2), encoding="utf-8")
    print(f"\nwrote {folder / 'state.json'}")
    print(f"deployed Rs {deployed:,.0f} / Rs {CAPITAL_INR:,.0f}  ({len(book)} names)")
    for r in book:
        print(f"  {r['ticker']:11s} w={r['weight']:5.1%}  {r['shares']:>5d} sh @ Rs {r['entry_price']:,.2f}"
              f"  = Rs {r['market_value']:,.0f}")


if __name__ == "__main__":
    main()
