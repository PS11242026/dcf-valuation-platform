"""India point-in-time backtest of engine_v3 (Phase B) — separate from the US battery.

Mirrors the US ``research_v3/v3_universe_backtest.py`` design (PIT bundle injected into
``EconomicDCFPipeline``; split-consistent shares x split-adjusted Close for valuation,
Adj Close for total-return forward windows; obs-weighted rank IC + Newey-West t + quintile
long-short via ``validation_common``), but on the **India** data layer:

- Fundamentals: ``IndiaFilingsClient.financial_periods`` (live NSE Ind-AS, disk-cached by the
  client). All periods fetched once per name, then filtered to ``report_date <= cutoff - lag``.
- Macro: ``live_india_macro()`` folded into Settings (G-Sec rf, India ERP, 25.17% tax, terminal
  band). **WACC = internal CAPM** — the valueinvesting.io provider WACC is *current-only*, not
  point-in-time, so a backtest MUST use the internal India-anchored CAPM (this is why backtest
  WACCs differ from the live provider WACCs the workbook shows).
- Sector/archetype/prior: the India tables (``india_sector_for_ticker`` / ``IN_SECTOR_PRIORS``).

HONEST LIMITATIONS (documented in the payload, not hidden):
- **Thin window.** India structured quarterly XBRL begins ~2018; the v3 engine needs ~20
  quarters of history, so the first valuable cutoff is ~2023. With "today" in 2026 the 1y
  forward window is fully observable, 2y partially, 3y only for the earliest cutoffs. The
  result is therefore PRELIMINARY/underpowered (1y-primary), exactly as the plan预期.
- **Survivorship.** The universe is the CURRENT large-cap map (``TICKER_SECTOR_IN``); free data
  has no point-in-time NIFTY/BSE membership or delisted names, so the long-short is
  survivorship-OPTIMISTIC. Flagged; the infra strengthens automatically if a delisted panel is
  added later.
- **Corporate actions.** Share re-expression uses ``validation_common``'s whitelisted INTEGER
  split factors; fractional Indian bonus ratios (e.g. 3:5) are not re-expressed (residual basis
  error on those names). Adj-Close forward returns already incorporate all actions.
- **Banks/NBFCs excluded** — a discounted-FCF model is not applicable to financials.

It also emits ``india_v3_reference.json``: the cross-sectional v3 discount distribution
(overall + per sector) at the latest cutoff — the India-relative reference the verdict layer
needs so an Indian name is ranked against Indian peers, not the US distribution.

Usage:
    PYTHONPATH=src python scripts/utilities/research_v3_india/india_v3_backtest.py --pilot
    PYTHONPATH=src python scripts/utilities/research_v3_india/india_v3_backtest.py --n 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "utilities"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import validation_common as vc  # noqa: E402

# The shared US battery only uses horizons {1,3,5}; register the 2y NW lag for the India run
# (8 quarters). Adding a key US code never reads is inert for the frozen US path.
vc.HORIZON_QUARTERS.setdefault(2, 8)

from historical_predictive_power import (  # noqa: E402
    REPORTING_LAG_DAYS,
    _latest,
    _PITPipeline,
    _price_asof,
    _quarter_ends,
)

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.contracts import MarketSnapshot, SourceMeta  # noqa: E402
from dcf_project.data_sources.fundamentals import FundamentalsBundle  # noqa: E402
from dcf_project.data_sources.india.filings import IndiaFilingsClient  # noqa: E402
from dcf_project.data_sources.india.macro import live_india_macro  # noqa: E402
from dcf_project.data_sources.india.sectors import india_sector_for_ticker  # noqa: E402
from dcf_project.data_sources.market import build_wacc  # noqa: E402
from dcf_project.engine_v3.drivers import DriverExtractionError  # noqa: E402
from dcf_project.engine_v3.pipeline import EconomicDCFPipeline  # noqa: E402
from dcf_project.paths import CACHE  # noqa: E402
from dcf_project.valuation.terminal import TerminalValueError  # noqa: E402

REPORT_DIR = ROOT / "reports" / "research_v3_india"
PRICE_CACHE = CACHE / "prices_india"
SPLIT_CACHE = CACHE / "splits_india" / "universe_splits.json"
BETA_CACHE = PRICE_CACHE / "_betas.json"
HORIZONS = (1, 2, 3)
MIN_PERIODS = 20            # engine_v3 reconstruction floor (US harness uses 24; IN window is thin)
START_YEAR = 2022          # cutoffs before ~2023 skip on insufficient_periods anyway

#: A representative liquid pilot spanning the DCF-applicable sectors.
PILOT = ["RELIANCE", "HINDUNILVR", "INFY", "TCS", "ITC", "SUNPHARMA", "MARUTI", "LT",
         "ASIANPAINT", "TITAN"]

#: Sectors a discounted-cash-flow model is not built for.
EXCLUDE_SECTORS = {"financials", "real_estate"}


def _india_universe(n: int | None) -> list[tuple[str, str]]:
    """(symbol, sector) for DCF-applicable India large caps, sorted, optionally truncated."""
    from dcf_project.data_sources.india.sectors import TICKER_SECTOR_IN

    names = sorted(sym for sym, sec in TICKER_SECTOR_IN.items() if sec not in EXCLUDE_SECTORS)
    if n is not None:
        names = names[:n]
    return [(s, india_sector_for_ticker(s)) for s in names]


class _PITPipelineV3(EconomicDCFPipeline):
    """engine_v3 pipeline that values a pre-built PIT bundle (no live data layer call)."""

    def __init__(self, settings, bundle: FundamentalsBundle):
        super().__init__(settings)
        self._bundle = bundle

    def load_data(self, ticker: str) -> FundamentalsBundle:
        return self._bundle


def _v1_intrinsic(result) -> float:
    """Distortion-removed frozen-v1 intrinsic: the state-space DCF diagnostic when a regime
    overrode the headline, else the headline per-share value (same convention as the US
    paired backtest's ``_v1_signal``). Divided by the SAME PIT price the v3 leg uses, so the
    v1 and v3 discounts share an identical denominator (a true paired comparison)."""
    diag = (result.diagnostics or {}).get("state_space_dcf_diagnostic_intrinsic")
    return float(diag) if isinstance(diag, (int, float)) else float(result.intrinsic_value_per_share)


SHARES_CACHE = PRICE_CACHE / "_shares.json"


def _yahoo_shares_cached(ticker_ns: str) -> float | None:
    """Current Yahoo sharesOutstanding (today's split basis). The NSE results XBRL does not
    carry a reliable share count, so PIT bundles fall back to this. Using today's shares with
    the today-split-basis Close is split/bonus-CONSISTENT (the product preserves market cap);
    only genuine issuance/buyback between the cutoff and today drifts (small for large caps)."""
    cache: dict[str, float] = {}
    if SHARES_CACHE.exists():
        try:
            cache = json.loads(SHARES_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}
    if ticker_ns in cache:
        return float(cache[ticker_ns]) or None
    shares = 0.0
    try:
        import yfinance as yf

        info = yf.Ticker(ticker_ns).get_info()
        s = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        shares = float(s) if s and float(s) > 0 else 0.0
    except Exception:  # noqa: BLE001
        shares = 0.0
    cache[ticker_ns] = shares
    vc._atomic_write_text(SHARES_CACHE, json.dumps(cache))
    return shares or None


def _beta_cached(ticker_ns: str) -> float:
    cache: dict[str, float] = {}
    if BETA_CACHE.exists():
        try:
            cache = json.loads(BETA_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}
    if ticker_ns in cache:
        return float(cache[ticker_ns])
    beta = 1.0
    try:
        import yfinance as yf

        b = yf.Ticker(ticker_ns).get_info().get("beta")
        beta = float(b) if b and float(b) > 0 else 1.0
    except Exception:  # noqa: BLE001
        beta = 1.0
    cache[ticker_ns] = beta
    vc._atomic_write_text(BETA_CACHE, json.dumps(cache))
    return beta


def _build_bundle(ticker, periods_tb, close_series, beta, cutoff, settings, yahoo_shares=None):
    """India PIT bundle: today-basis shares x today-basis Close, internal-CAPM WACC."""
    avail = [p for p in periods_tb if p.report_date <= cutoff - timedelta(days=REPORTING_LAG_DAYS)]
    if len(avail) < MIN_PERIODS:
        return None, "insufficient_periods"
    price = _price_asof(close_series, cutoff)
    if price is None or price <= 0:
        return None, "no_price"
    # NSE results XBRL rarely carries shares -> fall back to current Yahoo shares (split-basis
    # consistent with the today-basis Close); the engine divides equity value by this count.
    shares = _latest(avail, "shares_outstanding") or yahoo_shares
    cash = _latest(avail, "cash")
    debt = _latest(avail, "debt")
    if not shares or shares <= 0:
        return None, "no_shares"
    market = MarketSnapshot(
        ticker=ticker.upper(), price=price, market_cap=price * shares,
        shares_outstanding=shares, beta=beta, cash=cash, debt=debt, currency="INR",
        source=SourceMeta(provider="india_backtest_pit_v3", key=ticker.upper()),
    )
    latest = avail[-1]
    tax_rate = None
    if latest.tax_expense is not None and latest.ebit not in (None, 0):
        tax_rate = latest.tax_expense / abs(latest.ebit)
    try:
        wacc = build_wacc(market, tax_rate, debt, settings)
    except Exception as exc:  # noqa: BLE001
        return None, f"wacc_error:{type(exc).__name__}"
    return FundamentalsBundle(avail, market, wacc, [market.source], []), None


def _bump(skips: dict[str, int], key: str) -> None:
    skips[key] = skips.get(key, 0) + 1


def _process_ticker(symbol, sector, cutoffs, v1_settings, v3_settings, periods_client) -> dict[str, Any]:
    """Value one India name across all PIT cutoffs with BOTH engines on identical bundles
    (frozen v1 and economic v3, paired). Never raises; self-contained checkpoint payload."""
    ticker_ns = f"{symbol}.NS"
    try:
        periods_all, _ = periods_client.financial_periods(symbol)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": symbol, "rows": [], "skips": {"filings": 1},
                "outcome": f"failed:filings:{type(exc).__name__}"}
    if len(periods_all) < MIN_PERIODS:
        return {"ticker": symbol, "rows": [], "skips": {"too_few_periods": 1},
                "outcome": f"skipped:only_{len(periods_all)}_periods"}
    splits = vc.fetch_whitelisted_splits([ticker_ns], SPLIT_CACHE).get(ticker_ns, [])
    periods_tb = vc.reexpress_shares_today_basis(periods_all, splits)
    prices = vc.pit_price_frame(ticker_ns, PRICE_CACHE)
    if prices is None or not len(prices):
        return {"ticker": symbol, "rows": [], "skips": {"price": 1}, "outcome": "skipped:price"}
    close_series, adj_series = prices["close"], prices["adj_close"]
    beta = _beta_cached(ticker_ns)
    yahoo_shares = _yahoo_shares_cached(ticker_ns)
    last_price_date = prices.index[-1].date()

    rows: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    for cutoff in cutoffs:
        # The bundle (PIT periods, market, WACC) is built once with the v1 settings and shared
        # by BOTH engines, so any v3-v1 difference is forecast-driven, never data/WACC-driven.
        bundle, reason = _build_bundle(symbol, periods_tb, close_series, beta, cutoff, v1_settings,
                                       yahoo_shares=yahoo_shares)
        if bundle is None:
            _bump(skips, reason or "unknown")
            continue
        try:
            r1 = _PITPipeline(v1_settings, bundle).run(symbol)
        except Exception as exc:  # noqa: BLE001
            _bump(skips, f"v1_run:{type(exc).__name__}")
            continue
        try:
            r3 = _PITPipelineV3(v3_settings, bundle).run(symbol)
        except (DriverExtractionError, TerminalValueError) as exc:
            _bump(skips, f"v3_skip:{type(exc).__name__}")
            continue
        except Exception as exc:  # noqa: BLE001
            _bump(skips, f"v3_run:{type(exc).__name__}")
            continue

        px = bundle.market.price
        iv3 = float(r3.intrinsic_value_per_share)
        iv1 = _v1_intrinsic(r1)
        coherence = (r3.diagnostics or {}).get("valuation_coherence")
        row: dict[str, Any] = {
            "ticker": symbol, "sector": sector, "date": cutoff.isoformat(), "price": px,
            "v3_iv": iv3,
            "discount_to_intrinsic": iv3 / px - 1.0,   # the v3 signal (vc contract key)
            "v1_iv": iv1,
            "v1_discount": iv1 / px - 1.0,             # the paired frozen-v1 signal
            "wacc": float(r3.wacc),
            "terminal_growth": float(getattr(r3, "terminal_growth_rate", 0.0) or 0.0),
            "company_type": getattr(r3, "company_archetype", None),
            "coherence": coherence,
        }
        adj0 = _price_asof(adj_series, cutoff)
        for y in HORIZONS:
            target = cutoff + timedelta(days=round(365.25 * y))
            if target > last_price_date or not adj0 or adj0 <= 0:
                continue
            adj_fwd = _price_asof(adj_series, target)
            if adj_fwd and adj_fwd > 0:
                row[f"fwd_{y}y"] = float((adj_fwd / adj0) ** (1.0 / y) - 1.0)
        rows.append(row)
    return {"ticker": symbol, "rows": rows, "skips": skips, "outcome": f"ok:{len(rows)}"}


# ----------------------------- aggregation ----------------------------------- #

def _engine_metrics(rows: list[dict], signal_key: str = "discount_to_intrinsic") -> dict[str, Any]:
    return {f"{y}y": vc.core_metrics(rows, y, signal_key=signal_key, with_ci=False)
            for y in HORIZONS}


def _paired_diff(rows: list[dict]) -> dict[str, Any]:
    """Per-horizon paired v3-v1 rank-IC difference (Newey-West) on identical bundles. The India
    NW-t is inflated (thin/overlapping window) -- read the sign and magnitude, never the t."""
    out: dict[str, Any] = {}
    for y in HORIZONS:
        try:
            out[f"{y}y"] = vc.paired_ic_diff(rows, y, "discount_to_intrinsic", "v1_discount")
        except Exception:  # noqa: BLE001
            out[f"{y}y"] = {}
    return out


def _by_sector(rows: list[dict]) -> dict[str, Any]:
    secs: dict[str, list] = {}
    for o in rows:
        secs.setdefault(o.get("sector") or "unknown", []).append(o)
    return {
        sec: {"n_obs": len(sub), "n_tickers": len({o["ticker"] for o in sub}),
              "metrics": _engine_metrics(sub, "discount_to_intrinsic"),
              "metrics_v1": _engine_metrics(sub, "v1_discount"),
              "paired_ic_diff": _paired_diff(sub)}
        for sec, sub in sorted(secs.items())
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    import numpy as np

    a = np.array(sorted(values), dtype=float)
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    return {f"p{int(q * 100)}": float(np.quantile(a, q)) for q in qs} | {"mean": float(a.mean())}


def _reference_distribution(rows: list[dict]) -> dict[str, Any]:
    """Cross-sectional v3 discount distribution at the LATEST cutoff per name — the India
    relative reference the verdict layer ranks against (overall + per sector)."""
    latest: dict[str, dict] = {}
    for o in rows:
        cur = latest.get(o["ticker"])
        if cur is None or o["date"] > cur["date"]:
            latest[o["ticker"]] = o
    snap = list(latest.values())
    by_sec: dict[str, list[float]] = {}
    for o in snap:
        by_sec.setdefault(o.get("sector") or "unknown", []).append(o["discount_to_intrinsic"])
    all_d = [o["discount_to_intrinsic"] for o in snap]
    return {
        "as_of": max((o["date"] for o in snap), default=None),
        "n_names": len(snap),
        "overall": _quantiles(all_d) if all_d else {},
        "by_sector": {s: (_quantiles(v) | {"n": len(v)}) for s, v in sorted(by_sec.items()) if v},
        "names": sorted(({"ticker": o["ticker"], "sector": o["sector"],
                          "discount": round(o["discount_to_intrinsic"], 4),
                          "coherence": o.get("coherence")} for o in snap),
                        key=lambda r: r["discount"], reverse=True),
    }


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=None, help="cap the universe to the first N names")
    ap.add_argument("--pilot", action="store_true", help="run the fixed 10-name pilot")
    ap.add_argument("--no-resume", action="store_true", help="ignore per-ticker checkpoints")
    args = ap.parse_args()

    macro = live_india_macro()
    base = replace(
        get_settings(),
        wacc_source="internal", terminal_growth_source="internal",
        v3_mc_draws=0, monte_carlo_samples=0, scenario_monte_carlo_samples=0, run_sensitivity=False,
        **macro.settings_overrides(),
    )
    v1_settings = replace(base, engine_version="v1")
    v3_settings = replace(base, engine_version="v3")
    periods_client = IndiaFilingsClient(v3_settings)

    if args.pilot:
        universe = [(s, india_sector_for_ticker(s)) for s in PILOT]
        scope = "pilot"
    else:
        universe = _india_universe(args.n)
        scope = "broad" if args.n is None else f"n{args.n}"
    cutoffs = _quarter_ends(START_YEAR)

    print(f"India PAIRED v1+v3 PIT backtest: {len(universe)} names x {len(cutoffs)} cutoffs "
          f"(min_periods={MIN_PERIODS}, internal-CAPM WACC, scope={scope})", flush=True)
    print(f"  macro: rf={macro.risk_free:.3f} erp={macro.equity_risk_premium:.3f} "
          f"tax={macro.default_tax_rate:.4f} term_cap={macro.terminal_growth_cap:.3f}", flush=True)

    ckpt_dir = REPORT_DIR / "_checkpoints" / f"{scope}_v1v3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    obs: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    outcomes: dict[str, str] = {}
    started = time.time()
    for i, (symbol, sector) in enumerate(universe):
        ckpt = ckpt_dir / f"{symbol}.json"
        if ckpt.exists() and not args.no_resume:
            payload = json.loads(ckpt.read_text(encoding="utf-8"))
        else:
            payload = _process_ticker(symbol, sector, cutoffs, v1_settings, v3_settings, periods_client)
            _save(ckpt, payload)
        obs.extend(payload["rows"])
        for k, v in payload["skips"].items():
            skips[k] = skips.get(k, 0) + v
        outcomes[symbol] = payload["outcome"]
        print(f"  [{i + 1}/{len(universe)}] {symbol} ({sector}): {payload['outcome']}", flush=True)

    print(f"\ncollected {len(obs)} observations in {time.time() - started:.0f}s; computing metrics",
          flush=True)
    overall = _engine_metrics(obs, "discount_to_intrinsic")
    overall_v1 = _engine_metrics(obs, "v1_discount")
    paired_overall = _paired_diff(obs)
    by_sector = _by_sector(obs)
    reference = _reference_distribution(obs)

    payload = {
        "methodology": {
            "design": "PAIRED frozen-v1 + economic-v3 PIT backtest on the India data layer (both "
                      "engines valued in-process on IDENTICAL bundles, so any v3-v1 difference is "
                      "forecast-driven; mirrors the US v3_universe_backtest paired design)",
            "price_basis": "shares re-expressed to today's split basis (whitelisted integer factors); "
                           "valuation price = split-adjusted Close; forward returns = Adj Close (total return)",
            "wacc": "internal CAPM with India rf+ERP (PIT-safe; NOT the current-only provider WACC)",
            "signal": "discount_to_intrinsic = v3_iv/price - 1 (paired with v1_discount = v1_iv/price - 1)",
            "horizons_years": list(HORIZONS),
            "min_periods": MIN_PERIODS,
            "reporting_lag_days": REPORTING_LAG_DAYS,
            "start_year": START_YEAR,
            "limitations": [
                "PRELIMINARY/underpowered: India XBRL begins ~2018 and v3 needs ~20q, so cutoffs "
                "start ~2023 -> 1y-forward is primary, 2y partial, 3y only earliest cohorts.",
                "Survivorship-OPTIMISTIC: current large-cap universe, no PIT index membership or "
                "delisted names (free-data limit). Long-short reads high; infra strengthens if a "
                "delisted panel is added.",
                "Fractional Indian bonus ratios are not share-re-expressed (whitelist is integer "
                "split factors); Adj-Close forward returns are unaffected.",
                "Banks/NBFCs/real estate excluded (DCF not applicable).",
            ],
        },
        "scope": scope,
        "n_names": len(universe),
        "n_names_with_obs": len({o["ticker"] for o in obs}),
        "n_observations": len(obs),
        "skips": skips,
        "outcomes": outcomes,
        "overall_metrics_v3": overall,
        "overall_metrics_v1": overall_v1,
        "paired_ic_diff_overall": paired_overall,
        "by_sector": by_sector,
        "reference_distribution": reference,
        "observations": obs,
    }
    _save(REPORT_DIR / f"india_v3_backtest_{scope}.json", payload)
    _save(REPORT_DIR / "india_v3_reference.json", reference)

    def _f(v, nd=3):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "None"

    print(f"\n=== INDIA PAIRED v3 vs v1 PIT RESULTS ({scope}) ===")
    for y in HORIZONS:
        m = overall.get(f"{y}y", {})
        m1v = overall_v1.get(f"{y}y", {})
        pdf = paired_overall.get(f"{y}y", {}) or {}
        print(f"  {y}y: n={m.get('n')} v3_IC={_f(m.get('ic_obs_weighted'))} "
              f"v1_IC={_f(m1v.get('ic_obs_weighted'))} "
              f"paired(v3-v1)={_f(pdf.get('mean_ic_diff'))} NW-t={_f(pdf.get('nw_tstat'), 2)} "
              f"v3_q5-q1={_f(m.get('long_short_q5_minus_q1'))}")
    print("\n=== BY SECTOR (1y v3_IC / v1_IC / n) ===")
    for sec, block in by_sector.items():
        m1 = (block["metrics"].get("1y") or {})
        m1v = (block["metrics_v1"].get("1y") or {})
        print(f"  {sec:26} n={block['n_obs']:4} 1y v3_IC={_f(m1.get('ic_obs_weighted'))} "
              f"v1_IC={_f(m1v.get('ic_obs_weighted'))} q5-q1={_f(m1.get('long_short_q5_minus_q1'))}")
    print(f"\nreference: {reference['n_names']} names @ {reference['as_of']}; "
          f"overall median discount {_f(reference.get('overall', {}).get('p50'))}")
    print(f"skips: {skips}")
    print(f"wrote {REPORT_DIR / f'india_v3_backtest_{scope}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
