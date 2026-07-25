"""Canada PAIRED (frozen-v1 + economic-v3) point-in-time backtest — Phase B, Canada bolt-on.

Mirrors the India ``research_v3_india/india_v3_backtest.py`` design, but the fundamentals come
from the **existing SEC EDGAR pipeline** rather than a bespoke filings client: most TSX large
caps are SEC filers (full 10-K/10-Q, or MJDS 40-F / 6-K, or 20-F) and EDGAR ``companyfacts``
carries deep point-in-time quarterly history for them (empirically: most of the DCF-applicable
TSX names have >=20 usable quarters reaching back to 2011-2017). So Canada is a *light* bolt-on:

- Fundamentals: ``SECClient.financial_periods`` (same XBRL companyfacts path as the US), keyed by
  the EDGAR/US-listing symbol. True point-in-time, same ``FinancialPeriod`` contract.
- Reporting currency is **per name** and detected live from the EDGAR unit tag (USD vs CAD; each
  filer tags a single currency -- verified no dual-tagging). CAD reporters are priced on the TSX
  (``.TO``) and valued with the Canadian macro profile (Government of Canada rf, Canadian ERP,
  26.5% tax, CAD terminal band); USD reporters (global operators reporting in USD) are priced on
  their US listing and valued like a US name (USD rf + US ERP) with the Canadian statutory tax
  fallback. Either way each name's intrinsic value and price share a currency, so the discount
  ratio (and its cross-sectional rank) is well-posed.
- WACC = internal CAPM (PIT-safe). Macro/sector tables: ``data_sources.canada.{macro,sectors}``;
  sector priors fall back to the US table (developed market), via ``prior_for("CA", sector)``.

Both engines are valued in-process on the IDENTICAL bundle at every (name, cutoff), so any v3-v1
difference is forecast-driven (mirrors the US/India paired design). Honest limitations are
recorded in the payload: survivorship-optimistic (current TSX large-cap map, no delisted names),
SEC-filer subset only (TSX-only names without SEC filings are out of scope), financials/REITs
excluded (DCF not applicable).

Usage:
    PYTHONPATH=src python scripts/utilities/research_v3_canada/canada_v3_backtest.py --pilot
    PYTHONPATH=src python scripts/utilities/research_v3_canada/canada_v3_backtest.py            # broad
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

from historical_predictive_power import (  # noqa: E402
    REPORTING_LAG_DAYS,
    _latest,
    _PITPipeline,
    _price_asof,
    _quarter_ends,
)

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.contracts import MarketSnapshot, SourceMeta  # noqa: E402
from dcf_project.data_sources.canada.macro import canada_macro  # noqa: E402
from dcf_project.data_sources.canada.sectors import (  # noqa: E402
    TICKER_SECTOR_CA,
    canada_sector_for_ticker,
    currency_hint,
)
from dcf_project.data_sources.fundamentals import FundamentalsBundle  # noqa: E402
from dcf_project.data_sources.market import build_wacc  # noqa: E402
from dcf_project.data_sources.sec import SECClient, _reporting_taxonomies  # noqa: E402
from dcf_project.engine_v3.drivers import DriverExtractionError  # noqa: E402
from dcf_project.engine_v3.pipeline import EconomicDCFPipeline  # noqa: E402
from dcf_project.paths import CACHE  # noqa: E402
from dcf_project.valuation.terminal import TerminalValueError  # noqa: E402

REPORT_DIR = ROOT / "reports" / "research_v3_canada"
PRICE_CACHE = CACHE / "prices_canada"
SPLIT_CACHE = CACHE / "splits_canada" / "universe_splits.json"
BETA_CACHE = PRICE_CACHE / "_betas.json"
SHARES_CACHE = PRICE_CACHE / "_shares.json"
CCY_CACHE = PRICE_CACHE / "_currency.json"
HORIZONS = (1, 3, 5)
MIN_PERIODS = 20            # engine_v3 reconstruction floor (US harness uses 24; keep coverage)
START_YEAR = 2015           # deep SEC history -> long panel with 3y/5y forward windows

#: A representative liquid pilot spanning the DCF-applicable Canadian sectors + both currencies.
PILOT = ["ENB", "CNQ", "GOLD", "AEM", "CP", "WCN", "OTEX", "QSR", "MGA", "FTS"]


class _PITPipelineV3(EconomicDCFPipeline):
    """engine_v3 pipeline that values a pre-built PIT bundle (no live data layer call)."""

    def __init__(self, settings, bundle: FundamentalsBundle):
        super().__init__(settings)
        self._bundle = bundle

    def load_data(self, ticker: str) -> FundamentalsBundle:
        return self._bundle


def _v1_intrinsic(result) -> float:
    """Distortion-removed frozen-v1 intrinsic (state-space DCF diagnostic when a regime
    overrode the headline, else the headline per-share value) -- same convention as the US
    paired backtest. Divided by the SAME PIT price the v3 leg uses (true paired denominator)."""
    diag = (result.diagnostics or {}).get("state_space_dcf_diagnostic_intrinsic")
    return float(diag) if isinstance(diag, (int, float)) else float(result.intrinsic_value_per_share)


def _cached(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _reporting_currency(client: SECClient, symbol: str) -> str:
    """Authoritative reporting currency (``CAD``/``USD``) from the EDGAR unit tag, cached to
    disk. Each Canadian filer tags a single monetary currency, so the first monetary unit on a
    revenue/assets concept is decisive. Falls back to the curated hint if facts are unavailable."""
    cache = _cached(CCY_CACHE)
    if symbol in cache:
        return cache[symbol]
    ccy = None
    try:
        facts, _ = client.company_facts(symbol)
        taxonomies = _reporting_taxonomies(facts)
        probe = ("Revenues", "Revenue", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Assets", "OperatingIncomeLoss", "ProfitLoss")
        for taxname in ("ifrs-full", "us-gaap"):
            tx = taxonomies.get(taxname, {})
            for concept in probe:
                for unit in (tx.get(concept, {}) or {}).get("units", {}):
                    if unit in ("USD", "CAD"):
                        ccy = unit
                        break
                if ccy:
                    break
            if ccy:
                break
    except Exception:  # noqa: BLE001
        ccy = None
    ccy = ccy or currency_hint(symbol)
    cache[symbol] = ccy
    vc._atomic_write_text(CCY_CACHE, json.dumps(cache))
    return ccy


def _yahoo_meta_cached(price_ticker: str, cache_path: Path, key: str) -> float | None:
    """Cached Yahoo scalar (shares / beta) for the price listing. Listing-currency-independent
    for shares; beta is taken from the priced listing."""
    cache = _cached(cache_path)
    if price_ticker in cache:
        v = float(cache[price_ticker])
        return v or None
    val = 0.0
    try:
        import yfinance as yf

        info = yf.Ticker(price_ticker).get_info()
        raw = (info.get(key) or (info.get("impliedSharesOutstanding") if key == "sharesOutstanding"
                                 else None))
        val = float(raw) if raw and float(raw) > 0 else (1.0 if key == "beta" else 0.0)
    except Exception:  # noqa: BLE001
        val = 1.0 if key == "beta" else 0.0
    cache[price_ticker] = val
    vc._atomic_write_text(cache_path, json.dumps(cache))
    return val or None


def _build_bundle(symbol, ccy, periods_tb, close_series, beta, cutoff, settings, yahoo_shares=None):
    """Canada PIT bundle: today-basis shares x today-basis (listing-currency) Close, internal WACC."""
    avail = [p for p in periods_tb if p.report_date <= cutoff - timedelta(days=REPORTING_LAG_DAYS)]
    if len(avail) < MIN_PERIODS:
        return None, "insufficient_periods"
    price = _price_asof(close_series, cutoff)
    if price is None or price <= 0:
        return None, "no_price"
    shares = _latest(avail, "shares_outstanding") or yahoo_shares
    cash = _latest(avail, "cash")
    debt = _latest(avail, "debt")
    if not shares or shares <= 0:
        return None, "no_shares"
    market = MarketSnapshot(
        ticker=symbol.upper(), price=price, market_cap=price * shares,
        shares_outstanding=shares, beta=beta, cash=cash, debt=debt, currency=ccy,
        source=SourceMeta(provider="canada_backtest_pit", key=symbol.upper()),
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


def _process_ticker(symbol, sector, cutoffs, settings_by_ccy, sec_client) -> dict[str, Any]:
    """Value one Canadian name across all PIT cutoffs with BOTH engines on identical bundles.
    Never raises; self-contained checkpoint payload."""
    try:
        periods_all, _ = sec_client.financial_periods(symbol)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": symbol, "rows": [], "skips": {"sec": 1},
                "outcome": f"failed:sec:{type(exc).__name__}"}
    if len(periods_all) < MIN_PERIODS:
        return {"ticker": symbol, "rows": [], "skips": {"too_few_periods": 1},
                "outcome": f"skipped:only_{len(periods_all)}_periods"}
    ccy = _reporting_currency(sec_client, symbol)
    (v1_settings, v3_settings) = settings_by_ccy[ccy]
    price_ticker = f"{symbol}.TO" if ccy == "CAD" else symbol

    splits = vc.fetch_whitelisted_splits([price_ticker], SPLIT_CACHE).get(price_ticker, [])
    periods_tb = vc.reexpress_shares_today_basis(periods_all, splits)
    prices = vc.pit_price_frame(price_ticker, PRICE_CACHE)
    if prices is None or not len(prices):
        return {"ticker": symbol, "rows": [], "skips": {"price": 1},
                "outcome": f"skipped:price:{price_ticker}"}
    close_series, adj_series = prices["close"], prices["adj_close"]
    beta = _yahoo_meta_cached(price_ticker, BETA_CACHE, "beta") or 1.0
    yahoo_shares = _yahoo_meta_cached(price_ticker, SHARES_CACHE, "sharesOutstanding")
    last_price_date = prices.index[-1].date()

    rows: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    for cutoff in cutoffs:
        bundle, reason = _build_bundle(symbol, ccy, periods_tb, close_series, beta, cutoff,
                                       v1_settings, yahoo_shares=yahoo_shares)
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
            "ticker": symbol, "sector": sector, "currency": ccy, "date": cutoff.isoformat(),
            "price": px, "v3_iv": iv3,
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
    return {"ticker": symbol, "rows": rows, "skips": skips,
            "outcome": f"ok:{len(rows)}:{ccy}"}


# ----------------------------- aggregation ----------------------------------- #

def _engine_metrics(rows: list[dict], signal_key: str = "discount_to_intrinsic") -> dict[str, Any]:
    return {f"{y}y": vc.core_metrics(rows, y, signal_key=signal_key, with_ci=False)
            for y in HORIZONS}


def _paired_diff(rows: list[dict]) -> dict[str, Any]:
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
    """Cross-sectional v3 discount distribution at the LATEST cutoff per name (overall + sector)."""
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
        "names": sorted(({"ticker": o["ticker"], "sector": o["sector"], "currency": o.get("currency"),
                          "discount": round(o["discount_to_intrinsic"], 4),
                          "coherence": o.get("coherence")} for o in snap),
                        key=lambda r: r["discount"], reverse=True),
    }


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _settings_by_ccy() -> dict[str, tuple]:
    """(v1, v3) Settings per reporting currency. CAD names take the full Canadian macro profile;
    USD names keep US rf/ERP/terminal (USD cash flows) with the Canadian statutory tax fallback."""
    macro = canada_macro()
    base = replace(
        get_settings(),
        wacc_source="internal", terminal_growth_source="internal",
        v3_mc_draws=0, monte_carlo_samples=0, scenario_monte_carlo_samples=0, run_sensitivity=False,
    )
    out: dict[str, tuple] = {}
    for ccy, is_cad in (("CAD", True), ("USD", False)):
        b = replace(base, **macro.settings_overrides(currency_is_cad=is_cad))
        out[ccy] = (replace(b, engine_version="v1"), replace(b, engine_version="v3"))
    return out


def _ca_universe(n: int | None) -> list[tuple[str, str]]:
    names = sorted(TICKER_SECTOR_CA)
    if n is not None:
        names = names[:n]
    return [(s, canada_sector_for_ticker(s)) for s in names]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=None, help="cap the universe to the first N names")
    ap.add_argument("--pilot", action="store_true", help="run the fixed 10-name pilot")
    ap.add_argument("--no-resume", action="store_true", help="ignore per-ticker checkpoints")
    args = ap.parse_args()

    settings_by_ccy = _settings_by_ccy()
    sec_client = SECClient(get_settings())
    macro = canada_macro()

    if args.pilot:
        universe = [(s, canada_sector_for_ticker(s)) for s in PILOT]
        scope = "pilot"
    else:
        universe = _ca_universe(args.n)
        scope = "broad" if args.n is None else f"n{args.n}"
    cutoffs = _quarter_ends(START_YEAR)

    print(f"Canada PAIRED v1+v3 PIT backtest: {len(universe)} names x {len(cutoffs)} cutoffs "
          f"(min_periods={MIN_PERIODS}, internal-CAPM WACC, scope={scope})", flush=True)
    print(f"  CAD macro: rf={macro.risk_free:.3f} erp={macro.equity_risk_premium:.3f} "
          f"tax={macro.default_tax_rate:.4f} term_cap={macro.terminal_growth_cap:.3f}; "
          f"USD names valued on US rf/ERP", flush=True)

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
            payload = _process_ticker(symbol, sector, cutoffs, settings_by_ccy, sec_client)
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
            "design": "PAIRED frozen-v1 + economic-v3 PIT backtest on the Canada (TSX) data layer; "
                      "fundamentals via SEC EDGAR companyfacts (same XBRL pipeline as the US); both "
                      "engines valued in-process on IDENTICAL bundles",
            "fundamentals": "SEC EDGAR companyfacts (SEC-filer TSX names), keyed by US-listing symbol",
            "currency": "per-name reporting currency detected from the EDGAR unit tag; CAD reporters "
                        "priced on .TO + Canadian macro, USD reporters priced on US listing + US macro",
            "price_basis": "shares re-expressed to today's split basis; valuation price = split-adjusted "
                           "Close (listing currency); forward returns = Adj Close (total return)",
            "wacc": "internal CAPM; CAD: GoC rf + Canadian ERP; USD: US rf + US ERP; tax = Canadian "
                    "statutory fallback (effective tax used when available)",
            "signal": "discount_to_intrinsic = v3_iv/price - 1 (paired with v1_discount = v1_iv/price - 1)",
            "horizons_years": list(HORIZONS),
            "min_periods": MIN_PERIODS,
            "reporting_lag_days": REPORTING_LAG_DAYS,
            "start_year": START_YEAR,
            "limitations": [
                "Survivorship-OPTIMISTIC: current TSX large-cap map, no delisted names / PIT index "
                "membership (free-data limit).",
                "SEC-filer subset: TSX-only names without SEC filings are out of scope; the universe "
                "skews to the larger, cross-listed Canadian companies.",
                "Financials/REITs excluded (DCF not applicable).",
                "Mixed-currency cross-section: each name's discount is currency-internally consistent "
                "(intrinsic and price share a currency), so the cross-sectional RANK is well-posed; "
                "absolute discounts are not comparable across the CAD/USD split.",
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
    _save(REPORT_DIR / f"canada_v3_backtest_{scope}.json", payload)
    _save(REPORT_DIR / "canada_v3_reference.json", reference)

    def _f(v, nd=3):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "None"

    print(f"\n=== CANADA PAIRED v3 vs v1 PIT RESULTS ({scope}) ===")
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
    print(f"wrote {REPORT_DIR / f'canada_v3_backtest_{scope}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
