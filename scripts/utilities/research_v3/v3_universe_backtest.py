"""Split-consistent PAIRED point-in-time backtest: engine_v3 vs frozen v1 (Phase 5).

Split consistency BY CONSTRUCTION (not post-hoc rescaling): every period's shares are
re-expressed on today's split basis (validation_common.reexpress_shares_today_basis,
whitelisted factors), valuation price is the split-adjusted (non-dividend-adjusted)
Close, and forward returns come from Adj Close (total return). This reproduces the
research_v2 validated "fully corrected" semantics and removes the defect that
overturned the original broad validation.

PAIRED design: v1 and v3 are valued in-process on the IDENTICAL bundle at every
(ticker, cutoff) -- the only fair comparison, and the per-date IC difference gets a
Newey-West t-stat (validation_common.paired_ic_diff). v3 Monte Carlo runs with
--mc-draws (default 500) so the G6 calibration gate has p10/p90 bands per observation.

Tolerant + resumable: per-ticker checkpoints under reports/research_v3/_checkpoints/,
SEC disk cache ON (first run network-bound, reruns fast), run-tracker heartbeats at
reports/research_v3/run_tracker.json (readable via `python -m dcf_project research
status --state <path>` or directly).

Usage:
    PYTHONPATH=src python scripts/utilities/research_v3/v3_universe_backtest.py --n 50
    PYTHONPATH=src python scripts/utilities/research_v3/v3_universe_backtest.py --broad
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "utilities"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scipy import stats as _scipy_stats  # noqa: E402

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
from dcf_project.data_sources.fundamentals import FundamentalsBundle  # noqa: E402
from dcf_project.data_sources.market import build_wacc  # noqa: E402
from dcf_project.data_sources.sec import SECClient, SECDataError  # noqa: E402
from dcf_project.engine_v2.sector_map import sector_for_ticker  # noqa: E402
from dcf_project.engine_v3.drivers import DriverExtractionError  # noqa: E402
from dcf_project.engine_v3.pipeline import EconomicDCFPipeline  # noqa: E402
from dcf_project.paths import CACHE  # noqa: E402
from dcf_project.research.run_tracker import RunTracker  # noqa: E402
from dcf_project.valuation.terminal import TerminalValueError  # noqa: E402

REPORT_DIR = ROOT / "reports" / "research_v3"
PRICE_CACHE = CACHE / "prices"
SPLIT_CACHE = CACHE / "splits" / "universe_splits.json"
SPLIT_SEED_CACHES = (ROOT / "reports" / "research_v2" / "_cache" / "universe_splits.json",)
BETA_CACHE = PRICE_CACHE / "_betas.json"
HORIZONS = (1, 3, 5)

# The research_v2 diversified tranche (copied frozen from
# scripts/utilities/research_v2/forecasting_compare.py TRANCHE so the v3 n=50/n=100
# gates run on the SAME names the v2 program used).
TRANCHE = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "KO", "COST", "V", "ADBE",
    "ORCL", "CRM", "PEP", "WMT", "HD", "MA", "AMD", "INTC", "TXN", "NKE",
    "MCD", "SBUX", "QCOM", "CSCO", "ACN", "LIN", "LOW", "TGT", "DHR", "TMO",
    "ABT", "NOW", "INTU", "AMAT", "LRCX", "KLAC", "MU", "PG", "CL", "GIS",
    "EL", "MDLZ", "KMB", "SYK", "ISRG", "REGN", "VRTX", "BKNG", "EBAY", "ETSY",
    "AVGO", "ADI", "MCHP", "NXPI", "MRVL", "ON", "TER", "ENTG", "SWKS", "MPWR",
    "ADSK", "ANSS", "CDNS", "SNPS", "WDAY", "PANW", "FTNT", "PYPL", "EA", "TTWO",
    "TJX", "ROST", "ORLY", "AZO", "YUM", "CMG", "LULU", "DPZ", "MAR", "HLT",
    "KHC", "STZ", "KDP", "HSY", "MKC", "CHD", "CLX", "K", "SJM", "HRL",
    "ODFL", "FAST", "PAYX", "ADP", "VRSK", "CTAS", "ROL", "CPRT", "JKHY", "EXPD",
]


class _PITPipelineV3(EconomicDCFPipeline):
    def __init__(self, settings, bundle: FundamentalsBundle):
        super().__init__(settings)
        self._bundle = bundle

    def load_data(self, ticker: str) -> FundamentalsBundle:
        return self._bundle


def _sec_periods_with_retry(ticker: str, settings, attempts: int = 6,
                            base_delay: float = 8.0, on_retry=None):
    """SEC fetch with exponential backoff for *transient transport* failures only
    (copied from research_v2 forecasting_compare; with the disk cache ON this only
    matters on the first cold pass).

    ``SECDataError`` (no CIK for the ticker / no usable GAAP-IFRS facts) is PERMANENT --
    the data does not exist and never will, so retrying it is pointless and, with the
    8/16/32/64/128/256s backoff, burns ~8.4 minutes of pure sleep per such name. Many
    recent software/SPAC tickers in the broad universe aren't in SEC's company map, so
    that dead sleep is what made the broad run look frozen for minutes at a time. Raise
    those immediately; back off only on genuine HTTP/connection/timeout errors.
    """
    delay = base_delay
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return SECClient(settings).financial_periods(ticker)
        except SECDataError:
            raise  # permanent: no point retrying, no backoff
        except Exception as exc:  # noqa: BLE001 -- transient transport (429/503/timeout)
            last = exc
            if attempt == attempts:
                break
            if on_retry is not None:
                on_retry(ticker, attempt)
            time.sleep(delay)
            delay *= 2.0
    raise last  # type: ignore[misc]


def _beta_cached(ticker: str) -> float:
    cache: dict[str, float] = {}
    if BETA_CACHE.exists():
        try:
            cache = json.loads(BETA_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}
    if ticker in cache:
        return float(cache[ticker])
    beta = 1.0
    try:
        import yfinance as yf

        b = yf.Ticker(ticker).get_info().get("beta")
        beta = float(b) if b and float(b) > 0 else 1.0
    except Exception:  # noqa: BLE001
        beta = 1.0
    cache[ticker] = beta
    vc._atomic_write_text(BETA_CACHE, json.dumps(cache))
    return beta


def _v1_signal(result) -> float:
    """Distortion-removed v1 discount (state-space DCF diagnostic when a regime
    overrode the headline) -- same convention as the v2 research program."""
    diag = result.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
    iv = float(diag) if isinstance(diag, (int, float)) else float(result.intrinsic_value_per_share)
    return iv / result.market_price - 1.0 if result.market_price else float("nan")


def _build_bundle(ticker, periods_tb, close_series, beta, cutoff, settings):
    """Split-consistent PIT bundle: today-basis shares x today-basis Close price."""
    avail = [p for p in periods_tb if p.report_date <= cutoff - timedelta(days=REPORTING_LAG_DAYS)]
    if len(avail) < 24:
        return None, "insufficient_periods"
    price = _price_asof(close_series, cutoff)
    if price is None or price <= 0:
        return None, "no_price"
    shares = _latest(avail, "shares_outstanding")
    cash = _latest(avail, "cash")
    debt = _latest(avail, "debt")
    if not shares or shares <= 0:
        return None, "no_shares"
    market = MarketSnapshot(
        ticker=ticker.upper(),
        price=price,
        market_cap=price * shares,
        shares_outstanding=shares,
        beta=beta,
        cash=cash,
        debt=debt,
        source=SourceMeta(provider="backtest_pit_v3", key=ticker.upper()),
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


def _v3_extras(result) -> dict[str, Any]:
    audit = result.diagnostics.get("driver_audit", {})
    persistence = audit.get("persistence", {})
    drivers = audit.get("drivers", {})
    p10, p50, p90 = result.valuation_p10, result.valuation_p50, result.valuation_p90
    band_width = None
    if p50 and p90 > p10:
        band_width = (p90 - p10) / abs(p50)
    return {
        "v3_p10": p10, "v3_p50": p50, "v3_p90": p90,
        "v3_band_width": band_width,
        "v3_jensen_gap": result.monte_carlo_distribution.get("jensen_gap"),
        "v3_phi": persistence.get("phi"),
        "v3_cap_years": persistence.get("cap_years"),
        "v3_excess_roic_0": drivers.get("excess_roic_0"),
        "v3_quality": drivers.get("quality"),
        "v3_pv_terminal_share": audit.get("pv_terminal_share_of_ev"),
        "v3_n_band_breaches": len(audit.get("fcf_margin_breach_quarters") or []),
        "v3_n_growth_binds": len(audit.get("growth_binds_quarters") or []),
        "v3_company_type": result.company_archetype,
    }


def _process_ticker(ticker, family, cutoffs, v1_settings, v3_settings, splits_events,
                    on_sec_retry=None, heartbeat=None) -> dict[str, Any]:
    """Value one ticker across all PIT cutoffs with BOTH engines on identical bundles.
    Self-contained checkpoint payload; never raises."""
    try:
        periods_all, _ = _sec_periods_with_retry(ticker, v1_settings, on_retry=on_sec_retry)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "rows": [], "skips": {"sec": 1},
                "outcome": f"failed:sec:{type(exc).__name__}"}
    periods_tb = vc.reexpress_shares_today_basis(periods_all, splits_events)
    prices = vc.pit_price_frame(ticker, PRICE_CACHE)
    if prices is None or not len(prices):
        return {"ticker": ticker, "rows": [], "skips": {"price": 1}, "outcome": "skipped:price"}
    close_series, adj_series = prices["close"], prices["adj_close"]
    beta = _beta_cached(ticker)
    last_price_date = prices.index[-1].date()

    rows: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    for cutoff in cutoffs:
        if heartbeat is not None:
            heartbeat(ticker, f"cutoff {cutoff.isoformat()}")
        bundle, reason = _build_bundle(ticker, periods_tb, close_series, beta, cutoff, v1_settings)
        if bundle is None:
            _bump(skips, reason or "unknown")
            continue
        try:
            r1 = _PITPipeline(v1_settings, bundle).run(ticker)
        except Exception as exc:  # noqa: BLE001
            _bump(skips, f"v1_run:{type(exc).__name__}")
            continue
        try:
            r3 = _PITPipelineV3(v3_settings, bundle).run(ticker)
        except DriverExtractionError as exc:
            _bump(skips, "v3_not_applicable" if str(exc).startswith("v3_not_applicable")
                  else f"v3_skip:{str(exc)[:40]}")
            continue
        except TerminalValueError:
            _bump(skips, "v3_skip:terminal_spread")
            continue
        except Exception as exc:  # noqa: BLE001
            _bump(skips, f"v3_run:{type(exc).__name__}")
            continue

        px = bundle.market.price
        iv3 = float(r3.intrinsic_value_per_share)
        row: dict[str, Any] = {
            "ticker": ticker,
            "sector": family,
            "date": cutoff.isoformat(),
            "price": px,
            "cum_split": vc.cum_split_factor_after(splits_events, cutoff - timedelta(days=REPORTING_LAG_DAYS)),
            "v3_iv": iv3,
            "discount_to_intrinsic": iv3 / px - 1.0,  # the V3 signal (harness contract key)
            "v1_discount": _v1_signal(r1),
            "v1_iv": float(r1.intrinsic_value_per_share),
            "company_type": r1.diagnostics.get("resolved_company_type")
            or r1.diagnostics.get("company_type"),
            "wacc": float(r3.wacc),
            **_v3_extras(r3),
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
    return {"ticker": ticker, "rows": rows, "skips": skips, "outcome": f"ok:{len(rows)}"}


# ----------------------------- aggregation / gate ----------------------------- #

def _engine_metrics(rows: list[dict], signal_key: str, with_ci: bool) -> dict[str, Any]:
    return {f"{y}y": vc.core_metrics(rows, y, signal_key=signal_key, with_ci=with_ci) for y in HORIZONS}


def _family_blocks(rows: list[dict]) -> dict[str, Any]:
    fams: dict[str, list] = {}
    for o in rows:
        fams.setdefault(o.get("sector") or "unknown", []).append(o)
    out: dict[str, Any] = {}
    for fam, sub in sorted(fams.items()):
        out[fam] = {
            "n_obs": len(sub),
            "n_tickers": len({o["ticker"] for o in sub}),
            "v3": _engine_metrics(sub, "discount_to_intrinsic", with_ci=False),
            "v1_paired": _engine_metrics(sub, "v1_discount", with_ci=False),
            "paired_ic_diff": {f"{y}y": vc.paired_ic_diff(sub, y, "discount_to_intrinsic", "v1_discount")
                               for y in (3, 5)},
        }
    return out


def _gate_g2(overall_v3, overall_v1, paired_overall, families) -> dict[str, Any]:
    """Pre-registered G2 gate. PASS: v3 3y IC >= paired v1 in reliable families AND the
    overall per-date IC diff is not significantly negative (NW t > -2). KILL: staples
    3y IC < +0.10 or broad 3y IC < -0.05 (obs-weighted ICs)."""
    def ic3(block):
        return (block.get("3y") or {}).get("ic_obs_weighted")

    broad_v3 = ic3(overall_v3)
    staples = families.get("consumer_staples", {})
    staples_v3 = ic3(staples.get("v3", {}))
    staples_v1 = ic3(staples.get("v1_paired", {}))

    reliable = {}
    for fam, block in families.items():
        v1_ic = ic3(block.get("v1_paired", {}))
        if fam == "consumer_staples" or (isinstance(v1_ic, (int, float)) and v1_ic > 0):
            reliable[fam] = {"v3_3y_ic": ic3(block.get("v3", {})), "v1_3y_ic": v1_ic}

    cond_reliable = all(
        isinstance(v["v3_3y_ic"], (int, float)) and isinstance(v["v1_3y_ic"], (int, float))
        and v["v3_3y_ic"] >= v["v1_3y_ic"]
        for v in reliable.values()
    ) if reliable else None
    t_overall = (paired_overall.get("3y") or {}).get("nw_tstat")
    cond_not_below = (t_overall is None) or (t_overall > -2.0)

    kill_staples = isinstance(staples_v3, (int, float)) and staples_v3 < 0.10
    kill_broad = isinstance(broad_v3, (int, float)) and broad_v3 < -0.05
    killed = bool(kill_staples or kill_broad)
    passed = bool(cond_reliable) and cond_not_below and not killed
    return {
        "criteria": {
            "pass": "v3 3y IC >= paired v1 in reliable families AND overall paired NW-t > -2",
            "kill": "staples 3y IC < +0.10 OR broad 3y IC < -0.05 (obs-weighted)",
        },
        "inputs": {
            "broad_v3_3y_ic_obs": broad_v3,
            "staples_v3_3y_ic_obs": staples_v3,
            "staples_v1_3y_ic_obs": staples_v1,
            "overall_paired_3y_nw_t": t_overall,
            "reliable_families": reliable,
        },
        "conditions": {
            "v3_at_least_v1_in_reliable": cond_reliable,
            "overall_not_significantly_below_v1": cond_not_below,
            "kill_staples_triggered": kill_staples,
            "kill_broad_triggered": kill_broad,
        },
        "verdict": "KILL" if killed else ("PASS" if passed else "FAIL"),
        "note": "Gate numbers only; the G2 decision is recorded in the research journal.",
    }


def _save(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out_path)


def _family(ticker: str, family_of: dict[str, str]) -> str:
    """Universe family when the name is a universe member; else the static sector map
    (tranche mega-caps like AAPL/MSFT are not universe members). consumer_staples means
    the same cohort in both taxonomies, so the G2 staples gate is unaffected."""
    return family_of.get(ticker) or sector_for_ticker(ticker) or "unknown"


def main() -> int:
    # A 2-3-name cross-section makes the block bootstrap degenerate (constant ranks);
    # the >=4-names-per-date IC gate already handles it -- silence the scipy noise.
    warnings.filterwarnings("ignore", category=_scipy_stats.ConstantInputWarning)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50, help="names from the frozen research tranche")
    ap.add_argument("--broad", action="store_true", help="run the full ~500-name universe instead")
    ap.add_argument("--mc-draws", type=int, default=500, help="v3 MC draws per valuation (0 = skeleton only)")
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--no-resume", action="store_true", help="ignore existing per-ticker checkpoints")
    args = ap.parse_args()

    universe_meta = json.loads((ROOT / "reports" / "_universe_tickers.json").read_text(encoding="utf-8"))
    family_of: dict[str, str] = universe_meta["family_of"]
    if args.broad:
        tickers = list(universe_meta["tickers"])
        scope = "broad"
    else:
        tickers = TRANCHE[: args.n]
        scope = f"n{len(tickers)}"

    base = replace(
        get_settings(),
        wacc_source="internal", terminal_growth_source="internal",
        forecast_mode="state_space", consensus_weight=0.0,
        monte_carlo_samples=0, scenario_monte_carlo_samples=0, run_sensitivity=False,
        sec_cache_enabled=True,
    )
    v1_settings = base
    v3_settings = replace(base, engine_version="v3", v3_mc_draws=args.mc_draws)

    cutoffs = _quarter_ends(args.start_year)
    out_path = REPORT_DIR / f"v3_universe_backtest_{scope}.json"
    ckpt_dir = REPORT_DIR / "_checkpoints" / f"v3_{scope}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"v3 paired backtest: {len(tickers)} tickers x {len(cutoffs)} cutoffs "
          f"(mc_draws={args.mc_draws}, scope={scope})", flush=True)
    splits = vc.fetch_whitelisted_splits(tickers, SPLIT_CACHE, seed_caches=SPLIT_SEED_CACHES)
    n_splitters = sum(1 for t in tickers if splits.get(t))
    print(f"  {n_splitters} tickers with >=1 whitelisted split", flush=True)

    tracker = RunTracker.start(
        run_id=f"v3_backtest_{scope}_{date.today().isoformat()}",
        command=" ".join(sys.argv),
        phase="v3_paired_backtest",
        tranche_size=len(tickers),
        output_path=str(out_path.relative_to(ROOT)),
        state_path=REPORT_DIR / "run_tracker.json",
        log_path=REPORT_DIR / "run_tracker.log",
    )

    obs: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    outcomes: dict[str, str] = {}
    started = time.time()
    try:
        for i, ticker in enumerate(tickers):
            fam = _family(ticker, family_of)
            ckpt = ckpt_dir / f"{ticker}.json"
            if ckpt.exists() and not args.no_resume:
                payload = json.loads(ckpt.read_text(encoding="utf-8"))
            else:
                payload = _process_ticker(
                    ticker, fam, cutoffs, v1_settings, v3_settings, splits.get(ticker, []),
                    on_sec_retry=lambda t, a: tracker.heartbeat(t, f"sec retry {a}"),
                    heartbeat=lambda t, d: tracker.heartbeat(t, d),
                )
                _save(ckpt, payload)
            obs.extend(payload["rows"])
            for k, v in payload["skips"].items():
                skips[k] = skips.get(k, 0) + v
            outcomes[ticker] = payload["outcome"]
            if payload["outcome"].startswith("ok"):
                tracker.mark_completed(ticker)
            else:
                tracker.mark_failed(ticker, payload["outcome"])
            print(f"  [{i + 1}/{len(tickers)}] {ticker} ({fam}): {payload['outcome']}", flush=True)
    except KeyboardInterrupt:
        tracker.finish(status="paused", exit_code=130)
        print("interrupted -- checkpoints retained, rerun to resume", flush=True)
        return 130

    print(f"collected {len(obs)} paired observations in {time.time() - started:.0f}s; computing metrics", flush=True)
    overall_v3 = _engine_metrics(obs, "discount_to_intrinsic", with_ci=True)
    overall_v1 = _engine_metrics(obs, "v1_discount", with_ci=True)
    paired_overall = {f"{y}y": vc.paired_ic_diff(obs, y, "discount_to_intrinsic", "v1_discount")
                      for y in HORIZONS}
    families = _family_blocks(obs)
    gate = _gate_g2(overall_v3, overall_v1, paired_overall, families)

    payload = {
        "methodology": {
            "design": "paired in-process v3-vs-v1 on identical PIT bundles",
            "price_basis": "shares re-expressed on today's split basis (whitelisted factors, "
                           "per-period report_date); valuation price = split-adjusted Close "
                           "(auto_adjust=False); forward returns = Adj Close (total return)",
            "signal": "discount_to_intrinsic = v3_iv/price - 1 (v1_discount = paired v1 signal)",
            "reporting_lag_days": REPORTING_LAG_DAYS,
            "engines": {"v1": "frozen canonical (state_space, internal WACC/terminal)",
                        "v3": f"economic_fade, mc_draws={args.mc_draws}"},
            "start_year": args.start_year,
        },
        "n_tickers": len(tickers),
        "n_tickers_with_obs": len({o['ticker'] for o in obs}),
        "n_observations": len(obs),
        "skips": skips,
        "outcomes": outcomes,
        "overall_metrics_v3": overall_v3,
        "overall_metrics_v1_paired": overall_v1,
        "paired_ic_diff_overall": paired_overall,
        "by_family": families,
        "gate_g2": gate,
        "observations": obs,
    }
    _save(out_path, payload)
    summary = {k: payload[k] for k in ("n_tickers", "n_tickers_with_obs", "n_observations", "skips",
                                       "paired_ic_diff_overall", "gate_g2")}
    summary["headline"] = {
        f"{y}y": {
            "v3_ic_obs": (overall_v3.get(f"{y}y") or {}).get("ic_obs_weighted"),
            "v1_ic_obs": (overall_v1.get(f"{y}y") or {}).get("ic_obs_weighted"),
            "v3_ls_q5_q1": (overall_v3.get(f"{y}y") or {}).get("long_short_q5_minus_q1"),
        } for y in HORIZONS
    }
    _save(REPORT_DIR / f"v3_paired_summary_{scope}.json", summary)
    tracker.finish(status="completed", exit_code=0)

    def _f(v, nd=3):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "None"

    print(f"\n=== PAIRED RESULTS ({scope}) ===")
    for y in HORIZONS:
        m3, m1 = overall_v3.get(f"{y}y", {}), overall_v1.get(f"{y}y", {})
        pd_ = paired_overall.get(f"{y}y", {})
        print(f"  {y}y: n={m3.get('n')} v3 IC={_f(m3.get('ic_obs_weighted'))} "
              f"v1 IC={_f(m1.get('ic_obs_weighted'))} diff NW-t={_f(pd_.get('nw_tstat'), 2)} "
              f"v3 L/S={_f(m3.get('long_short_q5_minus_q1'))}")
    print("\n=== BY FAMILY (3y v3 IC / v1 IC / diff t) ===")
    for fam, block in families.items():
        ic3 = (block["v3"].get("3y") or {}).get("ic_obs_weighted")
        ic1 = (block["v1_paired"].get("3y") or {}).get("ic_obs_weighted")
        t = (block["paired_ic_diff"].get("3y") or {}).get("nw_tstat")
        print(f"  {fam:36} n={block['n_obs']:5} v3={_f(ic3)} v1={_f(ic1)} t={_f(t, 2)}")
    print(f"\nskips: {skips}")
    print(f"GATE G2: {gate['verdict']}  inputs={json.dumps(gate['inputs'], default=str)[:200]}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
