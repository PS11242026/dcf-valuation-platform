"""Tri-engine split-consistent PIT backtest: generic / canonical v1 / Fundamentals Fade v3.

A thin extension of ``v3_universe_backtest`` (whose split-consistent PIT bundle builder,
PIT pipelines and metric helpers are reused verbatim -- nothing about the v1/v3 math is
re-implemented here) that adds the third, textbook leg and a pathological-value guard so a
single blow-up name (e.g. MCD's negative-equity ~$172M/share) cannot dominate the ranking
metrics.

Design (unchanged from the paired harness, plus one leg):
  * All three engines are valued on the IDENTICAL PIT bundle at every (ticker, cutoff):
    today-basis shares x split-adjusted Close, forward returns from Adj Close (total return).
  * Headline is DETERMINISTIC: v3 runs mc_draws=0 (the MC layer failed calibration gate G6
    and never drives the headline). MC can be re-added later purely as a labeled diagnostic.
  * OUTLIER GUARD: a (ticker, cutoff) observation is dropped from the metric set iff ANY of
    the three engines returns a pathological intrinsic (non-finite, <= 0, or > GUARD_RATIO x
    price). Dropping the whole row keeps all three engines on an identical observation set
    (a fair, truly-paired comparison). Every drop is logged separately (engine, ticker,
    date, iv, price) and NEVER silently discarded.

No engine math is changed and v3 is NOT retuned. The question this run answers is whether
the economic fade model survives broad validation on its own deterministic merits.

Usage:
    PYTHONPATH=src python scripts/utilities/research_v3/tri_engine_backtest.py --broad
    PYTHONPATH=src python scripts/utilities/research_v3/tri_engine_backtest.py --n 50
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "utilities"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import warnings  # noqa: E402

from scipy import stats as _scipy_stats  # noqa: E402

# Reuse the frozen paired harness verbatim -- bundle builder, PIT pipelines, metrics, IO.
import v3_universe_backtest as vb  # noqa: E402
import validation_common as vc  # noqa: E402
from historical_predictive_power import REPORTING_LAG_DAYS, _price_asof, _quarter_ends  # noqa: E402

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.core import deterministic_baseline  # noqa: E402
from dcf_project.engine_v3.drivers import DriverExtractionError  # noqa: E402
from dcf_project.preprocessing.features import FeatureError, build_features  # noqa: E402
from dcf_project.research.run_tracker import RunTracker  # noqa: E402
from dcf_project.valuation.terminal import TerminalValueError  # noqa: E402

REPORT_DIR = ROOT / "reports" / "research_v3"
HORIZONS = vb.HORIZONS

#: Intrinsic > GUARD_RATIO x price (or non-finite / non-positive) is treated as pathological
#: -- a data/coherence blow-up, not an economic view. Wide enough that a genuine 10-bagger
#: thesis survives; tight enough to catch negative-equity / near-zero-IC explosions.
GUARD_RATIO = 100.0


def _generic_iv(bundle, settings) -> float | None:
    """Textbook single-stage normalized-margin DCF on the PIT bundle (mirrors
    ``valuation.generic.generic_valuation`` but on the injected, point-in-time bundle)."""
    features, _ = build_features(bundle.periods, settings)
    latest = features.iloc[-1]

    def _lv(col):
        try:
            v = float(latest.get(col))
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    baseline = deterministic_baseline(
        features, bundle.wacc, bundle.market, _lv("cash"), _lv("debt"),
        _lv("shares_outstanding"), settings, 20,
    )
    iv = baseline.get("intrinsic_value_per_share")
    return float(iv) if isinstance(iv, (int, float)) and math.isfinite(iv) else None


def _pathological(iv: float | None, price: float) -> bool:
    if iv is None or not math.isfinite(iv) or iv <= 0:
        return True
    return bool(price and iv / price > GUARD_RATIO)


def _process_ticker(ticker, family, cutoffs, gen_settings, v1_settings, v3_settings,
                    splits_events, on_sec_retry=None, heartbeat=None) -> dict[str, Any]:
    """Value one ticker across all PIT cutoffs with ALL THREE engines on identical bundles.
    Self-contained, resumable checkpoint payload; never raises."""
    try:
        periods_all, _ = vb._sec_periods_with_retry(ticker, v1_settings, on_retry=on_sec_retry)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "rows": [], "skips": {"sec": 1}, "pathological": [],
                "outcome": f"failed:sec:{type(exc).__name__}"}
    periods_tb = vc.reexpress_shares_today_basis(periods_all, splits_events)
    prices = vc.pit_price_frame(ticker, vb.PRICE_CACHE)
    if prices is None or not len(prices):
        return {"ticker": ticker, "rows": [], "skips": {"price": 1}, "pathological": [],
                "outcome": "skipped:price"}
    close_series, adj_series = prices["close"], prices["adj_close"]
    beta = vb._beta_cached(ticker)
    last_price_date = prices.index[-1].date()

    rows: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    pathological: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        if heartbeat is not None:
            heartbeat(ticker, f"cutoff {cutoff.isoformat()}")
        bundle, reason = vb._build_bundle(ticker, periods_tb, close_series, beta, cutoff, v1_settings)
        if bundle is None:
            vb._bump(skips, reason or "unknown")
            continue
        px = bundle.market.price

        # --- v1 (frozen canonical) ------------------------------------------------
        try:
            r1 = vb._PITPipeline(v1_settings, bundle).run(ticker)
        except Exception as exc:  # noqa: BLE001
            vb._bump(skips, f"v1_run:{type(exc).__name__}")
            continue
        # --- v3 (economic fade) ---------------------------------------------------
        try:
            r3 = vb._PITPipelineV3(v3_settings, bundle).run(ticker)
        except DriverExtractionError as exc:
            vb._bump(skips, "v3_not_applicable" if str(exc).startswith("v3_not_applicable")
                     else f"v3_skip:{str(exc)[:40]}")
            continue
        except TerminalValueError:
            vb._bump(skips, "v3_skip:terminal_spread")
            continue
        except Exception as exc:  # noqa: BLE001
            vb._bump(skips, f"v3_run:{type(exc).__name__}")
            continue
        # --- generic (textbook baseline) -----------------------------------------
        try:
            g_iv = _generic_iv(bundle, gen_settings)
        except (FeatureError, Exception) as exc:  # noqa: BLE001
            vb._bump(skips, f"generic_run:{type(exc).__name__}")
            continue

        v1_iv = float(r1.intrinsic_value_per_share)
        iv3 = float(r3.intrinsic_value_per_share)

        # --- OUTLIER GUARD: drop the whole row if ANY engine blew up (logged) ------
        blew = [(name, iv) for name, iv in
                (("generic", g_iv), ("v1", v1_iv), ("v3", iv3)) if _pathological(iv, px)]
        if blew:
            for name, iv in blew:
                pathological.append({"ticker": ticker, "sector": family, "date": cutoff.isoformat(),
                                     "engine": name, "iv": iv, "price": px,
                                     "ratio": (iv / px if (iv and px and math.isfinite(iv)) else None)})
            vb._bump(skips, "guard:" + "+".join(n for n, _ in blew))
            continue

        row: dict[str, Any] = {
            "ticker": ticker, "sector": family, "date": cutoff.isoformat(), "price": px,
            "cum_split": vc.cum_split_factor_after(splits_events, cutoff - timedelta(days=REPORTING_LAG_DAYS)),
            "generic_iv": g_iv, "generic_discount": g_iv / px - 1.0,
            "v1_iv": v1_iv, "v1_discount": vb._v1_signal(r1),
            "v3_iv": iv3, "discount_to_intrinsic": iv3 / px - 1.0,  # v3 signal (harness contract key)
            "company_type": r1.diagnostics.get("resolved_company_type") or r1.diagnostics.get("company_type"),
            "wacc": float(r3.wacc),
            **vb._v3_extras(r3),
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
    return {"ticker": ticker, "rows": rows, "skips": skips, "pathological": pathological,
            "outcome": f"ok:{len(rows)}"}


# ----------------------------- aggregation ----------------------------------- #

ENGINES = {"generic": "generic_discount", "v1": "v1_discount", "v3": "discount_to_intrinsic"}


def _three_engine_metrics(rows: list[dict], with_ci: bool) -> dict[str, Any]:
    return {eng: vb._engine_metrics(rows, key, with_ci=with_ci) for eng, key in ENGINES.items()}


def _family_blocks(rows: list[dict]) -> dict[str, Any]:
    fams: dict[str, list] = {}
    for o in rows:
        fams.setdefault(o.get("sector") or "unknown", []).append(o)
    out: dict[str, Any] = {}
    for fam, sub in sorted(fams.items()):
        block = {"n_obs": len(sub), "n_tickers": len({o["ticker"] for o in sub})}
        block.update(_three_engine_metrics(sub, with_ci=False))
        out[fam] = block
    return out


def _f(v, nd=3):
    return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else "  None"


def main() -> int:
    warnings.filterwarnings("ignore", category=_scipy_stats.ConstantInputWarning)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50, help="names from the frozen research tranche")
    ap.add_argument("--broad", action="store_true", help="run the full ~500-name universe")
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--no-resume", action="store_true", help="ignore existing per-ticker checkpoints")
    args = ap.parse_args()

    universe_meta = json.loads((ROOT / "reports" / "_universe_tickers.json").read_text(encoding="utf-8"))
    family_of: dict[str, str] = universe_meta["family_of"]
    if args.broad:
        tickers = list(universe_meta["tickers"])
        scope = "broad"
    else:
        tickers = vb.TRANCHE[: args.n]
        scope = f"n{len(tickers)}"

    base = replace(
        get_settings(),
        wacc_source="internal", terminal_growth_source="internal",
        forecast_mode="state_space", consensus_weight=0.0,
        monte_carlo_samples=0, scenario_monte_carlo_samples=0, run_sensitivity=False,
        sec_cache_enabled=True,
    )
    gen_settings = base                                  # textbook leg uses the same internal WACC/terminal
    v1_settings = base
    v3_settings = replace(base, engine_version="v3", v3_mc_draws=0)  # DETERMINISTIC headline

    cutoffs = _quarter_ends(args.start_year)
    out_path = REPORT_DIR / f"tri_engine_backtest_{scope}.json"
    ckpt_dir = REPORT_DIR / "_checkpoints" / f"tri_{scope}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"tri-engine backtest: {len(tickers)} tickers x {len(cutoffs)} cutoffs "
          f"(mc_draws=0 deterministic, guard={GUARD_RATIO}x, scope={scope})", flush=True)
    splits = vc.fetch_whitelisted_splits(tickers, vb.SPLIT_CACHE, seed_caches=vb.SPLIT_SEED_CACHES)

    tracker = RunTracker.start(
        run_id=f"tri_engine_{scope}_{date.today().isoformat()}",
        command=" ".join(sys.argv), phase="tri_engine_backtest",
        tranche_size=len(tickers), output_path=str(out_path.relative_to(ROOT)),
        state_path=REPORT_DIR / "tri_run_tracker.json", log_path=REPORT_DIR / "tri_run_tracker.log",
    )

    obs: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    pathological: list[dict[str, Any]] = []
    outcomes: dict[str, str] = {}
    started = time.time()
    try:
        for i, ticker in enumerate(tickers):
            fam = vb._family(ticker, family_of)
            ckpt = ckpt_dir / f"{ticker}.json"
            if ckpt.exists() and not args.no_resume:
                payload = json.loads(ckpt.read_text(encoding="utf-8"))
            else:
                payload = _process_ticker(
                    ticker, fam, cutoffs, gen_settings, v1_settings, v3_settings,
                    splits.get(ticker, []),
                    on_sec_retry=lambda t, a: tracker.heartbeat(t, f"sec retry {a}"),
                    heartbeat=lambda t, d: tracker.heartbeat(t, d),
                )
                vb._save(ckpt, payload)
            obs.extend(payload["rows"])
            for k, v in payload["skips"].items():
                skips[k] = skips.get(k, 0) + v
            pathological.extend(payload.get("pathological", []))
            outcomes[ticker] = payload["outcome"]
            (tracker.mark_completed if payload["outcome"].startswith("ok") else tracker.mark_failed)(
                ticker, *([] if payload["outcome"].startswith("ok") else [payload["outcome"]]))
            print(f"  [{i + 1}/{len(tickers)}] {ticker} ({fam}): {payload['outcome']}"
                  + (f"  [{len(payload.get('pathological', []))} guarded]" if payload.get("pathological") else ""),
                  flush=True)
    except KeyboardInterrupt:
        tracker.finish(status="paused", exit_code=130)
        print("interrupted -- checkpoints retained, rerun to resume", flush=True)
        return 130

    print(f"collected {len(obs)} paired observations in {time.time() - started:.0f}s; "
          f"{len(pathological)} guarded blow-ups; computing metrics", flush=True)
    overall = _three_engine_metrics(obs, with_ci=True)
    paired = {
        "v3_minus_v1": {f"{y}y": vc.paired_ic_diff(obs, y, "discount_to_intrinsic", "v1_discount") for y in HORIZONS},
        "v3_minus_generic": {f"{y}y": vc.paired_ic_diff(obs, y, "discount_to_intrinsic", "generic_discount") for y in HORIZONS},
        "v1_minus_generic": {f"{y}y": vc.paired_ic_diff(obs, y, "v1_discount", "generic_discount") for y in HORIZONS},
    }
    families = _family_blocks(obs)

    # guard ledger: blow-ups per engine and the worst offenders
    guard_by_engine: dict[str, int] = {}
    guard_by_ticker: dict[str, int] = {}
    for p in pathological:
        guard_by_engine[p["engine"]] = guard_by_engine.get(p["engine"], 0) + 1
        guard_by_ticker[p["ticker"]] = guard_by_ticker.get(p["ticker"], 0) + 1

    payload = {
        "methodology": {
            "design": "tri-engine (generic / v1 / v3) in-process on identical split-consistent PIT bundles",
            "headline": "DETERMINISTIC -- v3 mc_draws=0; Monte Carlo NOT used (failed calibration gate G6)",
            "outlier_guard": f"row dropped if ANY engine iv is non-finite, <=0, or > {GUARD_RATIO}x price; "
                             f"logged in 'pathological', never silently discarded",
            "signals": ENGINES,
            "price_basis": "today-basis shares x split-adjusted Close; forward returns = Adj Close (total return)",
            "reporting_lag_days": REPORTING_LAG_DAYS,
            "start_year": args.start_year,
            "no_engine_math_changes": True,
        },
        "n_tickers": len(tickers), "n_tickers_with_obs": len({o["ticker"] for o in obs}),
        "n_observations": len(obs), "n_guarded": len(pathological),
        "guard_by_engine": guard_by_engine,
        "guard_by_ticker": dict(sorted(guard_by_ticker.items(), key=lambda kv: -kv[1])),
        "skips": skips, "outcomes": outcomes,
        "overall_metrics": overall,
        "paired_ic_diff": paired,
        "by_family": families,
        "pathological": pathological,
        "observations": obs,
    }
    vb._save(out_path, payload)

    summary = {k: payload[k] for k in ("n_tickers", "n_tickers_with_obs", "n_observations",
                                       "n_guarded", "guard_by_engine", "guard_by_ticker", "skips")}
    summary["headline"] = {
        f"{y}y": {eng: {"ic_obs": (overall[eng].get(f"{y}y") or {}).get("ic_obs_weighted"),
                        "ic_eq": (overall[eng].get(f"{y}y") or {}).get("ic_equal_weight"),
                        "ls_q5_q1": (overall[eng].get(f"{y}y") or {}).get("long_short_q5_minus_q1")}
                 for eng in ENGINES}
        for y in HORIZONS
    }
    vb._save(REPORT_DIR / f"tri_engine_summary_{scope}.json", summary)
    tracker.finish(status="completed", exit_code=0)

    print(f"\n=== TRI-ENGINE RESULTS ({scope}) -- obs-weighted IC / L-S Q5-Q1 ===")
    print(f"  {'horizon':8} {'generic IC':>11} {'v1 IC':>9} {'v3 IC':>9}   "
          f"{'gen L/S':>9} {'v1 L/S':>9} {'v3 L/S':>9}")
    for y in HORIZONS:
        m = {eng: overall[eng].get(f"{y}y", {}) for eng in ENGINES}
        print(f"  {y}y{'':5} {_f(m['generic'].get('ic_obs_weighted')):>11} "
              f"{_f(m['v1'].get('ic_obs_weighted')):>9} {_f(m['v3'].get('ic_obs_weighted')):>9}   "
              f"{_f(m['generic'].get('long_short_q5_minus_q1')):>9} "
              f"{_f(m['v1'].get('long_short_q5_minus_q1')):>9} "
              f"{_f(m['v3'].get('long_short_q5_minus_q1')):>9}")
    print("\n=== BY FAMILY (3y obs-weighted IC: generic / v1 / v3) ===")
    for fam, block in families.items():
        def ic3(eng):
            return (block.get(eng, {}).get("3y") or {}).get("ic_obs_weighted")
        print(f"  {fam:26} n={block['n_obs']:5}  gen={_f(ic3('generic'))} "
              f"v1={_f(ic3('v1'))} v3={_f(ic3('v3'))}")
    print(f"\nguarded blow-ups: {len(pathological)} (by engine: {guard_by_engine}; "
          f"worst names: {dict(list(payload['guard_by_ticker'].items())[:5])})")
    print(f"skips: {skips}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
