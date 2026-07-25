"""P4B / Gate -- Sector-State Engine v1 candidate-signal backtest (diagnostic, additive only).

Does an industry-level *sector state* carry ranking information the company-level DCF cannot --
and, decisively, does **V2 discount + sector state beat V2 discount alone**?

Two-phase, on the SAME validation harness as P2.5/P3:

  Phase 1 (expensive, resumable per ticker): for each ticker x quarterly PIT cutoff, value the
    bundle with v1 + v2 (-> v1_discount, v2_discount) and record the raw sector inputs the panel
    needs (sector, market cap, TTM EBIT, TTM-EBIT YoY) -- all from the SAME SEC fundamentals the
    engine already pulls. Checkpointed per ticker so SEC throttling slows but cannot invalidate.
    Discounts + market caps are made SPLIT-CONSISTENT here (the frozen PIT harness pairs a
    split-adjusted price with as-reported pre-split shares, inflating splitters' pre-split discounts
    ~4-20x; corrected via the cumulative split factor -- frozen harness left untouched).

  Phase 2 (cheap, in-memory): aggregate the raw inputs into a per-(sector, cutoff) panel, add
    SPDR-ETF momentum vs SPY, z-score each component against the sector's own expanding history
    (point-in-time), blend into sector_state (companies inherit their sector's score), form the
    cross-sectional combined signal, then run the battery + the decisive orthogonality test.

The P3 lesson governs the verdict: a signal is FAVORABLE only if it adds information *orthogonal*
to the discount (partial-Spearman vs v2_discount) and/or the combined out-ranks v2_discount alone.
Nothing here is wired into valuation, construction, or the live conf_top10 book; v2 stays non-default.

Usage:
    PYTHONPATH=src python scripts/utilities/research_v2/sector_state_backtest.py --n 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts" / "utilities"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.engine_v2 import sector_state as ss  # noqa: E402
from dcf_project.engine_v2.sector_map import (  # noqa: E402
    etf_available_asof,
    etf_for_sector,
    sector_for_ticker,
)
from historical_predictive_power import (  # noqa: E402
    _beta,
    _build_pit_bundle,
    _metrics_for_horizon,
    _price_asof,
    _price_history,
    _quarter_ends,
    _PITPipeline,
)
from forecasting_compare import (  # noqa: E402
    TRANCHE,
    _PITPipelineV2,
    _sec_periods_with_retry,
    _signal_summary,
    _write_checkpoint,
)

# Discount baselines + the inherited sector_state, its z-scored components, and the combined signal.
COMBINED = "v2_discount_plus_sector_state"
SIGNALS = ["v1_discount", "v2_discount", "sector_state",
           "sector_valuation", "sector_revisions", "sector_momentum", COMBINED]


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _sum_attr(periods, attr: str) -> float | None:
    """Sum an attribute over the given periods; None if any is missing (no partial TTM)."""
    total = 0.0
    for p in periods:
        v = getattr(p, attr, None)
        if v is None:
            return None
        total += float(v)
    return total


def _ttm_ebit(bundle) -> tuple[float | None, float | None]:
    """TTM EBIT and its YoY growth for one ticker x cutoff (None if any quarter is missing)."""
    periods = bundle.periods
    ttm = _sum_attr(periods[-4:], "ebit") if len(periods) >= 4 else None
    prior = _sum_attr(periods[-8:-4], "ebit") if len(periods) >= 8 else None
    yoy = (ttm / prior - 1.0) if (ttm is not None and prior is not None and prior > 0) else None
    return ttm, yoy


def _ticker_splits(ticker: str):
    """Cumulative-split history for a ticker (yfinance), tz-naive dates. None on failure."""
    try:
        import yfinance as yf
        s = yf.Ticker(ticker).splits
    except Exception:  # noqa: BLE001
        return None
    if s is None or len(s) == 0:
        return None
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


# Clean forward-split factors. Applied to INDIVIDUAL yfinance split ratios (so products like
# NVDA 4x10=40 or ISRG 3x3=9 are kept), rejecting spin-offs (~1.1-1.5) and data errors (e.g. a
# lone 25x on BKNG) that yfinance encodes as "splits". yfinance is the consistent source here --
# it is what the price series was auto-adjusted with -- so matching it keeps shares <-> price coherent.
_SPLIT_WHITELIST = {2, 3, 4, 5, 6, 7, 8, 10, 15, 20}


def _clean_factor(ratio: float) -> float:
    """Snap a yfinance split ratio to a clean whitelist factor, else 1.0 (reject spin-off/garbage)."""
    r = float(ratio)
    f = round(r)
    return float(f) if (f in _SPLIT_WHITELIST and abs(r - f) / f <= 0.15) else 1.0


def _cum_split_after(splits, d) -> float:
    """Product of WHITELISTED split ratios occurring strictly after date ``d`` (1.0 if none).

    Brings as-reported (then-basis) shares onto the split-adjusted (today-basis) price so market
    cap and the discount are split-consistent -- the fix for the PIT harness's adjusted-price vs
    as-reported-shares mismatch (e.g. AAPL pre-2020 ~4x, AMZN pre-2022 ~20x). PIT-valid: a split is
    a known corporate action and true market cap is split-invariant. cum_split == 1.0 for every
    non-splitter and every post-split cutoff, so the corrected discount equals the frozen-harness
    discount everywhere except the genuine split artifacts.
    """
    if splits is None or len(splits) == 0:
        return 1.0
    f = 1.0
    for ts, ratio in splits.items():
        if ts.date() > d:
            f *= _clean_factor(ratio)
    return f


def _v1_intrinsic(result) -> float:
    """v1 diagnostic intrinsic (mirrors forecasting_compare._v1_signal's numerator)."""
    diag = result.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
    return float(diag) if isinstance(diag, (int, float)) else float(result.intrinsic_value_per_share)


def _checkpoint_dir(n: int) -> Path:
    d = ROOT / "reports" / "research_v2" / "_checkpoints" / f"sector_state_n{n}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Phase 1: per-ticker valuation + raw sector inputs ------------------------------------------

def _process_ticker(ticker, cutoffs, v1_settings, v2_settings, on_sec_retry, heartbeat):
    """Value one ticker across all PIT cutoffs; emit discount baselines + raw sector inputs.
    Self-contained, never raises -- failures are recorded in `outcome` for clean resume."""
    sector = sector_for_ticker(ticker)
    if sector is None:
        return {"ticker": ticker, "rows": [], "skips": {f"{ticker}:no_sector": 1},
                "outcome": "skipped:no_sector"}
    try:
        periods_all, _ = _sec_periods_with_retry(ticker, v1_settings, on_retry=on_sec_retry)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "rows": [], "skips": {f"{ticker}:sec": 1},
                "outcome": f"failed:sec:{type(exc).__name__}"}
    prices = _price_history(ticker)
    if prices is None:
        return {"ticker": ticker, "rows": [], "skips": {f"{ticker}:price": 1}, "outcome": "skipped:price"}
    beta = _beta(ticker)
    splits = _ticker_splits(ticker)  # for split-consistent discount + market cap
    last_price_date = prices.index[-1].date()
    rows: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    for cutoff in cutoffs:
        if heartbeat is not None:
            heartbeat(ticker, f"cutoff {cutoff.isoformat()}")
        bundle, reason = _build_pit_bundle(ticker, periods_all, prices, beta, cutoff, v1_settings)
        if bundle is None:
            skips[reason or "unknown"] = skips.get(reason or "unknown", 0) + 1
            continue
        try:
            r1 = _PITPipeline(v1_settings, bundle).run(ticker)
            r2 = _PITPipelineV2(v2_settings, bundle).run(ticker)
        except Exception as exc:  # noqa: BLE001
            skips[f"run:{type(exc).__name__}"] = skips.get(f"run:{type(exc).__name__}", 0) + 1
            continue
        price = bundle.market.price
        # Split-consistency: bring as-reported (then-basis) shares onto the adjusted (today-basis)
        # price. share_rd is the as-of date of the shares figure the bundle used.
        share_rd = next((p.report_date for p in reversed(bundle.periods)
                         if getattr(p, "shares_outstanding", None) is not None), cutoff)
        cum = _cum_split_after(splits, share_rd)
        disc_price = price * cum
        ttm_ebit, ttm_ebit_yoy = _ttm_ebit(bundle)
        mktcap = _num(getattr(bundle.market, "market_cap", None))
        if mktcap is not None:
            mktcap *= cum
        row: dict[str, Any] = {
            "ticker": ticker, "date": cutoff.isoformat(), "price": price, "sector": sector,
            "cum_split": cum,
            "v1_discount": _v1_intrinsic(r1) / disc_price - 1.0,
            "v2_discount": float(r2.intrinsic_value_per_share) / disc_price - 1.0,
            "market_cap": mktcap, "ttm_ebit": ttm_ebit, "ttm_ebit_yoy": ttm_ebit_yoy,
        }
        for years in (3, 5):
            target = cutoff + timedelta(days=round(365.25 * years))
            key = f"fwd_{years}y_ann_return"
            if target > last_price_date:
                row[key] = None
                continue
            p_fwd = _price_asof(prices, target)
            row[key] = float((p_fwd / price) ** (1.0 / years) - 1.0) if p_fwd and p_fwd > 0 else None
        rows.append(row)
    return {"ticker": ticker, "rows": rows, "skips": skips, "outcome": f"ok:{len(rows)}"}


# --- Phase 2: sector panel, momentum, z-score, join ---------------------------------------------

def _etf_cache(sectors: set[str]):
    """Fetch each present sector's SPDR ETF + SPY once (yfinance). Returns (etf_series_by_etf, spy)."""
    spy = _price_history("SPY")
    cache: dict[str, Any] = {}
    for sec in sectors:
        etf = etf_for_sector(sec)
        if etf and etf not in cache:
            cache[etf] = _price_history(etf)
    return cache, spy


def _build_panel(obs: list[dict[str, Any]], etf_cache, spy) -> ss.SectorPanel:
    """Aggregate raw per-(sector, cutoff) inputs + ETF momentum into a SectorPanel."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in obs:
        groups.setdefault((r["sector"], r["date"]), []).append(r)
    panel = ss.SectorPanel()
    for (sector, dstr), members in groups.items():
        asof = date.fromisoformat(dstr)
        valuation = ss.aggregate_earnings_yield([(m.get("market_cap"), m.get("ttm_ebit")) for m in members])
        revisions = ss.aggregate_revisions([m.get("ttm_ebit_yoy") for m in members])
        etf = etf_for_sector(sector)
        momentum = (ss.sector_momentum_asof(etf_cache.get(etf), spy, asof)
                    if etf_available_asof(etf, asof) else None)
        panel.set_raw(sector, asof, valuation=valuation, revisions=revisions, momentum=momentum)
    return panel


def _attach_sector_state(obs: list[dict[str, Any]], panel: ss.SectorPanel) -> None:
    for r in obs:
        st = panel.state(r["sector"], date.fromisoformat(r["date"]))
        r["sector_state"] = st["sector_state"]
        r["sector_coverage"] = st["coverage"]
        for key in ss.COMPONENT_KEYS:
            r[f"sector_{key}"] = st[f"sector_{key}"]


# --- orthogonality (the decisive test, per P3) --------------------------------------------------

def _spear(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def _orthogonality(rows, signal_key, control_key, fwd_key) -> dict[str, Any]:
    """Partial-Spearman of forward return on the signal, controlling for the discount."""
    pts = [(r[signal_key], r[control_key], r[fwd_key]) for r in rows
           if _num(r.get(signal_key)) is not None and _num(r.get(control_key)) is not None
           and _num(r.get(fwd_key)) is not None]
    n = len(pts)
    if n < 10:
        return {"n": n, "note": "insufficient"}
    g = [p[0] for p in pts]
    c = [p[1] for p in pts]
    f = [p[2] for p in pts]
    r_fg, r_fc, r_gc = _spear(f, g), _spear(f, c), _spear(g, c)
    denom = ((1.0 - r_fc ** 2) * (1.0 - r_gc ** 2)) ** 0.5
    rho = (r_fg - r_fc * r_gc) / denom if denom > 0 else None
    t = (rho * ((n - 3) / (1.0 - rho ** 2)) ** 0.5) if (rho is not None and abs(rho) < 1.0) else None
    return {"n": n, "corr_signal_vs_discount": r_gc, "spear_fwd_signal": r_fg,
            "spear_fwd_discount": r_fc, "partial_rho": rho, "partial_t": t}


def run(tickers, reduced=True, ckpt_dir=None, tracker=None) -> dict[str, Any]:
    base = replace(
        get_settings(),
        wacc_source="internal", terminal_growth_source="internal",
        forecast_mode="state_space", consensus_weight=0.0,
        monte_carlo_samples=0, run_sensitivity=False,
    )
    v1_settings = base
    v2_settings = replace(
        base, engine_version="v2",
        v2_paths=400 if reduced else 2000,
        v2_mcmc_draws=300 if reduced else 1000,
        v2_mcmc_chains=2 if reduced else 4,
        v2_mcmc_warmup=200 if reduced else 1000,
        v2_terminal_mode="equilibrium",
    )
    cutoffs = _quarter_ends(2015)
    obs: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    _sec_retries = {"n": 0}

    def _on_sec_retry(tk, attempt):
        _sec_retries["n"] += 1
        if tracker is not None:
            tracker.heartbeat(current_ticker=tk, current_detail=f"SEC retry {attempt}",
                              sec_retries=_sec_retries["n"])
        print(f"    {tk}: SEC retry {attempt} (total retries {_sec_retries['n']})", flush=True)

    def _heartbeat(tk, detail):
        if tracker is not None:
            tracker.heartbeat(current_ticker=tk, current_detail=detail)

    # ---- Phase 1: per-ticker valuation + raw sector inputs (resumable) ----
    for ti, ticker in enumerate(tickers, 1):
        ckpt_path = (ckpt_dir / f"{ticker}.json") if ckpt_dir else None
        if ckpt_path is not None and ckpt_path.exists():
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            obs.extend(payload.get("rows", []))
            for k, v in payload.get("skips", {}).items():
                skips[k] = skips.get(k, 0) + v
            if tracker is not None:
                (tracker.mark_failed(ticker, payload["outcome"])
                 if payload.get("outcome", "").startswith("failed")
                 else tracker.mark_completed(ticker))
            print(f"[{ti}/{len(tickers)}] {ticker}: resumed from checkpoint "
                  f"({len(payload.get('rows', []))} obs, {payload.get('outcome')})", flush=True)
            continue

        time.sleep(1.0)
        payload = _process_ticker(ticker, cutoffs, v1_settings, v2_settings, _on_sec_retry, _heartbeat)
        if ckpt_path is not None:
            _write_checkpoint(ckpt_path, payload)
        obs.extend(payload["rows"])
        for k, v in payload["skips"].items():
            skips[k] = skips.get(k, 0) + v
        if tracker is not None:
            (tracker.mark_failed(ticker, payload["outcome"])
             if payload["outcome"].startswith("failed") else tracker.mark_completed(ticker))
        print(f"[{ti}/{len(tickers)}] {ticker}: {len(payload['rows'])} PIT obs ({payload['outcome']})", flush=True)

    # ---- Phase 2: sector panel, momentum, z-score, join, combine ----
    sectors = {r["sector"] for r in obs}
    if tracker is not None:
        tracker.heartbeat(current_ticker="(panel)", current_detail=f"building sector panel: {sorted(sectors)}")
    etf_cache, spy = _etf_cache(sectors)
    panel = _build_panel(obs, etf_cache, spy)
    _attach_sector_state(obs, panel)
    ss.combined_cross_sectional(obs, "v2_discount", "sector_state", COMBINED)

    metrics = {sig: {f"{y}y": _metrics_for_horizon(obs, y, sig) for y in (3, 5)} for sig in SIGNALS}
    summary = {sig: {h: _signal_summary(metrics[sig][h]) for h in ("3y", "5y")} for sig in metrics}
    orthogonality = {
        sig: {f"{y}y": _orthogonality(obs, sig, "v2_discount", f"fwd_{y}y_ann_return") for y in (3, 5)}
        for sig in ("sector_state", COMBINED)
    }
    sector_coverage = sorted({r["sector"] for r in obs})
    return {"n_tickers": len(tickers), "n_obs": len(obs), "skips": skips,
            "sec_retries": _sec_retries["n"], "sectors": sector_coverage,
            "summary": summary, "metrics": metrics, "orthogonality": orthogonality,
            "observations": obs}


def _f(v, p=4) -> str:
    return f"{v:+.{p}f}" if isinstance(v, (int, float)) else "   n/a"


def _print_summary(payload, out_path) -> None:
    print(f"\n=== P4B Sector-State backtest (n_tickers={payload['n_tickers']}, "
          f"n_obs={payload['n_obs']}, sec_retries={payload['sec_retries']}) ===")
    print(f"sectors={payload['sectors']}")
    print(f"skips={payload['skips']}")
    print(f"{'horizon':>7} {'signal':>30} {'n':>5} {'IC':>8} {'IC_t':>6} {'spear':>8} "
          f"{'L/S':>8} {'decMono':>8} {'topDec':>8} {'hit>med':>8}")
    for y in (3, 5):
        for sig in SIGNALS:
            s = payload["summary"][sig][f"{y}y"]
            if "spearman" not in s:
                print(f"{y}y".rjust(7) + f" {sig:>30} {s.get('n_obs', 0):>5}  insufficient")
                continue
            print(f"{y}y".rjust(7) + f" {sig:>30} {s['n_obs']:>5} {_f(s['ic_mean']):>8} {_f(s['ic_t'], 2):>6} "
                  f"{_f(s['spearman']):>8} {_f(s['long_short']):>8} {_f(s['decile_monotonicity']):>8} "
                  f"{_f(s['top_decile_spread']):>8} {_f(s['hit_rate_beats_median'], 3):>8}")
        print()
    print("--- orthogonality vs v2_discount (partial-Spearman) ---")
    for sig in ("sector_state", COMBINED):
        for y in (3, 5):
            o = payload["orthogonality"][sig][f"{y}y"]
            if "partial_rho" not in o or o.get("partial_rho") is None:
                print(f"  {sig:>30} {y}y: n={o.get('n')} {o.get('note', '')}")
                continue
            print(f"  {sig:>30} {y}y: n={o['n']} corr_vs_disc={_f(o['corr_signal_vs_discount'])} "
                  f"partial_rho={_f(o['partial_rho'])} partial_t={_f(o['partial_t'], 2)}")
    print(f"\nwrote {out_path}")


def _journal_on_completion(payload, n, out_path, command) -> None:
    try:
        from journal import append_entry
    except Exception:  # noqa: BLE001
        return

    def _cell(sig, h):
        s = payload["summary"][sig][f"{h}y"]
        if "spearman" not in s:
            return "insufficient"
        return f"IC {s['ic_mean']:.4f} (t={s['ic_t']:.2f}) L/S {s['long_short']:.4f}"

    def _ocell(sig, h):
        o = payload["orthogonality"][sig][f"{h}y"]
        if o.get("partial_rho") is None:
            return o.get("note", "n/a")
        return f"partial_rho {o['partial_rho']:.4f} (t={o['partial_t']:.2f}), corr_vs_disc {o['corr_signal_vs_discount']:.3f}"

    metrics = {f"{h}y": {sig: _cell(sig, h) for sig in
                         ("v2_discount", "sector_state", COMBINED)} for h in (3, 5)}
    metrics["orthogonality"] = {f"{h}y": {sig: _ocell(sig, h) for sig in ("sector_state", COMBINED)}
                                for h in (3, 5)}
    n_names = len({r["ticker"] for r in payload["observations"]})
    append_entry({
        "title": f"P4B Sector-State backtest -- n={n} ({n_names} names, {payload['n_obs']} obs)",
        "hypothesis": "Does an industry-level sector_state add ranking information ORTHOGONAL to the "
                      "discount, and does v2_discount + sector_state beat v2_discount alone?",
        "sample_size": f"intended n={n}; names with obs={n_names}; total obs={payload['n_obs']}; "
                       f"sectors={payload['sectors']}; SEC retries={payload.get('sec_retries')}",
        "engine": "v2 (engine_v2) discount baseline + sector_state (0.35 val / 0.30 rev / 0.20 mom / 0.15 flows, "
                  "flows deferred). Free PIT data; v1 proxies for revisions. Diagnostic only.",
        "command": command,
        "output_artifact": str(out_path.relative_to(ROOT)),
        "metrics": metrics,
        "interpretation": "Auto-logged on completion. FAVORABLE only if sector_state's partial-Spearman vs "
                          "v2_discount is significant AND/OR the combined out-ranks v2_discount. Else NEGATIVE "
                          "(the P3 discipline).",
        "decision": "Pending review -- candidate signal stays OUT of construction until proven. v2 non-default.",
        "next_action": "Write P4B_sector_state_conclusion.md with the FAVORABLE/NEGATIVE verdict. If FAVORABLE, "
                       "run the full factor-uniqueness panel before any construction proposal.",
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of tickers from the shared tranche")
    ap.add_argument("--full-mcmc", action="store_true", help="use full MCMC budget (slow)")
    ap.add_argument("--no-tracker", action="store_true")
    ap.add_argument("--no-journal", action="store_true")
    args = ap.parse_args()
    tickers = TRANCHE[: args.n]
    ckpt_dir = _checkpoint_dir(args.n)
    out_dir = ROOT / "reports" / "research_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sector_state_backtest_{args.n}.json"
    command = f"PYTHONPATH=src python scripts/utilities/research_v2/sector_state_backtest.py --n {args.n}"

    tracker = None
    if not args.no_tracker:
        from dcf_project.research.run_tracker import RunTracker
        tracker = RunTracker.start(
            run_id=f"sector_state_n{args.n}", command=command, phase=f"P4B/sector/n={args.n}",
            tranche_size=len(tickers), output_path=str(out_path.relative_to(ROOT)),
        )

    try:
        payload = run(tickers, reduced=not args.full_mcmc, ckpt_dir=ckpt_dir, tracker=tracker)
    except BaseException as exc:  # noqa: BLE001
        if tracker is not None:
            tracker.finish(status="failed", exit_code=1)
        print(f"run FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise

    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _print_summary(payload, out_path)
    if not args.no_journal:
        _journal_on_completion(payload, args.n, out_path, command)
    if tracker is not None:
        tracker.finish(status="completed", exit_code=0)


if __name__ == "__main__":
    main()
