"""P3 / Gate 5 -- Expectation Gap candidate-signal backtest (diagnostic, additive only).

Does the reverse-DCF Expectation Gap carry independent ranking information beyond the discount?
For each ticker x quarterly PIT cutoff this values the bundle with v2 (engine_v2), then emits BOTH
the discount baseline (v1_discount, v2_discount == the aggregate value_gap) and the four decomposed
Expectation Gaps (eg_growth_gap, eg_margin_gap, eg_reinvestment_gap, eg_value_gap), joins realized
3y/5y forward returns, and reports Rank-IC / IC-t / Spearman / long-short / decile-monotonicity /
top-decile / hit-rate for each signal on the SAME validation harness as P2.5.

Reuses the frozen PIT helpers, the resumable per-ticker checkpoints, the run tracker, and the
metric functions -- so this is a clean parallel study, not a change to the P2.5 control. The gaps
are candidate signals ONLY: nothing here is wired into valuation or portfolio construction.

Usage:
    PYTHONPATH=src python scripts/utilities/research_v2/expectation_gap_backtest.py --n 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts" / "utilities"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.engine_v2 import expectation_gap as eg  # noqa: E402
from historical_predictive_power import (  # noqa: E402
    _beta,
    _build_pit_bundle,
    _metrics_for_horizon,
    _price_asof,
    _price_history,
    _quarter_ends,
    _PITPipeline,
)
# Reuse the P2.5 harness plumbing verbatim (frozen control infrastructure).
from forecasting_compare import (  # noqa: E402
    TRANCHE,
    _PITPipelineV2,
    _sec_periods_with_retry,
    _signal_summary,
    _v1_signal,
    _write_checkpoint,
)

# Signals backtested. value_gap == v2_discount by construction (the aggregate Expectation Gap);
# it is kept as an explicit consistency check alongside the decomposed driver gaps.
SIGNALS = ["v1_discount", "v2_discount",
           "eg_growth_gap", "eg_margin_gap", "eg_reinvestment_gap", "eg_value_gap"]


def _checkpoint_dir(n: int) -> Path:
    d = ROOT / "reports" / "research_v2" / "_checkpoints" / f"expectation_gap_n{n}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _process_ticker(ticker, cutoffs, v1_settings, v2_settings, on_sec_retry, heartbeat):
    """Value one ticker across all PIT cutoffs with v2, emitting discount + Expectation Gaps.
    Self-contained, never raises -- failures are recorded in `outcome` for clean resume."""
    try:
        periods_all, _ = _sec_periods_with_retry(ticker, v1_settings, on_retry=on_sec_retry)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "rows": [], "skips": {f"{ticker}:sec": 1},
                "outcome": f"failed:sec:{type(exc).__name__}"}
    prices = _price_history(ticker)
    if prices is None:
        return {"ticker": ticker, "rows": [], "skips": {f"{ticker}:price": 1}, "outcome": "skipped:price"}
    beta = _beta(ticker)
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
        row: dict[str, Any] = {
            "ticker": ticker, "date": cutoff.isoformat(), "price": price,
            "v1_discount": _v1_signal(r1),
            "v2_discount": float(r2.intrinsic_value_per_share) / price - 1.0,
        }
        row.update(eg.candidate_signals(r2))  # eg_growth_gap, eg_margin_gap, eg_reinvestment_gap, eg_value_gap
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


def run(tickers, reduced=True, ckpt_dir=None, tracker=None) -> dict[str, Any]:
    base = replace(
        get_settings(),
        wacc_source="internal", terminal_growth_source="internal",
        forecast_mode="state_space", consensus_weight=0.0,
        monte_carlo_samples=0, run_sensitivity=False,
        run_reverse_dcf=True,  # Expectation Gaps need the market-implied (reverse) block
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

    metrics = {}
    for signal in SIGNALS:
        metrics[signal] = {f"{y}y": _metrics_for_horizon(obs, y, signal) for y in (3, 5)}
    summary = {sig: {h: _signal_summary(metrics[sig][h]) for h in ("3y", "5y")} for sig in metrics}
    return {"n_tickers": len(tickers), "n_obs": len(obs), "skips": skips,
            "sec_retries": _sec_retries["n"], "summary": summary, "metrics": metrics, "observations": obs}


def _f(v, p=4) -> str:
    return f"{v:+.{p}f}" if isinstance(v, (int, float)) else "   n/a"


def _print_summary(payload, out_path) -> None:
    print(f"\n=== P3 Expectation Gap backtest (n_tickers={payload['n_tickers']}, "
          f"n_obs={payload['n_obs']}, sec_retries={payload['sec_retries']}) ===")
    print(f"skips={payload['skips']}")
    print(f"{'horizon':>7} {'signal':>18} {'n':>5} {'IC':>8} {'IC_t':>6} {'spear':>8} "
          f"{'L/S':>8} {'decMono':>8} {'topDec':>8} {'hit>med':>8}")
    for y in (3, 5):
        for sig in SIGNALS:
            s = payload["summary"][sig][f"{y}y"]
            if "spearman" not in s:
                print(f"{y}y".rjust(7) + f" {sig:>18} {s.get('n_obs', 0):>5}  insufficient")
                continue
            print(f"{y}y".rjust(7) + f" {sig:>18} {s['n_obs']:>5} {_f(s['ic_mean']):>8} {_f(s['ic_t'], 2):>6} "
                  f"{_f(s['spearman']):>8} {_f(s['long_short']):>8} {_f(s['decile_monotonicity']):>8} "
                  f"{_f(s['top_decile_spread']):>8} {_f(s['hit_rate_beats_median'], 3):>8}")
        print()
    print(f"wrote {out_path}")


def _journal_on_completion(payload, n, out_path, command) -> None:
    try:
        from journal import append_entry
    except Exception:  # noqa: BLE001
        return

    def _cell(sig, h):
        s = payload["summary"][sig][f"{h}y"]
        if "spearman" not in s:
            return "insufficient"
        return (f"IC {s['ic_mean']:.4f} (t={s['ic_t']:.2f}) L/S {s['long_short']:.4f} "
                f"decMono {s['decile_monotonicity']:.3f}")

    metrics = {f"{h}y": {sig: _cell(sig, h) for sig in SIGNALS} for h in (3, 5)}
    n_names = len({r["ticker"] for r in payload["observations"]})
    append_entry({
        "title": f"P3 Expectation Gap backtest -- n={n} ({n_names} names, {payload['n_obs']} obs)",
        "hypothesis": "Do the decomposed Expectation Gaps (growth/margin/reinvestment) add ranking "
                      "information beyond the discount baseline (v2_discount == aggregate value_gap)?",
        "sample_size": f"intended n={n}; names with obs={n_names}; total obs={payload['n_obs']}; "
                       f"SEC retries={payload.get('sec_retries')}",
        "engine": "v2 (engine_v2 + expectation_gap), reduced MCMC; v1_discount kept for reference. Diagnostic only.",
        "command": command,
        "output_artifact": str(out_path.relative_to(ROOT)),
        "metrics": metrics,
        "interpretation": "Auto-logged on completion. Compare each gap's Rank-IC / IC-t / long-short vs the "
                          "v2_discount baseline; value_gap should reproduce v2_discount (consistency check). "
                          "A gap 'adds information' only if it ranks comparably or better than the discount.",
        "decision": "Pending review -- candidate signals stay OUT of construction until proven. v2 non-default.",
        "next_action": "Summarize whether any gap is additive; if yes, design an orthogonality/combination test; "
                       "if not, record the negative result. Do not wire into conf_top10.",
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
    out_path = out_dir / f"expectation_gap_backtest_{args.n}.json"
    command = f"PYTHONPATH=src python scripts/utilities/research_v2/expectation_gap_backtest.py --n {args.n}"

    tracker = None
    if not args.no_tracker:
        from dcf_project.research.run_tracker import RunTracker
        tracker = RunTracker.start(
            run_id=f"expectation_gap_n{args.n}", command=command, phase=f"P3/EG/n={args.n}",
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
