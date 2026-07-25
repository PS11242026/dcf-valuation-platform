"""P2.5 RESEARCH FREEZE -- V1 vs V2 forecasting comparison (ranking quality).

Tests the PRIMARY hypothesis before any further engine sophistication: does the
posterior-coherent Bayesian state-space engine (v2) rank stocks better than the frozen
canonical state-space DCF (v1)?

For each ticker x quarterly PIT cutoff it builds ONE point-in-time bundle and values it with
BOTH engines (regime-free, internal WACC/terminal, no consensus lookahead), then joins
realized 3y/5y forward returns and reports, for v1 vs v2: IC / Rank-IC (mean cross-sectional
Spearman), pooled Spearman, decile + top-decile + long-short spreads, hit rate, and calibration.
v2 also emits its distribution-aware candidate signals for inspection.

Reduced MCMC budget (ranking only needs the posterior median, stable at fewer draws).
Reuses the frozen PIT helpers + metric function from historical_predictive_power.py.

Usage:
    PYTHONPATH=src python scripts/utilities/research_v2/forecasting_compare.py --n 10
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

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts" / "utilities"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.data_sources.sec import SECClient  # noqa: E402
from dcf_project.engine_v2.pipeline import BayesianDCFPipeline  # noqa: E402
from historical_predictive_power import (  # noqa: E402
    _beta,
    _build_pit_bundle,
    _metrics_for_horizon,
    _price_asof,
    _price_history,
    _quarter_ends,
    _PITPipeline,
)

# A diversified first tranche of long-history liquid names across sectors.
TRANCHE = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "KO", "COST", "V", "ADBE",
    "ORCL", "CRM", "PEP", "WMT", "HD", "MA", "AMD", "INTC", "TXN", "NKE",
    "MCD", "SBUX", "QCOM", "CSCO", "ACN", "LIN", "LOW", "TGT", "DHR", "TMO",
    "ABT", "NOW", "INTU", "AMAT", "LRCX", "KLAC", "MU", "PG", "CL", "GIS",
    "EL", "MDLZ", "KMB", "SYK", "ISRG", "REGN", "VRTX", "BKNG", "EBAY", "ETSY",
    # --- names 51-100: more breadth across the validated families ---------------
    "AVGO", "ADI", "MCHP", "NXPI", "MRVL", "ON", "TER", "ENTG", "SWKS", "MPWR",
    "ADSK", "ANSS", "CDNS", "SNPS", "WDAY", "PANW", "FTNT", "PYPL", "EA", "TTWO",
    "TJX", "ROST", "ORLY", "AZO", "YUM", "CMG", "LULU", "DPZ", "MAR", "HLT",
    "KHC", "STZ", "KDP", "HSY", "MKC", "CHD", "CLX", "K", "SJM", "HRL",
    "ODFL", "FAST", "PAYX", "ADP", "VRSK", "CTAS", "ROL", "CPRT", "JKHY", "EXPD",
]


class _PITPipelineV2(BayesianDCFPipeline):
    save_posterior = False

    def __init__(self, settings, bundle):
        super().__init__(settings)
        self._bundle = bundle

    def load_data(self, ticker):
        return self._bundle


def _sec_periods_with_retry(ticker: str, settings, attempts: int = 6,
                            base_delay: float = 8.0, on_retry=None):
    """SEC fetch with exponential backoff -- the SEC endpoint intermittently drops
    connections / throttles on long multi-ticker runs. Gentler defaults (more attempts,
    longer backoff) than the original to survive a full 100-name confirmation run.
    `on_retry(ticker, attempt)` is called before each backoff sleep so the run tracker
    can count SEC retries."""
    delay = base_delay
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return SECClient(settings).financial_periods(ticker)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if on_retry is not None:
                on_retry(ticker, attempt)
            time.sleep(delay)
            delay *= 2.0
    raise last


def _v1_signal(result) -> float:
    price = result.market_price
    diag = result.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
    intrinsic = float(diag) if isinstance(diag, (int, float)) else float(result.intrinsic_value_per_share)
    return intrinsic / price - 1.0


def _checkpoint_dir(n: int, suffix: str) -> Path:
    d = ROOT / "reports" / "research_v2" / "_checkpoints" / f"compare_n{n}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _process_ticker(ticker: str, cutoffs, v1_settings, v2_settings, on_sec_retry, heartbeat):
    """Value one ticker across all PIT cutoffs with both engines. Returns a self-contained
    checkpoint payload: {ticker, rows, skips, outcome}. Never raises -- failures are recorded
    in `outcome` so the orchestrator can resume cleanly."""
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
        row.update({f"v2_{k}": v for k, v in r2.monte_carlo_distribution.get("signals", {}).items()})
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


def run(tickers: list[str], reduced: bool = True, terminal_mode: str = "equilibrium",
        ckpt_dir: Path | None = None, tracker=None) -> dict[str, Any]:
    """Resumable V1-vs-V2 comparison. Per-ticker results are checkpointed to `ckpt_dir`;
    a re-run skips already-completed tickers. `tracker` (RunTracker) is updated as the run
    progresses so `research status` / `research watch` can monitor a backgrounded job."""
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
        v2_terminal_mode=terminal_mode,
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
                if payload.get("outcome", "").startswith("failed"):
                    tracker.mark_failed(ticker, payload["outcome"])
                else:
                    tracker.mark_completed(ticker)
            print(f"[{ti}/{len(tickers)}] {ticker}: resumed from checkpoint "
                  f"({len(payload.get('rows', []))} obs, {payload.get('outcome')})", flush=True)
            continue

        time.sleep(1.0)  # gentle inter-ticker pacing for the SEC endpoint
        payload = _process_ticker(ticker, cutoffs, v1_settings, v2_settings, _on_sec_retry, _heartbeat)
        if ckpt_path is not None:
            _write_checkpoint(ckpt_path, payload)
        obs.extend(payload["rows"])
        for k, v in payload["skips"].items():
            skips[k] = skips.get(k, 0) + v
        if tracker is not None:
            if payload["outcome"].startswith("failed"):
                tracker.mark_failed(ticker, payload["outcome"])
            else:
                tracker.mark_completed(ticker)
        print(f"[{ti}/{len(tickers)}] {ticker}: {len(payload['rows'])} PIT obs ({payload['outcome']})", flush=True)

    metrics = {}
    for signal in ("v1_discount", "v2_discount"):
        metrics[signal] = {f"{y}y": _metrics_for_horizon(obs, y, signal) for y in (3, 5)}
    summary = {sig: {h: _signal_summary(metrics[sig][h]) for h in ("3y", "5y")} for sig in metrics}
    return {"n_tickers": len(tickers), "n_obs": len(obs), "skips": skips, "sec_retries": _sec_retries["n"],
            "summary": summary, "metrics": metrics, "observations": obs}


def _signal_summary(m: dict[str, Any]) -> dict[str, Any]:
    if "spearman_corr_pooled" not in m:
        return {"n_obs": m.get("n_obs", 0), "note": "insufficient"}
    ic = m.get("information_coefficient_mean")
    ic_std = m.get("information_coefficient_std")
    n_p = m.get("information_coefficient_n_periods") or 0
    t_ic = (ic / ic_std * np.sqrt(n_p)) if (ic is not None and ic_std) else None
    dec = m.get("decile_table", [])
    mono = float(stats.spearmanr([r["bucket"] for r in dec],
                                 [r["mean_fwd_ann_return"] for r in dec]).statistic) if len(dec) >= 3 else None
    top_dec = (dec[-1]["mean_fwd_ann_return"] - float(np.mean([r["mean_fwd_ann_return"] for r in dec]))) if dec else None
    return {
        "n_obs": m["n_obs"], "ic_mean": ic, "ic_t": t_ic, "ic_n_periods": n_p,
        "spearman": m["spearman_corr_pooled"], "spearman_p": m["spearman_pvalue"],
        "long_short": m["long_short_top_minus_bottom"], "decile_monotonicity": mono,
        "top_decile_spread": top_dec, "hit_rate_beats_median": m.get("hit_rate_undervalued_beats_median"),
    }


def _f(v, p=4) -> str:
    return f"{v:+.{p}f}" if isinstance(v, (int, float)) else "   n/a"


def _print_summary(payload: dict[str, Any], out_path: Path) -> None:
    print(f"\n=== P2.5 V1 vs V2 forecasting comparison (n_tickers={payload['n_tickers']}, "
          f"n_obs={payload['n_obs']}) ===")
    print(f"skips={payload['skips']}")
    hdr = (f"{'horizon':>7} {'engine':>4} {'n':>4} {'IC':>8} {'IC_t':>6} {'spear':>8} "
           f"{'L/S':>8} {'decMono':>8} {'topDec':>8} {'hit>med':>8}")
    print(hdr)
    for y in (3, 5):
        for signal in ("v1_discount", "v2_discount"):
            s = payload["summary"][signal][f"{y}y"]
            tag = signal.split("_")[0]
            if "spearman" not in s:
                print(f"{y}y".rjust(7) + f" {tag:>4} {s.get('n_obs',0):>4}  insufficient")
                continue
            print(
                f"{y}y".rjust(7) + f" {tag:>4} {s['n_obs']:>4} {_f(s['ic_mean']):>8} {_f(s['ic_t'],2):>6} "
                f"{_f(s['spearman']):>8} {_f(s['long_short']):>8} {_f(s['decile_monotonicity']):>8} "
                f"{_f(s['top_decile_spread']):>8} {_f(s['hit_rate_beats_median'],3):>8}"
            )
        print()
    print(f"wrote {out_path}")


def _journal_on_completion(payload: dict[str, Any], n: int, out_path: Path, command: str) -> None:
    """Append an append-only journal entry summarizing this run (best-effort)."""
    try:
        from journal import append_entry  # scripts/utilities/research_v2 is on sys.path
    except Exception:  # noqa: BLE001
        return

    def _cell(sig, h):
        s = payload["summary"][sig][f"{h}y"]
        if "spearman" not in s:
            return "insufficient"
        return (f"IC {s['ic_mean']:.4f} (t={s['ic_t']:.2f}) spear {s['spearman']:.4f} "
                f"L/S {s['long_short']:.4f} decMono {s['decile_monotonicity']:.3f}")

    metrics = {f"{h}y": {"v1": _cell('v1_discount', h), "v2": _cell('v2_discount', h)} for h in (3, 5)}
    n_names = len({r["ticker"] for r in payload["observations"]})
    append_entry({
        "title": f"n={n} CLEAN confirmation run (resumable, tracker-monitored)",
        "hypothesis": "Confirm the n=50 favorable result: does posterior-coherent V2 out-rank V1 at n=100?",
        "sample_size": f"intended n={n}; tickers with obs={n_names}; total obs={payload['n_obs']}; "
                       f"SEC retries={payload.get('sec_retries')}",
        "engine": "v1 (frozen control) vs v2 (reduced MCMC: draws=300/chains=2/warmup=200/paths=400)",
        "command": command,
        "output_artifact": str(out_path.relative_to(ROOT)),
        "metrics": metrics,
        "interpretation": "Auto-logged on run completion. Compare v1 vs v2 deltas against the n=50 result; "
                          "see reports/research_v2/P2.5_conclusion.md for the confirms/contradicts verdict.",
        "decision": "Pending review in P2.5_conclusion.md (do not advance past P2.5 until confirmed/contradicted).",
        "next_action": "Update P2.5_conclusion.md with the confirms/contradicts verdict and present P3 go/no-go.",
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of tickers from the tranche")
    ap.add_argument("--full-mcmc", action="store_true", help="use full MCMC budget (slow)")
    ap.add_argument("--terminal", choices=["equilibrium", "v1"], default="equilibrium",
                    help="v2 terminal value: equilibrium (U3) or v1-held-constant (diagnostic)")
    ap.add_argument("--no-tracker", action="store_true", help="disable the run tracker")
    ap.add_argument("--no-journal", action="store_true", help="do not append a journal entry on completion")
    args = ap.parse_args()
    tickers = TRANCHE[: args.n]
    suffix = "" if args.terminal == "equilibrium" else f"_term-{args.terminal}"
    ckpt_dir = _checkpoint_dir(args.n, suffix)
    out_dir = ROOT / "reports" / "research_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forecasting_compare_{args.n}{suffix}.json"
    command = f"PYTHONPATH=src python scripts/utilities/research_v2/forecasting_compare.py --n {args.n}"

    tracker = None
    if not args.no_tracker:
        from dcf_project.research.run_tracker import RunTracker
        tracker = RunTracker.start(
            run_id=f"compare_n{args.n}{suffix}",
            command=command, phase=f"P2.5/n={args.n}",
            tranche_size=len(tickers), output_path=str(out_path.relative_to(ROOT)),
        )

    try:
        payload = run(tickers, reduced=not args.full_mcmc, terminal_mode=args.terminal,
                      ckpt_dir=ckpt_dir, tracker=tracker)
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
