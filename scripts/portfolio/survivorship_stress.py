"""Stage A.2 -- delisting stress test: bound the survivorship bias.

Survivorship bias is worst for a *value* strategy because the names that vanish (bankruptcies,
distressed take-unders) are disproportionately the cheap, distressed names the strategy BUYS. The
backtest never sees them. This test puts them back, synthetically, and asks: how much delisting in
the long book would it take to erase the edge?

Model (Monte Carlo): at each quarterly rebalance, every name in the long (deep-discount) book
independently "delists" with per-quarter probability d, realizing a loss-given-delisting -L instead
of its observed forward return; survivors keep their observed return. Sweep (d, L), average over many
seeds, and report:
  - long-only net CAGR & Jensen alpha vs SPY as a function of d
  - the long-short spread (delisting applied to the long leg only -- the conservative case)
  - the BREAKEVEN d* where the edge dies (top-N CAGR == SPY; and where long-short == 0)
  - the EDGAR-calibrated d (from survivorship_holes.json) with a distress multiplier, and how much
    alpha survives there.

Usage:
    python -m scripts.portfolio.survivorship_stress
    python -m scripts.portfolio.survivorship_stress --signal discount_x_conf --top-n 10 --sims 800
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.portfolio.portfolio_sim import SimConfig, load_price_panel, selection_paths

PORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "portfolio"
PPY = 4


def _cagr(qrets: np.ndarray) -> float:
    nav = float(np.prod(1.0 + qrets))
    yrs = len(qrets) / PPY
    return nav ** (1 / yrs) - 1 if nav > 0 else -1.0


def _alpha_ann(port: np.ndarray, bench: np.ndarray) -> float:
    if len(port) < 4:
        return float("nan")
    beta, intercept = np.polyfit(bench, port, 1)
    return (1 + intercept) ** PPY - 1


def edgar_hazard(distress_mult: float, base_window_years: int = 4) -> tuple[float, dict] | tuple[None, None]:
    """Convert the measured EDGAR hole rate into a per-quarter delisting hazard for held names."""
    p = PORT_DIR / "survivorship_holes.json"
    if not p.exists():
        return None, None
    res = json.load(open(p))
    hole_rate = res.get("totals", {}).get("hole_rate")
    if hole_rate is None:
        return None, None
    # window hole rate over the years the cohort had to delist (window->today ~ 8-11 yrs); use a
    # conservative effective horizon = base window span + ~5y to 2026.
    horizon_years = base_window_years + 6
    annual_hazard = 1 - (1 - hole_rate) ** (1 / horizon_years)
    q_hazard_avg = 1 - (1 - annual_hazard) ** (1 / 4)
    q_hazard_distress = min(0.25, q_hazard_avg * distress_mult)
    info = {"hole_rate": hole_rate, "horizon_years": horizon_years,
            "annual_hazard_avg": round(annual_hazard, 4),
            "q_hazard_avg": round(q_hazard_avg, 4),
            "distress_mult": distress_mult,
            "q_hazard_distress": round(q_hazard_distress, 4)}
    return q_hazard_distress, info


def stress(paths: list[dict], d: float, L: float, sims: int, rng: np.random.Generator,
           apply_short: float = 0.0) -> dict:
    """Monte Carlo: return mean long-net CAGR, alpha vs SPY, and long-short CAGR under hazard d."""
    spy = np.array([p["bench"].get("SPY", np.nan) for p in paths])
    valid = ~np.isnan(spy)
    long_cagrs, alphas, ls_cagrs = [], [], []
    for _ in range(sims):
        long_q, ls_q = [], []
        for p in paths:
            top = p["top"]
            n = len(top)
            base = np.array([r for _, r in top])
            delist = rng.random(n) < d
            rets = np.where(delist, -L, base)
            long_r = rets.mean()
            long_q.append(long_r)
            bot = p["bottom"]
            nb = len(bot)
            bret = np.array([r for _, r in bot])
            if apply_short > 0:
                bdel = rng.random(nb) < (d * apply_short)
                bret = np.where(bdel, -L, bret)
            ls_q.append(long_r - bret.mean())
        long_q = np.array(long_q)
        ls_q = np.array(ls_q)
        long_cagrs.append(_cagr(long_q))
        ls_cagrs.append(_cagr(ls_q))
        if valid.any():
            alphas.append(_alpha_ann(long_q[valid], spy[valid]))
    return {
        "d": d, "L": L,
        "long_cagr_mean": float(np.mean(long_cagrs)),
        "long_cagr_p10": float(np.percentile(long_cagrs, 10)),
        "alpha_spy_mean": float(np.nanmean(alphas)) if alphas else float("nan"),
        "alpha_pos_frac": float(np.mean(np.array(alphas) > 0)) if alphas else float("nan"),
        "long_short_cagr_mean": float(np.mean(ls_cagrs)),
        "long_short_pos_frac": float(np.mean(np.array(ls_cagrs) > 0)),
    }


def breakeven_d(paths: list[dict], L: float, sims: int, rng: np.random.Generator,
                target: str, spy_cagr: float) -> float | None:
    """Find d where the metric crosses its threshold (alpha->0 / long_short->0 / CAGR->SPY)."""
    grid = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20]
    prev_d, prev_v = None, None
    for d in grid:
        s = stress(paths, d, L, sims, rng)
        if target == "alpha":
            v = s["alpha_spy_mean"]
            thr = 0.0
        elif target == "cagr_vs_spy":
            v = s["long_cagr_mean"]
            thr = spy_cagr
        else:  # long_short
            v = s["long_short_cagr_mean"]
            thr = 0.0
        if prev_v is not None and (prev_v - thr) * (v - thr) <= 0 and prev_v != v:
            # linear interpolation between prev_d and d
            frac = (prev_v - thr) / (prev_v - v)
            return round(prev_d + frac * (d - prev_d), 4)
        prev_d, prev_v = d, v
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Delisting stress test (survivorship bound)")
    ap.add_argument("--signal", choices=["discount", "discount_x_conf"], default="discount")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--sims", type=int, default=600)
    ap.add_argument("--loss", type=float, default=0.7, help="loss-given-delisting (0.7 = -70%)")
    ap.add_argument("--distress-mult", type=float, default=2.0,
                    help="value names delist at this multiple of the average hazard")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prices = load_price_panel()
    cfg = SimConfig(signal=args.signal, top_n=args.top_n)
    paths = selection_paths(cfg, prices)
    spy = np.array([p["bench"].get("SPY", np.nan) for p in paths])
    spy_cagr = _cagr(spy[~np.isnan(spy)])
    base = stress(paths, 0.0, args.loss, 1, rng)  # d=0 reproduces the observed backtest
    print(f"signal={args.signal} top_n={args.top_n} | observed long CAGR "
          f"{base['long_cagr_mean'] * 100:.1f}% vs SPY {spy_cagr * 100:.1f}% "
          f"| long-short {base['long_short_cagr_mean'] * 100:.1f}%")

    # sweep d at the chosen loss
    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
    rows = [stress(paths, d, args.loss, args.sims, rng) for d in grid]

    print(f"\nDelisting hazard sweep (loss-given-delisting L={args.loss:.0%}, {args.sims} sims):")
    print(f"{'d/qtr':>7} {'longCAGR':>9} {'p10':>8} {'alphaSPY':>9} {'a>0%':>6} "
          f"{'LS CAGR':>8} {'LS>0%':>6}")
    for r in rows:
        print(f"{r['d']:>7.3f} {r['long_cagr_mean'] * 100:>8.1f}% {r['long_cagr_p10'] * 100:>7.1f}% "
              f"{r['alpha_spy_mean'] * 100:>8.1f}% {r['alpha_pos_frac'] * 100:>5.0f}% "
              f"{r['long_short_cagr_mean'] * 100:>7.1f}% {r['long_short_pos_frac'] * 100:>5.0f}%")

    be_alpha = breakeven_d(paths, args.loss, args.sims, rng, "alpha", spy_cagr)
    be_spy = breakeven_d(paths, args.loss, args.sims, rng, "cagr_vs_spy", spy_cagr)
    be_ls = breakeven_d(paths, args.loss, args.sims, rng, "long_short", spy_cagr)
    print(f"\nBreakeven per-quarter delisting hazard (loss {args.loss:.0%}):")
    print(f"  alpha vs SPY -> 0 at d* = {be_alpha}")
    print(f"  long CAGR    -> SPY at d* = {be_spy}")
    print(f"  long-short   -> 0 at d* = {be_ls}")

    q_haz, info = edgar_hazard(args.distress_mult)
    cal = None
    if q_haz is not None:
        cal = stress(paths, q_haz, args.loss, args.sims, rng)
        print(f"\nEDGAR-calibrated hazard: hole_rate={info['hole_rate']:.1%} -> "
              f"avg q-hazard {info['q_hazard_avg']:.1%} x{info['distress_mult']} distress "
              f"= {q_haz:.1%}/qtr")
        print(f"  at calibrated d: long CAGR {cal['long_cagr_mean'] * 100:.1f}% "
              f"(p10 {cal['long_cagr_p10'] * 100:.1f}%), alpha {cal['alpha_spy_mean'] * 100:.1f}%, "
              f"long-short {cal['long_short_cagr_mean'] * 100:.1f}%")

    out = {
        "config": {"signal": args.signal, "top_n": args.top_n, "loss_given_delisting": args.loss,
                   "sims": args.sims, "distress_mult": args.distress_mult},
        "observed": {"long_cagr": base["long_cagr_mean"], "spy_cagr": spy_cagr,
                     "long_short_cagr": base["long_short_cagr_mean"]},
        "hazard_sweep": rows,
        "breakeven": {"alpha_vs_spy": be_alpha, "cagr_vs_spy": be_spy, "long_short": be_ls},
        "edgar_calibration": info, "at_calibrated_hazard": cal,
    }
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    (PORT_DIR / "survivorship_stress.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    per_cfg = PORT_DIR / f"survivorship_stress_{args.signal}_top{args.top_n}.json"
    per_cfg.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote survivorship_stress.json and {per_cfg.name}")


if __name__ == "__main__":
    main()
