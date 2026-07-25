"""Backtest metrics + report + top-N sweep for the DCF-discount portfolio.

Consumes the quarterly return series from ``portfolio_sim.simulate`` and produces, vs SPY & QQQ:
  CAGR (gross & net), Jensen's alpha + beta, Sharpe, Sortino, max drawdown, annual turnover,
  sector concentration (max sector weight + HHI), tracking error, and year-by-year outperformance
  consistency. Writes equity_curve.csv, metrics.json, portfolio_report.md, and (with --sweep)
  topn_sweep.csv.

Quarterly periods -> annualized with 4 periods/yr. Risk-free assumed 0 for Sharpe/alpha (excess
returns dominated by equity risk premium at this horizon; stated in the report).

Usage:
    python -m scripts.portfolio.portfolio_metrics                       # baseline top-20, both signals
    python -m scripts.portfolio.portfolio_metrics --sweep              # + top-N sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.portfolio.portfolio_sim import (
    BENCHMARKS,
    SimConfig,
    SimResult,
    load_price_panel,
    simulate,
)

PORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "portfolio"
PPY = 4  # periods per year (quarterly)


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    nav = float((1 + r).prod())
    yrs = len(r) / PPY
    return nav ** (1 / yrs) - 1 if nav > 0 else float("nan")


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((r.mean() - rf / PPY) / sd * np.sqrt(PPY))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    r = r.dropna()
    downside = r[r < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float((r.mean() - rf / PPY) / dd * np.sqrt(PPY))


def max_drawdown(r: pd.Series) -> float:
    nav = (1 + r.dropna()).cumprod()
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def alpha_beta(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """Annualized Jensen's alpha and beta from OLS of portfolio on benchmark (rf=0)."""
    df = pd.concat([r, bench], axis=1).dropna()
    if len(df) < 4:
        return float("nan"), float("nan")
    y = df.iloc[:, 0].to_numpy()
    x = df.iloc[:, 1].to_numpy()
    beta, intercept = np.polyfit(x, y, 1)
    alpha_ann = (1 + intercept) ** PPY - 1  # per-period alpha annualized
    return float(alpha_ann), float(beta)


def tracking_error(r: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([r, bench], axis=1).dropna()
    if len(df) < 2:
        return float("nan")
    active = df.iloc[:, 0] - df.iloc[:, 1]
    return float(active.std(ddof=1) * np.sqrt(PPY))


def annual_outperformance(r: pd.Series, bench: pd.Series) -> dict:
    df = pd.concat([r.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty:
        return {}
    yearly = df.groupby(df.index.year).apply(lambda g: pd.Series({
        "port": (1 + g["p"]).prod() - 1,
        "bench": (1 + g["b"]).prod() - 1,
    }))
    wins = int((yearly["port"] > yearly["bench"]).sum())
    return {
        "years": int(len(yearly)),
        "win_years": wins,
        "win_rate": round(wins / len(yearly), 3),
        "by_year": {int(y): {"port": round(row["port"], 4), "bench": round(row["bench"], 4),
                              "beat": bool(row["port"] > row["bench"])}
                    for y, row in yearly.iterrows()},
    }


def concentration(res: SimResult) -> dict:
    """Average max single-sector weight and HHI across rebalances."""
    max_w, hhi = [], []
    for _, sw in res.sector_weights.items():
        if not sw:
            continue
        ws = np.array(list(sw.values()))
        max_w.append(ws.max())
        hhi.append(float((ws ** 2).sum()))
    return {"avg_max_sector_weight": round(float(np.mean(max_w)), 3) if max_w else None,
            "avg_sector_hhi": round(float(np.mean(hhi)), 3) if hhi else None}


def compute_metrics(res: SimResult) -> dict:
    out: dict = {
        "config": {"signal": res.cfg.signal, "top_n": res.cfg.top_n,
                   "cost_bps": res.cfg.cost_bps, "entry_lag_months": res.cfg.entry_lag_months},
        "n_periods": int(len(res.long_net)),
        "cagr_gross": round(cagr(res.long_gross), 4),
        "cagr_net": round(cagr(res.long_net), 4),
        "cagr_universe_ew": round(cagr(res.universe), 4),
        "cagr_short_leg": round(cagr(res.short_leg), 4),
        "cagr_long_short": round(cagr(res.long_short), 4),
        "sharpe_net": round(sharpe(res.long_net), 3),
        "sortino_net": round(sortino(res.long_net), 3),
        "max_drawdown_net": round(max_drawdown(res.long_net), 4),
        "avg_turnover_1sided_q": round(float(res.turnover.mean()), 3),
        "annual_turnover_approx": round(float(res.turnover.mean()) * PPY, 3),
        "concentration": concentration(res),
        "vs_benchmark": {},
    }
    for b, br in res.bench.items():
        a, beta = alpha_beta(res.long_net, br)
        out["vs_benchmark"][b] = {
            "bench_cagr": round(cagr(br), 4),
            "alpha_annual_net": round(a, 4),
            "beta": round(beta, 3),
            "tracking_error_annual": round(tracking_error(res.long_net, br), 4),
            "outperformance": annual_outperformance(res.long_net, br),
        }
    return out


def write_equity_curve(res: SimResult) -> Path:
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "long_net_ret": res.long_net,
        "long_gross_ret": res.long_gross,
        "universe_ret": res.universe,
        "short_leg_ret": res.short_leg,
        "long_short_ret": res.long_short,
        "turnover_1sided": res.turnover,
    })
    for b, br in res.bench.items():
        df[f"{b}_ret"] = br
    df["nav_long_net"] = (1 + res.long_net).cumprod()
    for b, br in res.bench.items():
        df[f"nav_{b}"] = (1 + br).cumprod()
    path = PORT_DIR / "equity_curve.csv"
    df.to_csv(path)
    return path


def _fmt_pct(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def write_report(metrics: dict, res: SimResult) -> Path:
    m = metrics
    lines: list[str] = []
    L = lines.append
    L("# DCF-Discount Portfolio -- Backtest Report")
    L("")
    L(f"Signal: **{m['config']['signal']}**, top-N = **{m['config']['top_n']}**, "
      f"cost = **{m['config']['cost_bps']} bps/side**, quarterly rebalance, "
      f"{m['n_periods']} quarters (~{m['n_periods'] / 4:.1f} yrs).")
    L("")
    L("## Headline (net of costs)")
    L("")
    L("| Metric | Portfolio (top-N) | Universe EW | SPY | QQQ |")
    L("|---|---|---|---|---|")
    spy = m["vs_benchmark"].get("SPY", {})
    qqq = m["vs_benchmark"].get("QQQ", {})
    L(f"| CAGR | **{_fmt_pct(m['cagr_net'])}** | {_fmt_pct(m['cagr_universe_ew'])} | "
      f"{_fmt_pct(spy.get('bench_cagr'))} | {_fmt_pct(qqq.get('bench_cagr'))} |")
    L(f"| Alpha (ann, net) | -- | -- | {_fmt_pct(spy.get('alpha_annual_net'))} | "
      f"{_fmt_pct(qqq.get('alpha_annual_net'))} |")
    L(f"| Beta | -- | -- | {spy.get('beta', 'N/A')} | {qqq.get('beta', 'N/A')} |")
    L(f"| Tracking error (ann) | -- | -- | {_fmt_pct(spy.get('tracking_error_annual'))} | "
      f"{_fmt_pct(qqq.get('tracking_error_annual'))} |")
    L("")
    L(f"- **Sharpe (net):** {m['sharpe_net']}  |  **Sortino (net):** {m['sortino_net']}  |  "
      f"**Max drawdown:** {_fmt_pct(m['max_drawdown_net'])}")
    L(f"- **Gross CAGR:** {_fmt_pct(m['cagr_gross'])} (cost drag "
      f"{_fmt_pct(m['cagr_gross'] - m['cagr_net'])}/yr at {m['config']['cost_bps']} bps/side)")
    L(f"- **Turnover:** {_fmt_pct(m['avg_turnover_1sided_q'])}/quarter one-sided "
      f"(~{_fmt_pct(m['annual_turnover_approx'])}/yr)")
    L(f"- **Sector concentration:** avg max sector weight "
      f"{_fmt_pct(m['concentration']['avg_max_sector_weight'])}, "
      f"HHI {m['concentration']['avg_sector_hhi']}")
    L("")
    L("## Signal diagnostic (is the tilt real?)")
    L("")
    L(f"- Top-N **net {_fmt_pct(m['cagr_net'])}** vs equal-weight-all-eligible "
      f"**{_fmt_pct(m['cagr_universe_ew'])}** vs bottom-N **{_fmt_pct(m['cagr_short_leg'])}**.")
    L(f"- Long-short (topN − bottomN) **{_fmt_pct(m['cagr_long_short'])}**/yr -- the clean spread the "
      f"validation IC predicts; positive = the discount ordering carries return.")
    L("  The decile/quintile monotonicity from the validation study should show up here as "
      "top-N > universe > bottom-N.")
    L("")
    L("## Year-by-year outperformance consistency")
    L("")
    for b in BENCHMARKS:
        op = m["vs_benchmark"].get(b, {}).get("outperformance", {})
        if not op:
            continue
        L(f"**vs {b}:** beat in {op['win_years']}/{op['years']} years "
          f"({_fmt_pct(op['win_rate'])}).")
        L("")
        L(f"| Year | Portfolio | {b} | Beat |")
        L("|---|---|---|---|")
        for y, row in op["by_year"].items():
            L(f"| {y} | {_fmt_pct(row['port'])} | {_fmt_pct(row['bench'])} | "
              f"{'✅' if row['beat'] else '❌'} |")
        L("")
    L("## Caveats (read before trusting the CAGR)")
    L("")
    L("- **Survivorship bias.** The 365-name universe is *current membership* of the validated "
      "sectors; delisted/bankrupt names are absent. The equal-weight universe itself returns "
      f"{_fmt_pct(m['cagr_universe_ew'])} (vs SPY {_fmt_pct(spy.get('bench_cagr'))}) -- most of that "
      "gap is survivorship, not skill. **Judge the signal by top-N *minus universe* "
      f"(~{_fmt_pct((m['cagr_net'] or 0) - (m['cagr_universe_ew'] or 0))}/yr) and by long-short, "
      "not by the raw CAGR vs SPY.**")
    L("- **Point-in-time discounts, forward returns.** Signal at quarter-end *t* (fundamentals "
      "public by then) → returns measured strictly *t→t+1Q*. No look-ahead in selection.")
    L("- **Costs.** Net series charges "
      f"{m['config']['cost_bps']} bps/side on traded notional; gross also reported.")
    L("- **rf = 0** for Sharpe/alpha (short sample, equity-dominated). Alpha is Jensen's "
      "(regression intercept) annualized.")
    L("")
    path = PORT_DIR / "portfolio_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_sweep(prices: pd.DataFrame, signals=("discount", "discount_x_conf"),
              ns=(10, 20, 30, 50, 100)) -> Path:
    rows = []
    for sig in signals:
        for n in ns:
            res = simulate(SimConfig(signal=sig, top_n=n), prices=prices)
            mx = compute_metrics(res)
            spy = mx["vs_benchmark"].get("SPY", {})
            qqq = mx["vs_benchmark"].get("QQQ", {})
            rows.append({
                "signal": sig, "top_n": n,
                "cagr_net": mx["cagr_net"], "cagr_gross": mx["cagr_gross"],
                "cagr_universe_ew": mx["cagr_universe_ew"],
                "cagr_long_short": mx["cagr_long_short"],
                "sharpe_net": mx["sharpe_net"], "sortino_net": mx["sortino_net"],
                "max_dd_net": mx["max_drawdown_net"],
                "alpha_spy": spy.get("alpha_annual_net"), "beta_spy": spy.get("beta"),
                "alpha_qqq": qqq.get("alpha_annual_net"), "beta_qqq": qqq.get("beta"),
                "turnover_1sided_q": mx["avg_turnover_1sided_q"],
                "spy_winrate": spy.get("outperformance", {}).get("win_rate"),
            })
    df = pd.DataFrame(rows)
    path = PORT_DIR / "topn_sweep.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Portfolio backtest metrics + report")
    ap.add_argument("--signal", choices=["discount", "discount_x_conf"], default="discount")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--sweep", action="store_true", help="also run the top-N sweep")
    args = ap.parse_args()

    PORT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_price_panel()
    res = simulate(SimConfig(signal=args.signal, top_n=args.top_n, cost_bps=args.cost_bps), prices=prices)
    metrics = compute_metrics(res)

    ec = write_equity_curve(res)
    (PORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rep = write_report(metrics, res)
    print(f"baseline: signal={args.signal} top_n={args.top_n} cost={args.cost_bps}bps")
    print(f"  net CAGR {metrics['cagr_net']:.4f} | universe {metrics['cagr_universe_ew']:.4f} | "
          f"SPY {metrics['vs_benchmark'].get('SPY', {}).get('bench_cagr')} | "
          f"QQQ {metrics['vs_benchmark'].get('QQQ', {}).get('bench_cagr')}")
    print(f"  Sharpe {metrics['sharpe_net']} maxDD {metrics['max_drawdown_net']} "
          f"alpha_SPY {metrics['vs_benchmark'].get('SPY', {}).get('alpha_annual_net')}")
    print(f"  wrote {ec.name}, metrics.json, {rep.name}")

    if args.sweep:
        sp = run_sweep(prices)
        print(f"  wrote {sp.name}")


if __name__ == "__main__":
    main()
