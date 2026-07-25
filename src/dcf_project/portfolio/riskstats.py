"""Book-agnostic risk statistics for a managed book -> <book>/risk_stats.json.

Generalized from scripts/portfolio/india_v3_riskstats.py so BOTH books get equal-depth risk
pages (the India script remains the India book's authoring tool for its extras: analyst targets,
logos, the Phase-B ex-ante block). The core here is market-data-driven and method-identical for
every book: 3y weekly total returns for each holding + the book's primary benchmark ->

  * per-name annualised vol, beta, correlation to benchmark, trailing-12m dividend yield
  * holding-to-holding correlation matrix and average pairwise correlation
  * portfolio (weighted) beta, vol, tracking error, dividend yield
  * diversification ratio, effective independent bets, HHI / effective N
  * realised CAPM alpha (regression intercept, annualised) and max drawdown -- BACKWARD-looking

rf convention is per-currency and documented (USD 0.0 matching portfolio_metrics.py's stated
convention; INR 0.068 matching the India export's assumption table). The EVIDENCE block cites
numbers computed by this repo's own studies -- nothing forward-looking is invented here.

Writes MERGE-style: keys already in risk_stats.json that this module does not compute (e.g. the
India script's `analyst` / `assumptions` blocks) are preserved, so the two tools compose.

CLI:  python -m dcf_project portfolio manage riskstats --book conf_top10
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from dcf_project.paths import MANAGED
from dcf_project.portfolio import registry, signals
from dcf_project.portfolio.manager import _benchmarks_of

WEEKS = 52.0
RF = {"USD": 0.0, "INR": 0.068}  # documented per-currency conventions (see module docstring)

# Cited from this repo's own backtests (reports/portfolio/topn_sweep.csv, discount_x_conf top10,
# 42 quarters 2015-2026, net 10 bps/side). The caveat is part of the record, not boilerplate.
EVIDENCE = {
    "conf_top10": {
        "backtest_alpha_spy": 0.0431, "backtest_beta_spy": 1.431,
        "backtest_alpha_qqq": 0.0555, "backtest_beta_qqq": 1.016,
        "source": "topn_sweep.csv discount_x_conf top_n=10, 42q 2015-2026, net 10bps/side",
        "caveat": "survivorship-biased universe; robust stats are long-short +15.6%/yr and "
                  "top-N minus universe +7.4%/yr",
    },
    "india_v3_aggressive": {
        "expected_alpha_1y": 0.0578, "basis": "Phase B 1y quintile L/S 0.289, long leg /2, "
        "60% survivorship haircut (india_v3_riskstats.py)",
        "caveat": "PRELIMINARY / survivorship-optimistic; NOT validated alpha",
    },
}


def _weights(holdings, closes) -> dict[str, float]:
    """Current weights: shares books by market value; equal-weight books by drifted growth
    (last/entry), matching manager._holding_rows so every surface agrees."""
    last = {t: float(c.iloc[-1]) for t, c in closes.items()}
    if any(h.get("shares") for h in holdings):
        mv = {h["ticker"]: (h.get("shares") or 0) * last.get(h["ticker"], 0.0)
              for h in holdings if h["ticker"] in last}
        tot = sum(mv.values())
        return {t: v / tot for t, v in mv.items()} if tot else {}
    grow = {h["ticker"]: last[h["ticker"]] / h["entry_price"]
            for h in holdings if h.get("entry_price") and h["ticker"] in last}
    tot = sum(grow.values())
    return {t: g / tot for t, g in grow.items()} if tot else {}


def stats_from_closes(holdings, closes, bench_close, *, rf: float = 0.0,
                      div_yield: dict[str, float | None] | None = None) -> dict:
    """Pure core: closes = {ticker: weekly Series}, bench_close = weekly Series. No network."""
    import pandas as pd

    div_yield = div_yield or {}
    px = pd.concat([bench_close.rename("__bench__"), *closes.values()], axis=1, join="inner")
    px.columns = ["__bench__", *closes.keys()]
    ret = px.pct_change().dropna()
    bench = ret["__bench__"].to_numpy(dtype=float)
    n = len(bench)
    var_m = float(np.var(bench, ddof=1))
    mean_m = float(np.mean(bench))
    tickers = [t for t in closes]
    R = np.vstack([ret[t].to_numpy(dtype=float) for t in tickers])

    per_name = {}
    w_map = _weights(holdings, closes)
    for i, t in enumerate(tickers):
        r = R[i]
        per_name[t] = {
            "weight": round(w_map.get(t, 0.0), 4),
            "vol": round(float(np.std(r, ddof=1) * np.sqrt(WEEKS)), 4),
            "beta": round(float(np.cov(r, bench, ddof=1)[0, 1] / var_m), 3) if var_m else None,
            "corr_bench": round(float(np.corrcoef(r, bench)[0, 1]), 3),
            "div_yield": round(div_yield[t], 4) if div_yield.get(t) is not None else None,
        }

    w = np.array([w_map.get(t, 0.0) for t in tickers])
    w = w / w.sum() if w.sum() else np.full(len(tickers), 1.0 / len(tickers))
    rp = w @ R
    cov = np.cov(R, ddof=1) if len(tickers) > 1 else np.array([[np.var(R[0], ddof=1)]])
    sig = np.sqrt(np.diag(cov))
    port_vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(WEEKS)
    weighted_avg_vol = float((w * sig).sum() * np.sqrt(WEEKS))
    div_ratio = weighted_avg_vol / port_vol if port_vol else None
    C = np.corrcoef(R) if len(tickers) > 1 else np.array([[1.0]])
    iu = np.triu_indices_from(C, k=1)
    hhi = float((w ** 2).sum())
    port_beta = float(np.cov(rp, bench, ddof=1)[0, 1] / var_m) if var_m else None
    te = float(np.std(rp - bench, ddof=1) * np.sqrt(WEEKS))
    rf_wk = rf / WEEKS
    realized_alpha = (float((np.mean(rp) - (rf_wk + port_beta * (mean_m - rf_wk))) * WEEKS)
                      if port_beta is not None else None)
    nav = np.cumprod(1.0 + rp)
    max_dd = float((nav / np.maximum.accumulate(nav) - 1.0).min())
    port_dy = float(sum(w[i] * (per_name[t]["div_yield"] or 0.0) for i, t in enumerate(tickers)))

    return {
        "window": f"{n} weekly obs (~{n / WEEKS:.1f}y), total-return basis",
        "benchmark_vol": round(float(np.std(bench, ddof=1) * np.sqrt(WEEKS)), 4),
        "rf": rf,
        "per_name": per_name,
        "corr_matrix": {"tickers": tickers, "matrix": np.round(C, 3).tolist()},
        "portfolio": {
            "beta": round(port_beta, 3) if port_beta is not None else None,
            "vol": round(port_vol, 4),
            "weighted_avg_vol": round(weighted_avg_vol, 4),
            "diversification_ratio": round(div_ratio, 3) if div_ratio else None,
            "effective_bets": round(div_ratio ** 2, 2) if div_ratio else None,
            "avg_pairwise_corr": round(float(C[iu].mean()), 3) if iu[0].size else 1.0,
            "hhi": round(hhi, 4),
            "effective_n_names": round(1.0 / hhi, 2) if hhi else None,
            "tracking_error": round(te, 4),
            "div_yield": round(port_dy, 4),
            "realized_alpha": round(realized_alpha, 4) if realized_alpha is not None else None,
            "max_drawdown": round(max_dd, 4),
        },
    }


def run_riskstats(book: str, years: float = 3.0) -> dict:
    p = registry.get(book)
    if not p:
        raise SystemExit(f"No managed portfolio '{book}'. See `portfolio list`.")
    folder = MANAGED / book
    state = json.load(open(folder / "state.json"))
    holdings = (state.get("books") or {}).get(p["book_id"]) or state.get("holdings", [])
    bench_entry = _benchmarks_of(state)
    primary = next(iter(bench_entry), None)
    if not primary:
        raise SystemExit(f"'{book}' has no benchmark in state.json.")
    ccy = state.get("currency", "USD")

    print(f"fetching ~{years:.0f}y weekly total returns for {len(holdings)} holdings + {primary} ...")
    bench_close = signals.weekly_close(primary, years)
    if bench_close is None:
        raise SystemExit(f"could not fetch benchmark ({primary}) returns; aborting.")
    closes, dy = {}, {}
    for h in holdings:
        sym, tk = h.get("symbol", h["ticker"]), h["ticker"]
        c = signals.weekly_close(sym, years)
        if c is None:
            print(f"  {tk:11s} skip (no data)")
            continue
        closes[tk] = c
        dy[tk] = signals.trailing_div_yield(sym, float(c.iloc[-1]))
        print(f"  {tk:11s} {len(c)} wk  divyld={(f'{dy[tk]:.1%}' if dy[tk] is not None else 'NA')}")
    if not closes:
        raise SystemExit("no holding price history fetched; aborting.")

    core = stats_from_closes(holdings, closes, bench_close, rf=RF.get(ccy, 0.0), div_yield=dy)
    core.update({"as_of": dt.date.today().isoformat(), "benchmark": primary,
                 "rf_basis": f"{ccy} convention (see riskstats.py docstring)"})
    if book in EVIDENCE:
        core["evidence"] = EVIDENCE[book]

    # MERGE: preserve keys another tool owns (India's analyst/assumptions blocks etc.)
    out_path = folder / "risk_stats.json"
    existing = json.load(open(out_path)) if out_path.exists() else {}
    existing.update(core)
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    pf = core["portfolio"]
    print(f"wrote {out_path}")
    print(f"  beta={pf['beta']}  vol={pf['vol']:.1%}  TE={pf['tracking_error']:.1%}  "
          f"divyld={pf['div_yield']:.1%}  realized alpha={pf['realized_alpha']}  "
          f"maxDD={pf['max_drawdown']:.1%}")
    return existing
