"""Rebalance-frequency & holding-horizon study on the frozen DCF-discount portfolio.

Every prior backtest rebalances **quarterly**. This module answers the user's "how long should we hold"
question along two axes -- without changing the DCF / forecast / WACC / ranking (a pure analysis layer):

  Analysis 1 -- REBALANCE FREQUENCY (implementable, non-overlapping). Hold a selected book for k quarters
    (k=1 quarterly, 2 semiannual, 4 annual), re-selecting only every k-th quarter and letting weights drift
    in between; charge turnover cost only at rebalances. Question: does alpha survive with fewer trades, does
    turnover drop ~1/k, and does CAPACITY improve? The capacity payoff is priced by reusing capacity_costs:
    fewer/larger-spaced trades -> higher 10-day-liquidation ceiling. (k=1 reproduces portfolio_sim.simulate.)

  Analysis 2 -- HOLDING HORIZON (overlapping cohorts). For each formation quarter t, buy the top-N by signal
    and the equal-weight universe, hold h quarters (h=4/8/12/20 = 1Y/2Y/3Y/5Y), and measure the annualized
    active return. Aggregate over overlapping cohorts with a Newey-West t-stat (lag h-1). Shows whether the
    signal strengthens with horizon -> whether to hold longer than the rebalance schedule implies.

Monthly rebalancing is intentionally OUT OF SCOPE: the signal panel and the price panel are quarter-end only
(fundamentals update quarterly), so rebalancing monthly on a quarterly-updating signal only churns cost with
no new information. Reported as a data limitation, not run.

Reuses (does NOT re-implement): portfolio_sim selection helpers, portfolio_metrics, capacity_costs cost model,
validation_common.newey_west_tstat. No src/ changes; portfolio_sim.py is NOT modified (helpers are composed).

Usage:
    python -m scripts.portfolio.rebalance_horizon
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.portfolio import capacity_costs as cc
from scripts.portfolio.build_liquidity_panel import load_liquidity_panel
from scripts.portfolio.portfolio_metrics import cagr, max_drawdown, sharpe
from scripts.portfolio.portfolio_sim import (
    BENCHMARKS, SimConfig, SimResult, _fwd_return, _signal_value, _waterfill_weights,
    load_price_panel, load_sector_reliability, load_signal_panel,
)
from scripts.validation_common import newey_west_tstat

PORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "portfolio"

FREQS = [(1, "Quarterly"), (2, "Semiannual"), (4, "Annual")]
HORIZONS_Q = [4, 8, 12, 20]  # 1Y / 2Y / 3Y / 5Y
# Books mirror the risk_ablation / capacity constructions so this ties to them.
BOOKS = [
    ("A_top10", dict(top_n=10)),
    ("C_sector20", dict(top_n=10, sector_cap=0.20)),
    ("D_sector25_pos5", dict(top_n=10, sector_cap=0.25, position_cap=0.05)),
]
CAP_AUMS = [1e7, 1e8]  # capacity tie-in reported at $10M and $100M


# --------------------------------------------------------------------------- Analysis 1: cadence engine
def simulate_cadence(cfg: SimConfig, k: int, prices: pd.DataFrame,
                     liquidity: pd.DataFrame | None = None) -> SimResult:
    """Hold a selected book for k quarters; re-select every k-th quarter, drift weights in between.

    Returns a SimResult populated with the fields downstream consumers need: long_gross/long_net/universe
    (quarterly series, comparable across k), turnover, n_names, bench, and book_weights recorded ONLY at
    rebalance quarters (so capacity_costs._book_deltas prices the trade at each rebalance). k=1 reproduces
    portfolio_sim.simulate to the basis point.
    """
    dates_str, panel = load_signal_panel()
    rel = load_sector_reliability()
    dates = [pd.Timestamp(d) for d in dates_str]
    position_cap = cfg.position_cap if cfg.position_cap else 1.0 / cfg.top_n

    res = SimResult(cfg=cfg)
    long_g, long_n, uni, turn, nn = {}, {}, {}, {}, {}
    prev_target: dict[str, float] = {}
    held: dict[str, float] = {}       # drifting weights of the currently held book
    qsr = k                            # quarters since rebalance; >=k forces the initial rebalance

    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        recs = panel.get(dates_str[i], {})
        elig: list[tuple[str, float, float, str]] = []
        for tk, rec in recs.items():
            r = _fwd_return(prices, tk, t, t1)
            if r is None:
                continue
            if cfg.liquidity_min_dollar_vol and liquidity is not None and tk in liquidity.columns:
                dv = liquidity.at[t, tk] if t in liquidity.index else None
                if dv is None or pd.isna(dv) or dv < cfg.liquidity_min_dollar_vol:
                    continue
            elig.append((tk, _signal_value(rec, cfg.signal, rel), r, rec["sector"]))
        # Skip a thin quarter entirely (no rebalance, no return, no hold-clock tick) -- exactly as simulate
        # does; this is what makes k=1 reproduce the canonical engine to the basis point.
        if len(elig) < cfg.min_eligible:
            continue
        uni_ret = sum(x[2] for x in elig) / len(elig)

        # Rebalance on schedule (initial entry, or k quarters since the last rebalance).
        cost = 0.0
        if not held or qsr >= k:
            elig.sort(key=lambda x: x[1], reverse=True)
            book = _waterfill_weights(elig, position_cap, cfg.sector_cap)
            target = {tk: w for tk, w, _r, _s in book}
            traded = sum(abs(target.get(tk, 0.0) - prev_target.get(tk, 0.0))
                         for tk in set(target) | set(prev_target))
            cost = traded * cfg.cost_bps / 1e4
            held = dict(target)
            res.book_weights[dates_str[i]] = dict(target)
            prev_target = target
            turn[t1] = traded / 2.0
            qsr = 0

        # Held-book quarterly return: drift held weights by realized returns; drop names that lose price
        # coverage mid-hold and renormalize over survivors (return measured on the tradeable portion).
        rsum, wsum, new_held = 0.0, 0.0, {}
        for tk, w in held.items():
            r = _fwd_return(prices, tk, t, t1)
            if r is None:
                continue
            rsum += w * r
            wsum += w
            new_held[tk] = w * (1.0 + r)
        book_ret = rsum / wsum if wsum > 1e-12 else 0.0
        if new_held:
            s = sum(new_held.values())
            held = {tk: v / s for tk, v in new_held.items()} if s > 0 else {}

        long_g[t1] = book_ret
        long_n[t1] = book_ret - cost
        uni[t1] = uni_ret
        nn[t1] = len(held)
        qsr += 1

    idx = pd.to_datetime(list(long_g))
    res.long_gross = pd.Series(long_g).reindex(idx)
    res.long_net = pd.Series(long_n).reindex(idx)
    res.universe = pd.Series(uni).reindex(idx)
    res.turnover = pd.Series(turn).reindex(idx).fillna(0.0)
    res.n_names = pd.Series(nn).reindex(idx)
    for b in BENCHMARKS:
        if b in prices.columns:
            br = {}
            for i in range(len(dates) - 1):
                t, t1 = dates[i], dates[i + 1]
                if t1 in idx:
                    r = _fwd_return(prices, b, t, t1)
                    if r is not None:
                        br[t1] = r
            res.bench[b] = pd.Series(br).reindex(idx)
    return res


def _ten_day_ceiling(res: SimResult, liquidity: pd.DataFrame) -> float:
    """AUM at which the largest single position would take >10 days to exit at <=20% ADV/day."""
    worst = cc._worst_weight_over_adv(res, liquidity)
    return (cc.DAYS_LIQ_LIMIT * cc.PARTICIPATION / worst) if worst > 0 else float("inf")


# --------------------------------------------------------------------------- Analysis 2: horizon decay
def _ann(cum: float, yrs: float) -> float:
    return (1.0 + cum) ** (1.0 / yrs) - 1.0 if (1.0 + cum) > 0 else -1.0


def holding_horizon(cfg: SimConfig, prices: pd.DataFrame) -> pd.DataFrame:
    """Overlapping-cohort h-quarter buy-and-hold active return (top-N vs universe) + decile spread, NW t."""
    dates_str, panel = load_signal_panel()
    rel = load_sector_reliability()
    dates = [pd.Timestamp(d) for d in dates_str]
    rows = []
    for h in HORIZONS_Q:
        yrs = h / 4.0
        actives, dec_spreads = [], []
        for i in range(len(dates) - h):
            t, th = dates[i], dates[i + h]
            elig = []
            for tk, rec in panel.get(dates_str[i], {}).items():
                p0 = prices.at[t, tk] if (t in prices.index and tk in prices.columns) else None
                p1 = prices.at[th, tk] if (th in prices.index and tk in prices.columns) else None
                if p0 is None or p1 is None or pd.isna(p0) or pd.isna(p1) or p0 <= 0:
                    continue
                elig.append((tk, _signal_value(rec, cfg.signal, rel), float(p1 / p0 - 1.0)))
            if len(elig) < cfg.min_eligible:
                continue
            elig.sort(key=lambda x: x[1], reverse=True)
            top = elig[:cfg.top_n]
            uni_cum = sum(x[2] for x in elig) / len(elig)
            top_cum = sum(x[2] for x in top) / len(top)
            actives.append(_ann(top_cum, yrs) - _ann(uni_cum, yrs))
            d = max(1, len(elig) // 10)  # decile tails
            dec_spreads.append(_ann(sum(x[2] for x in elig[:d]) / d, yrs)
                               - _ann(sum(x[2] for x in elig[-d:]) / d, yrs))
        nw = newey_west_tstat(actives, lag=h - 1)
        nwd = newey_west_tstat(dec_spreads, lag=h - 1)
        rows.append({
            "horizon_q": h, "horizon_y": yrs, "n_cohorts": len(actives),
            "mean_active_ann": round(nw["mean"], 4) if nw["mean"] is not None else None,
            "active_t_nw": round(nw["t_stat"], 2) if nw["t_stat"] is not None else None,
            "decile_spread_ann": round(nwd["mean"], 4) if nwd["mean"] is not None else None,
            "decile_t_nw": round(nwd["t_stat"], 2) if nwd["t_stat"] is not None else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- driver
def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = load_price_panel()
    liquidity = load_liquidity_panel()
    sigma = cc.estimate_daily_sigma(prices)

    combos = [(label, kw, "discount") for label, kw in BOOKS]
    combos.append(("C_sector20", dict(top_n=10, sector_cap=0.20), "discount_x_conf"))

    freq_rows, cap_rows = [], []
    for label, kw, signal in combos:
        for k, kname in FREQS:
            res = simulate_cadence(SimConfig(signal=signal, **kw), k, prices, liquidity)
            net, gross, u = cagr(res.long_net), cagr(res.long_gross), cagr(res.universe)
            n_rebal = len(res.book_weights)
            n_q = int(res.long_net.notna().sum())
            row = {
                "book": label, "signal": signal, "freq": kname, "hold_q": k,
                "n_rebalances": n_rebal, "n_quarters": n_q,
                "gross_cagr": round(gross, 4), "net_cagr": round(net, 4),
                "alpha_uni": round(net - u, 4), "sharpe": round(sharpe(res.long_net), 2),
                "max_dd": round(max_drawdown(res.long_net), 3),
                "turnover_yr": round(float(res.turnover.mean()) * 4.0, 3),
            }
            for b in BENCHMARKS:
                if b in res.bench:
                    row[f"alpha_{b.lower()}"] = round(net - cagr(res.bench[b]), 4)
            freq_rows.append(row)

            # Capacity tie-in (discount only, concentrated A vs breadth D) -- the "trade less buys capacity".
            if signal == "discount" and label in ("A_top10", "D_sector25_pos5"):
                cap = {"book": label, "freq": kname, "hold_q": k,
                       "turnover_yr": row["turnover_yr"],
                       "ceiling_10day": _ten_day_ceiling(res, liquidity)}
                for aum in CAP_AUMS:
                    real = cc.net_cagr(res, aum, liquidity, sigma, "realistic")
                    cap[f"alpha_uni_{int(aum/1e6)}M"] = round(real - u, 4)
                cap_rows.append(cap)

    horizon = holding_horizon(SimConfig(signal="discount", top_n=10), prices)
    return pd.DataFrame(freq_rows), pd.DataFrame(cap_rows), horizon


# --------------------------------------------------------------------------- report
def _pct(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:.1f}%"


def write_report(freq: pd.DataFrame, cap: pd.DataFrame, horizon: pd.DataFrame,
                 bench_cagr: dict[str, float]) -> Path:
    L: list[str] = []
    A = L.append
    A("# Rebalance Frequency & Holding Horizon")
    A("")
    A("How long should the DCF-discount book be held? Two complementary cuts on the *holding-period* axis, "
      "both pure analysis layers on the frozen signal (no DCF/forecast/WACC/ranking change). Primary metric: "
      "**alpha vs the equal-weight universe** (survivorship-neutral). Benchmarks over the sample: "
      f"SPY {_pct(bench_cagr.get('SPY'))}, QQQ {_pct(bench_cagr.get('QQQ'))} CAGR.")
    A("")
    A("> **Monthly is out of scope** -- the signal and price panels are quarter-end only (fundamentals update "
      "quarterly), so rebalancing monthly on a quarterly-updating signal only churns cost with no new "
      "information. Reported as a data limitation, not run.")
    A("")

    # ---- Analysis 1
    A("## 1. Rebalance frequency (implementable, non-overlapping)")
    A("")
    A("Hold each selected book for *k* quarters, re-selecting only every *k*-th quarter (weights drift in "
      "between); turnover cost (10 bps/side) is charged only at rebalances. Quarterly (*k*=1) reproduces the "
      "canonical simulator.")
    A("")
    for signal in ("discount", "discount_x_conf"):
        sub = freq[freq.signal == signal]
        if sub.empty:
            continue
        tag = "primary" if signal == "discount" else "survivorship-robust spot-check"
        A(f"### Signal: `{signal}`  *({tag})*")
        A("")
        A("| Book | Frequency | Rebals | Gross | Net | Alpha/Uni | vs SPY | Sharpe | Max DD | Turnover/yr |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            A(f"| {r['book']} | {r['freq']} | {int(r['n_rebalances'])} | {_pct(r['gross_cagr'])} | "
              f"{_pct(r['net_cagr'])} | **{_pct(r['alpha_uni'])}** | {_pct(r.get('alpha_spy'))} | "
              f"{r['sharpe']:.2f} | {_pct(r['max_dd'])} | {_pct(r['turnover_yr'])} |")
        A("")

    # ---- Capacity tie-in
    A("## 2. Does trading less buy capacity?")
    A("")
    A("Each frequency's actual trade profile repriced through the capacity cost model "
      "([[capacity-cost-resilience]]): the 10-day-liquidation ceiling (largest single position exitable in "
      "<=10 trading days at <=20% ADV/day) and net alpha vs universe at $10M / $100M (realistic execution).")
    A("")
    A("| Book | Frequency | Turnover/yr | 10-day capacity ceiling | Alpha/Uni @$10M | Alpha/Uni @$100M |")
    A("|---|---|---|---|---|---|")
    for _, r in cap.iterrows():
        A(f"| {r['book']} | {r['freq']} | {_pct(r['turnover_yr'])} | {cc._usd(r['ceiling_10day'])} | "
          f"{_pct(r['alpha_uni_10M'])} | {_pct(r['alpha_uni_100M'])} |")
    A("")

    # ---- Analysis 2
    A("## 3. Holding horizon (overlapping cohorts, signal decay)")
    A("")
    A("For each formation quarter, buy the top-10 by `discount` and the equal-weight universe and hold *h* "
      "quarters; annualized active return averaged across overlapping cohorts with a Newey-West t-stat "
      "(lag *h*-1). The decile spread is the top-vs-bottom-10% tail over the same horizon.")
    A("")
    A("| Horizon | Cohorts | Active/yr (top-10 vs uni) | NW t | Decile spread/yr | NW t |")
    A("|---|---|---|---|---|---|")
    for _, r in horizon.iterrows():
        flag = "  ⚠ thin" if r["horizon_q"] >= 20 else ""
        A(f"| {r['horizon_y']:.0f}Y ({int(r['horizon_q'])}q){flag} | {int(r['n_cohorts'])} | "
          f"**{_pct(r['mean_active_ann'])}** | {r['active_t_nw']} | {_pct(r['decile_spread_ann'])} | "
          f"{r['decile_t_nw']} |")
    A("")

    A("## Reading it")
    A("")
    _write_reading(L, freq, cap, horizon)
    path = PORT_DIR / "rebalance_horizon.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _write_reading(L: list[str], freq: pd.DataFrame, cap: pd.DataFrame, horizon: pd.DataFrame) -> None:
    A = L.append
    g = freq[freq.signal == "discount"]

    def fcell(book, kname, col):
        row = g[(g.book == book) & (g.freq == kname)]
        return row.iloc[0][col] if len(row) else float("nan")

    def ccell(book, kname, col):
        row = cap[(cap.book == book) & (cap.freq == kname)]
        return row.iloc[0][col] if len(row) else float("nan")

    qa = fcell("A_top10", "Quarterly", "alpha_uni")
    sa = fcell("A_top10", "Semiannual", "alpha_uni")
    aa = fcell("A_top10", "Annual", "alpha_uni")
    A(f"- **Quarterly is materially best; the edge is front-loaded.** Top-10 alpha vs universe falls from "
      f"{_pct(qa)} (quarterly) to {_pct(sa)} (semiannual) / {_pct(aa)} (annual) -- semiannual keeps only "
      f"~{sa / qa * 100:.0f}% of the edge, annual ~{aa / qa * 100:.0f}%. The risk-controlled book C holds up "
      f"better ({_pct(fcell('C_sector20', 'Quarterly', 'alpha_uni'))} → "
      f"{_pct(fcell('C_sector20', 'Annual', 'alpha_uni'))}, ~"
      f"{fcell('C_sector20', 'Annual', 'alpha_uni') / fcell('C_sector20', 'Quarterly', 'alpha_uni') * 100:.0f}% "
      "retained), but in every book **trading less sacrifices alpha** -- this is a medium-frequency value "
      "signal whose discount mostly closes within a year, not a slow one. (Semiannual lands below annual in "
      "the concentrated books -- a small-sample wrinkle on 11-21 rebalances; both sit well under quarterly.)")
    A(f"- **Trading less does NOT buy capacity -- breadth does.** The binding 10-day-liquidation ceiling is "
      f"set by the largest *position* (10% of book), which is identical at every cadence: top-10 stays at "
      f"**{cc._usd(ccell('A_top10', 'Quarterly', 'ceiling_10day'))}** quarterly→annual. Only spreading across "
      f"more names moves it -- the 20-name book sits at **{cc._usd(ccell('D_sector25_pos5', 'Annual', 'ceiling_10day'))}**. "
      "Lower frequency only trims *cost*, which is non-binding until ~$100M; there annual's lower turnover "
      f"lets it overtake quarterly (A @$100M: quarterly {_pct(ccell('A_top10', 'Quarterly', 'alpha_uni_100M'))} "
      f"vs annual {_pct(ccell('A_top10', 'Annual', 'alpha_uni_100M'))}), while at $10M quarterly still wins "
      f"({_pct(ccell('A_top10', 'Quarterly', 'alpha_uni_10M'))} vs {_pct(ccell('A_top10', 'Annual', 'alpha_uni_10M'))}). "
      "So the cadence/AUM choice has a crossover between $10M and $100M.")
    hr = horizon.set_index("horizon_q")
    A(f"- **The per-year edge is strongest early and decays.** Annualized active return (top-10 vs universe) "
      f"by hold horizon: 1Y {_pct(hr.loc[4, 'mean_active_ann'])} (NW t={hr.loc[4, 'active_t_nw']}), "
      f"2Y {_pct(hr.loc[8, 'mean_active_ann'])} (t={hr.loc[8, 'active_t_nw']}), "
      f"3Y {_pct(hr.loc[12, 'mean_active_ann'])} (t={hr.loc[12, 'active_t_nw']}), "
      f"5Y {_pct(hr.loc[20, 'mean_active_ann'])} (t={hr.loc[20, 'active_t_nw']}, thin). Total alpha keeps "
      "accruing over longer holds, but at a *declining rate* -- consistent with the rebalance finding that "
      "the discount mostly converges within ~1 year. The 5Y up-tick rests on 23 heavily-overlapping cohorts "
      "(NW lag 19) and should not be over-read. **Net implication:** harvest quarterly when AUM is small; "
      "let cadence lengthen toward annual only as size forces it.")


def main() -> None:
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    freq, cap, horizon = run()
    freq.to_csv(PORT_DIR / "rebalance_frequency.csv", index=False)
    horizon.to_csv(PORT_DIR / "holding_horizon.csv", index=False)
    (PORT_DIR / "rebalance_horizon.json").write_text(json.dumps(
        {"frequency": freq.to_dict(orient="records"),
         "capacity_by_freq": cap.to_dict(orient="records"),
         "holding_horizon": horizon.to_dict(orient="records")}, indent=2, default=str), encoding="utf-8")

    # benchmark CAGR over the realized quarterly index of the quarterly top-10 book (for the report header)
    ref = simulate_cadence(SimConfig(signal="discount", top_n=10), 1, load_price_panel())
    bench_cagr = {b: cagr(ref.bench[b]) for b in BENCHMARKS if b in ref.bench}

    rep = write_report(freq, cap, horizon, bench_cagr)
    print(freq.to_string(index=False))
    print("\ncapacity by frequency:")
    print(cap.to_string(index=False))
    print("\nholding horizon:")
    print(horizon.to_string(index=False))
    print(f"\nwrote rebalance_frequency.csv, holding_horizon.csv, rebalance_horizon.json, {rep.name}")


if __name__ == "__main__":
    main()
