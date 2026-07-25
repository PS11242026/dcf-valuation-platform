"""Quarterly portfolio simulator over the frozen DCF discount signal.

Reuses:
  - reports/final_universe_backtest.json  -> point-in-time discount signal panel (the SIGNAL)
  - reports/portfolio/_prices.csv          -> quarter-end total-return price panel (build_price_panel)
  - scripts.scorecard._sector_reliability  -> evidence-based sector confidence (0-1) for the
                                              `discount_x_conf` signal variant

No DCF / forecast / WACC / ranking changes -- this is a portfolio layer on top of a frozen model.

Engine (per quarter t, hold one quarter t -> t+1):
  eligible = names with a (|discount|<=5 filtered) signal at t AND a price at both t and t+1
  rank by signal (desc); LONG = top-N equal weight, SHORT-leg diagnostic = bottom-N
  portfolio quarterly return = mean forward return of held names; compound NAV
  transaction cost charged on traded notional (sum|dw| * cost_bps)

Look-ahead guard: the signal at quarter-end t (fundamentals public by then) selects names whose
returns are measured strictly forward t -> t+1. An optional +1-month entry lag is supported.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for `import scorecard`
from scorecard import _sector_reliability  # noqa: E402

from scripts.portfolio.build_price_panel import load_price_panel  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"
PORT_DIR = REPORT_DIR / "portfolio"
DISCOUNT_CLIP = 5.0  # |discount| > 5 = ADR/currency blow-ups (519 obs); drop from selection
BENCHMARKS = ["SPY", "QQQ"]


def load_sector_reliability() -> dict[str, float]:
    inst = json.load(open(REPORT_DIR / "final_institutional_validation.json")).get("by_sector_family", {})
    return {f: _sector_reliability(b)["score"] for f, b in inst.items()}


def load_signal_panel() -> tuple[list[str], dict[str, dict[str, dict]]]:
    """Return (sorted dates, {date: {ticker: {"discount":x, "sector":s}}}) with blow-ups filtered."""
    obs = json.load(open(REPORT_DIR / "final_universe_backtest.json"))["observations"]
    panel: dict[str, dict[str, dict]] = {}
    for o in obs:
        d = o.get("discount_to_intrinsic")
        if not isinstance(d, (int, float)) or abs(d) > DISCOUNT_CLIP:
            continue
        panel.setdefault(o["date"], {})[o["ticker"]] = {"discount": d, "sector": o["sector"]}
    return sorted(panel), panel


@dataclass
class SimConfig:
    signal: str = "discount"          # "discount" | "discount_x_conf"
    top_n: int = 20
    cost_bps: float = 10.0            # per side, on traded notional
    entry_lag_months: int = 0         # 0 = enter at quarter-end t; 1 = +1 month
    min_eligible: int = 40            # skip a rebalance if cross-section is too thin
    # --- risk controls (None = off; baseline equal-weight = position_cap=1/top_n, sector_cap=None) ---
    position_cap: float | None = None       # max weight per name (e.g. 0.05); None => 1/top_n (equal wt)
    sector_cap: float | None = None         # max weight per sector (e.g. 0.25); None => uncapped
    liquidity_min_dollar_vol: float | None = None  # drop names below this median $-volume at t


@dataclass
class SimResult:
    cfg: SimConfig
    # quarterly returns indexed by the END date of each holding period (t+1)
    long_gross: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    long_net: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    short_leg: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    long_short: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    universe: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bench: dict[str, pd.Series] = field(default_factory=dict)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    holdings: dict[str, list[str]] = field(default_factory=dict)
    sector_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    n_names: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    max_position: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # per-quarter Brinson inputs for sector_attribution.py: {date, W, R, U, RU} (sector dicts)
    attribution: list[dict] = field(default_factory=list)
    # per-quarter per-name book weights {date: {ticker: weight}} -- consumed by capacity_costs.py to
    # reprice traded notional under an AUM-dependent cost model (additive; no effect on returns).
    book_weights: dict[str, dict[str, float]] = field(default_factory=dict)


def _waterfill_weights(elig_sorted: list[tuple], position_cap: float | None,
                       sector_cap: float | None) -> list[tuple[str, float, float, str]]:
    """Assign fractional weights down the signal ranking, capping per-name and per-sector.

    Each name i gets w_i = min(position_cap, sector_cap - sector_weight_so_far, 1 - total). Walking the
    ranked list and stopping at sum=1 reproduces top-N equal weight when position_cap=1/N & sector_cap=None,
    and naturally expands the book + redistributes weight when caps bind. elig_sorted items are
    (ticker, signal, fwd_ret, sector), already sorted by signal desc.
    """
    pc = position_cap if position_cap else 1.0
    out: list[tuple[str, float, float, str]] = []
    sec_w: dict[str, float] = {}
    total = 0.0
    for tk, _sig, ret, sector in elig_sorted:
        if total >= 1.0 - 1e-9:
            break
        head_sec = (sector_cap - sec_w.get(sector, 0.0)) if sector_cap is not None else 1.0
        w = min(pc, head_sec, 1.0 - total)
        if w <= 1e-9:
            continue  # this name's sector is full -> skip, take the next sector's name
        out.append((tk, w, ret, sector))
        sec_w[sector] = sec_w.get(sector, 0.0) + w
        total += w
    if 0.0 < total < 1.0 - 1e-6:  # ran out of names/sectors -> renormalize what we hold
        out = [(tk, w / total, ret, s) for tk, w, ret, s in out]
    return out


def _signal_value(rec: dict, sig: str, rel: dict[str, float]) -> float:
    d = rec["discount"]
    if sig == "discount":
        return d
    if sig == "discount_x_conf":
        return d * rel.get(rec["sector"], 0.0)
    raise ValueError(f"unknown signal {sig}")


def _fwd_return(prices: pd.DataFrame, ticker: str, t: pd.Timestamp, t1: pd.Timestamp) -> float | None:
    if ticker not in prices.columns:
        return None
    p0 = prices.at[t, ticker] if t in prices.index else None
    p1 = prices.at[t1, ticker] if t1 in prices.index else None
    if p0 is None or p1 is None or pd.isna(p0) or pd.isna(p1) or p0 <= 0:
        return None
    return float(p1 / p0 - 1.0)


def selection_paths(cfg: SimConfig, prices: pd.DataFrame | None = None) -> list[dict]:
    """Per-quarter name-level selection for stress testing.

    Returns one record per holding period: {date, top:[(ticker, fwd_ret)], bottom:[(ticker, fwd_ret)],
    universe_ret, bench:{SPY,QQQ}}. Same selection logic as ``simulate`` but exposes name-level returns
    so a delisting shock can be injected position-by-position.
    """
    if prices is None:
        prices = load_price_panel()
    dates_str, panel = load_signal_panel()
    rel = load_sector_reliability()
    dates = [pd.Timestamp(d) for d in dates_str]
    out: list[dict] = []
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        recs = panel.get(dates_str[i], {})
        elig = []
        for tk, rec in recs.items():
            r = _fwd_return(prices, tk, t, t1)
            if r is None:
                continue
            elig.append((tk, _signal_value(rec, cfg.signal, rel), r))
        if len(elig) < cfg.min_eligible:
            continue
        elig.sort(key=lambda x: x[1], reverse=True)
        n = min(cfg.top_n, len(elig) // 2)
        bench = {b: _fwd_return(prices, b, t, t1) for b in BENCHMARKS if b in prices.columns}
        out.append({
            "date": t1,
            "top": [(x[0], x[2]) for x in elig[:n]],
            "bottom": [(x[0], x[2]) for x in elig[-n:]],
            "universe_ret": sum(x[2] for x in elig) / len(elig),
            "bench": bench,
        })
    return out


def simulate(cfg: SimConfig, prices: pd.DataFrame | None = None,
             liquidity: pd.DataFrame | None = None) -> SimResult:
    if prices is None:
        prices = load_price_panel()
    dates_str, panel = load_signal_panel()
    rel = load_sector_reliability()
    dates = [pd.Timestamp(d) for d in dates_str]
    position_cap = cfg.position_cap if cfg.position_cap else 1.0 / cfg.top_n

    res = SimResult(cfg=cfg)
    prev_w: dict[str, float] = {}
    long_g, long_n, short_l, ls, uni, turn, nn, mp = {}, {}, {}, {}, {}, {}, {}, {}

    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        recs = panel.get(dates_str[i], {})
        # eligible: has signal at t and a realizable forward return t->t+1
        elig: list[tuple[str, float, float, str]] = []  # (ticker, signal, fwd_ret, sector)
        for tk, rec in recs.items():
            r = _fwd_return(prices, tk, t, t1)
            if r is None:
                continue
            # liquidity screen: drop names below the dollar-volume floor at t
            if cfg.liquidity_min_dollar_vol and liquidity is not None and tk in liquidity.columns:
                dv = liquidity.at[t, tk] if t in liquidity.index else None
                if dv is None or pd.isna(dv) or dv < cfg.liquidity_min_dollar_vol:
                    continue
            elig.append((tk, _signal_value(rec, cfg.signal, rel), r, rec["sector"]))
        if len(elig) < cfg.min_eligible:
            continue
        elig.sort(key=lambda x: x[1], reverse=True)

        # LONG book via water-filling (position + sector caps); equal weight is the no-cap special case.
        book = _waterfill_weights(elig, position_cap, cfg.sector_cap)
        long_ret = sum(w * ret for _tk, w, ret, _s in book)

        # SHORT leg (diagnostic): bottom-N equal weight, unconstrained, consistent across portfolios.
        n_short = min(cfg.top_n, len(elig) // 2)
        bot = elig[-n_short:]
        short_ret = sum(x[2] for x in bot) / n_short
        uni_ret = sum(x[2] for x in elig) / len(elig)

        # turnover vs previous book weights: sum|dw| over the union of holdings.
        w_cur = {tk: w for tk, w, _r, _s in book}
        traded = sum(abs(w_cur.get(tk, 0.0) - prev_w.get(tk, 0.0))
                     for tk in set(w_cur) | set(prev_w))
        cost = traded * cfg.cost_bps / 1e4

        long_g[t1] = long_ret
        long_n[t1] = long_ret - cost
        short_l[t1] = short_ret
        ls[t1] = long_ret - short_ret
        uni[t1] = uni_ret
        turn[t1] = traded / 2.0  # one-sided turnover
        nn[t1] = len(book)
        mp[t1] = max((w for _t, w, _r, _s in book), default=0.0)

        res.holdings[dates_str[i]] = [tk for tk, _w, _r, _s in book]
        res.book_weights[dates_str[i]] = dict(w_cur)
        sw: dict[str, float] = {}
        for _tk, w, _r, s in book:
            sw[s] = sw.get(s, 0.0) + w
        res.sector_weights[dates_str[i]] = sw

        # Brinson inputs: book weight/return and universe weight/return per sector.
        W, R, U, RU = {}, {}, {}, {}
        for _tk, w, ret, s in book:
            W[s] = W.get(s, 0.0) + w
            R[s] = R.get(s, 0.0) + w * ret
        for s in list(R):
            R[s] /= W[s]
        nu = len(elig)
        for _tk, _sig, ret, s in elig:
            U[s] = U.get(s, 0.0) + 1.0 / nu
            RU[s] = RU.get(s, 0.0) + ret / nu
        for s in list(RU):  # convert summed ret/nu into a per-sector mean
            RU[s] = RU[s] / U[s] if U[s] else 0.0
        res.attribution.append({"date": dates_str[i], "W": W, "R": R, "U": U, "RU": RU})
        prev_w = w_cur

    idx = pd.to_datetime(list(long_g))
    res.long_gross = pd.Series(long_g).reindex(idx)
    res.long_net = pd.Series(long_n).reindex(idx)
    res.short_leg = pd.Series(short_l).reindex(idx)
    res.long_short = pd.Series(ls).reindex(idx)
    res.universe = pd.Series(uni).reindex(idx)
    res.turnover = pd.Series(turn).reindex(idx)
    res.n_names = pd.Series(nn).reindex(idx)
    res.max_position = pd.Series(mp).reindex(idx)

    for b in BENCHMARKS:
        if b in prices.columns:
            br = {}
            for i in range(len(dates) - 1):
                t, t1 = dates[i], dates[i + 1]
                if t1 in res.long_net.index:
                    r = _fwd_return(prices, b, t, t1)
                    if r is not None:
                        br[t1] = r
            res.bench[b] = pd.Series(br).reindex(res.long_net.index)
    return res


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Quarterly DCF-discount portfolio simulator")
    ap.add_argument("--signal", choices=["discount", "discount_x_conf"], default="discount")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--entry-lag-months", type=int, default=0)
    args = ap.parse_args()
    cfg = SimConfig(signal=args.signal, top_n=args.top_n, cost_bps=args.cost_bps,
                    entry_lag_months=args.entry_lag_months)
    res = simulate(cfg)
    nav = (1 + res.long_net).cumprod()
    print(f"signal={cfg.signal} top_n={cfg.top_n} cost_bps={cfg.cost_bps}")
    print(f"periods={len(res.long_net)} final NAV(net)={nav.iloc[-1]:.2f}")
    print(res.long_net.to_string())


if __name__ == "__main__":
    main()
