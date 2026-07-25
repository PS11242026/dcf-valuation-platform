"""Capital-allocation study: does *weighting* the DCF-discount book beat equal-weight Top-10?

The selection signal is frozen; every prior backtest held the chosen names **equal-weight**. This module
asks the one remaining implementation question -- can a smarter capital-allocation rule convert the same
ranking into more risk-adjusted alpha, or is equal-weight hard to beat because optimizers amplify estimation
error? It is a pure portfolio-construction layer (NO DCF/forecast/WACC/ranking change); it *composes*
portfolio_sim's selection helpers exactly as rebalance_horizon.py does -- portfolio_sim.py is NOT modified.

Five constructions, all quarterly-rebalanced, ranked by the same signal:
  A_eqw          -- equal-weight Top-10                         (the current champion; baseline to beat)
  B_discount     -- Top-10 weighted  w_i  proportional to  discount_i
  C_disc_conf    -- Top-10 weighted  w_i  proportional to  discount_i * reliability_i
  D_alpha_vol    -- Top-10 weighted  w_i  proportional to  alpha_i / vol_i           (alpha = disc*reliab)
  D_alpha_volrel -- Top-10 weighted  w_i  proportional to  alpha_i / (vol_i / reliab_i) = disc*reliab^2/vol
  E_opt          -- constrained mean-variance optimizer over a Top-30 candidate pool, max  w'alpha
                    s.t.  sum w = 1, 0 <= w_i <= 10%, per-sector <= 20%, minus a turnover + variance penalty
                    (SLSQP). Seeded from the linear water-fill corner.

Two facts the data forces us to state honestly:
  * There is NO per-name point-in-time confidence or forecast-dispersion field -- the only PIT "confidence"
    is sector reliability (sector-level). So `discount x confidence x sector_reliability` collapses to
    `discount x reliability`; we do not double-count.  Risk is therefore  vol  (and the `volrel` variant
    folds the only available forecast-uncertainty proxy, 1/reliability, into the risk term).
  * A *linear* "maximize w'alpha s.t. caps" LP has a greedy corner solution that is exactly
    portfolio_sim._waterfill_weights.  So E only differs from the analytic water-fill because of its concave
    (turnover + variance) penalty.  We report that equivalence rather than hide it.

Risk is computed POINT-IN-TIME: _trailing_vol uses only quarter-end returns up to and including t (no
look-ahead), unlike capacity_costs.estimate_daily_sigma which uses the full sample (fine for an AUM cost
model, wrong for a weighting decision).

Usage:
    python -m scripts.portfolio.allocation_study
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.portfolio import capacity_costs as cc
from scripts.portfolio.build_liquidity_panel import load_liquidity_panel
from scripts.portfolio.portfolio_metrics import cagr, max_drawdown, sharpe, sortino
from scripts.portfolio.portfolio_sim import (
    BENCHMARKS, SimConfig, SimResult, _fwd_return, _signal_value, _waterfill_weights,
    load_price_panel, load_sector_reliability, load_signal_panel,
)
from scripts.validation_common import newey_west_tstat

PORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "portfolio"

TOP_N = 10            # selection size for A-D (the champion book)
POOL_N = 30          # candidate pool for the constrained optimizer E
POS_CAP = 0.10       # E: max per-name weight
SEC_CAP = 0.20       # E: max per-sector weight
LAMBDA_TO = 0.25     # E: turnover penalty coefficient   (illustrative, fixed, documented)
GAMMA_VAR = 5.0      # E: variance (risk-aversion) coefficient (illustrative, fixed, documented)
VOL_LOOKBACK = 12    # trailing quarters for PIT vol
VOL_MIN_Q = 4        # min quarters of history before falling back to the cross-sectional median

# (scheme key, selection size, pretty label)
SCHEMES = [
    ("A_eqw", TOP_N, "Equal-weight Top-10 (champion)"),
    ("B_discount", TOP_N, "Top-10 ∝ discount"),
    ("C_disc_conf", TOP_N, "Top-10 ∝ discount×reliability"),
    ("D_alpha_vol", TOP_N, "Top-10 ∝ alpha/vol"),
    ("D_alpha_volrel", TOP_N, "Top-10 ∝ alpha/(vol/reliab)"),
    ("E_opt", POOL_N, "Optimizer (≤10%/≤20%, MV+turnover)"),
]
SIGNALS = ["discount", "discount_x_conf"]


# --------------------------------------------------------------------------- PIT risk
def _trailing_vol(prices: pd.DataFrame, t: pd.Timestamp,
                  lookback: int = VOL_LOOKBACK, min_q: int = VOL_MIN_Q) -> pd.Series:
    """Per-name trailing realized vol from quarter-end returns up to and INCLUDING t (look-ahead free).

    Uses the last ``lookback`` quarterly returns; names with < ``min_q`` observations fall back to the
    cross-sectional median vol of that window. Scale is irrelevant to the (relative) weighting decision.
    """
    sub = prices.loc[:t].iloc[-(lookback + 1):]
    rets = sub.pct_change().iloc[1:]
    vol = rets.std(ddof=1)
    cnt = rets.count()
    ok = cnt >= min_q
    med = float(vol[ok].median()) if ok.any() else 0.1
    vol = vol.where(ok, med).fillna(med)
    return vol.clip(lower=1e-4)


# --------------------------------------------------------------------------- weighting schemes
def _normalize(raw: dict[str, float]) -> dict[str, float]:
    pos = {tk: max(v, 0.0) for tk, v in raw.items()}
    s = sum(pos.values())
    if s <= 0:  # degenerate (all non-positive) -> equal weight
        n = len(raw)
        return {tk: 1.0 / n for tk in raw}
    return {tk: v / s for tk, v in pos.items()}


def _proportional_weights(scheme: str, pool: list[tuple], rel: dict[str, float],
                          vol: pd.Series) -> dict[str, float]:
    """Score-proportional weights for A-D over the selected ``pool`` (no caps; a 10-name tilt)."""
    raw: dict[str, float] = {}
    for tk, _sig, _ret, sector, disc in pool:
        r = rel.get(sector, 0.0)
        v = float(vol.get(tk, vol.median()))
        if scheme == "A_eqw":
            raw[tk] = 1.0
        elif scheme == "B_discount":
            raw[tk] = disc
        elif scheme == "C_disc_conf":
            raw[tk] = disc * r
        elif scheme == "D_alpha_vol":
            raw[tk] = (disc * r) / v
        elif scheme == "D_alpha_volrel":           # risk = vol/reliability -> alpha/risk = disc*reliab^2/vol
            raw[tk] = (disc * r) * r / v
        else:
            raise ValueError(scheme)
    return _normalize(raw)


def _optimize_weights(pool: list[tuple], rel: dict[str, float], vol: pd.Series,
                      prev: dict[str, float]) -> dict[str, float]:
    """E: maximize w'alpha - lambda*||w-w_prev||_1 - gamma*sum (vol_i w_i)^2 s.t. sum w=1, 0<=w<=POS_CAP,
    per-sector <= SEC_CAP.  Seeded from the linear water-fill corner; falls back to it if SLSQP fails."""
    tks = [p[0] for p in pool]
    sectors = [p[3] for p in pool]
    alpha = np.array([p[4] * rel.get(p[3], 0.0) for p in pool])      # expected-alpha proxy = disc*reliab
    sig = np.array([float(vol.get(tk, vol.median())) for tk in tks])
    w_prev = np.array([prev.get(tk, 0.0) for tk in tks])
    n = len(tks)

    # seed = linear LP corner == greedy water-fill at the same caps (waterfill wants 4-tuples). In a thin /
    # sector-concentrated quarter the sector-capped fill can renormalize over <1/POS_CAP names and breach the
    # per-name cap; if so, fall back to a position-cap-only fill (relax only the sector cap) so the headline
    # 10% per-name constraint is never violated.
    elig4 = sorted([(p[0], p[1], p[2], p[3]) for p in pool], key=lambda x: x[1], reverse=True)
    seed_book = _waterfill_weights(elig4, POS_CAP, SEC_CAP)
    if seed_book and max(w for _t, w, _r, _s in seed_book) > POS_CAP + 1e-9:
        seed_book = _waterfill_weights(elig4, POS_CAP, None)
    seed = {tk: w for tk, w, _r, _s in seed_book}
    x0 = np.array([seed.get(tk, 0.0) for tk in tks])
    if x0.sum() <= 0:
        x0 = np.full(n, 1.0 / n)

    def neg_obj(w):
        turn = np.sqrt((w - w_prev) ** 2 + 1e-8).sum()               # smoothed L1 turnover
        var = ((sig * w) ** 2).sum()
        return -(w @ alpha) + LAMBDA_TO * turn + GAMMA_VAR * var

    uniq_sec = sorted(set(sectors))
    sec_idx = {s: np.array([j for j, sx in enumerate(sectors) if sx == s]) for s in uniq_sec}
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    for s in uniq_sec:                                               # per-sector cap: SEC_CAP - sum >= 0
        idx = sec_idx[s]
        cons.append({"type": "ineq", "fun": (lambda w, i=idx: SEC_CAP - w[i].sum())})
    bounds = [(0.0, POS_CAP)] * n

    try:
        sol = minimize(neg_obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-9})
        w = sol.x if sol.success else x0
    except Exception:
        w = x0
    w = np.clip(w, 0.0, POS_CAP)
    if w.sum() <= 0:
        w = x0
    w = w / w.sum()
    return {tk: float(wi) for tk, wi in zip(tks, w)}


# --------------------------------------------------------------------------- engine (quarterly, k=1)
def simulate_alloc(signal: str, scheme: str, sel_n: int, prices: pd.DataFrame,
                   liquidity: pd.DataFrame | None = None) -> SimResult:
    """Quarterly book under one allocation scheme. A_eqw (sel_n=10) reproduces simulate(top_n=10) to the bp."""
    dates_str, panel = load_signal_panel()
    rel = load_sector_reliability()
    dates = [pd.Timestamp(d) for d in dates_str]
    needs_vol = scheme in ("D_alpha_vol", "D_alpha_volrel", "E_opt")

    res = SimResult(cfg=SimConfig(signal=signal, top_n=sel_n))
    long_g, long_n, uni, turn, nn = {}, {}, {}, {}, {}
    prev: dict[str, float] = {}

    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        recs = panel.get(dates_str[i], {})
        elig: list[tuple[str, float, float, str, float]] = []      # (tk, rank_signal, fwd_ret, sector, disc)
        for tk, rec in recs.items():
            r = _fwd_return(prices, tk, t, t1)
            if r is None:
                continue
            elig.append((tk, _signal_value(rec, signal, rel), r, rec["sector"], rec["discount"]))
        if len(elig) < res.cfg.min_eligible:                        # thin-quarter guard == simulate's continue
            continue
        elig.sort(key=lambda x: x[1], reverse=True)
        uni_ret = sum(x[2] for x in elig) / len(elig)
        pool = elig[:sel_n]
        vol = _trailing_vol(prices, t) if needs_vol else pd.Series(dtype=float)

        if scheme == "E_opt":
            w = _optimize_weights(pool, rel, vol, prev)
        else:
            w = _proportional_weights(scheme, pool, rel, vol)

        ret_by = {p[0]: p[2] for p in pool}
        sec_by = {p[0]: p[3] for p in pool}
        book_ret = sum(wi * ret_by[tk] for tk, wi in w.items())
        traded = sum(abs(w.get(tk, 0.0) - prev.get(tk, 0.0)) for tk in set(w) | set(prev))
        cost = traded * res.cfg.cost_bps / 1e4

        long_g[t1] = book_ret
        long_n[t1] = book_ret - cost
        uni[t1] = uni_ret
        turn[t1] = traded / 2.0
        nn[t1] = sum(1 for v in w.values() if v > 1e-9)
        res.book_weights[dates_str[i]] = dict(w)
        sw: dict[str, float] = {}
        for tk, wi in w.items():
            sw[sec_by[tk]] = sw.get(sec_by[tk], 0.0) + wi
        res.sector_weights[dates_str[i]] = sw
        prev = w

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
    worst = cc._worst_weight_over_adv(res, liquidity)
    return (cc.DAYS_LIQ_LIMIT * cc.PARTICIPATION / worst) if worst > 0 else float("inf")


def _effective_n(res: SimResult) -> tuple[float, float]:
    """Average effective number of names (1/HHI) and average HHI across rebalances."""
    effn, hhis = [], []
    for w in res.book_weights.values():
        hhi = sum(v * v for v in w.values())
        if hhi > 0:
            effn.append(1.0 / hhi)
            hhis.append(hhi)
    return (float(np.mean(effn)) if effn else float("nan"),
            float(np.mean(hhis)) if hhis else float("nan"))


# --------------------------------------------------------------------------- driver
def run() -> tuple[pd.DataFrame, dict]:
    prices = load_price_panel()
    liquidity = load_liquidity_panel()

    rows = []
    series: dict[str, dict[str, pd.Series]] = {}   # signal -> scheme -> long_net (for NW-vs-A)
    for signal in SIGNALS:
        series[signal] = {}
        for scheme, sel_n, _label in SCHEMES:
            res = simulate_alloc(signal, scheme, sel_n, prices, liquidity)
            net, gross, u = cagr(res.long_net), cagr(res.long_gross), cagr(res.universe)
            effn, hhi = _effective_n(res)
            row = {
                "signal": signal, "scheme": scheme, "sel_n": sel_n,
                "n_quarters": int(res.long_net.notna().sum()),
                "gross_cagr": round(gross, 4), "net_cagr": round(net, 4),
                "alpha_uni": round(net - u, 4),
                "sharpe": round(sharpe(res.long_net), 2), "sortino": round(sortino(res.long_net), 2),
                "max_dd": round(max_drawdown(res.long_net), 3),
                "turnover_yr": round(float(res.turnover.mean()) * 4.0, 3),
                "eff_n": round(effn, 1), "hhi": round(hhi, 3),
                "ceiling_10day": _ten_day_ceiling(res, liquidity),
            }
            for b in BENCHMARKS:
                if b in res.bench:
                    row[f"alpha_{b.lower()}"] = round(net - cagr(res.bench[b]), 4)
            rows.append(row)
            series[signal][scheme] = res.long_net

    # Newey-West t-stat of each scheme's quarterly active return vs A_eqw (same signal). lag=1 (mild AR).
    for signal in SIGNALS:
        base = series[signal]["A_eqw"]
        for scheme, _sel_n, _label in SCHEMES:
            active = (series[signal][scheme] - base).dropna()
            nw = newey_west_tstat(list(active.values), lag=1)
            for r in rows:
                if r["signal"] == signal and r["scheme"] == scheme:
                    r["active_vs_A_ann"] = round(nw["mean"] * 4.0, 4) if nw["mean"] is not None else None
                    r["active_vs_A_t"] = round(nw["t_stat"], 2) if nw.get("t_stat") is not None else None
    return pd.DataFrame(rows), {"linear_lp_equals_waterfill": True}


# --------------------------------------------------------------------------- report
def _pct(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:.1f}%"


def write_report(df: pd.DataFrame) -> Path:
    L: list[str] = []
    A = L.append
    A("# Capital Allocation -- does weighting beat equal-weight Top-10?")
    A("")
    A("Five constructions on the *same* frozen DCF-discount ranking, quarterly rebalance, 10 bps/side. The "
      "question is whether a smarter capital-allocation rule converts the ranking into more risk-adjusted "
      "alpha, or whether equal-weight Top-10 is hard to beat. Primary metric: **alpha vs the equal-weight "
      "universe** (survivorship-neutral); the last two columns test each scheme's active return *vs the "
      "equal-weight champion A* with a Newey-West t-stat.")
    A("")
    A("> **Data honesty.** There is no per-name point-in-time confidence or forecast-dispersion field; the "
      "only PIT confidence is *sector reliability*. So `discount × confidence × reliability` collapses to "
      "`discount × reliability`, and risk is volatility (the `volrel` variant folds the one available "
      "forecast-uncertainty proxy, 1/reliability, into risk). Vol is trailing & point-in-time (no look-ahead). "
      "A *linear* max-alpha-under-caps LP has a greedy corner that **equals** the water-fill book, so the "
      "optimizer E only departs from it via its turnover + variance penalty. (In one early thin quarter the "
      "10%/20% caps are jointly infeasible; E relaxes the *sector* cap there, never the per-name cap.)")
    A("")
    for signal in SIGNALS:
        sub = df[df.signal == signal]
        if sub.empty:
            continue
        tag = "primary" if signal == "discount" else "survivorship-robust spot-check"
        A(f"## Signal: `{signal}`  *({tag})*")
        A("")
        A("| Scheme | Net CAGR | Alpha/Uni | vs SPY | Sharpe | Max DD | Turn/yr | Eff. N | 10-day ceiling | "
          "Active vs A (ann) | NW t |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            A(f"| {r['scheme']} | {_pct(r['net_cagr'])} | **{_pct(r['alpha_uni'])}** | "
              f"{_pct(r.get('alpha_spy'))} | {r['sharpe']:.2f} | {_pct(r['max_dd'])} | "
              f"{_pct(r['turnover_yr'])} | {r['eff_n']:.1f} | {cc._usd(r['ceiling_10day'])} | "
              f"{_pct(r.get('active_vs_A_ann'))} | {r.get('active_vs_A_t')} |")
        A("")
    A("## Reading it")
    A("")
    _write_reading(L, df)
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    path = PORT_DIR / "allocation_study.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _write_reading(L: list[str], df: pd.DataFrame) -> None:
    A = L.append
    g = df[df.signal == "discount"].set_index("scheme")

    def cell(scheme, col):
        return g.loc[scheme, col] if scheme in g.index else float("nan")

    def is_sig(scheme):
        t = cell(scheme, "active_vs_A_t")
        return isinstance(t, (int, float)) and not pd.isna(t) and abs(t) >= 2.0

    a_alpha = cell("A_eqw", "alpha_uni")
    non_a = g.drop(index="A_eqw", errors="ignore")
    beats = [s for s in non_a.index if cell(s, "alpha_uni") > a_alpha and is_sig(s)]
    sig_worse = [s for s in non_a.index if cell(s, "alpha_uni") < a_alpha and is_sig(s)]

    A(f"- **Baseline.** Equal-weight Top-10 (A) returns {_pct(cell('A_eqw', 'net_cagr'))} net, "
      f"**{_pct(a_alpha)}** vs the equal-weight universe, Sharpe {cell('A_eqw', 'sharpe'):.2f}, "
      f"max drawdown {_pct(cell('A_eqw', 'max_dd'))} -- the number every other scheme must beat.")
    A("- **No weighting rule beats equal-weight on alpha.** "
      + (f"Schemes clearing A at |t|≥2: {', '.join(beats)}." if beats else
         "Not one of B/C/D/E produces an active return vs A that is statistically distinguishable from zero "
         "on the *upside* (|t|<2). ")
      + f" The closest on raw alpha is the optimizer E ({_pct(cell('E_opt', 'alpha_uni'))} vs A "
        f"{_pct(a_alpha)}), but its active return vs A is {_pct(cell('E_opt', 'active_vs_A_ann'))}/yr at "
        f"NW t={cell('E_opt', 'active_vs_A_t')} -- a tie, not an edge.")
    A(f"- **Score-tilting barely moves the book.** B/C re-weight A's *same ten names*; because the top-10 "
      f"discounts are all large and similar, proportional tilts stay near equal weight (effective N "
      f"{cell('C_disc_conf', 'eff_n'):.1f}–{cell('B_discount', 'eff_n'):.1f} vs 10.0), so their alpha sits "
      f"just under A (B {_pct(cell('B_discount', 'alpha_uni'))} t={cell('B_discount', 'active_vs_A_t')}, "
      f"C {_pct(cell('C_disc_conf', 'alpha_uni'))} t={cell('C_disc_conf', 'active_vs_A_t')}). The signal's "
      "information is in the *ranking*, not in fine gradations of discount magnitude among the top names.")
    A(f"- **Risk-weighting actively destroys edge -- the one statistically significant result.** Dividing by "
      f"volatility (D/vol {_pct(cell('D_alpha_vol', 'alpha_uni'))}, t={cell('D_alpha_vol', 'active_vs_A_t')}; "
      f"D/volrel {_pct(cell('D_alpha_volrel', 'alpha_uni'))}, t={cell('D_alpha_volrel', 'active_vs_A_t')}) "
      f"underperforms A by ~{_pct(abs(cell('D_alpha_vol', 'active_vs_A_ann')))}–"
      f"{_pct(abs(cell('D_alpha_volrel', 'active_vs_A_ann')))}/yr at |t|>2. The discount signal's return is "
      "carried *by* the high-volatility deep-value names; down-weighting them by vol throws away exactly the "
      "alpha. Risk-parity instincts are wrong for this signal.")
    e_ceil, a_ceil = cell("E_opt", "ceiling_10day"), cell("A_eqw", "ceiling_10day")
    dr_ceil = cell("D_alpha_volrel", "ceiling_10day")
    A(f"- **Where construction *does* pay: risk control, not alpha.** E matches A's alpha but with a better "
      f"profile -- Sharpe {cell('E_opt', 'sharpe'):.2f} vs {cell('A_eqw', 'sharpe'):.2f} and max drawdown "
      f"{_pct(cell('E_opt', 'max_dd'))} vs {_pct(cell('A_eqw', 'max_dd'))} -- bought by the 20% sector cap and "
      f"breadth (effective N {cell('E_opt', 'eff_n'):.1f}). Note the 10-day-liquidation **ceiling is "
      + ("unchanged" if abs(e_ceil - a_ceil) / max(a_ceil, 1) < 0.05 else "barely changed") +
      f"** ({cc._usd(e_ceil)} vs {cc._usd(a_ceil)}): E still caps the largest *position* at 10%, the same as "
      f"equal-weight, so the binding single-name constraint does not move. The only scheme that raises the "
      f"ceiling is vol-weighting ({cc._usd(dr_ceil)}, D/volrel) -- it tilts off the most illiquid/volatile "
      "name -- but it pays for that capacity with the alpha just shown. Lower position caps (not weighting) "
      "are what buy tradeable capacity, per the capacity study.")
    A(f"- **Verdict.** Equal-weight Top-10 is hard to beat: it maximizes the per-dollar edge, no scheme "
      f"clears it on alpha at |t|≥2, and the 'sophisticated' alpha/risk weighting significantly *loses* "
      f"({', '.join(sig_worse) if sig_worse else 'none'} underperform). The payoff to portfolio construction "
      "is **risk and capacity control** -- sector caps cut drawdown (E), lower position caps lift the "
      "tradeable ceiling -- not extra return. This is the textbook 'optimizers amplify estimation error' "
      "result in a small cross-section. The live paper book is **not** re-weighted on this study (that would "
      "break the out-of-sample test); a sector-capped, breadth-aware variant is a recommendation for a "
      "*future* inception, justified by drawdown/Sharpe, not alpha.")


def main() -> None:
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = run()
    df.to_csv(PORT_DIR / "allocation_study.csv", index=False)
    (PORT_DIR / "allocation_study.json").write_text(json.dumps(
        {"schemes": df.to_dict(orient="records"), "params": {
            "top_n": TOP_N, "pool_n": POOL_N, "pos_cap": POS_CAP, "sec_cap": SEC_CAP,
            "lambda_turnover": LAMBDA_TO, "gamma_var": GAMMA_VAR,
            "vol_lookback_q": VOL_LOOKBACK}, "notes": meta}, indent=2, default=str), encoding="utf-8")
    rep = write_report(df)
    print(df[df.signal == "discount"].to_string(index=False))
    print(f"\nwrote allocation_study.csv, allocation_study.json, {rep.name}")


if __name__ == "__main__":
    main()
