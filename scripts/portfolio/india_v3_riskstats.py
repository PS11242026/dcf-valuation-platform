"""India v3 book -- portfolio risk / return statistics (the network step for the export).

Computes the portfolio-management stats the export needs from REAL market data (3y weekly total
returns for the five holdings + NIFTY 50), then writes ``risk_stats.json`` into the book folder so
the export stays deterministic / offline. If this file is absent the export falls back to documented
constants, so it is optional but recommended (re-run after any re-valuation).

What it computes (all from data unless noted):
  * per-name annualised vol, beta vs NIFTY, correlation to NIFTY, trailing-12m dividend yield
  * holding-to-holding correlation matrix and average pairwise correlation
  * portfolio (market-value-weighted) beta, vol, tracking error, dividend yield
  * diversification ratio = (sum w_i sigma_i) / sigma_portfolio  and effective independent bets = DR^2
  * weight concentration: HHI and effective number of names = 1 / HHI
  * realised 3y CAPM alpha (regression intercept, annualised) and 3y max drawdown -- BACKWARD-looking
  * EX-ANTE expectations over the ~1y forecasting horizon: E[Rm], expected alpha (signal-derived with an
    explicit survivorship haircut), E[Rp] via CAPM, expected Sharpe, information ratio

Authoring note: prose avoids en/em dashes per the book owner's request.

Run:  python scripts/portfolio/india_v3_riskstats.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from dcf_project.paths import MANAGED

BOOK_NAME = "india_v3_aggressive"
FOLDER = MANAGED / BOOK_NAME
BENCHMARK = "^NSEI"
WEEKS = 52.0

# --- ex-ante assumptions (explicit + documented; consistent with the assumption table) -------------
RF_INDIA = 0.068          # India 10y G-Sec (matches the export assumption table)
ERP_MARKET = 0.060        # NIFTY equity premium for E[Rm] (the MARKET premium; distinct from the ~8%
                          # Damodaran total ERP used inside the WACC, which adds country/credit risk)
ENERGY_1Y_IC = 0.476      # India Phase B, per-sector
OVERALL_1Y_LS = 0.289     # India Phase B, 1y quintile long-short
SURVIVORSHIP_HAIRCUT = 0.60   # Phase B is survivorship-optimistic + overlapping + inflated t-stats


def _weekly_close(symbol: str, years: float = 3.0):
    """Weekly total-return CLOSE series (auto_adjust folds in splits + dividends), indexed by date so
    holdings and the benchmark can be aligned on common weeks (position alignment silently misaligns
    series of different length and collapses beta). Returns a pandas Series or None on failure."""
    import yfinance as yf

    raw = yf.download(symbol, period=f"{int(years * 12)}mo", interval="1wk",
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return None
    close = raw["Close"].dropna()
    if hasattr(close, "columns"):           # multiindex -> single column
        close = close.iloc[:, 0]
    close.name = symbol
    return close if len(close) >= 30 else None


LOGO_BASE = ("https://raw.githubusercontent.com/dharunashokkumar/"
             "indian-listed-company-logos/main/nse/NSE_{ticker}.svg")


def _fetch_logo(ticker: str, out_dir: Path) -> bool:
    """Download the NSE company logo (SVG) and rasterise it to figures/logos/<TICKER>.png via PyMuPDF
    (no cairo dependency). Best-effort: returns True on success, False if unavailable. The dashboard
    uses the PNG if present and falls back to a branded badge otherwise."""
    out = out_dir / f"{ticker}.png"
    if out.exists():
        return True
    import urllib.request
    url = LOGO_BASE.format(ticker=ticker)
    try:
        svg = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read()
        import fitz  # PyMuPDF
        doc = fitz.open(stream=svg, filetype="svg")
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(16, 16), alpha=True)   # high-res, transparent
        out_dir.mkdir(parents=True, exist_ok=True)
        pix.save(str(out))
        return True
    except Exception:  # noqa: BLE001
        return False


def _analyst(symbol: str) -> dict:
    """Street consensus from Yahoo: mean/high/low price target, analyst count, recommendation. Each
    field is best-effort (PSU coverage is good but the feed is flaky), so missing values stay None."""
    import yfinance as yf

    info = {}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:  # noqa: BLE001
        info = {}
    return {
        "mean_target": info.get("targetMeanPrice"),
        "high_target": info.get("targetHighPrice"),
        "low_target": info.get("targetLowPrice"),
        "n_analysts": info.get("numberOfAnalystOpinions"),
        "reco_key": info.get("recommendationKey"),
        "reco_mean": info.get("recommendationMean"),
    }


def _trailing_div_yield(symbol: str, price: float) -> float | None:
    import yfinance as yf

    try:
        divs = yf.Ticker(symbol).dividends
    except Exception:  # noqa: BLE001
        return None
    if divs is None or len(divs) == 0 or not price:
        return None
    cutoff = (dt.datetime.now(divs.index.tz) if divs.index.tz is not None
              else dt.datetime.now()) - dt.timedelta(days=365)
    last12 = divs[divs.index >= cutoff]
    if len(last12) == 0:
        return None
    return float(last12.sum()) / float(price)


def main() -> None:
    import pandas as pd

    state = json.loads((FOLDER / "state.json").read_text(encoding="utf-8"))
    holdings = state["holdings"]
    deployed = state.get("deployed") or sum(h["market_value"] or 0 for h in holdings)
    weights = {h["ticker"]: (h["market_value"] or 0) / deployed for h in holdings}

    print(f"fetching 3y weekly total returns for {len(holdings)} holdings + NIFTY ...")
    bench_close = _weekly_close(BENCHMARK)
    if bench_close is None:
        raise SystemExit("could not fetch benchmark (^NSEI) returns; aborting.")

    logo_dir = FOLDER / "figures" / "logos"
    closes: dict[str, "pd.Series"] = {"^NSEI": bench_close}
    div_yield: dict[str, float | None] = {}
    analyst: dict[str, dict] = {}
    logos_ok = 0
    for h in holdings:
        sym, tk = h["symbol"], h["ticker"]
        c = _weekly_close(sym)
        if c is None:
            print(f"  {tk:11s} skip (no data)")
            continue
        closes[tk] = c
        dy = _trailing_div_yield(sym, h["entry_price"])
        div_yield[tk] = dy
        analyst[tk] = _analyst(sym)
        got_logo = _fetch_logo(tk, logo_dir)
        logos_ok += int(got_logo)
        tgt = analyst[tk].get("mean_target")
        print(f"  {tk:11s} {len(c)} wk  divyld={(f'{dy:.1%}' if dy is not None else 'NA')}"
              f"  street={analyst[tk].get('reco_key') or 'NA'}"
              f"  target={(f'Rs {tgt:,.0f}' if tgt else 'NA')}  logo={'ok' if got_logo else 'NA'}")
    print(f"logos fetched: {logos_ok}/{len(holdings)} -> {logo_dir}")

    # Align on COMMON weeks by date (inner join), then take weekly simple returns.
    px = pd.concat(closes.values(), axis=1, join="inner")
    px.columns = list(closes.keys())
    ret_df = px.pct_change().dropna()
    bench = ret_df["^NSEI"].to_numpy(dtype=float)
    rets = {tk: ret_df[tk].to_numpy(dtype=float) for tk in closes if tk != "^NSEI"}
    n = len(bench)
    print(f"  aligned to {n} common weekly observations")
    var_m = float(np.var(bench, ddof=1))
    mean_m = float(np.mean(bench))
    bench_vol = float(np.std(bench, ddof=1) * np.sqrt(WEEKS))

    per_name = {}
    for tk, r in rets.items():
        beta = float(np.cov(r, bench, ddof=1)[0, 1] / var_m) if var_m else None
        per_name[tk] = {
            "weight": round(weights.get(tk, 0.0), 4),
            "vol": round(float(np.std(r, ddof=1) * np.sqrt(WEEKS)), 4),
            "beta": round(beta, 3) if beta is not None else None,
            "corr_nifty": round(float(np.corrcoef(r, bench)[0, 1]), 3),
            "div_yield": round(div_yield[tk], 4) if div_yield.get(tk) is not None else None,
        }

    tickers = list(rets.keys())
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    R = np.vstack([rets[t] for t in tickers])             # names x weeks
    rp = w @ R                                             # weighted portfolio weekly returns
    cov = np.cov(R, ddof=1)                                # weekly covariance
    sig = np.sqrt(np.diag(cov))
    port_vol_wk = float(np.sqrt(w @ cov @ w))
    port_vol = port_vol_wk * np.sqrt(WEEKS)
    weighted_avg_vol = float((w * sig).sum() * np.sqrt(WEEKS))
    div_ratio = weighted_avg_vol / port_vol if port_vol else None
    eff_bets = div_ratio ** 2 if div_ratio else None

    # average pairwise correlation (off-diagonal)
    C = np.corrcoef(R)
    iu = np.triu_indices_from(C, k=1)
    avg_corr = float(C[iu].mean()) if iu[0].size else 1.0

    hhi = float((w ** 2).sum())
    eff_n = 1.0 / hhi

    port_beta = float(np.cov(rp, bench, ddof=1)[0, 1] / var_m) if var_m else None
    te = float(np.std(rp - bench, ddof=1) * np.sqrt(WEEKS))   # tracking error vs NIFTY
    # realised CAPM alpha (annualised regression intercept), backward-looking context only
    rf_wk = RF_INDIA / WEEKS
    realized_alpha = float((np.mean(rp) - (rf_wk + port_beta * (mean_m - rf_wk))) * WEEKS) if port_beta is not None else None

    # 3y max drawdown of the weighted book (real data)
    nav = np.cumprod(1.0 + rp)
    peak = np.maximum.accumulate(nav)
    max_dd = float((nav / peak - 1.0).min())

    port_div_yield = float(sum(w[i] * (per_name[t]["div_yield"] or 0.0) for i, t in enumerate(tickers)))

    # --- ex-ante (forward) expectations over the ~1y horizon -------------------------------------
    e_rm = RF_INDIA + ERP_MARKET
    gross_long_leg_alpha = OVERALL_1Y_LS / 2.0                 # long leg approx of the L/S spread
    expected_alpha = gross_long_leg_alpha * (1.0 - SURVIVORSHIP_HAIRCUT)
    e_rp = RF_INDIA + (port_beta or 1.0) * (e_rm - RF_INDIA) + expected_alpha
    expected_sharpe = (e_rp - RF_INDIA) / port_vol if port_vol else None
    info_ratio = expected_alpha / te if te else None

    out = {
        "as_of": dt.date.today().isoformat(),
        "window": f"{n} weekly obs (~3y), total-return basis",
        "benchmark": BENCHMARK,
        "benchmark_vol": round(bench_vol, 4),
        "assumptions": {
            "rf": RF_INDIA, "erp_market": ERP_MARKET, "e_rm": round(e_rm, 4),
            "energy_1y_ic": ENERGY_1Y_IC, "overall_1y_ls": OVERALL_1Y_LS,
            "gross_long_leg_alpha": round(gross_long_leg_alpha, 4),
            "survivorship_haircut": SURVIVORSHIP_HAIRCUT,
        },
        "per_name": per_name,
        "analyst": analyst,
        "corr_matrix": {"tickers": tickers, "matrix": np.round(C, 3).tolist()},
        "portfolio": {
            "beta": round(port_beta, 3) if port_beta is not None else None,
            "vol": round(port_vol, 4),
            "weighted_avg_vol": round(weighted_avg_vol, 4),
            "diversification_ratio": round(div_ratio, 3) if div_ratio else None,
            "effective_bets": round(eff_bets, 2) if eff_bets else None,
            "avg_pairwise_corr": round(avg_corr, 3),
            "hhi": round(hhi, 4),
            "effective_n_names": round(eff_n, 2),
            "tracking_error": round(te, 4),
            "div_yield": round(port_div_yield, 4),
            "realized_alpha_3y": round(realized_alpha, 4) if realized_alpha is not None else None,
            "max_drawdown_3y": round(max_dd, 4),
            "e_rp": round(e_rp, 4),
            "expected_alpha": round(expected_alpha, 4),
            "expected_sharpe": round(expected_sharpe, 3) if expected_sharpe else None,
            "information_ratio": round(info_ratio, 3) if info_ratio else None,
        },
    }
    (FOLDER / "risk_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    p = out["portfolio"]
    print(f"\nwrote {FOLDER / 'risk_stats.json'}")
    print(f"  beta={p['beta']}  vol={p['vol']:.1%}  TE={p['tracking_error']:.1%}  "
          f"divyld={p['div_yield']:.1%}")
    print(f"  diversification ratio={p['diversification_ratio']}  effective bets={p['effective_bets']}  "
          f"avg corr={p['avg_pairwise_corr']}  eff N (HHI)={p['effective_n_names']}")
    print(f"  E[Rp]={p['e_rp']:.1%}  expected Sharpe={p['expected_sharpe']}  "
          f"expected alpha={p['expected_alpha']:.1%}  IR={p['information_ratio']}")
    print(f"  realised 3y alpha={p['realized_alpha_3y']}  3y maxDD={p['max_drawdown_3y']:.1%}")


if __name__ == "__main__":
    main()
