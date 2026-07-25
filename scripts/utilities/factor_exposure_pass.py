"""Suite 4 -- Factor exposure / additivity (evaluation only; NO production change, NO DCF run).

Question: is `discount_to_intrinsic` ADDITIVE over standard factors, or a repackaging of them?

For each (ticker, date) in the existing 10,057-obs dataset, rebuild the SAME point-in-time
information set used to form the signal (same 75d filing lag, same price-asof) and compute
PIT factors -- but do NOT run the DCF (factors are cheap):
  Value:        earnings_yield (ttm_ebit/mktcap), fcf_yield (ttm_fcf/mktcap), book_to_market
  Quality:      roic, operating_margin
  Momentum:     trailing 12m price return
  Size:         log(market_cap)
  Profitability:fcf_margin, nopat_margin
(Gross margin unavailable -- no COGS.)

Then join to obs and test additivity:
  (i)   collinearity: per-date Spearman(discount, factor)
  (ii)  univariate obs-weighted IC of each factor alone
  (iii) Fama-MacBeth: per-date OLS fwd ~ discount + factors (standardized); discount coef
        averaged with Newey-West t-stat (HEADLINE). Value-only AND full-block controls.
  (iii-alt) factor-neutralized IC: residualize discount on the factor block per date, re-run
        obs-weighted IC vs fwd (rank-robust confirmation).

Writes reports/validation_factor_exposure.json. Run in the background.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import scripts.validation_common as C  # noqa: E402
import scripts.historical_predictive_power as bt  # noqa: E402
from dcf_project.config import get_settings  # noqa: E402
from dcf_project.preprocessing.reconstruction import reconstruct_historical_fcf  # noqa: E402

REPORT_DIR = ROOT / "reports"
HORIZONS = (1, 3, 5)
# the standardized factor block used in the multivariate / neutralized tests
FACTOR_BLOCK = ["earnings_yield", "roic", "momentum_12m", "log_mktcap", "fcf_margin"]
ALL_FACTORS = FACTOR_BLOCK + ["fcf_yield", "book_to_market", "operating_margin", "nopat_margin"]


def _base_settings():
    return replace(get_settings(), wacc_source="internal", terminal_growth_source="internal",
                   forecast_mode="state_space", consensus_weight=0.0, monte_carlo_samples=0, run_sensitivity=False)


def _factors_at(bundle, price_series, cutoff) -> dict | None:
    try:
        hist = reconstruct_historical_fcf(bundle.periods,
                                          use_proxy_invested_capital=get_settings().use_proxy_invested_capital).history
    except Exception:
        return None
    if hist is None or len(hist) < 4:
        return None
    last = hist.iloc[-1]
    tail4 = hist.tail(4)
    mktcap = bundle.market.market_cap or (bundle.market.price * (bundle.market.shares_outstanding or 0))
    if not mktcap or mktcap <= 0:
        return None

    def _ttm(col):
        if col not in tail4:
            return None
        v = float(tail4[col].sum())
        return v if np.isfinite(v) else None
    ttm_ebit = _ttm("ebit"); ttm_fcf = _ttm("fcf"); ttm_rev = _ttm("revenue"); ttm_nopat = _ttm("nopat")

    def _g(col):  # NaN-safe getter for a balance-sheet stock
        v = last.get(col)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None
    # book equity reconstruction (PIT); require the core components to be present
    ca, ppe, gw, intang, cl, debt = (_g("current_assets"), _g("ppe"), _g("goodwill"),
                                     _g("intangible_assets"), _g("current_liabilities"), _g("debt"))
    be = None
    if ca is not None and cl is not None:
        be = ca + (ppe or 0) + (gw or 0) + (intang or 0) - cl - (debt or 0)
    # momentum
    p0 = bt._price_asof(price_series, cutoff)
    p12 = bt._price_asof(price_series, cutoff - timedelta(days=365))
    mom = float(p0 / p12 - 1.0) if (p0 and p12 and p12 > 0) else None

    f = {
        "earnings_yield": (ttm_ebit / mktcap) if ttm_ebit is not None else None,
        "fcf_yield": (ttm_fcf / mktcap) if ttm_fcf is not None else None,
        "book_to_market": (be / mktcap) if be is not None else None,
        "roic": float(last["roic"]) if "roic" in last and last["roic"] is not None else None,
        "operating_margin": float(last["operating_margin"]) if "operating_margin" in last and last["operating_margin"] is not None else None,
        "momentum_12m": mom,
        "log_mktcap": float(np.log(mktcap)),
        "fcf_margin": (ttm_fcf / ttm_rev) if (ttm_fcf is not None and ttm_rev) else None,
        "nopat_margin": (ttm_nopat / ttm_rev) if (ttm_nopat is not None and ttm_rev) else None,
    }
    # final NaN/inf -> None sweep so nothing poisons the cross-sectional stats
    for k, v in list(f.items()):
        if isinstance(v, float) and not np.isfinite(v):
            f[k] = None
    return f


def build_factor_panel(obs: list[dict]) -> list[dict]:
    base = _base_settings()
    needed: dict[str, set] = {}
    for o in obs:
        needed.setdefault(o["ticker"], set()).add(o["date"])
    panel = []
    tickers = sorted(needed)
    n = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        try:
            periods_all, _ = bt.SECClient(base).financial_periods(ticker)
        except Exception:
            continue
        ps = bt._price_history(ticker)
        if ps is None:
            continue
        beta = bt._beta(ticker)
        for date_str in sorted(needed[ticker]):
            cutoff = bt.date.fromisoformat(date_str)
            bundle, _ = bt._build_pit_bundle(ticker, periods_all, ps, beta, cutoff, base)
            if bundle is None:
                continue
            f = _factors_at(bundle, ps, cutoff)
            if f is None:
                continue
            f["ticker"] = ticker
            f["date"] = date_str
            panel.append(f)
        if i % 50 == 0:
            print(f"[{i}/{n}] panel rows={len(panel)}", flush=True)
    return panel


# ---------------- statistics ----------------
def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def _rank_z(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank, then z-scored. Robust to the heavy tails of the raw discount
    (e.g. the 900x intrinsic blowups) that destabilize OLS on z-scored raw values."""
    from scipy import stats as _s
    r = _s.rankdata(x)
    return _zscore(r)


def collinearity(joined: list[dict]) -> dict:
    """Per-date Spearman(discount, factor), obs-weighted across dates."""
    from scipy import stats
    out = {}
    for fac in ALL_FACTORS:
        byd: dict[str, list] = {}
        for o in joined:
            if isinstance(o.get(fac), (int, float)) and isinstance(o.get("discount_to_intrinsic"), (int, float)):
                byd.setdefault(o["date"], []).append((o["discount_to_intrinsic"], o[fac]))
        series = []
        for v in byd.values():
            if len(v) >= 6:
                c = stats.spearmanr([p[0] for p in v], [p[1] for p in v]).statistic
                if np.isfinite(c):
                    series.append((len(v), c))
        out[fac] = (float(np.average([c for _, c in series], weights=[n for n, _ in series]))
                    if series else None)
    return out


def univariate_ic(joined: list[dict], y: int) -> dict:
    out = {}
    for fac in ALL_FACTORS:
        rows = [o for o in joined if isinstance(o.get(fac), (int, float))]
        series = C.per_date_ic_series(rows, y, signal_key=fac)
        out[fac] = C.obs_weighted_ic(series)
    # discount benchmark
    out["discount_to_intrinsic"] = C.obs_weighted_ic(C.per_date_ic_series(joined, y, "discount_to_intrinsic"))
    return out


def fama_macbeth(joined: list[dict], y: int, controls: list[str]) -> dict:
    """Per-date cross-sectional OLS: fwd ~ discount + controls (standardized). Average discount coef, NW t."""
    rk = f"fwd_{y}y"
    byd: dict[str, list] = {}
    for o in joined:
        if isinstance(o.get(rk), (int, float)) and isinstance(o.get("discount_to_intrinsic"), (int, float)) \
           and all(isinstance(o.get(c), (int, float)) for c in controls):
            byd.setdefault(o["date"], []).append(o)
    coefs, r2s = [], []
    for dt, g in byd.items():
        if len(g) < 20:
            continue
        yv = np.array([o[rk] for o in g], float)
        cols = ["discount_to_intrinsic"] + controls
        X = np.column_stack([_rank_z(np.array([o[c] for o in g], float)) for c in cols])
        yv = _zscore(yv)  # standardize target so coef is in return-SD per signal-rank-SD units
        X = np.column_stack([np.ones(len(g)), X])
        try:
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            pred = X @ beta
            ss_res = float(np.sum((yv - pred) ** 2)); ss_tot = float(np.sum((yv - yv.mean()) ** 2))
            coefs.append(float(beta[1]))  # discount coefficient (after intercept)
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        except Exception:
            continue
    nw = C.newey_west_tstat(coefs, C.HORIZON_QUARTERS[y])
    return {"controls": controls, "n_dates": len(coefs),
            "discount_coef_mean": nw.get("mean"), "discount_coef_nw_tstat": nw.get("t_stat"),
            "nw_lag_quarters": C.HORIZON_QUARTERS[y], "mean_r2": float(np.nanmean(r2s)) if r2s else None}


def neutralized_ic(joined: list[dict], y: int, block: list[str]) -> dict:
    """Residualize discount on the factor block per date; obs-weighted IC of residual vs fwd."""
    rk = f"fwd_{y}y"
    byd: dict[str, list] = {}
    for o in joined:
        if isinstance(o.get(rk), (int, float)) and isinstance(o.get("discount_to_intrinsic"), (int, float)) \
           and all(isinstance(o.get(c), (int, float)) for c in block):
            byd.setdefault(o["date"], []).append(o)
    resid_rows = []
    for dt, g in byd.items():
        if len(g) < 20:
            continue
        d = np.array([o["discount_to_intrinsic"] for o in g], float)
        dz = _rank_z(d)
        X = np.column_stack([np.ones(len(g))] + [_rank_z(np.array([o[c] for o in g], float)) for c in block])
        try:
            beta, *_ = np.linalg.lstsq(X, dz, rcond=None)
            resid = dz - X @ beta
        except Exception:
            continue
        for o, r in zip(g, resid):
            resid_rows.append({"date": dt, "neutral_discount": float(r), rk: o[rk]})
    series = C.per_date_ic_series(resid_rows, y, signal_key="neutral_discount")
    return {"neutralized_obs_weighted_ic": C.obs_weighted_ic(series), "n_periods": len(series)}


def main() -> None:
    obs = C.load_obs()
    panel_path = REPORT_DIR / "_factor_panel.json"
    if panel_path.exists():
        print("reusing cached factor panel (delete reports/_factor_panel.json to rebuild)", flush=True)
        panel = json.load(open(panel_path))
    else:
        print(f"building PIT factor panel for {len({o['ticker'] for o in obs})} tickers ...", flush=True)
        panel = build_factor_panel(obs)
        panel_path.write_text(json.dumps(panel), encoding="utf-8")
    pmap = {(p["ticker"], p["date"]): p for p in panel}
    joined = []
    for o in obs:
        f = pmap.get((o["ticker"], o["date"]))
        if f is None:
            continue
        row = dict(o)
        for k in ALL_FACTORS:
            row[k] = f.get(k)
        joined.append(row)
    print(f"joined {len(joined)}/{len(obs)} obs with factors", flush=True)

    result = {
        "method": "PIT factors (no DCF run); standardized within date. Additivity headline = 3Y/5Y "
                  "Fama-MacBeth discount coefficient NW t-stat after full-block control, corroborated by "
                  "factor-neutralized obs-weighted IC. Gross margin unavailable (no COGS).",
        "n_obs_total": len(obs), "n_obs_joined": len(joined),
        "factor_block": FACTOR_BLOCK,
        "collinearity_spearman_discount_vs_factor": collinearity(joined),
        "univariate_obs_weighted_ic": {f"{y}y": univariate_ic(joined, y) for y in HORIZONS},
        "fama_macbeth_full_block": {f"{y}y": fama_macbeth(joined, y, FACTOR_BLOCK) for y in HORIZONS},
        "fama_macbeth_value_only": {f"{y}y": fama_macbeth(joined, y, ["earnings_yield"]) for y in HORIZONS},
        "factor_neutralized_ic": {f"{y}y": neutralized_ic(joined, y, FACTOR_BLOCK) for y in HORIZONS},
    }
    (REPORT_DIR / "validation_factor_exposure.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("\n=== FACTOR ADDITIVITY ===")
    print("collinearity (Spearman discount vs factor):")
    for fac, v in result["collinearity_spearman_discount_vs_factor"].items():
        print(f"   {fac:18} {('None' if v is None else f'{v:+.3f}')}")
    for y in HORIZONS:
        fm = result["fama_macbeth_full_block"][f"{y}y"]
        nz = result["factor_neutralized_ic"][f"{y}y"]
        print(f" {y}y FM discount coef={fm['discount_coef_mean']} NW_t={fm['discount_coef_nw_tstat']} (n_dates={fm['n_dates']}) | neutralized IC={nz['neutralized_obs_weighted_ic']}")
    print("\nwrote reports/validation_factor_exposure.json")


if __name__ == "__main__":
    main()
