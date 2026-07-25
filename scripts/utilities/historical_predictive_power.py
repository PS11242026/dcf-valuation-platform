"""Backtest calibration / predictive-power study.

For each ticker and each quarter from 2015 onward:
  1. Truncate fundamentals to that date (period_end <= cutoff - reporting_lag).
  2. Generate the model's intrinsic value at that date using POINT-IN-TIME inputs
     (historical adjusted price, internal CAPM WACC, internal terminal growth,
     pure state-space forecast -- NO live provider/consensus lookahead).
  3. Compare discount-to-intrinsic with realized 3y and 5y annualized returns
     (adjusted close = total return).

Measures: Pearson/Spearman correlation, decile performance, hit rate,
information coefficient (mean cross-sectional Spearman), and calibration error.

This evaluates predictive power ONLY. No valuation assumption is changed and no
parameter is fit to market prices.

Output: reports/historical_predictive_power.json

Usage:
    PYTHONPATH=src python scripts/historical_predictive_power.py
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.contracts import FinancialPeriod, MarketSnapshot, SourceMeta  # noqa: E402
from dcf_project.data_sources.fundamentals import FundamentalsBundle  # noqa: E402
from dcf_project.data_sources.market import build_wacc  # noqa: E402
from dcf_project.data_sources.sec import SECClient  # noqa: E402
from dcf_project.pipeline import DCFPipeline  # noqa: E402

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "KO", "NVDA", "TSLA"]
REPORT_DIR = ROOT / "reports"
REPORTING_LAG_DAYS = 75  # approximate 10-Q/10-K filing lag after period end
TODAY = date(2026, 6, 1)


class _PITPipeline(DCFPipeline):
    def __init__(self, settings, bundle: FundamentalsBundle):
        super().__init__(settings)
        self._bundle = bundle

    def load_data(self, ticker: str) -> FundamentalsBundle:
        return self._bundle


def _price_history(ticker: str) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.Ticker(ticker).history(start="2008-01-01", auto_adjust=True)
    except Exception:
        return None
    if df is None or df.empty or "Close" not in df:
        return None
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _beta(ticker: str) -> float:
    try:
        import yfinance as yf

        b = yf.Ticker(ticker).get_info().get("beta")
        return float(b) if b and float(b) > 0 else 1.0
    except Exception:
        return 1.0


def _price_asof(series: pd.Series, d: date) -> float | None:
    ts = pd.Timestamp(d)
    sub = series[series.index <= ts]
    return float(sub.iloc[-1]) if len(sub) else None


def _quarter_ends(start_year: int = 2015) -> list[date]:
    out = []
    for year in range(start_year, TODAY.year + 1):
        for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            d = date(year, month, day)
            if d <= TODAY:
                out.append(d)
    return out


def _latest(values: list[FinancialPeriod], attr: str) -> float | None:
    for p in reversed(values):
        v = getattr(p, attr, None)
        if v is not None:
            return float(v)
    return None


def _build_pit_bundle(ticker, periods_all, price_series, beta, cutoff, settings):
    avail = [p for p in periods_all if p.report_date <= cutoff - timedelta(days=REPORTING_LAG_DAYS)]
    if len(avail) < 24:
        return None, "insufficient_periods"
    price = _price_asof(price_series, cutoff)
    if price is None or price <= 0:
        return None, "no_price"
    shares = _latest(avail, "shares_outstanding")
    cash = _latest(avail, "cash")
    debt = _latest(avail, "debt")
    if not shares or shares <= 0:
        return None, "no_shares"
    market = MarketSnapshot(
        ticker=ticker.upper(),
        price=price,
        market_cap=price * shares,
        shares_outstanding=shares,
        beta=beta,
        cash=cash,
        debt=debt,
        source=SourceMeta(provider="backtest_pit", key=ticker.upper()),
    )
    latest = avail[-1]
    tax_rate = None
    if latest.tax_expense is not None and latest.ebit not in (None, 0):
        tax_rate = latest.tax_expense / abs(latest.ebit)
    try:
        wacc = build_wacc(market, tax_rate, debt, settings)
    except Exception as exc:
        return None, f"wacc_error:{exc}"
    return FundamentalsBundle(avail, market, wacc, [market.source], []), None


def run_backtest() -> dict[str, Any]:
    # Point-in-time settings: strip all live-provider / consensus lookahead.
    settings = replace(
        get_settings(),
        wacc_source="internal",
        terminal_growth_source="internal",
        forecast_mode="state_space",
        consensus_weight=0.0,
        monte_carlo_samples=0,
        run_sensitivity=False,
    )
    observations: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    cutoffs = _quarter_ends(2015)
    for ticker in UNIVERSE:
        try:
            periods_all, _ = SECClient(settings).financial_periods(ticker)
        except Exception as exc:
            skips[f"{ticker}:sec"] = skips.get(f"{ticker}:sec", 0) + 1
            print(f"{ticker}: SEC fetch failed: {exc}")
            continue
        price_series = _price_history(ticker)
        if price_series is None:
            print(f"{ticker}: price history failed")
            continue
        beta = _beta(ticker)
        last_price_date = price_series.index[-1].date()
        for cutoff in cutoffs:
            bundle, reason = _build_pit_bundle(ticker, periods_all, price_series, beta, cutoff, settings)
            if bundle is None:
                skips[reason or "unknown"] = skips.get(reason or "unknown", 0) + 1
                continue
            try:
                result = _PITPipeline(settings, bundle).run(ticker)
            except Exception as exc:
                skips[f"run:{type(exc).__name__}"] = skips.get(f"run:{type(exc).__name__}", 0) + 1
                continue
            price = bundle.market.price
            intrinsic = float(result.intrinsic_value_per_share)
            # Disciplined DCF intrinsic: when a regime (e.g. AMZN/TSLA
            # optionality_hypergrowth) overrides the headline value with a
            # speculative scenario number, fall back to the underlying DCF so every
            # ticker is compared on the same intrinsic-DCF basis.
            dcf_diag = result.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
            dcf_intrinsic = float(dcf_diag) if isinstance(dcf_diag, (int, float)) else intrinsic
            discount = intrinsic / price - 1.0  # headline discount (regime value)
            discount_dcf = dcf_intrinsic / price - 1.0  # distortion-removed signal
            obs: dict[str, Any] = {
                "ticker": ticker,
                "date": cutoff.isoformat(),
                "price": price,
                "intrinsic": intrinsic,
                "dcf_intrinsic": dcf_intrinsic,
                "regime_overrode": isinstance(dcf_diag, (int, float)),
                "discount_to_intrinsic": discount,
                "discount_dcf": discount_dcf,
                "wacc": float(result.wacc),
                "terminal_growth": float(result.terminal_growth_rate),
            }
            for years in (3, 5):
                target = cutoff + timedelta(days=round(365.25 * years))
                if target > last_price_date:
                    obs[f"fwd_{years}y_ann_return"] = None
                    continue
                p_fwd = _price_asof(price_series, target)
                if p_fwd is None or p_fwd <= 0:
                    obs[f"fwd_{years}y_ann_return"] = None
                    continue
                obs[f"fwd_{years}y_ann_return"] = float((p_fwd / price) ** (1.0 / years) - 1.0)
                obs[f"fwd_{years}y_predicted_ann_return"] = float(
                    (intrinsic / price) ** (1.0 / years) - 1.0
                )  # convergence-to-intrinsic implied annual return
            observations.append(obs)

    return {
        "observations": observations,
        "skips": skips,
        "metrics": {f"{y}y": _metrics_for_horizon(observations, y, "discount_to_intrinsic") for y in (3, 5)},
        "metrics_distortion_removed": {
            f"{y}y": _metrics_for_horizon(observations, y, "discount_dcf") for y in (3, 5)
        },
    }


def _metrics_for_horizon(observations: list[dict[str, Any]], years: int, signal_key: str) -> dict[str, Any]:
    key = f"fwd_{years}y_ann_return"
    pred_key = f"fwd_{years}y_predicted_ann_return"
    rows = [o for o in observations if isinstance(o.get(key), (int, float)) and isinstance(o.get(signal_key), (int, float))]
    if len(rows) < 10:
        return {"n_obs": len(rows), "note": "insufficient observations"}
    signal = np.array([o[signal_key] for o in rows])
    fwd = np.array([o[key] for o in rows])
    pred = np.array([o.get(pred_key, np.nan) for o in rows])

    pear = stats.pearsonr(signal, fwd)
    spear = stats.spearmanr(signal, fwd)

    # Information coefficient: mean cross-sectional Spearman per rebalance date.
    by_date: dict[str, list[tuple[float, float]]] = {}
    for o in rows:
        by_date.setdefault(o["date"], []).append((o[signal_key], o[key]))
    period_ics = []
    for pairs in by_date.values():
        if len(pairs) >= 3:
            s = np.array([p[0] for p in pairs])
            f = np.array([p[1] for p in pairs])
            ic = stats.spearmanr(s, f).statistic
            if np.isfinite(ic):
                period_ics.append(float(ic))

    # Decile performance (by signal). Use deciles if enough obs, else quintiles.
    n_buckets = 10 if len(rows) >= 50 else 5
    order = np.argsort(signal)
    buckets = np.array_split(order, n_buckets)
    decile_table = []
    for i, idx in enumerate(buckets):
        decile_table.append(
            {
                "bucket": i + 1,
                "n": int(len(idx)),
                "mean_discount_to_intrinsic": float(np.mean(signal[idx])),
                "mean_fwd_ann_return": float(np.mean(fwd[idx])),
                "median_fwd_ann_return": float(np.median(fwd[idx])),
            }
        )
    long_short = decile_table[-1]["mean_fwd_ann_return"] - decile_table[0]["mean_fwd_ann_return"]

    # Hit rate: of stocks flagged undervalued (discount>0), fraction with positive forward return.
    under = fwd[signal > 0.0]
    over = fwd[signal <= 0.0]
    hit_rate_positive = float(np.mean(under > 0.0)) if len(under) else None
    # Directional skill: did undervalued names beat the cross-period median forward return?
    median_fwd = float(np.median(fwd))
    hit_rate_vs_median = float(np.mean(under > median_fwd)) if len(under) else None

    # Calibration: realized vs predicted (convergence-implied) annualized return.
    valid = np.isfinite(pred)
    calib = {}
    if valid.sum() >= 10:
        p = pred[valid]
        r = fwd[valid]
        bias = float(np.mean(r - p))
        rmse = float(np.sqrt(np.mean((r - p) ** 2)))
        slope, intercept, rval, pval, _ = stats.linregress(p, r)
        # Binned calibration curve.
        qs = np.quantile(p, np.linspace(0, 1, 6))
        bins = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (p >= lo) & (p <= hi)
            if m.sum() >= 3:
                bins.append(
                    {
                        "predicted_range": [float(lo), float(hi)],
                        "n": int(m.sum()),
                        "mean_predicted": float(np.mean(p[m])),
                        "mean_realized": float(np.mean(r[m])),
                    }
                )
        calib = {
            "mean_bias_realized_minus_predicted": bias,
            "rmse": rmse,
            "regression_slope": float(slope),
            "regression_intercept": float(intercept),
            "regression_r": float(rval),
            "calibration_curve": bins,
        }

    return {
        "n_obs": len(rows),
        "n_undervalued_signal": int((signal > 0).sum()),
        "n_overvalued_signal": int((signal <= 0).sum()),
        "pearson_corr": float(pear.statistic),
        "pearson_pvalue": float(pear.pvalue),
        "spearman_corr_pooled": float(spear.statistic),
        "spearman_pvalue": float(spear.pvalue),
        "information_coefficient_mean": float(np.mean(period_ics)) if period_ics else None,
        "information_coefficient_std": float(np.std(period_ics)) if period_ics else None,
        "information_coefficient_n_periods": len(period_ics),
        "decile_table": decile_table,
        "long_short_top_minus_bottom": float(long_short),
        "hit_rate_undervalued_positive_return": hit_rate_positive,
        "hit_rate_undervalued_beats_median": hit_rate_vs_median,
        "mean_fwd_return_undervalued": float(np.mean(under)) if len(under) else None,
        "mean_fwd_return_overvalued": float(np.mean(over)) if len(over) else None,
        "calibration": calib,
        "overlap_caveat": "Quarterly cutoffs with multi-year horizons overlap; observations are autocorrelated, inflating apparent significance.",
    }


def _conclusion(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for h in ("3y", "5y"):
        m = metrics.get(h, {})
        if "spearman_corr_pooled" not in m:
            out[h] = "insufficient_data"
            continue
        ic = m.get("information_coefficient_mean")
        ls = m.get("long_short_top_minus_bottom")
        under = m.get("mean_fwd_return_undervalued")
        over = m.get("mean_fwd_return_overvalued")
        positive_signal = (m["spearman_corr_pooled"] > 0) and (ls is not None and ls > 0)
        out[h] = {
            "spearman": m["spearman_corr_pooled"],
            "mean_IC": ic,
            "long_short_spread": ls,
            "undervalued_minus_overvalued_return": (under - over) if (under is not None and over is not None) else None,
            "undervalued_outperformed": bool(positive_signal),
            "verdict": (
                "Undervalued names subsequently outperformed (positive but modest signal)."
                if positive_signal
                else "No reliable evidence that model-undervalued names outperformed."
            ),
        }
    return out


def _compare_signals(headline: dict[str, Any], removed: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for h in ("3y", "5y"):
        a, b = headline.get(h, {}), removed.get(h, {})
        if "spearman_corr_pooled" not in a or "spearman_corr_pooled" not in b:
            out[h] = {"note": "insufficient data"}
            continue
        out[h] = {
            metric: {
                "headline_regime_intrinsic": a.get(metric),
                "distortion_removed_dcf_intrinsic": b.get(metric),
                "delta": (b.get(metric) - a.get(metric))
                if isinstance(a.get(metric), (int, float)) and isinstance(b.get(metric), (int, float))
                else None,
            }
            for metric in (
                "spearman_corr_pooled",
                "information_coefficient_mean",
                "long_short_top_minus_bottom",
                "hit_rate_undervalued_beats_median",
            )
        }
    return out


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    result = run_backtest()
    payload = {
        "methodology": {
            "universe": UNIVERSE,
            "period": "quarter-ends 2015 -> present, point-in-time",
            "reporting_lag_days": REPORTING_LAG_DAYS,
            "lookahead_controls": [
                "historical adjusted-close prices (total return) for valuation date and forward returns",
                "internal CAPM WACC (no live provider scrape)",
                "internal terminal growth (no live provider scrape)",
                "forecast_mode=state_space (no live analyst consensus)",
                "fundamentals truncated to period_end <= cutoff - reporting_lag",
            ],
            "known_limitations": [
                "Beta and risk-free rate held at current values (not point-in-time); shifts WACC level but is ~constant across the cross-section.",
                "Overlapping multi-year horizons on quarterly cutoffs -> autocorrelated observations.",
                "SEC facts may be lightly restated vs the original filing.",
                "Survivorship: universe is 8 large survivors, not a full cross-section.",
            ],
            "signal": "discount_to_intrinsic = intrinsic/price - 1 (higher = more undervalued)",
        },
        "n_observations": len(result["observations"]),
        "skips": result["skips"],
        "metrics": result["metrics"],
        "metrics_distortion_removed": result["metrics_distortion_removed"],
        "headline_vs_distortion_removed_comparison": _compare_signals(
            result["metrics"], result["metrics_distortion_removed"]
        ),
        "conclusion": _conclusion(result["metrics"]),
        "conclusion_distortion_removed": _conclusion(result["metrics_distortion_removed"]),
        "observations": result["observations"],
    }
    (REPORT_DIR / "historical_predictive_power.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    # Console summary
    print(f"\nTotal observations: {len(result['observations'])}  skips={result['skips']}")
    for label, metrics in (("HEADLINE (regime)", result["metrics"]), ("DISTORTION-REMOVED (DCF)", result["metrics_distortion_removed"])):
        print(f"\n--- {label} ---")
        for h in ("3y", "5y"):
            m = metrics.get(h, {})
            if "spearman_corr_pooled" not in m:
                print(f"[{h}] {m.get('note')}")
                continue
            print(
                f"[{h}] n={m['n_obs']} spearman={m['spearman_corr_pooled']:.3f} "
                f"IC={m['information_coefficient_mean']:.4f}  L/S={m['long_short_top_minus_bottom']:.3f} "
                f"hit(beats med)={m['hit_rate_undervalued_beats_median']}"
            )


if __name__ == "__main__":
    main()
