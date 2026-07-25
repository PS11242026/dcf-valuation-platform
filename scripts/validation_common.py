"""Shared statistics layer for the comprehensive validation suite (evaluation only).

The signal under test is `discount_to_intrinsic` from the existing 10,057-observation
point-in-time dataset (reports/final_universe_backtest.json). Headline metric is the
obs-weighted IC (per-date cross-sectional Spearman averaged weighted by per-date obs
count) -- equal-weight IC is reported but flagged untrustworthy (small-n early-period
artifact, documented in software_outlier_audit). Overlapping multi-year forward windows
induce autocorrelation, so every IC/spread headline carries a Newey-West t-stat (lag =
horizon in quarters) and a contiguous-date block-bootstrap 95% CI.

No production code is imported or modified.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
SIGNAL_KEY = "discount_to_intrinsic"
HORIZON_QUARTERS = {1: 4, 3: 12, 5: 20}  # NW lag / bootstrap block length


def load_obs(path: Path | None = None) -> list[dict]:
    data = json.load(open(path or (REPORT_DIR / "final_universe_backtest.json")))
    return data["observations"]


def _valid(o: dict, rk: str, signal_key: str = SIGNAL_KEY) -> bool:
    return isinstance(o.get(rk), (int, float)) and isinstance(o.get(signal_key), (int, float))


def per_date_ic_series(rows: list[dict], y: int, signal_key: str = SIGNAL_KEY) -> list[tuple[str, int, float]]:
    """Per-date cross-sectional Spearman(signal, fwd). Gate n_d >= 4 (established convention)."""
    rk = f"fwd_{y}y"
    byd: dict[str, list[tuple[float, float]]] = {}
    for o in rows:
        if _valid(o, rk, signal_key):
            byd.setdefault(o["date"], []).append((o[signal_key], o[rk]))
    out = []
    for dt in sorted(byd):
        v = byd[dt]
        if len(v) < 4:
            continue
        ic = stats.spearmanr([p[0] for p in v], [p[1] for p in v]).statistic
        if np.isfinite(ic):
            out.append((dt, len(v), float(ic)))
    return out


def obs_weighted_ic(series: list[tuple[str, int, float]]) -> float | None:
    if not series:
        return None
    return float(np.average([ic for _, _, ic in series], weights=[n for _, n, _ in series]))


def equal_weight_ic(series: list[tuple[str, int, float]]) -> float | None:
    if not series:
        return None
    return float(np.mean([ic for _, _, ic in series]))


def newey_west_tstat(values: list[float], lag: int) -> dict[str, Any]:
    """HAC (Newey-West) t-stat for mean(values)=0 on an autocorrelated series."""
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 3:
        return {"mean": float(x.mean()) if n else None, "t_stat": None, "n": n}
    mean = float(x.mean())
    d = x - mean
    gamma0 = float(np.dot(d, d) / n)
    lag = max(0, min(lag, n - 1))
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gamma_k = float(np.dot(d[k:], d[:-k]) / n)
        var += 2.0 * w * gamma_k
    var = max(var, 1e-12)
    se = float(np.sqrt(var / n))
    return {"mean": mean, "t_stat": float(mean / se) if se > 0 else None, "nw_se": se, "n": n, "lag": lag}


def block_bootstrap_ci(rows: list[dict], y: int, stat_fn: Callable[[list[dict]], float | None],
                       block_q: int, n_boot: int = 3000, seed: int = 7) -> dict[str, Any]:
    """Contiguous-date block bootstrap CI (preserves overlapping-window autocorrelation)."""
    rk = f"fwd_{y}y"
    rows = [o for o in rows if _valid(o, rk)]
    dates = sorted({o["date"] for o in rows})
    if len(dates) < 4:
        return {"ci95": None, "n_dates": len(dates)}
    by_date: dict[str, list[dict]] = {}
    for o in rows:
        by_date.setdefault(o["date"], []).append(o)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(dates) / block_q))
    ests = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(dates), size=n_blocks)
        sample_rows: list[dict] = []
        for s in starts:
            for dt in dates[s:s + block_q]:
                sample_rows.extend(by_date[dt])
        est = stat_fn(sample_rows)
        if est is not None and np.isfinite(est):
            ests.append(est)
    if len(ests) < 50:
        return {"ci95": None, "n_dates": len(dates)}
    lo, hi = np.percentile(ests, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "boot_mean": float(np.mean(ests)),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_dates": len(dates), "n_boot": len(ests)}


def per_date_buckets(rows: list[dict], y: int, signal_key: str = SIGNAL_KEY) -> dict[str, Any]:
    """Rank WITHIN each date, split into buckets, mean fwd per bucket; average across dates.

    Time-neutral (avoids the time-confounding of pooled deciles). Adaptive cross-section gate:
    n_d>=50 -> deciles (10); 30<=n_d<50 -> quintiles (5); n_d<30 -> dropped.
    Returns equal-date and obs-weighted bucket-return curves + per-date top-minus-bottom spread.
    """
    rk = f"fwd_{y}y"
    byd: dict[str, list[dict]] = {}
    for o in rows:
        if _valid(o, rk, signal_key):
            byd.setdefault(o["date"], []).append(o)

    decile_curves: list[tuple[int, list[float]]] = []   # (n_d, [10 bucket means])
    quintile_curves: list[tuple[int, list[float]]] = []
    spreads: list[tuple[str, float]] = []               # (date, top-bottom) at native granularity
    dropped = []
    for dt in sorted(byd):
        g = byd[dt]
        n_d = len(g)
        if n_d < 30:
            dropped.append({"date": dt, "n": n_d})
            continue
        nb = 10 if n_d >= 50 else 5
        sig = np.array([o[signal_key] for o in g])
        fwd = np.array([o[rk] for o in g])
        order = np.argsort(sig)
        buckets = np.array_split(order, nb)
        means = [float(np.mean(fwd[idx])) for idx in buckets]
        spreads.append((dt, means[-1] - means[0]))
        (decile_curves if nb == 10 else quintile_curves).append((n_d, means))

    def _avg_curve(curves):
        if not curves:
            return None
        arr = np.array([c for _, c in curves]); w = np.array([n for n, _ in curves], dtype=float)
        return {"equal_date": [round(float(x), 4) for x in arr.mean(axis=0)],
                "obs_weighted": [round(float(x), 4) for x in np.average(arr, axis=0, weights=w)],
                "n_dates": len(curves)}

    dec = _avg_curve(decile_curves)
    monotonicity = None
    if dec is not None:
        curve = dec["obs_weighted"]
        monotonicity = float(stats.spearmanr(range(len(curve)), curve).statistic)
        dec["monotonicity_spearman"] = monotonicity
        dec["monotonic_steps_increasing"] = int(sum(curve[i + 1] >= curve[i] for i in range(len(curve) - 1)))
        dec["n_steps"] = len(curve) - 1
    return {
        "decile_curve": dec,
        "quintile_curve_fallback": _avg_curve(quintile_curves),
        "per_date_spread": spreads,
        "dropped_dates": dropped,
    }


def spread_significance(per_date_spread: list[tuple[str, float]], y: int,
                        rows: list[dict] | None = None) -> dict[str, Any]:
    """NW t-stat + block-bootstrap CI on the per-date top-minus-bottom decile spread series."""
    lag = HORIZON_QUARTERS[y]
    vals = [s for _, s in per_date_spread]
    nw = newey_west_tstat(vals, lag)
    out = {"mean_spread": nw.get("mean"), "nw_tstat": nw.get("t_stat"), "nw_lag_quarters": lag,
           "n_dates": nw.get("n")}
    if rows is not None:
        def _stat(r):
            b = per_date_buckets(r, y)
            sp = [s for _, s in b["per_date_spread"]]
            return float(np.mean(sp)) if sp else None
        out["block_bootstrap_ci95"] = block_bootstrap_ci(rows, y, _stat, lag)
    return out


def pooled_buckets(rows: list[dict], y: int, n_buckets: int = 10, signal_key: str = SIGNAL_KEY) -> dict[str, Any]:
    """Time-confounded reference: rank all obs at once (kept only for comparison)."""
    rk = f"fwd_{y}y"
    rows = [o for o in rows if _valid(o, rk, signal_key)]
    if len(rows) < n_buckets * 2:
        return {"note": "insufficient", "n": len(rows)}
    sig = np.array([o[signal_key] for o in rows]); fwd = np.array([o[rk] for o in rows])
    order = np.argsort(sig)
    buckets = np.array_split(order, n_buckets)
    table = [{"bucket": i + 1, "n": int(len(idx)), "mean_discount": float(np.mean(sig[idx])),
              "mean_fwd": float(np.mean(fwd[idx])), "median_fwd": float(np.median(fwd[idx]))}
             for i, idx in enumerate(buckets)]
    return {"table": table, "spread_top_minus_bottom": table[-1]["mean_fwd"] - table[0]["mean_fwd"],
            "label": "time-confounded reference (pooled across dates)"}


def core_metrics(rows: list[dict], y: int, signal_key: str = SIGNAL_KEY,
                 with_ci: bool = True) -> dict[str, Any]:
    """obs-weighted IC (+ NW t-stat + block-bootstrap CI), equal-weight IC, pooled Spearman,
    hit rate, quintile L/S. The standard per-cohort metric block."""
    rk = f"fwd_{y}y"
    valid = [o for o in rows if _valid(o, rk, signal_key)]
    if len(valid) < 12:
        return {"n": len(valid), "note": "insufficient"}
    series = per_date_ic_series(rows, y, signal_key)
    ic_obs = obs_weighted_ic(series)
    lag = HORIZON_QUARTERS[y]
    # NW t-stat is computed on the per-date IC series, whose mean is the EQUAL-WEIGHT IC
    # (untrustworthy for small-n early cohorts). It is therefore reported as the equal-weight
    # significance; the obs-weighted IC's significance comes from the block-bootstrap CI below.
    nw = newey_west_tstat([ic for _, _, ic in series], lag)
    sig = np.array([o[signal_key] for o in valid]); fwd = np.array([o[rk] for o in valid])
    sp = stats.spearmanr(sig, fwd)
    order = np.argsort(sig); qs = np.array_split(order, 5)
    qm = [float(np.mean(fwd[idx])) for idx in qs]
    med = float(np.median(fwd)); under = fwd[sig > 0]
    out = {
        "n": len(valid), "n_periods": len(series),
        "ic_obs_weighted": ic_obs, "ic_equal_weight": equal_weight_ic(series),
        "equal_weight_ic_nw_tstat": nw.get("t_stat"), "nw_lag_quarters": lag,
        "spearman_pooled": float(sp.statistic),
        "spearman_p_naive": float(sp.pvalue),  # autocorrelation-inflated; reference only
        "hit_rate": float(np.mean(under > med)) if len(under) else None,
        "long_short_q5_minus_q1": float(qm[-1] - qm[0]),
        "effective_sample_size": round(len(series) / lag, 1),
    }
    if with_ci:
        out["ic_block_bootstrap_ci95"] = block_bootstrap_ci(
            rows, y, lambda r: obs_weighted_ic(per_date_ic_series(r, y, signal_key)), lag)
    return out


# --------------------------------------------------------------------------- #
# Price-basis primitives: split consistency BY CONSTRUCTION.
#
# Extracted from scripts/utilities/research_v2/split_corrected_universe_revalidation.py
# (the validated split correction, 35/35 manual audit) so new backtest runners build
# split-consistent panels instead of rescaling defective ones. The research_v2
# original stays frozen.
#
# Basis convention (the validated "fully corrected" semantics):
# - shares: as-reported x cumulative WHITELISTED split factors dated after each
#   period's report_date => every period on TODAY's split basis (unit re-expression,
#   zero information, PIT-safe).
# - valuation price: yfinance Close with auto_adjust=False (split-adjusted to today's
#   basis, NOT dividend-adjusted) -- the same basis as the re-expressed shares.
# - forward returns: Adj Close ratios (split + dividend = total return).
# --------------------------------------------------------------------------- #

SPLIT_WHITELIST = {2, 3, 4, 5, 6, 7, 8, 10, 15, 20}


def clean_split_factor(ratio: float) -> float:
    """Whitelisted integer split factor (within 15%), else 1.0 (ignore junk ratios)."""
    r = float(ratio)
    f = round(r)
    return float(f) if (f in SPLIT_WHITELIST and abs(r - f) / f <= 0.15) else 1.0


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def fetch_whitelisted_splits(
    tickers: list[str],
    cache_path: Path,
    seed_caches: tuple[Path, ...] = (),
    sleep_s: float = 0.15,
) -> dict[str, list]:
    """Whitelisted forward-split events per ticker ([['YYYY-MM-DD', factor], ...]).

    Resumable: own cache at `cache_path` is extended and rewritten every 25 fetches;
    `seed_caches` (e.g. the frozen research_v2 split cache) are read-only donors.
    """
    cache: dict[str, list] = {}
    for seed in seed_caches:
        seed = Path(seed)
        if seed.exists():
            try:
                cache.update(json.loads(seed.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
    cache_path = Path(cache_path)
    own: dict[str, list] = {}
    if cache_path.exists():
        try:
            own = json.loads(cache_path.read_text(encoding="utf-8"))
            cache.update(own)
        except (json.JSONDecodeError, OSError):
            own = {}
    todo = [t for t in tickers if t not in cache]
    if todo:
        import yfinance as yf

        for i, tk in enumerate(todo):
            events: list = []
            try:
                s = yf.Ticker(tk).splits
                if s is not None and len(s):
                    import pandas as pd

                    for ts, ratio in s.items():
                        f = clean_split_factor(ratio)
                        if f > 1.0:
                            events.append([pd.Timestamp(ts).strftime("%Y-%m-%d"), f])
            except Exception:  # noqa: BLE001 -- missing splits = no correction
                events = []
            cache[tk] = events
            own[tk] = events
            if (i + 1) % 25 == 0:
                _atomic_write_text(cache_path, json.dumps(own))
                print(f"  splits {i + 1}/{len(todo)} fetched", flush=True)
            time.sleep(sleep_s)
        _atomic_write_text(cache_path, json.dumps(own))
    return {t: cache.get(t, []) for t in tickers}


def cum_split_factor_after(events: list, ref: date) -> float:
    """Product of whitelisted split factors dated strictly after `ref`."""
    f = 1.0
    for ds, fac in events:
        if date.fromisoformat(ds) > ref:
            f *= float(fac)
    return f


def reexpress_shares_today_basis(periods: list, events: list) -> list:
    """Re-express each FinancialPeriod's shares_outstanding on today's split basis.

    Multiplies by the cumulative whitelisted split factor dated after that period's
    report_date. Returns new period objects (pydantic model_copy); the input list is
    untouched. With no events, the original list is returned as-is.
    """
    if not events:
        return list(periods)
    out = []
    for p in periods:
        factor = cum_split_factor_after(events, p.report_date)
        if factor != 1.0 and p.shares_outstanding:
            out.append(p.model_copy(update={"shares_outstanding": float(p.shares_outstanding) * factor}))
        else:
            out.append(p)
    return out


def pit_price_frame(ticker: str, cache_dir: Path, max_age_days: float = 7.0):
    """Daily {close, adj_close} frame for PIT valuation/returns, disk-cached (csv.gz).

    close = split-adjusted (today's basis), no dividend adjustment -> valuation price.
    adj_close = split+dividend adjusted -> total-return forward windows.
    Returns a pandas DataFrame or None on failure.
    """
    import pandas as pd

    cache_dir = Path(cache_dir)
    fp = cache_dir / f"{ticker.upper()}.csv.gz"
    if fp.exists() and (time.time() - fp.stat().st_mtime) / 86400.0 <= max_age_days:
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if len(df) and {"close", "adj_close"} <= set(df.columns):
                return df
        except Exception:  # noqa: BLE001 -- corrupt cache entry = miss
            pass
    try:
        import yfinance as yf

        raw = yf.Ticker(ticker).history(start="2008-01-01", auto_adjust=False)
        if raw is None or raw.empty or "Adj Close" not in raw.columns:
            raw = yf.download(ticker, start="2008-01-01", auto_adjust=False,
                              progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
    except Exception:  # noqa: BLE001
        return None
    if raw is None or raw.empty or "Close" not in raw.columns:
        return None
    df = pd.DataFrame({"close": raw["Close"]})
    df["adj_close"] = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(how="all")
    if not len(df):
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(".tmp.gz")
    try:
        df.to_csv(tmp, compression="gzip")
        os.replace(tmp, fp)
    except OSError:
        pass
    return df


def paired_ic_diff(rows: list[dict], y: int, key_a: str, key_b: str) -> dict[str, Any]:
    """NW t-stat on the per-date IC difference (engine A minus engine B) -- the paired
    comparison statistic for same-bundle engine A/B tests."""
    sa = {d: ic for d, _, ic in per_date_ic_series(rows, y, key_a)}
    sb = {d: ic for d, _, ic in per_date_ic_series(rows, y, key_b)}
    dates = sorted(set(sa) & set(sb))
    diffs = [sa[d] - sb[d] for d in dates]
    nw = newey_west_tstat(diffs, HORIZON_QUARTERS[y])
    return {
        "n_dates": len(dates),
        "mean_ic_diff": nw.get("mean"),
        "nw_tstat": nw.get("t_stat"),
        "nw_lag_quarters": HORIZON_QUARTERS[y],
    }


REGIMES = [
    ("2015-2018_zirp_bull", "2015-01-01", "2018-12-31"),
    ("2019-2020Q1_pre_covid", "2019-01-01", "2020-03-31"),
    ("2020Q2-2021_covid_meltup", "2020-04-01", "2021-12-31"),
    ("2022_rate_shock", "2022-01-01", "2022-12-31"),
    ("2023-2026_higher_for_longer_AI", "2023-01-01", "2026-12-31"),
]


def regime_of(date_str: str) -> str | None:
    for name, lo, hi in REGIMES:
        if lo <= date_str <= hi:
            return name
    return None
