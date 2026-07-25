from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcf_project.contracts import FinancialPeriod


REQUIRED_COLUMNS = [
    "date",
    "revenue",
    "ebit",
    "tax_expense",
    "da",
    "r_and_d",
    "capex",
    "current_assets",
    "cash",
    "current_liabilities",
    "ppe",
    "goodwill",
    "intangible_assets",
    "operating_lease_liability",
    "debt",
    "shares",
]

STATE_COLUMNS = [
    "revenue_growth",
    "operating_margin",
    "reinvestment_rate",
    "roic",
]


@dataclass(frozen=True)
class ReconstructionResult:
    history: pd.DataFrame
    warnings: list[str]


def standardized_quarterly_frame(periods: list[FinancialPeriod]) -> pd.DataFrame:
    rows = []
    for period in periods:
        rows.append(
            {
                "date": period.report_date,
                "fiscal_period": period.fiscal_period,
                "revenue": period.revenue,
                "ebit": period.ebit,
                "tax_expense": period.tax_expense,
                "da": period.da,
                "r_and_d": period.r_and_d,
                "capex": period.capex,
                "current_assets": period.current_assets,
                "cash": period.cash,
                "current_liabilities": period.current_liabilities,
                "ppe": period.ppe,
                "goodwill": period.goodwill,
                "intangible_assets": period.intangible_assets,
                "operating_lease_liability": period.operating_lease_liability,
                "source_nwc": period.nwc,
                "debt": period.debt,
                "shares": period.shares_outstanding,
                "sbc": period.sbc,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No financial periods supplied")
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    for column in REQUIRED_COLUMNS:
        if column != "date":
            df[column] = pd.to_numeric(df.get(column), errors="coerce")
    return df


def reconstruct_historical_fcf(
    periods: list[FinancialPeriod] | pd.DataFrame,
    use_proxy_invested_capital: bool = False,
    sbc_addback_fraction: float = 0.0,
) -> ReconstructionResult:
    df = periods.copy() if isinstance(periods, pd.DataFrame) else standardized_quarterly_frame(periods)
    warnings: list[str] = []
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    for column in REQUIRED_COLUMNS:
        if column not in df:
            df[column] = np.nan
    for column in REQUIRED_COLUMNS:
        if column != "date":
            df[column] = pd.to_numeric(df[column], errors="coerce")
            missing = int(df[column].isna().sum())
            if missing:
                warnings.append(f"{column} missing in {missing} periods.")

    df["capex"] = df["capex"].abs()
    _normalize_da(df, warnings)
    df["nwc"] = (df["current_assets"] - df["cash"]) - df["current_liabilities"]
    legacy_nwc = df["nwc"].copy()
    if "source_nwc" in df:
        legacy_nwc = pd.to_numeric(df["source_nwc"], errors="coerce")
    df["nwc"] = df["nwc"].where(df["nwc"].notna(), legacy_nwc)
    df["delta_nwc"] = df["nwc"].diff().fillna(0.0)
    df["tax_rate"] = _safe_divide(df["tax_expense"], df["ebit"]).clip(0.0, 0.45)
    df["tax_rate"] = df["tax_rate"].ffill().bfill().fillna(0.21)
    # SBC treatment lever (default 0.0 = production: SBC fully expensed). The effective
    # tax rate above is computed on GAAP EBIT first; the add-back then lifts operating
    # income so the higher margin flows into nopat/fcf/roic and the state-space fit.
    if sbc_addback_fraction and "sbc" in df:
        sbc_series = pd.to_numeric(df["sbc"], errors="coerce").fillna(0.0)
        df["ebit"] = df["ebit"] + float(sbc_addback_fraction) * sbc_series
    df["nopat"] = df["ebit"] * (1.0 - df["tax_rate"])
    df["fcf"] = df["nopat"] + df["da"].fillna(0.0) - df["capex"].fillna(0.0) - df["delta_nwc"]
    df["net_capex"] = df["capex"].fillna(0.0) - df["da"].fillna(0.0)
    df["reinvestment"] = df["net_capex"] + df["delta_nwc"]
    df["invested_capital"] = _invested_capital(df, warnings, use_proxy_invested_capital)
    df["invested_capital_qoq_change"] = df["invested_capital"].pct_change()
    raw_jump = (
        df["raw_invested_capital_qoq_change"].abs() > 0.30
        if "raw_invested_capital_qoq_change" in df
        else pd.Series(False, index=df.index)
    )
    df["invested_capital_continuity_flag"] = raw_jump | (df["invested_capital_qoq_change"].abs() > 0.30)
    flag_count = int(df["invested_capital_continuity_flag"].fillna(False).sum())
    if flag_count:
        warnings.append(f"Invested capital changed by more than 30% QoQ in {flag_count} periods.")

    df["revenue_growth"] = df["revenue"].pct_change()
    df["ebit_margin"] = _safe_divide(df["ebit"], df["revenue"])
    df["operating_margin"] = df["ebit_margin"]
    df["da_ratio"] = _safe_divide(df["da"], df["revenue"])
    df["r_and_d_ratio"] = _safe_divide(df["r_and_d"], df["revenue"]).fillna(0.0)
    df["capex_ratio"] = _safe_divide(df["capex"], df["revenue"])
    df["nwc_ratio"] = _safe_divide(df["delta_nwc"], df["revenue"])
    df["roic"] = _safe_divide(df["nopat"], df["ic_used_for_roic"])
    df["reinvestment_rate"] = _safe_divide(df["reinvestment"], df["nopat"].abs())
    df["raw_reinvestment_rate"] = df["reinvestment_rate"]
    df["raw_roic"] = df["roic"]
    df["reinvestment_rate"] = _normalized_reinvestment_rate(df)
    df["roic"] = _normalized_roic(df)

    for column, low, high in [
        ("revenue_growth", -0.70, 1.50),
        ("ebit_margin", -0.50, 0.80),
        ("operating_margin", -0.50, 0.80),
        ("tax_rate", 0.00, 0.45),
        ("da_ratio", 0.00, 0.60),
        ("capex_ratio", 0.00, 0.80),
        ("nwc_ratio", -0.80, 0.80),
        ("reinvestment_rate", -1.00, 1.50),
        ("roic", -0.25, 0.40),
    ]:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan).clip(low, high)

    if os.environ.get("DCF_DEBUG_NA_COUNTS"):
        _diag_cols = STATE_COLUMNS + ["fcf", "revenue"]
        _na = {col: int(df[col].isna().sum()) for col in _diag_cols if col in df}
        print(f"[reconstruct_historical_fcf] rows={len(df)} NA counts before dropna: {_na}")

    usable = df.dropna(subset=STATE_COLUMNS + ["fcf", "revenue"]).copy()
    if len(usable) < 20:
        raise ValueError("Insufficient complete quarterly history for state-space DCF; need at least 20 quarters")
    if len(usable) < 32:
        warnings.append(f"Only {len(usable)} complete quarters available; 32+ preferred.")
    if len(usable) < len(df):
        warnings.append(f"Dropped {len(df) - len(usable)} incomplete periods before modeling.")
    return ReconstructionResult(history=usable.reset_index(drop=True), warnings=warnings)


def state_matrix(history: pd.DataFrame) -> np.ndarray:
    return history[STATE_COLUMNS].astype(float).to_numpy()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype(float).replace(0.0, np.nan)
    denom = denom.mask(denom.abs() < 1e-9)
    return numerator.astype(float) / denom


def _normalize_da(df: pd.DataFrame, warnings: list[str]) -> None:
    raw_ratio = _safe_divide(df["da"], df["revenue"]).replace([np.inf, -np.inf], np.nan)
    non_fy_mask = df.get("fiscal_period", pd.Series("", index=df.index)).ne("FY")
    quarterly_ratios = raw_ratio[non_fy_mask & raw_ratio.notna() & raw_ratio.between(0.0, 0.20)]
    if quarterly_ratios.empty:
        quarterly_ratios = raw_ratio[raw_ratio.notna() & raw_ratio.between(0.0, 0.20)]
    if quarterly_ratios.empty:
        return
    ratio = float(quarterly_ratios.tail(12).median())
    missing_da = df["da"].isna()
    fy_outlier = (
        df.get("fiscal_period", pd.Series("", index=df.index)).eq("FY")
        & raw_ratio.notna()
        & (raw_ratio > max(ratio * 1.75, 0.08))
    )
    replace_mask = missing_da | fy_outlier
    if replace_mask.any():
        df.loc[replace_mask, "da"] = df.loc[replace_mask, "revenue"] * ratio
        warnings.append(
            f"D&A normalized in {int(replace_mask.sum())} periods using quarterly median D&A/revenue {ratio:.2%}."
        )


def _invested_capital(
    df: pd.DataFrame,
    warnings: list[str] | None = None,
    use_proxy_invested_capital: bool = False,
) -> pd.Series:
    warnings = warnings if warnings is not None else []
    current_assets = _stock_account(df, "current_assets")
    cash = _stock_account(df, "cash")
    current_liabilities = _stock_account(df, "current_liabilities")
    ppe = _stock_account(df, "ppe")
    goodwill = _stock_account(df, "goodwill")
    intangibles = _stock_account(df, "intangible_assets")
    leases = _stock_account(df, "operating_lease_liability")
    nwc = (current_assets - cash) - current_liabilities
    source_nwc = pd.to_numeric(df.get("source_nwc", pd.Series(np.nan, index=df.index)), errors="coerce")
    nwc = nwc.where(nwc.notna(), source_nwc)
    df["nwc"] = nwc
    df["operating_assets"] = (current_assets - cash).clip(lower=0.0) + ppe + goodwill + intangibles + leases
    df["operating_liabilities"] = current_liabilities
    df["cash_excluded"] = cash
    df["goodwill"] = goodwill
    df["intangible_assets"] = intangibles
    df["lease_assets"] = leases
    accounting_ic = nwc + ppe + goodwill + intangibles + leases
    proxy_ic = df["revenue"].astype(float) * 1.25
    low_accounting_ic = accounting_ic < (df["revenue"].astype(float) * 0.05)
    if bool(low_accounting_ic.fillna(False).any()):
        warnings.append(
            f"Accounting invested capital below 5% of revenue in {int(low_accounting_ic.sum())} periods; "
            "proxy IC not used unless DCF_USE_PROXY_INVESTED_CAPITAL=true."
        )
    df["accounting_ic"] = accounting_ic
    df["proxy_ic"] = proxy_ic
    df["proxy_used"] = bool(use_proxy_invested_capital)
    df["normalized_ic"] = proxy_ic.where(low_accounting_ic, accounting_ic) if use_proxy_invested_capital else accounting_ic
    df["ic_used_for_roic"] = df["normalized_ic"]
    df["raw_invested_capital_reconstructed"] = accounting_ic
    df["raw_invested_capital_qoq_change"] = accounting_ic.pct_change()
    _flag_stale_carry_forward(df, warnings)
    return df["ic_used_for_roic"]


def _stock_account(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(0.0, index=df.index)
    raw = pd.to_numeric(df[column], errors="coerce")
    filled = raw.ffill().fillna(0.0)
    age = []
    source_idx = None
    source_periods = []
    for idx, value in enumerate(raw):
        if pd.notna(value):
            source_idx = idx
        age.append((idx - source_idx) if source_idx is not None else None)
        source_periods.append(df.iloc[source_idx]["date"] if source_idx is not None and "date" in df else None)
    df[f"{column}_raw_value"] = raw
    df[f"{column}_carried_forward_value"] = filled
    df[f"{column}_carry_forward_age"] = age
    df[f"{column}_source_period"] = source_periods
    return filled


def _flag_stale_carry_forward(df: pd.DataFrame, warnings: list[str], threshold_quarters: int = 8) -> None:
    for column in ["goodwill", "intangible_assets", "operating_lease_liability"]:
        age_column = f"{column}_carry_forward_age"
        if age_column not in df:
            continue
        stale = pd.to_numeric(df[age_column], errors="coerce") > threshold_quarters
        df[f"{column}_stale_carry_forward_flag"] = stale
        if bool(stale.fillna(False).any()):
            warnings.append(
                f"{column} carried forward more than {threshold_quarters} quarters in {int(stale.sum())} periods."
            )


def _smooth_large_stock_jumps(values: pd.Series, threshold: float) -> pd.Series:
    smoothed = values.astype(float).copy()
    for idx in range(1, len(smoothed)):
        previous = float(smoothed.iloc[idx - 1])
        current = float(smoothed.iloc[idx])
        if previous <= 0.0 or not np.isfinite(previous) or not np.isfinite(current):
            continue
        change = current / previous - 1.0
        if abs(change) > threshold:
            smoothed.iloc[idx] = previous * (1.0 + float(np.clip(change, -threshold, threshold)))
    return smoothed


def _normalized_reinvestment_rate(df: pd.DataFrame) -> pd.Series:
    raw = df["reinvestment_rate"].replace([np.inf, -np.inf], np.nan).astype(float)
    winsor = raw.clip(raw.quantile(0.05), raw.quantile(0.95)) if raw.notna().sum() >= 12 else raw
    median = winsor.rolling(12, min_periods=4).median()
    average = winsor.rolling(8, min_periods=4).mean()
    growth_consistent = _safe_divide(df["revenue_growth"], df["roic"].replace(0.0, np.nan))
    blended = 0.45 * median + 0.35 * average + 0.20 * growth_consistent
    fallback = winsor.expanding(min_periods=1).median()
    return blended.where(blended.notna(), fallback).clip(0.0, 0.85)


def _normalized_roic(df: pd.DataFrame) -> pd.Series:
    raw = df["roic"].replace([np.inf, -np.inf], np.nan).astype(float)
    winsor = raw.clip(raw.quantile(0.05), raw.quantile(0.95)) if raw.notna().sum() >= 12 else raw
    median = winsor.rolling(12, min_periods=4).median()
    average = winsor.rolling(8, min_periods=4).mean()
    blended = 0.65 * median + 0.35 * average
    fallback = winsor.expanding(min_periods=1).median()
    return blended.where(blended.notna(), fallback).clip(0.025, 0.40)
