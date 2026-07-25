from __future__ import annotations

import numpy as np
import pandas as pd

from dcf_project.config import Settings, get_settings
from dcf_project.contracts import FeatureFrame, FinancialPeriod


FEATURE_COLUMNS = [
    "revenue_growth",
    "ebit_margin",
    "capex_ratio",
    "nwc_ratio",
    "da_ratio",
    "tax_rate",
]


class FeatureError(RuntimeError):
    pass


def build_features(
    periods: list[FinancialPeriod],
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, FeatureFrame]:
    settings = settings or get_settings()
    if len(periods) < 5:
        raise FeatureError("At least five financial periods are required to build growth features")
    ticker = periods[0].ticker.upper()
    df = pd.DataFrame([p.model_dump() for p in periods]).sort_values("report_date")
    df = df.drop_duplicates(subset=["report_date"], keep="last").reset_index(drop=True)
    warnings: list[str] = []
    _warn_irregular_quarters(df, warnings)

    required = ["revenue", "ebit", "capex", "da", "tax_expense", "nwc"]
    numeric_columns = required + [
        "sbc",
        "operating_lease_liability",
        "operating_lease_cost",
        "ppe",
        "goodwill",
        "intangible_assets",
        "cash",
        "debt",
        "shares_outstanding",
        "current_assets",
        "current_liabilities",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    missing_counts = {col: int(df[col].isna().sum()) for col in required if col in df}
    for col, count in missing_counts.items():
        if count:
            warnings.append(f"{col} missing in {count} periods.")

    df["revenue_growth"] = df["revenue"].pct_change()
    df["ebit_margin"] = _safe_divide(df["ebit"], df["revenue"])
    df["capex_ratio"] = _safe_divide(df["capex"].abs(), df["revenue"])
    df["da_ratio"] = _safe_divide(df["da"], df["revenue"])
    df["tax_rate"] = _safe_divide(df["tax_expense"], df["ebit"].abs())
    df["delta_nwc"] = df["nwc"].diff()
    df["nwc_ratio"] = _safe_divide(df["delta_nwc"], df["revenue"])
    df["nopat"] = df["ebit"] * (1.0 - df["tax_rate"].clip(0.0, 0.5))
    _add_capex_decomposition(df, settings, ticker)
    df["sbc"] = df["sbc"].fillna(0.0) if "sbc" in df else 0.0
    df["sbc_haircut_factor"] = _sbc_haircut_factor(ticker, settings)
    df["sbc_adjustment"] = df["sbc"] * df["sbc_haircut_factor"]
    df["sbc_addback"] = df["sbc"] - df["sbc_adjustment"]
    df["lease_adjustment"] = (
        df["operating_lease_cost"].fillna(0.0) if "operating_lease_cost" in df else 0.0
    )
    df["reported_fcf"] = df["nopat"] + df["da"] - df["reported_capex"] - df["delta_nwc"].fillna(0.0)
    df["economic_fcf"] = df["nopat"] + df["da"] - df["maintenance_capex"] - df["delta_nwc"].fillna(0.0)
    df["owner_earnings"] = (
        df["nopat"]
        + df["da"]
        + df["sbc_addback"]
        - df["maintenance_capex"]
        - df["delta_nwc"].fillna(0.0)
    )
    df["reported_fcf_margin"] = _safe_divide(df["reported_fcf"], df["revenue"])
    df["economic_fcf_margin"] = _safe_divide(df["economic_fcf"], df["revenue"])
    df["owner_earnings_margin"] = _safe_divide(df["owner_earnings"], df["revenue"])
    _add_invested_capital_metrics(df, settings)
    df["fcf"] = df[_selected_fcf_column(settings)]
    df["fcf_growth"] = df["fcf"].diff() / df["fcf"].shift(1).abs()

    for col, low, high in FEATURE_BOUNDS:
        before = df[col].copy()
        percentile_low, percentile_high = _winsor_bounds(df[col], low, high)
        df[col] = df[col].clip(percentile_low, percentile_high)
        changed = int((before.notna() & (before != df[col])).sum())
        if changed:
            warnings.append(
                f"{col} winsorized in {changed} periods "
                f"to [{percentile_low:.4f}, {percentile_high:.4f}]."
            )

    usable = df.dropna(subset=FEATURE_COLUMNS + ["fcf", "revenue"]).copy()
    if len(usable) < 4:
        raise FeatureError("Insufficient complete periods after feature engineering")
    if len(usable) < len(df):
        warnings.append(f"Dropped {len(df) - len(usable)} incomplete periods before modeling.")

    rows = _jsonable_rows(usable)
    return usable, FeatureFrame(ticker=ticker, rows=rows, warnings=warnings)


FEATURE_BOUNDS = [
    ("revenue_growth", -0.7, 1.5),
    ("ebit_margin", -0.5, 0.8),
    ("capex_ratio", 0.0, 0.8),
    ("nwc_ratio", -0.8, 0.8),
    ("da_ratio", 0.0, 0.6),
    ("tax_rate", 0.0, 0.5),
    ("fcf_growth", -2.0, 2.0),
]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].astype(float).to_numpy()


def latest_base_values(df: pd.DataFrame) -> dict[str, float]:
    latest = df.iloc[-1]
    return {
        "revenue": float(latest["revenue"]),
        "fcf": float(latest["fcf"]),
        "reported_fcf": float(latest["reported_fcf"]),
        "economic_fcf": float(latest["economic_fcf"]),
        "owner_earnings": float(latest["owner_earnings"]),
        "reported_capex": float(latest["reported_capex"]),
        "maintenance_capex": float(latest["maintenance_capex"]),
        "growth_capex": float(latest["growth_capex"]),
        "operating_lease_liability": _last_valid(df, "operating_lease_liability"),
        "invested_capital": float(latest["invested_capital"]),
        "roic": float(latest["roic"]),
        "reinvestment_rate": float(latest["reinvestment_rate"]),
        "cash": _last_valid(df, "cash"),
        "debt": _last_valid(df, "debt"),
        "shares_outstanding": _last_valid(df, "shares_outstanding"),
    }


def _selected_fcf_column(settings: Settings) -> str:
    if settings.valuation_mode in {"owner_earnings", "owner_earnings_dcf"} or (
        settings.valuation_mode == "auto" and settings.company_type == "marketplace_hybrid"
    ):
        return "owner_earnings"
    if settings.valuation_mode in {"economic", "economic_dcf"} or (
        settings.valuation_mode == "auto" and settings.company_type in {"hyperscaler", "platform"}
    ):
        return "economic_fcf"
    return "reported_fcf"


def _add_invested_capital_metrics(df: pd.DataFrame, settings: Settings) -> None:
    revenue = df["revenue"].astype(float)
    nwc = df["nwc"].astype(float).fillna(0.0)
    ppe = df["ppe"].astype(float) if "ppe" in df else pd.Series(np.nan, index=df.index)
    intangibles = (
        (df["goodwill"].fillna(0.0) if "goodwill" in df else 0.0)
        + (df["intangible_assets"].fillna(0.0) if "intangible_assets" in df else 0.0)
    )
    leases = (
        df["operating_lease_liability"].fillna(0.0)
        if "operating_lease_liability" in df and settings.normalize_leases
        else 0.0
    )
    turnover = _invested_capital_turnover(settings)
    proxy_ic = revenue / turnover
    accounting_ic = nwc.clip(lower=0.0) + ppe.fillna(0.0) + intangibles + leases
    df["invested_capital"] = accounting_ic.where(accounting_ic > revenue * 0.05, proxy_ic)
    df["invested_capital"] = df["invested_capital"].clip(lower=revenue * 0.03)
    df["roic"] = _safe_divide(df["nopat"] * 4.0, df["invested_capital"])
    if "growth_capex" in df:
        df["reinvestment"] = df["growth_capex"].clip(lower=0.0) + df["delta_nwc"].fillna(0.0).clip(lower=0.0)
    else:
        net_capex = df["capex"].abs() - df["da"].fillna(0.0)
        df["reinvestment"] = net_capex.clip(lower=0.0) + df["delta_nwc"].fillna(0.0).clip(lower=0.0)
    df["reinvestment_rate"] = _safe_divide(df["reinvestment"], df["nopat"].abs()).clip(0.0, 0.90)
    df["growth_from_roic"] = (df["reinvestment_rate"] * df["roic"]).clip(-0.20, 0.40)


def _invested_capital_turnover(settings: Settings) -> float:
    defaults = {
        "consumer_staple": 3.0,
        "platform": 2.8,
        "hyperscaler": 1.8,
        "marketplace_hybrid": 2.2,
        "industrial": 1.4,
        "telecom": 0.8,
        "semiconductor": 1.2,
        "asset_light": 2.5,
    }
    return defaults.get(settings.company_type, 2.0)


def _sbc_haircut_factor(ticker: str, settings: Settings) -> float:
    if settings.sbc_mode == "ignore":
        return 0.0
    if settings.sbc_mode == "full_expense":
        return 1.0
    if settings.sbc_haircut_factor is not None:
        return float(np.clip(settings.sbc_haircut_factor, 0.0, 1.0))
    overrides = {"AMZN": 0.65, "META": 0.80, "GOOGL": 0.75, "SNOW": 1.00, "UBER": 0.85}
    if ticker.upper() in overrides:
        return overrides[ticker.upper()]
    defaults = {
        "marketplace_hybrid": 0.65,
        "hyperscaler": 0.75,
        "platform": 0.80,
        "asset_light": 0.70,
        "semiconductor": 0.75,
    }
    return defaults.get(settings.company_type, 1.0)


def _add_capex_decomposition(df: pd.DataFrame, settings: Settings, ticker: str) -> None:
    reported_capex = df["capex"].abs().astype(float)
    floor = df["revenue"].astype(float) * settings.maintenance_capex_sector_floor_ratio
    estimates = _maintenance_capex_estimates(df, settings, ticker)
    method = settings.maintenance_capex_method
    if method == "ensemble":
        maintenance = pd.DataFrame(estimates).median(axis=1).combine(floor, max)
    elif method == "depreciation_anchor":
        maintenance = estimates["depreciation_anchor"].combine(floor, max)
    elif method == "growth_adjusted":
        maintenance = estimates["growth_adjusted"].combine(floor, max)
    elif method == "percentile":
        maintenance = estimates["historical_lower_band"].combine(floor, max)
    elif method == "asset_turnover":
        maintenance = estimates["asset_turnover"].combine(floor, max)
    else:
        raise ValueError(f"Unsupported maintenance capex method: {method}")
    df["reported_capex"] = reported_capex
    df["maintenance_capex"] = maintenance.fillna(reported_capex).clip(upper=reported_capex)
    df["growth_capex"] = (reported_capex - df["maintenance_capex"]).clip(lower=0.0)
    df["maintenance_capex_ratio"] = _safe_divide(df["maintenance_capex"], df["revenue"])
    df["growth_capex_ratio"] = _safe_divide(df["growth_capex"], df["revenue"])
    for name, values in estimates.items():
        df[f"maintenance_capex_{name}"] = values.fillna(df["maintenance_capex"]).clip(upper=reported_capex)
        df[f"maintenance_capex_{name}_ratio"] = _safe_divide(
            df[f"maintenance_capex_{name}"],
            df["revenue"],
        )
    df["maintenance_capex_method"] = method


def _maintenance_capex_estimates(
    df: pd.DataFrame,
    settings: Settings,
    ticker: str,
) -> dict[str, pd.Series]:
    revenue = df["revenue"].astype(float)
    capex = df["capex"].abs().astype(float)
    capex_ratio = df["capex_ratio"].astype(float)
    da_ratio = df["da_ratio"].astype(float)
    calibration_floor = revenue * _calibration_min_maintenance_ratio(ticker, settings)
    depreciation_anchor = df["da"].astype(float).rolling(12, min_periods=1).mean()
    growth_adjusted = capex * (1.0 - _growth_investment_factor(df, ticker, settings))
    historical_lower_band = revenue * _lower_band_capex_ratio(capex_ratio, settings)
    asset_turnover = revenue * _asset_turnover_maintenance_ratio(capex_ratio, da_ratio, ticker, settings)
    return {
        "depreciation_anchor": depreciation_anchor.combine(calibration_floor, max),
        "growth_adjusted": growth_adjusted.combine(calibration_floor, max),
        "historical_lower_band": historical_lower_band.combine(calibration_floor, max),
        "asset_turnover": asset_turnover.combine(calibration_floor, max),
    }


def _growth_investment_factor(df: pd.DataFrame, ticker: str, settings: Settings) -> pd.Series:
    growth = df["revenue_growth"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    acceleration = growth.diff().fillna(0.0).clip(lower=0.0)
    normalized = (growth.clip(lower=0.0) + acceleration).rolling(4, min_periods=1).mean()
    return (normalized / 0.30).clip(lower=0.0, upper=1.0) * _growth_investment_factor_cap(
        ticker,
        settings,
    )


def _lower_band_capex_ratio(capex_ratio: pd.Series, settings: Settings) -> float:
    clean = capex_ratio.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return settings.maintenance_capex_sector_floor_ratio
    q25 = float(clean.quantile(0.25))
    q40 = float(clean.quantile(0.40))
    return 0.5 * (q25 + q40)


def _asset_turnover_maintenance_ratio(
    capex_ratio: pd.Series,
    da_ratio: pd.Series,
    ticker: str,
    settings: Settings,
) -> float:
    clean_capex = capex_ratio.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    clean_da = da_ratio.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean_capex.empty:
        return settings.maintenance_capex_sector_floor_ratio
    da_component = float(clean_da.median()) if not clean_da.empty else float(clean_capex.median())
    sustaining_ratio = 0.65 * float(clean_capex.median()) + 0.35 * da_component
    return max(
        sustaining_ratio * _maintenance_burden_multiplier(ticker, settings),
        settings.maintenance_capex_sector_floor_ratio,
    )


def _calibration_min_maintenance_ratio(ticker: str, settings: Settings) -> float:
    overrides = {"AMZN": 0.060, "GOOGL": 0.050, "META": 0.070, "MSFT": 0.045}
    if ticker.upper() in overrides:
        return overrides[ticker.upper()]
    defaults = {
        "hyperscaler": 0.055,
        "platform": 0.060,
        "semiconductor": 0.055,
        "telecom": 0.100,
        "industrial": 0.045,
        "consumer_staple": 0.030,
        "asset_light": 0.025,
    }
    return defaults.get(settings.company_type, settings.maintenance_capex_sector_floor_ratio)


def _growth_investment_factor_cap(ticker: str, settings: Settings) -> float:
    overrides = {"AMZN": 0.35, "GOOGL": 0.40, "META": 0.25, "MSFT": 0.35}
    if ticker.upper() in overrides:
        return overrides[ticker.upper()]
    defaults = {
        "hyperscaler": 0.35,
        "platform": 0.25,
        "semiconductor": 0.30,
        "consumer_staple": 0.05,
        "industrial": 0.10,
        "telecom": 0.05,
    }
    return defaults.get(settings.company_type, 0.15)


def _maintenance_burden_multiplier(ticker: str, settings: Settings) -> float:
    overrides = {"AMZN": 1.15, "GOOGL": 1.00, "META": 1.20, "MSFT": 1.00}
    if ticker.upper() in overrides:
        return overrides[ticker.upper()]
    defaults = {
        "hyperscaler": 1.05,
        "platform": 1.15,
        "semiconductor": 1.10,
        "telecom": 1.25,
        "consumer_staple": 1.00,
        "industrial": 1.00,
    }
    return defaults.get(settings.company_type, 1.00)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype(float).replace(0.0, np.nan)
    denom = denom.mask(denom.abs() < 1e-9)
    return numerator.astype(float) / denom


def _winsor_bounds(series: pd.Series, hard_low: float, hard_high: float) -> tuple[float, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(clean) < 10:
        return hard_low, hard_high
    q_low = float(clean.quantile(0.01))
    q_high = float(clean.quantile(0.99))
    return max(hard_low, q_low), min(hard_high, q_high)


def _warn_irregular_quarters(df: pd.DataFrame, warnings: list[str]) -> None:
    dates = pd.to_datetime(df["report_date"], errors="coerce").dropna()
    if len(dates) < 2:
        return
    gaps = dates.diff().dt.days.dropna()
    irregular = int(((gaps < 70) | (gaps > 115)).sum())
    if irregular:
        warnings.append(f"Detected {irregular} irregular report-date gaps before feature filtering.")


def _last_valid(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _jsonable_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if hasattr(value, "isoformat"):
                clean[key] = value.isoformat()
            elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                clean[key] = None
            else:
                clean[key] = value
        rows.append(clean)
    return rows
