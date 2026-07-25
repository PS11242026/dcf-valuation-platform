from __future__ import annotations

import numpy as np
import pandas as pd

from dcf_project.config import Settings
from dcf_project.contracts import GrowthForecast
from dcf_project.models.state_space import FCFStateSpaceModel, UnivariateLocalLinearModel
from dcf_project.preprocessing.features import FEATURE_COLUMNS, feature_matrix


def forecast_growth(
    ticker: str,
    features: pd.DataFrame,
    horizon_quarters: int,
    settings: Settings,
) -> GrowthForecast:
    warnings: list[str] = []
    x = feature_matrix(features)
    can_use_multivariate = len(features) >= settings.min_multivariate_periods and np.isfinite(x).all()
    if can_use_multivariate:
        try:
            return _multivariate_forecast(ticker, features, x, horizon_quarters, settings)
        except Exception as exc:
            warnings.append(f"Multivariate model failed and univariate fallback was used: {exc}")
    fallback = _univariate_forecast(ticker, features, horizon_quarters)
    fallback.warnings.extend(warnings)
    return fallback


def reconstruct_fcf_from_states(
    states: np.ndarray,
    base_revenue: float,
    base_fcf: float,
    normalized_capex_ratio: float | None = None,
    maintenance_capex_ratio: float | None = None,
    sbc_ratio: float = 0.0,
    sbc_haircut_factor: float = 1.0,
    settings: Settings | None = None,
) -> tuple[list[float], list[float], list[dict[str, float]]]:
    revenue = float(base_revenue)
    fcf_path: list[float] = []
    growth_path: list[float] = []
    components: list[dict[str, float]] = []
    previous_fcf = float(base_fcf)
    long_run_revenue_growth = float(np.median(states[:, 0])) if len(states) else 0.0
    for quarter, state in enumerate(states, start=1):
        revenue_growth, margin, capex_ratio, nwc_ratio, da_ratio, tax_rate = state
        raw_revenue_growth = float(revenue_growth)
        smoothing_weight = min(0.75, quarter / 20.0)
        smoothed_revenue_growth = (
            (1.0 - smoothing_weight) * raw_revenue_growth
            + smoothing_weight * long_run_revenue_growth
        )
        revenue_growth = float(np.clip(smoothed_revenue_growth, -0.20, 0.30))
        revenue *= 1.0 + revenue_growth
        raw_capex_ratio = float(capex_ratio)
        if normalized_capex_ratio is not None:
            capex_ratio = min(raw_capex_ratio, normalized_capex_ratio)
        ebit = revenue * float(np.clip(margin, -0.5, 0.8))
        nopat = ebit * (1.0 - float(np.clip(tax_rate, 0.0, 0.5)))
        da = revenue * float(np.clip(da_ratio, 0.0, 0.6))
        capex = revenue * float(np.clip(capex_ratio, 0.0, 0.8))
        maintenance_capex = _forecast_maintenance_capex(
            revenue,
            da,
            capex,
            maintenance_capex_ratio,
            settings,
        )
        growth_capex = max(capex - maintenance_capex, 0.0)
        delta_nwc = revenue * float(np.clip(nwc_ratio, -0.8, 0.8))
        reported_fcf = nopat + da - capex - delta_nwc
        economic_fcf = nopat + da - maintenance_capex - delta_nwc
        sbc = revenue * float(np.clip(sbc_ratio, 0.0, 0.5))
        sbc_adjustment = sbc * float(np.clip(sbc_haircut_factor, 0.0, 1.0))
        sbc_addback = sbc - sbc_adjustment
        owner_earnings = economic_fcf + sbc_addback
        fcf = _select_cash_flow(reported_fcf, economic_fcf, owner_earnings, settings)
        growth = (fcf - previous_fcf) / max(abs(previous_fcf), 1.0)
        fcf_path.append(float(fcf))
        growth_path.append(float(np.clip(growth, -2.0, 2.0)))
        cumulative_cagr = _annual_cagr(base_fcf, fcf, quarter)
        components.append(
            {
                "quarter": float(quarter),
                "revenue": float(revenue),
                "revenue_growth": revenue_growth,
                "raw_revenue_growth": raw_revenue_growth,
                "smoothed_revenue_growth": smoothed_revenue_growth,
                "ebit_margin": float(margin),
                "ebit": float(ebit),
                "nopat": float(nopat),
                "da": float(da),
                "da_ratio": float(da_ratio),
                "capex": float(capex),
                "reported_capex": float(capex),
                "maintenance_capex": float(maintenance_capex),
                "growth_capex": float(growth_capex),
                "raw_capex_ratio": raw_capex_ratio,
                "used_capex_ratio": float(capex_ratio),
                "delta_nwc": float(delta_nwc),
                "nwc_ratio": float(nwc_ratio),
                "tax_rate": float(tax_rate),
                "reported_fcf": float(reported_fcf),
                "economic_fcf": float(economic_fcf),
                "owner_earnings": float(owner_earnings),
                "sbc": float(sbc),
                "sbc_adjustment": float(sbc_adjustment),
                "sbc_addback": float(sbc_addback),
                "fcf": float(fcf),
                "fcf_growth": float(np.clip(growth, -2.0, 2.0)),
                "cumulative_cagr": float(cumulative_cagr) if np.isfinite(cumulative_cagr) else float("nan"),
            }
        )
        previous_fcf = fcf
    return fcf_path, growth_path, components


def _multivariate_forecast(
    ticker: str,
    features: pd.DataFrame,
    x: np.ndarray,
    horizon_quarters: int,
    settings: Settings,
) -> GrowthForecast:
    model = FCFStateSpaceModel().fit(x)
    states, covs = model.forecast_state(horizon_quarters)
    base_revenue = float(features.iloc[-1]["revenue"])
    base_fcf = float(features.iloc[-1]["fcf"])
    capex_policy = _capex_policy(features)
    normalized_capex_ratio = capex_policy["normalized_capex_ratio"]
    maintenance_capex_ratio = capex_policy["maintenance_capex_ratio"]
    latest = features.iloc[-1]
    sbc_ratio = float(latest.get("sbc", 0.0) / max(float(latest["revenue"]), 1.0))
    sbc_haircut_factor = float(latest.get("sbc_haircut_factor", 1.0))
    fcf_path, growth_path, components = reconstruct_fcf_from_states(
        states,
        base_revenue,
        base_fcf,
        normalized_capex_ratio if settings.normalize_capex else None,
        maintenance_capex_ratio,
        sbc_ratio,
        sbc_haircut_factor,
        settings,
    )
    std = np.sqrt(np.maximum(np.asarray([np.trace(cov) for cov in covs]), 1e-9))
    lower = [float(np.clip(g - 1.96 * s, -2.0, 2.0)) for g, s in zip(growth_path, std)]
    upper = [float(np.clip(g + 1.96 * s, -2.0, 2.0)) for g, s in zip(growth_path, std)]
    raw_single, collapse_warning = _collapse_to_annual_cagr(base_fcf, fcf_path, horizon_quarters)
    warnings = []
    if collapse_warning:
        warnings.append(collapse_warning)
    if raw_single < -0.30 or raw_single > 0.40:
        warnings.append(
            f"Forecast annual FCF CAGR {raw_single:.2%} is outside mature-company sanity range."
        )
    smoothed_single = _smoothed_growth_metric(growth_path, raw_single)
    emergency_low, emergency_high = -0.50, 0.50
    single = float(np.clip(smoothed_single, emergency_low, emergency_high))
    growth_clamped = not np.isclose(smoothed_single, single, rtol=0.0, atol=1e-12)
    if growth_clamped:
        warnings.append("Emergency descriptive FCF growth clamp triggered.")
    return GrowthForecast(
        ticker=ticker.upper(),
        model_type="multivariate_dlm",
        single_growth_rate=single,
        raw_fcf_growth_rate=float(raw_single),
        smoothed_fcf_growth_rate=float(smoothed_single),
        clamped_fcf_growth_rate=single,
        growth_clamped=bool(growth_clamped),
        clamp_triggered=bool(growth_clamped),
        horizon_quarters=horizon_quarters,
        growth_path=growth_path,
        fcf_path=fcf_path,
        lower_growth_path=lower,
        upper_growth_path=upper,
        diagnostics={
            "feature_columns": FEATURE_COLUMNS,
            "log_likelihood": model.result.log_likelihood if model.result else None,
            "state_mean": model.mu.tolist() if model.mu is not None else None,
            "phi_diagonal": np.diag(model.phi).tolist() if model.phi is not None else None,
            "phi_eigenvalues": np.linalg.eigvals(model.phi).real.tolist() if model.phi is not None else None,
            "process_covariance_diagonal": np.diag(model.q).tolist() if model.q is not None else None,
            "observation_covariance_diagonal": np.diag(model.h).tolist() if model.h is not None else None,
            "filtered_state": model.result.filtered_state.tolist() if model.result else None,
            "forecast_state_path": states.tolist(),
            "forecast_components": components,
            "capex_policy": capex_policy | {"normalization_enabled": settings.normalize_capex},
            "maintenance_capex_method": capex_policy.get("maintenance_capex_method"),
            "maintenance_capex_estimates": capex_policy.get("maintenance_capex_estimates"),
            "valuation_mode": _effective_valuation_mode(settings),
            "terminal_fcf_path": [
                float(row["owner_earnings"] if _effective_valuation_mode(settings) == "owner_earnings" else row["economic_fcf"])
                for row in components
            ],
            "growth_collapse": "annual_cagr_from_base_to_forecast_end_clamped_mature_company",
            "primary_valuation_driver": "full_20_quarter_operating_forecast_path",
        },
        warnings=warnings,
    )


def _univariate_forecast(ticker: str, features: pd.DataFrame, horizon_quarters: int) -> GrowthForecast:
    model = UnivariateLocalLinearModel().fit(features["fcf_growth"].to_numpy())
    growth, variance = model.forecast_growth(horizon_quarters)
    previous_fcf = float(features.iloc[-1]["fcf"])
    fcf_path = []
    for g in growth:
        previous_fcf = previous_fcf * (1.0 + float(g))
        fcf_path.append(float(previous_fcf))
    std = np.sqrt(variance)
    raw_single = _annual_cagr(float(features.iloc[-1]["fcf"]), fcf_path[-1], horizon_quarters)
    smoothed_single = _smoothed_growth_metric([float(x) for x in growth], raw_single)
    single = float(np.clip(smoothed_single, -0.50, 0.50))
    return GrowthForecast(
        ticker=ticker.upper(),
        model_type="univariate_local_linear",
        single_growth_rate=single,
        raw_fcf_growth_rate=float(raw_single),
        smoothed_fcf_growth_rate=float(smoothed_single),
        clamped_fcf_growth_rate=single,
        growth_clamped=bool(not np.isclose(smoothed_single, single, rtol=0.0, atol=1e-12)),
        clamp_triggered=bool(not np.isclose(smoothed_single, single, rtol=0.0, atol=1e-12)),
        horizon_quarters=horizon_quarters,
        growth_path=[float(x) for x in growth],
        fcf_path=fcf_path,
        lower_growth_path=[float(np.clip(g - 1.96 * s, -2.0, 2.0)) for g, s in zip(growth, std)],
        upper_growth_path=[float(np.clip(g + 1.96 * s, -2.0, 2.0)) for g, s in zip(growth, std)],
        diagnostics={
            "level": model.level,
            "trend": model.trend,
            "variance": model.variance,
            "valuation_mode": "reported_univariate_fallback",
            "growth_collapse": "annual_cagr_from_base_to_forecast_end_clamped_mature_company",
            "primary_valuation_driver": "full_20_quarter_operating_forecast_path",
        },
        warnings=["Used univariate fallback model because multivariate history was insufficient."],
    )


def _collapse_to_annual_cagr(
    base_fcf: float,
    fcf_path: list[float],
    horizon_quarters: int,
) -> tuple[float, str | None]:
    if base_fcf <= 0 or not fcf_path or fcf_path[-1] <= 0:
        return float(np.median(fcf_path) / max(abs(base_fcf), 1.0) - 1.0), (
            "FCF CAGR fallback used because base or ending FCF was non-positive."
        )
    return _annual_cagr(base_fcf, fcf_path[-1], horizon_quarters), None


def _annual_cagr(base_fcf: float, ending_fcf: float, horizon_quarters: int) -> float:
    years = max(horizon_quarters / 4.0, 0.25)
    if base_fcf <= 0 or ending_fcf <= 0:
        return float("nan")
    return float((ending_fcf / base_fcf) ** (1.0 / years) - 1.0)


def _smoothed_growth_metric(growth_path: list[float], raw_cagr: float) -> float:
    clean = np.asarray([g for g in growth_path if np.isfinite(g)], dtype=float)
    if clean.size == 0:
        return raw_cagr
    median_quarterly = float(np.median(clean))
    annualized_median = (1.0 + np.clip(median_quarterly, -0.75, 1.0)) ** 4 - 1.0
    if not np.isfinite(raw_cagr):
        return annualized_median
    return float(0.5 * raw_cagr + 0.5 * annualized_median)


def _capex_policy(features: pd.DataFrame) -> dict[str, float]:
    capex = features["capex_ratio"].dropna().astype(float)
    if capex.empty:
        return {
            "weighted_last_12_capex_ratio": 0.0,
            "maintenance_capex_ratio": 0.0,
            "normalized_capex_ratio": 0.0,
            "maintenance_capex_method": "unknown",
            "maintenance_capex_estimates": {},
        }
    last = capex.tail(12)
    weights = np.arange(1, len(last) + 1, dtype=float)
    weighted_last_12 = float(np.average(last, weights=weights))
    if "maintenance_capex_ratio" in features:
        maintenance = float(features["maintenance_capex_ratio"].dropna().astype(float).tail(12).median())
    else:
        maintenance = float(capex.median())
    normalized = float(min(weighted_last_12, maintenance * 1.25))
    return {
        "weighted_last_12_capex_ratio": weighted_last_12,
        "maintenance_capex_ratio": maintenance,
        "normalized_capex_ratio": normalized,
        "maintenance_capex_method": str(features["maintenance_capex_method"].iloc[-1])
        if "maintenance_capex_method" in features
        else "legacy",
        "maintenance_capex_estimates": _latest_maintenance_estimates(features),
    }


def _forecast_maintenance_capex(
    revenue: float,
    da: float,
    capex: float,
    maintenance_capex_ratio: float | None,
    settings: Settings | None,
) -> float:
    if settings is None:
        floor = revenue * (settings.maintenance_capex_sector_floor_ratio if settings else 0.02)
        return float(min(max(da, floor), capex))
    ratio = maintenance_capex_ratio
    if ratio is None or not np.isfinite(ratio):
        ratio = settings.maintenance_capex_sector_floor_ratio
    floor = settings.maintenance_capex_sector_floor_ratio
    return float(min(revenue * max(ratio, floor), capex))


def _latest_maintenance_estimates(features: pd.DataFrame) -> dict[str, float]:
    latest = features.iloc[-1]
    estimates = {}
    for name in [
        "depreciation_anchor",
        "growth_adjusted",
        "historical_lower_band",
        "asset_turnover",
    ]:
        value = latest.get(f"maintenance_capex_{name}_ratio")
        if value is not None and np.isfinite(value):
            estimates[name] = float(value)
    return estimates


def _select_cash_flow(
    reported_fcf: float,
    economic_fcf: float,
    owner_earnings: float,
    settings: Settings | None,
) -> float:
    mode = _effective_valuation_mode(settings)
    if mode == "owner_earnings":
        return owner_earnings
    if mode == "economic":
        return economic_fcf
    return reported_fcf


def _effective_valuation_mode(settings: Settings | None) -> str:
    if settings is None:
        return "reported"
    if settings.valuation_mode in {"owner_earnings", "owner_earnings_dcf"} or (
        settings.valuation_mode == "auto" and settings.company_type == "marketplace_hybrid"
    ):
        return "owner_earnings"
    if settings.valuation_mode in {"economic", "economic_dcf"} or (
        settings.valuation_mode == "auto" and settings.company_type in {"hyperscaler", "platform"}
    ):
        return "economic"
    return "reported"
