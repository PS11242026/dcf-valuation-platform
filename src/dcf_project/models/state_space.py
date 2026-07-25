from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcf_project.models.kalman import KalmanResult, forecast_states, kalman_filter
from dcf_project.models.priors import prior_for
from dcf_project.preprocessing.reconstruction import STATE_COLUMNS


@dataclass
class StateSpaceForecast:
    states: np.ndarray
    covariances: np.ndarray
    financials: pd.DataFrame
    fcf_growth_rate: float


@dataclass
class FCFStateSpaceModel:
    mu: np.ndarray | None = None
    phi: np.ndarray | None = None
    q: np.ndarray | None = None
    r: np.ndarray | None = None
    z: np.ndarray | None = None
    result: KalmanResult | None = None
    raw_mu: np.ndarray | None = None
    standardization_center: np.ndarray | None = None
    standardization_scale: np.ndarray | None = None
    empirical_phi: np.ndarray | None = None
    prior_influence: dict[str, object] | None = None
    state_columns: tuple[str, ...] = tuple(STATE_COLUMNS)
    company_type: str = "asset_light"
    calibration_mode: str = "base"
    ticker: str | None = None

    @property
    def h(self) -> np.ndarray | None:
        return self.r

    def fit(self, x: np.ndarray) -> "FCFStateSpaceModel":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.state_columns):
            raise ValueError(f"Expected a matrix with {len(self.state_columns)} state columns")
        if len(x) < 8:
            raise ValueError("At least eight observations are required")
        x = _interpolate_missing(x)
        # Long-run reversion anchor. Defaults to the full-history median (production
        # behavior). DCF_ANCHOR_METHOD selects an alternative anchor for the
        # structural-break research study WITHOUT changing the default; only the mu
        # reversion target is affected (standardization center stays the median).
        raw_mu = _state_anchor(x, self.company_type, self.ticker)
        self.raw_mu = raw_mu
        self.mu = _apply_long_run_mean_priors(
            raw_mu,
            self.company_type,
            self.calibration_mode,
            self.ticker,
        )
        self.standardization_center = np.nanmedian(x, axis=0)
        scale = np.nanstd(x, axis=0)
        self.standardization_scale = np.where(scale > 1e-6, scale, 1.0)
        x_std = (x - self.standardization_center) / self.standardization_scale
        mu_std = (self.mu - self.standardization_center) / self.standardization_scale
        centered = x_std - mu_std
        self.empirical_phi = _estimate_transition(centered)
        self.phi = _apply_company_type_priors(
            self.empirical_phi,
            self.company_type,
            self.calibration_mode,
            self.ticker,
        )
        residuals = centered[1:] - centered[:-1] @ self.phi.T
        residuals = residuals[np.isfinite(residuals).all(axis=1)]
        empirical_var = np.maximum(np.nanvar(x_std, axis=0), 1e-6)
        q = np.cov(residuals, rowvar=False) if len(residuals) >= 2 else np.diag(empirical_var * 0.25)
        q_std = _nearest_psd(q, floor=1e-7)
        r_std = _nearest_psd(np.diag(empirical_var * 0.10), floor=1e-7)
        self.q = _unscale_covariance(q_std, self.standardization_scale)
        self.r = _unscale_covariance(r_std, self.standardization_scale)
        self.z = np.eye(len(self.state_columns))
        result_std = kalman_filter(x_std, mu_std, self.phi, q_std, r_std, self.z)
        self.result = _unscale_kalman_result(result_std, self.standardization_center, self.standardization_scale)
        self.prior_influence = _prior_influence_trace(
            self.state_columns,
            raw_mu,
            self.mu,
            self.empirical_phi,
            self.phi,
            self.standardization_center,
            self.standardization_scale,
        )
        return self

    def filter(self) -> KalmanResult:
        if self.result is None:
            raise RuntimeError("Model must be fit before filtering")
        return self.result

    def smooth(self) -> tuple[np.ndarray, np.ndarray]:
        if self.result is None:
            raise RuntimeError("Model must be fit before smoothing")
        return self.result.smoothed_states, self.result.smoothed_covariances

    def forecast(self, steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
        if self.result is None or self.mu is None or self.phi is None or self.q is None:
            raise RuntimeError("Model must be fit before forecasting")
        return forecast_states(
            self.result.filtered_state,
            self.result.filtered_cov,
            self.mu,
            self.phi,
            self.q,
            steps,
        )

    def forecast_state(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        return self.forecast(horizon)

    def forecast_financials(
        self,
        base_revenue: float,
        base_fcf: float,
        steps: int = 20,
        base_invested_capital: float | None = None,
        tax_rate: float = 0.21,
    ) -> StateSpaceForecast:
        states, covs = self.forecast(steps)
        financials = financials_from_states(
            states=states,
            base_revenue=base_revenue,
            base_fcf=base_fcf,
            base_invested_capital=base_invested_capital,
            tax_rate=tax_rate,
            company_type=self.company_type,
            calibration_mode=self.calibration_mode,
            ticker=self.ticker,
            roic_center=float(self.mu[3]) if self.mu is not None else None,
        )
        growth = model_derived_fcf_growth(float(base_fcf), financials["owner_earnings"].to_numpy(float), steps)
        return StateSpaceForecast(states=states, covariances=covs, financials=financials, fcf_growth_rate=growth)


@dataclass
class UnivariateLocalLinearModel:
    level: float | None = None
    trend: float | None = None
    variance: float | None = None

    def fit(self, y: np.ndarray) -> "UnivariateLocalLinearModel":
        clean = np.asarray(y, dtype=float)
        clean = clean[np.isfinite(clean)]
        if len(clean) < 3:
            raise ValueError("At least three FCF growth observations are required")
        self.level = float(np.median(clean[-4:]))
        self.trend = float(np.median(np.diff(clean[-5:]))) if len(clean) >= 5 else 0.0
        self.variance = float(max(np.var(clean - np.median(clean)), 1e-4))
        return self

    def forecast_growth(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        if self.level is None or self.trend is None or self.variance is None:
            raise RuntimeError("Model must be fit before forecasting")
        steps = np.arange(1, horizon + 1, dtype=float)
        growth = np.clip(self.level + steps * self.trend, -0.5, 0.5)
        variance = np.full(horizon, self.variance)
        return growth, variance


def financials_from_states(
    states: np.ndarray,
    base_revenue: float,
    base_fcf: float,
    base_invested_capital: float | None = None,
    tax_rate: float = 0.21,
    company_type: str = "asset_light",
    accounting_policy: dict[str, float] | None = None,
    calibration_mode: str = "base",
    ticker: str | None = None,
    roic_center: float | None = None,
) -> pd.DataFrame:
    revenue = float(base_revenue)
    invested_capital = float(base_invested_capital or max(base_revenue * 1.25, 1.0))
    previous_fcf = float(base_fcf)
    tax_rate = float(np.clip(tax_rate, 0.0, 0.35))
    policy = _accounting_policy_defaults(company_type, calibration_mode, ticker) | (accounting_policy or {})
    rows: list[dict[str, float]] = []
    for quarter, raw_state in enumerate(np.asarray(states, dtype=float), start=1):
        bounded = _bound_state_with_trace(
            raw_state,
            company_type,
            calibration_mode,
            ticker,
            roic_center,
        )
        revenue_growth, margin, reinvestment_rate, roic = bounded["values"]
        clamp_flags = bounded["clamp_flags"]
        fallback_triggered = False
        if company_type == "marketplace_hybrid" and (not np.isfinite(reinvestment_rate) or reinvestment_rate > 0.95):
            reinvestment_rate = _marketplace_reinvestment_path(quarter)
            fallback_triggered = True
        identity_growth = roic * reinvestment_rate
        revenue_growth = float(0.65 * revenue_growth + 0.35 * identity_growth)
        revenue_growth = float(np.clip(revenue_growth, -0.08, _growth_cap(company_type, calibration_mode, ticker)))
        revenue *= 1.0 + revenue_growth
        ebit = revenue * margin
        nopat = ebit * (1.0 - tax_rate)
        r_and_d = revenue * policy["r_and_d_ratio"]
        r_and_d_capitalized = r_and_d * policy["r_and_d_capitalization_factor"]
        nopat_adjusted = nopat + r_and_d_capitalized * (1.0 - tax_rate)
        reinvestment = nopat * reinvestment_rate
        invested_capital = max(invested_capital + reinvestment + r_and_d_capitalized, revenue * 0.15)
        economic_fcf = nopat_adjusted * (1.0 - reinvestment_rate) - r_and_d_capitalized
        if nopat <= 0.0 and company_type == "marketplace_hybrid":
            owner_earnings_margin = max(margin, 0.04)
            economic_fcf = revenue * owner_earnings_margin * 0.35
            reinvestment = nopat - economic_fcf
            fallback_triggered = True
        da = revenue * policy["da_ratio"]
        delta_nwc = revenue * policy["delta_nwc_ratio"]
        capex = revenue * policy["capex_ratio"]
        maintenance_capex = min(capex, revenue * policy["maintenance_capex_ratio"])
        sbc = revenue * policy["sbc_ratio"]
        sbc_adjustment = sbc * policy["sbc_haircut"]
        accounting_fcf = nopat + da - capex - delta_nwc
        owner_earnings = nopat_adjusted + da - maintenance_capex - delta_nwc - sbc_adjustment - r_and_d_capitalized
        fcf = economic_fcf
        rows.append(
            {
                "quarter": float(quarter),
                "revenue_growth": revenue_growth,
                "raw_model_revenue_growth": float(raw_state[0]),
                "revenue_growth_clamped": float(clamp_flags["revenue_growth"]),
                "identity_growth": identity_growth,
                "revenue": revenue,
                "operating_margin": margin,
                "raw_model_operating_margin": float(raw_state[1]),
                "operating_margin_clamped": float(clamp_flags["operating_margin"]),
                "ebit_margin": margin,
                "ebit": ebit,
                "tax_rate": tax_rate,
                "nopat": nopat,
                "nopat_adjusted": nopat_adjusted,
                "r_and_d": r_and_d,
                "r_and_d_capitalized": r_and_d_capitalized,
                "reinvestment_rate": reinvestment_rate,
                "raw_reinvestment": float(raw_state[2]),
                "raw_model_reinvestment_rate": float(raw_state[2]),
                "normalized_reinvestment": reinvestment_rate,
                "effective_reinvestment": reinvestment_rate,
                "reinvestment_rate_clamped": float(clamp_flags["reinvestment_rate"]),
                "reinvestment": reinvestment,
                "invested_capital": invested_capital,
                "roic": roic,
                "raw_model_roic": float(raw_state[3]),
                "roic_clamped": float(clamp_flags["roic"]),
                "annualized_roic": roic * 4.0,
                "accounting_fcf": accounting_fcf,
                "economic_fcf": economic_fcf,
                "owner_earnings": owner_earnings,
                "maintenance_capex": maintenance_capex,
                "sbc": sbc,
                "sbc_adjustment": sbc_adjustment,
                "fcf": fcf,
                "fcf_growth": (fcf - previous_fcf) / max(abs(previous_fcf), 1.0),
                "fallback_triggered": float(fallback_triggered),
                "da": da,
                "capex": capex,
                "delta_nwc": delta_nwc,
            }
        )
        previous_fcf = fcf
    return pd.DataFrame(rows)


def _accounting_policy_defaults(
    company_type: str,
    calibration_mode: str = "base",
    ticker: str | None = None,
) -> dict[str, float]:
    prior = prior_for(ticker, company_type)
    policies = {
        "asset_light_quality": {
            "da_ratio": 0.035,
            "capex_ratio": 0.025,
            "maintenance_capex_ratio": 0.020,
            "delta_nwc_ratio": 0.002,
            "sbc_ratio": 0.010,
            "sbc_haircut": 0.50,
            "r_and_d_ratio": prior.default_rd_ratio,
            "r_and_d_capitalization_factor": 0.70,
        },
        "hyperscaler": {
            "da_ratio": 0.055,
            "capex_ratio": 0.075,
            "maintenance_capex_ratio": 0.035,
            "delta_nwc_ratio": 0.002,
            "sbc_ratio": 0.025,
            "sbc_haircut": 0.65,
            "r_and_d_ratio": prior.default_rd_ratio,
            "r_and_d_capitalization_factor": 0.75,
        },
        "platform": {
            "da_ratio": 0.045,
            "capex_ratio": 0.055,
            "maintenance_capex_ratio": 0.030,
            "delta_nwc_ratio": 0.001,
            "sbc_ratio": 0.035,
            "sbc_haircut": 0.70,
            "r_and_d_ratio": prior.default_rd_ratio,
            "r_and_d_capitalization_factor": 0.75,
        },
        "marketplace_hybrid": {
            "da_ratio": 0.040,
            "capex_ratio": 0.065,
            "maintenance_capex_ratio": 0.025,
            "delta_nwc_ratio": 0.003,
            "sbc_ratio": 0.020,
            "sbc_haircut": 0.60,
            "r_and_d_ratio": prior.default_rd_ratio,
            "r_and_d_capitalization_factor": 0.60,
        },
        "consumer_compounder": {
            "da_ratio": 0.018,
            "capex_ratio": 0.018,
            "maintenance_capex_ratio": 0.012,
            "delta_nwc_ratio": 0.001,
            "sbc_ratio": 0.003,
            "sbc_haircut": 1.00,
            "r_and_d_ratio": prior.default_rd_ratio,
            "r_and_d_capitalization_factor": 0.30,
        },
    }
    policy = policies.get(
        company_type,
        {
            "da_ratio": 0.030,
            "capex_ratio": 0.035,
            "maintenance_capex_ratio": 0.020,
            "delta_nwc_ratio": 0.002,
            "sbc_ratio": 0.010,
            "sbc_haircut": 0.75,
            "r_and_d_ratio": prior.default_rd_ratio,
            "r_and_d_capitalization_factor": 0.50,
        },
    )
    if calibration_mode == "consensus_calibrated":
        policy = dict(policy)
        policy["capex_ratio"] *= 0.90
        policy["maintenance_capex_ratio"] *= 0.90
        policy["sbc_haircut"] *= 0.80
    return policy


def model_derived_fcf_growth(base_fcf: float, projected_fcf: np.ndarray, horizon_quarters: int = 20) -> float:
    _ = base_fcf, horizon_quarters
    projected_fcf = np.asarray(projected_fcf, dtype=float)
    if projected_fcf.size < 8:
        return float("nan")
    starting_base = float(np.nanmean(projected_fcf[:4]))
    ending_base = float(np.nanmean(projected_fcf[-4:]))
    if not np.isfinite(starting_base) or not np.isfinite(ending_base) or starting_base <= 0.0 or ending_base <= 0.0:
        return float("nan")
    return float((ending_base / starting_base) ** 0.25 - 1.0)


def _bound_state(
    state: np.ndarray,
    company_type: str,
    calibration_mode: str = "base",
    ticker: str | None = None,
    roic_center: float | None = None,
) -> tuple[float, float, float, float]:
    prior = prior_for(ticker, company_type)
    reinvestment_high = 0.65 if company_type in {"hyperscaler", "platform"} else 0.55
    if prior.archetype == "capital_intensive_growth":
        reinvestment_high = 0.90
    if company_type == "marketplace_hybrid":
        reinvestment_high = 1.20
    reinvestment_low = {
        "asset_light_quality": 0.00,
        "hyperscaler": 0.02,
        "platform": 0.02,
        "consumer_compounder": 0.06,
        "consumer_staple": 0.04,
        "marketplace_hybrid": 0.12,
    }.get(company_type, 0.03)
    if prior.archetype == "capital_intensive_growth":
        reinvestment_low = 0.12
    if calibration_mode == "consensus_calibrated":
        reinvestment_low *= 0.75
        reinvestment_high = min(reinvestment_high, 0.50)
    roic_low, roic_high = _roic_bounds(prior, roic_center)
    return (
        float(np.clip(state[0], -0.08, _growth_cap(company_type, calibration_mode, ticker))),
        float(np.clip(state[1], prior.margin_floor, prior.margin_cap)),
        float(np.clip(state[2], reinvestment_low, reinvestment_high)),
        float(np.clip(state[3], roic_low, roic_high)),
    )


def _bound_state_with_trace(
    state: np.ndarray,
    company_type: str,
    calibration_mode: str = "base",
    ticker: str | None = None,
    roic_center: float | None = None,
) -> dict[str, object]:
    values = _bound_state(state, company_type, calibration_mode, ticker, roic_center)
    names = ("revenue_growth", "operating_margin", "reinvestment_rate", "roic")
    return {
        "values": values,
        "clamp_flags": {
            name: bool(not np.isclose(float(raw), float(final), rtol=1e-9, atol=1e-12))
            for name, raw, final in zip(names, state, values)
        },
    }


def _marketplace_reinvestment_path(quarter: int) -> float:
    return float(max(0.35, 0.95 - 0.035 * quarter))


def _growth_cap(company_type: str, calibration_mode: str = "base", ticker: str | None = None) -> float:
    prior = prior_for(ticker, company_type)
    if calibration_mode == "consensus_calibrated":
        return {
            "hyperscaler": 0.10,
            "platform": 0.10,
            "marketplace_hybrid": 0.12,
            "asset_light_quality": 0.075,
            "consumer_compounder": 0.05,
            "consumer_staple": 0.04,
        }.get(company_type, prior.revenue_growth_cap)
    return {
        "hyperscaler": 0.08,
        "platform": 0.08,
        "marketplace_hybrid": 0.10,
        "asset_light_quality": 0.06,
        "consumer_compounder": 0.04,
        "consumer_staple": 0.035,
    }.get(company_type, prior.revenue_growth_cap)


def _estimate_transition(centered: np.ndarray) -> np.ndarray:
    return _stabilize_transition(np.diag(_estimate_persistence(centered)))


def _estimate_persistence(centered: np.ndarray) -> np.ndarray:
    values = []
    for idx in range(centered.shape[1]):
        series = centered[:, idx]
        mask = np.isfinite(series[1:]) & np.isfinite(series[:-1])
        if mask.sum() < 4:
            values.append(0.65)
            continue
        x = series[:-1][mask]
        y = series[1:][mask]
        denom = float(np.dot(x, x))
        values.append(0.50 if denom <= 1e-12 else float(np.dot(x, y) / denom))
    return np.clip(np.asarray(values, dtype=float), 0.05, 0.96)


def _state_anchor(x: np.ndarray, company_type: str, ticker: str | None = None) -> np.ndarray:
    """Long-run reversion anchor for the state mean.

    Default 'median' reproduces production behavior exactly. Alternatives are used
    only by the structural-break anchor research study (DCF_ANCHOR_METHOD).
    """
    import os

    method = os.environ.get("DCF_ANCHOR_METHOD", "median").lower()
    if method == "median":
        return np.nanmedian(x, axis=0)
    if method in {"rolling5y", "rolling_5y", "rolling"}:
        window = x[-20:] if len(x) >= 20 else x
        return np.nanmedian(window, axis=0)
    if method in {"ewma", "exponential"}:
        return _ewma_anchor(x, halflife=8.0)
    if method == "bayesian":
        recent = _nan_reduce(x[-8:] if len(x) >= 8 else x, np.nanmean)
        long_run = np.nanmedian(x, axis=0)
        prior = _archetype_anchor(company_type, ticker)
        blended = 0.50 * recent + 0.30 * long_run + 0.20 * prior
        return np.where(np.isfinite(blended), blended, long_run)
    return np.nanmedian(x, axis=0)


def _nan_reduce(window: np.ndarray, fn) -> np.ndarray:
    with np.errstate(all="ignore"):
        return fn(window, axis=0)


def _ewma_anchor(x: np.ndarray, halflife: float = 8.0) -> np.ndarray:
    n = len(x)
    ages = np.arange(n - 1, -1, -1, dtype=float)  # most recent row has age 0
    weights = 0.5 ** (ages / halflife)
    out = np.empty(x.shape[1], dtype=float)
    for col in range(x.shape[1]):
        values = x[:, col]
        mask = np.isfinite(values)
        if not mask.any():
            out[col] = np.nan
            continue
        w = weights[mask]
        out[col] = float(np.dot(values[mask], w) / w.sum())
    return out


def _archetype_anchor(company_type: str, ticker: str | None) -> np.ndarray:
    prior = prior_for(ticker, company_type)
    return np.array(
        [
            0.5 * (prior.revenue_growth_floor + prior.revenue_growth_cap),
            0.5 * (prior.margin_floor + prior.margin_cap),
            prior.reinvestment_burden,
            0.5 * (prior.roic_floor + prior.roic_cap),
        ],
        dtype=float,
    )


def _apply_long_run_mean_priors(
    mu: np.ndarray,
    company_type: str,
    calibration_mode: str = "base",
    ticker: str | None = None,
) -> np.ndarray:
    adjusted = np.asarray(mu, dtype=float).copy()
    prior = prior_for(ticker, company_type)
    floors = {
        "asset_light_quality": [0.005, 0.26, 0.03, 0.08],
        "hyperscaler": [0.008, 0.28, 0.06, 0.08],
        "platform": [0.008, 0.26, 0.05, 0.08],
        "consumer_compounder": [0.004, 0.18, 0.03, 0.06],
        "consumer_staple": [0.003, 0.14, 0.03, 0.035],
        "marketplace_hybrid": [0.010, 0.06, 0.20, 0.025],
    }.get(company_type)
    caps = {
        "asset_light_quality": [0.035, 0.44, 0.38, 0.22],
        "hyperscaler": [0.045, 0.48, 0.45, 0.22],
        "platform": [0.045, 0.48, 0.45, 0.22],
        "consumer_compounder": [0.025, 0.36, 0.35, 0.16],
        "consumer_staple": [0.020, 0.30, 0.35, 0.10],
        "marketplace_hybrid": [0.055, 0.22, 0.95, 0.10],
    }.get(company_type)
    if floors is None or caps is None:
        floors = [prior.revenue_growth_floor, prior.margin_floor, 0.03, prior.roic_floor / 4.0]
        caps = [prior.revenue_growth_cap, prior.margin_cap, 0.85, prior.roic_cap / 4.0]
    historical_roic = float(np.asarray(mu, dtype=float)[3])
    adjusted = np.minimum(np.maximum(adjusted, floors), caps)
    adjusted[2] = 0.50 * adjusted[2] + 0.50 * prior.reinvestment_burden
    # Ticker-specific ROIC calibration: blend the company's own historical ROIC
    # with the archetype midpoint instead of pinning the long-run ROIC mean to a
    # fixed archetype band. This lets companies with materially different realized
    # ROIC (e.g. PLTR vs NVDA, both "industrial" archetype) express distinct
    # economics in the mean-reversion target the Kalman forecast pulls toward.
    # Set DCF_DISABLE_ROIC_CALIBRATION=1 to restore the legacy archetype pinning
    # (used for A/B isolation of the calibration's valuation impact, and rollback).
    if not _roic_calibration_disabled():
        adjusted[3] = _calibrated_roic_center(historical_roic, prior)
    if calibration_mode == "consensus_calibrated":
        adjusted[0] = min(adjusted[0] + 0.008, _growth_cap(company_type, calibration_mode, ticker))
        adjusted[1] = min(adjusted[1] + 0.025, 0.55)
        adjusted[2] = max(adjusted[2] * 0.80, 0.04)
        adjusted[3] = min(adjusted[3] * 1.15, 0.30)
    return adjusted


def _roic_calibration_disabled() -> bool:
    import os

    return os.environ.get("DCF_DISABLE_ROIC_CALIBRATION", "").lower() in {"1", "true", "yes"}


def _calibrated_roic_center(historical_roic: float, prior) -> float:
    """Blend a company's realized ROIC with its archetype midpoint.

    Replaces hard archetype floor/cap pinning with a history-weighted center so
    different historical ROIC profiles produce different forecast ROIC. The
    result is kept inside a generous absolute envelope for sanity, not snapped to
    the archetype band.
    """
    archetype_mid = 0.5 * (prior.roic_floor + prior.roic_cap)
    if not np.isfinite(historical_roic):
        return float(archetype_mid)
    blended = 0.65 * float(historical_roic) + 0.35 * archetype_mid
    return float(np.clip(blended, 0.02, 0.45))


def _roic_bounds(prior, roic_center: float | None) -> tuple[float, float]:
    """Bounded-persistence ROIC band around a calibrated center.

    When a ticker-specific center is supplied the per-quarter ROIC is allowed to
    vary within +/-35% of that center (a persistence band) rather than being
    clipped to the fixed archetype floor/cap. Falls back to the archetype band
    when no center is available (e.g. legacy callers / generic fixtures).
    """
    if roic_center is None or not np.isfinite(float(roic_center)) or _roic_calibration_disabled():
        return prior.roic_floor, prior.roic_cap
    center = float(roic_center)
    half_band = max(0.015, 0.35 * abs(center))
    low = max(0.01, center - half_band)
    high = max(low + 1e-3, center + half_band)
    return low, high


def _apply_company_type_priors(
    phi: np.ndarray,
    company_type: str,
    calibration_mode: str = "base",
    ticker: str | None = None,
) -> np.ndarray:
    prior = prior_for(ticker, company_type)
    floors_by_type = {
        "asset_light_quality": [0.78, 0.96, 0.70, 0.985],
        "platform": [0.80, 0.965, 0.72, 0.985],
        "hyperscaler": [0.82, 0.97, 0.74, 0.99],
        "consumer_compounder": [0.72, 0.96, 0.68, 0.985],
        "consumer_staple": [0.68, 0.92, 0.68, 0.92],
        "marketplace_hybrid": [0.84, 0.86, 0.86, 0.90],
    }
    caps_by_type = {
        "asset_light_quality": [0.92, 0.995, 0.88, 0.995],
        "platform": [0.94, 0.995, 0.90, 0.995],
        "hyperscaler": [0.94, 0.995, 0.92, 0.995],
        "consumer_compounder": [0.88, 0.99, 0.86, 0.995],
        "consumer_staple": [0.84, 0.97, 0.84, 0.97],
        "marketplace_hybrid": [0.95, 0.94, 0.96, 0.97],
    }
    floors = floors_by_type.get(company_type)
    if floors is None:
        floors = [
            prior.growth_persistence,
            prior.margin_persistence,
            prior.reinvestment_persistence,
            prior.roic_persistence,
        ]
        caps = [0.96, 0.998, 0.94, 0.998]
    else:
        caps = caps_by_type[company_type]
    adjusted = phi.copy()
    diagonal = np.diag(adjusted).copy()
    for idx, floor in enumerate(floors):
        diagonal[idx] = min(max(float(diagonal[idx]), floor), caps[idx])
    if calibration_mode == "consensus_calibrated":
        diagonal[1] = min(max(diagonal[1], 0.985), 0.998)
        diagonal[3] = min(max(diagonal[3], 0.992), 0.998)
    return _stabilize_transition(np.diag(diagonal))


def _stabilize_transition(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals = np.linalg.eigvals(phi)
    radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    if radius >= 0.99:
        phi = phi * (0.99 / radius)
    return phi


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 0:
        matrix = np.asarray([[float(matrix)]])
    matrix = np.atleast_2d(matrix)
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T + np.eye(matrix.shape[0]) * floor


def _unscale_kalman_result(result: KalmanResult, center: np.ndarray, scale: np.ndarray) -> KalmanResult:
    return KalmanResult(
        predicted_states=_unscale_states(result.predicted_states, center, scale),
        predicted_covariances=_unscale_covariances(result.predicted_covariances, scale),
        filtered_states=_unscale_states(result.filtered_states, center, scale),
        filtered_covariances=_unscale_covariances(result.filtered_covariances, scale),
        smoothed_states=_unscale_states(result.smoothed_states, center, scale),
        smoothed_covariances=_unscale_covariances(result.smoothed_covariances, scale),
        filtered_state=_unscale_state(result.filtered_state, center, scale),
        filtered_cov=_unscale_covariance(result.filtered_cov, scale),
        log_likelihood=result.log_likelihood,
        innovations=result.innovations,
    )


def _unscale_states(states: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return center + states * scale


def _unscale_state(state: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return center + state * scale


def _unscale_covariances(covariances: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray([_unscale_covariance(cov, scale) for cov in covariances], dtype=float)


def _unscale_covariance(covariance: np.ndarray, scale: np.ndarray) -> np.ndarray:
    d = np.diag(scale)
    return d @ covariance @ d


def _prior_influence_trace(
    columns: tuple[str, ...],
    raw_mu: np.ndarray,
    prior_mu: np.ndarray,
    empirical_phi: np.ndarray,
    prior_phi: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    return {
        "standardization": {
            column: {"center": float(c), "scale": float(s)}
            for column, c, s in zip(columns, center, scale)
        },
        "long_run_mean": {
            column: {
                "model_contribution": float(raw),
                "prior_adjusted_value": float(prior),
                "prior_contribution": float(prior - raw),
            }
            for column, raw, prior in zip(columns, raw_mu, prior_mu)
        },
        "transition_persistence": {
            column: {
                "model_contribution": float(emp),
                "prior_adjusted_value": float(adj),
                "prior_contribution": float(adj - emp),
            }
            for column, emp, adj in zip(columns, np.diag(empirical_phi), np.diag(prior_phi))
        },
    }


def _interpolate_missing(x: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(x)
    frame = frame.replace([np.inf, -np.inf], np.nan).interpolate().ffill().bfill()
    return frame.to_numpy(dtype=float)
