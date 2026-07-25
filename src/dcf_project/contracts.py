from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceMeta(BaseModel):
    provider: str
    key: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    url: str | None = None


class FinancialPeriod(BaseModel):
    ticker: str
    report_date: date
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    revenue: float | None = None
    ebit: float | None = None
    tax_expense: float | None = None
    capex: float | None = None
    da: float | None = None
    r_and_d: float | None = None
    sbc: float | None = None
    operating_lease_liability: float | None = None
    operating_lease_cost: float | None = None
    ppe: float | None = None
    goodwill: float | None = None
    intangible_assets: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    nwc: float | None = None
    cash: float | None = None
    debt: float | None = None
    shares_outstanding: float | None = None
    source: str = "unknown"


class MarketSnapshot(BaseModel):
    ticker: str
    price: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None
    beta: float | None = None
    cash: float | None = None
    debt: float | None = None
    currency: str | None = None
    source: SourceMeta


class WACCInputs(BaseModel):
    risk_free_rate: float
    beta: float
    equity_risk_premium: float
    cost_of_equity: float
    pre_tax_cost_of_debt: float
    tax_rate: float
    market_value_equity: float
    market_value_debt: float
    equity_weight: float
    debt_weight: float
    after_tax_cost_of_debt: float
    wacc: float
    source_notes: list[str] = Field(default_factory=list)


class FeatureFrame(BaseModel):
    ticker: str
    rows: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)


class GrowthForecast(BaseModel):
    ticker: str
    model_type: Literal["multivariate_dlm", "univariate_local_linear"]
    single_growth_rate: float
    raw_fcf_growth_rate: float
    smoothed_fcf_growth_rate: float
    clamped_fcf_growth_rate: float
    growth_clamped: bool
    clamp_triggered: bool
    horizon_quarters: int
    growth_path: list[float]
    fcf_path: list[float]
    lower_growth_path: list[float]
    upper_growth_path: list[float]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DCFResult(BaseModel):
    ticker: str
    explicit_forecast_quarters: int
    intrinsic_value_per_share: float
    accounting_intrinsic: float | None = None
    economic_intrinsic: float | None = None
    owner_intrinsic: float | None = None
    ensemble_intrinsic: float | None = None
    ensemble_weights: dict[str, float] = Field(default_factory=dict)
    enterprise_value: float
    equity_value: float
    net_debt: float
    projected_fcf: list[float]
    explicit_pv: float
    pv_projected_fcf: float
    terminal_value: float
    pv_terminal_value: float
    pv_terminal: float | None = None
    fcf_terminal_base: float
    wacc: float
    terminal_growth_rate: float
    fcf_growth_rate: float
    raw_fcf_growth_rate: float
    smoothed_fcf_growth_rate: float
    clamped_fcf_growth_rate: float
    growth_clamped: bool
    clamp_triggered: bool
    valuation_p10: float
    valuation_p50: float
    valuation_p90: float
    diluted_shares_outstanding: float
    market_price: float | None = None
    implied_upside: float | None = None
    forecast_horizon_quarters: int | None = None
    forecast_mode: str | None = None
    company_archetype: str | None = None
    consensus_weight: float | None = None
    buyback_rate: float | None = None
    r_and_d_capitalized: bool | None = None
    segment_mode: str | None = None
    terminal_growth_source_type: str | None = None
    # Human-readable WACC provenance for the report front end (e.g. "ValueInvesting.io",
    # "Internal CAPM", "Internal CAPM (provider unavailable)"). Same provider path for US and India.
    wacc_source: str | None = None
    provider_long_term_growth: float | None = None
    provider_terminal_growth: float | None = None
    valuation_regime: str | None = None
    scenario_values: dict[str, float] = Field(default_factory=dict)
    probability_weights: dict[str, float] = Field(default_factory=dict)
    reverse_dcf_expectations: dict[str, Any] = Field(default_factory=dict)
    scenario_generation_method: str | None = None
    probability_method: str | None = None
    # Engine provenance stamp. Defaults to "v1" (frozen canonical engine); the
    # Bayesian engine_v2 sets this to "v2". Additive metadata only.
    engine_version: str = "v1"
    # Output-coherence verdict (engine_v3 Research_V4 Mod 1): "ok" for a valuation the
    # engine stands behind, "incomplete" when the headline is numerically broken and
    # should be withheld. None when the guard did not run (v1/v2, or guard disabled).
    valuation_coherence: str | None = None
    monte_carlo_distribution: dict[str, Any] = Field(default_factory=dict)
    forecast_financials: list[dict[str, Any]] = Field(default_factory=list)
    confidence_interval: dict[str, float] = Field(default_factory=dict)
    sensitivity: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    sources: list[SourceMeta] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
