from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal


ValuationMode = Literal[
    "auto",
    "accounting_dcf",
    "economic_dcf",
    "owner_earnings_dcf",
    "roic_dcf",
    "reported",
    "economic",
    "owner_earnings",
    "consensus_calibrated",
]
CompanyType = Literal[
    "auto",
    "asset_light",
    "asset_light_quality",
    "industrial",
    "consumer_staple",
    "consumer_compounder",
    "bank",
    "insurance",
    "hyperscaler",
    "platform",
    "marketplace_hybrid",
    "capital_intensive_growth",
    "semiconductor",
    "telecom",
    "retail",
]
SBCMode = Literal["ignore", "full_expense", "partial"]
MaintenanceCapexMethod = Literal[
    "ensemble",
    "depreciation_anchor",
    "growth_adjusted",
    "percentile",
    "asset_turnover",
]
WACCSource = Literal["auto", "valueinvesting", "internal"]
TerminalGrowthSource = Literal["auto", "valueinvesting", "internal"]
ForecastMode = Literal["state_space", "consensus_blend"]
ValuationRegime = Literal[
    "auto",
    "mature_compounder",
    "platform_growth",
    "optionality_hypergrowth",
    "capital_intensive_cyclical",
    "consumer_defensive",
]


@dataclass(frozen=True)
class Settings:
    sec_user_agent: str = os.getenv(
        "DCF_SEC_USER_AGENT",
        "dcf-project/0.1 contact@example.com",
    )
    # Market/country selection. "US" (default) is the frozen production behaviour: SEC
    # filings, US Treasury risk-free, and the US macro anchors below. "IN" routes to the
    # India bolt-on (NSE/BSE XBRL + live India macro). Resolved once at the valuation entry
    # point (see data_sources.markets.resolve_country) and threaded through unchanged.
    country: str = os.getenv("DCF_COUNTRY", "US")
    equity_risk_premium: float = float(os.getenv("DCF_EQUITY_RISK_PREMIUM", "0.045"))
    default_cost_of_debt: float = float(os.getenv("DCF_DEFAULT_COST_OF_DEBT", "0.045"))
    wacc_source: WACCSource = os.getenv("DCF_WACC_SOURCE", "auto")  # type: ignore[assignment]
    wacc_override: float | None = (
        float(os.getenv("DCF_WACC_OVERRIDE"))
        if os.getenv("DCF_WACC_OVERRIDE") is not None
        else None
    )
    # Risk-free override (annualised, decimal). When set, MarketClient.risk_free_rate returns
    # it verbatim instead of fetching the US Treasury 10Y. Default None keeps the US path
    # byte-identical; the India bolt-on sets it to the live G-Sec yield so US Treasury state
    # never leaks into an Indian valuation via the class-level risk-free cache.
    risk_free_override: float | None = (
        float(os.getenv("DCF_RISK_FREE_OVERRIDE"))
        if os.getenv("DCF_RISK_FREE_OVERRIDE") is not None
        else None
    )
    terminal_growth_source: TerminalGrowthSource = os.getenv("DCF_TERMINAL_GROWTH_SOURCE", "auto")  # type: ignore[assignment]
    terminal_growth_override: float | None = (
        float(os.getenv("DCF_TERMINAL_GROWTH_OVERRIDE"))
        if os.getenv("DCF_TERMINAL_GROWTH_OVERRIDE") is not None
        else None
    )
    strict_provider_terminal_growth: bool = (
        os.getenv("DCF_STRICT_PROVIDER_TERMINAL_GROWTH", "false").lower() in {"1", "true", "yes"}
    )
    default_tax_rate: float = float(os.getenv("DCF_DEFAULT_TAX_RATE", "0.21"))
    terminal_growth_cap: float = float(os.getenv("DCF_TERMINAL_GROWTH_CAP", "0.04"))
    default_terminal_growth: float = float(os.getenv("DCF_DEFAULT_TERMINAL_GROWTH", "0.025"))
    long_run_nominal_gdp: float = float(os.getenv("DCF_LONG_RUN_NOMINAL_GDP", "0.04"))
    inflation_anchor: float = float(os.getenv("DCF_INFLATION_ANCHOR", "0.025"))
    terminal_growth_floor: float = float(os.getenv("DCF_TERMINAL_GROWTH_FLOOR", "0.015"))
    forecast_mode: ForecastMode = os.getenv("DCF_FORECAST_MODE", "consensus_blend")  # type: ignore[assignment]
    consensus_weight: float = float(os.getenv("DCF_CONSENSUS_WEIGHT", "0.70"))
    buyback_model_enabled: bool = os.getenv("DCF_BUYBACK_MODEL_ENABLED", "true").lower() in {"1", "true", "yes"}
    max_buyback_rate: float = float(os.getenv("DCF_MAX_BUYBACK_RATE", "0.04"))
    capitalize_rd: bool = os.getenv("DCF_CAPITALIZE_RD", "true").lower() in {"1", "true", "yes"}
    rd_capitalization_factor: float = float(os.getenv("DCF_RD_CAPITALIZATION_FACTOR", "0.70"))
    segment_mode: str = os.getenv("DCF_SEGMENT_MODE", "disabled")
    segment_assumptions_json: str | None = os.getenv("DCF_SEGMENT_ASSUMPTIONS_JSON")
    valuation_regime: ValuationRegime = os.getenv("DCF_VALUATION_REGIME", "auto")  # type: ignore[assignment]
    valuation_regime_mapping_json: str | None = os.getenv("DCF_VALUATION_REGIME_MAPPING_JSON")
    scenario_probability_method: str = os.getenv("DCF_SCENARIO_PROBABILITY_METHOD", "bayesian")
    scenario_probability_weights_json: str | None = os.getenv("DCF_SCENARIO_PROBABILITY_WEIGHTS_JSON")
    scenario_monte_carlo_samples: int = int(os.getenv("DCF_SCENARIO_MONTE_CARLO_SAMPLES", "2000"))
    normalize_capex: bool = os.getenv("DCF_NORMALIZE_CAPEX", "true").lower() in {"1", "true", "yes"}
    valuation_mode: ValuationMode = os.getenv("DCF_VALUATION_MODE", "roic_dcf")  # type: ignore[assignment]
    company_type: CompanyType = os.getenv("DCF_COMPANY_TYPE", "auto")  # type: ignore[assignment]
    sbc_mode: SBCMode = os.getenv("DCF_SBC_MODE", "partial")  # type: ignore[assignment]
    # Research lever (default 0.0 = current production behaviour: SBC fully expensed).
    # Fraction of stock-based comp added back to operating income (EBIT) in the
    # historical reconstruction, flowing consistently into the state-space fit and
    # forecast. 0.0 = expensed, 0.5 = partial, 1.0 = market-style (SBC added back,
    # dilution reflected only via share count). Used by the SBC treatment study.
    sbc_addback_fraction: float = float(os.getenv("DCF_SBC_ADDBACK_FRACTION", "0.0"))
    sbc_haircut_factor: float | None = (
        float(os.getenv("DCF_SBC_HAIRCUT_FACTOR"))
        if os.getenv("DCF_SBC_HAIRCUT_FACTOR") is not None
        else None
    )
    normalize_leases: bool = os.getenv("DCF_NORMALIZE_LEASES", "true").lower() in {"1", "true", "yes"}
    maintenance_capex_method: MaintenanceCapexMethod = os.getenv(
        "DCF_MAINTENANCE_CAPEX_METHOD",
        "ensemble",
    )  # type: ignore[assignment]
    maintenance_capex_sector_floor_ratio: float = float(
        os.getenv("DCF_MAINTENANCE_CAPEX_SECTOR_FLOOR_RATIO", "0.02")
    )
    maintenance_capex_percentile: float = float(os.getenv("DCF_MAINTENANCE_CAPEX_PERCENTILE", "0.25"))
    use_proxy_invested_capital: bool = (
        os.getenv("DCF_USE_PROXY_INVESTED_CAPITAL", "false").lower() in {"1", "true", "yes"}
    )
    allow_high_economic_fcf_margin: bool = (
        os.getenv("DCF_ALLOW_HIGH_ECONOMIC_FCF_MARGIN", "false").lower() in {"1", "true", "yes"}
    )
    min_mature_growth: float = float(os.getenv("DCF_MIN_MATURE_GROWTH", "-0.05"))
    max_mature_growth: float = float(os.getenv("DCF_MAX_MATURE_GROWTH", "0.15"))
    explicit_forecast_quarters: int = int(os.getenv("DCF_EXPLICIT_FORECAST_QUARTERS", "20"))
    request_timeout_seconds: int = int(os.getenv("DCF_REQUEST_TIMEOUT_SECONDS", "30"))
    # --- SEC disk cache (additive; default OFF keeps the v1 network path identical).
    # When ON, companyfacts responses are stored verbatim under .cache/sec/ so PIT
    # backtest campaigns reprice a frozen, internally consistent filing snapshot
    # instead of refetching ~500 names per run (6-8h with throttling -> minutes).
    sec_cache_enabled: bool = os.getenv("DCF_SEC_CACHE_ENABLED", "false").lower() in {"1", "true", "yes"}
    sec_cache_max_age_days: int = int(os.getenv("DCF_SEC_CACHE_MAX_AGE_DAYS", "30"))
    min_multivariate_periods: int = 10
    random_seed: int = 7
    run_backtest: bool = os.getenv("DCF_RUN_BACKTEST", "false").lower() in {"1", "true", "yes"}
    run_reverse_dcf: bool = os.getenv("DCF_RUN_REVERSE_DCF", "true").lower() in {"1", "true", "yes"}
    run_sensitivity: bool = os.getenv("DCF_RUN_SENSITIVITY", "true").lower() in {"1", "true", "yes"}
    monte_carlo_samples: int = int(os.getenv("DCF_MONTE_CARLO_SAMPLES", "2000"))
    verbose: bool = os.getenv("DCF_VERBOSE", "false").lower() in {"1", "true", "yes"}
    # --- engine_v2 (Bayesian state-space Monte Carlo) selection + knobs ---------
    # Additive only. Default "v1" keeps the frozen canonical engine as the scientific
    # control; "v2" selects the Bayesian posterior-coherent engine (see engine_v2/).
    # None of these alter v1 numeric outputs.
    engine_version: str = os.getenv("DCF_ENGINE_VERSION", "v1")
    v2_paths: int = int(os.getenv("DCF_V2_PATHS", "2000"))
    v2_mcmc_draws: int = int(os.getenv("DCF_V2_MCMC_DRAWS", "1000"))
    v2_mcmc_chains: int = int(os.getenv("DCF_V2_MCMC_CHAINS", "4"))
    v2_mcmc_warmup: int = int(os.getenv("DCF_V2_MCMC_WARMUP", "1000"))
    v2_mcmc_thin: int = int(os.getenv("DCF_V2_MCMC_THIN", "2"))
    v2_sample_wacc: bool = os.getenv("DCF_V2_SAMPLE_WACC", "false").lower() in {"1", "true", "yes"}
    # Terminal-value source for v2: "equilibrium" (U3, derive g from x*=mu) or "v1" (hold the
    # v1 baseline terminal growth). "v1" is the P2.5 diagnostic that isolates the posterior
    # distribution effect from the equilibrium-terminal change.
    v2_terminal_mode: str = os.getenv("DCF_V2_TERMINAL_MODE", "equilibrium")
    # --- engine_v3 (economic-drivers fade/CAP/persistence) knobs -----------------
    # Additive only; selected via engine_version="v3". Every knob is an explicit
    # economic assumption (documented in engine_v3/config.py). None alter v1 outputs.
    # Explicit fade horizon before terminal value. 40 quarters (10 years) lets the
    # company-specific driver half-lives substantially complete before the perpetuity,
    # so PV(terminal)/EV stays defensible. Decoupled from the v1 setting
    # ``explicit_forecast_quarters`` (which stays 20) -- v1 is never affected.
    v3_horizon_quarters: int = int(os.getenv("DCF_V3_HORIZON_QUARTERS", "40"))
    # Monte Carlo over the economic drivers is OPTIONAL and DIAGNOSTIC-ONLY (default 0
    # = off). It failed the pre-registered calibration gate G6 (p10-p90 coverage ~5%),
    # so it never drives the headline and carries no return-forecast claim; enable it
    # only to render a driver-sensitivity range.
    v3_mc_draws: int = int(os.getenv("DCF_V3_MC_DRAWS", "0"))
    # Headline intrinsic: "deterministic" (economic skeleton; MC is uncertainty only,
    # since valuation is convex in drivers and the MC mean is Jensen-inflated) or
    # "mc_median" (ablatable alternative signal in backtests).
    v3_headline: str = os.getenv("DCF_V3_HEADLINE", "deterministic")
    v3_phi_prior_strength: float = float(os.getenv("DCF_V3_PHI_PRIOR_STRENGTH", "12.0"))
    # Empirical-Bayes prior strength for the per-driver fade half-life fits (growth,
    # margin): the company AR(1) estimate gets weight n/(n+n0) against the sector
    # half-life prior. Lower than phi's n0 because the half-life is a coarser quantity.
    v3_half_life_prior_strength: float = float(os.getenv("DCF_V3_HALF_LIFE_PRIOR_STRENGTH", "8.0"))
    v3_cap_max_years: float = float(os.getenv("DCF_V3_CAP_MAX_YEARS", "15.0"))
    v3_moat_premium_cap: float = float(os.getenv("DCF_V3_MOAT_PREMIUM_CAP", "0.02"))
    v3_reinvestment_cap: float = float(os.getenv("DCF_V3_REINVESTMENT_CAP", "0.95"))
    v3_terminal_dilution_haircut: float = float(os.getenv("DCF_V3_TERMINAL_DILUTION_HAIRCUT", "0.01"))
    v3_terminal_g_sigma: float = float(os.getenv("DCF_V3_TERMINAL_G_SIGMA", "0.004"))
    v3_wacc_sigma: float = float(os.getenv("DCF_V3_WACC_SIGMA", "0.008"))
    v3_sample_wacc: bool = os.getenv("DCF_V3_SAMPLE_WACC", "false").lower() in {"1", "true", "yes"}
    v3_sector_state_kappa: float = float(os.getenv("DCF_V3_SECTOR_STATE_KAPPA", "0.0"))
    # Output-coherence guard (Research_V4, Modification 1). Pure robustness: flags an
    # engine_v3 result as "incomplete" when its output is numerically broken (corrupt
    # share count, negative equity, terminal blow-up, or an intrinsic implausible vs the
    # traded price) instead of emitting nonsense. Default ON: it never alters a healthy
    # valuation (those stay bit-identical) and only fires on already-pathological output.
    v3_coherence_guard: bool = os.getenv("DCF_V3_COHERENCE_GUARD", "true").lower() in {"1", "true", "yes"}


def get_settings() -> Settings:
    return Settings()
