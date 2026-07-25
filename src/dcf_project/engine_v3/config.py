"""Configuration for the economic-drivers engine (engine_v3).

A thin view over the additive ``v3_*`` knobs on the frozen :class:`Settings`,
mirroring ``engine_v2.config.BayesianV2Config``. Every knob is an explicit economic
assumption (documented at the field), not a statistical tuning parameter.
"""
from __future__ import annotations

from dataclasses import dataclass

from dcf_project.config import Settings


@dataclass(frozen=True)
class V3Config:
    """Resolved engine_v3 knobs."""

    #: Explicit fade horizon in quarters before terminal value (40 = 10 years). Decoupled
    #: from the v1 ``explicit_forecast_quarters`` so v1 numerics are never touched.
    horizon_quarters: int = 40
    #: Monte Carlo draws over the economic drivers (0 = OFF, the default). The MC layer is
    #: diagnostic-only (failed calibration gate G6); it never drives the headline.
    mc_draws: int = 0
    #: Headline intrinsic: "deterministic" (the economic skeleton; default -- valuation is
    #: convex in drivers, so the MC mean/median is Jensen-inflated) or "mc_median".
    headline: str = "deterministic"
    #: Empirical-Bayes prior strength n0 for the excess-ROIC persistence fit: the company
    #: estimate gets weight n/(n + n0·(1+2·quality_penalty)) against the sector prior.
    phi_prior_strength: float = 12.0
    #: Empirical-Bayes prior strength n0 for the per-driver fade half-life fits.
    half_life_prior_strength: float = 8.0
    #: Hard bounds (years) on every fade half-life -- company estimate, sector prior, and
    #: CAP-derived fallback are all clipped here. Floor: even a no-moat driver does not
    #: revert instantaneously; ceiling: no driver stays out of equilibrium past ~8y.
    half_life_min_years: float = 0.5
    half_life_max_years: float = 8.0
    #: Competitive Advantage Period ceiling (years) -- no company gets a longer runway.
    cap_max_years: float = 15.0
    #: Cap on the terminal excess return (RONIC − WACC) that survives the explicit horizon.
    moat_premium_cap: float = 0.02
    #: Sustainable-growth bind: max reinvestment share of NOPAT; growth above
    #: reinvestment_cap × RONIC is reduced (growth must be paid for, never free).
    reinvestment_cap: float = 0.95
    #: Grinold-Kroner dilution haircut subtracted from the nominal-GDP growth anchor
    #: (net share issuance dilutes per-share claims on aggregate earnings).
    terminal_dilution_haircut: float = 0.01
    #: MC sampling sigmas (terminal growth band-clipped; WACC sampled only if sample_wacc).
    terminal_g_sigma: float = 0.004
    wacc_sigma: float = 0.008
    sample_wacc: bool = False
    #: Sector-state tilt weight on the near-term growth anchor (default 0 = off;
    #: adopted only if the G5 ablation shows a significant paired-IC improvement).
    sector_state_kappa: float = 0.0
    #: Output-coherence guard (Research_V4 Mod 1). When True, a numerically broken
    #: result is classified ``incomplete`` (withheld as a headline) rather than emitted.
    #: Robustness-only: never moves a healthy intrinsic.
    coherence_guard: bool = True
    seed: int = 7

    @classmethod
    def from_settings(cls, settings: Settings) -> "V3Config":
        return cls(
            horizon_quarters=int(getattr(settings, "v3_horizon_quarters", 40)),
            mc_draws=int(getattr(settings, "v3_mc_draws", 0)),
            headline=str(getattr(settings, "v3_headline", "deterministic")),
            phi_prior_strength=float(getattr(settings, "v3_phi_prior_strength", 12.0)),
            half_life_prior_strength=float(getattr(settings, "v3_half_life_prior_strength", 8.0)),
            cap_max_years=float(getattr(settings, "v3_cap_max_years", 15.0)),
            moat_premium_cap=float(getattr(settings, "v3_moat_premium_cap", 0.02)),
            reinvestment_cap=float(getattr(settings, "v3_reinvestment_cap", 0.95)),
            terminal_dilution_haircut=float(getattr(settings, "v3_terminal_dilution_haircut", 0.01)),
            terminal_g_sigma=float(getattr(settings, "v3_terminal_g_sigma", 0.004)),
            wacc_sigma=float(getattr(settings, "v3_wacc_sigma", 0.008)),
            sample_wacc=bool(getattr(settings, "v3_sample_wacc", False)),
            sector_state_kappa=float(getattr(settings, "v3_sector_state_kappa", 0.0)),
            coherence_guard=bool(getattr(settings, "v3_coherence_guard", True)),
            seed=int(getattr(settings, "random_seed", 7)),
        )
