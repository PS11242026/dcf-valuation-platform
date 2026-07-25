"""Monte Carlo over the economic drivers (engine_v3).

Uncertainty enters where it actually lives -- in the ESTIMATED DRIVERS (base revenue
growth, normalized margin target, persistence phi, terminal growth, optionally WACC) --
not in latent states. Each draw re-runs the *identical* deterministic fade skeleton
(``fade.build_fade_forecast`` via its override arguments) and re-values it with the
*identical* bridge (``assemble.value_fade_path``), so zero-variance draws reproduce
the skeleton exactly (a tested invariant: there is one implementation of the
economics, not a sampler-side copy).

The headline intrinsic stays the deterministic skeleton: valuation is convex in the
drivers, so the MC mean/median is Jensen-inflated and reported as uncertainty
(p10/p50/p90 bands, fan charts, probability-undervalued signals), never silently as
the point estimate. Draws are independent across drivers by design; a (g, margin)
correlation is added only if the G6 calibration gate shows over-wide bands.

Payload schema reuses the engine_v2 distribution functions so downstream consumers
(Excel renderer, backtest runners) see one schema family.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from dcf_project.contracts import MarketSnapshot
from dcf_project.engine_v2.result import _fan, candidate_signals, distribution_outputs
from dcf_project.engine_v3.assemble import value_fade_path
from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.drivers import YOY_GROWTH_BAND, DriverEstimates
from dcf_project.engine_v3.fade import build_fade_forecast
from dcf_project.engine_v3.halflives import HalfLifeEstimates
from dcf_project.engine_v3.persistence import PersistenceFit
from dcf_project.engine_v3.sector_priors import SectorPrior
from dcf_project.engine_v3.terminal import WACC_GUARD_GAP

#: Sampled normalized-margin targets are kept strictly positive: the value-driver
#: terminal needs positive normalized operating profit (negative-NOPAT names are a
#: typed skip upstream), and 60% NOPAT margin is beyond any sustained large-cap record.
MARGIN_TARGET_BOUNDS = (0.002, 0.60)
WACC_BOUNDS = (0.045, 0.20)
#: Below this effective sample the distribution is reported as unavailable.
MIN_EFFECTIVE_DRAWS = 50


def _draw_summary(clipped: np.ndarray, raw: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(clipped)),
        "p5": float(np.percentile(clipped, 5)),
        "p95": float(np.percentile(clipped, 95)),
        "clipped": int(np.sum(clipped != raw)),
    }


def run_driver_mc(
    drivers: DriverEstimates,
    persistence: PersistenceFit,
    sector_prior: SectorPrior,
    cfg: V3Config,
    wacc: float,
    g_terminal: float,
    terminal_growth_floor: float,
    terminal_growth_cap: float,
    latest_cash: float | None,
    latest_debt: float | None,
    latest_shares: float | None,
    market: MarketSnapshot,
    half_lives: HalfLifeEstimates | None = None,
    horizon: int = 40,
    sector_tilt: float = 0.0,
) -> dict[str, Any] | None:
    """Sample the drivers, re-run the skeleton per draw, return the distribution payload.

    Returns ``{"distribution": ..., "percentiles": {p10,p50,p90}}`` (the shape
    ``assemble.run_economic_dcf`` consumes), or None when the MC layer is disabled
    (``mc_draws<=0``) or degenerate (too few finite draws) -- the deterministic
    skeleton then stands alone.
    """
    n = int(cfg.mc_draws)
    if n <= 0:
        return None
    rng = np.random.default_rng(cfg.seed)

    g_raw = rng.normal(drivers.g_base.value, max(drivers.g_base.se, 0.0), n)
    m_raw = rng.normal(drivers.margin_target_company.value, max(drivers.margin_target_company.se, 0.0), n)
    phi_raw = rng.normal(persistence.phi, max(persistence.phi_se, 0.0), n)
    gt_raw = rng.normal(g_terminal, max(cfg.terminal_g_sigma, 0.0), n)
    if cfg.sample_wacc:
        wacc_raw = rng.normal(wacc, max(cfg.wacc_sigma, 0.0), n)
        wacc_draws = np.clip(wacc_raw, *WACC_BOUNDS)
    else:
        wacc_raw = wacc_draws = np.full(n, float(wacc))

    g_draws = np.clip(g_raw, *YOY_GROWTH_BAND)
    m_draws = np.clip(m_raw, *MARGIN_TARGET_BOUNDS)
    phi_draws = np.clip(phi_raw, 0.0, 0.99)
    # Terminal growth re-clipped to the same Grinold-Kroner band per draw (the ceiling
    # moves with WACC when WACC is sampled) so no draw can violate the spread guard.
    gt_ceiling = np.minimum(terminal_growth_cap, wacc_draws - WACC_GUARD_GAP)
    gt_draws = np.clip(gt_raw, terminal_growth_floor, gt_ceiling)

    horizon_years = horizon / 4.0
    excess_base = max(persistence.excess_0, 0.0)

    iv = np.full(n, np.nan)
    rev_paths = np.full((n, horizon), np.nan)
    fcf_paths = np.full((n, horizon), np.nan)
    failed = 0
    for i in range(n):
        try:
            frame, _ = build_fade_forecast(
                drivers,
                persistence,
                sector_prior,
                cfg,
                float(wacc_draws[i]),
                float(gt_draws[i]),
                half_lives=half_lives,
                horizon=horizon,
                sector_tilt=sector_tilt,
                g_base_override=float(g_draws[i]),
                margin_target_override=float(m_draws[i]),
                phi_override=float(phi_draws[i]),
            )
            # The persistence->terminal coupling is preserved per draw: phi uncertainty
            # propagates into the surviving terminal excess (same formula as the fit).
            excess_t = float(np.clip(excess_base * phi_draws[i] ** horizon_years, 0.0, cfg.moat_premium_cap))
            valuation = value_fade_path(
                frame,
                float(wacc_draws[i]),
                float(gt_draws[i]),
                float(wacc_draws[i]) + excess_t,
                latest_cash,
                latest_debt,
                latest_shares,
                market,
            )
            iv[i] = valuation["intrinsic"]
            rev_paths[i] = frame["revenue"].to_numpy()
            fcf_paths[i] = frame["fcf"].to_numpy()
        except Exception:  # a pathological draw is data, not a crash: count it
            failed += 1
    ok = np.isfinite(iv)
    n_effective = int(ok.sum())
    if n_effective < min(n, MIN_EFFECTIVE_DRAWS):
        return None

    ivv = iv[ok]
    price = market.price
    counts, edges = np.histogram(ivv, bins=50)
    distribution: dict[str, Any] = {
        "engine": "v3",
        "n_draws": n,
        "n_effective": n_effective,
        "failed_draws": int(failed),
        "seed": int(cfg.seed),
        "iv": distribution_outputs(ivv, price),
        "signals": candidate_signals(ivv, price),
        "histogram": {"bin_edges": edges.tolist(), "counts": counts.tolist()},
        "fan_chart": {"revenue": _fan(rev_paths[ok]), "fcf": _fan(fcf_paths[ok])},
        "driver_draws": {
            "g_base": _draw_summary(g_draws, g_raw),
            "margin_target": _draw_summary(m_draws, m_raw),
            "phi": _draw_summary(phi_draws, phi_raw),
            "terminal_growth": _draw_summary(gt_draws, gt_raw),
            "wacc": _draw_summary(wacc_draws, wacc_raw) if cfg.sample_wacc else None,
        },
    }
    percentiles = {
        "p10": float(np.percentile(ivv, 10)),
        "p50": float(np.percentile(ivv, 50)),
        "p90": float(np.percentile(ivv, 90)),
    }
    return {"distribution": distribution, "percentiles": percentiles}
