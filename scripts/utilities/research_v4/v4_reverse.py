"""V3-native reverse-DCF for Research_V4 (research-only; engine untouched).

Given a loaded fundamentals bundle, this reconstructs the *exact* engine_v3 intermediate
state (drivers, persistence, terminal anchor, half-lives) by calling the engine's own
building blocks, then re-prices the company under perturbed drivers via the engine's own
``build_fade_forecast`` (which already exposes ``g_base_override`` / ``margin_target_override``
/ ``phi_override``) and ``value_fade_path``. Root-finding those re-pricings against the
current market price backs out what the market is implying:

    implied effective growth   -- hold margin/RONIC fixed, solve g so IV == price
    implied normalized margin   -- hold growth/RONIC fixed, solve margin so IV == price
    implied terminal RONIC      -- closed-form inversion of the value-driver terminal

This deliberately does NOT touch the frozen v1 ``reverse_dcf_expectations`` path. The
orchestration mirrors ``EconomicDCFPipeline.run`` (engine_v3/pipeline.py); v3 is frozen so
the mirror is stable, and a round-trip test (solve against the engine's own IV) recovers
the engine's own driver, guarding against drift.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dcf_project.data_sources.fundamentals import FundamentalsBundle  # noqa: E402
from dcf_project.engine_v3.assemble import value_fade_path  # noqa: E402
from dcf_project.engine_v3.config import V3Config  # noqa: E402
from dcf_project.engine_v3.drivers import DriverEstimates, extract_drivers  # noqa: E402
from dcf_project.engine_v3.fade import build_fade_forecast  # noqa: E402
from dcf_project.engine_v3.halflives import HalfLifeEstimates, estimate_half_lives  # noqa: E402
from dcf_project.engine_v3.persistence import PersistenceFit, fit_persistence  # noqa: E402
from dcf_project.engine_v3.sector_priors import prior_for_sector, sector_for_ticker  # noqa: E402
from dcf_project.engine_v3.terminal import gk_terminal_growth  # noqa: E402
from dcf_project.models.classification import resolve_company_type  # noqa: E402
from dcf_project.pipeline import _last_valid, _latest_debt_stock  # noqa: E402
from dcf_project.preprocessing.reconstruction import reconstruct_historical_fcf  # noqa: E402


@dataclass
class EngineContext:
    """All engine_v3 intermediates needed to re-price one company under perturbed drivers."""

    ticker: str
    drivers: DriverEstimates
    persistence: PersistenceFit
    sector_prior: Any
    cfg: V3Config
    wacc: float
    g_terminal: float
    half_lives: HalfLifeEstimates
    ronic_terminal: float
    latest_cash: float | None
    latest_debt: float | None
    latest_shares: float | None
    market: Any
    g_base_effective: float  # the engine's own effective base growth (g_base + tilt)

    def price_per_share(
        self,
        g_base: float | None = None,
        margin_target: float | None = None,
        phi: float | None = None,
        terminal_growth: float | None = None,
        ronic_terminal: float | None = None,
        half_lives: HalfLifeEstimates | None = None,
    ) -> float:
        """Intrinsic per share under optional driver overrides (None = engine value).

        Driver overrides (g_base/margin_target/phi) are treated as EFFECTIVE values:
        sector_tilt is set to 0 here so the solved-for growth is the effective near-term
        anchor directly comparable to ``g_base_effective`` (in all live runs the tilt is 0
        anyway, kappa=0). ``terminal_growth`` / ``ronic_terminal`` let the root-cause layer
        re-price the terminal block (terminal-growth conservatism / moat-premium) levers.
        """
        g_term = self.g_terminal if terminal_growth is None else float(terminal_growth)
        frame, _ = build_fade_forecast(
            self.drivers,
            self.persistence,
            self.sector_prior,
            self.cfg,
            self.wacc,
            g_term,
            half_lives=self.half_lives if half_lives is None else half_lives,
            horizon=self.cfg.horizon_quarters,
            sector_tilt=0.0,
            g_base_override=self.g_base_effective if g_base is None else g_base,
            margin_target_override=margin_target,
            phi_override=phi,
        )
        valuation = value_fade_path(
            frame,
            self.wacc,
            g_term,
            self.ronic_terminal if ronic_terminal is None else float(ronic_terminal),
            self.latest_cash,
            self.latest_debt,
            self.latest_shares,
            self.market,
        )
        return float(valuation["intrinsic"])


def build_context(settings, bundle: FundamentalsBundle, ticker: str) -> EngineContext:
    """Reconstruct engine_v3 intermediates from a bundle (mirrors EconomicDCFPipeline.run)."""
    cfg = V3Config.from_settings(settings)
    horizon = cfg.horizon_quarters
    reconstruction = reconstruct_historical_fcf(
        bundle.periods,
        use_proxy_invested_capital=settings.use_proxy_invested_capital,
        sbc_addback_fraction=getattr(settings, "sbc_addback_fraction", 0.0),
    )
    history = reconstruction.history
    resolve_company_type(ticker, settings.company_type, history)  # diagnostics only
    wacc = bundle.wacc.wacc
    sector = sector_for_ticker(ticker)
    sector_prior = prior_for_sector(sector)

    drivers = extract_drivers(history, wacc)
    persistence = fit_persistence(
        drivers.excess_roic_annual,
        drivers.excess_0,
        sector,
        sector_prior.phi,
        drivers.quality,
        cfg,
        horizon_years=horizon / 4.0,
    )
    g_terminal, _ = gk_terminal_growth(settings, cfg, wacc)
    margin_w = float(sector_prior.anchor_weight)
    margin_target_blend = (
        (1.0 - margin_w) * drivers.margin_target_company.value + margin_w * sector_prior.margin_anchor
    )
    half_lives = estimate_half_lives(
        history, g_terminal, margin_target_blend, sector_prior, persistence, cfg
    )
    ronic_terminal = wacc + persistence.excess_terminal
    return EngineContext(
        ticker=ticker,
        drivers=drivers,
        persistence=persistence,
        sector_prior=sector_prior,
        cfg=cfg,
        wacc=wacc,
        g_terminal=g_terminal,
        half_lives=half_lives,
        ronic_terminal=ronic_terminal,
        latest_cash=_last_valid(history, "cash"),
        latest_debt=_latest_debt_stock(history),
        latest_shares=_last_valid(history, "shares") or bundle.market.shares_outstanding,
        market=bundle.market,
        g_base_effective=float(drivers.g_base.value),
    )


def _bisect(f: Callable[[float], float], lo: float, hi: float, target: float,
            tol: float = 1e-4, iters: int = 80) -> float | None:
    """Solve f(x) == target on [lo, hi] assuming f monotone increasing. None if unbracketed."""
    try:
        f_lo, f_hi = f(lo) - target, f(hi) - target
    except Exception:  # noqa: BLE001 -- terminal-spread / divestment guards at extremes
        return None
    if f_lo != f_lo or f_hi != f_hi:  # NaN
        return None
    if f_lo * f_hi > 0:
        return None  # target not bracketed -> market implies a driver outside the band
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        try:
            f_mid = f(mid) - target
        except Exception:  # noqa: BLE001
            return None
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def implied_growth(ctx: EngineContext, target_price: float,
                   lo: float = -0.50, hi: float = 1.00) -> float | None:
    """Effective base revenue growth the market price implies (margin/RONIC held at model)."""
    return _bisect(lambda g: ctx.price_per_share(g_base=g), lo, hi, target_price)


def implied_margin(ctx: EngineContext, target_price: float,
                   lo: float = 0.0, hi: float = 0.60) -> float | None:
    """Normalized NOPAT margin the market price implies (growth/RONIC held at model)."""
    return _bisect(lambda m: ctx.price_per_share(margin_target=m), lo, hi, target_price)


def implied_terminal_ronic(ctx: EngineContext, target_price: float) -> float | None:
    """Closed-form terminal RONIC implied by the market, inverting the value-driver terminal.

    Holding the explicit FCF path (hence explicit PV) and terminal NOPAT/g fixed, the
    market's equity price implies a target enterprise value, hence a target terminal value,
    hence (inverting TV = NOPAT_{T+1}(1 - g/RONIC)/(WACC-g)) a market-implied RONIC_T.
    """
    base = ctx.price_per_share()  # model frame & valuation, no override
    frame, _ = build_fade_forecast(
        ctx.drivers, ctx.persistence, ctx.sector_prior, ctx.cfg, ctx.wacc, ctx.g_terminal,
        half_lives=ctx.half_lives, horizon=ctx.cfg.horizon_quarters, sector_tilt=0.0,
        g_base_override=ctx.g_base_effective,
    )
    valuation = value_fade_path(
        frame, ctx.wacc, ctx.g_terminal, ctx.ronic_terminal,
        ctx.latest_cash, ctx.latest_debt, ctx.latest_shares, ctx.market,
    )
    shares = valuation["shares"]
    qw = ctx.wacc / 4.0
    n = len(valuation["projected_fcf"])
    g = ctx.g_terminal
    spread = ctx.wacc - g
    nopat_terminal_annual = float(frame["nopat"].tail(4).sum())
    nopat_t1 = nopat_terminal_annual * (1.0 + g)
    if nopat_t1 <= 0 or spread <= 0:
        return None
    target_ev = target_price * shares + valuation["net_debt"]
    target_pv_terminal = target_ev - valuation["explicit_pv"]
    if target_pv_terminal <= 0:
        return None
    target_tv = target_pv_terminal * (1.0 + qw) ** n
    # TV = NOPAT_t1 * (1 - g/RONIC) / spread  ->  RONIC = g / (1 - TV*spread/NOPAT_t1)
    denom = 1.0 - (target_tv * spread) / nopat_t1
    if denom <= 0:
        return None
    ronic = g / denom
    return float(ronic) if ronic > 0 else None
