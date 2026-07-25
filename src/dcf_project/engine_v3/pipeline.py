"""EconomicDCFPipeline -- the engine_v3 entry point.

A :class:`DCFPipeline` subclass that reuses the frozen v1 data layer verbatim
(``load_data`` inherited, so the backtests' PIT bundle-injection pattern works
identically) and the frozen historical reconstruction, then replaces the forecast
layer with the economic-drivers skeleton:

    drivers -> ROIC persistence/CAP -> fade forecast -> GK/value-driver terminal -> MC

Deliberately NOT used (the v1 forecast machinery this engine replaces): the Kalman
state-space fit, consensus blending, optionality regimes, the 3-FCF-base ensemble,
buyback/R&D overlays. Classification is computed for diagnostics/stratification only.
"""
from __future__ import annotations

import numpy as np

from dcf_project.contracts import DCFResult
from dcf_project.engine_v3.assemble import run_economic_dcf
from dcf_project.engine_v3.config import V3Config
from dcf_project.engine_v3.drivers import extract_drivers
from dcf_project.engine_v3.fade import build_fade_forecast
from dcf_project.engine_v3.halflives import estimate_half_lives
from dcf_project.engine_v3.mc import run_driver_mc
from dcf_project.engine_v3.persistence import fit_persistence
from dcf_project.engine_v3.sector_priors import prior_for, sector_for_ticker
from dcf_project.engine_v3.terminal import gk_terminal_growth
from dcf_project.models.classification import resolve_company_type
from dcf_project.pipeline import DCFPipeline, _frame_records, _last_valid, _latest_debt_stock
from dcf_project.preprocessing.reconstruction import reconstruct_historical_fcf

#: Bound on the sector-state growth tilt in annual growth units (±2pp at most).
SECTOR_TILT_BOUND = 0.02


class EconomicDCFPipeline(DCFPipeline):
    """Economically grounded fade/CAP/persistence valuation (engine_v3)."""

    #: Optional sector-state composite score (z-scored), injected by backtest runners
    #: for the G5 ablation. None (the default, and all live runs while kappa=0) means
    #: no tilt. The tilt shifts the near-term growth anchor by kappa·score, bounded.
    sector_state_score: float | None = None

    def run(self, ticker: str, horizon_quarters: int | None = None) -> DCFResult:
        # The explicit fade horizon is a v3 economic assumption (default 40q = 10y),
        # deliberately decoupled from v1's fixed ``explicit_forecast_quarters`` (20) so
        # the frozen v1 engine is never perturbed. Callers may override it explicitly.
        cfg = V3Config.from_settings(self.settings)
        if horizon_quarters is None:
            horizon_quarters = cfg.horizon_quarters
        bundle = self.load_data(ticker)
        reconstruction = reconstruct_historical_fcf(
            bundle.periods,
            use_proxy_invested_capital=self.settings.use_proxy_invested_capital,
            sbc_addback_fraction=getattr(self.settings, "sbc_addback_fraction", 0.0),
        )
        history = reconstruction.history
        wacc = bundle.wacc.wacc
        country = getattr(self.settings, "country", "US")
        if country == "IN":
            from dcf_project.data_sources.india.sectors import (
                india_archetype_for_sector,
                india_sector_for_ticker,
            )

            sector = india_sector_for_ticker(ticker)
        else:
            sector = sector_for_ticker(ticker)
        sector_prior = prior_for(country, sector)

        effective_company_type, classification = resolve_company_type(
            ticker, self.settings.company_type, history
        )
        # India override: the asset-intensity classifier needs a capex line the NSE results
        # XBRL does not carry, so it mislabels every Indian name ``asset_light``. When the type
        # is auto-resolved, relabel the archetype from the (reliable) GICS sector instead. This
        # is diagnostic/stratification only -- the fade/terminal/persistence read ``sector`` and
        # its prior, never this archetype -- so the headline valuation is unchanged.
        if country == "IN" and str(self.settings.company_type or "auto") == "auto":
            archetype = india_archetype_for_sector(sector)
            if archetype:
                effective_company_type = archetype
                classification = {
                    "method": "india_sector_archetype",
                    "resolved_company_type": archetype,
                    "sector": sector,
                }

        drivers = extract_drivers(history, wacc)
        persistence = fit_persistence(
            drivers.excess_roic_annual,
            drivers.excess_0,
            sector,
            sector_prior.phi,
            drivers.quality,
            cfg,
            horizon_years=horizon_quarters / 4.0,
        )
        g_terminal, gk_detail = gk_terminal_growth(self.settings, cfg, wacc)
        tilt = self._sector_tilt(cfg)

        # Company-specific fade half-lives (growth + margin), estimated from the same
        # history and shrunk to sector priors. The blended margin target is the anchor
        # the margin reverts toward (computed identically inside build_fade_forecast).
        margin_w = float(sector_prior.anchor_weight)
        margin_target_blend = (
            (1.0 - margin_w) * drivers.margin_target_company.value + margin_w * sector_prior.margin_anchor
        )
        half_lives = estimate_half_lives(
            history, g_terminal, margin_target_blend, sector_prior, persistence, cfg
        )

        frame, fade_audit = build_fade_forecast(
            drivers,
            persistence,
            sector_prior,
            cfg,
            wacc,
            g_terminal,
            half_lives=half_lives,
            horizon=horizon_quarters,
            sector_tilt=tilt,
        )

        latest_cash = _last_valid(history, "cash")
        latest_debt = _latest_debt_stock(history)
        latest_shares = _last_valid(history, "shares") or bundle.market.shares_outstanding

        mc_payload = run_driver_mc(
            drivers,
            persistence,
            sector_prior,
            cfg,
            wacc,
            g_terminal,
            self.settings.terminal_growth_floor,
            self.settings.terminal_growth_cap,
            latest_cash,
            latest_debt,
            latest_shares,
            bundle.market,
            half_lives=half_lives,
            horizon=horizon_quarters,
            sector_tilt=tilt,
        )

        driver_audit = dict(fade_audit)
        driver_audit["grinold_kroner"] = gk_detail
        driver_audit["half_lives"] = {
            "growth_years": half_lives.h_growth_years,
            "margin_years": half_lives.h_margin_years,
            "growth": half_lives.growth_audit,
            "margin": half_lives.margin_audit,
        }
        driver_audit["persistence"] = {
            "phi": persistence.phi,
            "phi_se": persistence.phi_se,
            "phi_company": persistence.phi_company,
            "n_years": persistence.n_years,
            "shrink_weight": persistence.shrink_weight,
            "excess_terminal": persistence.excess_terminal,
            "cap_years": persistence.cap_years,
            "sector": persistence.sector,
            "notes": persistence.notes,
        }
        driver_audit["drivers"] = {
            "g_base": {"value": drivers.g_base.value, "se": drivers.g_base.se, "window": drivers.g_base.window},
            "margin_current": drivers.margin_current,
            "margin_target_company": {
                "value": drivers.margin_target_company.value,
                "se": drivers.margin_target_company.se,
                "window": drivers.margin_target_company.window,
            },
            "roic_annual_0": drivers.roic_annual_0,
            "excess_roic_0": drivers.excess_0,
            "base_revenue_ttm": drivers.base_revenue_ttm,
            "base_nopat_ttm": drivers.base_nopat_ttm,
            "base_invested_capital": drivers.base_invested_capital,
            "tax_rate": drivers.tax_rate,
            "quality": drivers.quality,
            "fcf_margin_current": drivers.fcf_margin_current,
            "notes": drivers.notes,
        }

        # Reporting currency + filing provenance, so the workbook labels money in the right
        # symbol (₹ vs $) and names the source it was pulled from. US runs resolve to USD/EDGAR
        # (unchanged); India to INR/NSE. Presentation only -- never read by the valuation math.
        currency = (getattr(bundle.market, "currency", None) or ("INR" if country == "IN" else "USD")).upper()
        filing_source = "NSE" if country == "IN" else "EDGAR"
        diagnostics: dict[str, object] = {
            "company_classification": classification,
            "effective_company_type": effective_company_type,
            "resolved_company_type": effective_company_type,
            "sector": sector,
            "currency": currency,
            "filing_source": filing_source,
            "engine": "engine_v3 economic_fade",
            # Reconstructed quarterly history (EDGAR-sourced) for the workbook's Historical
            # Financials sheet. The v1 pipeline emits this in its own _diagnostics(); the v3
            # forecast layer replaces that path, so surface it explicitly here too. Each record
            # carries date + fiscal_period, from which the report derives the SEC filing.
            "historical_financials": _frame_records(history.tail(20)),
        }

        ronic_terminal = wacc + persistence.excess_terminal
        return run_economic_dcf(
            ticker=ticker,
            frame=frame,
            driver_audit=driver_audit,
            wacc_inputs=bundle.wacc,
            market=bundle.market,
            latest_cash=latest_cash,
            latest_debt=latest_debt,
            latest_shares=latest_shares,
            terminal_growth=g_terminal,
            ronic_terminal=ronic_terminal,
            sources=bundle.sources,
            warnings=bundle.warnings + reconstruction.warnings,
            cfg=cfg,
            diagnostics=diagnostics,
            mc_payload=mc_payload,
        )

    def _sector_tilt(self, cfg: V3Config) -> float:
        if self.sector_state_score is None or not cfg.sector_state_kappa:
            return 0.0
        raw = cfg.sector_state_kappa * float(self.sector_state_score) * 0.01
        return float(np.clip(raw, -SECTOR_TILT_BOUND, SECTOR_TILT_BOUND))
