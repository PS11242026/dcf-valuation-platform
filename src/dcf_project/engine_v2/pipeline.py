"""BayesianDCFPipeline -- the engine_v2 entry point.

Mirrors ``scripts/utilities/historical_predictive_power._PITPipeline``: a
:class:`DCFPipeline` subclass that reuses the entire frozen v1 sequence
(load -> reconstruct -> classify -> fit state space -> forecast -> overlays -> baseline DCF)
and then runs the posterior-coherent Monte Carlo on top via
:func:`engine_v2.assemble.run_bayesian_dcf`. ``pipeline.py`` / ``state_space.py`` /
``kalman.py`` / ``dcf.py`` stay byte-identical; the v1 module-level helpers are imported and
reused rather than copied.

It deliberately values the *raw* state-space DCF (no ``apply_valuation_regime`` overlay) --
the same regime-free intrinsic the backtests use -- so the only change vs v1 is the forecast
distribution.
"""
from __future__ import annotations

from dataclasses import replace

from dcf_project.contracts import DCFResult
from dcf_project.data_sources.consensus import resolve_forward_consensus
from dcf_project.engine_v2.assemble import run_bayesian_dcf
from dcf_project.models.classification import resolve_company_type
from dcf_project.models.overlays import apply_forecast_overlays
from dcf_project.models.state_space import FCFStateSpaceModel
from dcf_project.paths import REPORTS
from dcf_project.pipeline import (
    DCFPipeline,
    _diagnostics,
    _forecast_decomposition,
    _forecast_sanity,
    _last_valid,
    _latest_debt_stock,
    _stabilized_base_fcf,
)
from dcf_project.preprocessing.reconstruction import reconstruct_historical_fcf, state_matrix
from dcf_project.valuation.dcf import build_state_space_dcf_result
from dcf_project.valuation.regimes import classify_valuation_regime


class BayesianDCFPipeline(DCFPipeline):
    """Bayesian state-space Monte Carlo valuation (engine_v2)."""

    #: Where posterior draws are written. Overridable (e.g. in tests) to avoid
    #: touching the live ``reports/`` tree. Defaults to reports/research_v2/posterior.
    posterior_dir = None
    #: Whether to persist posterior draws. Backtests set this False (hundreds of fits).
    save_posterior = True

    def run(self, ticker: str, horizon_quarters: int = 20) -> DCFResult:
        if horizon_quarters != self.settings.explicit_forecast_quarters:
            raise ValueError(
                f"DCF valuation requires a fixed {self.settings.explicit_forecast_quarters}-quarter "
                "explicit forecast horizon before terminal value."
            )
        bundle = self.load_data(ticker)
        reconstruction = reconstruct_historical_fcf(
            bundle.periods,
            use_proxy_invested_capital=self.settings.use_proxy_invested_capital,
            sbc_addback_fraction=getattr(self.settings, "sbc_addback_fraction", 0.0),
        )
        history = reconstruction.history
        effective_company_type, classification = resolve_company_type(ticker, self.settings.company_type, history)
        settings = replace(self.settings, company_type=effective_company_type)  # type: ignore[arg-type]
        model = FCFStateSpaceModel(
            company_type=settings.company_type,
            calibration_mode=settings.valuation_mode,
            ticker=ticker.upper(),
        ).fit(state_matrix(history))
        base_fcf = _stabilized_base_fcf(history)
        forecast = model.forecast_financials(
            base_revenue=float(history.iloc[-1]["revenue"]),
            base_fcf=base_fcf,
            steps=horizon_quarters,
            base_invested_capital=float(history.iloc[-1]["invested_capital"]),
            tax_rate=float(history.iloc[-1]["tax_rate"]),
        )
        consensus = resolve_forward_consensus(ticker, history, self.settings)
        forecast_financials, overlay_diagnostics = apply_forecast_overlays(
            ticker=ticker.upper(),
            financials=forecast.financials,
            history=history,
            consensus=consensus,
            settings=settings,
        )
        latest_cash = _last_valid(history, "cash")
        latest_debt = _latest_debt_stock(history)
        current_shares = _last_valid(history, "shares")
        latest_shares = current_shares or bundle.market.shares_outstanding
        diagnostics = _diagnostics(history, model, forecast_financials, forecast.states)
        diagnostics.update(overlay_diagnostics)
        diagnostics["forecast_sanity"] = _forecast_sanity(forecast.states, forecast.financials)
        diagnostics["prior_influence"] = model.prior_influence
        diagnostics["forecast_decomposition"] = _forecast_decomposition(
            forecast.financials,
            forecast_financials,
            overlay_diagnostics.get("field_source_map", []),
            model.prior_influence or {},
        )
        diagnostics["company_classification"] = classification
        diagnostics["effective_company_type"] = settings.company_type
        diagnostics["resolved_company_type"] = settings.company_type
        diagnostics["valuation_regime"] = classify_valuation_regime(ticker, settings)
        # Additive (v2-only, diagnostic): surface the trailing revenue/margin the lean
        # state-space valuation path does not carry, so engine_v2.expectation_gap can solve
        # the reverse DCF (market-implied expectations) for the Gate-5 candidate signals.
        # Never consumed by valuation or construction.
        _last = history.iloc[-1]
        _rev = float(_last["revenue"]) if ("revenue" in history and _last["revenue"]) else None
        _fcf_ttm = float(_last["fcf"]) if "fcf" in history else None
        diagnostics["expectation_gap_inputs"] = {
            "latest_revenue": _rev,
            "latest_margin": (_fcf_ttm / _rev) if (_rev and _fcf_ttm is not None) else None,
            "latest_maintenance_capex": float(_last["net_capex"]) if "net_capex" in history else None,
        }
        baseline = build_state_space_dcf_result(
            ticker=ticker,
            forecast_financials=forecast_financials,
            base_fcf=base_fcf,
            fcf_growth_rate=forecast.fcf_growth_rate,
            wacc_inputs=bundle.wacc,
            market=bundle.market,
            latest_cash=latest_cash,
            latest_debt=latest_debt,
            latest_shares=latest_shares,
            sources=bundle.sources,
            warnings=bundle.warnings + reconstruction.warnings,
            settings=settings,
            diagnostics=diagnostics,
        )
        posterior_path = None
        if self.save_posterior:
            base_dir = self.posterior_dir or (REPORTS / "research_v2" / "posterior")
            posterior_path = base_dir / f"{ticker.upper()}.npz"
        return run_bayesian_dcf(
            baseline=baseline,
            model=model,
            history=history,
            base_fcf=base_fcf,
            wacc_inputs=bundle.wacc,
            market=bundle.market,
            settings=settings,
            posterior_path=posterior_path,
        )
