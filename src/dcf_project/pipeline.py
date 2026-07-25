from __future__ import annotations

from dataclasses import replace

import numpy as np

from dcf_project.config import Settings, get_settings
from dcf_project.contracts import DCFResult
from dcf_project.data_sources.consensus import resolve_forward_consensus
from dcf_project.data_sources.fundamentals import FundamentalsBundle, load_fundamentals
from dcf_project.models.overlays import apply_forecast_overlays
from dcf_project.models.classification import resolve_company_type
from dcf_project.models.state_space import FCFStateSpaceModel
from dcf_project.preprocessing.reconstruction import (
    STATE_COLUMNS,
    reconstruct_historical_fcf,
    state_matrix,
)
from dcf_project.valuation.dcf import build_state_space_dcf_result
from dcf_project.valuation.regimes import apply_valuation_regime, classify_valuation_regime


class DCFPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def load_data(self, ticker: str) -> FundamentalsBundle:
        return load_fundamentals(ticker, self.settings)

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
        forecast_terminal_shares = _last_valid(forecast_financials, "shares")
        latest_shares = current_shares or bundle.market.shares_outstanding
        diagnostics = _diagnostics(history, model, forecast_financials, forecast.states)
        diagnostics.update(overlay_diagnostics)
        diagnostics["share_count_trace"] = {
            "valuation_denominator": latest_shares,
            "valuation_denominator_source": "historical_latest" if current_shares else "market_snapshot",
            "historical_latest_shares": current_shares,
            "market_snapshot_shares": bundle.market.shares_outstanding,
            "forecast_terminal_shares": forecast_terminal_shares,
            "forecast_share_path_used_for_denominator": False,
            "buyback_rate": overlay_diagnostics.get("buyback_rate"),
            "note": "DCF per-share value uses current diluted shares; forecast shares are retained only for overlay audit.",
        }
        diagnostics["forecast_sanity"] = _forecast_sanity(forecast.states, forecast.financials)
        diagnostics["prior_influence"] = model.prior_influence
        diagnostics["forecast_decomposition"] = _forecast_decomposition(
            forecast.financials,
            forecast_financials,
            overlay_diagnostics.get("field_source_map", []),
            model.prior_influence or {},
        )
        diagnostics["company_classification"] = classification
        diagnostics["configured_company_type"] = self.settings.company_type
        diagnostics["effective_company_type"] = settings.company_type
        diagnostics["resolved_company_type"] = settings.company_type
        diagnostics["classification_method"] = classification.get("method")
        diagnostics["valuation_regime"] = classify_valuation_regime(ticker, settings)
        result = build_state_space_dcf_result(
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
        return apply_valuation_regime(result, settings)

    def inspect_data(self, ticker: str) -> dict[str, object]:
        bundle = self.load_data(ticker)
        reconstruction = reconstruct_historical_fcf(
            bundle.periods,
            use_proxy_invested_capital=self.settings.use_proxy_invested_capital,
        )
        history = reconstruction.history
        return {
            "ticker": ticker.upper(),
            "period_count": len(bundle.periods),
            "complete_period_count": len(history),
            "first_period": str(history.iloc[0]["date"]),
            "last_period": str(history.iloc[-1]["date"]),
            "warnings": bundle.warnings + reconstruction.warnings,
            "market": bundle.market.model_dump(mode="json"),
            "wacc": bundle.wacc.model_dump(mode="json"),
            "sources": [source.model_dump(mode="json") for source in bundle.sources],
        }

    def backtest(self, ticker: str, horizon_quarters: int = 4, min_train_periods: int = 12) -> dict[str, object]:
        bundle = self.load_data(ticker)
        reconstruction = reconstruct_historical_fcf(
            bundle.periods,
            use_proxy_invested_capital=self.settings.use_proxy_invested_capital,
        )
        history = reconstruction.history
        errors = []
        directions = []
        for end in range(min_train_periods, len(history) - horizon_quarters):
            train = history.iloc[:end].copy()
            actual = float(history.iloc[end + horizon_quarters - 1]["fcf"])
            base = float(train.iloc[-1]["fcf"])
            try:
                model = FCFStateSpaceModel(
                    company_type=self.settings.company_type,
                    calibration_mode=self.settings.valuation_mode,
                    ticker=ticker.upper(),
                ).fit(state_matrix(train))
                forecast = model.forecast_financials(
                    float(train.iloc[-1]["revenue"]),
                    base,
                    horizon_quarters,
                    base_invested_capital=float(train.iloc[-1]["invested_capital"]),
                    tax_rate=float(train.iloc[-1]["tax_rate"]),
                )
            except Exception:
                continue
            predicted = float(forecast.financials.iloc[-1]["fcf"])
            actual_growth = (actual - base) / max(abs(base), 1.0)
            predicted_growth = (predicted - base) / max(abs(base), 1.0)
            errors.append(predicted_growth - actual_growth)
            directions.append((predicted_growth >= 0.0) == (actual_growth >= 0.0))
        if not errors:
            return {
                "ticker": ticker.upper(),
                "windows": 0,
                "warnings": reconstruction.warnings + ["Not enough history for backtest windows."],
            }
        abs_errors = [abs(error) for error in errors]
        sq_errors = [error * error for error in errors]
        return {
            "ticker": ticker.upper(),
            "windows": len(errors),
            "mae": sum(abs_errors) / len(abs_errors),
            "rmse": (sum(sq_errors) / len(sq_errors)) ** 0.5,
            "directional_accuracy": sum(directions) / len(directions),
            "warnings": bundle.warnings + reconstruction.warnings,
        }


def _stabilized_base_fcf(history) -> float:
    """Representative quarterly FCF base (trailing-4-quarter median).

    A single quarter's FCF is highly volatile (capex/working-capital timing), e.g.
    AMZN's latest quarter was -$78.6B while its TTM is positive. base_fcf does not
    drive intrinsic value (the forecast path is revenue/margin driven), but it seeds
    the reported FCF-growth diagnostic; using one anomalous quarter produces an
    economically invalid headline growth rate. The trailing-4q median is robust.
    """
    if "fcf" not in history or history.empty:
        return 0.0
    tail = history["fcf"].dropna().tail(4)
    if tail.empty:
        return float(history["fcf"].iloc[-1]) if "fcf" in history else 0.0
    return float(tail.median())


def _last_valid(frame, column: str) -> float | None:
    if column not in frame:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _latest_debt_stock(frame) -> float | None:
    """Most complete recent debt reading (trailing-4-quarter max).

    Debt is a sticky balance-sheet stock; some issuers disclose total debt only
    annually, leaving interim quarters with an incomplete (current-portion-only)
    figure. The trailing max recovers the full annual reading without overstating
    leverage for consistent reporters.
    """
    if "debt" not in frame:
        return None
    series = frame["debt"].dropna()
    if series.empty:
        return None
    return float(series.tail(4).max())


def _diagnostics(history, model: FCFStateSpaceModel, forecast_financials, forecast_states=None) -> dict[str, object]:
    return {
        "historical_financials": _frame_records(history.tail(32)),
        "historical_fcf": _frame_records(
            history[
                [
                    "date",
                    "revenue",
                    "ebit",
                    "tax_expense",
                    "tax_rate",
                    "nopat",
                    "da",
                    "capex",
                    "nwc",
                    "delta_nwc",
                    "fcf",
                    "net_capex",
                    "reinvestment",
                    "reinvestment_rate",
                    "operating_assets",
                    "operating_liabilities",
                    "cash_excluded",
                    "goodwill",
                    "intangible_assets",
                    "lease_assets",
                    "accounting_ic",
                    "proxy_ic",
                    "normalized_ic",
                    "ic_used_for_roic",
                    "proxy_used",
                    "goodwill_carry_forward_age",
                    "intangible_assets_carry_forward_age",
                    "operating_lease_liability_carry_forward_age",
                    "goodwill_stale_carry_forward_flag",
                    "intangible_assets_stale_carry_forward_flag",
                    "operating_lease_liability_stale_carry_forward_flag",
                    "raw_invested_capital_reconstructed",
                    "raw_invested_capital_qoq_change",
                    "invested_capital",
                    "invested_capital_qoq_change",
                    "invested_capital_continuity_flag",
                    "roic",
                ]
            ].tail(32)
        ),
        "state_vector_columns": STATE_COLUMNS,
        "state_vector_history": _frame_records(history[["date"] + STATE_COLUMNS].tail(32)),
        "state_vector": model.result.filtered_state.tolist() if model.result else None,
        "state_vector_means": model.mu.tolist() if model.mu is not None else None,
        "transition_matrix": model.phi.tolist() if model.phi is not None else None,
        "empirical_transition_matrix": model.empirical_phi.tolist() if model.empirical_phi is not None else None,
        "transition_eigenvalues": np.linalg.eigvals(model.phi).real.tolist() if model.phi is not None else None,
        "process_covariance": model.q.tolist() if model.q is not None else None,
        "observation_covariance": model.r.tolist() if model.r is not None else None,
        "state_standardization": {
            "center": model.standardization_center.tolist() if model.standardization_center is not None else None,
            "scale": model.standardization_scale.tolist() if model.standardization_scale is not None else None,
        },
        "smoothed_state_path": model.result.smoothed_states.tolist() if model.result else None,
        "forecast_latent_path": forecast_states.tolist() if forecast_states is not None else None,
        "forecast_quarterly_path": _frame_records(forecast_financials),
        "fcf_path": [float(x) for x in forecast_financials["fcf"].to_list()],
        "invested_capital_components": _frame_records(
            history[
                [
                    "date",
                    "operating_assets",
                    "operating_liabilities",
                    "cash_excluded",
                    "goodwill",
                    "intangible_assets",
                    "lease_assets",
                    "accounting_ic",
                    "proxy_ic",
                    "normalized_ic",
                    "ic_used_for_roic",
                    "proxy_used",
                    "goodwill_carry_forward_age",
                    "intangible_assets_carry_forward_age",
                    "operating_lease_liability_carry_forward_age",
                    "goodwill_stale_carry_forward_flag",
                    "intangible_assets_stale_carry_forward_flag",
                    "operating_lease_liability_stale_carry_forward_flag",
                    "raw_invested_capital_reconstructed",
                    "raw_invested_capital_qoq_change",
                    "invested_capital",
                    "invested_capital_qoq_change",
                    "invested_capital_continuity_flag",
                ]
            ].tail(32)
        ),
        "invested_capital_continuity_flags": _invested_capital_flags(history),
    }


def _frame_records(frame) -> list[dict[str, object]]:
    rows = []
    for record in frame.to_dict(orient="records"):
        row = {}
        for key, value in record.items():
            if hasattr(value, "isoformat"):
                row[key] = value.isoformat()
            elif isinstance(value, float) and not np.isfinite(value):
                row[key] = None
            else:
                row[key] = value
        rows.append(row)
    return rows


def _invested_capital_flags(history) -> list[dict[str, object]]:
    if "invested_capital_continuity_flag" not in history:
        return []
    flagged = history[history["invested_capital_continuity_flag"].fillna(False)]
    return _frame_records(
        flagged[
            [
                "date",
                "invested_capital",
                "invested_capital_qoq_change",
                "operating_assets",
                "operating_liabilities",
            ]
        ]
    )


def _forecast_sanity(states, raw_financials) -> dict[str, object]:
    raw_states = np.asarray(states, dtype=float)
    growth = raw_states[:, 0] if raw_states.ndim == 2 and raw_states.shape[1] >= 1 else np.asarray([])
    return {
        "raw_growth_cagr": float(np.prod(1.0 + growth) ** (4.0 / len(growth)) - 1.0) if len(growth) else None,
        "raw_growth_path": [float(x) for x in growth.tolist()] if len(growth) else [],
        "raw_margin_path": _series(raw_financials, "raw_model_operating_margin"),
        "raw_roic_path": _series(raw_financials, "raw_model_roic"),
        "raw_reinvestment_path": _series(raw_financials, "raw_model_reinvestment_rate"),
    }


def _forecast_decomposition(raw_financials, final_financials, field_source_map, prior_influence) -> list[dict[str, object]]:
    rows = []
    prior_mu = (prior_influence.get("long_run_mean") or {}) if isinstance(prior_influence, dict) else {}
    source_by_quarter = {
        int(row.get("quarter", idx + 1)): row
        for idx, row in enumerate(field_source_map if isinstance(field_source_map, list) else [])
        if isinstance(row, dict)
    }
    for idx, final in final_financials.head(20).iterrows():
        quarter = int(final.get("quarter", idx + 1))
        raw = raw_financials.iloc[idx]
        sources = source_by_quarter.get(quarter, {})
        rows.append(
            {
                "quarter": float(quarter),
                "revenue": _decomp_field(raw, final, sources, "revenue", None),
                "growth": _decomp_field(raw, final, sources, "revenue_growth", prior_mu.get("revenue_growth")),
                "operating_margin": _decomp_field(raw, final, sources, "operating_margin", prior_mu.get("operating_margin")),
                "nopat": _decomp_field(raw, final, sources, "nopat", None),
                "invested_capital": _decomp_field(raw, final, sources, "invested_capital", None),
                "roic": _decomp_field(raw, final, sources, "roic", prior_mu.get("roic")),
                "reinvestment": _decomp_field(raw, final, sources, "reinvestment", None),
                "reinvestment_rate": _decomp_field(raw, final, sources, "reinvestment_rate", prior_mu.get("reinvestment_rate")),
                "fcf": _decomp_field(raw, final, sources, "fcf", None),
            }
        )
    return rows


def _decomp_field(raw, final, sources, field: str, prior_info) -> dict[str, object]:
    source_info = sources.get(field, {}) if isinstance(sources, dict) else {}
    prior_value = prior_info.get("prior_adjusted_value") if isinstance(prior_info, dict) else None
    raw_model = raw.get(field) if field in raw else None
    final_value = final.get(field) if field in final else None
    consensus_value = source_info.get("blended_value") if isinstance(source_info, dict) else None
    return {
        "raw_model": _float_or_none(raw_model),
        "prior": _float_or_none(prior_value),
        "consensus": _float_or_none(consensus_value),
        "overlay": _float_or_none(source_info.get("capped_value")) if isinstance(source_info, dict) else None,
        "final": _float_or_none(final_value),
        "source": source_info.get("source") if isinstance(source_info, dict) else "model",
        "clamp_triggered": bool(source_info.get("clamp_triggered", False)) if isinstance(source_info, dict) else False,
    }


def _series(frame, column: str) -> list[float]:
    if column not in frame:
        return []
    return [float(x) for x in frame[column].astype(float).to_list()]


def _float_or_none(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None
