from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dcf_project.config import Settings
from dcf_project.data_sources.consensus import ForwardConsensus
from dcf_project.models.priors import prior_for


FORECAST_TRACE_FIELDS = ("revenue_growth", "operating_margin", "reinvestment_rate", "roic", "fcf", "shares")


def apply_forecast_overlays(
    ticker: str,
    financials: pd.DataFrame,
    history: pd.DataFrame,
    consensus: ForwardConsensus,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = financials.copy()
    raw_model_forecast = _forecast_trace(result)
    raw_layer = result.copy()
    requested_consensus_weight = float(np.clip(settings.consensus_weight, 0.0, 1.0))
    if settings.forecast_mode == "state_space":
        requested_consensus_weight = 0.0
    segment_trace = _segment_assumptions_trace(settings)
    target_growth = _segment_growth(segment_trace) or ((1.0 + consensus.revenue_growth_annual) ** 0.25 - 1.0)
    target_margin = _segment_margin(segment_trace) or consensus.ebit_margin
    consensus_forecast = _consensus_trace(result, target_growth, target_margin)
    calibration_trace = _historical_forecast_calibration(history)
    if segment_trace:
        dynamic_trace = _static_consensus_weight_trace(requested_consensus_weight, len(result), "manual_segment_override")
    else:
        dynamic_trace = _dynamic_consensus_weight_trace(
            ticker,
            result,
            history,
            consensus,
            settings,
            requested_consensus_weight,
            calibration_trace,
        )
    consensus_weights = [float(row["weight"]) for row in dynamic_trace["quarterly_weights"]]
    consensus_weight = float(np.mean(consensus_weights)) if consensus_weights else requested_consensus_weight
    result = _blend_revenue_and_margin(
        result,
        target_growth,
        target_margin,
        consensus_weights,
        preserve_trajectory=segment_trace is None,
    )
    if settings.forecast_mode == "state_space" or segment_trace:
        fade_trace = _no_fade_trace(ticker, result, settings, "disabled_for_state_space_or_manual_segment")
    else:
        result, fade_trace = _apply_economic_fade(ticker, result, history, settings)
    blended_layer = result.copy()
    result = _apply_rd_capitalization(ticker, result, settings)
    overlay_layer = result.copy()
    buyback_rate = _estimate_buyback_rate(ticker, history, settings)
    result = _apply_share_forecast(result, history, buyback_rate)
    final_layer = result.copy()
    field_source_map = _field_source_map(
        raw_layer,
        blended_layer,
        overlay_layer,
        final_layer,
        consensus_weight,
        buyback_rate,
    )
    clamp_triggered_map = _clamp_triggered_map(final_layer)
    diagnostics = {
        "forecast_mode": settings.forecast_mode,
        "company_archetype": prior_for(ticker, settings.company_type).archetype,
        "requested_consensus_weight": requested_consensus_weight,
        "consensus_weight": consensus_weight,
        "dynamic_consensus_weight_trace": dynamic_trace,
        "historical_forecast_calibration": calibration_trace,
        "fade_trace": fade_trace,
        "steady_state_trace": fade_trace.get("steady_state_trace"),
        "consensus_provider": consensus.provider,
        "consensus_fallback_used": consensus.fallback_used,
        "consensus_revenue_growth_annual": consensus.revenue_growth_annual,
        "consensus_revenue_growth_quarterly": target_growth,
        "consensus_growth_conversion": {
            "input_basis": "annualized_cagr",
            "formula": "(1 + annual_growth) ** 0.25 - 1",
            "annual_growth": consensus.revenue_growth_annual,
            "quarterly_growth": target_growth,
        },
        "consensus_ebit_margin": consensus.ebit_margin,
        "consensus_fcf_margin": consensus.fcf_margin,
        "buyback_rate": buyback_rate,
        "r_and_d_capitalized": bool(settings.capitalize_rd),
        "segment_mode": settings.segment_mode,
        "segment_assumptions": segment_trace,
        "raw_model_forecast": raw_model_forecast,
        "consensus_forecast": consensus_forecast,
        "blended_forecast": _forecast_trace(blended_layer),
        "overlay_forecast": _forecast_trace(overlay_layer),
        "final_forecast": _forecast_trace(final_layer),
        "field_source_map": field_source_map,
        "clamp_triggered_map": clamp_triggered_map,
    }
    return result, diagnostics


def _blend_revenue_and_margin(
    financials: pd.DataFrame,
    target_quarterly_growth: float,
    target_margin: float | None,
    weight: float | list[float],
    *,
    preserve_trajectory: bool = True,
) -> pd.DataFrame:
    weights = _weight_schedule(weight, len(financials))
    if max(weights, default=0.0) <= 0.0:
        return financials
    result = financials.copy()
    for column in ["da", "capex", "delta_nwc", "maintenance_capex", "sbc_adjustment"]:
        if column in result:
            result[f"_{column}_ratio"] = result[column] / result["revenue"].replace(0.0, np.nan)
    if preserve_trajectory:
        avg_weight = float(np.mean(weights)) if weights else 0.0
        model_growth_mean = float(result["revenue_growth"].astype(float).mean())
        growth_shift = avg_weight * (target_quarterly_growth - model_growth_mean)
        result["revenue_growth"] = result["revenue_growth"].astype(float) + growth_shift
        if target_margin is not None:
            model_margin_mean = float(result["operating_margin"].astype(float).mean())
            margin_shift = avg_weight * (target_margin - model_margin_mean)
            result["operating_margin"] = result["operating_margin"].astype(float) + margin_shift
            result["ebit_margin"] = result["operating_margin"]
        previous_revenue = float(financials.iloc[0]["revenue"] / (1.0 + financials.iloc[0]["revenue_growth"]))
        for idx in result.index:
            growth = float(result.at[idx, "revenue_growth"])
            revenue = previous_revenue * (1.0 + growth)
            result.at[idx, "revenue"] = revenue
            previous_revenue = revenue
        return _recompute_cash_flows(result)

    previous_revenue = float(result.iloc[0]["revenue"] / (1.0 + result.iloc[0]["revenue_growth"]))
    for position, idx in enumerate(result.index):
        row_weight = float(weights[position])
        model_growth = float(result.at[idx, "revenue_growth"])
        growth = (1.0 - row_weight) * model_growth + row_weight * target_quarterly_growth
        result.at[idx, "revenue_growth"] = growth
        revenue = previous_revenue * (1.0 + growth)
        result.at[idx, "revenue"] = revenue
        if target_margin is not None:
            model_margin = float(result.at[idx, "operating_margin"])
            margin = (1.0 - row_weight) * model_margin + row_weight * target_margin
            result.at[idx, "operating_margin"] = margin
            result.at[idx, "ebit_margin"] = margin
        previous_revenue = revenue
    return _recompute_cash_flows(result)


def _weight_schedule(weight: float | list[float], rows: int) -> list[float]:
    if isinstance(weight, list):
        values = [float(np.clip(value, 0.0, 1.0)) for value in weight]
        if len(values) >= rows:
            return values[:rows]
        return values + [values[-1] if values else 0.0] * (rows - len(values))
    return [float(np.clip(weight, 0.0, 1.0))] * rows


def _dynamic_consensus_weight_trace(
    ticker: str,
    financials: pd.DataFrame,
    history: pd.DataFrame,
    consensus: ForwardConsensus,
    settings: Settings,
    requested_weight: float,
    calibration_trace: dict[str, object],
) -> dict[str, object]:
    prior = prior_for(ticker, settings.company_type)
    raw_growth_var = _series_variance(financials.get("revenue_growth"))
    historical_growth_var = _series_variance(history.get("revenue_growth"))
    model_confidence = _model_confidence(raw_growth_var, historical_growth_var, calibration_trace)
    analyst_dispersion = _analyst_dispersion_proxy(consensus, history)
    maturity_factor = {
        "asset_light_quality": 0.95,
        "hyperscaler": 0.85,
        "platform": 0.80,
        "consumer_compounder": 0.90,
        "consumer_staple": 0.95,
        "capital_intensive_growth": 0.55,
        "marketplace_hybrid": 0.60,
    }.get(settings.company_type, 0.75)
    if prior.archetype in {"capital_intensive_growth", "amzn"}:
        maturity_factor = min(maturity_factor, 0.60)
    dispersion_factor = float(np.clip(1.0 - 0.25 * analyst_dispersion, 0.75, 1.0))
    confidence_factor = float(np.clip(1.0 - 0.25 * model_confidence, 0.75, 1.0))
    maturity_blend = 0.65 + 0.35 * maturity_factor
    floor = 0.25 if prior.archetype in {"capital_intensive_growth", "amzn"} else 0.45
    base_weight = float(np.clip(requested_weight * maturity_blend * dispersion_factor * confidence_factor, min(floor, requested_weight), requested_weight))
    quarterly_weights = []
    horizon = max(len(financials), 1)
    for quarter in range(1, horizon + 1):
        horizon_factor = float(np.clip(1.0 - 0.15 * ((quarter - 1) / max(horizon - 1, 1)), 0.85, 1.0))
        quarterly_weights.append({"quarter": float(quarter), "weight": float(base_weight * horizon_factor)})
    return {
        "method": "dynamic_consensus_weight_from_dispersion_confidence_maturity_horizon",
        "requested_weight": requested_weight,
        "average_weight": float(np.mean([row["weight"] for row in quarterly_weights])) if quarterly_weights else 0.0,
        "maturity_factor": maturity_factor,
        "maturity_blend": maturity_blend,
        "analyst_dispersion_proxy": analyst_dispersion,
        "dispersion_factor": dispersion_factor,
        "model_confidence": model_confidence,
        "model_confidence_factor": confidence_factor,
        "historical_accuracy_factor": calibration_trace.get("confidence_score"),
        "quarterly_weights": quarterly_weights,
    }


def _static_consensus_weight_trace(weight: float, rows: int, reason: str) -> dict[str, object]:
    quarterly_weights = [{"quarter": float(quarter), "weight": float(weight)} for quarter in range(1, rows + 1)]
    return {
        "method": "static_consensus_weight",
        "reason": reason,
        "requested_weight": float(weight),
        "average_weight": float(weight),
        "maturity_factor": None,
        "analyst_dispersion_proxy": None,
        "dispersion_factor": None,
        "model_confidence": None,
        "model_confidence_factor": None,
        "historical_accuracy_factor": None,
        "quarterly_weights": quarterly_weights,
    }


def _model_confidence(raw_growth_var: float | None, historical_growth_var: float | None, calibration_trace: dict[str, object]) -> float:
    variance_signal = 0.5
    if raw_growth_var is not None and historical_growth_var not in (None, 0.0):
        variance_signal = float(np.clip(raw_growth_var / historical_growth_var, 0.0, 1.0))
    accuracy_signal = calibration_trace.get("confidence_score")
    if not isinstance(accuracy_signal, (int, float)):
        accuracy_signal = 0.5
    return float(np.clip(0.55 * variance_signal + 0.45 * float(accuracy_signal), 0.0, 1.0))


def _analyst_dispersion_proxy(consensus: ForwardConsensus, history: pd.DataFrame) -> float:
    if consensus.fallback_used:
        return 0.35
    growth_vol = float(history["revenue_growth"].dropna().tail(12).std()) if "revenue_growth" in history else 0.05
    return float(np.clip(growth_vol * 4.0, 0.05, 0.45))


def _historical_forecast_calibration(history: pd.DataFrame) -> dict[str, object]:
    rows = []
    fields = ["revenue_growth", "operating_margin", "roic", "reinvestment_rate", "fcf"]
    for idx in range(4, len(history) - 1):
        previous = history.iloc[: idx + 1]
        actual = history.iloc[idx + 1]
        row: dict[str, object] = {"index": int(idx), "target_index": int(idx + 1)}
        for field in fields:
            if field not in history or not np.isfinite(float(actual.get(field, np.nan))):
                continue
            forecast = _one_step_forecast(previous, field)
            error = float(actual[field]) - forecast
            row[field] = {"forecast": forecast, "actual": float(actual[field]), "error": error}
        rows.append(row)
    summary = {}
    scores = []
    for field in fields:
        errors = [
            float(row[field]["error"])
            for row in rows
            if isinstance(row.get(field), dict) and np.isfinite(float(row[field]["error"]))
        ]
        actuals = [
            float(row[field]["actual"])
            for row in rows
            if isinstance(row.get(field), dict) and np.isfinite(float(row[field]["actual"]))
        ]
        if not errors:
            summary[field] = {"bias": None, "rmse": None, "directional_accuracy": None}
            continue
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        scale = float(np.nanstd(actuals)) if len(actuals) >= 2 else max(abs(float(np.nanmean(actuals))), 1.0)
        score = 1.0 / (1.0 + rmse / max(scale, 1e-9))
        scores.append(score)
        summary[field] = {
            "bias": float(np.mean(errors)),
            "rmse": rmse,
            "directional_accuracy": _directional_accuracy(rows, field),
            "confidence_score": float(np.clip(score, 0.0, 1.0)),
        }
    return {
        "method": "rolling_one_step_median_persistence_backtest",
        "observations": len(rows),
        "summary": summary,
        "confidence_score": float(np.mean(scores)) if scores else 0.5,
    }


def _one_step_forecast(history: pd.DataFrame, field: str) -> float:
    values = history[field].dropna().tail(4)
    if values.empty:
        return 0.0
    if len(values) == 1:
        return float(values.iloc[-1])
    return float(0.65 * values.iloc[-1] + 0.35 * values.median())


def _directional_accuracy(rows: list[dict[str, object]], field: str) -> float | None:
    hits = []
    previous_actual = None
    for row in rows:
        item = row.get(field)
        if not isinstance(item, dict):
            continue
        actual = item.get("actual")
        forecast = item.get("forecast")
        if not isinstance(actual, (int, float)) or not isinstance(forecast, (int, float)):
            continue
        if previous_actual is not None:
            hits.append(np.sign(forecast - previous_actual) == np.sign(actual - previous_actual))
        previous_actual = actual
    return float(np.mean(hits)) if hits else None


def _apply_economic_fade(
    ticker: str,
    financials: pd.DataFrame,
    history: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = financials.copy()
    prior = prior_for(ticker, settings.company_type)
    volatility = float(history["revenue_growth"].dropna().tail(12).std()) if "revenue_growth" in history else 0.05
    historical_growth = _historical_median(history, "revenue_growth", fallback=prior.terminal_growth_policy / 4.0)
    historical_margin = _historical_median(history, "operating_margin", fallback=(prior.margin_floor + prior.margin_cap) / 2.0)
    historical_roic = _historical_median(history, "roic", fallback=(prior.roic_floor + prior.roic_cap) / 2.0)
    historical_reinvestment = _historical_median(history, "reinvestment_rate", fallback=prior.reinvestment_burden)
    empirical_persistence = {
        field: _empirical_persistence(history, field, fallback=fallback)
        for field, fallback in {
            "revenue_growth": prior.growth_persistence,
            "operating_margin": prior.margin_persistence,
            "roic": prior.roic_persistence,
            "reinvestment_rate": prior.reinvestment_persistence,
        }.items()
    }
    max_fade_by_field = {
        field: float(np.clip((1.0 - persistence) * 0.75, 0.015, 0.12))
        for field, persistence in empirical_persistence.items()
    }
    steady = {
        "revenue_growth": float(np.clip(historical_growth, prior.revenue_growth_floor, prior.revenue_growth_cap)),
        "operating_margin": float(np.clip(0.70 * historical_margin + 0.30 * ((prior.margin_floor + prior.margin_cap) / 2.0), prior.margin_floor, prior.margin_cap)),
        "roic": float(np.clip(0.70 * historical_roic + 0.30 * ((prior.roic_floor + prior.roic_cap) / 2.0), prior.roic_floor, prior.roic_cap)),
        "reinvestment_rate": float(np.clip(0.70 * historical_reinvestment + 0.30 * prior.reinvestment_burden, 0.0, 0.6)),
    }
    speed = float(np.clip(0.04 + volatility * 0.60, 0.03, 0.12))
    trace_rows = []
    horizon = max(len(result), 1)
    for position, idx in enumerate(result.index, start=1):
        if position <= horizon * 0.40:
            fade_weight = 0.0
        else:
            fade_weight = float(np.clip(((position - horizon * 0.40) / (horizon * 0.60)) * speed, 0.0, 0.12))
        before = {field: float(result.at[idx, field]) for field in steady if field in result}
        applied_fade_weights = {}
        for field, target in steady.items():
            if field in result:
                field_fade_weight = min(fade_weight, max_fade_by_field.get(field, fade_weight))
                applied_fade_weights[field] = field_fade_weight
                result.at[idx, field] = (1.0 - field_fade_weight) * float(result.at[idx, field]) + field_fade_weight * target
        trace_rows.append(
            {
                "quarter": float(position),
                "fade_weight": fade_weight,
                "field_fade_weights": applied_fade_weights,
                "before": before,
                "after": {field: float(result.at[idx, field]) for field in before},
            }
        )
    return _recompute_cash_flows(result), {
        "method": "classification_volatility_scaled_economic_fade",
        "speed": speed,
        "empirical_persistence": empirical_persistence,
        "max_fade_by_field": max_fade_by_field,
        "steady_state_trace": steady,
        "quarters": trace_rows,
    }


def _no_fade_trace(ticker: str, financials: pd.DataFrame, settings: Settings, reason: str) -> dict[str, object]:
    prior = prior_for(ticker, settings.company_type)
    steady = {
        "revenue_growth": float(np.clip(prior.terminal_growth_policy / 4.0, -0.01, prior.revenue_growth_cap)),
        "operating_margin": float((prior.margin_floor + prior.margin_cap) / 2.0),
        "roic": float((prior.roic_floor + prior.roic_cap) / 2.0),
        "reinvestment_rate": float(np.clip(prior.reinvestment_burden, 0.0, 0.6)),
    }
    return {
        "method": "no_economic_fade",
        "reason": reason,
        "speed": 0.0,
        "steady_state_trace": steady,
        "quarters": [{"quarter": float(idx + 1), "fade_weight": 0.0} for idx in range(len(financials))],
    }


def _historical_median(history: pd.DataFrame, field: str, fallback: float) -> float:
    if field not in history:
        return float(fallback)
    values = history[field].dropna().tail(12)
    return float(values.median()) if not values.empty else float(fallback)


def _empirical_persistence(history: pd.DataFrame, field: str, fallback: float) -> float:
    if field not in history:
        return float(np.clip(fallback, 0.0, 0.995))
    values = history[field].dropna().astype(float).tail(20)
    if len(values) < 6:
        return float(np.clip(fallback, 0.0, 0.995))
    previous = values.iloc[:-1].to_numpy(dtype=float)
    current = values.iloc[1:].to_numpy(dtype=float)
    if float(np.nanstd(previous)) < 1e-9 or float(np.nanstd(current)) < 1e-9:
        return float(np.clip(fallback, 0.0, 0.995))
    corr = float(np.corrcoef(previous, current)[0, 1])
    return float(np.clip(corr, 0.0, 0.995)) if np.isfinite(corr) else float(np.clip(fallback, 0.0, 0.995))


def _series_variance(series) -> float | None:
    if series is None:
        return None
    values = pd.Series(series).dropna().astype(float)
    if len(values) < 2:
        return None
    return float(values.var(ddof=0))


def _apply_rd_capitalization(ticker: str, financials: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    result = financials.copy()
    if not settings.capitalize_rd:
        result["r_and_d_capitalized"] = 0.0
        return result
    prior = prior_for(ticker, settings.company_type)
    rd_ratio = prior.default_rd_ratio
    result["r_and_d"] = result.get("r_and_d", result["revenue"] * rd_ratio)
    result["r_and_d"] = result["r_and_d"].where(result["r_and_d"].notna(), result["revenue"] * rd_ratio)
    result["r_and_d_capitalized"] = result["r_and_d"] * settings.rd_capitalization_factor
    return _recompute_cash_flows(result)


def _apply_share_forecast(financials: pd.DataFrame, history: pd.DataFrame, buyback_rate: float) -> pd.DataFrame:
    result = financials.copy()
    latest_shares = _latest_shares(history)
    if latest_shares is None:
        result["shares"] = np.nan
        return result
    shares = latest_shares
    values = []
    quarterly_rate = max(buyback_rate, -0.05) / 4.0
    for _ in result.index:
        shares *= 1.0 - quarterly_rate
        values.append(shares)
    result["shares"] = values
    return result


def _estimate_buyback_rate(ticker: str, history: pd.DataFrame, settings: Settings) -> float:
    if not settings.buyback_model_enabled or "shares" not in history:
        return 0.0
    shares = history["shares"].dropna().tail(20)
    if len(shares) < 8 or shares.iloc[0] <= 0 or shares.iloc[-1] <= 0:
        return prior_for(ticker, settings.company_type).default_buyback_rate
    years = max((len(shares) - 1) / 4.0, 0.25)
    shrink = 1.0 - float((shares.iloc[-1] / shares.iloc[0]) ** (1.0 / years))
    prior = prior_for(ticker, settings.company_type).default_buyback_rate
    blended = 0.70 * shrink + 0.30 * prior
    return float(np.clip(blended, -0.02, settings.max_buyback_rate))


def _recompute_cash_flows(financials: pd.DataFrame) -> pd.DataFrame:
    result = financials.copy()
    tax_rate = result["tax_rate"].astype(float)
    for column in ["da", "capex", "delta_nwc", "maintenance_capex", "sbc_adjustment"]:
        ratio_col = f"_{column}_ratio"
        if ratio_col in result:
            result[column] = result["revenue"] * result[ratio_col].fillna(0.0)
    result["ebit"] = result["revenue"] * result["operating_margin"]
    result["nopat"] = result["ebit"] * (1.0 - tax_rate)
    rd_cap = result.get("r_and_d_capitalized", pd.Series(0.0, index=result.index)).fillna(0.0)
    result["nopat_adjusted"] = result["nopat"] + rd_cap * (1.0 - tax_rate)
    result["reinvestment"] = result["nopat"] * result["reinvestment_rate"]
    result["economic_fcf"] = result["nopat_adjusted"] * (1.0 - result["reinvestment_rate"]) - rd_cap
    result["accounting_fcf"] = result["nopat"] + result["da"] - result["capex"] - result["delta_nwc"]
    result["owner_earnings"] = (
        result["nopat_adjusted"]
        + result["da"]
        - result["maintenance_capex"]
        - result["delta_nwc"]
        - result["sbc_adjustment"]
        - rd_cap
    )
    result["fcf"] = result["economic_fcf"]
    return result


def _latest_shares(history: pd.DataFrame) -> float | None:
    if "shares" not in history:
        return None
    values = history["shares"].dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _segment_assumptions_trace(settings: Settings) -> dict[str, object] | None:
    if settings.segment_mode != "manual" or not settings.segment_assumptions_json:
        return None
    try:
        parsed = json.loads(settings.segment_assumptions_json)
    except json.JSONDecodeError:
        return {"error": "invalid_segment_assumptions_json"}
    return parsed if isinstance(parsed, dict) else {"error": "segment_assumptions_must_be_object"}


def _segment_growth(segment_trace: dict[str, object] | None) -> float | None:
    if not segment_trace or "segments" not in segment_trace:
        return None
    total = 0.0
    weight_sum = 0.0
    for segment in segment_trace.get("segments", []):
        if not isinstance(segment, dict):
            continue
        weight = float(segment.get("weight", 0.0))
        growth = segment.get("revenue_growth_annual")
        if growth is None:
            continue
        total += weight * ((1.0 + float(growth)) ** 0.25 - 1.0)
        weight_sum += weight
    return total / weight_sum if weight_sum > 0.0 else None


def _segment_margin(segment_trace: dict[str, object] | None) -> float | None:
    if not segment_trace or "segments" not in segment_trace:
        return None
    total = 0.0
    weight_sum = 0.0
    for segment in segment_trace.get("segments", []):
        if not isinstance(segment, dict):
            continue
        weight = float(segment.get("weight", 0.0))
        margin = segment.get("operating_margin")
        if margin is None:
            continue
        total += weight * float(margin)
        weight_sum += weight
    return total / weight_sum if weight_sum > 0.0 else None


def _forecast_trace(financials: pd.DataFrame) -> list[dict[str, float | None]]:
    columns = [
        "quarter",
        "revenue",
        "revenue_growth",
        "operating_margin",
        "reinvestment_rate",
        "roic",
        "fcf",
        "shares",
        "raw_model_revenue_growth",
        "raw_model_operating_margin",
        "raw_model_reinvestment_rate",
        "raw_model_roic",
        "revenue_growth_clamped",
        "operating_margin_clamped",
        "reinvestment_rate_clamped",
        "roic_clamped",
    ]
    rows = []
    available = [column for column in columns if column in financials]
    for record in financials[available].head(20).to_dict(orient="records"):
        row = {}
        for key, value in record.items():
            row[key] = float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None
        rows.append(row)
    return rows


def _field_source_map(
    raw: pd.DataFrame,
    blended: pd.DataFrame,
    overlay: pd.DataFrame,
    final: pd.DataFrame,
    consensus_weight: float,
    buyback_rate: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in final.index[:20]:
        row: dict[str, object] = {"quarter": float(final.at[idx, "quarter"]) if "quarter" in final else float(idx + 1)}
        for field in FORECAST_TRACE_FIELDS:
            raw_value = _field_value(raw, idx, field)
            blended_value = _field_value(blended, idx, field)
            overlay_value = _field_value(overlay, idx, field)
            final_value = _field_value(final, idx, field)
            row[field] = {
                "raw_value": raw_value,
                "blended_value": blended_value,
                "capped_value": overlay_value,
                "final_value": final_value,
                "source": _field_source(field, raw_value, blended_value, overlay_value, final_value, consensus_weight, buyback_rate),
                "clamp_triggered": _field_clamped(final, idx, field),
                "fallback_reason": _field_fallback_reason(final, idx, field),
            }
        rows.append(row)
    return rows


def _clamp_triggered_map(final: pd.DataFrame) -> dict[str, bool]:
    return {
        field: bool(final.get(f"{field}_clamped", pd.Series(False, index=final.index)).astype(bool).any())
        for field in ("revenue_growth", "operating_margin", "reinvestment_rate", "roic")
    }


def _field_value(frame: pd.DataFrame, idx: int, field: str) -> float | None:
    if field not in frame:
        return None
    value = frame.at[idx, field]
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _field_source(
    field: str,
    raw_value: float | None,
    blended_value: float | None,
    overlay_value: float | None,
    final_value: float | None,
    consensus_weight: float,
    buyback_rate: float,
) -> str:
    if field in {"revenue_growth", "operating_margin"} and consensus_weight > 0.0 and _changed(raw_value, blended_value):
        return "consensus"
    if field == "shares" and buyback_rate != 0.0:
        return "overlay"
    if _changed(blended_value, overlay_value):
        return "overlay"
    if _changed(overlay_value, final_value):
        return "overlay"
    return "model"


def _field_clamped(frame: pd.DataFrame, idx: int, field: str) -> bool:
    column = f"{field}_clamped"
    if column not in frame:
        return False
    return bool(frame.at[idx, column])


def _field_fallback_reason(frame: pd.DataFrame, idx: int, field: str) -> str | None:
    if _field_clamped(frame, idx, field):
        return "policy_cap"
    if field == "shares" and "shares" in frame and not np.isfinite(float(frame.at[idx, "shares"])):
        return "missing_share_history"
    return None


def _changed(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left != right
    return not np.isclose(left, right, rtol=1e-9, atol=1e-12)


def _consensus_trace(
    financials: pd.DataFrame,
    target_growth: float,
    target_margin: float | None,
) -> list[dict[str, float]]:
    base_revenue = float(financials.iloc[0]["revenue"] / (1.0 + financials.iloc[0]["revenue_growth"]))
    margin = target_margin if target_margin is not None else float(financials["operating_margin"].median())
    rows = []
    revenue = base_revenue
    for quarter in range(1, len(financials.head(20)) + 1):
        revenue *= 1.0 + target_growth
        rows.append(
            {
                "quarter": float(quarter),
                "revenue": float(revenue),
                "revenue_growth": float(target_growth),
                "operating_margin": float(margin),
            }
        )
    return rows
