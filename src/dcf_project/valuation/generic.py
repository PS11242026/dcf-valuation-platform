"""Generic DCF -- textbook single-stage normalized-margin baseline (deterministic_baseline)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from dcf_project.core import deterministic_baseline, get_settings
from dcf_project.data_sources.fundamentals import load_fundamentals
from dcf_project.paths import VALUATION
from dcf_project.preprocessing.features import build_features
from dcf_project.reporting_kit import save_model, xlsx
from dcf_project.valuation.report import build_generic_model


def generic_valuation(ticker: str, settings=None) -> dict:
    settings = settings or get_settings()
    bundle = load_fundamentals(ticker, settings)
    features, _ = build_features(bundle.periods, settings)
    latest = features.iloc[-1]

    def _lv(col):
        try:
            v = float(latest.get(col))
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    baseline = deterministic_baseline(features, bundle.wacc, bundle.market, _lv("cash"), _lv("debt"),
                                      _lv("shares_outstanding"), settings, 20)
    price = float(bundle.market.price) if bundle.market.price else None
    return {"ticker": ticker.upper(), "baseline": baseline, "price": price}


def run_generic(ticker: str) -> Path:
    ticker = ticker.upper()
    folder = VALUATION / ticker
    folder.mkdir(parents=True, exist_ok=True)
    data = generic_valuation(ticker)
    model = build_generic_model(ticker, data["baseline"], data["price"], folder)
    out = xlsx.render(model, folder / "generic_dcf.xlsx")
    save_model(model, folder)
    (folder / "assumptions.json").write_text(
        json.dumps({"mode": "generic", **data["baseline"]}, indent=2, default=str), encoding="utf-8")
    iv = data["baseline"].get("intrinsic_value_per_share")
    print(f"wrote {out}  (intrinsic ${iv:,.2f}, price ${data['price']})" if iv else f"wrote {out}")
    return out
