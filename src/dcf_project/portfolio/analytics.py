"""Expected-performance profile for a constructed book -- sourced from the validated backtest.

Reads the frozen research outputs (reports/portfolio/metrics.json, capacity_costs.json,
final_universe_backtest.json) so an IC proposal cites validated evidence rather than re-deriving it.
"""
from __future__ import annotations

import json
from pathlib import Path

from dcf_project.paths import REPORTS


def _load(p: Path):
    try:
        return json.load(open(p)) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _dig(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def expected_profile() -> dict:
    m = _load(REPORTS / "portfolio" / "metrics.json") or {}
    bt = _load(REPORTS / "final_universe_backtest.json") or {}
    cap = _load(REPORTS / "portfolio" / "capacity_costs.json") or {}

    prof: dict = {}
    prof["expected_return_cagr"] = m.get("cagr_net")
    prof["expected_sharpe"] = m.get("sharpe_net")
    prof["expected_max_drawdown"] = m.get("max_drawdown_net")
    prof["expected_turnover_annual"] = m.get("annual_turnover_approx")
    cu = m.get("cagr_universe_ew")
    vb = m.get("vs_benchmark") or {}
    # alpha vs SPY if present, else net CAGR minus equal-weight universe CAGR
    alpha = _dig(vb, "SPY", "alpha") or _dig(vb, "spy", "alpha") or _dig(vb, "SPY", "alpha_annualized")
    if alpha is None and prof["expected_return_cagr"] is not None and cu is not None:
        alpha = prof["expected_return_cagr"] - cu
    prof["expected_alpha"] = alpha
    if prof["expected_return_cagr"] and prof["expected_sharpe"]:
        prof["expected_volatility"] = prof["expected_return_cagr"] / prof["expected_sharpe"]
    om3 = _dig(bt, "overall_metrics", "3y") or {}
    prof["ic_3y"] = om3.get("information_coefficient")
    prof["long_short_3y"] = om3.get("long_short_q5_minus_q1")
    prof["capacity_ceilings"] = cap.get("ceilings")
    prof["config"] = m.get("config")
    return prof
