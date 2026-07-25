"""Build the split-corrected observations panel -- the shared input for the Blast Radius Review.

Produces reports/research_v2/final_universe_backtest_split_corrected.json with the SAME schema as
the frozen reports/final_universe_backtest.json, but with `discount_to_intrinsic` replaced by the
split-corrected value `(1+discount)/cum_split - 1` (the original kept as `discount_original`). Uses
the whitelisted yfinance split cache already built by split_corrected_universe_revalidation.py.

Every downstream evaluation study (factor_uniqueness, factor_exposure_pass, portfolio_sim, allocation,
survivorship) reads its discount signal from final_universe_backtest.json, so pointing them at this
corrected copy re-validates them analytically -- no MCMC, frozen artifact untouched.

Usage: PYTHONPATH=src python scripts/utilities/research_v2/build_corrected_obs_panel.py
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "reports" / "final_universe_backtest.json"
SPLIT_CACHE = ROOT / "reports" / "research_v2" / "_cache" / "universe_splits.json"
OUT = ROOT / "reports" / "research_v2" / "final_universe_backtest_split_corrected.json"
LAG_DAYS = 75


def _cum_after(events: list, cutoff: date) -> float:
    ref = cutoff - timedelta(days=LAG_DAYS)
    f = 1.0
    for ds, fac in events or []:
        if date.fromisoformat(ds) > ref:
            f *= float(fac)
    return f


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    splits = json.loads(SPLIT_CACHE.read_text(encoding="utf-8"))
    obs = payload["observations"]
    n_corr = 0
    for o in obs:
        cum = _cum_after(splits.get(o["ticker"], []), date.fromisoformat(o["date"]))
        d = o.get("discount_to_intrinsic")
        o["discount_original"] = d
        o["cum_split"] = cum
        if isinstance(d, (int, float)) and cum != 1.0:
            o["discount_to_intrinsic"] = (1.0 + d) / cum - 1.0
            n_corr += 1
    payload["split_corrected"] = True
    payload["n_obs_split_corrected"] = n_corr
    payload["note"] = ("discount_to_intrinsic is SPLIT-CORRECTED (whitelisted yfinance splits, "
                       "cum after cutoff-75d); discount_original retains the frozen-harness value.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(obs)} obs, {n_corr} split-corrected)")


if __name__ == "__main__":
    main()
