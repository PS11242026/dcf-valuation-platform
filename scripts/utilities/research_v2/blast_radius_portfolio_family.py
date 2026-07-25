"""Blast-radius re-runs of the portfolio-family studies on the split-corrected discount.

Re-runs allocation_study.run() and rebalance_horizon.run() (and reports the capacity tie-in they
carry) on the ORIGINAL vs SPLIT-CORRECTED signal panel by patching the shared load_signal_panel.
Calls run() (not main()) so the original reports/portfolio/* artifacts are NOT overwritten.

Caveat: discount_x_conf and sector-reliability weights still load the UNCORRECTED
final_institutional_validation.json; recomputing those is a rebuild-phase step.

Usage: PYTHONPATH=src python scripts/utilities/research_v2/blast_radius_portfolio_family.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scripts.portfolio.portfolio_sim as PS  # noqa: E402
import scripts.portfolio.allocation_study as AL  # noqa: E402
import scripts.portfolio.rebalance_horizon as RB  # noqa: E402

CLIP = PS.DISCOUNT_CLIP
ORIG = json.loads((ROOT / "reports" / "final_universe_backtest.json").read_text())["observations"]
CORR = json.loads((ROOT / "reports" / "research_v2" / "final_universe_backtest_split_corrected.json"
                   ).read_text())["observations"]
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _panel(obs):
    panel: dict[str, dict[str, dict]] = {}
    for o in obs:
        d = o.get("discount_to_intrinsic")
        if not isinstance(d, (int, float)) or abs(d) > CLIP:
            continue
        panel.setdefault(o["date"], {})[o["ticker"]] = {"discount": d, "sector": o["sector"]}
    return sorted(panel), panel


def _patch(obs):
    loader = lambda: _panel(obs)  # noqa: E731
    PS.load_signal_panel = loader
    AL.load_signal_panel = loader
    RB.load_signal_panel = loader


def main() -> None:
    out = {}
    for label, obs in (("ORIGINAL", ORIG), ("CORRECTED", CORR)):
        _patch(obs)
        print(f"\n################  {label}  ################")
        try:
            al_df, _meta = AL.run()
            print("\n--- allocation_study.run() (schemes) ---")
            print(al_df.to_string(index=False))
            out[f"allocation:{label}"] = al_df.to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            print(f"allocation run failed: {type(exc).__name__}: {exc}")
        try:
            rb = RB.run()
            freq = rb[0]
            print("\n--- rebalance_horizon.run() (by frequency) ---")
            print(freq.to_string(index=False))
            out[f"rebalance_freq:{label}"] = freq.to_dict(orient="records")
            if len(rb) > 1 and hasattr(rb[1], "to_string"):
                print("\n--- holding horizon ---")
                print(rb[1].to_string(index=False))
                out[f"rebalance_horizon:{label}"] = rb[1].to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            print(f"rebalance run failed: {type(exc).__name__}: {exc}")

    p = ROOT / "reports" / "research_v2" / "blast_radius_portfolio_family.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
