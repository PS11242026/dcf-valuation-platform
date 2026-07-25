"""Blast-radius re-validation: factor uniqueness under the split-corrected discount.

Re-runs factor_uniqueness's horse-race (univariate Newey-West IC) and bidirectional spanning (does
the discount span the rival value/quality factors, with the full factor block as controls) on the
ORIGINAL vs the SPLIT-CORRECTED discount. Pure analytic re-run -- patches the shared obs loader; no
MCMC; frozen artifacts untouched.

Usage: PYTHONPATH=src python scripts/utilities/research_v2/blast_radius_factor_uniqueness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scripts.validation_common as C  # noqa: E402
import scripts.utilities.factor_uniqueness as FU  # noqa: E402

FU.REPORT_DIR = ROOT / "reports"   # module computes ROOT=parents[1]; point panel reads at repo/reports
C.REPORT_DIR = ROOT / "reports"

ORIG = json.loads((ROOT / "reports" / "final_universe_backtest.json").read_text())["observations"]
CORR = json.loads((ROOT / "reports" / "research_v2" / "final_universe_backtest_split_corrected.json"
                   ).read_text())["observations"]


def _run(obs):
    C.load_obs = lambda path=None: obs          # patch the shared loader factor_uniqueness uses
    joined = FU._load_joined()
    hr = FU.horse_race(joined)
    sp3 = FU.spanning_matrix(joined, 3, FU.FACTOR_BLOCK)
    sp5 = FU.spanning_matrix(joined, 5, FU.FACTOR_BLOCK)
    return hr, sp3, sp5


def _discount_row(hr):
    r = hr[hr["signal"] == FU.PRETTY[FU.FOCAL]].iloc[0]
    return {f"{y}y": (r[f"ic_{y}y"], r[f"t_{y}y"]) for y in FU.HORIZONS}


def main() -> None:
    print("Running factor-uniqueness on ORIGINAL discount ...", flush=True)
    hr0, s30, s50 = _run(ORIG)
    print("Running factor-uniqueness on SPLIT-CORRECTED discount ...", flush=True)
    hr1, s31, s51 = _run(CORR)

    print("\n=== DCF discount univariate IC (Newey-West t): original -> corrected ===")
    d0, d1 = _discount_row(hr0), _discount_row(hr1)
    for y in FU.HORIZONS:
        print(f"  {y}y: IC {d0[f'{y}y'][0]} (t {d0[f'{y}y'][1]})  ->  {d1[f'{y}y'][0]} (t {d1[f'{y}y'][1]})")

    print("\n=== Full horse-race (corrected) -- IC by signal ===")
    print(hr1.to_string(index=False))

    print("\n=== 3y spanning vs rivals WITH factor block -- discount coef/t (orig -> corr), rival coef/t ===")
    print("  rival                         disc_t_orig  disc_t_corr   rival_t_orig  rival_t_corr")
    for _, r0 in s30.iterrows():
        r1 = s31[s31["rival"] == r0["rival"]].iloc[0]
        print(f"  {r0['rival']:28} {str(r0['discount_t']):>11}  {str(r1['discount_t']):>11}   "
              f"{str(r0['rival_t']):>11}  {str(r1['rival_t']):>11}")

    out = {
        "discount_ic": {"original": d0, "corrected": d1},
        "horse_race_corrected": hr1.to_dict(orient="records"),
        "spanning_3y": {"original": s30.to_dict(orient="records"), "corrected": s31.to_dict(orient="records")},
        "spanning_5y": {"original": s50.to_dict(orient="records"), "corrected": s51.to_dict(orient="records")},
    }
    p = ROOT / "reports" / "research_v2" / "blast_radius_factor_uniqueness.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
