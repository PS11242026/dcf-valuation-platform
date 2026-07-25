"""Paired significance of the v3-v1 IC difference on the corrected tri-engine panel.

Tests the PAIRED per-date IC difference (v3 minus v1) directly -- not the raw ICs --
with two overlap-aware estimators so the 3y/5y windows are not overstated:

  * Newey-West HAC t-stat on the per-date IC-difference series (lag = horizon quarters,
    i.e. 4/12/20 for 1/3/5y) -- ``validation_common.paired_ic_diff`` verbatim.
  * Contiguous-date BLOCK BOOTSTRAP of the obs-weighted IC difference (block length =
    horizon quarters), which preserves overlapping-window autocorrelation without relying
    on the NW lag being small relative to the date count. This is the trustworthy check at
    5y, where only ~24 dates exist against a lag of 20.

Run overall and per-family, with staples called out (the one family where v1 still led v3
after the margin-seasonality fix). Read-only; writes a JSON ledger, changes no engine math.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for c in (ROOT / "scripts", ROOT / "scripts" / "utilities"):
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))


import validation_common as vc  # noqa: E402

PAYLOAD = ROOT / "reports" / "research_v3" / "tri_engine_backtest_broad.json"
OUT = ROOT / "reports" / "research_v3" / "tri_paired_significance.json"
HORIZONS = (1, 3, 5)
V3, V1 = "discount_to_intrinsic", "v1_discount"


def _obs_ic_diff(rows: list[dict], y: int) -> float | None:
    """Obs-weighted IC(v3) - IC(v1) on a row set (the block-bootstrap statistic)."""
    a = vc.obs_weighted_ic(vc.per_date_ic_series(rows, y, V3))
    b = vc.obs_weighted_ic(vc.per_date_ic_series(rows, y, V1))
    if a is None or b is None:
        return None
    return a - b


def _analyze(rows: list[dict], label: str) -> dict:
    block = {"label": label, "n_obs": len(rows), "horizons": {}}
    for y in HORIZONS:
        nw = vc.paired_ic_diff(rows, y, V3, V1)
        boot = vc.block_bootstrap_ci(rows, y, lambda r, _y=y: _obs_ic_diff(r, _y),
                                     block_q=vc.HORIZON_QUARTERS[y], n_boot=3000)
        block["horizons"][f"{y}y"] = {
            "n_dates": nw["n_dates"],
            "mean_ic_diff": nw["mean_ic_diff"],
            "nw_tstat": nw["nw_tstat"],
            "nw_lag_q": nw["nw_lag_quarters"],
            "boot_ci95": boot.get("ci95"),
            "boot_excludes_zero": boot.get("excludes_zero"),
            "boot_mean": boot.get("boot_mean"),
        }
    return block


def _fmt(blk: dict) -> str:
    rows = [f"  {blk['label']:34} (n_obs={blk['n_obs']})"]
    for y in HORIZONS:
        h = blk["horizons"][f"{y}y"]
        t = h["nw_tstat"]
        ci = h["boot_ci95"]
        ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "    n/a     "
        star = "*" if (t is not None and abs(t) >= 2.0) else " "
        bstar = "B" if h["boot_excludes_zero"] else " "
        md = h["mean_ic_diff"]
        rows.append(f"      {y}y  dIC={md:+.3f} (nd={h['n_dates']:2})  "
                    f"NW t={t:+.2f}{star} (lag {h['nw_lag_q']})  "
                    f"boot95 {ci_s}{bstar}")
    return "\n".join(rows)


def main() -> int:
    d = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    obs = d["observations"]
    fams: dict[str, list] = {}
    for o in obs:
        fams.setdefault(o.get("sector") or "unknown", []).append(o)

    results = {"overall": _analyze(obs, "OVERALL (v3 - v1)")}
    print("=== PAIRED v3 - v1 IC DIFFERENCE: overlap-aware inference ===")
    print("  (NW t on per-date IC diff; block-bootstrap 95% CI of obs-wtd IC diff,")
    print("   block = horizon quarters.  '*' = |NW t|>=2,  'B' = bootstrap CI excludes 0)\n")
    print(_fmt(results["overall"]))
    print("\n=== BY FAMILY ===")
    for fam in sorted(fams, key=lambda f: -len(fams[f])):
        blk = _analyze(fams[fam], fam)
        results[fam] = blk
        print(_fmt(blk))

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
