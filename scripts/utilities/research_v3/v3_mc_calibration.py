"""G6: Monte Carlo calibration of the engine_v3 valuation bands (evaluation only).

Reads a paired v3 panel (v3_universe_backtest_*.json) and asks three pre-registered
questions about the driver-MC uncertainty layer:

(i)  COVERAGE -- if price converges toward value over 3y, the implied-return band
     [r(p10), r(p90)] (r(q) = (iv_q/price)^(1/3) - 1) should bracket realized 3y
     returns ~80% of the time (nominal). Wide tolerance: realized returns carry
     market noise the iv band does not model.
(ii) "KNOWS WHEN IT DOESN'T KNOW" -- rank correlation between band width and the
     realized absolute error |fwd_3y - r(p50)|. THE KILL CRITERION: corr <= 0 kills
     the MC layer (keep the deterministic skeleton).
(iii) SHARPNESS -- band-width distribution overall and by family (a band can only be
     informative if it is also not vacuously wide).

Usage: PYTHONPATH=src python scripts/utilities/research_v3/v3_mc_calibration.py \
           [--panel reports/research_v3/v3_universe_backtest_n50.json] [--horizon 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import validation_common as vc  # noqa: E402


def implied_return(iv: float, price: float, years: int) -> float:
    return (iv / price) ** (1.0 / years) - 1.0


def usable_rows(obs: list[dict], years: int) -> list[dict]:
    rk = f"fwd_{years}y"
    out = []
    for o in obs:
        if not isinstance(o.get(rk), (int, float)):
            continue
        p10, p50, p90, px = o.get("v3_p10"), o.get("v3_p50"), o.get("v3_p90"), o.get("price")
        if not all(isinstance(x, (int, float)) and x > 0 for x in (p10, p50, p90, px)):
            continue
        if p90 <= p10:  # degenerate band (mc off / collapsed)
            continue
        out.append(o)
    return out


def evaluate(obs: list[dict], years: int) -> dict:
    rk = f"fwd_{years}y"
    rows = usable_rows(obs, years)
    if len(rows) < 30:
        return {"n": len(rows), "note": "insufficient rows with bands + forward returns"}

    fwd = np.array([o[rk] for o in rows])
    r10 = np.array([implied_return(o["v3_p10"], o["price"], years) for o in rows])
    r50 = np.array([implied_return(o["v3_p50"], o["price"], years) for o in rows])
    r90 = np.array([implied_return(o["v3_p90"], o["price"], years) for o in rows])
    width = np.array([
        o["v3_band_width"] if isinstance(o.get("v3_band_width"), (int, float))
        else (o["v3_p90"] - o["v3_p10"]) / o["v3_p50"]
        for o in rows
    ])
    abs_err = np.abs(fwd - r50)

    inside = (fwd >= r10) & (fwd <= r90)
    coverage = float(np.mean(inside))

    # (ii) overall rank correlation + per-date series with NW t (overlap-aware).
    sp = stats.spearmanr(width, abs_err)
    byd: dict[str, list[tuple[float, float]]] = {}
    for o, w, e in zip(rows, width, abs_err):
        byd.setdefault(o["date"], []).append((float(w), float(e)))
    per_date = []
    for dt in sorted(byd):
        pairs = byd[dt]
        if len(pairs) >= 4:
            ic = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
            if np.isfinite(ic):
                per_date.append(float(ic))
    nw = vc.newey_west_tstat(per_date, vc.HORIZON_QUARTERS[years])

    # Coverage stability across halves of the sample period.
    dates = sorted({o["date"] for o in rows})
    mid = dates[len(dates) // 2]
    early = inside[[o["date"] < mid for o in rows]]
    late = inside[[o["date"] >= mid for o in rows]]

    by_family: dict[str, dict] = {}
    fams: dict[str, list[int]] = {}
    for i, o in enumerate(rows):
        fams.setdefault(o.get("sector") or "unknown", []).append(i)
    for fam, idx in sorted(fams.items()):
        if len(idx) >= 20:
            by_family[fam] = {
                "n": len(idx),
                "coverage": float(np.mean(inside[idx])),
                "median_width": float(np.median(width[idx])),
                "width_abs_err_spearman": float(stats.spearmanr(width[idx], abs_err[idx]).statistic),
            }

    width_err_corr = float(sp.statistic)
    return {
        "n": len(rows),
        "horizon_years": years,
        "coverage_p10_p90": coverage,
        "coverage_nominal": 0.80,
        "coverage_tolerance": [0.55, 0.95],
        "coverage_within_tolerance": bool(0.55 <= coverage <= 0.95),
        "coverage_early_half": float(np.mean(early)) if len(early) else None,
        "coverage_late_half": float(np.mean(late)) if len(late) else None,
        "width_abs_err_spearman_pooled": width_err_corr,
        "width_abs_err_spearman_p_naive": float(sp.pvalue),
        "width_abs_err_per_date": {"mean": nw.get("mean"), "nw_tstat": nw.get("t_stat"),
                                   "n_dates": nw.get("n")},
        "kill_criterion_width_err_corr_le_0": bool(width_err_corr <= 0),
        "sharpness": {
            "median_width": float(np.median(width)),
            "p25_width": float(np.percentile(width, 25)),
            "p75_width": float(np.percentile(width, 75)),
        },
        "by_family": by_family,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=str(ROOT / "reports" / "research_v3" / "v3_universe_backtest_n50.json"))
    ap.add_argument("--horizon", type=int, default=3, choices=(1, 3, 5))
    args = ap.parse_args()

    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    obs = panel["observations"]
    result = evaluate(obs, args.horizon)
    out_path = Path(args.panel).parent / f"v3_mc_calibration_{Path(args.panel).stem.split('_')[-1]}_{args.horizon}y.json"
    out_path.write_text(json.dumps({"panel": str(args.panel), **result}, indent=2), encoding="utf-8")

    print(f"G6 MC calibration ({args.horizon}y, n={result.get('n')}):")
    if "note" in result:
        print(f"  {result['note']}")
        return 1
    print(f"  coverage p10-p90: {result['coverage_p10_p90']:.3f} (nominal 0.80, tol {result['coverage_tolerance']}) "
          f"-> {'OK' if result['coverage_within_tolerance'] else 'OUT OF TOLERANCE'}")
    print(f"  early/late halves: {result['coverage_early_half']:.3f} / {result['coverage_late_half']:.3f}")
    print(f"  width<->|err| spearman: pooled {result['width_abs_err_spearman_pooled']:.3f}; "
          f"per-date mean {result['width_abs_err_per_date']['mean']} "
          f"(NW t {result['width_abs_err_per_date']['nw_tstat']})")
    print(f"  KILL (corr<=0): {result['kill_criterion_width_err_corr_le_0']}")
    print(f"  sharpness: median width {result['sharpness']['median_width']:.3f}")
    for fam, b in result["by_family"].items():
        print(f"    {fam:36} n={b['n']:4} cov={b['coverage']:.2f} width={b['median_width']:.2f} "
              f"corr={b['width_abs_err_spearman']:.2f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
