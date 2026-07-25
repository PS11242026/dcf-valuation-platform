"""G4: validation suites + Book/Market control over a paired v3 panel (evaluation only).

Re-runs the research_v2 validation-suite battery on the SPLIT-CONSISTENT v3 panel
(v3_universe_backtest_{n50,broad}.json) for BOTH engines (v3 signal and the paired
in-process v1 signal), then adds the pre-registered factor control:

  Suite 1  Family stratification (v3 vs paired v1, paired per-date IC diff w/ NW t)
  Suite 2  Regime stability (validation_common.REGIMES taxonomy)
  Suite 3  Decile portfolios (3y forward return by v3-discount decile)
  Suite 5  Freeze-and-test holdout (obs dated after 2021-06-30 only)
  B/M      Book/Market control from reports/_factor_panel.json:
           (a) per-date OLS fwd ~ rank_z(v3_discount) + rank_z(book_to_market),
               NW t on the discount-coefficient series (overlap-aware);
           (b) residual IC: per-date Spearman of (v3 discount rank residualized
               on B/M rank) vs forward return -- overall and per family.

G4 kill (plan): kill the BROAD claim if B/M subsumes v3 everywhere including the
reliable families; otherwise scope the claim honestly. This script reports numbers;
the decision is recorded in the research journal.

Usage: PYTHONPATH=src python scripts/utilities/research_v3/v3_validation_suites.py \
           [--panel reports/research_v3/v3_universe_backtest_broad.json]
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

V3_KEY = "discount_to_intrinsic"
V1_KEY = "v1_discount"
FREEZE_DATE = "2021-06-30"
HORIZONS = (1, 3, 5)
MIN_CROSS_SECTION = 4  # same per-date floor validation_common uses


def _rank_z(values: list[float]) -> np.ndarray:
    r = stats.rankdata(values)
    return (r - r.mean()) / (r.std() or 1.0)


def _engine_block(rows: list[dict]) -> dict:
    return {
        "v3": {f"{y}y": vc.core_metrics(rows, y, signal_key=V3_KEY) for y in HORIZONS},
        "v1_paired": {f"{y}y": vc.core_metrics(rows, y, signal_key=V1_KEY) for y in HORIZONS},
        "paired_ic_diff": {f"{y}y": vc.paired_ic_diff(rows, y, V3_KEY, V1_KEY) for y in (3, 5)},
    }


def suite1_families(obs: list[dict]) -> dict:
    fams: dict[str, list] = {}
    for o in obs:
        fams.setdefault(o.get("sector") or "unknown", []).append(o)
    return {fam: {"n_obs": len(sub), **_engine_block(sub)}
            for fam, sub in sorted(fams.items()) if len(sub) >= 30}


def suite2_regimes(obs: list[dict]) -> dict:
    blocks: dict[str, list] = {}
    for o in obs:
        reg = vc.regime_of(o["date"])
        if reg:
            blocks.setdefault(reg, []).append(o)
    return {reg: {"n_obs": len(sub),
                  "v3_3y": vc.core_metrics(sub, 3, signal_key=V3_KEY),
                  "v1_3y": vc.core_metrics(sub, 3, signal_key=V1_KEY)}
            for reg, sub in sorted(blocks.items())}


def suite3_deciles(obs: list[dict]) -> dict:
    return {f"{y}y": vc.per_date_buckets(obs, y, signal_key=V3_KEY) for y in (3, 5)}


def suite5_holdout(obs: list[dict]) -> dict:
    post = [o for o in obs if o["date"] > FREEZE_DATE]
    return {"freeze_date": FREEZE_DATE, "n_obs_post": len(post), **_engine_block(post)}


# ----------------------------- B/M control ----------------------------- #

def _factor_join(obs: list[dict]) -> tuple[list[dict], dict]:
    panel = json.loads((ROOT / "reports" / "_factor_panel.json").read_text(encoding="utf-8"))
    bm = {(r["ticker"], r["date"]): r.get("book_to_market") for r in panel}
    joined = []
    for o in obs:
        v = bm.get((o["ticker"], o["date"]))
        if isinstance(v, (int, float)) and np.isfinite(v):
            joined.append({**o, "book_to_market": float(v)})
    cov = {"n_obs": len(obs), "n_joined": len(joined),
           "join_rate": round(len(joined) / len(obs), 3) if obs else None,
           "n_tickers_joined": len({o["ticker"] for o in joined})}
    return joined, cov


def bm_control(obs: list[dict], y: int) -> dict:
    """Per-date OLS coefficients + B/M-residualized discount IC, NW/overlap-aware."""
    rk = f"fwd_{y}y"
    byd: dict[str, list[dict]] = {}
    for o in obs:
        if isinstance(o.get(rk), (int, float)) and isinstance(o.get(V3_KEY), (int, float)):
            byd.setdefault(o["date"], []).append(o)

    coef_disc, coef_bm, resid_ic, dates = [], [], [], []
    for dt in sorted(byd):
        sub = byd[dt]
        if len(sub) < MIN_CROSS_SECTION + 1:
            continue
        zd = _rank_z([o[V3_KEY] for o in sub])
        zb = _rank_z([o["book_to_market"] for o in sub])
        fwd = np.array([o[rk] for o in sub])
        X = np.column_stack([np.ones(len(sub)), zd, zb])
        try:
            beta, *_ = np.linalg.lstsq(X, fwd, rcond=None)
        except np.linalg.LinAlgError:
            continue
        coef_disc.append(float(beta[1]))
        coef_bm.append(float(beta[2]))
        # residualize discount rank on B/M rank, then Spearman vs fwd
        slope = float(np.dot(zd, zb) / (np.dot(zb, zb) or 1.0))
        resid = zd - slope * zb
        ic = stats.spearmanr(resid, fwd).statistic
        if np.isfinite(ic):
            resid_ic.append(float(ic))
        dates.append(dt)

    lag = vc.HORIZON_QUARTERS[y]
    return {
        "n_dates": len(dates),
        "discount_coef": vc.newey_west_tstat(coef_disc, lag),
        "bm_coef": vc.newey_west_tstat(coef_bm, lag),
        "residual_ic": vc.newey_west_tstat(resid_ic, lag),
    }


def bm_by_family(obs: list[dict], y: int) -> dict:
    fams: dict[str, list] = {}
    for o in obs:
        fams.setdefault(o.get("sector") or "unknown", []).append(o)
    return {fam: bm_control(sub, y) for fam, sub in sorted(fams.items()) if len(sub) >= 100}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel",
                    default=str(ROOT / "reports" / "research_v3" / "v3_universe_backtest_broad.json"))
    args = ap.parse_args()

    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    obs = panel["observations"]
    joined, coverage = _factor_join(obs)

    result = {
        "panel": str(args.panel),
        "n_observations": len(obs),
        "suite1_families": suite1_families(obs),
        "suite2_regimes": suite2_regimes(obs),
        "suite3_deciles": suite3_deciles(obs),
        "suite5_holdout": suite5_holdout(obs),
        "bm_control": {
            "join_coverage": coverage,
            "overall": {f"{y}y": bm_control(joined, y) for y in (3, 5)},
            "by_family_3y": bm_by_family(joined, 3),
        },
    }
    stem = Path(args.panel).stem.split("_")[-1]
    out_path = Path(args.panel).parent / f"v3_validation_suites_{stem}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"G4 suites + B/M control over {len(obs)} obs (B/M join {coverage['join_rate']}):")
    o3 = result["bm_control"]["overall"]["3y"]
    print(f"  3y discount coef | B/M control: mean {o3['discount_coef'].get('mean')} "
          f"NW t {o3['discount_coef'].get('t_stat')}")
    print(f"  3y B/M coef:                    mean {o3['bm_coef'].get('mean')} "
          f"NW t {o3['bm_coef'].get('t_stat')}")
    print(f"  3y B/M-residual discount IC:    mean {o3['residual_ic'].get('mean')} "
          f"NW t {o3['residual_ic'].get('t_stat')}")
    for fam, b in result["bm_control"]["by_family_3y"].items():
        ric = b["residual_ic"]
        print(f"    {fam:36} resid IC {ric.get('mean')} (NW t {ric.get('t_stat')}, n_d {b['n_dates']})")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
