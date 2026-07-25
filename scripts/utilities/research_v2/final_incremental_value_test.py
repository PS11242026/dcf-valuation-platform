"""FINAL incremental-value test -- does the DCF discount add anything beyond Book/Market, OUT OF SAMPLE?

The closing study of the conditioner-search program (user-specified). Four signals on the finalized
fully-corrected panel joined to Book/Market, evaluated with a true regime boundary (train pre-2020,
test 2020+, and the reverse where sample permits):

  1. Book/Market alone
  2. DCF discount alone
  3. Book/Market + DCF discount composite (equal-weight within-date rank)
  4. Residualized DCF discount (per-date orthogonalization against Book/Market)

Battery per signal per split: obs-weighted Rank-IC (+Newey-West t +block-bootstrap CI), pooled Spearman,
decile long-short spread (+NW t), decile monotonicity. Plus partial Spearman correlations (discount⊥B/M
and B/M⊥discount) and per-date incremental R² (fwd~B/M vs fwd~B/M+discount, and the reverse).

THE decisive OOS statistic -- a FROZEN combiner: fit per-date-mean OLS weights (a·rankz(B/M)+b·rankz(disc))
on TRAIN, freeze, apply to TEST, compare the combined IC to B/M-alone IC on TEST. Because the discount's
sign flips across the regime boundary, train-learned weights are the honest test of whether the discount
helps a B/M investor who could NOT see the future regime.

Verdict: A) incremental value survives  vs  B) does not survive. Eval-only, no MCMC, no src change.

Usage: python scripts/utilities/research_v2/final_incremental_value_test.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.validation_common as C  # noqa: E402

RV = ROOT / "reports" / "research_v2"
PANEL = RV / "final_universe_backtest_fully_corrected.json"
FACTORS = ROOT / "reports" / "_factor_panel.json"
CUT = "2020-01-01"
SIGNALS = {"book_to_market": "Book/Market", "discount_to_intrinsic": "DCF discount",
           "composite": "B/M + discount composite", "resid_disc": "discount ⊥ B/M (residual)"}


def _rank_z(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x)
    s = r.std()
    return (r - r.mean()) / s if s > 0 else r * 0.0


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def load_eval() -> list[dict]:
    obs = json.loads(PANEL.read_text(encoding="utf-8"))["observations"]
    pmap = {(p["ticker"], p["date"]): p for p in json.loads(FACTORS.read_text(encoding="utf-8"))}
    ev = []
    for o in obs:
        f = pmap.get((o["ticker"], o["date"]))
        if f is None or not isinstance(f.get("book_to_market"), (int, float)):
            continue
        if not isinstance(o.get("discount_to_intrinsic"), (int, float)):
            continue
        r = dict(o)
        r["book_to_market"] = f["book_to_market"]
        ev.append(r)
    # per-date composite (equal-weight pct-rank) + residualized discount (rank_z(disc) ⊥ rank_z(B/M))
    byd: dict[str, list[dict]] = defaultdict(list)
    for o in ev:
        byd[o["date"]].append(o)
    for g in byd.values():
        disc = np.array([o["discount_to_intrinsic"] for o in g], float)
        bm = np.array([o["book_to_market"] for o in g], float)
        pr_d = stats.rankdata(disc) / (len(g) + 1.0)
        pr_b = stats.rankdata(bm) / (len(g) + 1.0)
        rz_d, rz_b = _rank_z(disc), _rank_z(bm)
        denom = float(np.dot(rz_b, rz_b))
        beta = float(np.dot(rz_b, rz_d) / denom) if denom > 0 else 0.0
        resid = rz_d - beta * rz_b
        for i, o in enumerate(g):
            o["composite"] = float(0.5 * pr_d[i] + 0.5 * pr_b[i])
            o["resid_disc"] = float(resid[i])
    print(f"eval universe: {len(ev)} obs with BOTH discount and Book/Market")
    return ev


def metrics_for(rows: list[dict], y: int, key: str) -> dict:
    lag = C.HORIZON_QUARTERS[y]
    series = C.per_date_ic_series(rows, y, key)
    ic = C.obs_weighted_ic(series)
    nw = C.newey_west_tstat([v for _, _, v in series], lag)
    rk = f"fwd_{y}y"
    valid = [o for o in rows if isinstance(o.get(rk), (int, float)) and isinstance(o.get(key), (int, float))]
    sp = stats.spearmanr([o[key] for o in valid], [o[rk] for o in valid]).statistic if len(valid) > 4 else None
    buckets = C.per_date_buckets(rows, y, key)
    dec = buckets.get("decile_curve") or buckets.get("quintile_curve_fallback")
    mono = dec.get("monotonicity_spearman") if dec else None
    spr = C.spread_significance(buckets["per_date_spread"], y)
    boot = C.block_bootstrap_ci(rows, y, lambda r: C.obs_weighted_ic(C.per_date_ic_series(r, y, key)), lag)
    return {"n_dates": len(series), "ic_obs": ic, "ic_nw_t": nw.get("t_stat"),
            "boot_ci95": boot.get("ci95"), "boot_excl0": boot.get("excludes_zero"),
            "spearman_pooled": float(sp) if sp is not None and np.isfinite(sp) else None,
            "ls_spread": spr.get("mean_spread"), "ls_nw_t": spr.get("nw_tstat"),
            "decile_monotonicity": mono}


def partial_spearman(rows, y, x, z):
    """per-date partial Spearman of (x, fwd | z); returns mean + NW t."""
    rk = f"fwd_{y}y"
    byd: dict[str, list[dict]] = defaultdict(list)
    for o in rows:
        if all(isinstance(o.get(k), (int, float)) for k in (rk, x, z)):
            byd[o["date"]].append(o)
    vals = []
    for dt in sorted(byd):
        g = byd[dt]
        if len(g) < 8:
            continue
        xv = stats.rankdata([o[x] for o in g]); zv = stats.rankdata([o[z] for o in g])
        yv = stats.rankdata([o[rk] for o in g])
        rxy = np.corrcoef(xv, yv)[0, 1]; rxz = np.corrcoef(xv, zv)[0, 1]; ryz = np.corrcoef(yv, zv)[0, 1]
        denom = np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
        pc = (rxy - rxz * ryz) / denom
        if np.isfinite(pc):
            vals.append(pc)
    nw = C.newey_west_tstat(vals, C.HORIZON_QUARTERS[y])
    return {"mean": nw.get("mean"), "nw_t": nw.get("t_stat"), "n_dates": len(vals)}


def incremental_r2(rows, y, base, add):
    """mean per-date ΔR² from adding `add` to a regression of fwd on `base`."""
    rk = f"fwd_{y}y"
    byd: dict[str, list[dict]] = defaultdict(list)
    for o in rows:
        if all(isinstance(o.get(k), (int, float)) for k in (rk, base, add)):
            byd[o["date"]].append(o)
    d = []
    for dt, g in byd.items():
        if len(g) < 20:
            continue
        yv = _z(np.array([o[rk] for o in g], float))
        xb = _rank_z(np.array([o[base] for o in g], float))
        xa = _rank_z(np.array([o[add] for o in g], float))
        def r2(X):
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ beta
            return 1.0 - float(np.dot(resid, resid)) / float(np.dot(yv, yv))
        r2_base = r2(np.column_stack([np.ones(len(g)), xb]))
        r2_both = r2(np.column_stack([np.ones(len(g)), xb, xa]))
        d.append(r2_both - r2_base)
    return float(np.mean(d)) if d else None, len(d)


def fit_combiner(rows, y):
    rk = f"fwd_{y}y"
    byd: dict[str, list[dict]] = defaultdict(list)
    for o in rows:
        if all(isinstance(o.get(k), (int, float)) for k in (rk, "book_to_market", "discount_to_intrinsic")):
            byd[o["date"]].append(o)
    betas = []
    for g in byd.values():
        if len(g) < 20:
            continue
        yv = _z(np.array([o[rk] for o in g], float))
        X = np.column_stack([np.ones(len(g)),
                             _rank_z(np.array([o["book_to_market"] for o in g], float)),
                             _rank_z(np.array([o["discount_to_intrinsic"] for o in g], float))])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        betas.append(beta[1:])
    if not betas:
        return None, None, 0
    B = np.array(betas)
    return float(B[:, 0].mean()), float(B[:, 1].mean()), len(B)


def apply_combiner_ic(rows, y, a, b):
    rk = f"fwd_{y}y"
    byd: dict[str, list[dict]] = defaultdict(list)
    for o in rows:
        if all(isinstance(o.get(k), (int, float)) for k in (rk, "book_to_market", "discount_to_intrinsic")):
            byd[o["date"]].append(o)
    series = []
    for dt in sorted(byd):
        g = byd[dt]
        if len(g) < 4:
            continue
        score = (a * _rank_z(np.array([o["book_to_market"] for o in g], float)) +
                 b * _rank_z(np.array([o["discount_to_intrinsic"] for o in g], float)))
        fwd = np.array([o[rk] for o in g], float)
        ic = stats.spearmanr(score, fwd).statistic
        if np.isfinite(ic):
            series.append((dt, len(g), float(ic)))
    return C.obs_weighted_ic(series), len(series)


def split(rows, lo, hi):
    return [o for o in rows if lo <= o["date"] < hi]


def main():
    ev = load_eval()
    pre = split(ev, "0000-00-00", CUT)
    post = split(ev, CUT, "9999-99-99")
    result = {"cut": CUT, "n_eval": len(ev), "n_pre": len(pre), "n_post": len(post)}

    for y in (1, 3, 5):
        print("\n" + "=" * 96)
        print(f"  HORIZON {y}y")
        print("=" * 96)
        ry = {}
        for split_name, rows in (("TRAIN pre-2020", pre), ("TEST 2020+", post)):
            print(f"\n--- {split_name}  (n_obs={len(rows)}) ---")
            print(f"{'signal':>28} {'n_d':>4} {'IC_obs':>9} {'NW_t':>7} {'Spearman':>9} "
                  f"{'LS':>8} {'LS_t':>7} {'mono':>6}  boot95")
            ms = {}
            for key, pretty in SIGNALS.items():
                m = metrics_for(rows, y, key)
                ms[key] = m
                def g(v, f="{:+.4f}"):
                    return f.format(v) if isinstance(v, (int, float)) else "   n/a"
                boot = (f"[{m['boot_ci95'][0]:+.3f},{m['boot_ci95'][1]:+.3f}]{'*' if m['boot_excl0'] else ''}"
                        if m["boot_ci95"] else "n/a")
                print(f"{pretty:>28} {m['n_dates']:>4} {g(m['ic_obs'])} {g(m['ic_nw_t'],'{:+.2f}'):>7} "
                      f"{g(m['spearman_pooled']):>9} {g(m['ls_spread'],'{:+.3f}'):>8} "
                      f"{g(m['ls_nw_t'],'{:+.2f}'):>7} {g(m['decile_monotonicity'],'{:+.2f}'):>6}  {boot}")
            # partials + incremental R²
            pd_ = partial_spearman(rows, y, "discount_to_intrinsic", "book_to_market")
            pb_ = partial_spearman(rows, y, "book_to_market", "discount_to_intrinsic")
            dr2, ndr = incremental_r2(rows, y, "book_to_market", "discount_to_intrinsic")
            br2, nbr = incremental_r2(rows, y, "discount_to_intrinsic", "book_to_market")
            print(f"    partial ρ(disc,fwd|B/M)={pd_['mean']:+.4f} (t{pd_['nw_t']:+.2f}) | "
                  f"ρ(B/M,fwd|disc)={pb_['mean']:+.4f} (t{pb_['nw_t']:+.2f})")
            print(f"    incremental R²: disc adds to B/M {dr2:+.5f} | B/M adds to disc {br2:+.5f}")
            ry[split_name] = {"metrics": ms, "partial_disc_given_bm": pd_, "partial_bm_given_disc": pb_,
                              "incr_r2_disc_over_bm": dr2, "incr_r2_bm_over_disc": br2}

        # ---- decisive FROZEN-combiner OOS test (both directions where sample permits) ----
        print(f"\n  >>> FROZEN-COMBINER OOS TEST ({y}y) -- fit a·B/M + b·disc on TRAIN, apply to TEST <<<")
        oos = {}
        for tr_name, tr, te_name, te in (("pre-2020", pre, "2020+", post), ("2020+", post, "pre-2020", pre)):
            a, b, nfit = fit_combiner(tr, y)
            if a is None:
                continue
            bm_only_te, _ = apply_combiner_ic(te, y, 1.0, 0.0)
            comb_te, nte = apply_combiner_ic(te, y, a, b)
            lift = (comb_te - bm_only_te) if (comb_te is not None and bm_only_te is not None) else None
            print(f"    train {tr_name:8} -> weights a(B/M)={a:+.3f} b(disc)={b:+.3f} (n_fit={nfit}); "
                  f"test {te_name:8}: B/M-only IC={bm_only_te:+.4f}  combined IC={comb_te:+.4f}  "
                  f"lift={lift:+.4f}" + ("" if nte else ""))
            oos[f"train_{tr_name}_test_{te_name}"] = {"a_bm": a, "b_disc": b, "n_fit": nfit,
                                                      "bm_only_test_ic": bm_only_te, "combined_test_ic": comb_te,
                                                      "lift": lift, "n_test_dates": nte}
        ry["frozen_combiner_oos"] = oos
        result[f"{y}y"] = ry

    (RV / "final_incremental_value_test.json").write_text(json.dumps(result, indent=2, default=str),
                                                          encoding="utf-8")
    print(f"\nwrote {(RV / 'final_incremental_value_test.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
