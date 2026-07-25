"""Three-way comparison: V3 vs V1 vs Book/Market on the corrected broad panel.

Phase 2 asks whether v3's ranking edge is UNIQUE versus the Book/Market value factor,
or merely a (better-expressed) reload of it -- and whether v1 behaves the same way.

For each engine signal (v3 = discount_to_intrinsic, v1 = v1_discount) we compute, per
quarterly cross-section, the B/M-RESIDUALIZED discount IC: rank-z the discount, rank-z
book_to_market, residualize the discount rank on the B/M rank (OLS), Spearman the residual
against the forward return. The per-date IC series then gets a Newey-West HAC t-stat
(lag = horizon quarters) so overlapping windows are not overstated. Reported overall and
per family, for v3 and v1, at 3y and 5y. Read-only; writes a JSON ledger.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
for c in (ROOT, ROOT / "scripts"):
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))

import validation_common as vc  # noqa: E402

PANEL = ROOT / "reports" / "research_v3" / "tri_engine_backtest_broad.json"
FACTORS = ROOT / "reports" / "_factor_panel.json"
OUT = ROOT / "reports" / "research_v3" / "tri_bm_residual.json"
SIGNALS = {"v3": "discount_to_intrinsic", "v1": "v1_discount"}
HORIZONS = (3, 5)
MIN_XS = 5  # need >= MIN_XS names in a cross-section to residualize + correlate


def _rank_z(values) -> np.ndarray:
    r = stats.rankdata(values)
    return (r - r.mean()) / (r.std() or 1.0)


def _join_bm(obs: list[dict]) -> list[dict]:
    panel = json.loads(FACTORS.read_text(encoding="utf-8"))
    bm = {(r["ticker"], r["date"]): r.get("book_to_market") for r in panel}
    out = []
    for o in obs:
        v = bm.get((o["ticker"], o["date"]))
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.append({**o, "book_to_market": float(v)})
    return out


def _residual_ic_series(rows: list[dict], y: int, sig_key: str) -> list[float]:
    rk = f"fwd_{y}y"
    byd: dict[str, list[dict]] = {}
    for o in rows:
        if isinstance(o.get(rk), (int, float)) and isinstance(o.get(sig_key), (int, float)):
            byd.setdefault(o["date"], []).append(o)
    series = []
    for dt in sorted(byd):
        sub = byd[dt]
        if len(sub) < MIN_XS:
            continue
        zd = _rank_z([o[sig_key] for o in sub])
        zb = _rank_z([o["book_to_market"] for o in sub])
        fwd = np.array([o[rk] for o in sub])
        slope = float(np.dot(zd, zb) / (np.dot(zb, zb) or 1.0))
        resid = zd - slope * zb
        ic = stats.spearmanr(resid, fwd).statistic
        if np.isfinite(ic):
            series.append(float(ic))
    return series


def _raw_ic_series(rows: list[dict], y: int, sig_key: str) -> list[float]:
    return [ic for _dt, _n, ic in vc.per_date_ic_series(rows, y, sig_key)]


def _block(rows: list[dict]) -> dict:
    out = {}
    for y in HORIZONS:
        lag = vc.HORIZON_QUARTERS[y]
        h = {}
        for name, key in SIGNALS.items():
            raw = _raw_ic_series(rows, y, key)
            res = _residual_ic_series(rows, y, key)
            h[name] = {
                "raw_ic": vc.newey_west_tstat(raw, lag),
                "bm_resid_ic": vc.newey_west_tstat(res, lag),
            }
        out[f"{y}y"] = h
    return out


def _fmt_block(label: str, blk: dict) -> str:
    lines = [f"  {label}"]
    for y in HORIZONS:
        h = blk[f"{y}y"]
        for name in SIGNALS:
            raw = h[name]["raw_ic"]; res = h[name]["bm_resid_ic"]
            rt = raw.get("t_stat"); st = res.get("t_stat")
            rm = raw.get("mean"); sm = res.get("mean")
            def s(v):
                return f"{v:+.3f}" if isinstance(v, (int, float)) else "  None"
            def t(v):
                return f"{v:+.2f}" if isinstance(v, (int, float)) else " n/a"
            star = "*" if isinstance(st, (int, float)) and abs(st) >= 2.0 else " "
            lines.append(f"      {y}y {name}: raw IC {s(rm)} (t {t(rt)})  |  "
                         f"net of B/M {s(sm)} (t {t(st)}){star}")
    return "\n".join(lines)


def main() -> int:
    d = json.loads(PANEL.read_text(encoding="utf-8"))
    joined = _join_bm(d["observations"])
    fams: dict[str, list] = {}
    for o in joined:
        fams.setdefault(o.get("sector") or "unknown", []).append(o)

    results = {"join": {"n_obs": len(d["observations"]), "n_joined": len(joined)},
               "overall": _block(joined),
               "by_family": {}}
    print("=== V3 vs V1 vs Book/Market  (B/M-residualized discount IC, NW HAC) ===")
    print(f"  joined {len(joined)}/{len(d['observations'])} obs to factor panel\n")
    print(_fmt_block("OVERALL", results["overall"]))
    print("\n  -- BY FAMILY --")
    for fam in sorted(fams, key=lambda f: -len(fams[f])):
        if len(fams[fam]) < 100:
            continue
        blk = _block(fams[fam])
        results["by_family"][fam] = blk
        print(_fmt_block(f"{fam} (n={len(fams[fam])})", blk))

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
