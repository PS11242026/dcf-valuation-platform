"""Research_V4 Phase B -- diagnostics & cross-sectional analysis of the snapshot panel.

Consumes reports/research_v4/v4_snapshot.json (Phase A) and produces, deterministically:
  * the three per-company gaps (vs market, vs consensus, market vs consensus);
  * aggregate distribution (mean / median / cap-weighted mean / dispersion / percentiles);
  * a per-sector structural profile (gap, CAP, fade speed, terminal contribution, ROIC
    persistence);
  * cross-sectional relationships of the valuation gap to the engine's structural drivers
    (Spearman rank correlations + a standardized OLS), to attribute *which* assumptions move
    the gap.

Outputs: v4_diagnostics.json, tables/v4_sector_breakdown.csv, tables/v4_cross_section.csv,
and figures under reports/research_v4/figures/. Reads nothing from src/ except as data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "research_v4"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"

# Structural drivers we regress the valuation gap on (engine outputs -> economic levers).
DRIVERS = [
    "cap_years", "roic_annual_0", "excess_roic_0", "phi", "half_life_growth_years",
    "half_life_margin_years", "terminal_growth", "pv_tv_share_of_ev",
    "g_base_effective", "margin_target_blend",
]


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x else None


# Two numerical/degenerate pathologies are flagged and EXCLUDED from the gap distribution,
# then reported as their own diagnostic category (they are bugs, not economic views):
#   (a) upside blow-up: intrinsic > 5x price (capital-structure sign error / terminal blow-up
#       / bad share count -- e.g. MCD at $234M/share, GIS at 21x);
#   (b) non-positive intrinsic: net debt exceeds EV so per-share value is <= 0 (degenerate).
# A positive-but-tiny intrinsic (e.g. SNPS worth <1% of price) is NOT flagged: that is the
# documented SBC / low-GAAP-NOPAT conservatism that is the very phenomenon under study.
PATHOLOGY_HIGH = 5.0


def load_rows() -> list[dict]:
    data = json.loads((REPORT_DIR / "v4_snapshot.json").read_text(encoding="utf-8"))
    return [r for r in data["rows"] if r.get("status") == "ok"]


def is_pathological(r: dict) -> bool:
    # Prefer the engine's OWN coherence verdict (Mod 1 shipped 2026-06-12): the guard now
    # classifies broken output as "incomplete" inside the result. Fall back to the legacy
    # numeric rule only for rows produced before the field existed.
    coh = r.get("valuation_coherence")
    if coh is not None:
        return coh == "incomplete"
    iv, px = _num(r.get("intrinsic")), _num(r.get("market_price"))
    if iv is None or px is None or px <= 0:
        return True
    return iv <= 0 or (iv / px) > PATHOLOGY_HIGH


def add_gaps(rows: list[dict]) -> list[dict]:
    for r in rows:
        iv, px, tgt = _num(r.get("intrinsic")), _num(r.get("market_price")), _num(r.get("consensus_target_mean"))
        r["valuation_gap"] = (iv / px - 1.0) if iv and px else None
        r["consensus_gap"] = (iv / tgt - 1.0) if iv and tgt else None
        r["market_consensus_gap"] = (px / tgt - 1.0) if px and tgt else None
    return rows


def _dist(values: list[float]) -> dict[str, Any]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "share_negative": float((arr < 0).mean()),
    }


def _cap_weighted_mean(rows: list[dict], key: str) -> float | None:
    num = den = 0.0
    for r in rows:
        v, w = _num(r.get(key)), _num(r.get("market_cap"))
        if v is not None and w:
            num += v * w
            den += w
    return float(num / den) if den else None


def aggregate(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for gap in ("valuation_gap", "consensus_gap", "market_consensus_gap"):
        vals = [r.get(gap) for r in rows]
        d = _dist([v for v in vals if v is not None])
        d["cap_weighted_mean"] = _cap_weighted_mean(rows, gap)
        out[gap] = d
    return out


def by_sector(rows: list[dict]) -> dict[str, Any]:
    sectors: dict[str, list[dict]] = {}
    for r in rows:
        sectors.setdefault(r["sector"], []).append(r)
    out: dict[str, Any] = {}
    for sec, srows in sectors.items():
        def avg(key: str) -> float | None:
            vals = [_num(r.get(key)) for r in srows]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None
        out[sec] = {
            "n": len(srows),
            "valuation_gap_mean": avg("valuation_gap"),
            "valuation_gap_median": float(np.median([r["valuation_gap"] for r in srows if r.get("valuation_gap") is not None])) if any(r.get("valuation_gap") is not None for r in srows) else None,
            "consensus_gap_mean": avg("consensus_gap"),
            "market_consensus_gap_mean": avg("market_consensus_gap"),
            "cap_years": avg("cap_years"),
            "half_life_growth_years": avg("half_life_growth_years"),
            "half_life_margin_years": avg("half_life_margin_years"),
            "pv_tv_share_of_ev": avg("pv_tv_share_of_ev"),
            "phi": avg("phi"),
            "excess_terminal": avg("excess_terminal"),
            "excess_roic_0": avg("excess_roic_0"),
            "terminal_growth": avg("terminal_growth"),
            "g_base_effective": avg("g_base_effective"),
            "implied_growth": avg("implied_growth"),
            "implied_growth_minus_model": avg("implied_growth_minus_model"),
            "margin_compression": _margin_compression(srows),
        }
    return out


def _margin_compression(rows: list[dict]) -> float | None:
    """Mean (margin_target_blend - margin_current): negative = model compresses margins."""
    deltas = []
    for r in rows:
        mt, mc = _num(r.get("margin_target_blend")), _num(r.get("margin_current"))
        if mt is not None and mc is not None:
            deltas.append(mt - mc)
    return float(np.mean(deltas)) if deltas else None


def cross_section(rows: list[dict]) -> dict[str, Any]:
    """Spearman(gap, driver) + standardized OLS of valuation_gap on all drivers + log size."""
    y = np.array([r.get("valuation_gap") for r in rows], dtype=object)
    mask = np.array([isinstance(v, (int, float)) for v in y])
    spearman: dict[str, Any] = {}
    cols = DRIVERS + ["log_market_cap"]
    X_full = {}
    for d in cols:
        if d == "log_market_cap":
            xv = [np.log(_num(r.get("market_cap"))) if _num(r.get("market_cap")) else None for r in rows]
        else:
            xv = [_num(r.get(d)) for r in rows]
        X_full[d] = xv
        pairs = [(x, r["valuation_gap"]) for x, r in zip(xv, rows)
                 if x is not None and isinstance(r.get("valuation_gap"), (int, float))]
        xa = [a for a, _ in pairs]
        if len(pairs) >= 6 and len(set(xa)) > 1 and len({b for _, b in pairs}) > 1:
            rho, p = stats.spearmanr(xa, [b for _, b in pairs])
            spearman[d] = {"rho": float(rho), "p": float(p), "n": len(pairs)}
        else:
            spearman[d] = {"rho": None, "p": None, "n": len(pairs),
                           "note": "constant or too few" if len(pairs) >= 6 else "too few"}

    ols = _standardized_ols(rows, cols, X_full)
    return {"spearman": spearman, "standardized_ols": ols}


def _standardized_ols(rows: list[dict], cols: list[str], X_full: dict[str, list]) -> dict[str, Any]:
    """OLS of valuation_gap on z-scored drivers (complete cases). Pure-numpy, deterministic."""
    yvals, rowsX = [], []
    for i, r in enumerate(rows):
        y = r.get("valuation_gap")
        xs = [X_full[c][i] for c in cols]
        if isinstance(y, (int, float)) and all(x is not None for x in xs):
            yvals.append(float(y))
            rowsX.append([float(x) for x in xs])
    if len(yvals) < len(cols) + 2:
        return {"n": len(yvals), "note": "insufficient complete cases for OLS", "coef": {}}
    X = np.array(rowsX)
    y = np.array(yvals)
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    Xd = np.column_stack([np.ones(len(y)), Xz])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else None
    return {
        "n": len(yvals),
        "r2": r2,
        "intercept": float(beta[0]),
        "coef": {c: float(b) for c, b in zip(cols, beta[1:])},
        "note": "coefficients are change in valuation_gap per 1 sd of the z-scored driver",
    }


def make_figures(rows: list[dict], sector: dict[str, Any]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    gaps = [r["valuation_gap"] for r in rows if r.get("valuation_gap") is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gaps, bins=20, color="#3b6ea5", edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(float(np.median(gaps)), color="#c0392b", ls="--", label=f"median {np.median(gaps):.0%}")
    ax.set_title("V3 valuation gap (intrinsic/price - 1)")
    ax.set_xlabel("valuation gap"); ax.legend()
    p = FIG_DIR / "v4_gap_distribution.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    made.append(p.name)

    # per-sector mean gap bars
    secs = list(sector.keys())
    means = [sector[s]["valuation_gap_mean"] for s in secs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(secs)), means, color="#3b6ea5")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(secs))); ax.set_xticklabels(secs, rotation=30, ha="right", fontsize=8)
    ax.set_title("Mean V3 valuation gap by sector"); ax.set_ylabel("mean gap")
    p = FIG_DIR / "v4_sector_gap.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    made.append(p.name)

    # gap vs PV(TV)/EV scatter
    xs = [(r.get("pv_tv_share_of_ev"), r.get("valuation_gap")) for r in rows]
    xs = [(a, b) for a, b in xs if isinstance(a, (int, float)) and isinstance(b, (int, float))]
    if xs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([a for a, _ in xs], [b for _, b in xs], s=20, color="#3b6ea5")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("PV(TV)/EV"); ax.set_ylabel("valuation gap"); ax.set_title("Gap vs terminal-value reliance")
        p = FIG_DIR / "v4_gap_vs_pvtv.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        made.append(p.name)
    return made


def write_csvs(rows: list[dict], sector: dict[str, Any], xsec: dict[str, Any]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sec_cols = ["sector", "n", "valuation_gap_mean", "valuation_gap_median", "consensus_gap_mean",
                "market_consensus_gap_mean", "cap_years", "half_life_growth_years",
                "half_life_margin_years", "pv_tv_share_of_ev", "phi", "excess_roic_0",
                "terminal_growth", "g_base_effective", "implied_growth",
                "implied_growth_minus_model", "margin_compression"]
    with (TABLE_DIR / "v4_sector_breakdown.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(sec_cols)
        for sec, d in sector.items():
            w.writerow([sec] + [d.get(c) for c in sec_cols[1:]])

    with (TABLE_DIR / "v4_cross_section.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["driver", "spearman_rho", "spearman_p", "n", "ols_std_coef"])
        ols = xsec["standardized_ols"]["coef"]
        for d, s in xsec["spearman"].items():
            w.writerow([d, s["rho"], s["p"], s["n"], ols.get(d)])


def main() -> None:
    all_rows = add_gaps(load_rows())
    pathological = [r for r in all_rows if is_pathological(r)]
    rows = [r for r in all_rows if not is_pathological(r)]
    diagnostics = {
        "n_ok": len(all_rows),
        "n_clean": len(rows),
        "pathological": [
            {"ticker": r["ticker"], "sector": r["sector"], "intrinsic": r.get("intrinsic"),
             "market_price": r.get("market_price"),
             "intrinsic_over_price": (r.get("intrinsic") / r["market_price"])
             if r.get("market_price") else None,
             "valuation_coherence": r.get("valuation_coherence"),
             "coherence_flags": r.get("coherence_flags")}
            for r in pathological
        ],
        "pathology_rule": "engine_v3 coherence guard verdict == 'incomplete' "
                          "(corrupt_shares / negative_equity / terminal_blowup / implausible_vs_price)",
        "aggregate": aggregate(rows),
        "by_sector": by_sector(rows),
        "cross_section": cross_section(rows),
    }
    figs = make_figures(rows, diagnostics["by_sector"])
    diagnostics["figures"] = figs
    (REPORT_DIR / "v4_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    write_csvs(rows, diagnostics["by_sector"], diagnostics["cross_section"])

    agg = diagnostics["aggregate"]["valuation_gap"]
    print(f"clean n={len(rows)} (+{len(pathological)} pathological excluded)")
    print(f"valuation_gap median={agg['median']:.1%} mean={agg['mean']:.1%} "
          f"cap-wtd={agg['cap_weighted_mean']:.1%}  share_neg={agg['share_negative']:.0%}")
    print("pathological:", [r["ticker"] for r in pathological])
    print("Wrote v4_diagnostics.json + sector/cross-section CSVs + figures:", figs)


if __name__ == "__main__":
    main()
