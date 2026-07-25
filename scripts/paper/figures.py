"""Standalone figure generation for the research package -> reports/paper/figures/*.png.

Every figure is derived from an already-produced report artifact under reports/ (no model re-run). matplotlib
is imported lazily and guarded: if it is unavailable the module reports which figures it skipped and exits 0,
so the rest of the package still builds. Idempotent -- re-running overwrites the PNGs.

Usage:
    python -m scripts.paper.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PORT = ROOT / "reports" / "portfolio"
REPORTS = ROOT / "reports"
OUT = ROOT / "reports" / "paper" / "figures"


def _load_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def _load_json(p: Path) -> dict | None:
    return json.load(open(p)) if p.exists() else None


def generate() -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - environment-dependent
        print(f"[figures] matplotlib unavailable ({e}); skipping all figures.")
        return []

    OUT.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    def save(fig, name: str):
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=140, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    # 1) Equity curve: long-net NAV vs SPY/QQQ ------------------------------------------------------
    eq = _load_csv(PORT / "equity_curve.csv")
    if eq is not None:
        x = pd.to_datetime(eq.iloc[:, 0])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, eq["nav_long_net"], label="DCF-discount top-20 (net)", lw=2)
        if "nav_SPY" in eq:
            ax.plot(x, eq["nav_SPY"], label="SPY", lw=1.3, ls="--")
        if "nav_QQQ" in eq:
            ax.plot(x, eq["nav_QQQ"], label="QQQ", lw=1.3, ls=":")
        ax.set_yscale("log")
        ax.set_title("Growth of $1 -- DCF-discount long book (net) vs benchmarks")
        ax.set_ylabel("NAV (log)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        save(fig, "equity_curve.png")

    # 2) Top-N sweep: net CAGR and alpha vs SPY ----------------------------------------------------
    tn = _load_csv(PORT / "topn_sweep.csv")
    if tn is not None:
        d = tn[tn.signal == "discount"].sort_values("top_n")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(d["top_n"], d["cagr_net"] * 100, marker="o", label="Net CAGR")
        ax.plot(d["top_n"], d["alpha_spy"] * 100, marker="s", label="Alpha vs SPY")
        ax.set_xscale("log")
        ax.set_xticks(d["top_n"])
        ax.set_xticklabels(d["top_n"])
        ax.set_xlabel("Book size (top-N)")
        ax.set_ylabel("Annualized %")
        ax.set_title("Edge concentrates in the most-undervalued names")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        save(fig, "topn_sweep.png")

    # 3) Survivorship: long-short CAGR & alpha vs SPY across delisting hazard -----------------------
    sv = _load_json(PORT / "survivorship_stress.json")
    if sv is not None:
        hs = pd.DataFrame(sv["hazard_sweep"])
        be = sv.get("breakeven", {})
        cal = sv.get("edgar_calibration", {}).get("q_hazard_distress")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hs["d"] * 100, hs["long_short_cagr_mean"] * 100, marker="o", label="Long-short CAGR")
        ax.plot(hs["d"] * 100, hs["alpha_spy_mean"] * 100, marker="s", label="Long-only alpha vs SPY")
        ax.axhline(0, color="k", lw=0.8)
        if cal:
            ax.axvline(cal * 100, color="crimson", ls="--", lw=1.2,
                       label=f"EDGAR distress hazard {cal * 100:.1f}%/q")
        ax.set_xlabel("Per-quarter delisting hazard on held names (%)")
        ax.set_ylabel("Annualized %")
        ax.set_title("Survivorship stress: confidence-weighted long-short is robust")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "survivorship_curve.png")

    # 4) Capacity: realistic alpha vs universe across AUM ------------------------------------------
    cap = _load_csv(PORT / "capacity_costs.csv")
    if cap is not None:
        d = cap[cap.signal == "discount"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for book, g in d.groupby("book"):
            g = g.sort_values("aum")
            ax.plot(g["aum"], g["alpha_uni_real"] * 100, marker="o", label=book)
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("AUM ($, log)")
        ax.set_ylabel("Net alpha vs universe (%/yr, realistic)")
        ax.set_title("Capacity: alpha survives cost to scale; tradeability binds first")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "capacity_curve.png")

    # 5) Sector attribution: allocation vs selection (Brinson) ------------------------------------
    sa = _load_csv(PORT / "sector_attribution.csv")
    if sa is not None:
        d = sa[sa.signal == "discount"].sort_values("contribution_ann")
        import numpy as np
        y = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(y - 0.2, d["selection_ann"] * 100, height=0.4, label="Selection", color="#3b7")
        ax.barh(y + 0.2, d["allocation_ann"] * 100, height=0.4, label="Allocation", color="#c84")
        ax.set_yticks(y)
        ax.set_yticklabels(d["sector"], fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Annualized contribution (%)")
        ax.set_title("Edge is stock selection, not sector timing (Brinson)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3, axis="x")
        save(fig, "sector_attribution.png")

    # 6) IC by sector family (3y, obs-weighted) ----------------------------------------------------
    val = _load_json(REPORTS / "final_institutional_validation.json")
    if val is not None and "by_sector_family" in val:
        fams, ics = [], []
        for fam, b in val["by_sector_family"].items():
            h = b.get("3y") or b.get("1y")
            if h and h.get("ic_obs_weighted") is not None:
                fams.append(fam.replace("_", " "))
                ics.append(h["ic_obs_weighted"])
        order = sorted(range(len(ics)), key=lambda i: ics[i])
        fams = [fams[i] for i in order]
        ics = [ics[i] for i in order]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(fams, [v * 100 for v in ics], color=["#c43" if v < 0 else "#37b" for v in ics])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("3Y rank IC (obs-weighted, %)")
        ax.set_title("Predictive power is archetype-conditional")
        ax.grid(alpha=0.3, axis="x")
        save(fig, "ic_by_sector.png")

    # 7) Allocation schemes: alpha vs universe by construction (equal-weight is hard to beat) --------
    al = _load_csv(PORT / "allocation_study.csv")
    if al is not None:
        order = ["A_eqw", "B_discount", "C_disc_conf", "D_alpha_vol", "D_alpha_volrel", "E_opt"]
        d = al[al.signal == "discount"].set_index("scheme").reindex(order).dropna(how="all")
        a_alpha = d.loc["A_eqw", "alpha_uni"] if "A_eqw" in d.index else 0.0
        colors = []
        for s in d.index:
            t = d.loc[s, "active_vs_A_t"]
            if s == "A_eqw":
                colors.append("#37b")                                   # champion
            elif pd.notna(t) and abs(t) >= 2.0 and d.loc[s, "alpha_uni"] < a_alpha:
                colors.append("#c43")                                   # significantly worse
            else:
                colors.append("#9bb")                                   # tie
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(range(len(d)), d["alpha_uni"] * 100, color=colors)
        ax.axhline(a_alpha * 100, color="#37b", ls="--", lw=1,
                   label=f"Equal-weight Top-10 ({a_alpha * 100:.1f}%)")
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Alpha vs universe (%/yr, net)")
        ax.set_title("Weighting does not beat equal-weight; alpha/risk weighting loses")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        save(fig, "allocation_bars.png")

    # 8) Factor horse-race: 3Y IC by signal (discount vs value/quality/composites) -----------------
    fu = _load_csv(PORT / "factor_uniqueness.csv")
    if fu is not None:
        d = fu.sort_values("ic_3y")
        colors = ["#37b" if s == "DCF discount" else "#bbb" for s in d["signal"]]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(range(len(d)), d["ic_3y"] * 100, color=colors)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["signal"], fontsize=8)
        ax.set_xlabel("3Y obs-weighted rank IC (%)")
        ax.set_title("Factor horse-race: DCF discount vs value/quality strategies (3Y)")
        ax.grid(alpha=0.3, axis="x")
        save(fig, "factor_horse_race.png")

    print(f"[figures] wrote {len(made)} figures to {OUT}: {', '.join(made)}")
    return made


if __name__ == "__main__":
    generate()
