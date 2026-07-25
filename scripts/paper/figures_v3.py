"""Cross-market figures for the v3 (three-market) manuscript -> reports/paper_v3/figures/.

New figures only (US figures are reused verbatim from ``figures_v2.generate``):
- ``where_dcf_works_heatmap``  : market x sector rank-IC matrix (the centrepiece overview).
- ``india_canada_sector_ic``   : India vs Canada 1y by-sector IC, same horizon + sector vocab
                                 (the controlled emerging-vs-developed-commodity comparison).
- ``india_sector_ic`` / ``canada_sector_ic`` : per-market by-sector 1y IC, v3 with v1 overlay.
- ``market_discount_distribution`` : India + Canada cross-sectional discount medians by sector.
- ``cross_market_paired`` : paired v3-v1 IC by market (the economic engine adds little broadly).

Matplotlib-guarded: any figure failure prints and returns None, never breaks the build.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
R2 = REPORTS / "research_v2"
R3 = REPORTS / "research_v3"
IN_BROAD = REPORTS / "research_v3_india" / "india_v3_backtest_broad.json"
CA_BROAD = REPORTS / "research_v3_canada" / "canada_v3_backtest_broad.json"

NAVY, BLUE, RED, GREEN, GREY = "#1F3864", "#2E5496", "#B23A1E", "#2E7D32", "#9AA3B2"

#: Common coarse sector vocabulary (column order) for the cross-market matrix.
SECTORS = ["energy", "materials", "industrials", "utilities", "consumer_staples",
           "consumer_discretionary", "technology", "communication_services", "health_care"]
SECTOR_SHORT = {
    "energy": "Energy", "materials": "Materials", "industrials": "Industrials",
    "utilities": "Utilities", "consumer_staples": "Cons.\nstaples",
    "consumer_discretionary": "Cons.\ndisc.", "technology": "Tech", "communication_services": "Comm.",
    "health_care": "Health\ncare",
}
#: US research families -> coarse sector (US tech families are averaged into "technology").
US_FAMILY_TO_SECTOR = {
    "consumer_staples": "consumer_staples", "consumer_discretionary": "consumer_discretionary",
    "software_saas_infra": "technology", "internet_platforms_marketplaces": "technology",
    "semiconductors": "technology", "semiconductor_equipment": "technology",
    "industrials_assetlight_services": "industrials", "health_care": "health_care",
    "technology": "technology",
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _j(path: Path):
    return json.load(open(path)) if path.exists() else None


def _guard(fn):
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:  # noqa: BLE001
            print(f"  [fig_v3] {fn.__name__} skipped: {type(exc).__name__}: {exc}")
            return None
    return wrap


def _save(fig, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out.name


def _ic_obs(block, hz):
    return ((block or {}).get(hz, {}) or {}).get("ic_obs_weighted")


def _us_sector_ic_3y() -> dict[str, float]:
    """US corrected 3y IC mapped to coarse sectors (tech families averaged)."""
    cv = _j(R2 / "correction_validation.json") or {}
    fam = cv.get("by_family_3y", {})
    buckets: dict[str, list[float]] = {}
    for f, v in fam.items():
        sec = US_FAMILY_TO_SECTOR.get(f)
        if sec and v.get("full") is not None:
            buckets.setdefault(sec, []).append(v["full"])
    return {s: sum(vs) / len(vs) for s, vs in buckets.items()}


def _market_sector_ic_1y(path: Path) -> dict[str, float]:
    """India/Canada by-sector 1y obs-weighted IC."""
    d = _j(path) or {}
    out: dict[str, float] = {}
    for sec, blk in (d.get("by_sector", {}) or {}).items():
        ic = _ic_obs(blk.get("metrics", {}), "1y")
        if ic is not None:
            out[sec] = ic
    return out


# ---------------------------------------------------------------- centrepiece
@_guard
def where_dcf_works_heatmap(figs: Path) -> str | None:
    """Market x sector rank-IC matrix. US row = corrected 3y by-family (coarse); India/Canada
    rows = 1y by-sector. Green = signal works, red = wrong-signed, grey = sector absent."""
    import numpy as np

    us = _us_sector_ic_3y()
    ca = _market_sector_ic_1y(CA_BROAD)
    ind = _market_sector_ic_1y(IN_BROAD)
    if not (us or ca or ind):
        return None
    rows = [("United States (3y)", us), ("Canada (1y)", ca), ("India (1y)", ind)]
    M = np.full((len(rows), len(SECTORS)), np.nan)
    for i, (_, d) in enumerate(rows):
        for j, sec in enumerate(SECTORS):
            if sec in d:
                M[i, j] = d[sec]
    plt = _mpl()
    from matplotlib.colors import TwoSlopeNorm
    masked = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#E8E8E8")
    vmax = max(0.25, float(np.nanmax(np.abs(M)))) if np.isfinite(M).any() else 0.25
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(SECTORS)))
    ax.set_xticklabels([SECTOR_SHORT[s] for s in SECTORS], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="#111" if abs(M[i, j]) < 0.30 else "white")
    ax.set_title("Where the DCF discount ranks forward returns (rank IC, by market x sector)",
                 fontsize=11, color=NAVY, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Rank IC")
    fig.text(0.5, -0.04, "US at 3-year horizon (corrected by-family, tech families averaged); "
             "India & Canada at 1-year. Grey = sector not in that market's DCF universe.",
             ha="center", fontsize=7.5, color="#555")
    return _save(fig, figs / "where_dcf_works_heatmap.png")


@_guard
def india_canada_sector_ic(figs: Path) -> str | None:
    """India vs Canada 1y by-sector IC (same horizon + sector vocab): does the cyclical-sectors-
    predict pattern hold in BOTH an emerging (India) and a developed-commodity (Canada) market?"""
    import numpy as np

    ind = _market_sector_ic_1y(IN_BROAD)
    ca = _market_sector_ic_1y(CA_BROAD)
    if not (ind and ca):
        return None
    secs = [s for s in SECTORS if s in ind or s in ca]
    y = np.arange(len(secs))
    iv = [ind.get(s, np.nan) for s in secs]
    cv = [ca.get(s, np.nan) for s in secs]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.barh(y - 0.2, iv, 0.4, label="India (1y)", color=RED)
    ax.barh(y + 0.2, cv, 0.4, label="Canada (1y)", color=NAVY)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([s.replace("_", " ").title() for s in secs], fontsize=8)
    ax.set_xlabel("1-year obs-weighted rank IC")
    ax.set_title("Cyclical vs stable: 1y by-sector IC, India vs Canada",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="x", alpha=0.25)
    return _save(fig, figs / "india_canada_sector_ic.png")


def _per_market_sector_ic(path: Path, title: str, fname: str):
    @_guard
    def _fig(figs: Path) -> str | None:
        import numpy as np
        d = _j(path) or {}
        bysec = d.get("by_sector", {}) or {}
        rows = []
        for sec, blk in bysec.items():
            v3 = _ic_obs(blk.get("metrics", {}), "1y")
            v1 = _ic_obs(blk.get("metrics_v1", {}), "1y")
            n = blk.get("n_obs", 0)
            if v3 is not None:
                rows.append((sec, v3, v1, n))
        if not rows:
            return None
        rows.sort(key=lambda r: (r[1] if r[1] is not None else -9))
        labels = [f"{r[0].replace('_', ' ').title()} (n={r[3]})" for r in rows]
        v3v = [r[1] for r in rows]
        v1v = [r[2] if r[2] is not None else np.nan for r in rows]
        y = np.arange(len(rows))
        plt = _mpl()
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        ax.barh(y - 0.2, v3v, 0.4, label="v3 (economic fade)", color=RED)
        ax.barh(y + 0.2, v1v, 0.4, label="v1 (frozen)", color=BLUE)
        ax.axvline(0, color="#444", linewidth=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("1-year obs-weighted rank IC")
        ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
        ax.legend(fontsize=8, frameon=False); ax.grid(axis="x", alpha=0.25)
        return _save(fig, figs / fname)
    return _fig


india_sector_ic = _per_market_sector_ic(
    IN_BROAD, "India: 1y by-sector IC (v3 vs frozen v1)", "india_sector_ic.png")
canada_sector_ic = _per_market_sector_ic(
    CA_BROAD, "Canada: 1y by-sector IC (v3 vs frozen v1)", "canada_sector_ic.png")


@_guard
def market_discount_distribution(figs: Path) -> str | None:
    """Median v3 discount by sector for India and Canada (the 'what looks cheap' map): the
    standout call is Indian PSU energy (the only sector the model finds cheap)."""
    import numpy as np

    ind = _j(IN_BROAD) or {}
    ca = _j(CA_BROAD) or {}
    in_ref = (ind.get("reference_distribution", {}) or {}).get("by_sector", {})
    ca_ref = (ca.get("reference_distribution", {}) or {}).get("by_sector", {})
    if not (in_ref or ca_ref):
        return None
    secs = [s for s in SECTORS if s in in_ref or s in ca_ref]
    y = np.arange(len(secs))
    iv = [in_ref.get(s, {}).get("p50", np.nan) for s in secs]
    cv = [ca_ref.get(s, {}).get("p50", np.nan) for s in secs]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.barh(y - 0.2, iv, 0.4, label="India median discount", color=RED)
    ax.barh(y + 0.2, cv, 0.4, label="Canada median discount", color=NAVY)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([s.replace("_", " ").title() for s in secs], fontsize=8)
    ax.set_xlabel("Median v3 discount-to-intrinsic (latest cross-section)")
    ax.set_title("What the DCF finds cheap: median discount by sector",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="x", alpha=0.25)
    return _save(fig, figs / "market_discount_distribution.png")


@_guard
def cross_market_paired(figs: Path) -> str | None:
    """Paired v3 - v1 rank-IC diff by market (US 3y corrected; India/Canada 1y). The economic
    engine is, at best, a narrow improvement and is not a broad alpha source in any market."""

    psig = _j(R3 / "tri_paired_significance.json") or {}
    us3 = ((psig.get("overall", {}).get("horizons", {}) or {}).get("3y", {}) or {}).get("mean_ic_diff")
    ind = _j(IN_BROAD) or {}
    ca = _j(CA_BROAD) or {}
    in1 = ((ind.get("paired_ic_diff_overall", {}) or {}).get("1y", {}) or {}).get("mean_ic_diff")
    ca1 = ((ca.get("paired_ic_diff_overall", {}) or {}).get("1y", {}) or {}).get("mean_ic_diff")
    pairs = [("US (3y)", us3), ("India (1y)", in1), ("Canada (1y)", ca1)]
    pairs = [(lbl, v) for lbl, v in pairs if v is not None]
    if not pairs:
        return None
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [GREEN if v > 0 else RED for v in vals]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(labels, vals, color=colors, width=0.55)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_ylabel("Paired v3 - v1 rank IC")
    ax.set_title("Economic fade vs frozen baseline, by market",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, figs / "cross_market_paired.png")


def generate(base: Path) -> list[str]:
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    made = [where_dcf_works_heatmap(figs), india_canada_sector_ic(figs),
            india_sector_ic(figs), canada_sector_ic(figs),
            market_discount_distribution(figs), cross_market_paired(figs)]
    return [m for m in made if m]
