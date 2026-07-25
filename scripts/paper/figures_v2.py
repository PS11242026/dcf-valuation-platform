"""Figures for the corrected manuscript (v2) -> reports/paper_v2/figures/.

Matplotlib-guarded (a figure failure prints and returns None, never breaks the build).
Reads the same frozen JSON artifacts as build_paper_v2 plus, when present, the KO
fade-paths driver-audit sidecar produced by `valuation dcf KO`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
R2 = REPORTS / "research_v2"
R3 = REPORTS / "research_v3"
KO_AUDIT = REPORTS / "valuation" / "KO" / "driver_audit.json"

NAVY, BLUE, RED, GREEN, GREY = "#1F3864", "#2E5496", "#B23A1E", "#2E7D32", "#9AA3B2"
FAMILY_LABEL = {
    "consumer_staples": "Cons. staples", "consumer_discretionary": "Cons. disc.",
    "internet_platforms_marketplaces": "Internet platforms", "semiconductors": "Semiconductors",
    "semiconductor_equipment": "Semi equip.", "software_saas_infra": "Software/SaaS",
    "industrials_assetlight_services": "Industrials (AL)", "health_care": "Health care",
    "technology": "Technology", "financials": "Financials",
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
            print(f"  [fig_v2] {fn.__name__} skipped: {type(exc).__name__}: {exc}")
            return None
    return wrap


def _save(fig, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out.name


@_guard
def orig_vs_corrected_ic(figs: Path) -> str | None:
    cv = _j(R2 / "correction_validation.json")
    if not cv:
        return None
    plt = _mpl()
    hz = ["1y", "3y", "5y"]
    orig = [cv["overall"][h]["original"]["ic"] for h in hz]
    corr = [cv["overall"][h]["full"]["ic"] for h in hz]
    import numpy as np
    x = np.arange(len(hz))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - 0.2, orig, 0.4, label="Original (uncorrected)", color=GREY)
    ax.bar(x + 0.2, corr, 0.4, label="Fully corrected", color=NAVY)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(hz)
    ax.set_ylabel("Rank IC"); ax.set_title("DCF discount: original vs split+dividend corrected IC",
                                            fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.25)
    return _save(fig, figs / "orig_vs_corrected_ic.png")


@_guard
def correction_decomposition(figs: Path) -> str | None:
    cv = _j(R2 / "correction_validation.json")
    if not cv:
        return None
    plt = _mpl()
    b = cv["overall"]["3y"]
    keys = [("original", "Original"), ("split_only", "Split only"),
            ("div_only", "Dividend only"), ("full", "Fully corrected")]
    vals = [b[k]["ic"] for k, _ in keys]
    colors = [GREY, RED, GREEN, NAVY]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar([lbl for _, lbl in keys], vals, color=colors)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_ylabel("3-year rank IC")
    ax.set_title("Correction decomposition (3y): the split adjustment is decisive",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=0.25)
    return _save(fig, figs / "correction_decomposition.png")


@_guard
def ic_by_family(figs: Path) -> str | None:
    cv = _j(R2 / "correction_validation.json")
    if not cv:
        return None
    plt = _mpl()
    fam = cv["by_family_3y"]
    items = sorted(fam.items(), key=lambda kv: kv[1]["full"])
    labels = [FAMILY_LABEL.get(k, k) for k, _ in items]
    vals = [v["full"] for _, v in items]
    colors = [GREEN if x > 0.05 else (RED if x < 0 else GREY) for x in vals]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Corrected 3-year rank IC")
    ax.set_title("What survives correction (3y IC by family)", fontsize=11, color=NAVY, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8); ax.grid(axis="x", alpha=0.25)
    return _save(fig, figs / "ic_by_family.png")


@_guard
def v3_paired_ic(figs: Path) -> str | None:
    paired = _j(R3 / "v3_paired_summary_n50.json")
    if not paired:
        return None
    plt = _mpl()
    import numpy as np
    hz = ["1y", "3y", "5y"]
    head = paired["headline"]
    v3 = [head[h]["v3_ic_obs"] for h in hz]
    v1 = [head[h]["v1_ic_obs"] for h in hz]
    x = np.arange(len(hz))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - 0.2, v1, 0.4, label="v1 (frozen)", color=BLUE)
    ax.bar(x + 0.2, v3, 0.4, label="v3 (economic fade)", color=RED)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(hz)
    ax.set_ylabel("Obs-weighted rank IC")
    ax.set_title("G2: economic engine does not out-rank the baseline", fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.25)
    return _save(fig, figs / "v3_paired_ic.png")


@_guard
def mc_coverage(figs: Path) -> str | None:
    mc3, mc5 = _j(R3 / "v3_mc_calibration_n50_3y.json"), _j(R3 / "v3_mc_calibration_n50_5y.json")
    if not mc3:
        return None
    plt = _mpl()
    labels, cov = [], []
    for tag, mc in (("3y", mc3), ("5y", mc5)):
        if mc:
            labels.append(tag); cov.append(mc["coverage_p10_p90"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(labels, cov, color=RED, width=0.5, label="Realized p10-p90 coverage")
    ax.axhline(0.80, color=NAVY, linestyle="--", linewidth=1.4, label="Nominal 80%")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Coverage of realized return")
    ax.set_title("G6: driver-MC bands are uninformative (coverage << nominal)",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.25)
    return _save(fig, figs / "mc_coverage.png")


@_guard
def v3_corrected_paired(figs: Path) -> str | None:
    """The reversal: overall paired v3-v1 IC by horizon on the CORRECTED panel, with
    block-bootstrap 95% CIs (the kill panel's defect is corrected here)."""
    psig = _j(R3 / "tri_paired_significance.json")
    if not psig:
        return None
    plt = _mpl()
    import numpy as np
    hz = ["1y", "3y", "5y"]
    h = psig.get("overall", {}).get("horizons", {})
    # plot the block-bootstrap mean so the bar and its 95% CI are the SAME estimator
    centers = [h.get(z, {}).get("boot_mean") for z in hz]
    cis = [h.get(z, {}).get("boot_ci95") or [None, None] for z in hz]
    lo = [max(0.0, c - ci[0]) if (c is not None and ci[0] is not None) else 0 for c, ci in zip(centers, cis)]
    hiv = [max(0.0, ci[1] - c) if (c is not None and ci[1] is not None) else 0 for c, ci in zip(centers, cis)]
    x = np.arange(len(hz))
    colors = [GREEN if (h.get(z, {}).get("boot_excludes_zero")) else GREY for z in hz]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x, centers, 0.5, color=colors, yerr=[lo, hiv], capsize=5,
           error_kw={"ecolor": "#444", "elinewidth": 1.2})
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(hz)
    ax.set_ylabel("Paired v3 - v1 rank IC (block-bootstrap mean, 95% CI)")
    ax.set_title("Corrected panel: v3 out-ranks v1 at 3-5y (bootstrap CI excludes 0)",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, figs / "v3_corrected_paired.png")


@_guard
def v3_bm_control(figs: Path) -> str | None:
    """The boundary: v3 discount IC net of Book/Market by family (3y). Green = survives the
    value factor (scoped alpha), red = wrong-signed net of value, grey = spanned/inconclusive."""
    bm = _j(R3 / "tri_bm_residual.json")
    if not bm:
        return None
    plt = _mpl()
    T = 2.0
    rows = []
    for fam, blk in bm.get("by_family", {}).items():
        r = blk.get("3y", {}).get("v3", {}).get("bm_resid_ic", {})
        m, t = r.get("mean"), r.get("t_stat")
        if m is not None and t is not None:
            rows.append((FAMILY_LABEL.get(fam, fam), m, t))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [GREEN if (t >= T and m > 0) else (RED if (t <= -T and m < 0) else GREY)
              for _, m, t in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("v3 discount rank IC, net of Book/Market (3y)")
    ax.set_title("Net of Book/Market: scoped edge in consumer families only",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8); ax.grid(axis="x", alpha=0.25)
    return _save(fig, figs / "v3_bm_control.png")


@_guard
def fade_paths_ko(figs: Path) -> str | None:
    audit = _j(KO_AUDIT)
    if not audit or "paths" not in audit:
        print("  [fig_v2] fade_paths_KO skipped: run `valuation dcf KO` to generate the sidecar.")
        return None
    plt = _mpl()
    p = audit["paths"]
    q = list(range(1, len(p["growth_annual"]) + 1))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(q, p["growth_annual"], color=NAVY, label="Revenue growth")
    ax.plot(q, p["nopat_margin"], color=GREEN, label="NOPAT margin")
    ax.plot(q, p["ronic"], color=BLUE, label="RONIC")
    ax.plot(q, p["reinvestment_rate"], color=GREY, linestyle="--", label="Reinvestment (derived)")
    gk = (audit.get("grinold_kroner") or {}).get("terminal_growth")
    if gk is not None:
        ax.axhline(gk, color=RED, linestyle=":", linewidth=1.2, label="Terminal growth")
    ax.set_xlabel("Quarter"); ax.set_ylabel("Annual rate")
    ax.set_title("Worked example (KO): fade paths with derived reinvestment",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=2); ax.grid(alpha=0.25)
    return _save(fig, figs / "fade_paths_KO.png")


def generate(base: Path) -> list[str]:
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    made = [orig_vs_corrected_ic(figs), correction_decomposition(figs), ic_by_family(figs),
            v3_paired_ic(figs), v3_corrected_paired(figs), v3_bm_control(figs),
            mc_coverage(figs), fade_paths_ko(figs)]
    return [m for m in made if m]
