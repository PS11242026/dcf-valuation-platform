"""India v3 book -- full institutional export (Excel + PowerPoint + PDF).

The US managed book has a one-command dashboard/export; the India v3 book did not. This builds the
same three-format package for the standalone India book by assembling ONE ``ReportModel`` from the
locked book state (``reports/managed_portfolios/india_v3_aggressive/state.json``) plus the India
Phase B evidence, then rendering it through the shared reporting kit
(``reporting_kit.xlsx / pdf / pptx``). It reuses the kit's figure helpers (``viz3d``) and adds a few
INR-aware, portfolio-specific charts. No live valuation is performed: the book is already valued and
locked, so the export reads the snapshot and stays deterministic / offline.

Run:  python scripts/portfolio/india_v3_export.py
Then: python -m dcf_project export pdf india_v3_aggressive   (model is also persisted for the CLI)

Authoring note: all prose written here deliberately avoids en/em dashes (per the book owner's request
for the investment-philosophy essay); only hyphens, commas, semicolons and parentheses are used.
"""
from __future__ import annotations

import json
from pathlib import Path

from dcf_project.paths import MANAGED
from dcf_project.reporting_kit import viz3d
from dcf_project.reporting_kit.model import Image, Metric, Section, Table
from dcf_project.reporting_kit.pdf import render as render_pdf
from dcf_project.reporting_kit.persist import save_model
from dcf_project.reporting_kit.pptx import render as render_pptx
from dcf_project.reporting_kit.templates import new_report
from dcf_project.reporting_kit.xlsx import render as render_xlsx

BOOK_NAME = "india_v3_aggressive"
FOLDER = MANAGED / BOOK_NAME
FIGS = FOLDER / "figures"

# Human sector descriptors for the holdings table (the state stores the bare GICS "energy").
SECTOR_DESC = {
    "IOC": "Energy (PSU refiner/mktg)",
    "BPCL": "Energy (PSU refiner/mktg)",
    "ONGC": "Energy (PSU upstream)",
    "COALINDIA": "Energy (PSU coal)",
    "HINDPETRO": "Energy (PSU refiner/mktg)",
}

# India Phase B evidence (reports/research_v3_india/india_v3_phase_b_report.md). Static, cited.
PHASE_B_OVERALL = [          # horizon, n, IC(obs-wt), quintile L/S
    ["1y", 636, 0.218, 0.289],
    ["2y", 272, 0.368, 0.195],
]
PHASE_B_SECTOR_IC = [        # sector, 1y IC (obs-wt), n, read
    ["Materials", 0.489, 192, "strong"],
    ["Energy", 0.476, 91, "strong"],
    ["Utilities", 0.464, 90, "strong"],
    ["Industrials", 0.363, 108, "solid"],
    ["Consumer discretionary", 0.218, 142, "moderate"],
    ["Technology", 0.003, 102, "flat"],
    ["Consumer staples", -0.053, 113, "none"],
    ["Health care", -0.439, 146, "anti-predictive"],
]
VALIDATED = {"Materials", "Energy", "Utilities", "Industrials"}

# Key assumption differences vs the US book (assumption, US, India). Sourced from the India bolt-on
# code paths (data_sources/india/macro.py, engine_v3/sector_priors.py, portfolio/scoring.py).
ASSUMPTION_DIFFS = [
    ["Currency", "USD ($)", "INR (Rs)"],
    ["Filings / accounting", "SEC EDGAR, US GAAP", "NSE / BSE XBRL, Ind-AS"],
    ["Risk-free rate", "US 10Y Treasury", "India 10Y G-Sec (live, ~6.8%)"],
    ["Equity risk premium", "~4.5%", "~8.0% (Damodaran India, live)"],
    ["Statutory tax rate", "21%", "25.17% (new corporate regime)"],
    ["Cost of debt", "Blended issuer curve", "~9.0% (India lending proxy)"],
    ["Terminal growth floor", "~2.0% (fixed)", "India CPI (live, ~5%)"],
    ["Terminal growth cap", "~3.5% (fixed)", "India 10Y G-Sec (live, ~6.8%)"],
    ["Archetype source", "Asset intensity (capex/PPE/R&D)", "GICS-derived (NSE results omit capex)"],
    ["Verdict reference", "US cross-section percentile", "India cross-section percentile"],
    ["WACC source", "Provider-first, internal fallback", "Provider-first via .NS, same fallback"],
    ["Where the signal works", "Consumer staples reliable", "Energy/materials/utilities/industrials"],
]


# ---------------------------------------------------------------------------
# INR-aware, portfolio-specific figures (the kit's viz3d hardcodes a $ axis on
# football/tornado, so these few are authored here; concentration is reused).
# ---------------------------------------------------------------------------
NAVY, BLUE, GREEN, RED, GREY, AMBER = "#1F3864", "#2E5496", "#2E7D32", "#B23A1E", "#9AA3B2", "#BF8F00"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def fig_allocation(holdings, out: Path) -> Path:
    plt = _mpl()
    h = sorted(holdings, key=lambda r: r["market_value"] or 0)
    names = [r["ticker"] for r in h]
    vals = [(r["market_value"] or 0) / 1000.0 for r in h]   # Rs thousands
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.barh(range(len(h)), vals, color=BLUE)
    for i, (v, r) in enumerate(zip(vals, h)):
        ax.text(v, i, f"  Rs {v * 1000:,.0f}  ({r['weight']:.1%})", va="center", fontsize=8, color="#333")
    ax.set_yticks(range(len(h))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Market value (Rs thousands)")
    ax.set_title("Allocation by holding (conviction-weighted)", fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def fig_price_vs_intrinsic(holdings, out: Path) -> Path:
    plt = _mpl()
    names = [r["ticker"] for r in holdings]
    px = [r["entry_price"] for r in holdings]
    iv = [r["intrinsic"] for r in holdings]
    x = range(len(holdings))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    w = 0.4
    ax.bar([i - w / 2 for i in x], px, width=w, color=GREY, label="Entry price")
    ax.bar([i + w / 2 for i in x], iv, width=w, color=NAVY, label="Intrinsic (v3, live WACC)")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Rs / share")
    ax.set_title("Price vs intrinsic per holding (live provider WACC)", fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def fig_discount(holdings, out: Path) -> Path:
    plt = _mpl()
    # Use the at-entry discount (intrinsic / entry price - 1) so this chart reconciles exactly with the
    # price-vs-intrinsic chart; both are on the price the book actually paid.
    h = sorted(holdings, key=lambda r: (r.get("discount_at_entry") if r.get("discount_at_entry")
                                        is not None else r.get("discount_to_intrinsic") or 0))
    names = [r["ticker"] for r in h]
    disc = [((r.get("discount_at_entry") if r.get("discount_at_entry") is not None
              else r.get("discount_to_intrinsic")) or 0) * 100 for r in h]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.barh(range(len(h)), disc, color=[GREEN if d >= 0 else RED for d in disc])
    for i, d in enumerate(disc):
        ax.text(d, i, f" {d:+.1f}% ", va="center", ha="left" if d >= 0 else "right", fontsize=8, color="#333")
    ax.axvline(0, color="#444", linewidth=0.9)
    ax.set_yticks(range(len(h))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Discount to intrinsic at entry (%)  [negative = entry price above intrinsic]")
    ax.set_title("Discount to intrinsic per holding (at entry price)", fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def fig_sector_ic(out: Path) -> Path:
    plt = _mpl()
    rows = sorted(PHASE_B_SECTOR_IC, key=lambda r: r[1])
    names = [r[0] for r in rows]
    ic = [r[1] for r in rows]
    colors = [GREEN if r[0] in VALIDATED else (GREY if r[1] >= 0 else RED) for r in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.barh(range(len(rows)), ic, color=colors)
    for i, v in enumerate(ic):
        ax.text(v, i, f" {v:+.2f} ", va="center", ha="left" if v >= 0 else "right", fontsize=8, color="#333")
    ax.axvline(0, color="#444", linewidth=0.9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("1-year rank IC (discount -> forward return)")
    ax.set_title("India Phase B: where the discount predicts (green = book's validated sectors)",
                 fontsize=10.5, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def fig_risk_return(holdings, rs, out: Path) -> Path:
    """Per-name risk/return bubbles (x=vol, y=CAPM expected return, size=weight) with the portfolio
    and NIFTY plotted, plus the capital-market line from the risk-free rate through NIFTY."""
    plt = _mpl()
    a = rs["assumptions"]
    rf, e_rm = a["rf"], a["e_rm"]
    pf, pn = rs["portfolio"], rs["per_name"]
    bench_vol = rs.get("benchmark_vol", 0.15)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    # capital-market line: Rf through the NIFTY point
    xmax = max([pn[t]["vol"] for t in pn] + [pf["vol"], bench_vol]) * 1.15
    slope = (e_rm - rf) / bench_vol if bench_vol else 0.0
    ax.plot([0, xmax * 100], [rf * 100, (rf + slope * xmax) * 100], color=GREY, ls="--", lw=1.0,
            label="Capital-market line (Rf -> NIFTY)")
    for t, d in pn.items():
        er = rf + (d["beta"] or 1.0) * (e_rm - rf)         # CAPM expected return per name
        ax.scatter(d["vol"] * 100, er * 100, s=2400 * d["weight"], color=BLUE, alpha=0.55,
                   edgecolor=NAVY, zorder=3)
        ax.annotate(t, (d["vol"] * 100, er * 100), fontsize=8, ha="center", va="center", color=NAVY)
    ax.scatter(bench_vol * 100, e_rm * 100, marker="s", s=120, color=AMBER, zorder=4, label="NIFTY 50")
    ax.scatter(pf["vol"] * 100, pf["e_rp"] * 100, marker="*", s=420, color=GREEN, zorder=5,
               edgecolor="#14532d", label="Portfolio (incl. signal alpha)")
    ax.annotate(f"  E[R]={pf['e_rp']:.0%}\n  vol={pf['vol']:.0%}\n  Sharpe={pf['expected_sharpe']:.2f}",
                (pf["vol"] * 100, pf["e_rp"] * 100), fontsize=8, va="center", color="#14532d")
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Expected annual return (%)")
    ax.set_title("Risk / return (bubble = weight; ex-ante, ~1y horizon)",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def fig_nav_projection(capital, rs, out: Path) -> Path:
    """Illustrative forward NAV cone over the 1-year forecasting horizon (expected path +/- 1 sigma),
    with the beta-implied NIFTY path. This is an ex-ante projection, NOT a realised track record."""
    import numpy as np
    plt = _mpl()
    pf, a = rs["portfolio"], rs["assumptions"]
    e_rp, vol, e_rm = pf["e_rp"], pf["vol"], a["e_rm"]
    t = np.linspace(0, 1, 13)                               # months -> years
    exp_nav = capital * (1 + e_rp) ** t
    band = vol * np.sqrt(t) * exp_nav                       # 1-sigma absolute band
    nifty_nav = capital * (1 + e_rm) ** t
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.fill_between(t * 12, (exp_nav - band) / 1e5, (exp_nav + band) / 1e5, color=BLUE, alpha=0.15,
                    label="+/- 1 sigma band")
    ax.plot(t * 12, exp_nav / 1e5, color=NAVY, lw=2.2, label="Book expected NAV")
    ax.plot(t * 12, nifty_nav / 1e5, color=AMBER, ls="--", lw=1.4, label="NIFTY (beta-implied)")
    ax.axhline(capital / 1e5, color=GREY, lw=0.9, ls=":")
    ax.annotate(f"Rs {exp_nav[-1]:,.0f}\n(+{e_rp:.0%})", (12, exp_nav[-1] / 1e5), fontsize=8,
                color=NAVY, va="bottom", ha="right")
    ax.annotate(f"1 sigma: Rs {(exp_nav[-1]-band[-1]):,.0f} .. Rs {(exp_nav[-1]+band[-1]):,.0f}",
                (12, (exp_nav[-1] + band[-1]) / 1e5), fontsize=7.5, color="#555", va="bottom", ha="right")
    ax.set_xlabel("Months from inception")
    ax.set_ylabel("NAV (Rs lakh)")
    ax.set_title("Projected NAV over the 1-year edge horizon (ex-ante, illustrative; not a track record)",
                 fontsize=10.5, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


def fig_corr_heatmap(rs, out: Path) -> Path:
    """Holding-to-holding return correlation heatmap. Shows that even inside one sector the book holds
    distinguishable bets (the OMC refiners cluster; ONGC upstream and COALINDIA sit apart)."""
    import numpy as np
    plt = _mpl()
    cm = rs["corr_matrix"]
    tickers, M = cm["tickers"], np.array(cm["matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(tickers))); ax.set_xticklabels(tickers, fontsize=8, rotation=40, ha="right")
    ax.set_yticks(range(len(tickers))); ax.set_yticklabels(tickers, fontsize=8)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#111" if M[i, j] < 0.7 else "#fff")
    ax.set_title("Holding return correlations (3y weekly)", fontsize=10.5, color=NAVY, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out)


def _save_full(fig, out: Path) -> Path:
    """Save a full-bleed figure (dashboard cover) without tight-bbox cropping, so the 16:9 layout
    and the full-width banner are preserved. High DPI for a crisp, high-resolution cover."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


# Distinct per-company brand colours (the allocation pie is colour-coded by company, not by shade).
COMPANY_COLORS = {
    "IOC": "#E8531B", "BPCL": "#1B998B", "ONGC": "#C8102E", "COALINDIA": "#1F3864", "HINDPETRO": "#7B4FA3",
}
COMPANY_SHORT = {"IOC": "IOC", "BPCL": "BPCL", "ONGC": "ONGC", "COALINDIA": "COAL", "HINDPETRO": "HPCL"}


def _place_company_mark(ax, plt, ticker: str, weight: float, theta_deg: float, color: str,
                        base_r: float = 0.62) -> None:
    """Put a company mark on its pie slice, sized proportionally to the slice weight. Uses a real logo
    PNG from figures/logos/<TICKER>.png when present; otherwise draws a branded initials badge."""
    import numpy as np
    th = np.deg2rad(theta_deg)
    x, y = base_r * np.cos(th), base_r * np.sin(th)
    logo = FIGS / "logos" / f"{ticker}.png"
    if logo.exists():
        try:
            from matplotlib.offsetbox import AnnotationBbox, OffsetImage
            img = plt.imread(str(logo))
            # OffsetImage zoom maps image PIXELS to points, so normalise by the native height and
            # scale the target size with the slice weight (proportional representation).
            ih = float(img.shape[0]) or 1.0
            target_pts = 26.0 + 70.0 * weight                  # ~40pt mark for a ~20% slice
            oi = OffsetImage(img, zoom=target_pts / ih)
            ab = AnnotationBbox(oi, (x, y), frameon=False, pad=0.0, zorder=6)
            ax.add_artist(ab)
            return
        except Exception:  # noqa: BLE001
            pass
    rad = 0.17 + 0.55 * weight                                  # badge radius scales with weight
    ax.add_patch(plt.Circle((x, y), rad, color=color, zorder=5, ec="white", lw=1.2))
    ax.text(x, y, COMPANY_SHORT.get(ticker, ticker), color="white", fontsize=5.6 + 6.0 * weight,
            fontweight="bold", ha="center", va="center", zorder=6)


def fig_dashboard(state, rs, holdings, out: Path) -> Path:
    """A single-image, visually appealing dashboard used as the report cover (16:9): banner, a strip
    of KPI tiles, and three mini-charts (allocation donut, projected NAV, risk/return)."""
    import numpy as np
    import matplotlib.patches as mpatches
    plt = _mpl()
    pf, a = rs["portfolio"], rs["assumptions"]
    capital = state.get("capital") or 0.0
    e_rp, vol, e_rm, rf = pf["e_rp"], pf["vol"], a["e_rm"], a["rf"]

    fig = plt.figure(figsize=(13.33, 7.5)); fig.patch.set_facecolor("white")

    # banner
    ban = fig.add_axes([0, 0.865, 1, 0.135]); ban.axis("off")
    ban.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ban.transAxes, color=NAVY))
    ban.text(0.016, 0.60, "India v3 Aggressive Concentrated Book", transform=ban.transAxes,
             color="white", fontsize=20, fontweight="bold", va="center")
    ban.text(0.016, 0.22, "Engine_v3 discount signal  ·  single-sector PSU energy  ·  Rs 10,00,000  "
             "·  benchmark NIFTY 50", transform=ban.transAxes, color="#C7D2E5", fontsize=10.5, va="center")
    ban.add_patch(mpatches.FancyBboxPatch((0.85, 0.30), 0.13, 0.40, transform=ban.transAxes,
                  boxstyle="round,pad=0.008", linewidth=0, facecolor=AMBER))
    ban.text(0.915, 0.50, "PRELIMINARY", transform=ban.transAxes, color="white", fontsize=9.5,
             fontweight="bold", ha="center", va="center")

    # KPI tiles
    tiles = [
        ("Capital", f"Rs {capital / 1e5:,.1f}L"),
        ("Holdings", f"{len(holdings)}"),
        ("Exp. return 1y", f"{e_rp:.1%}"),
        ("Exp. Sharpe", f"{pf['expected_sharpe']:.2f}"),
        ("Exp. alpha", f"+{pf['expected_alpha']:.1%}"),
        ("Portfolio beta", f"{pf['beta']:.2f}"),
        ("Dividend yield", f"{pf['div_yield']:.1%}"),
        ("Effective bets", f"{pf['effective_bets']:.1f}"),
    ]
    axk = fig.add_axes([0.018, 0.625, 0.964, 0.200]); axk.axis("off")
    axk.set_xlim(0, len(tiles)); axk.set_ylim(0, 1)
    for i, (lab, val) in enumerate(tiles):
        hero = lab == "Exp. return 1y"
        if hero:
            # the expected-return tile is the hero: green panel, white text, up-arrow, larger.
            axk.add_patch(mpatches.FancyBboxPatch((i + 0.04, 0.04), 0.92, 0.92, boxstyle="round,pad=0.02",
                          linewidth=0, facecolor="#1E7A46", zorder=3))
            axk.add_patch(mpatches.FancyBboxPatch((i + 0.04, 0.04), 0.92, 0.92, boxstyle="round,pad=0.02",
                          fill=False, edgecolor="#14532D", linewidth=1.6, zorder=4))
            axk.text(i + 0.52, 0.60, f"▲ {val}", ha="center", va="center", fontsize=17.5,
                     fontweight="bold", color="white", zorder=5)
            axk.text(i + 0.52, 0.235, "EXPECTED RETURN 1Y", ha="center", va="center", fontsize=6.6,
                     color="#CDEBD7", fontweight="bold", zorder=5)
        else:
            axk.add_patch(mpatches.FancyBboxPatch((i + 0.06, 0.10), 0.88, 0.80, boxstyle="round,pad=0.02",
                          linewidth=0, facecolor="#EEF2F9"))
            axk.add_patch(mpatches.Rectangle((i + 0.06, 0.10), 0.04, 0.80, color=BLUE))
            axk.text(i + 0.52, 0.62, val, ha="center", va="center", fontsize=14.5, fontweight="bold", color=NAVY)
            axk.text(i + 0.52, 0.26, lab.upper(), ha="center", va="center", fontsize=7.0, color="#5A6473")

    # allocation pie: colour-coded per company, each slice carrying that company's mark (logo or
    # branded badge) sized in proportion to its weight, with the percentage labelled outside.
    axd = fig.add_axes([0.030, 0.085, 0.275, 0.43])
    vals = [h["market_value"] or 0 for h in holdings]
    names = [h["ticker"] for h in holdings]
    tot = sum(vals) or 1.0
    colors = [COMPANY_COLORS.get(n, BLUE) for n in names]
    wedges, _ = axd.pie(vals, labels=[f"{n}  {v / tot:.0%}" for n, v in zip(names, vals)],
                        colors=colors, startangle=90, labeldistance=1.16,
                        wedgeprops=dict(edgecolor="white", linewidth=1.5),
                        textprops=dict(fontsize=6.6, color="#333", fontweight="bold"))
    axd.set_xlim(-1.3, 1.3); axd.set_ylim(-1.3, 1.3); axd.set_autoscale_on(False)  # lock vs marks
    for wge, n, v in zip(wedges, names, vals):
        _place_company_mark(axd, plt, n, v / tot, (wge.theta1 + wge.theta2) / 2.0,
                            COMPANY_COLORS.get(n, BLUE))
    axd.set_title("Allocation by company", fontsize=9.5, color=NAVY, fontweight="bold")

    # projected NAV mini
    axn = fig.add_axes([0.375, 0.115, 0.255, 0.40])
    t = np.linspace(0, 1, 13)
    exp_nav = capital * (1 + e_rp) ** t
    band = vol * np.sqrt(t) * exp_nav
    axn.fill_between(t * 12, (exp_nav - band) / 1e5, (exp_nav + band) / 1e5, color=BLUE, alpha=0.15)
    axn.plot(t * 12, exp_nav / 1e5, color=NAVY, lw=2.0)
    axn.plot(t * 12, capital * (1 + e_rm) ** t / 1e5, color=AMBER, ls="--", lw=1.2)
    axn.set_title("Projected NAV (1y, +/- 1 sigma)", fontsize=9.5, color=NAVY, fontweight="bold")
    axn.set_xlabel("months", fontsize=7); axn.set_ylabel("Rs lakh", fontsize=7)
    axn.tick_params(labelsize=6.5); axn.spines[["top", "right"]].set_visible(False)

    # risk/return mini
    axr = fig.add_axes([0.705, 0.115, 0.265, 0.40])
    for _t, d in rs["per_name"].items():
        er = rf + (d["beta"] or 1.0) * (e_rm - rf)
        axr.scatter(d["vol"] * 100, er * 100, s=520 * d["weight"], color=BLUE, alpha=0.55, edgecolor=NAVY)
    axr.scatter(rs.get("benchmark_vol", 0.15) * 100, e_rm * 100, marker="s", s=55, color=AMBER)
    axr.scatter(pf["vol"] * 100, e_rp * 100, marker="*", s=240, color=GREEN, edgecolor="#14532d")
    axr.set_title("Risk / return (* = portfolio)", fontsize=9.5, color=NAVY, fontweight="bold")
    axr.set_xlabel("volatility %", fontsize=7); axr.set_ylabel("E[R] %", fontsize=7)
    axr.tick_params(labelsize=6.5); axr.spines[["top", "right"]].set_visible(False)

    fig.text(0.018, 0.018, f"Inception {state.get('inception', '')}  ·  currency INR  ·  signal horizon "
             "~1 year  ·  status PRELIMINARY (directional, not validated alpha)", fontsize=7.5, color="#5A6473")
    fig.text(0.982, 0.018, "engine_v3 Fundamentals-Fade", fontsize=7.5, color="#9AA3B2", ha="right")
    return _save_full(fig, out)


def fig_expert(holdings, rs, out: Path) -> Path:
    """Street consensus mean target vs our v3 intrinsic vs entry price, per holding."""
    plt = _mpl()
    an = rs.get("analyst", {})
    names = [h["ticker"] for h in holdings]
    entry = [h["entry_price"] for h in holdings]
    target = [((an.get(h["ticker"]) or {}).get("mean_target")) for h in holdings]
    intr = [h["intrinsic"] for h in holdings]
    x = list(range(len(names)))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar([i - w for i in x], entry, width=w, color=GREY, label="Entry price")
    ax.bar(x, [t or 0 for t in target], width=w, color=GREEN, label="Street mean target")
    ax.bar([i + w for i in x], intr, width=w, color=NAVY, label="Our v3 intrinsic")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Rs / share")
    ax.set_title("Expert opinion vs our model: street target, entry, intrinsic",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out)


_RECO_LABEL = {"strong_buy": "Strong Buy", "buy": "Buy", "outperform": "Outperform", "hold": "Hold",
               "underperform": "Underperform", "sell": "Sell", "strong_sell": "Strong Sell"}


# ---------------------------------------------------------------------------
# Risk-stats loader (computed by india_v3_riskstats.py; documented fallback keeps the
# export deterministic / offline if the JSON is not present).
# ---------------------------------------------------------------------------
_RISK_FALLBACK = {
    "window": "fallback constants (run india_v3_riskstats.py for live values)",
    "benchmark_vol": 0.15,
    "assumptions": {"rf": 0.068, "erp_market": 0.06, "e_rm": 0.128, "energy_1y_ic": 0.476,
                    "overall_1y_ls": 0.289, "gross_long_leg_alpha": 0.1445, "survivorship_haircut": 0.60},
    "per_name": {
        "IOC": {"weight": 0.204, "vol": 0.309, "beta": 1.41, "corr_nifty": 0.57, "div_yield": 0.071},
        "BPCL": {"weight": 0.204, "vol": 0.308, "beta": 1.33, "corr_nifty": 0.54, "div_yield": 0.074},
        "ONGC": {"weight": 0.204, "vol": 0.260, "beta": 0.83, "corr_nifty": 0.40, "div_yield": 0.055},
        "COALINDIA": {"weight": 0.204, "vol": 0.272, "beta": 0.73, "corr_nifty": 0.33, "div_yield": 0.060},
        "HINDPETRO": {"weight": 0.185, "vol": 0.383, "beta": 1.66, "corr_nifty": 0.54, "div_yield": 0.040},
    },
    "corr_matrix": {"tickers": ["IOC", "BPCL", "ONGC", "COALINDIA", "HINDPETRO"],
                    "matrix": [[1.0, 0.86, 0.42, 0.45, 0.78], [0.86, 1.0, 0.36, 0.41, 0.84],
                               [0.42, 0.36, 1.0, 0.47, 0.35], [0.45, 0.41, 0.47, 1.0, 0.35],
                               [0.78, 0.84, 0.35, 0.35, 1.0]]},
    "analyst": {
        "IOC": {"mean_target": 158.0, "high_target": 203.0, "low_target": 110.0, "n_analysts": 31,
                "reco_key": "hold", "reco_mean": 2.55},
        "BPCL": {"mean_target": 332.16, "high_target": 518.0, "low_target": 233.0, "n_analysts": 31,
                 "reco_key": "hold", "reco_mean": 2.58},
        "ONGC": {"mean_target": 308.1, "high_target": 405.0, "low_target": 230.0, "n_analysts": 30,
                 "reco_key": "buy", "reco_mean": 2.13},
        "COALINDIA": {"mean_target": 459.27, "high_target": 550.0, "low_target": 290.0, "n_analysts": 26,
                      "reco_key": "buy", "reco_mean": 2.50},
        "HINDPETRO": {"mean_target": 398.19, "high_target": 679.0, "low_target": 275.0, "n_analysts": 31,
                      "reco_key": "hold", "reco_mean": 2.87},
    },
    "portfolio": {"beta": 1.18, "vol": 0.244, "weighted_avg_vol": 0.305, "diversification_ratio": 1.25,
                  "effective_bets": 1.57, "avg_pairwise_corr": 0.53, "hhi": 0.20, "effective_n_names": 4.99,
                  "tracking_error": 0.195, "div_yield": 0.060, "realized_alpha_3y": 0.198,
                  "max_drawdown_3y": -0.301, "e_rp": 0.197, "expected_alpha": 0.058,
                  "expected_sharpe": 0.528, "information_ratio": 0.296},
}


def load_risk_stats() -> dict:
    p = FOLDER / "risk_stats.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return _RISK_FALLBACK


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def build_model(state: dict):
    holdings = state["holdings"]
    deployed = state.get("deployed") or sum(h["market_value"] or 0 for h in holdings)
    capital = state.get("capital") or 0.0
    wsum = sum(h["weight"] or 0 for h in holdings) or 1.0
    wtd_disc = sum((h["weight"] or 0) * (h["discount_to_intrinsic"] or 0) for h in holdings) / wsum

    rs = load_risk_stats()
    pf = rs["portfolio"]

    # At-entry discount (intrinsic / ENTRY price - 1): the basis the book actually transacted on. The
    # state.json `discount_to_intrinsic` was computed off the valuation-snapshot price, which differs
    # from the fill by up to ~2.5pp; we recompute on a single consistent basis for the charts/tables.
    for h in holdings:
        ep, iv = h.get("entry_price"), h.get("intrinsic")
        h["discount_at_entry"] = (iv / ep - 1.0) if (ep and iv) else None
    wtd_disc_entry = sum((h["weight"] or 0) * (h["discount_at_entry"] or 0) for h in holdings) / wsum

    FIGS.mkdir(parents=True, exist_ok=True)
    f_alloc = fig_allocation(holdings, FIGS / "alloc_india.png")
    f_pxiv = fig_price_vs_intrinsic(holdings, FIGS / "price_vs_intrinsic.png")
    f_disc = fig_discount(holdings, FIGS / "discount_india.png")
    f_conc = viz3d.concentration([h["weight"] or 0 for h in holdings], FIGS / "conc_india.png",
                                 title="Concentration (weight by holding)")
    f_ic = fig_sector_ic(FIGS / "sector_ic_india.png")
    f_rr = fig_risk_return(holdings, rs, FIGS / "risk_return_india.png")
    f_nav = fig_nav_projection(capital, rs, FIGS / "nav_projection_india.png")
    f_corr = fig_corr_heatmap(rs, FIGS / "corr_india.png")
    f_dash = fig_dashboard(state, rs, holdings, FIGS / "dashboard_india.png")
    f_expert = fig_expert(holdings, rs, FIGS / "expert_india.png")

    model = new_report(
        title="India v3 Aggressive Concentrated Book",
        subtitle="Engine_v3 discount signal (India Phase B) | single-sector PSU energy bet | Rs 10,00,000",
        persona="pm",
        Book=BOOK_NAME,
        Country="India (NSE)",
        Currency="INR (Rs)",
        Benchmark=f"NIFTY 50 (^NSEI @ {state.get('benchmark_entry', 0):,.2f})",
        Inception=state.get("inception", ""),
        Engine="v3 (Fundamentals-Fade)",
        Signal=state.get("signal", ""),
        Status="PRELIMINARY / not validated alpha",
    )
    model.meta["As of"] = state.get("inception", "")
    model.cover_image = f_dash   # visually appealing dashboard replaces the default meta cover

    # ---- 1. Executive summary ----
    s = Section(name="Summary", title="Executive Summary")
    s.metrics = [
        Metric("Capital", capital, "inr0"),
        Metric("Deployed", deployed, "inr0", note="Cash residual held in INR"),
        Metric("Holdings", len(holdings), "num0"),
        Metric("Effective bets", pf["effective_bets"], "num",
               note=f"5 names but ~{pf['effective_bets']:.1f} independent bets (single sector)", good=False),
        Metric("Expected return (1y)", pf["e_rp"], "pct1", note="Ex-ante CAPM + signal alpha"),
        Metric("Expected Sharpe (1y)", pf["expected_sharpe"], "num"),
        Metric("Expected alpha vs NIFTY", pf["expected_alpha"], "pct1", note="After survivorship haircut"),
        Metric("Portfolio beta", pf["beta"], "num", note="High-beta cyclical book"),
        Metric("Dividend yield", pf["div_yield"], "pct1", note="Trailing 12m; cash regardless of price"),
        Metric("Wtd discount at entry", wtd_disc_entry, "pct1",
               note="Negative => entry price above intrinsic (see Discount & Verdict Audit)"),
        Metric("Signal horizon", "~1 year", "text"),
        Metric("Validation", "Preliminary", "text", good=False),
    ]
    s.narrative = [
        "A high-conviction, deliberately concentrated India book sized for upside on a Rs 10,00,000 "
        "real-money allocation, picked purely on the engine_v3 discount signal and restricted to the "
        "sectors where that signal is shown to forecast returns. It is standalone and does not touch "
        "the US conf_top10 registry.",
        "This is a one-year position in the Indian state-owned oil, gas and coal complex. Over a two-week "
        "window its path is crude, rupee and policy noise, not the model working; the expected short-horizon "
        "return is about the NIFTY drift, with a large down move as likely as a large up move.",
    ]
    s.prose = True
    model.sections.append(s)

    # ---- 1b. Diversification & portfolio risk (placed at the top, as requested) ----
    a = rs["assumptions"]
    s = Section(name="Diversification", title="Portfolio Diversification & Risk")
    s.metrics = [
        Metric("Holdings", len(holdings), "num0"),
        Metric("Effective bets", pf["effective_bets"], "num",
               note="Diversification ratio squared; independent bets, not name count", good=False),
        Metric("Diversification ratio", pf["diversification_ratio"], "num",
               note="(Sum w.sigma) / portfolio sigma; 1.0 = no diversification"),
        Metric("Avg pairwise correlation", pf["avg_pairwise_corr"], "num"),
        Metric("Weight HHI", pf["hhi"], "num", note=f"Effective N (HHI) = {pf['effective_n_names']:.1f}"),
        Metric("Sector concentration", "Energy 100%", "text", good=False),
        Metric("Portfolio volatility", pf["vol"], "pct1", note="Annualised, 3y weekly"),
        Metric("Portfolio beta", pf["beta"], "num"),
        Metric("Tracking error vs NIFTY", pf["tracking_error"], "pct1"),
    ]
    s.tables = [Table(
        title="Formulas used in the risk and diversification calculations (standard forms)",
        columns=["Measure", "Formula (plain-English variables)", "What it means"],
        rows=[
            ["Beta", "Beta = Covariance(Holding Return, Market Return) / Variance(Market Return)",
             "A holding's sensitivity to the market."],
            ["Portfolio volatility",
             "Portfolio Volatility = Square Root(Weights transposed x Covariance Matrix x Weights)",
             "Overall return variability, after correlations."],
            ["Diversification ratio",
             "Diversification Ratio = (Sum of Weight x Volatility) / Portfolio Volatility",
             "How much correlation lowers risk vs the names held alone; 1.0 means none."],
            ["Effective bets", "Effective Bets = Diversification Ratio squared",
             "Independent risk bets, not the count of names."],
            ["Concentration (HHI)", "Herfindahl Index = Sum of (Weight squared)",
             "Weight concentration, from 0 (spread out) to 1 (one name)."],
            ["Effective names", "Effective Names = 1 / Herfindahl Index",
             "Equal-weight equivalent number of positions."],
        ],
        formats=["text", "text", "text"],
        col_widths=[18, 66, 40],
    ), Table(
        title="Per-name risk (3y weekly, total-return basis)",
        columns=["Ticker", "Weight", "Volatility", "Beta", "Corr to NIFTY", "Div yield"],
        rows=[[t, d["weight"], d["vol"], d["beta"], d["corr_nifty"], d["div_yield"]]
              for t, d in rs["per_name"].items()],
        formats=["text", "pct1", "pct1", "num", "num", "pct1"],
    )]
    s.images = [Image("Holding return correlations", f_corr, 560)]
    s.narrative = [
        "Diversification is reported even though every name sits in one sector, because the honest answer "
        "is two-sided. By weight the book is well spread: the weight HHI is about 0.20, an effective five "
        "names, since the positions are deliberately near equal. By risk it is not five bets: the "
        "diversification ratio is about " + f"{pf['diversification_ratio']:.2f}" + ", which squares to roughly "
        + f"{pf['effective_bets']:.1f}" + " independent bets, because the holdings share a common crude, rupee and "
        "PSU-policy driver (average pairwise correlation about " + f"{pf['avg_pairwise_corr']:.2f}" + ").",
        "There is real structure inside the sector, however. The oil-marketing refiners (IOC, BPCL, "
        "HINDPETRO) move together at roughly 0.8 correlation because they share the same refining and "
        "marketing-margin driver, while ONGC (upstream crude realisations) and COALINDIA (coal and power "
        "demand) correlate only about 0.35 to 0.46 with that cluster. So the book carries roughly two to "
        "three distinguishable risk engines rather than one, which is why the effective-bets number is "
        "above one rather than at one.",
        "What diversification cannot do here is remove the single-sector tail. The sector HHI is 1.0, so a "
        "crude spike, a windfall-tax or fuel-pricing shock, or a broad high-beta selloff hits all five names "
        "at once. The realised three-year maximum drawdown of this exact weighted book was about "
        + f"{pf['max_drawdown_3y']:.0%}" + ", which is the kind of single-sector drawdown to size for.",
    ]
    s.prose = True
    model.sections.append(s)

    # ---- 1c. Expected performance & CAPM ----
    beta, e_rm, rf = pf["beta"], a["e_rm"], a["rf"]
    s = Section(name="Expected", title="Expected Performance & CAPM (1-year horizon)")
    s.metrics = [
        Metric("Expected return E[Rp]", pf["e_rp"], "pct1"),
        Metric("Expected alpha vs NIFTY", pf["expected_alpha"], "pct1", note="CAPM alpha, post-haircut"),
        Metric("Expected Sharpe", pf["expected_sharpe"], "num", note="(E[Rp] - Rf) / vol"),
        Metric("Information ratio", pf["information_ratio"], "num", note="Alpha / tracking error"),
        Metric("Tracking error", pf["tracking_error"], "pct1"),
        Metric("Beta", beta, "num"),
        Metric("E[NIFTY]", e_rm, "pct1", note="Rf + market premium"),
        Metric("Realised 3y alpha", pf["realized_alpha_3y"], "pct1",
               note="Backward-looking context only, not the forecast"),
    ]
    s.tables = [Table(
        title="Formulas used in the return calculations (standard forms)",
        columns=["Measure", "Formula (plain-English variables)", "What it means"],
        rows=[
            ["CAPM expected return",
             "Expected Return = Risk-Free Rate + Beta x (Expected Market Return - Risk-Free Rate)",
             "The return required for the market risk taken, before any skill."],
            ["Alpha (Jensen)",
             "Alpha = Portfolio Return - [Risk-Free Rate + Beta x (Market Return - Risk-Free Rate)]",
             "Return earned above the CAPM baseline; the edge."],
            ["Sharpe ratio",
             "Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility",
             "Excess return per unit of total risk."],
            ["Tracking error",
             "Tracking Error = Standard Deviation of (Portfolio Return - Benchmark Return)",
             "How far the book drifts from its benchmark."],
            ["Information ratio", "Information Ratio = Alpha / Tracking Error",
             "Active return per unit of active risk."],
            ["Expected alpha (our estimate)",
             "Expected Alpha = (One-Year Long-Short Spread / 2) x (1 - Survivorship Haircut)",
             "Turns the backtest spread into a conservative forward alpha."],
        ],
        formats=["text", "text", "text"],
        col_widths=[20, 64, 40],
    ), Table(
        title="CAPM decomposition of the 1-year expected return",
        columns=["Component", "Annual %", "Note"],
        rows=[
            ["Risk-free rate (India 10Y G-Sec)", rf, "live macro anchor"],
            # residual so the three components foot exactly to E[Rp] (beta carries unrounded precision)
            ["+ Beta x equity market premium", pf["e_rp"] - rf - pf["expected_alpha"],
             f"beta {beta:.2f} x premium {(e_rm - rf):.1%}"],
            ["+ Expected alpha (signal, post-haircut)", pf["expected_alpha"],
             f"L/S {a['overall_1y_ls']:.2f} / 2 x (1 - {a['survivorship_haircut']:.0%} haircut)"],
            ["= Expected total return E[Rp]", pf["e_rp"], "ex-ante, ~1 year"],
            ["Memo: trailing dividend yield", pf["div_yield"], "delivered as cash regardless of price path"],
        ],
        formats=["text", "pct2", "text"],
    )]
    s.images = [
        Image("Risk / return (per name, portfolio, NIFTY)", f_rr, 800),
        Image("Projected NAV over the 1-year edge horizon", f_nav, 820),
    ]
    s.narrative = [
        "The expected one-year return is built bottom-up from the capital asset pricing model and then "
        "credited with the signal's expected alpha. Start at the risk-free rate (the India ten-year G-Sec, "
        "about 6.8%), add the book's market exposure (a beta of about " + f"{beta:.2f}" + " times an equity "
        "market premium of about " + f"{(e_rm - rf):.1%}" + "), and add the cross-sectional edge the Phase B "
        "evidence supports. That edge is taken as roughly half the one-year quintile long-short spread (the "
        "long leg), then cut by a 60% survivorship haircut because Phase B is survivorship-optimistic and its "
        "t-statistics are inflated, which lands expected alpha near " + f"{pf['expected_alpha']:.1%}" + " and an "
        "expected total return near " + f"{pf['e_rp']:.1%}" + ".",
        "The expected Sharpe of about " + f"{pf['expected_sharpe']:.2f}" + " and information ratio of about "
        + f"{pf['information_ratio']:.2f}" + " are modest by design: a single-sector, high-beta book carries a "
        "lot of volatility (about " + f"{pf['vol']:.0%}" + ") and tracking error (about "
        + f"{pf['tracking_error']:.0%}" + ") per unit of expected edge. The realised three-year alpha of about "
        + f"{pf['realized_alpha_3y']:.0%}" + " is shown only as backward-looking context; it reflects the 2023 "
        "to 2025 PSU re-rating and is not repeatable, so it is deliberately excluded from the forecast.",
        "Read these as ex-ante model estimates with wide error bars, not promises. Because the signal is "
        "preliminary, the realised one-year alpha could plausibly be zero or negative; the dividend yield of "
        "about " + f"{pf['div_yield']:.0%}" + " is the one component that arrives as cash largely independent of "
        "the price path.",
    ]
    s.prose = True
    model.sections.append(s)

    # ---- 2. Investment philosophy (the essay) ----
    essay = (
        "This book expresses one disciplined idea: in India the engine_v3 fundamentals-fade discount is "
        "not a universal cheapness gauge but a narrow, sector-conditional ranking signal, so the only "
        "honest way to use it is to deploy capital exactly where the evidence says it forecasts returns. "
        "The Phase B point-in-time backtest of 105 names and roughly 1,004 observations shows a positive "
        "one-year information coefficient of about +0.22 overall, but the cross-section is the real story: "
        "the discount ranks forward returns strongly in cyclical, capital-intensive sectors (energy +0.48, "
        "materials +0.49, utilities +0.46, industrials +0.36) and is flat to anti-predictive in technology, "
        "staples and health care, which is the mirror image of the United States where staples was the one "
        "reliable family. An Indian value signal therefore earns its keep in the parts of the market whose "
        "cash flows actually revert, not in the high-multiple compounders the market already prices for "
        "perfection. The model's one coherent value call follows the same logic, because the state-owned "
        "oil, gas and coal complex is the only sector trading below a conservative DCF and is also among the "
        "highest in predictive power, so the signal finds value precisely where it has demonstrated "
        "forecasting ability. We concentrate the entire book there, conviction-weighting by relative "
        "selection rank rather than by absolute discount, since what survives both the internal and the "
        "provider WACC is the ordinal ranking and not the magnitude; we hold it as a roughly one-year "
        "position, we treat the result as preliminary and survivorship-optimistic rather than validated "
        "alpha, and we size it as speculative, losable capital whose two-week path will be set by crude, "
        "the rupee and policy rather than by the model working."
    )
    s = Section(name="Philosophy", title="Investment Philosophy")
    s.narrative = [essay]
    s.prose = True
    # The sector-IC chart lives once, in the Validation section (was duplicated here).
    model.sections.append(s)

    # ---- 3. Holdings ----
    s = Section(name="Holdings", title="Holdings")
    rows = []
    for h in holdings:
        rows.append([
            h["ticker"], SECTOR_DESC.get(h["ticker"], h.get("sector", "")), h["weight"], h["shares"],
            h["entry_price"], h["market_value"], h["intrinsic"], h.get("discount_at_entry"),
            h["selection_rank"], h["verdict"],
        ])
    rows.append(["Total", "", wsum, "", "", deployed, "", wtd_disc_entry, "", ""])
    s.tables = [Table(
        title="Concentrated top-5 (entry prices 2026-06-12 close; discount at entry)",
        columns=["Ticker", "Sector", "Weight", "Shares", "Entry", "Mkt value", "Intrinsic",
                 "Disc @ entry", "Sel. rank", "Verdict"],
        rows=rows,
        formats=["text", "text", "pct1", "num0", "inr", "inr0", "inr", "pct1", "num0", "text"],
        heatmap_cols=[7],
        total_row=True,
    )]
    s.images = [
        Image("Allocation by holding", f_alloc, 720),
        Image("Concentration / HHI", f_conc, 720),
    ]
    s.narrative = [
        "The four rank-88 names take about 20.4% each; HINDPETRO (rank 80) takes about 18.5%. Equal-ish by "
        "design: the allocation study found equal weight hard to beat and vol-weighting a loser.",
        "HCLTECH, TCS and INFY were deliberately excluded despite high cheapness ranks, because Phase B "
        "shows technology has a 1-year IC near zero; a high rank there is cheap but the signal does not work.",
    ]
    model.sections.append(s)

    # ---- 4. Valuation ----
    s = Section(name="Valuation", title="Valuation Detail")
    s.images = [
        Image("Entry price vs v3 intrinsic per holding", f_pxiv, 800),
        Image("Discount to intrinsic per holding", f_disc, 720),
    ]
    s.narrative = [
        "Two WACCs disagree on the sign of the value call. The Phase B reference (internal CAPM, PIT-safe) "
        "sees these PSUs as roughly 20% to 35% undervalued; the live provider WACC (about 1 to 1.5 points "
        "higher) sees them 18% to 37% overvalued, which is why the verdicts read Neutral to Unattractive. "
        "What survives both bases is only the relative ranking: energy is the cheapest sector either way.",
        "The bull case therefore rests on the relative rank plus the energy one-year IC plus PSU catalysts "
        "(high dividend yields, buybacks, crude and marketing-margin tailwinds). On absolute value under the "
        "live WACC there is no margin-of-safety cushion.",
    ]
    s.prose = True
    model.sections.append(s)

    # ---- 4b. Discount & verdict audit (is it a bug or optics?) ----
    s = Section(name="Audit", title="Discount & Verdict Audit")
    s.tables = [Table(
        title="Reconciliation: entry vs intrinsic, both discount bases, rank vs verdict",
        columns=["Ticker", "Entry (Rs)", "Intrinsic (Rs)", "Disc @ entry", "Disc @ snapshot",
                 "Sel. rank", "Verdict"],
        rows=[[h["ticker"], h["entry_price"], h["intrinsic"], h.get("discount_at_entry"),
               h.get("discount_to_intrinsic"), h.get("selection_rank"), h.get("verdict")]
              for h in holdings],
        formats=["text", "inr", "inr", "pct1", "pct1", "num0", "text"],
    )]
    s.narrative = [
        "Is the discount or verdict bugged? No. The discount is intrinsic divided by price minus one, "
        "verified to the rupee; a negative number means the price sits above that name's own DCF intrinsic. "
        "The only defect found is a basis mismatch: the stored discount was computed off the valuation "
        "snapshot price, while the fill happened at a separately downloaded entry price, so the two differ "
        "by up to about 2.5 points (IOC and BPCL). This pack recomputes every discount on the single "
        "at-entry basis, which is the column to trust; the snapshot column is shown only for the audit.",
        "The Neutral and Unattractive verdicts are intentional, not a bug. The India scorer applies an "
        "absolute-value cap: a name trading above its own intrinsic cannot read Attractive no matter how "
        "cheap it is relative to an all-expensive market. A discount between minus 10% and minus 30% caps "
        "the label at Neutral, and below minus 30% it caps at Unattractive, which is exactly why IOC, BPCL, "
        "ONGC and COALINDIA read Neutral and HINDPETRO reads Unattractive. The selection rank of 88 and 80 "
        "is the uncapped relative cheapness percentile, so both numbers are true at once: these are the "
        "cheapest tail of the India cross-section, and they still trade above their own intrinsic.",
        "Why is every name above intrinsic? Two reasons, both structural. First, the two WACCs disagree on "
        "the sign of the value call: the Phase B reference (internal CAPM) sees these PSUs as 20% to 35% "
        "undervalued, while the live provider WACC, about 1 to 1.5 points higher, sees them 18% to 37% "
        "overvalued. Same cash flows, a higher discount rate, opposite sign. Second, India's high nominal "
        "discount rate (a roughly 12% to 16% WACC) makes almost the entire market look expensive against a "
        "DCF, so energy is simply the least expensive tail of a structurally expensive-looking market. The "
        "live pipeline also ranks a provider-WACC discount against an internal-WACC reference, an "
        "inconsistency that is the documented Phase C gating item rather than a code defect.",
        "So if every entry price is above intrinsic, how does the book make money? Not from absolute "
        "mean-reversion to the printed intrinsic; on the live-WACC basis there is no margin of safety. The "
        "edge is a cross-sectional rank claim, not an absolute one. The Phase B evidence is an information "
        "coefficient, a rank correlation between the discount and the forward return; an energy IC of about "
        "+0.48 means the cheapest-ranked names systematically outperform the most expensive-ranked names, "
        "not that the cheap names must rise in absolute terms. A long-only book of the top-ranked names "
        "harvests that relative spread as alpha, which is why the expected return decomposes as the "
        "risk-free rate plus beta times the market premium plus that alpha. The book can therefore beat "
        "NIFTY even in a falling market (the cheapest names fall least), and it makes money in absolute "
        "terms whenever the market return plus the alpha is positive.",
        "Two further return sources the live-WACC DCF deliberately understates make the picture less "
        "paradoxical. If the internal-CAPM WACC is the correct basis, these names are genuinely 20% to 35% "
        "cheap with real absolute upside. And independently of the valuation debate, the book yields about "
        "6% in trailing dividends, with PSU buybacks and cyclical crude and marketing-margin upswings on "
        "top, which deliver cash and re-rating that a conservative DCF does not capture. In short the "
        "negative headline discount is mostly an optics artifact of the conservative provider-WACC basis, "
        "not a calculation error, and the return mechanism is alpha plus dividends plus the more favourable "
        "internal-WACC value case rather than reversion to the printed intrinsic.",
    ]
    s.prose = True
    model.sections.append(s)

    # ---- 4c. Expert opinion / street consensus comparison ----
    an = rs.get("analyst", {})
    exp_rows, ups, n_an_list, nbuy = [], [], [], 0
    for h in holdings:
        info = an.get(h["ticker"]) or {}
        tgt = info.get("mean_target")
        up = (tgt / h["entry_price"] - 1.0) if (tgt and h["entry_price"]) else None
        if up is not None:
            ups.append((h["weight"] or 0, up))
        reco = _RECO_LABEL.get(info.get("reco_key"),
                               (info.get("reco_key") or "NA").replace("_", " ").title())
        if info.get("reco_key") in ("buy", "strong_buy", "outperform"):
            nbuy += 1
        if info.get("n_analysts"):
            n_an_list.append(info["n_analysts"])
        exp_rows.append([h["ticker"], h["entry_price"], tgt, up, info.get("n_analysts"),
                         reco, h["intrinsic"], h["verdict"]])
    wsum_up = sum(w for w, _ in ups) or 1.0
    wtd_upside = sum(w * u for w, u in ups) / wsum_up
    avg_an = (sum(n_an_list) / len(n_an_list)) if n_an_list else None

    s = Section(name="Experts", title="Expert Opinion: Street Consensus vs Our Model")
    s.metrics = [
        Metric("Street weighted upside", wtd_upside, "pct1", note="Mean target vs entry, weighted"),
        Metric("Names rated Buy", f"{nbuy} of {len(holdings)}", "text"),
        Metric("Avg analysts / name", avg_an, "num0"),
        Metric("Our wtd discount at entry", wtd_disc_entry, "pct1",
               note="Our DCF: entry above intrinsic", good=False),
    ]
    s.tables = [Table(
        title="Per-name: street mean target vs our v3 intrinsic (source: Yahoo Finance consensus)",
        columns=["Ticker", "Entry", "Street target", "Upside to target", "# analysts",
                 "Street rating", "Our intrinsic", "Our verdict"],
        rows=exp_rows,
        formats=["text", "inr", "inr", "pct1", "num0", "text", "inr", "text"],
        heatmap_cols=[3],
    )]
    s.images = [Image("Street target vs entry vs our intrinsic", f_expert, 800)]
    s.narrative = [
        "The street and our model disagree in a revealing way, and the comparison is the point. Sell-side "
        "consensus (26 to 31 analysts per name) sets twelve-month price targets that sit above the current "
        "price, a weighted upside of about " + f"{wtd_upside:.0%}" + ", and rates ONGC and COALINDIA outright "
        "Buy with the oil-marketing names at Hold. Our conservative DCF does the opposite: it places "
        "intrinsic value below the current price, so the verdicts read Neutral to Unattractive.",
        "Both can be right because they answer different questions. A street target is a forecast of where "
        "the price will trade in a year, driven by earnings momentum, dividend and buyback re-rating, and "
        "value relative to the stock's own history. Our intrinsic is an estimate of absolute fair value "
        "discounted at a high India WACC of roughly 12% to 16%. The street is markedly more bullish than our "
        "intrinsic, with mean targets roughly 40% to 65% above our DCF value, which is the same gap the audit "
        "describes: the market is paying for PSU dividends, buybacks and a cyclical re-rating that a "
        "conservative cash-flow model deliberately does not capitalise.",
        "Read this as one external cross-check, not as validation. The targets are twelve-month consensus, "
        "not guarantees, and the high-to-low target bands are wide (HINDPETRO alone spans about Rs 275 to Rs "
        "679), which signals low agreement. The useful takeaway is directional: the street agrees the energy "
        "PSUs are at worst fairly valued with positive expected price action, which is consistent with this "
        "book being a defensible relative-value and income position rather than a deep-value bargain.",
    ]
    s.prose = True
    model.sections.append(s)

    # ---- 5. Key assumption differences ----
    s = Section(name="Assumptions", title="Key Assumption Differences: US vs India")
    s.tables = [Table(
        title="What changes when the engine is pointed at India",
        columns=["Assumption", "United States", "India"],
        rows=[list(r) for r in ASSUMPTION_DIFFS],
        formats=["text", "text", "text"],
    )]
    s.narrative = [
        "The India book runs the identical engine_v3 machinery as the US book; only the data layer and a "
        "small set of country anchors change. The biggest economic difference is the discount-rate and "
        "terminal-growth environment: India carries a higher risk-free rate, a higher equity risk premium "
        "and a higher tax rate, while the terminal-growth band is tied live to India macro with a floor at "
        "CPI inflation and a cap at the 10-year G-Sec yield. The other structural difference is behavioural "
        "rather than mechanical: the sectors where the discount has forecasting power are the cyclical, "
        "capital-intensive families in India, the opposite of the consumer-staples reliability seen in the "
        "United States, which is exactly why this book is built on PSU energy and not on the high-quality "
        "compounders that dominate the US selections.",
    ]
    s.prose = True
    model.sections.append(s)

    # ---- 6. Validation evidence ----
    s = Section(name="Validation", title="Validation Evidence: India Phase B")
    s.tables = [
        Table(title="Overall rank IC (discount -> forward return)",
              columns=["Horizon", "n", "IC (obs-wt)", "Quintile L/S"],
              rows=[list(r) for r in PHASE_B_OVERALL],
              formats=["text", "num0", "num", "num"]),
        Table(title="Per-sector 1-year IC (the accuracy map)",
              columns=["Sector", "1y IC (obs-wt)", "n", "Read"],
              rows=[list(r) for r in PHASE_B_SECTOR_IC],
              formats=["text", "num", "num0", "text"],
              heatmap_cols=[1]),
    ]
    s.images = [Image("1-year IC by sector", f_ic, 780)]
    s.narrative = [
        "Thin and overlapping window: India structured XBRL begins around 2018 and v3 needs about 20 "
        "quarters, so usable cutoffs only start around 2023, the one-year windows overlap heavily and "
        "three-year returns do not exist yet. The Newey-West t-stats are massively inflated; read the sign "
        "and the per-sector pattern, never the t-stats, as the signal.",
        "Survivorship-optimistic: the universe is the current large-cap map, with no point-in-time index "
        "membership or delisted names in free data, so the long-short spreads are inflated.",
        "Conclusion: a directional one-year ranking signal concentrated in cyclical sectors with one "
        "coherent standout call (PSU energy undervaluation). It is not validated alpha; Phase C stays gated "
        "pending review and a survivorship-robust, provider-WACC-consistent rebuild.",
    ]
    model.sections.append(s)

    # ---- 7. Risk and honesty contract ----
    s = Section(name="Risk", title="Risk and the Honesty Contract")
    hon = state.get("honesty", {})
    s.metrics = [
        Metric("Horizon of edge", "~1 year", "text"),
        Metric("10-day expectation", "~market drift", "text"),
        Metric("Concentration", "Single sector", "text", good=False),
        Metric("Sector HHI", 1.0, "num", note="All holdings in energy"),
    ]
    s.narrative = [
        f"Horizon of edge: {hon.get('horizon_of_edge', '')}",
        f"Ten-day expectation: {hon.get('ten_day_expectation', '')}",
        f"Validation status: {hon.get('validation_status', '')}",
        f"Concentration risk: {hon.get('concentration_risk', '')}",
        "Up case (+10% to +12% possible): crude pulls back so OMC marketing margins expand, plus a PSU "
        "dividend or re-rating wave in a risk-on tape.",
        "Down case (-10% to -12% just as possible): crude spikes and margins squeeze, or a windfall-tax or "
        "policy shock, or a broad selloff that hits high-beta PSUs hardest.",
        "If sized with real money, size this as a speculative, losable allocation.",
    ]
    model.sections.append(s)

    return model


def main() -> None:
    state_path = FOLDER / "state.json"
    if not state_path.exists():
        raise SystemExit(f"book state not found: {state_path}. Build it first: "
                         f"python scripts/portfolio/india_v3_book.py")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    model = build_model(state)

    xlsx_path = render_xlsx(model, FOLDER / f"{BOOK_NAME}.xlsx")
    pdf_path = render_pdf(model, FOLDER / f"{BOOK_NAME}.pdf")
    pptx_path = render_pptx(model, FOLDER / f"{BOOK_NAME}.pptx")
    save_model(model, FOLDER)   # enables: python -m dcf_project export pdf|ppt india_v3_aggressive

    print("wrote India v3 export package:")
    for p in (xlsx_path, pdf_path, pptx_path):
        print(f"  {p}  ({p.stat().st_size:,} bytes)")
    print(f"  model persisted -> {FOLDER / '.report_model.pkl'}")


if __name__ == "__main__":
    main()
