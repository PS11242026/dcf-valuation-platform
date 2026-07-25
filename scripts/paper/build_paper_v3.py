"""Assemble the three-market FAJ-style manuscript (v3) -> reports/paper_v3/.

This is the cross-market paper: the same fundamentals-only DCF discount signal and the same two
engines (frozen state-space **v1**; economic-driver **v3 Fundamentals Fade**), tested point-in-time
in **three markets** -- the United States, India, and Canada -- to answer *where and when* a DCF
discount ranks forward returns, and how that answer changes across markets.

Presentation layer only: it reads already-produced JSON/CSV under ``reports/`` and emits

    reports/paper_v3/master_research_report_v3.md   -- FAJ-style manuscript (Exec summary ->
        Problem/Claim -> Theory -> Observations -> Conclusion -> Glossary -> References)
    reports/paper_v3/tables/*.csv                    -- exhibit tables (US tables reused from
        build_paper_v2, plus India/Canada by-sector + the cross-market matrices)
    reports/paper_v3/figures/*.png                   -- US figures reused from figures_v2, plus
        the new cross-market figures from figures_v3

and (when pandoc is available) a DOCX. Every headline number is pulled live from its source file,
so the manuscript regenerates from the frozen research artifacts. It does NOT re-run any study.

Sources: US -- reports/research_v2/correction_validation.json, reports/research_v3/{tri_paired_
significance,tri_bm_residual,v3_validation_suites_broad,v3_mc_calibration_n50_*}.json. India --
reports/research_v3_india/india_v3_backtest_broad.json. Canada -- reports/research_v3_canada/
canada_v3_backtest_broad.json.

Usage:
    python -m scripts.paper.build_paper_v3
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from scripts.paper import figures_v2, figures_v3
from scripts.paper.build_paper_v2 import build_tables_v2

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
R2 = REPORTS / "research_v2"
R3 = REPORTS / "research_v3"
IN_BROAD = REPORTS / "research_v3_india" / "india_v3_backtest_broad.json"
CA_BROAD = REPORTS / "research_v3_canada" / "canada_v3_backtest_broad.json"
DEFAULT_BASE = REPORTS / "paper_v3"

SECTOR_LABEL = {
    "energy": "energy", "materials": "materials", "industrials": "industrials",
    "utilities": "utilities", "consumer_staples": "consumer staples",
    "consumer_discretionary": "consumer discretionary", "technology": "technology",
    "communication_services": "communication services", "health_care": "health care",
}


def _j(path: Path) -> dict | None:
    return json.load(open(path)) if path.exists() else None


def _pct(x, d=0) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:.{d}f}%"


def _ic(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:+.3f}"


def _t(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.2f}"


# --------------------------------------------------------------- artifact accessors
def _ic_obs(block: dict, hz: str):
    return ((block or {}).get(hz, {}) or {}).get("ic_obs_weighted")


def _overall_ic(market: dict, engine: str, hz: str):
    key = "overall_metrics_v3" if engine == "v3" else "overall_metrics_v1"
    return _ic_obs(market.get(key, {}), hz)


def _paired(market: dict, hz: str):
    b = (market.get("paired_ic_diff_overall", {}) or {}).get(hz, {}) or {}
    return b.get("mean_ic_diff"), b.get("nw_tstat")


def _ref_median(market: dict):
    return (market.get("reference_distribution", {}) or {}).get("overall", {}).get("p50")


def _ref_sector_median(market: dict, sector: str):
    return ((market.get("reference_distribution", {}) or {}).get("by_sector", {}) or {}
            ).get(sector, {}).get("p50")


def _sector_ranked(market: dict, hz: str, engine: str = "v3"):
    """[(sector, ic, n_obs)] sorted desc by IC for the given engine/horizon."""
    metrics_key = "metrics" if engine == "v3" else "metrics_v1"
    out = []
    for sec, blk in (market.get("by_sector", {}) or {}).items():
        ic = _ic_obs(blk.get(metrics_key, {}), hz)
        if ic is not None:
            out.append((sec, ic, blk.get("n_obs", 0)))
    return sorted(out, key=lambda r: -r[1])


def _fmt_sector_list(ranked, k, label=SECTOR_LABEL):
    return ", ".join(f"{label.get(s, s)} ({ic:+.2f})" for s, ic, _ in ranked[:k])


def _fmt_pos(ranked, thresh=0.05, label=SECTOR_LABEL):
    """Format only the sectors whose IC clears a positive threshold (honest 'where it works')."""
    pos = [(s, ic) for s, ic, _ in ranked if ic >= thresh]
    return ", ".join(f"{label.get(s, s)} ({ic:+.2f})" for s, ic in pos) or "no sector"


def _fmt_neg(ranked, thresh=-0.05, label=SECTOR_LABEL):
    neg = sorted([(s, ic) for s, ic, _ in ranked if ic <= thresh], key=lambda r: r[1])
    return ", ".join(f"{label.get(s, s)} ({ic:+.2f})" for s, ic in neg) or "no sector"


def _sector_ic(market: dict, sector: str, hz: str = "1y", engine: str = "v3"):
    key = "metrics" if engine == "v3" else "metrics_v1"
    return _ic_obs(((market.get("by_sector", {}) or {}).get(sector, {}) or {}).get(key, {}), hz)


# --------------------------------------------------------------- tables
def build_tables_v3(base: Path) -> list[str]:
    tables = base / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    made = build_tables_v2(base)  # US exhibits (correction, ic-by-family, paired, B/M, MC, gate ledger)

    ind = _j(IN_BROAD) or {}
    ca = _j(CA_BROAD) or {}

    # India / Canada by-sector IC (v3 & v1) at 1y -----------------------------------
    for tag, market in (("india", ind), ("canada", ca)):
        rows = []
        for sec, blk in sorted((market.get("by_sector", {}) or {}).items()):
            m, m1 = blk.get("metrics", {}), blk.get("metrics_v1", {})
            rows.append({
                "sector": SECTOR_LABEL.get(sec, sec), "n_obs": blk.get("n_obs"),
                "n_tickers": blk.get("n_tickers"),
                "ic_1y_v3": round(_ic_obs(m, "1y"), 4) if _ic_obs(m, "1y") is not None else None,
                "ic_1y_v1": round(_ic_obs(m1, "1y"), 4) if _ic_obs(m1, "1y") is not None else None,
                "median_discount": round(_ref_sector_median(market, sec), 4)
                if _ref_sector_median(market, sec) is not None else None,
            })
        if rows:
            pd.DataFrame(rows).to_csv(tables / f"{tag}_sector_ic.csv", index=False)
            made.append(f"{tag}_sector_ic.csv")

    # Cross-market overall matrix (market x horizon x engine) -----------------------
    cv = _j(R2 / "correction_validation.json") or {}
    psig = _j(R3 / "tri_paired_significance.json") or {}
    rows = []
    # US: discount IC (corrected) at 1/3/5y + paired v3-v1 diff at 3/5y
    ov = cv.get("overall", {})
    for hz in ("1y", "3y", "5y"):
        b = ov.get(hz, {})
        pd_b = (psig.get("overall", {}).get("horizons", {}) or {}).get(hz, {})
        rows.append({"market": "United States", "horizon": hz,
                     "discount_ic_corrected": round(b.get("full", {}).get("ic"), 4)
                     if b.get("full", {}).get("ic") is not None else None,
                     "paired_v3_minus_v1": round(pd_b.get("mean_ic_diff"), 4)
                     if pd_b.get("mean_ic_diff") is not None else None,
                     "paired_nw_t": round(pd_b.get("nw_tstat"), 2)
                     if pd_b.get("nw_tstat") is not None else None})
    for mkt_name, market in (("India", ind), ("Canada", ca)):
        for hz in ("1y", "2y", "3y", "5y"):
            v3 = _overall_ic(market, "v3", hz)
            if v3 is None:
                continue
            md, mt = _paired(market, hz)
            rows.append({"market": mkt_name, "horizon": hz,
                         "discount_ic_corrected": round(v3, 4),
                         "paired_v3_minus_v1": round(md, 4) if md is not None else None,
                         "paired_nw_t": round(mt, 2) if mt is not None else None})
    if rows:
        pd.DataFrame(rows).to_csv(tables / "cross_market_overall.csv", index=False)
        made.append("cross_market_overall.csv")
    return made


# --------------------------------------------------------------- manuscript
def build_master_v3(base: Path) -> Path:
    cv = _j(R2 / "correction_validation.json") or {}
    ov = cv.get("overall", {})
    fam3 = cv.get("by_family_3y", {})
    psig = _j(R3 / "tri_paired_significance.json") or {}
    bm = _j(R3 / "tri_bm_residual.json") or {}
    suites = _j(R3 / "v3_validation_suites_broad.json") or {}
    mc3 = _j(R3 / "v3_mc_calibration_n50_3y.json") or {}
    ind = _j(IN_BROAD) or {}
    ca = _j(CA_BROAD) or {}

    def f_full(hz):
        b = ov.get(hz, {}); return b.get("full", {}).get("ic"), b.get("full", {}).get("t")

    def f_orig(hz):
        b = ov.get(hz, {}); return b.get("original", {}).get("ic"), b.get("original", {}).get("t")

    def us_paired(hz):
        b = (psig.get("overall", {}).get("horizons", {}) or {}).get(hz, {})
        return b.get("mean_ic_diff"), b.get("nw_tstat"), b.get("boot_ci95")

    def us_bm_overall(hz="3y"):
        b = ((bm.get("overall", {}) or {}).get(hz, {}) or {}).get("v3", {}).get("bm_resid_ic", {})
        return b.get("mean"), b.get("t_stat")

    def us_bm_fam(fam, hz="3y"):
        b = ((bm.get("by_family", {}).get(fam, {}) or {}).get(hz, {}) or {}).get("v3", {}).get("bm_resid_ic", {})
        return b.get("mean"), b.get("t_stat")

    # India / Canada headline numbers
    in_1y = _overall_ic(ind, "v3", "1y"); in_2y = _overall_ic(ind, "v3", "2y")
    in_rank = _sector_ranked(ind, "1y", "v3")
    in_ref = _ref_median(ind)
    in_energy_disc = _ref_sector_median(ind, "energy")
    in_paired_1y = _paired(ind, "1y")
    in_nobs = ind.get("n_observations"); in_nnames = ind.get("n_names_with_obs")

    ca_1y_v3 = _overall_ic(ca, "v3", "1y"); ca_1y_v1 = _overall_ic(ca, "v1", "1y")
    ca_3y_v3 = _overall_ic(ca, "v3", "3y"); ca_3y_v1 = _overall_ic(ca, "v1", "3y")
    ca_5y_v3 = _overall_ic(ca, "v3", "5y")
    ca_rank = _sector_ranked(ca, "1y", "v3")
    ca_ref = _ref_median(ca)
    ca_paired_1y = _paired(ca, "1y"); ca_paired_3y = _paired(ca, "3y")
    ca_nobs = ca.get("n_observations"); ca_nnames = ca.get("n_names_with_obs")

    # the materials sign-flip: India materials (domestic-demand cyclicals) vs Canada materials
    # (commodity-price-taking miners) -- the controlled test of "forecastability, not GICS label".
    in_mat = _sector_ic(ind, "materials", "1y", "v3")
    ca_mat = _sector_ic(ca, "materials", "1y", "v3")
    ca_ind = _sector_ic(ca, "industrials", "1y", "v3")

    L: list[str] = []
    A = L.append

    # ---- YAML title block (rendered by pandoc as a title page) ----
    A('---\n'
      'title: "Where Discounted Cash Flow Works"\n'
      'subtitle: "A Point-in-Time, Three-Market, Two-Engine Study of the DCF Discount as a '
      'Cross-Sectional Return Signal in the United States, India, and Canada"\n'
      'author: Pulkit Sharma\n'
      'date: June 2026\n'
      '---')
    A("")

    # ---- 1. Executive summary ----
    A("## 1. Executive Summary")
    A("")
    A("We ask a deceptively simple question: if one values a company with a disciplined, "
      "fundamentals-only discounted-cash-flow (DCF) model and buys it when it trades below that "
      "intrinsic value, does the resulting *discount-to-intrinsic* rank stocks on their subsequent "
      "returns? We answer it the same way in three markets -- the United States, India, and Canada -- "
      "using strict point-in-time (PIT) data and two independently constructed valuation engines, and "
      "we report a result that is more interesting than a simple yes or no.")
    A("")
    A("**The signal is real but conditional, and the condition is not the same in every market.** In "
      "the United States, once the return panel is correctly adjusted for corporate actions, the broad "
      f"cross-sectional signal is statistically indistinguishable from zero (corrected 3-year rank "
      f"information coefficient {_ic(f_full('3y')[0])}, t {_t(f_full('3y')[1])}, versus "
      f"{_ic(f_orig('3y')[0])} before the adjustment); it survives only where cash flows are stable "
      f"and forecastable -- consumer staples retains a corrected 3-year IC of "
      f"{_ic(fam3.get('consumer_staples', {}).get('full'))}. In India, the same signal is a "
      f"**directional 1-year ranker** (obs-weighted rank IC {_ic(in_1y)}; 2-year {_ic(in_2y)}) that "
      f"concentrates in *cyclical, capital-intensive* sectors -- "
      f"{_fmt_sector_list(in_rank, 3)} -- and is flat-to-negative in the very families (staples, "
      f"technology, health care) where the US signal lives. In Canada -- a developed market whose "
      f"investable DCF universe is, like India's, dominated by commodity and cyclical sectors -- the "
      f"discount is a positive value signal at longer horizons (3-year rank IC {_ic(ca_3y_v3)}, "
      f"5-year {_ic(ca_5y_v3)}) but flat at one year ({_ic(ca_1y_v3)}); decisively, its *sectoral* "
      f"pattern separates two competing explanations of the reversal -- *market development* versus "
      f"*cash-flow forecastability*: forecastable industrials predict (1-year IC {_ic(ca_ind)}) while "
      f"commodity-price-taking materials are wrong-signed ({_ic(ca_mat)}), the mirror image of India's "
      f"domestic-demand materials ({_ic(in_mat)}). The data favour forecastability, not development.")
    A("")
    A("**A better forecast is not broader alpha.** Replacing the forecast layer with an economically "
      "grounded engine (the v3 Fundamentals Fade: competitive-advantage-period / return-on-capital "
      "persistence with a Grinold-Kroner terminal) produces a *strictly better-behaved* valuation and, "
      f"on the corrected US panel, paired rank-IC improvement over the frozen baseline (3-year diff "
      f"{_ic(us_paired('3y')[0])}, Newey-West t {_t(us_paired('3y')[1])}). But the Book-to-Market value "
      f"factor **spans** that broad edge (US discount IC net of Book/Market overall "
      f"{_ic(us_bm_overall()[0])}, t {_t(us_bm_overall()[1])}); the incremental-to-value signal survives "
      f"only in consumer staples and consumer discretionary. In India and Canada the economic engine "
      f"does **not** broadly out-rank the frozen one (paired 1-year diff {_ic(in_paired_1y[0])} India, "
      f"{_ic(ca_paired_1y[0])} Canada), so its value is interpretability and auditability, not extra "
      f"cross-sectional power.")
    A("")
    A("**Contributions.** (i) A *cross-market* characterisation of DCF-discount predictability that "
      "reverses by sector between a developed growth market (US) and commodity-cyclical markets "
      "(India, Canada), isolating *cash-flow forecastability* -- not market development per se -- as the "
      "operative condition. (ii) A reusable methods kit: PIT return-panel hygiene (a single missing "
      "split adjustment manufactured the original US 'signal'), in-process *paired* engine comparison, "
      "overlap-aware inference (Newey-West and block bootstrap), pre-registered kill-gates, and a "
      "documented defect post-mortem that reversed a kill. (iii) A transparent, auditable valuation "
      "engine offered with a precisely-bounded ranking claim rather than a broad alpha promise.")
    A("")
    A("> **Scope and honesty caveats.** The US result is the most mature (≈10,000 firm-quarters, "
      "corrected panel, full factor controls). The India and Canada results are **preliminary**: the "
      "structured-data windows are short and overlapping, the universes are survivorship-optimistic "
      "(current large-cap constituents, no delisted names in free data), and the reported t-statistics "
      "are inflated by cross-sectional dependence and few independent periods. Read the **sign, "
      "magnitude, and sectoral pattern** of the India/Canada ICs as the evidence -- never the "
      "t-statistics as proof of significance.")
    A("")

    # ---- 2. Problem statement / claim ----
    A("## 2. Problem Statement and Claim")
    A("")
    A("### 2.1 The question")
    A("")
    A("The discount-to-intrinsic of a disciplined DCF is the oldest quantitative value signal there "
      "is (Williams 1938; Gordon and Shapiro 1956): estimate the present value of a firm's future free "
      "cash flows, compare it to price, and prefer what trades cheap. A large literature establishes "
      "that *value* predicts the cross-section of returns (Basu 1977; Fama and French 1992, 1993; "
      "Lakonishok, Shleifer, and Vishny 1994), and that model-based *intrinsic-value-to-price* ratios "
      "in particular carry predictive content (Frankel and Lee 1998; Lee, Myers, and Swaminathan 1999). "
      "Yet most of this evidence is single-market (US) and single-model. Two questions are "
      "under-examined and are the subject of this paper:")
    A("")
    A("1. **Is DCF-discount predictability universal or conditional** -- does it hold across the whole "
      "cross-section, or only in identifiable economic regimes; and if conditional, *does the "
      "conditioning travel across markets* with very different sector compositions and stages of "
      "development?")
    A("2. **Does a more economically coherent forecast buy predictive power**, or only "
      "interpretability -- i.e., is a transparent CAP/fade engine a better *ranker* than a frozen "
      "baseline, and does any edge survive a control for the cheapest available value factor?")
    A("")
    A("### 2.2 The claim")
    A("")
    A("We defend three propositions with the data:")
    A("")
    A("- **C1 (Conditionality).** A fundamentals-only DCF discount is *not* a broad cross-sectional "
      "signal once prices are correctly adjusted; it is a **regime-conditional** signal that works "
      "where future cash flows are forecastable and fails where value is dominated by unmodelled "
      "optionality or violent multiple cycles.")
    A("- **C2 (Cross-market reversal, and its true driver).** The *identity* of the forecastable "
      "regime is **market-specific**: in the US it is the stable-cash-flow consumer families; in "
      "India it is the domestic-demand cyclicals (energy, utilities, materials, industrials); in "
      "Canada it is the forecastable industrials (rails, waste) and a broad value signal that "
      "strengthens with horizon. The operative condition is **cash-flow forecastability**, not the "
      "GICS sector label and not the level of market development. The decisive evidence is a *sign "
      f"reversal within the same sector label*: 'materials' predicts in India ({_ic(in_mat)} -- "
      f"domestic-demand cement/chemical cyclicals) but is wrong-signed in Canada ({_ic(ca_mat)} -- "
      "gold and metal miners whose equity tracks exogenous commodity prices, not modellable "
      "fundamentals). A developed market does not behave like the US by virtue of being developed; it "
      "behaves according to which of its businesses have forecastable cash flows.")
    A("- **C3 (Coherence ≠ alpha).** An economically coherent forecast engine is a strictly better, "
      "auditable *valuation*, and on the corrected US panel a marginally better *ranker*, but a free "
      "Book-to-Market factor spans its broad edge and it adds no broad ranking power in India or "
      "Canada. Economic coherence and broad predictive power are distinct objectives.")
    A("")

    # ---- 3. Theory ----
    A("## 3. Theory and Method")
    A("")
    A("### 3.1 Intrinsic value and the discount signal")
    A("")
    A("For a firm with free cash flow to the firm $FCFF_t$ discounted at the weighted-average cost of "
      "capital $r$ (WACC), enterprise value is "
      "$EV_0 = \\sum_{t=1}^{N} \\frac{FCFF_t}{(1+r)^t} + \\frac{TV_N}{(1+r)^N}$, where the terminal "
      "value $TV_N$ closes the horizon. Equity value per share follows after netting debt and cash and "
      "dividing by shares. The signal we test is the **discount-to-intrinsic**, "
      "$\\text{disc} = V/P - 1$, where $V$ is modelled intrinsic value per share and $P$ is price: "
      "positive means the model judges the stock cheap. This is the ratio whose cross-sectional "
      "predictive content Frankel and Lee (1998) first documented for the US.")
    A("")
    A("### 3.2 Two engines")
    A("")
    A("We hold the data layer fixed and vary only the forecast layer, comparing two engines that "
      "consume an identical PIT data bundle:")
    A("")
    A("- **Engine v1 (frozen state-space baseline).** The production model: a state-space projection of "
      "the firm's own fundamentals, deliberately conservative, used as the live and benchmark engine. "
      "It is held byte-identical throughout so that every cross-engine difference is attributable to "
      "the forecast, not the data.")
    A("- **Engine v3 (Fundamentals Fade).** An economic-driver model built for auditability. The "
      "forecast is specified by three mean-reverting primitives -- revenue growth, operating margin, "
      "and the return on *new* invested capital (RONIC) -- that fade toward sector anchors at "
      "company-specific half-lives (estimated by a through-origin AR(1) on the firm's own history, "
      "empirical-Bayes shrunk toward a sector prior). Reinvestment is **derived** every period from the "
      "sustainable-growth identity $g = \\text{reinvestment rate} \\times RONIC$, so growth is always "
      "paid for. Excess returns on capital decay at a fitted persistence rate $\\phi$; the implied "
      "**competitive advantage period** (CAP) is the time for excess ROIC to reach a floor (Mauboussin "
      "and Johnson 1997; Rappaport and Mauboussin 2001). The terminal growth rate is *derived*, not "
      "chosen, from the **Grinold-Kroner** decomposition (long-run nominal GDP less a dilution "
      "haircut, clamped to a macro band) (Grinold and Kroner 2002). An output-coherence guard "
      "withholds arithmetically corrupt valuations (negative equity, terminal blow-ups). A worked "
      "example of the fade paths -- growth and margin reverting to anchors while RONIC decays toward "
      "the cost of capital and the derived reinvestment line tracks both -- is shown in Exhibit 1.")
    A("")
    A("![**Exhibit 1.** Worked example (Coca-Cola): the v3 fade paths -- revenue growth and operating "
      "margin reverting to anchors, RONIC decaying toward WACC, and reinvestment derived from "
      "g = reinvestment x RONIC.](figures/fade_paths_KO.png)")
    A("")
    A("### 3.3 Point-in-time discipline and the markets")
    A("")
    A("The signal at quarter-end $t$ uses only fundamentals filed and public by $t$ (a reporting lag is "
      "enforced); forward total returns are measured strictly after $t$ from split- and "
      "dividend-adjusted prices. The three markets differ only in their data plumbing, never in the "
      "engine or the statistics:")
    A("")
    A("- **United States** -- SEC EDGAR XBRL fundamentals; US Treasury risk-free and US equity risk "
      "premium; ≈10,000 firm-quarters across a large-cap universe.")
    A(f"- **India** -- live NSE Ind-AS XBRL fundamentals; Government-of-India G-Sec risk-free and a "
      f"Damodaran India total equity-risk premium (≈8%); internal-CAPM WACC; "
      f"{in_nnames or 'n'} names / {in_nobs or 'n'} firm-quarter observations.")
    A(f"- **Canada** -- SEC EDGAR XBRL fundamentals for the (large, cross-listed) SEC-filer subset of "
      f"the TSX, valued in each firm's *reporting currency* (Government-of-Canada risk-free and "
      f"Canadian equity-risk premium for CAD reporters; US anchors for USD reporters); "
      f"{ca_nnames or 'n'} names / {ca_nobs or 'n'} firm-quarter observations.")
    A("")
    A("### 3.4 Statistics")
    A("")
    A("The headline metric is the **rank information coefficient** (Spearman rank correlation between "
      "the signal and forward return), computed per cross-section and aggregated obs- and "
      "equal-weighted. Because multi-year forward windows overlap, naive t-statistics overstate "
      "significance; we report **Newey-West** heteroskedasticity- and autocorrelation-consistent "
      "t-statistics with horizon-scaled lags (Newey and West 1987) and, for the US engine comparison, a "
      "contiguous-block bootstrap (Politis and Romano 1994). Engine claims are **pre-registered**: "
      "PASS/KILL thresholds are fixed before the run and honoured. Cross-engine comparisons are "
      "**paired**: v1 and v3 value the identical bundle at every name-date, so a difference is never an "
      "artifact of differing data or split treatment -- the precise error that produced the original "
      "US false positive (Section 4.1).")
    A("")

    # ---- 4. Observations ----
    A("## 4. Observations")
    A("")
    A("### 4.1 The United States: a correction, a conditional survivor, and a spanned engine")
    A("")
    A(f"The original, uncorrected US study showed a strong, horizon-increasing signal (3-year rank IC "
      f"{_ic(f_orig('3y')[0])}, t {_t(f_orig('3y')[1])}). A corporate-action audit of the return panel "
      f"overturned it. Decomposing the adjustment isolates the cause: the **split adjustment is "
      f"decisive**. Dividend adjustment alone would have made the signal look slightly stronger; it is "
      f"the split adjustment that collapses the fully corrected broad 3-year IC to {_ic(f_full('3y')[0])} "
      f"(t {_t(f_full('3y')[1])}) -- statistically indistinguishable from zero (Exhibit 2; Table 1). "
      f"The lesson is methodological and sobering: a single missing return-panel adjustment, not "
      f"model error, manufactured years of apparent signal.")
    A("")
    A("Table 1. The correction decomposition (US): the split adjustment alone is decisive.")
    A("")
    A("| Horizon | Original IC | Split-only | Dividend-only | Corrected IC |")
    A("|:--|--:|--:|--:|--:|")
    for _hz in ("1y", "3y", "5y"):
        _b = ov.get(_hz, {})
        A(f"| {_hz} | {_ic(_b.get('original', {}).get('ic'))} | {_ic(_b.get('split_only', {}).get('ic'))} "
          f"| {_ic(_b.get('div_only', {}).get('ic'))} | {_ic(_b.get('full', {}).get('ic'))} |")
    A("")
    A(f"Correction does not zero the signal *everywhere*. By sector family at 3 years (Exhibit 3), "
      f"the clear survivor is "
      f"**consumer staples** (corrected 3-year IC {_ic(fam3.get('consumer_staples', {}).get('full'))}), "
      f"with consumer discretionary retaining a weaker edge "
      f"({_ic(fam3.get('consumer_discretionary', {}).get('full'))}). The most spectacular "
      f"pre-correction numbers were the most spurious -- internet platforms collapsed from "
      f"{_ic(fam3.get('internet_platforms_marketplaces', {}).get('original'))} to "
      f"{_ic(fam3.get('internet_platforms_marketplaces', {}).get('full'))}, and semiconductors flipped "
      f"negative. A fundamentals DCF discount predicts where cash flows are stable and forecastable, "
      f"and not where value is dominated by unmodelled optionality or multiple cycles -- the first "
      f"statement of conditionality (C1).")
    A("")
    A(f"Does a more coherent forecast recover broad predictiveness? A pre-registered paired backtest "
      f"first **killed** the v3 engine; the kill was then traced to a margin-seasonality defect (the "
      f"fade targeted seasonal per-quarter margins rather than trailing-twelve-month margins, injecting "
      f"a spurious compression and a systemic half-life floor). On the corrected engine the verdict "
      f"reverses: v3 **out-ranks** v1 on identical bundles (paired 3-year IC diff {_ic(us_paired('3y')[0])}, "
      f"NW t {_t(us_paired('3y')[1])}; 5-year diff {_ic(us_paired('5y')[0])}), surviving a post-2021 "
      f"out-of-sample holdout and every macro regime (Exhibit 4). "
      f"**But a better fade is not a unique alpha source.** Residualising the discount on the "
      f"Book-to-Market rank each cross-section, the engine's broad edge net of value is gone (overall "
      f"net-of-B/M IC {_ic(us_bm_overall()[0])}, t {_t(us_bm_overall()[1])}); it survives only in "
      f"consumer discretionary (net {_ic(us_bm_fam('consumer_discretionary')[0])}, t "
      f"{_t(us_bm_fam('consumer_discretionary')[1])}) and consumer staples "
      f"(net {_ic(us_bm_fam('consumer_staples')[0])}, t {_t(us_bm_fam('consumer_staples')[1])}), and is "
      f"wrong-signed in semiconductors (Exhibit 5). The driver-level Monte "
      f"Carlo bands are *not* calibrated uncertainty (3-year p10-p90 coverage "
      f"{_pct(mc3.get('coverage_p10_p90'), 1)} versus an 80% nominal) and are rendered as a "
      f"sensitivity range only. This is C3.")
    A("")
    A("![**Exhibit 2.** US DCF-discount rank IC, original (unadjusted) vs split- and dividend-corrected: "
      "the broad signal collapses once corporate actions are applied.](figures/orig_vs_corrected_ic.png)")
    A("")
    A("![**Exhibit 3.** What survives correction: US corrected 3-year rank IC by sector family. Only "
      "stable-cash-flow consumer staples retains a material edge.](figures/ic_by_family.png)")
    A("")
    A("![**Exhibit 4.** The reversed kill: on the corrected panel the economic v3 engine out-ranks the "
      "frozen v1 at 3-5 years (block-bootstrap 95% CI excludes zero).](figures/v3_corrected_paired.png)")
    A("")
    A("![**Exhibit 5.** The boundary: US v3 discount IC net of Book/Market by family. The incremental-"
      "to-value edge survives only in consumer staples and discretionary; semiconductors are "
      "wrong-signed.](figures/v3_bm_control.png)")
    A("")
    A("### 4.2 India: the conditioning inverts")
    A("")
    A(f"Re-running the same machinery on the India data layer ({in_nnames or 'n'} names, "
      f"{in_nobs or 'n'} firm-quarter observations) yields a positive, directional 1-year ranker (v3 "
      f"obs-weighted rank IC {_ic(in_1y)}; 2-year {_ic(in_2y)}). Its sectoral map is the striking part "
      f"and is the *opposite* of the US (Exhibit 6). The discount predicts best in **cyclical, "
      f"capital-intensive** "
      f"sectors -- {_fmt_pos(in_rank)} -- and is anti-predictive in {_fmt_neg(in_rank)}. The very "
      f"families that carry the US signal (consumer staples, technology, health care) are exactly "
      f"where the India signal is weakest.")
    A("")
    A(f"The cross-sectional discount distribution explains why. Against a 12-16% nominal-WACC DCF, the "
      f"median Indian large cap looks *expensive* (median discount {_pct(in_ref, 0)}): high-quality "
      f"Indian compounders trade at 45-60x earnings, which a disciplined DCF reads as richly valued -- "
      f"structurally, not as a bug (the data were verified to the rupee). The one sector the model finds "
      f"genuinely **cheap** is energy (median discount {_pct(in_energy_disc, 0)}) -- the state-owned oil "
      f"names at ~5-8x earnings -- and energy is *also* among the highest-IC sectors: in India the "
      f"discount predicts returns precisely where it finds value (Exhibit 7). The economic engine "
      f"does not broadly help here (paired v3-v1 1-year diff {_ic(in_paired_1y[0])}).")
    A("")
    A("![**Exhibit 6.** India 1-year by-sector rank IC (v3 vs frozen v1): the discount ranks returns "
      "in cyclical, capital-intensive sectors and fails in staples, technology, and health care -- the "
      "inverse of the US.](figures/india_sector_ic.png)")
    A("")
    A("![**Exhibit 7.** Median DCF discount by sector (India vs Canada): India's only cheap sector is "
      "energy (state-owned oil), which is also where the discount predicts best.]"
      "(figures/market_discount_distribution.png)")
    A("")
    A("### 4.3 Canada: a natural experiment separating development from forecastability")
    A("")
    A(f"India is an emerging market; one might attribute its cyclical tilt to market immaturity. Canada "
      f"tests that directly: it is a **developed** market (deep, liquid, well-regulated) whose "
      f"investable DCF universe is, like India's, dominated by **commodity and cyclical** sectors "
      f"(energy, materials, utilities, rails). If the reversal is about market *development*, Canada "
      f"should look like the US; if it is about cash-flow *forecastability*, Canada should track "
      f"whichever of its sectors have modellable cash flows. We value the SEC-filer TSX large caps "
      f"point-in-time ({ca_nnames or 'n'} names, {ca_nobs or 'n'} firm-quarter observations), each in "
      f"its own reporting currency (a per-name detection: CAD reporters priced on the TSX, USD "
      f"reporters on their US listing).")
    A("")
    A(f"The Canadian evidence is more refined than 'Canada equals India', and that refinement is the "
      f"paper's sharpest result. As a *value* signal the discount works in Canada at longer horizons "
      f"(3-year rank IC {_ic(ca_3y_v3)}, 5-year {_ic(ca_5y_v3)}) but is flat at one year "
      f"({_ic(ca_1y_v3)}; frozen v1 {_ic(ca_1y_v1)}) -- consistent with a slow-moving value premium "
      f"rather than a fast signal. The one-year *sectoral* map is where the mechanism shows itself "
      f"(Exhibits 8-9): it predicts in the **forecastable** industrials -- rails and "
      f"waste, toll-like cash flows -- ({_ic(ca_ind)}) and is **wrong-signed in materials** "
      f"({_ic(ca_mat)}). The median Canadian large cap is cheap on the model (median discount "
      f"{_pct(ca_ref, 0)}) -- a value/commodity market, not a high-multiple growth market.")
    A("")
    A(f"The materials cell is the controlled experiment. In India 'materials' *predicts* "
      f"({_ic(in_mat)}); in Canada the **same GICS label** is *anti-predictive* ({_ic(ca_mat)}). The "
      f"reason is economic, not statistical: Indian materials are domestic-demand cement and chemical "
      f"cyclicals whose cash flows track a forecastable domestic capex cycle, whereas Canadian "
      f"materials are gold and base-metal **miners whose equity returns track exogenous, "
      f"violently-cycling commodity prices** that no trailing-fundamentals DCF can anticipate -- the "
      f"exact pathology that makes US semiconductors wrong-signed net of value (Section 4.1). This is "
      f"the central evidence for **C2**: the operative condition is **cash-flow forecastability**, not "
      f"the sector label and not the level of market development. A developed market (Canada) and an "
      f"emerging one (India) agree wherever cash flows are forecastable and disagree exactly where the "
      f"same label hides different economics. As in the US and India, the economic-fade engine does "
      f"not out-rank the frozen baseline in Canada (paired v3-v1 1-year diff {_ic(ca_paired_1y[0])}; "
      f"3-year {_ic(ca_paired_3y[0])}), reinforcing **C3** across all three markets.")
    A("")
    A("![**Exhibit 8.** Canada 1-year by-sector rank IC (v3 vs frozen v1): forecastable industrials "
      "predict; commodity-price-taking materials are wrong-signed.](figures/canada_sector_ic.png)")
    A("")
    A("![**Exhibit 9.** The controlled experiment: 1-year by-sector IC, India vs Canada. Energy and "
      "industrials agree; 'materials' flips sign (India domestic-demand cyclicals vs Canadian "
      "commodity-price-taking miners).](figures/india_canada_sector_ic.png)")
    A("")
    A("### 4.4 The cross-market picture")
    A("")
    A("Exhibit 10 is the paper in one image: a market-by-sector matrix of rank IC. The green cells -- "
      "where the DCF discount ranks returns -- sit in different columns in different markets. The US "
      "lights up in consumer staples and discretionary; India in its domestic-demand cyclicals "
      "(energy, materials, utilities, industrials); Canada in its forecastable industrials, with "
      "energy weakly positive and commodity-price-taking materials conspicuously red. The signal is "
      "neither absent nor universal: it tracks the *forecastable* corner of each market, wherever the "
      "market's economic composition places it (Table 2). Exhibit 11 shows the complementary negative "
      "result -- the economic engine's paired improvement over the frozen baseline is, at best, narrow "
      "in the US and absent in India and Canada.")
    A("")
    A("Table 2. Cross-market summary: corrected DCF-discount rank IC and the paired v3 - v1 "
      "improvement, by market and horizon. (US discount IC is the frozen-engine corrected panel; "
      "India and Canada are preliminary -- read sign and magnitude, not significance.)")
    A("")
    A("| Market | Horizon | Discount rank IC | Paired v3 - v1 |")
    A("|:--|:--|--:|--:|")
    for _hz in ("1y", "3y", "5y"):
        A(f"| United States | {_hz} | {_ic(ov.get(_hz, {}).get('full', {}).get('ic'))} "
          f"| {_ic(us_paired(_hz)[0])} |")
    for _hz in ("1y", "2y", "3y"):
        _disc = _overall_ic(ind, "v3", _hz)
        if _disc is not None:
            A(f"| India | {_hz} | {_ic(_disc)} | {_ic(_paired(ind, _hz)[0])} |")
    for _hz in ("1y", "3y", "5y"):
        _disc = _overall_ic(ca, "v3", _hz)
        if _disc is not None:
            A(f"| Canada | {_hz} | {_ic(_disc)} | {_ic(_paired(ca, _hz)[0])} |")
    A("")
    A("![**Exhibit 10 (central exhibit).** Where the DCF discount ranks forward returns: a market x "
      "sector matrix of rank IC. The green cells sit in different columns in each market, and the same "
      "'materials' label is green in India yet deep-red in Canada.](figures/where_dcf_works_heatmap.png)")
    A("")
    A("![**Exhibit 11.** Paired v3 - v1 rank-IC improvement by market: the economically grounded engine "
      "is, at best, a narrow improvement and never a broad alpha source.](figures/cross_market_paired.png)")
    A("")

    # ---- 5. Conclusion ----
    A("## 5. Conclusion")
    A("")
    A("A disciplined DCF discount is a **conditional, market-specific** value signal. In the United "
      "States it is, broadly, indistinguishable from zero once prices are correctly adjusted, surviving "
      "only in stable-cash-flow consumer families; in India it lives in the domestic-demand cyclicals, "
      "and in Canada in the forecastable industrials and a slower, longer-horizon value premium. The "
      "reversal is not about developed versus emerging markets -- developed Canada and emerging India "
      "agree wherever cash flows are forecastable, and the *same* GICS label (materials) flips sign "
      "between them because its economic content differs (domestic-demand cyclicals in India, "
      "commodity-price-taking miners in Canada). What conditions the signal is cash-flow "
      "forecastability, not market development. Economic sophistication in "
      "the forecast layer changes the picture less than one would hope: the Fundamentals Fade engine is "
      "a strictly better-behaved, auditable valuation and, on the corrected US panel, a marginally "
      "better ranker, but the Book-to-Market value factor spans its broad edge and it adds no broad "
      "ranking power abroad. The durable deliverables are therefore methodological -- PIT return-panel "
      "hygiene, paired in-process engine comparison, overlap-aware inference, pre-registered gates, and "
      "a kill reversed by its own post-mortem -- and a transparent valuation engine sold with a "
      "precisely-bounded, scoped claim rather than a broad alpha promise.")
    A("")
    A("**Limitations and future work.** The US evidence is mature; the India and Canada panels are "
      "preliminary -- short, overlapping windows, survivorship-optimistic universes, and inflated "
      "t-statistics. The natural next steps are a survivorship-robust (delisting-inclusive) panel in "
      "both markets, a Book-to-Market control abroad (as done for the US) to test whether the cyclical "
      "edge is incremental to value, and live, out-of-sample accrual of the conditional signal. Until "
      "then the India and Canada findings should be read as directional and structural, not as "
      "validated alpha.")
    A("")

    # ---- Glossary ----
    A("## Glossary of Key Terms and Metrics")
    A("")
    glossary = [
        ("Discounted cash flow (DCF)", "Valuation that sets a firm's value to the present value of its "
         "expected future free cash flows, discounted at a required rate of return."),
        ("Free cash flow to the firm (FCFF)", "After-tax operating cash flow available to all capital "
         "providers, after reinvestment (capex and working capital), before financing flows."),
        ("Intrinsic value (V)", "The DCF-modelled per-share value, as opposed to the market price (P)."),
        ("Discount-to-intrinsic", "The signal tested here: V/P − 1. Positive = the model judges the "
         "stock cheap relative to price."),
        ("WACC (r)", "Weighted-average cost of capital: the blended, tax-adjusted required return on "
         "debt and equity used to discount FCFF."),
        ("CAPM / Equity risk premium (ERP)", "The capital-asset-pricing model sets cost of equity = "
         "risk-free rate + beta × ERP; the ERP is the extra return demanded for equity risk."),
        ("Terminal value (TV)", "The value of cash flows beyond the explicit forecast horizon, "
         "capitalised into a single figure at the horizon."),
        ("Grinold-Kroner decomposition", "A framework deriving long-run equity/earnings growth from "
         "income, real growth, and inflation components; used here to *derive* (not pick) terminal "
         "growth from long-run nominal GDP."),
        ("ROIC / RONIC", "Return on invested capital / return on *new* invested capital -- the "
         "profitability of the existing book and of incremental investment, respectively."),
        ("Competitive advantage period (CAP)", "The length of time a firm is expected to earn returns "
         "above its cost of capital before competition erodes them to normal."),
        ("Fade", "The mean-reversion of a forecast driver (growth, margin, RONIC) toward a long-run "
         "anchor at an estimated half-life."),
        ("Sustainable-growth identity", "g = reinvestment rate × RONIC: growth must be funded by "
         "reinvestment, so it cannot be assumed for free."),
        ("NOPAT", "Net operating profit after tax: operating profit (EBIT) taxed at the marginal rate, "
         "before financing."),
        ("TTM", "Trailing twelve months -- a four-quarter sum that removes seasonality from flow "
         "measures such as margins."),
        ("Point-in-time (PIT) / look-ahead bias", "Using only information available on the decision "
         "date. A look-ahead bias contaminates a backtest with data that was not yet public."),
        ("Rank information coefficient (rank IC)", "The Spearman rank correlation between a signal and "
         "subsequent return, measured per cross-section; the standard measure of a ranking signal's "
         "strength (Spearman 1904)."),
        ("Newey-West t-statistic", "A heteroskedasticity- and autocorrelation-consistent (HAC) "
         "t-statistic; necessary when forward-return windows overlap (Newey and West 1987)."),
        ("Block bootstrap", "Resampling contiguous blocks of dates to build confidence intervals that "
         "respect serial dependence (Politis and Romano 1994)."),
        ("Long-short (quintile/decile) spread", "The return difference between the top and bottom "
         "fractile ranked on the signal -- a portfolio read of predictive strength."),
        ("Book-to-Market (B/M)", "Book equity divided by market equity; the canonical academic value "
         "factor (Fama and French 1992, 1993)."),
        ("Spanning / residualisation", "Testing whether a signal adds information *beyond* a known "
         "factor by regressing it on that factor and correlating the residual with returns."),
        ("Survivorship bias", "Upward bias from studying only firms that survived to the present, "
         "omitting delisted failures (Brown et al. 1992)."),
        ("Paired comparison", "Valuing two engines on the identical data bundle per name-date, so any "
         "difference is forecast-driven, not data-driven."),
        ("Coherence guard", "A product safeguard that withholds arithmetically corrupt valuations "
         "(negative equity, terminal blow-ups) rather than display them."),
    ]
    for term, definition in glossary:
        A(f"- **{term}.** {definition}")
    A("")

    # ---- References ----
    A("## References")
    A("")
    refs = [
        "Basu, S. 1977. \"Investment Performance of Common Stocks in Relation to Their "
        "Price-Earnings Ratios: A Test of the Efficient Market Hypothesis.\" *Journal of Finance* "
        "32 (3): 663-682.",
        "Brown, S. J., W. N. Goetzmann, R. G. Ibbotson, and S. A. Ross. 1992. \"Survivorship Bias in "
        "Performance Studies.\" *Review of Financial Studies* 5 (4): 553-580.",
        "Damodaran, A. 2012. *Investment Valuation: Tools and Techniques for Determining the Value of "
        "Any Asset.* 3rd ed. Hoboken, NJ: John Wiley & Sons. (Country-risk-premium and "
        "equity-risk-premium datasets, Damodaran Online.)",
        "Fama, E. F., and K. R. French. 1992. \"The Cross-Section of Expected Stock Returns.\" "
        "*Journal of Finance* 47 (2): 427-465.",
        "Fama, E. F., and K. R. French. 1993. \"Common Risk Factors in the Returns on Stocks and "
        "Bonds.\" *Journal of Financial Economics* 33 (1): 3-56.",
        "Fama, E. F., and J. D. MacBeth. 1973. \"Risk, Return, and Equilibrium: Empirical Tests.\" "
        "*Journal of Political Economy* 81 (3): 607-636.",
        "Frankel, R., and C. M. C. Lee. 1998. \"Accounting Valuation, Market Expectation, and "
        "Cross-Sectional Stock Returns.\" *Journal of Accounting and Economics* 25 (3): 283-319.",
        "Gordon, M. J., and E. Shapiro. 1956. \"Capital Equipment Analysis: The Required Rate of "
        "Profit.\" *Management Science* 3 (1): 102-110.",
        "Grinold, R. C., and K. F. Kroner. 2002. \"The Equity Risk Premium: Analyzing the Long-Run "
        "Prospects for the Stock Market.\" *Investment Insights* (Barclays Global Investors) 5 (3).",
        "Lakonishok, J., A. Shleifer, and R. W. Vishny. 1994. \"Contrarian Investment, Extrapolation, "
        "and Risk.\" *Journal of Finance* 49 (5): 1541-1578.",
        "Lee, C. M. C., J. Myers, and B. Swaminathan. 1999. \"What Is the Intrinsic Value of the "
        "Dow?\" *Journal of Finance* 54 (5): 1693-1741.",
        "Mauboussin, M. J., and P. Johnson. 1997. \"Competitive Advantage Period: The Neglected Value "
        "Driver.\" *Financial Management* 26 (2): 67-74.",
        "Newey, W. K., and K. D. West. 1987. \"A Simple, Positive Semi-Definite, Heteroskedasticity "
        "and Autocorrelation Consistent Covariance Matrix.\" *Econometrica* 55 (3): 703-708.",
        "Politis, D. N., and J. P. Romano. 1994. \"The Stationary Bootstrap.\" *Journal of the "
        "American Statistical Association* 89 (428): 1303-1313.",
        "Penman, S. H. 2011. *Accounting for Value.* New York: Columbia University Press.",
        "Rappaport, A., and M. J. Mauboussin. 2001. *Expectations Investing: Reading Stock Prices for "
        "Better Returns.* Boston: Harvard Business School Press.",
        "Rouwenhorst, K. G. 1999. \"Local Return Factors and Turnover in Emerging Stock Markets.\" "
        "*Journal of Finance* 54 (4): 1439-1464.",
        "Spearman, C. 1904. \"The Proof and Measurement of Association between Two Things.\" "
        "*American Journal of Psychology* 15 (1): 72-101.",
        "Williams, J. B. 1938. *The Theory of Investment Value.* Cambridge, MA: Harvard University "
        "Press.",
    ]
    for r in refs:
        A(r)
        A("")

    A("")
    A("### Appendix A -- Reproducibility")
    A("")
    A("Regenerate this manuscript: `python -m scripts.paper.build_paper_v3`. US artifacts: "
      "`reports/research_v2/correction_validation.json`, `reports/research_v3/"
      "{tri_paired_significance,tri_bm_residual,v3_validation_suites_broad,"
      "v3_mc_calibration_n50_3y,v3_mc_calibration_n50_5y}.json`. India: "
      "`reports/research_v3_india/india_v3_backtest_broad.json` (paired v1+v3, produced by "
      "`scripts/utilities/research_v3_india/india_v3_backtest.py`). Canada: "
      "`reports/research_v3_canada/canada_v3_backtest_broad.json` (paired v1+v3, produced by "
      "`scripts/utilities/research_v3_canada/canada_v3_backtest.py`, fundamentals via SEC EDGAR). "
      "The single-market provenance manuscripts are `reports/paper/` (pre-correction US) and "
      "`reports/paper_v2/` (corrected US).")

    out = base / "master_research_report_v3.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def render_docx(md_path: Path) -> Path | None:
    if shutil.which("pandoc") is None:
        print("  [docx] pandoc not found; skipping DOCX render")
        return None
    docx = md_path.with_suffix(".docx")
    # Professional styling: a restyled reference doc (serif body, navy headings, page numbers).
    ref = Path(__file__).resolve().parent / "reference.docx"
    if not ref.exists():
        try:
            from scripts.paper import make_reference_docx
            make_reference_docx.main()
        except Exception as exc:  # noqa: BLE001
            print(f"  [docx] reference-doc build skipped: {exc}")
    cmd = ["pandoc", str(md_path), "-o", str(docx), "--toc", "--toc-depth=2",
           "--resource-path", str(md_path.parent)]
    if ref.exists():
        cmd += ["--reference-doc", str(ref)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=md_path.parent)
        print(f"  [docx] wrote {docx}" + (" (styled)" if ref.exists() else ""))
        return docx
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"  [docx] render failed: {exc}")
        return None


def main() -> int:
    base = DEFAULT_BASE
    base.mkdir(parents=True, exist_ok=True)
    print("building paper v3 (three-market FAJ manuscript) ->", base)
    tables = build_tables_v3(base)
    print(f"  tables: {len(tables)} -> {', '.join(tables)}")
    us_figs = figures_v2.generate(base)
    xm_figs = figures_v3.generate(base)
    print(f"  figures: US {len(us_figs)} + cross-market {len(xm_figs)}")
    md = build_master_v3(base)
    print(f"  manuscript: {md}")
    render_docx(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
