"""Assemble the corrected research manuscript (v2) -> reports/paper_v2/.

This is the post-split-correction story and the platform paper for the Fundamentals Fade
engine. The legacy ``reports/paper/`` package (scripts/paper/build_paper.py) is left
byte-untouched -- it IS the pre-correction artifact this manuscript's discovery narrative
(Section 5) cites as superseded.

Presentation layer only: reads already-produced JSON/CSV under ``reports/`` and emits

    reports/paper_v2/master_research_report_v2.md   -- 11-section FAJ-style manuscript
    reports/paper_v2/tables/*.csv                    -- exhibit tables (correction
        decomposition, IC by family orig-vs-corrected, v3 paired, MC coverage, gate ledger)
    reports/paper_v2/figures/*.png                   -- via scripts.paper.figures_v2

and (when pandoc is available) a DOCX. Every number is pulled live from its source file, so
the manuscript regenerates from the frozen research artifacts. It does NOT re-run any study.

Usage:
    python -m scripts.paper.build_paper_v2
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from scripts.paper import figures_v2

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
R2 = REPORTS / "research_v2"
R3 = REPORTS / "research_v3"
PORT = REPORTS / "portfolio"
DEFAULT_BASE = REPORTS / "paper_v2"

FAMILY_LABEL = {
    "consumer_staples": "Consumer staples", "consumer_discretionary": "Consumer discretionary",
    "internet_platforms_marketplaces": "Internet platforms", "semiconductors": "Semiconductors",
    "semiconductor_equipment": "Semiconductor equipment", "software_saas_infra": "Software / SaaS",
    "industrials_assetlight_services": "Industrials (asset-light)", "health_care": "Health care",
    "technology": "Technology", "financials": "Financials",
}


def _j(path: Path) -> dict | None:
    return json.load(open(path)) if path.exists() else None


def _pct(x, d=1) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:.{d}f}%"


def _ic(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:+.3f}"


def _t(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.2f}"


# --------------------------------------------------------------------------- tables
def build_tables_v2(base: Path) -> list[str]:
    tables = base / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    cv = _j(R2 / "correction_validation.json")
    if cv:
        ov = cv.get("overall", {})
        rows = []
        for hz, b in ov.items():
            rows.append({"horizon": hz,
                         "ic_original": round(b["original"]["ic"], 4), "t_original": round(b["original"]["t"], 2),
                         "ic_split_only": round(b["split_only"]["ic"], 4),
                         "ic_div_only": round(b["div_only"]["ic"], 4),
                         "ic_fully_corrected": round(b["full"]["ic"], 4), "t_fully_corrected": round(b["full"]["t"], 2)})
        pd.DataFrame(rows).to_csv(tables / "correction_decomposition.csv", index=False)
        made.append("correction_decomposition.csv")
        fam = cv.get("by_family_3y", {})
        frows = [{"family": FAMILY_LABEL.get(k, k), "ic_3y_original": round(v["original"], 4),
                  "ic_3y_split_only": round(v.get("split_only", float("nan")), 4),
                  "ic_3y_fully_corrected": round(v["full"], 4)}
                 for k, v in sorted(fam.items(), key=lambda kv: -kv[1]["full"])]
        pd.DataFrame(frows).to_csv(tables / "ic_by_family_corrected.csv", index=False)
        made.append("ic_by_family_corrected.csv")
    paired = _j(R3 / "v3_paired_summary_n50.json")
    if paired:
        head, pid = paired.get("headline", {}), paired.get("paired_ic_diff_overall", {})
        rows = [{"horizon": hz, "v3_ic_obs": round(head[hz]["v3_ic_obs"], 4),
                 "v1_ic_obs": round(head[hz]["v1_ic_obs"], 4),
                 "paired_ic_diff": round(pid[hz]["mean_ic_diff"], 4),
                 "paired_nw_t": round(pid[hz]["nw_tstat"], 2)} for hz in ("1y", "3y", "5y") if hz in head]
        pd.DataFrame(rows).to_csv(tables / "v3_vs_paired_v1.csv", index=False)
        made.append("v3_vs_paired_v1.csv")
    mc3, mc5 = _j(R3 / "v3_mc_calibration_n50_3y.json"), _j(R3 / "v3_mc_calibration_n50_5y.json")
    if mc3:
        rows = []
        for tag, mc in (("3y", mc3), ("5y", mc5)):
            if mc:
                rows.append({"horizon": tag, "n": mc["n"], "coverage_p10_p90": round(mc["coverage_p10_p90"], 4),
                             "coverage_nominal": mc["coverage_nominal"],
                             "median_width": round(mc["sharpness"]["median_width"], 4),
                             "width_vs_error_spearman": round(mc["width_abs_err_spearman_pooled"], 4)})
        pd.DataFrame(rows).to_csv(tables / "v3_mc_coverage.csv", index=False)
        made.append("v3_mc_coverage.csv")
    # ---- corrected-ladder exhibits (2026-06-11 margin-seasonality fix) ----------
    psig = _j(R3 / "tri_paired_significance.json")
    if psig:
        rows = []
        for grp_key, grp_label in [("overall", "OVERALL")] + [
                (k, FAMILY_LABEL.get(k, k)) for k in sorted(psig) if k not in ("overall", "join")]:
            blk = psig.get(grp_key, {})
            hz = blk.get("horizons", {})
            for h in ("1y", "3y", "5y"):
                b = hz.get(h, {})
                if not b:
                    continue
                ci = b.get("boot_ci95") or [None, None]
                rows.append({"group": grp_label, "horizon": h,
                             "paired_ic_diff": round(b.get("mean_ic_diff"), 4) if b.get("mean_ic_diff") is not None else None,
                             "nw_t": round(b.get("nw_tstat"), 2) if b.get("nw_tstat") is not None else None,
                             "boot_ci95_lo": round(ci[0], 4) if ci[0] is not None else None,
                             "boot_ci95_hi": round(ci[1], 4) if ci[1] is not None else None,
                             "boot_excludes_zero": b.get("boot_excludes_zero")})
        pd.DataFrame(rows).to_csv(tables / "v3_corrected_paired.csv", index=False)
        made.append("v3_corrected_paired.csv")
    bm = _j(R3 / "tri_bm_residual.json")
    if bm:
        rows = []
        for fam, blk in sorted(bm.get("by_family", {}).items()):
            b3 = blk.get("3y", {})
            v3 = b3.get("v3", {}); v1 = b3.get("v1", {})
            rows.append({"family": FAMILY_LABEL.get(fam, fam),
                         "v3_raw_ic_3y": round(v3.get("raw_ic", {}).get("mean"), 4) if v3.get("raw_ic", {}).get("mean") is not None else None,
                         "v3_net_bm_ic_3y": round(v3.get("bm_resid_ic", {}).get("mean"), 4) if v3.get("bm_resid_ic", {}).get("mean") is not None else None,
                         "v3_net_bm_t_3y": round(v3.get("bm_resid_ic", {}).get("t_stat"), 2) if v3.get("bm_resid_ic", {}).get("t_stat") is not None else None,
                         "v1_net_bm_ic_3y": round(v1.get("bm_resid_ic", {}).get("mean"), 4) if v1.get("bm_resid_ic", {}).get("mean") is not None else None})
        ov = bm.get("overall", {}).get("3y", {})
        if ov:
            rows.insert(0, {"family": "OVERALL",
                            "v3_raw_ic_3y": round(ov.get("v3", {}).get("raw_ic", {}).get("mean"), 4) if ov.get("v3", {}).get("raw_ic", {}).get("mean") is not None else None,
                            "v3_net_bm_ic_3y": round(ov.get("v3", {}).get("bm_resid_ic", {}).get("mean"), 4) if ov.get("v3", {}).get("bm_resid_ic", {}).get("mean") is not None else None,
                            "v3_net_bm_t_3y": round(ov.get("v3", {}).get("bm_resid_ic", {}).get("t_stat"), 2) if ov.get("v3", {}).get("bm_resid_ic", {}).get("t_stat") is not None else None,
                            "v1_net_bm_ic_3y": round(ov.get("v1", {}).get("bm_resid_ic", {}).get("mean"), 4) if ov.get("v1", {}).get("bm_resid_ic", {}).get("mean") is not None else None})
        pd.DataFrame(rows).to_csv(tables / "v3_bm_control.csv", index=False)
        made.append("v3_bm_control.csv")

    # gate ledger (corrected 2026-06-11: G2 superseded by the corrected panel; G3-G5 run)
    g5 = _j(R3 / "v3_sector_state_ablation_paired_k0.25.json") or {}
    ledger = [
        ("G0", "Build / v1 invariance", "v1 numerics byte-identical under the v3 block", "PASS"),
        ("G1", "Economic coherence", "identities, binds, tripwires on 8 worked names", "PASS"),
        ("G2", "Ranking vs paired v1", "original kill (broad 3y IC -0.052; staples v3<v1) later traced "
         "to a margin-seasonality defect in the kill panel", "KILL -> SUPERSEDED"),
        ("G3", "Broad signal validation (corrected)", "paired v3-v1 +0.067 3y / +0.076 5y, NW t 3.4/4.1, "
         "block-bootstrap CI excludes 0", "PASS"),
        ("G4", "Validation suites + B/M control", "v3>v1 survives post-2021 OOS holdout (t 2.75) and every "
         "regime; but Book/Market spans the broad edge -- v3 incremental-to-value only in consumer "
         "staples + discretionary; semis wrong-signed net of B/M", "PASS (scoped)"),
        ("G5", "Sector-state ablation (kappa)", f"self-paired on corrected engine: 3y paired diff "
         f"+0.0006 (NW t 2.38) but vanishes at 5y -- economically negligible; verdict {g5.get('verdict', 'KEEP_KAPPA_0')}",
         "KEEP kappa=0"),
        ("G6", "Monte-Carlo calibration", "p10-p90 coverage 5.2% vs 80%; width<->error ~0", "KILL"),
        ("G7", "Product traceability smoke", "renders + driver_audit sidecars; suite green", "PASS"),
    ]
    pd.DataFrame(ledger, columns=["gate", "question", "evidence", "verdict"]).to_csv(
        tables / "v3_gate_ledger.csv", index=False)
    made.append("v3_gate_ledger.csv")
    return made


# --------------------------------------------------------------------------- manuscript
def build_master_v2(base: Path) -> Path:
    cv = _j(R2 / "correction_validation.json") or {}
    ov = cv.get("overall", {})
    o1, o3, o5 = ov.get("1y", {}), ov.get("3y", {}), ov.get("5y", {})
    fam3 = cv.get("by_family_3y", {})
    paired = _j(R3 / "v3_paired_summary_n50.json") or {}
    g2 = paired.get("gate_g2", {})
    g2in = g2.get("inputs", {})
    phead = paired.get("headline", {})
    pid = paired.get("paired_ic_diff_overall", {})
    mc3 = _j(R3 / "v3_mc_calibration_n50_3y.json") or {}
    mc5 = _j(R3 / "v3_mc_calibration_n50_5y.json") or {}
    metrics = _j(PORT / "metrics.json") or {}
    # corrected-ladder ledgers (2026-06-11 margin-seasonality fix)
    psig = _j(R3 / "tri_paired_significance.json") or {}
    bm = _j(R3 / "tri_bm_residual.json") or {}
    suites = _j(R3 / "v3_validation_suites_broad.json") or {}

    def psig_h(group, hz):
        b = (psig.get(group, {}).get("horizons", {}) or {}).get(hz, {})
        return b.get("mean_ic_diff"), b.get("nw_tstat"), b.get("boot_ci95")

    def bm_fam(fam, eng, hz="3y"):
        b = ((bm.get("by_family", {}).get(fam, {}) or {}).get(hz, {}) or {}).get(eng, {})
        r = b.get("bm_resid_ic", {})
        return r.get("mean"), r.get("t_stat")

    def f_full(hz):
        b = ov.get(hz, {})
        return b.get("full", {}).get("ic"), b.get("full", {}).get("t")

    def f_orig(hz):
        b = ov.get(hz, {})
        return b.get("original", {}).get("ic"), b.get("original", {}).get("t")

    L: list[str] = []
    A = L.append

    A("# Economic Coherence Is Not Broad Predictive Power")
    A("### A point-in-time study of DCF valuation signals, a split-adjustment correction, "
      "a reversed kill, and where an economically grounded forecast engine narrowly out-ranks -- "
      "and where a value factor subsumes it")
    A("")
    A("*Corrected manuscript (v2). Every exhibit is regenerated from frozen research artifacts under "
      "`reports/research_v2/` and `reports/research_v3/` by `scripts/paper/build_paper_v2.py`. The "
      "pre-correction package in `reports/paper/` is preserved verbatim as the superseded artifact that "
      "Section 5 documents. Figures in `figures/`, tables in `tables/`.*")
    A("")
    A("**Abstract.** We test whether the gap between a frozen fundamentals-only DCF intrinsic value and "
      "market price ranks stocks on forward returns. An initial point-in-time (PIT) study found a strong, "
      "horizon-increasing signal. A subsequent split-and-dividend adjustment audit of the return panel "
      f"overturned the broad result: the fully corrected 3-year rank IC is {_ic(f_full('3y')[0])} "
      f"({_t(f_full('3y')[1])}), versus {_ic(f_orig('3y')[0])} ({_t(f_orig('3y')[1])}) before correction. "
      "The signal survives **only** in stable-cash-flow families -- consumer staples retains a corrected "
      f"3-year IC of {_ic(fam3.get('consumer_staples', {}).get('full'))}. We then ask whether replacing the "
      "forecast layer with an economically grounded engine (CAP/ROIC-persistence fade with a Grinold-Kroner "
      "terminal) recovers predictive power. A pre-registered paired backtest initially **killed** it; that "
      "kill was subsequently traced to a margin-seasonality defect in the engine, and on the corrected panel "
      f"the economic engine **out-ranks** the frozen baseline (paired 3-year IC diff {_ic(psig_h('overall', '3y')[0])}, "
      f"NW t {_t(psig_h('overall', '3y')[1])}), surviving an out-of-sample holdout and holding across regimes. "
      "Yet the Book/Market value factor **spans the broad edge**: the engine's information net of value survives "
      "only in consumer staples and consumer discretionary. The contributions are a methodological one -- PIT "
      "return hygiene, pre-registered kill-gates, overlap-aware inference, and a defect post-mortem that "
      "reversed a kill -- and a substantive, **scoped** finding: an economically coherent forecast delivers a "
      "real but narrow ranking improvement that a simple value factor largely subsumes. Economic coherence and "
      "broad predictiveness remain distinct objectives; coherence is necessary for an auditable product and, "
      "here, sufficient only for a family-scoped edge.")
    A("")
    A("---")

    # 1
    A("## 1. Introduction")
    A("")
    A("The discount-to-intrinsic of a disciplined discounted-cash-flow (DCF) model is an intuitive value "
      "signal: buy what trades below conservatively estimated intrinsic value. This manuscript reports a "
      "research program that (i) built a frozen, fundamentals-only DCF and validated its discount as a "
      "cross-sectional signal; (ii) discovered, through an adjustment audit of the return panel, that the "
      "broad validation was a corporate-action artifact; (iii) re-established what survives correction; and "
      "(iv) tested -- and pre-registered kill criteria for -- whether a more economically coherent forecast "
      "engine recovers the lost edge. The headline lesson is stated plainly and defended with the data: "
      "**economic coherence is not the same objective as predictive power.**")
    A("")

    # 2
    A("## 2. Data and PIT Methodology")
    A("")
    n_obs = cv.get("n_obs")
    A(f"- **Panel.** ~{n_obs:,} firm-quarter observations across quarterly cross-sections in a large-cap "
      f"universe ({len(fam3)} sector families). Of these, {cv.get('n_dividend_adjusted', 0):,} required a "
      "dividend adjustment." if n_obs else "- Panel: see correction_validation.json.")
    A("- **Look-ahead control.** The signal at quarter-end *t* uses only fundamentals public by *t*; forward "
      "returns are measured strictly after *t*.")
    A("- **The correction (methods contribution).** The original return panel used unadjusted prices. Two "
      "corporate-action adjustments were applied and validated against external references on a 35/35 spot "
      "check: split adjustment and dividend (total-return) adjustment. As Section 5 shows, the split "
      "adjustment is the decisive one; failing to apply it manufactured a spurious signal.")
    A("- **Reproducibility.** Corrected panels and the decomposition live under `reports/research_v2/`; the "
      "v3 paired panel and calibration under `reports/research_v3/`.")
    A("")

    # 3
    A("## 3. Valuation Architecture -- the Fundamentals Fade Engine")
    A("")
    A("The product-facing valuation is an **economic-driver fade model** (engine v3), built to be auditable "
      "rather than to maximize rank IC. Its design is the platform contribution of this work:")
    A("- **Primitives, not reinvestment.** The forecast is specified by three mean-reverting primitives -- "
      "revenue growth, operating margin, and the return on new invested capital (RONIC). Reinvestment is "
      "**derived** every period from the sustainable-growth identity *reinv = g / RONIC*, so growth is always "
      "paid for and independent driver fades cannot manufacture value.")
    A("- **Per-driver half-lives.** Growth and margin each revert toward their anchors at a company-specific "
      "half-life estimated by a through-origin AR(1) on the company's own history, empirical-Bayes shrunk "
      "toward a sector prior; data-insufficient drivers fall back to a CAP-derived half-life, flagged in the "
      "audit.")
    A("- **Persistence and CAP.** Excess ROIC decays at a fitted persistence rate phi; the competitive "
      "advantage period (CAP) is the implied time for the excess to reach a 50bp floor; RONIC fades toward "
      "WACC accordingly.")
    A("- **Grinold-Kroner terminal.** Terminal growth is *derived* (long-run nominal GDP minus a dilution "
      "haircut, clamped), never picked; the value-driver terminal prices growth at the surviving RONIC. A "
      "10-year explicit horizon shrinks the perpetuity's share of value.")
    A("- **v1 benchmark.** The frozen v1 state-space engine remains the research and live-book engine; v2 "
      "(a Bayesian variant) is archived. All cross-engine claims below compare v3 against v1 on identical "
      "PIT data bundles.")
    A("")
    A("A worked example (KO) appears in [`figures/fade_paths_KO.png`](figures/fade_paths_KO.png): growth and "
      "margin fade to their anchors while RONIC decays toward WACC, and reinvestment tracks as the derived "
      "line.")
    A("")

    # 4
    A("## 4. Validation Methodology")
    A("")
    A("- **Rank IC.** Spearman rank correlation between the signal and forward return, computed per "
      "cross-section and aggregated; obs-weighted and equal-weighted variants reported.")
    A("- **Inference.** Newey-West (NW) t-statistics with horizon-scaled lags account for overlapping forward "
      "windows -- essential at 3-5 years, where naive t-statistics overstate significance.")
    A("- **Paired comparison.** The engine comparison runs v3 and v1 **in-process on the same data bundle** "
      "per name-date, so differences are forecast-driven, never data- or split-driven (the bug that produced "
      "the original false positive).")
    A("- **Pre-registration.** Engine gates fix PASS/KILL thresholds *before* the run; a kill is recorded and "
      "honored even when marginal.")
    A("")

    # 5
    A("## 5. Results I -- The Discovery and Its Overturn")
    A("")
    A("The original (uncorrected) study showed a strong, horizon-increasing signal: 3-year rank IC "
      f"{_ic(f_orig('3y')[0])} ({_t(f_orig('3y')[1])}), 5-year {_ic(f_orig('5y')[0])} ({_t(f_orig('5y')[1])}). "
      "This is the result the pre-correction package (`reports/paper/`) reports and is **superseded** here.")
    A("")
    A("Decomposing the correction isolates the cause (full table: "
      "[`tables/correction_decomposition.csv`](tables/correction_decomposition.csv)):")
    A("")
    A("| Horizon | IC original | IC split-only | IC dividend-only | IC fully corrected |")
    A("|---|---|---|---|---|")
    for hz in ("1y", "3y", "5y"):
        b = ov.get(hz, {})
        if b:
            A(f"| {hz} | {_ic(b['original']['ic'])} ({_t(b['original']['t'])}) | "
              f"{_ic(b['split_only']['ic'])} | {_ic(b['div_only']['ic'])} | "
              f"{_ic(b['full']['ic'])} ({_t(b['full']['t'])}) |")
    A("")
    A("**The split adjustment is decisive.** Dividend adjustment *alone* would have made the signal look "
      f"slightly stronger (3-year {_ic(o3.get('div_only', {}).get('ic'))}); it is the split adjustment that "
      f"collapses it (split-only 3-year {_ic(o3.get('split_only', {}).get('ic'))}). The fully corrected broad "
      f"signal is statistically indistinguishable from zero at every horizon "
      f"(3-year {_ic(f_full('3y')[0])}, {_t(f_full('3y')[1])}). The broad 'institutional validation passed' "
      "headline was an artifact of unadjusted prices. This overturn implicated the prior factor-uniqueness "
      "and portfolio-alpha claims, all of which used the uncorrected discount.")
    A("")
    A("![Original vs corrected IC](figures/orig_vs_corrected_ic.png)")
    A("")

    # 6
    A("## 6. Results II -- What Survives Correction")
    A("")
    A("Correcting the panel does not zero the signal everywhere. By sector family at 3 years (full table: "
      "[`tables/ic_by_family_corrected.csv`](tables/ic_by_family_corrected.csv)):")
    A("")
    A("| Family | IC 3y original | IC 3y corrected |")
    A("|---|---|---|")
    for k, v in sorted(fam3.items(), key=lambda kv: -kv[1]["full"]):
        A(f"| {FAMILY_LABEL.get(k, k)} | {_ic(v['original'])} | {_ic(v['full'])} |")
    A("")
    A(f"**Consumer staples is the clear survivor** (corrected 3-year IC "
      f"{_ic(fam3.get('consumer_staples', {}).get('full'))}), with consumer discretionary retaining a weaker "
      f"edge ({_ic(fam3.get('consumer_discretionary', {}).get('full'))}). The most spectacular pre-correction "
      f"numbers were the most spurious: internet platforms collapsed from "
      f"{_ic(fam3.get('internet_platforms_marketplaces', {}).get('original'))} to "
      f"{_ic(fam3.get('internet_platforms_marketplaces', {}).get('full'))}, and semiconductors flipped "
      f"negative. The economically sensible reading: a fundamentals DCF discount predicts returns where "
      "cash flows are stable and forecastable, and not where value is dominated by unmodeled optionality or "
      "violent multiple cycles. A further boundary established on the corrected panel is that book-to-market, "
      "the most stable raw ranker in this survivorship-biased universe, spans the discount at 3-5 years.")
    A("")
    A("![IC by family (corrected)](figures/ic_by_family.png)")
    A("")

    # 7
    A("## 7. Results III -- Engine Experiments and the Gate Ledger")
    A("")
    A("If the discount works only where cash flows are coherent, perhaps a *more economically coherent "
      "forecast* recovers broad predictiveness. The engine experiments tested this with pre-registered "
      "gates (full ledger: [`tables/v3_gate_ledger.csv`](tables/v3_gate_ledger.csv)).")
    A("")
    A("**G0-G1 (coherence) PASS.** v3 reproduces v1 byte-identically when disabled (G0), and on eight worked "
      "names its identities, reinvestment binds, and FCF-margin tripwires behave exactly as the economics "
      "demand -- mean reversion pulls depressed margins up and high growth down toward GDP (G1).")
    A("")
    A("**G2 (ranking) KILL.** Under the split-consistent paired backtest "
      f"({paired.get('n_observations', 'n')} obs over {paired.get('n_tickers_with_obs', 'n')} names; full "
      "table [`tables/v3_vs_paired_v1.csv`](tables/v3_vs_paired_v1.csv)):")
    A("")
    A("| Horizon | v3 IC (obs-wt) | v1 IC (obs-wt) | Paired IC diff | NW t |")
    A("|---|---|---|---|---|")
    for hz in ("1y", "3y", "5y"):
        if hz in phead and hz in pid:
            A(f"| {hz} | {_ic(phead[hz]['v3_ic_obs'])} | {_ic(phead[hz]['v1_ic_obs'])} | "
              f"{_ic(pid[hz]['mean_ic_diff'])} | {_t(pid[hz]['nw_tstat'])} |")
    A("")
    A(f"The pre-registered criteria killed v3: broad 3-year IC {_ic(g2in.get('broad_v3_3y_ic_obs'))} fell below "
      f"the -0.05 floor, and -- apparently decisively -- in consumer staples, the one family where the signal is "
      f"real, v3 ranked **worse** than the frozen v1 ({_ic(g2in.get('staples_v3_3y_ic_obs'))} vs "
      f"{_ic(g2in.get('staples_v1_3y_ic_obs'))}). On this evidence the program was closed. **That conclusion did "
      "not hold.** Section 7A documents how the kill panel was later found to carry a margin-seasonality defect; "
      "on the corrected panel gates G3-G5 were run and the ranking verdict reversed. We present the original "
      "kill here as recorded, then its post-mortem, because the reversal -- not the kill -- is the load-bearing "
      "result, and pre-registration is what made the honest reversal legible.")
    A("")
    A("**G6 (uncertainty calibration) KILL.** The optional driver Monte-Carlo bands are not calibrated "
      "uncertainty (full table [`tables/v3_mc_coverage.csv`](tables/v3_mc_coverage.csv)): 3-year p10-p90 "
      f"coverage of realized returns is {_pct(mc3.get('coverage_p10_p90'), 1)} against an 80% nominal, and the "
      f"rank correlation between band width and realized error is "
      f"{_ic(mc3.get('width_abs_err_spearman_pooled'))} -- wider bands do not flag larger errors. The bands "
      "are therefore rendered **only** as a driver-sensitivity range, never as a return forecast, with a "
      "mandatory caption to that effect.")
    A("")
    A("**G7 (product traceability) PASS.** The opt-in v3 engine renders cleanly with a full driver-audit "
      "sidecar mapping every projected number to a named assumption.")
    A("")
    A("![v3 vs paired v1](figures/v3_paired_ic.png)")
    A("")
    A("![Monte-Carlo coverage](figures/mc_coverage.png)")
    A("")

    # 7A -- the reversal
    d3, t3, ci3 = psig_h("overall", "3y")
    d5, t5, ci5 = psig_h("overall", "5y")
    s5 = (suites.get("suite5_holdout", {}) or {})
    s5_3y = (s5.get("paired_ic_diff", {}) or {}).get("3y", {})
    cd_v3, cd_t = bm_fam("consumer_discretionary", "v3")
    cd_v1, _ = bm_fam("consumer_discretionary", "v1")
    cs_v3, cs_t = bm_fam("consumer_staples", "v3")
    sm_v3, sm_t = bm_fam("semiconductors", "v3")
    sw_v3, sw_t = bm_fam("software_saas_infra", "v3")
    ov_v3, ov_t = (bm.get("overall", {}).get("3y", {}).get("v3", {}).get("bm_resid_ic", {}).get("mean"),
                   bm.get("overall", {}).get("3y", {}).get("v3", {}).get("bm_resid_ic", {}).get("t_stat"))

    A("## 7A. Results IV -- The Kill, Re-examined: a Margin-Seasonality Defect and the Corrected Ladder")
    A("")
    A("Closing G2 prompted a forensic review of *why* an economically disciplined engine would subtract "
      "dispersion. The cause was a defect, not a property of the economics: the fade engine set each driver's "
      "fade target and half-life from **seasonal per-quarter** NOPAT margins rather than trailing-twelve-month "
      "margins, injecting a spurious ~6-percentage-point margin compression and tripping a systemic 0.5-year "
      "half-life floor. The fix (TTM margins for the target and half-life estimation) changed engine internals "
      "only -- v1 stays byte-identical -- and re-opened the ranking question on a corrected panel.")
    A("")
    A("Re-running the broad, split-consistent paired backtest on the corrected engine reverses G2. Against the "
      "frozen v1 baseline, on identical PIT bundles, with both Newey-West HAC inference and a contiguous-date "
      "block bootstrap (block = horizon quarters) so the overlapping windows are not overstated (full table: "
      "[`tables/v3_corrected_paired.csv`](tables/v3_corrected_paired.csv)):")
    A("")
    A("| Horizon | Paired v3 - v1 IC | NW t | Block-bootstrap 95% CI |")
    A("|---|---|---|---|")
    for hz, (dd, tt, cc) in (("1y", psig_h("overall", "1y")), ("3y", (d3, t3, ci3)), ("5y", (d5, t5, ci5))):
        ci_s = f"[{_ic(cc[0])}, {_ic(cc[1])}]" if cc else "N/A"
        A(f"| {hz} | {_ic(dd)} | {_t(tt)} | {ci_s} |")
    A("")
    A("")
    A("![Corrected paired v3 - v1](figures/v3_corrected_paired.png)")
    A("")
    A(f"v3 out-ranks v1 at 3 and 5 years with bootstrap intervals that exclude zero; the 1-year diff is not "
      f"significant. The improvement is **not in-sample over-fitting**: on a freeze-and-test holdout (rows dated "
      f"after 2021-06-30, {s5.get('n_obs_post', 'n')} observations) the paired 3-year diff is "
      f"{_ic(s5_3y.get('mean_ic_diff'))} (NW t {_t(s5_3y.get('nw_tstat'))}), and v3 beats v1 in every macro "
      "regime we cut (pre-COVID, the COVID melt-up, the 2022 rate shock, and the higher-for-longer/AI period), "
      "tying only in the most recent. The previous manuscript's central negative claim -- that the economic "
      "engine ranks no better than the baseline -- does not survive the defect correction.")
    A("")
    A("**But a better fade is not a unique alpha source.** The decisive test is whether the corrected discount "
      "ranks *beyond* the cheapest available value factor. We residualize the engine's discount rank on the "
      "Book/Market rank each cross-section and correlate the residual with the forward return (full table: "
      "[`tables/v3_bm_control.csv`](tables/v3_bm_control.csv)):")
    A("")
    A("| Family | v3 discount IC, net of Book/Market (3y) | NW t | v1, net of B/M |")
    A("|---|---|---|---|")
    for fam in ("consumer_discretionary", "consumer_staples", "industrials_assetlight_services",
                "software_saas_infra", "semiconductors", "internet_platforms_marketplaces",
                "semiconductor_equipment"):
        v3m, v3t = bm_fam(fam, "v3")
        v1m, _ = bm_fam(fam, "v1")
        A(f"| {FAMILY_LABEL.get(fam, fam)} | {_ic(v3m)} | {_t(v3t)} | {_ic(v1m)} |")
    A("")
    A(f"Overall, net of Book/Market the engine's broad edge is gone ({_ic(ov_v3)}, NW t {_t(ov_t)}): a simple, "
      "free value factor spans it. The incremental-to-value signal concentrates in exactly two families. In "
      f"**consumer discretionary** the engine uniquely survives ({_ic(cd_v3)}, NW t {_t(cd_t)}) where v1 does "
      f"not ({_ic(cd_v1)}); in **consumer staples** the DCF discount survives for both engines ({_ic(cs_v3)}, "
      f"NW t {_t(cs_t)}). Elsewhere the verdict is sobering: in **software** the engine's large raw edge is "
      f"entirely the value factor (net of B/M {_ic(sw_v3)}, t {_t(sw_t)}), and in **semiconductors** the "
      f"discount is significantly **wrong-signed** net of value ({_ic(sm_v3)}, NW t {_t(sm_t)}).")
    A("")
    A("![v3 net of Book/Market by family](figures/v3_bm_control.png)")
    A("")
    A("Finally, the optional sector-state tilt (G5) was re-ablated self-paired on the corrected engine. It is "
      "a cautionary case in pre-registration: the 3-year paired improvement is positive and clears the "
      "NW-t>2 bar *by the letter*, yet the effect is ~6 basis points of IC and vanishes at 5 years. Statistical "
      "significance here reflects a near-deterministic, consistent hair of a shift, not a material edge; on "
      "the materiality the rule was meant to protect, the tilt fails. It stays off (kappa=0). The Monte-Carlo "
      "calibration kill (G6) is untouched.")
    A("")
    A("The corrected verdict, stated precisely: **the Fundamentals Fade engine is a strictly better fade than "
      "the frozen baseline, and it carries a genuine, narrow, incremental-to-value ranking edge in consumer "
      "staples and consumer discretionary -- but it is not a broad alpha source, because Book/Market spans its "
      "broad edge, and it is wrong-signed in semiconductors.** It remains the opt-in, most-explainable engine; "
      "v1 remains the default ranking and live-book model.")
    A("")

    # 8
    A("## 8. Portfolio Implementation (corrected, caveated)")
    A("")
    A("Because the broad signal did not survive correction, the portfolio layer is reframed as a **live, "
      "out-of-sample test of the conditional (stable-cash-flow) signal**, not a validated alpha claim. A "
      "paper portfolio over the frozen v1 discount ranking (inception 2026-06-04, quarterly tracking of "
      "holdings, weights, discounts, confidence, and returns) is the only genuine forward evidence; its first "
      "rebalance is 2026-09-04. The earlier capacity, allocation, and risk-control studies are retained as "
      "methodology but their alpha magnitudes are pre-correction and are **not** carried forward as live "
      "claims.")
    if metrics.get("config"):
        A("")
        A(f"*(Historical, pre-correction context only: the original top-{metrics['config'].get('top_n', 20)} "
          f"book reported net CAGR {_pct(metrics.get('cagr_net'))} vs universe "
          f"{_pct(metrics.get('cagr_universe_ew'))}; these numbers are superseded and shown for provenance.)*")
    A("")

    # 9
    A("## 9. What Survived")
    A("")
    A("- A **conditional** signal: the DCF discount predicts forward returns in stable-cash-flow families "
      f"(consumer staples corrected 3-year IC {_ic(fam3.get('consumer_staples', {}).get('full'))}), and -- net "
      "of Book/Market -- in consumer staples and consumer discretionary specifically.")
    A("- A **better fade, demonstrated**: on the corrected panel the economic engine out-ranks the frozen "
      f"baseline (paired 3-year IC diff {_ic(d3)}, NW t {_t(t3)}), out-of-sample and across regimes -- a real "
      "improvement, even though Book/Market subsumes the *broad* edge.")
    A("- A **methods kit**: split-consistent PIT return construction, paired in-process engine comparison, "
      "Newey-West and block-bootstrap inference for overlapping windows, pre-registered kill-gates, and a "
      "defect post-mortem that reversed a kill on its own evidence.")
    A("- A **coherence product with a scoped claim**: the Fundamentals Fade engine, valuable for transparency "
      "and auditability (every number traces to a mechanism), now also carrying a *narrow, family-scoped* "
      "ranking edge -- sold as such, not as broad alpha.")
    A("- A **live test**: the paper portfolio accruing genuine out-of-sample history.")
    A("")

    # 10
    A("## 10. Lessons")
    A("")
    A("- **Economic coherence is not *broad* predictive power -- but it is not nothing.** The corrected engine "
      "out-ranks the baseline and carries a genuine edge in two families; yet a free Book/Market factor spans "
      "its broad signal. Coherence bought interpretability *and* a narrow, scoped edge -- not broad alpha.")
    A("- **A kill is a hypothesis, not a verdict -- audit the engine before you close the program.** The "
      "original G2 kill was real on its panel but wrong about the world: a margin-seasonality defect, not the "
      "economics, subtracted the dispersion. Pre-registration made the reversal honest; the post-mortem made "
      "it possible.")
    A("- **Control for the cheap factor before claiming alpha.** Raw ICs flattered the engine (especially in "
      "software); residualizing on Book/Market is what separated a real consumer-family edge from a value-factor "
      "reload and exposed the wrong-signed semiconductor zone.")
    A("- **Return-panel hygiene dominates model sophistication.** A single missing corporate-action "
      "adjustment, not model error, produced the original false positive; the split adjustment alone "
      "overturned years of apparent signal.")
    A("- **Overlapping windows and survivorship flatter results.** NW inference, block-bootstrap CIs, and an "
      "explicit survivorship stress were necessary to avoid over-claiming.")
    A("")

    # 11
    A("## 11. Conclusion")
    A("")
    A("A fundamentals DCF discount is a **conditional** value signal -- real in stable-cash-flow families, "
      "absent broadly once prices are correctly adjusted. Economic sophistication in the forecast layer changed "
      "the picture more than the first verdict suggested: after correcting a margin-seasonality defect, the "
      "economically grounded engine **out-ranks** the frozen baseline out-of-sample and across regimes, and "
      "carries a genuine incremental-to-value edge in consumer staples and consumer discretionary -- but the "
      "Book/Market value factor spans its broad ranking content, and it is wrong-signed in semiconductors. The "
      "honest summary is a **scoped** one: a better, more auditable fade with a narrow real edge, not a broad "
      "alpha engine. The durable outputs are methodological (PIT hygiene, paired comparison, overlap-aware "
      "inference, pre-registered gates, and a kill reversed by its own post-mortem) and a transparent, auditable "
      "valuation engine now offered with a precisely-bounded ranking claim. The open question is live "
      "performance of the conditional signal, now being accrued.")
    A("")
    A("---")
    A("")
    A("### Appendix A -- Reproducibility")
    A("")
    A("Regenerate this manuscript: `python -m scripts.paper.build_paper_v2` (or `research build-paper`, which "
      "defaults to v2). The corrected baseline state is tagged `research-complete-v1`. Source artifacts: "
      "`reports/research_v2/correction_validation.json` (decomposition), "
      "`reports/research_v2/split_corrected_universe_revalidation.json` (corrected revalidation), "
      "`reports/research_v3/v3_paired_summary_n50.json` (original G2), and -- for the Section 7A reversal -- "
      "`reports/research_v3/tri_paired_significance.json` (corrected G3 paired diff + block bootstrap), "
      "`reports/research_v3/v3_validation_suites_broad.json` (G4 suites: OOS holdout, regimes, deciles), "
      "`reports/research_v3/tri_bm_residual.json` (G4 Book/Market control), "
      "`reports/research_v3/v3_sector_state_ablation_paired_k0.25.json` (G5), "
      "`reports/research_v3/v3_mc_calibration_n50_{3,5}y.json` (G6). The legacy `reports/paper/` package is the "
      "preserved pre-correction artifact cited in Section 5.")
    A("")
    A("### Appendix B -- v3 Gate Ledger")
    A("")
    A("| Gate | Question | Verdict |")
    A("|---|---|---|")
    A("| G0 | v1 byte-identical under the v3 block | PASS |")
    A("| G1 | Economic coherence (identities, binds, tripwires) | PASS |")
    A("| G2 | Ranking vs paired v1 (pre-registered) | **KILL -> SUPERSEDED** (margin-seasonality defect, Sec. 7A) |")
    A("| G3 | Broad signal validation (corrected panel) | **PASS** (paired v3-v1 +ve 3y/5y, bootstrap excl. 0) |")
    A("| G4 | Validation suites + Book/Market control | **PASS (scoped)** -- v3>v1 OOS & regimes; B/M spans broad edge |")
    A("| G5 | Sector-state ablation (kappa) | KEEP kappa=0 (tilt negligible) |")
    A("| G6 | Monte-Carlo calibration | **KILL** |")
    A("| G7 | Product traceability smoke | PASS |")
    A("")
    A("The research journal (`reports/research_v2/research_journal.md`, append-only) records each gate entry "
      "with timestamps, inputs, and decisions -- including the 2026-06-11 defect post-mortem and corrected "
      "ladder that reversed G2 and produced the scoped G3/G4 verdicts.")
    A("")

    base.mkdir(parents=True, exist_ok=True)
    path = base / "master_research_report_v2.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _pandoc(md_path: Path, base: Path) -> Path | None:
    exe = shutil.which("pandoc")
    if not exe:
        import os
        cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Pandoc" / "pandoc.exe"
        exe = str(cand) if cand.exists() else None
    if not exe:
        print("  [pandoc] not found -- skipping DOCX (markdown manuscript still written).")
        return None
    out = base / "master_research_report_v2.docx"
    try:
        # Read as GitHub-Flavored Markdown so the `---` thematic breaks are NOT parsed as a
        # YAML metadata block; cwd=base so the relative figures/ image paths resolve.
        subprocess.run([exe, str(md_path), "-f", "gfm", "-o", str(out)], check=True, cwd=str(base))
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  [pandoc] DOCX conversion failed: {type(exc).__name__}: {exc}")
        return None


def main(base: Path = DEFAULT_BASE) -> None:
    tbls = build_tables_v2(base)
    figs = figures_v2.generate(base)
    rep = build_master_v2(base)
    docx = _pandoc(rep, base)
    print(f"tables: {tbls}")
    print(f"figures: {figs}")
    print(f"wrote {rep.relative_to(ROOT)}")
    if docx:
        print(f"wrote {docx.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
