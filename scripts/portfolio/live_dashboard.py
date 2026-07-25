"""Live portfolio dashboard over the frozen paper-portfolio book.

One command turns the frozen inception book (reports/portfolio/paper_portfolio.json, created by
paper_portfolio.py) plus the accruing track history into a durable, human-readable record under
reports/live/: holdings, returns vs SPY/QQQ, sector weights, per-name attribution and -- on demand
or at a quarterly rebalance -- current intrinsic / discount.

Commands:
    python -m scripts.portfolio.live_dashboard                      # = dashboard (fast, seconds)
    python -m scripts.portfolio.live_dashboard dashboard
    python -m scripts.portfolio.live_dashboard refresh-intrinsics   # re-score holdings (~1-3 min)
    python -m scripts.portfolio.live_dashboard rebalance --dry-run   # preview the next rebalance
    python -m scripts.portfolio.live_dashboard rebalance             # execute (quarterly)

Headline book = discount_x_conf_top10 (confidence-weighted top-10); the other three books are
reported as secondary summary rows. This is a REPORTING layer -- portfolio construction and the
DCF signal are unchanged. Return math matches paper_portfolio.py: equal-weight buy-and-hold =
mean of per-name (current / entry - 1).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # scripts/portfolio -> _live_common
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # src -> dcf_project
from _live_common import (  # noqa: E402
    BENCHMARKS,
    PORT_DIR,
    REPORT_DIR,
    SNAP,
    STATE,
    _live_prices,
    _rank_books,
)

# Corrected-panel ranking verdict (lightweight JSON-only import; no numpy/pandas). Guarded so the
# dashboard still runs if the research ledgers are absent. Construction is UNCHANGED -- this is a
# scoped, read-only overlay on the frozen v1 book (v1 stays the live signal; the fade engine's
# incremental-to-value edge is consumer-family-only, per the 2026-06-11 corrected ladder).
try:
    from dcf_project.research import validation_evidence as ve  # noqa: E402
    _VE_OK = ve.available()
except Exception:  # noqa: BLE001
    ve = None  # type: ignore[assignment]
    _VE_OK = False

HEADLINE = "discount_x_conf_top10"
LIVE_DIR = REPORT_DIR / "live"
TRACK = PORT_DIR / "paper_portfolio_track.json"
INTRINSICS = LIVE_DIR / "intrinsics.json"
REBAL_LOG = LIVE_DIR / "rebalance_log.json"
ALPHA_MIN_PERIODS = 8  # below this we report simple excess return; above, regression alpha

SECTOR_LABEL = {
    "software_saas_infra": "Software/SaaS",
    "internet_platforms_marketplaces": "Internet/Marketplace",
    "semiconductors": "Semiconductors",
    "semiconductor_equipment": "Semicap",
    "consumer_staples": "Consumer Staples",
    "consumer_discretionary": "Consumer Disc.",
    "industrials_assetlight_services": "Industrials (AL)",
}


# ----------------------------- small formatters / io -----------------------------
def _pct(x, dp: int = 1, signed: bool = False) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:+.{dp}f}%" if signed else f"{x * 100:.{dp}f}%"


def _usd(x) -> str:
    return "N/A" if x is None else f"${x:,.2f}"


def _today() -> str:
    return dt.date.today().isoformat()


def _load_json(p: Path, default=None):
    return json.load(open(p)) if p.exists() else default


def _sector_name(s: str) -> str:
    return SECTOR_LABEL.get(s, (s or "unknown").replace("_", " ").title())


# ----------------------------- return math (matches paper_portfolio) -----------------------------
def _book_return(book: list[dict], cur: dict[str, float]) -> float | None:
    """Equal-weight buy-and-hold book return = mean of per-name (current/entry - 1)."""
    rets = [cur[r["ticker"]] / r["entry_price"] - 1.0
            for r in book
            if r.get("entry_price") and cur.get(r["ticker"]) and r["entry_price"] > 0]
    return sum(rets) / len(rets) if rets else None


def _bench_returns(state: dict, cur: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for bm in BENCHMARKS:
        e, c = state["benchmark_entry"].get(bm), cur.get(bm)
        out[bm] = (c / e - 1.0) if (e and c and e > 0) else None
    return out


def _headline_rows(book: list[dict], cur: dict[str, float], intr: dict | None) -> list[dict]:
    """Per-name detail for the headline book. Contributions sum to the equal-weight book return."""
    priced = [r for r in book if r.get("entry_price") and cur.get(r["ticker"]) and r["entry_price"] > 0]
    n_valid = len(priced)
    growth = {r["ticker"]: cur[r["ticker"]] / r["entry_price"] for r in priced}  # drifted value factor
    denom = sum(growth.values())
    intr_h = (intr or {}).get("holdings", {}) if intr else {}
    rows = []
    for r in book:
        t = r["ticker"]
        e, c = r.get("entry_price"), cur.get(t)
        ret = (c / e - 1.0) if (e and c and e > 0) else None
        ci = intr_h.get(t) or {}
        rows.append({
            "ticker": t,
            "sector": r.get("sector"),
            "entry_price": e,
            "current_price": c,
            "return": ret,
            "weight_now": (growth[t] / denom) if (t in growth and denom) else None,
            "contribution": (ret / n_valid) if (ret is not None and n_valid) else None,
            "inception_intrinsic": r.get("intrinsic"),
            "inception_discount": r.get("discount"),
            "current_intrinsic": ci.get("intrinsic"),
            "current_discount": ci.get("discount"),
        })
    # most-attractive (highest inception discount) first
    rows.sort(key=lambda x: (x["inception_discount"] is not None, x["inception_discount"]), reverse=True)
    return rows


def _ranking_overlay(rows: list[dict]) -> dict | None:
    """Bucket the current (drifted) headline book by the fade engine's corrected-panel ranking
    tier per family. Read-only context: tells the reader what share of the v1 book sits in
    families where the fade engine adds incremental-to-value ranking information (consumer
    staples / discretionary), versus families where the discount is spanned by Book/Market or
    wrong-signed. Does NOT change construction. Returns None if the ledgers are unavailable."""
    if not _VE_OK or ve is None:
        return None
    tiers: dict[str, float] = {}
    per_name: list[dict] = []
    for r in rows:
        fam = r.get("sector")
        w = r.get("weight_now")
        verdict = ve.family_verdict(fam) if fam else None
        tier = verdict.tier if verdict else "insufficient"
        if w:
            tiers[tier] = tiers.get(tier, 0.0) + w
        per_name.append({"ticker": r["ticker"], "family": fam,
                         "tier": tier, "badge": verdict.badge if verdict else "Inconclusive",
                         "weight_now": w})
    return {"weight_by_tier": tiers, "per_name": per_name,
            "value_orthogonal_weight": tiers.get("value_orthogonal", 0.0),
            "wrong_signed_weight": tiers.get("wrong_signed", 0.0)}


def _sector_weights(rows: list[dict]) -> dict[str, float]:
    agg: dict[str, float] = {}
    for r in rows:
        w = r.get("weight_now")
        if w:
            agg[r["sector"]] = agg.get(r["sector"], 0.0) + w
    return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))


def _max_drawdown(navs: list[float]) -> float | None:
    navs = [v for v in navs if v is not None]
    if len(navs) < 2:
        return None
    peak, mdd = navs[0], 0.0
    for v in navs:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def _equity_curve(track: dict | None, rebal_log: dict | None) -> list[dict]:
    """Continuous NAV path: chained closed rebalance segments + the current open segment.

    Pre-first-rebalance this is simply 1 + return-since-inception at each track observation.
    """
    pts: list[dict] = []
    nav = spy = qqq = 1.0
    for row in (rebal_log or {}).get("rows", []):
        nav = float(row.get("cumulative_nav", nav))
        br = row.get("benchmark_returns_since_last", {}) or {}
        spy *= (1 + br["SPY"]) if br.get("SPY") is not None else 1.0
        qqq *= (1 + br["QQQ"]) if br.get("QQQ") is not None else 1.0
        pts.append({"date": row["date"], "port_nav": nav, "spy_nav": spy, "qqq_nav": qqq})
    base_nav, base_spy, base_qqq = nav, spy, qqq
    for row in (track or {}).get("rows", []):
        r = (row.get("books") or {}).get(HEADLINE)
        b = row.get("bench", {}) or {}
        pts.append({
            "date": row["date"],
            "port_nav": base_nav * (1 + r) if r is not None else None,
            "spy_nav": base_spy * (1 + b["SPY"]) if b.get("SPY") is not None else None,
            "qqq_nav": base_qqq * (1 + b["QQQ"]) if b.get("QQQ") is not None else None,
        })
    return pts


def _cum_anchor() -> tuple[float, str | None]:
    log = _load_json(REBAL_LOG)
    if log and log.get("rows"):
        last = log["rows"][-1]
        return float(last.get("cumulative_nav", 1.0)), last.get("date")
    return 1.0, None


def _portfolio_metrics(eq: list[dict]) -> dict:
    if not eq:
        return {"basis": "no_data"}
    last = eq[-1]
    port = (last.get("port_nav") or 1.0) - 1.0
    spy = (last.get("spy_nav") or 1.0) - 1.0
    qqq = (last.get("qqq_nav") or 1.0) - 1.0
    m = {
        "basis": "since_inception_simple",
        "port_total_return": port,
        "spy_total_return": spy,
        "qqq_total_return": qqq,
        "excess_vs_spy": port - spy,
        "excess_vs_qqq": port - qqq,
        "max_drawdown": _max_drawdown([p.get("port_nav") for p in eq]),
        "alpha_basis": "simple_excess_return",
    }
    # Once enough history accrues, upgrade "alpha" to an annualized regression alpha using the
    # validated portfolio_metrics helpers (reused, not reimplemented). Untested until live history
    # reaches ALPHA_MIN_PERIODS observations.
    navs = [p["port_nav"] for p in eq if p.get("port_nav") is not None]
    snav = [p["spy_nav"] for p in eq if p.get("spy_nav") is not None]
    if len(navs) >= ALPHA_MIN_PERIODS and len(snav) == len(navs):
        try:
            import pandas as pd  # noqa: E402

            from scripts.portfolio.portfolio_metrics import alpha_beta, sharpe
            pr = pd.Series(navs).pct_change().dropna()
            br = pd.Series(snav).pct_change().dropna()
            a, b = alpha_beta(pr, br)
            m.update({"alpha_basis": "regression_annualized", "jensen_alpha_vs_spy": a,
                      "beta_vs_spy": b, "sharpe": sharpe(pr)})
        except Exception as exc:  # noqa: BLE001
            m["alpha_upgrade_error"] = f"{type(exc).__name__}: {exc}"
    return m


def _update_track(state: dict, bench: dict, book_rets: dict, today: str, days: int) -> dict:
    """Append (or replace today's) tracking row -- shares paper_portfolio_track.json schema."""
    hist = _load_json(TRACK, {"inception": state.get("inception"), "rows": []})
    hist.setdefault("inception", state.get("inception"))
    row = {"date": today, "days": days, "bench": bench, "books": book_rets}
    # drop any existing rows for today (from a prior dashboard run or paper_portfolio.track) so the
    # history stays one-row-per-date regardless of which writer last touched it
    rows = [r for r in hist.get("rows", []) if r.get("date") != today]
    rows.append(row)
    hist["rows"] = rows
    TRACK.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    return hist


# ----------------------------- output writers -----------------------------
def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _write_figure(eq: list[dict]) -> Path | None:
    pts = [p for p in eq if p.get("port_nav") is not None]
    if len(pts) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 -- matplotlib is optional; absence must not break the dashboard
        return None
    x = [p["date"] for p in pts]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, [p["port_nav"] for p in pts], label=f"{HEADLINE} (NAV)", linewidth=2)
    if any(p.get("spy_nav") for p in pts):
        ax.plot(x, [p.get("spy_nav") for p in pts], label="SPY", linestyle="--")
    if any(p.get("qqq_nav") for p in pts):
        ax.plot(x, [p.get("qqq_nav") for p in pts], label="QQQ", linestyle=":")
    ax.set_title("Live paper portfolio -- cumulative NAV")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = LIVE_DIR / "figures" / "equity_curve.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_markdown(state, rows, sectors, book_rets, bench, metrics, eq, intr, days, today, changed) -> None:
    refreshed = (intr or {}).get("asof") if intr else None
    L: list[str] = []
    A = L.append
    A(f"# Live Paper Portfolio -- {HEADLINE}")
    A("")
    A(f"_As of **{today}** ({days} days since inception {state['inception']}). "
      f"Snapshot {state.get('snapshot_asof', 'n/a')}; "
      f"rebalances: {state.get('rebalance_count', 0)}._")
    A("")
    A("## Headline")
    A("")
    A("| Metric | Value |")
    A("|---|---|")
    A(f"| Portfolio return (since inception) | {_pct(book_rets.get(HEADLINE), 2)} |")
    A(f"| SPY | {_pct(bench.get('SPY'), 2)} |")
    A(f"| QQQ | {_pct(bench.get('QQQ'), 2)} |")
    A(f"| Excess vs SPY ({metrics.get('alpha_basis')}) | {_pct(metrics.get('excess_vs_spy'), 2, signed=True)} |")
    A(f"| Excess vs QQQ | {_pct(metrics.get('excess_vs_qqq'), 2, signed=True)} |")
    if metrics.get("jensen_alpha_vs_spy") is not None:
        A(f"| Jensen alpha vs SPY (annualized) | {_pct(metrics.get('jensen_alpha_vs_spy'), 2, signed=True)} |")
    A(f"| Max drawdown | {_pct(metrics.get('max_drawdown'), 2) if metrics.get('max_drawdown') is not None else 'n/a (need 2+ obs)'} |")
    A("")
    if changed:
        A(f"**Since last run:** {changed}")
        A("")
    A("## Holdings (equal-weight)")
    A("")
    cur_hdr = " Cur disc |" if refreshed else ""
    A(f"| Ticker | Sector | Entry | Current | Return | Wt now | Contrib | Incep disc |{cur_hdr}")
    A(f"|---|---|--:|--:|--:|--:|--:|--:|{'--:|' if refreshed else ''}")
    for r in rows:
        cur_cell = f" {_pct(r['current_discount'], 0)} |" if refreshed else ""
        A(f"| {r['ticker']} | {_sector_name(r['sector'])} | {_usd(r['entry_price'])} | "
          f"{_usd(r['current_price'])} | {_pct(r['return'], 1, signed=True)} | "
          f"{_pct(r['weight_now'], 1)} | {_pct(r['contribution'], 2, signed=True)} | "
          f"{_pct(r['inception_discount'], 0)} |{cur_cell}")
    A("")
    A("## Sector weights (current, drifted)")
    A("")
    A("| Sector | Weight |")
    A("|---|--:|")
    for s, w in sectors.items():
        A(f"| {_sector_name(s)} | {_pct(w, 1)} |")
    A("")
    overlay = _ranking_overlay(rows)
    if overlay:
        TIER_LABEL = {"value_orthogonal": "Incremental-to-value (scoped fade edge)",
                      "value_proxy": "Spanned by Book/Market value factor",
                      "wrong_signed": "Wrong-signed net of value",
                      "insufficient": "Inconclusive"}
        A("## Fade-engine ranking context (scoped, read-only)")
        A("")
        A("_The live book ranks on the **frozen v1** discount; construction is unchanged. This panel "
          "maps the current book onto the corrected-panel (2026-06-11) finding that the fade engine's "
          "incremental-to-value ranking edge is confined to consumer staples and consumer "
          "discretionary. It is context, not a trading signal._")
        A("")
        A("| Family ranking tier | Book weight |")
        A("|---|--:|")
        order = ["value_orthogonal", "value_proxy", "wrong_signed", "insufficient"]
        for tier in order:
            w = overlay["weight_by_tier"].get(tier)
            if w:
                A(f"| {TIER_LABEL[tier]} | {_pct(w, 1)} |")
        A("")
        A(f"_Scoped-edge families (consumer staples + discretionary) are "
          f"**{_pct(overlay['value_orthogonal_weight'], 1)}** of the drifted book; "
          f"wrong-signed-net-of-value families are {_pct(overlay['wrong_signed_weight'], 1)}._")
        A("")
    A("## All books")
    A("")
    A("| Book | Return | vs SPY | vs QQQ |")
    A("|---|--:|--:|--:|")
    for name in ("discount_x_conf_top10", "discount_x_conf_top20", "discount_top10", "discount_top20"):
        br = book_rets.get(name)
        vs_spy = (br - bench["SPY"]) if (br is not None and bench.get("SPY") is not None) else None
        vs_qqq = (br - bench["QQQ"]) if (br is not None and bench.get("QQQ") is not None) else None
        star = " (headline)" if name == HEADLINE else ""
        A(f"| {name}{star} | {_pct(br, 2)} | {_pct(vs_spy, 2, signed=True)} | {_pct(vs_qqq, 2, signed=True)} |")
    A("")
    A("---")
    A(f"_Intrinsic/discount {'refreshed ' + refreshed if refreshed else 'shown at inception only'}; "
      "run `refresh-intrinsics` to re-score holdings (pinned internal WACC/terminal-growth)._  ")
    A("_Equal-weight, quarterly rebalance intended. Research output, not investment advice._")
    (LIVE_DIR / "dashboard.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------- commands -----------------------------
def cmd_dashboard() -> None:
    state = _load_json(STATE)
    if not state:
        raise SystemExit(f"{STATE} missing; run `python -m scripts.portfolio.paper_portfolio open` first")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / "figures").mkdir(exist_ok=True)
    books = state["books"]
    all_names = sorted({r["ticker"] for b in books.values() for r in b} | set(BENCHMARKS))
    print(f"fetching current prices for {len(all_names)} names...")
    cur = _live_prices(all_names)
    today = _today()
    days = (dt.date.fromisoformat(today) - dt.date.fromisoformat(state["inception"])).days
    bench = _bench_returns(state, cur)
    intr = _load_json(INTRINSICS)

    book_rets = {name: _book_return(b, cur) for name, b in books.items()}
    rows = _headline_rows(books[HEADLINE], cur, intr)
    sectors = _sector_weights(rows)

    # detect change vs the previous tracking observation BEFORE we write today's row
    prev = (_load_json(TRACK, {"rows": []}) or {}).get("rows", [])
    prev_port = prev[-1]["books"].get(HEADLINE) if (prev and prev[-1].get("date") != today) else None
    _update_track(state, bench, book_rets, today, days)

    eq = _equity_curve(_load_json(TRACK), _load_json(REBAL_LOG))
    metrics = _portfolio_metrics(eq)
    changed = ""
    if prev_port is not None and book_rets.get(HEADLINE) is not None:
        changed = f"headline return {_pct(prev_port, 2)} -> {_pct(book_rets[HEADLINE], 2)}"

    # write artifacts
    _write_csv(LIVE_DIR / "holdings.csv",
               ["ticker", "sector", "entry_price", "current_price", "return", "weight_now",
                "contribution", "inception_intrinsic", "inception_discount",
                "current_intrinsic", "current_discount"], rows)
    _write_csv(LIVE_DIR / "equity_curve.csv", ["date", "port_nav", "spy_nav", "qqq_nav"], eq)
    fig = _write_figure(eq)
    summary = {
        "asof": today, "inception": state["inception"], "days": days,
        "rebalance_count": state.get("rebalance_count", 0), "headline_book": HEADLINE,
        "snapshot_asof": state.get("snapshot_asof"), "n_holdings": len(books[HEADLINE]),
        "portfolio_return": book_rets.get(HEADLINE), "benchmarks": bench,
        "book_returns": book_rets, "metrics": metrics, "sector_weights": sectors,
        "intrinsics_asof": (intr or {}).get("asof") if intr else None,
        "fade_ranking_overlay": _ranking_overlay(rows),
    }
    (LIVE_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(state, rows, sectors, book_rets, bench, metrics, eq, intr, days, today, changed)

    # terminal print
    print(f"\nLive paper portfolio -- {HEADLINE} | inception {state['inception']} ({days}d) | as of {today}")
    print(f"  portfolio {_pct(book_rets.get(HEADLINE), 2)}   "
          f"SPY {_pct(bench.get('SPY'), 2)}   QQQ {_pct(bench.get('QQQ'), 2)}   "
          f"excess vs SPY {_pct(metrics.get('excess_vs_spy'), 2, signed=True)}")
    if not intr:
        print("  (intrinsic/discount shown at inception only -- run `refresh-intrinsics` for current values)")
    print("  wrote reports/live/: dashboard.md, holdings.csv, equity_curve.csv, summary.json"
          + (", figures/equity_curve.png" if fig else ""))


def cmd_refresh_intrinsics(all_books: bool) -> None:
    state = _load_json(STATE)
    if not state:
        raise SystemExit(f"{STATE} missing; run `python -m scripts.portfolio.paper_portfolio open` first")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if all_books:
        names = sorted({r["ticker"] for b in state["books"].values() for r in b})
    else:
        names = [r["ticker"] for r in state["books"][HEADLINE]]

    # heavy imports kept local so the fast `dashboard` path never pays for them
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/  -> scorecard
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # src/      -> dcf_project
    from dataclasses import replace

    from dcf_project.config import get_settings
    from scorecard import Reference, score_ticker

    # Pin internal WACC + terminal-growth sources for run-to-run reproducibility (provider WACC/
    # terminal-growth are the documented source of valuation drift). MC/sensitivity off for speed --
    # the discount is deterministic and independent of them (same flags as build_live_snapshot).
    settings = replace(get_settings(), wacc_source="internal", terminal_growth_source="internal",
                       run_sensitivity=False, monte_carlo_samples=0, scenario_monte_carlo_samples=0)
    ref = Reference()
    print(f"re-scoring {len(names)} holdings with pinned internal sources...")
    holdings: dict[str, dict] = {}
    for t in names:
        try:
            sc = score_ticker(t, ref, settings)
            a = sc.get("assumptions", {})
            holdings[t] = {"intrinsic": sc.get("intrinsic_value"), "price": sc.get("price"),
                           "discount": sc.get("discount_to_intrinsic"), "wacc": a.get("wacc"),
                           "terminal_growth": a.get("terminal_growth"), "verdict": sc.get("verdict")}
            print(f"  {t:6} discount {_pct(sc.get('discount_to_intrinsic'), 0):>8}   iv {_usd(sc.get('intrinsic_value'))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {t:6} failed ({type(exc).__name__}: {exc})")
    out = {"asof": _today(), "wacc_source": "internal", "terminal_growth_source": "internal",
           "scope": "all_books" if all_books else HEADLINE, "holdings": holdings}
    INTRINSICS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote reports/live/intrinsics.json ({len(holdings)}/{len(names)} priced). "
          "Re-run `dashboard` to fold current discounts into the card.")


def cmd_rebalance(dry_run: bool) -> None:
    state = _load_json(STATE)
    if not state:
        raise SystemExit(f"{STATE} missing; run `python -m scripts.portfolio.paper_portfolio open` first")
    if not SNAP.exists():
        raise SystemExit(f"{SNAP} missing; run `python scripts/build_live_snapshot.py` first (~15-25 min)")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    snap_asof = dt.datetime.fromtimestamp(SNAP.stat().st_mtime).date().isoformat()

    new_books = _rank_books()
    new_book, cur_book = new_books[HEADLINE], state["books"][HEADLINE]
    prior = [r["ticker"] for r in cur_book]
    newt = [r["ticker"] for r in new_book]
    prior_s, new_s = set(prior), set(newt)
    adds = [t for t in newt if t not in prior_s]
    drops = [t for t in prior if t not in new_s]
    holds = [t for t in newt if t in prior_s]
    w_old = {t: 1.0 / len(prior) for t in prior}
    w_new = {t: 1.0 / len(newt) for t in newt}
    turnover = 0.5 * sum(abs(w_new.get(t, 0.0) - w_old.get(t, 0.0)) for t in prior_s | new_s)

    cur = _live_prices(sorted(prior_s | set(BENCHMARKS)))
    realized = _book_return(cur_book, cur)
    bench = _bench_returns(state, cur)
    cum0, last_date = _cum_anchor()
    new_cum = cum0 * (1 + realized) if realized is not None else cum0

    print(f"Rebalance {'PREVIEW' if dry_run else 'EXECUTE'} -- {HEADLINE} (snapshot {snap_asof})")
    print(f"  realized return of current book since {last_date or state['inception']}: {_pct(realized, 2)}")
    print(f"  turnover (1-sided): {_pct(turnover, 1)}   adds {len(adds)}  drops {len(drops)}  holds {len(holds)}")
    print(f"  DROP: {drops}")
    print(f"  ADD : {adds}")

    if dry_run:
        preview = {"asof": _today(), "snapshot_asof": snap_asof, "headline_book": HEADLINE,
                   "prior_holdings": prior, "new_holdings": newt, "adds": adds, "drops": drops,
                   "holds": holds, "turnover_1sided": round(turnover, 4),
                   "realized_return_since_last": realized, "prior_cumulative_nav": cum0,
                   "would_be_cumulative_nav": new_cum, "benchmark_returns_since_last": bench}
        (LIVE_DIR / "rebalance_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
        print(f"\n[dry-run] would carry cumulative NAV {cum0:.4f} -> {new_cum:.4f}. "
              "Wrote reports/live/rebalance_preview.json. No live state changed.")
        return

    # --- execute: archive, re-lock entry prices, roll NAV, log, reset open segment ---
    (LIVE_DIR / "books").mkdir(exist_ok=True)
    stamp = state.get("last_rebalance") or state["inception"]
    (LIVE_DIR / "books" / f"book_{stamp}.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    fresh = _live_prices(sorted({r["ticker"] for b in new_books.values() for r in b} | set(BENCHMARKS)))
    for b in new_books.values():
        for r in b:
            r["entry_price"] = fresh.get(r["ticker"])
    new_state = {"inception": state["inception"], "last_rebalance": _today(),
                 "rebalance_count": int(state.get("rebalance_count", 0)) + 1, "snapshot_asof": snap_asof,
                 "benchmark_entry": {bm: fresh.get(bm) for bm in BENCHMARKS}, "books": new_books,
                 "note": state.get("note", "")}
    STATE.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    log = _load_json(REBAL_LOG, {"inception": state["inception"], "rows": []})
    log["rows"].append({"date": _today(), "headline_book": HEADLINE, "prior_holdings": prior,
                        "new_holdings": newt, "adds": adds, "drops": drops,
                        "turnover_1sided": round(turnover, 4), "realized_return_since_last": realized,
                        "cumulative_nav": new_cum, "benchmark_returns_since_last": bench})
    REBAL_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    TRACK.write_text(json.dumps({"inception": state["inception"], "last_rebalance": _today(),
                                 "rows": []}, indent=2), encoding="utf-8")
    print(f"\nrebalanced: cumulative NAV {cum0:.4f} -> {new_cum:.4f}; archived prior book to "
          f"reports/live/books/book_{stamp}.json; logged to rebalance_log.json.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="live_dashboard", description="Live paper-portfolio dashboard")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("dashboard", help="fast price-based dashboard -> reports/live/ (default)")
    ri = sub.add_parser("refresh-intrinsics", help="re-score holdings with pinned internal sources")
    ri.add_argument("--all-books", action="store_true", help="refresh the union of all 4 books")
    rb = sub.add_parser("rebalance", help="quarterly rebalance (use --dry-run to preview)")
    rb.add_argument("--dry-run", action="store_true", help="preview adds/drops/turnover; change nothing")
    args = ap.parse_args()
    cmd = args.cmd or "dashboard"
    if cmd == "dashboard":
        cmd_dashboard()
    elif cmd == "refresh-intrinsics":
        cmd_refresh_intrinsics(all_books=args.all_books)
    elif cmd == "rebalance":
        cmd_rebalance(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
