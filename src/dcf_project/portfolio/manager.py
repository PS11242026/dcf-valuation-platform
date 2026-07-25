"""PRODUCT 3 -- Portfolio manager (the live/managed books; PM dashboard).

Ported from the verified scripts/portfolio/live_dashboard.py, then generalized so EVERY managed
book flows through one code path:

  * equal-weight books (conf_top10): return = mean of per-name (current/entry - 1)  [unchanged]
  * fixed-shares books (india_v3_aggressive): return = (sum shares*price + cash)/capital - 1
  * per-book benchmarks read from state.json (SPY/QQQ dict, or a single ^NSEI scalar)
  * per-book currency (USD/INR) via the reporting kit's native formats
  * `update` additionally writes status.json (website-ready payload with benchmark actives +
    comparables) and news.json (best-effort per-holding headlines)

The live state is mutated ONLY by an explicit `rebalance` (update/preview never write state);
manual conviction books (no auto signal) refuse preview/rebalance with a clear message.
Signal/construction unchanged.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from dcf_project.paths import MANAGED
from dcf_project.portfolio import registry, signals
from dcf_project.portfolio.scoring import sector_label
from dcf_project.reporting_kit import Image, Metric, ReportModel, Table, save_model, viz3d, xlsx

BENCHMARKS = signals.BENCHMARKS  # default US pair; per-book set comes from state.json

# Currency -> (big-amount fmt, per-share fmt) in the reporting kit's format vocabulary.
CCY_FMT = {"USD": ("usd0", "usd"), "INR": ("inr0", "inr")}
CCY_SYM = {"USD": "$", "INR": "Rs "}

# Books whose DEFAULT model slot / <name>.xlsx belong to a dedicated pitch exporter
# (scripts/portfolio/india_v3_export.py). For these the dashboard writes <name>_dashboard.xlsx
# and persists to the dashboard model slot so `manage update` can never clobber the pitch package.
MODEL_OWNER_PITCH = {"india_v3_aggressive"}

# Fund/index comparators per managed book (the website's head-to-head block). `run_rate` lines are
# MODELED (no daily NAV published) and labeled as such; `index` lines are fetched over the window.
# Add an India PMS here as {"kind": "run_rate", "annual": <reported>} once one is chosen.
COMPARABLES = {
    "conf_top10": [
        {"name": "Titan Flagship (modeled)", "kind": "run_rate", "annual": 0.1072,
         "note": "reported ~10.72%/yr net run-rate compounded over the window; no daily NAV published"},
        {"name": "Russell 2000 (IWM)", "kind": "index", "symbol": "IWM",
         "note": "small/mid-cap universe proxy"},
        {"name": "Russell 1000 Value (IWD)", "kind": "index", "symbol": "IWD",
         "note": "value-factor benchmark"},
        {"name": "Consumer Staples (XLP)", "kind": "index", "symbol": "XLP",
         "note": "sector-fair for the staples-heavy book"},
        {"name": "Berkshire Hathaway (BRK-B)", "kind": "index", "symbol": "BRK-B",
         "note": "flagship value manager, tradeable comparison"},
    ],
    "india_v3_aggressive": [
        {"name": "NIFTY Energy", "kind": "index", "symbol": "^CNXENERGY",
         "note": "sector-fair benchmark for an all-energy PSU book"},
    ],
}


# ----------------------------- book/benchmark shape helpers -----------------------------
def _benchmarks_of(state) -> dict[str, float]:
    """Normalize benchmark entry marks: {'SPY': px, 'QQQ': px} or benchmark+scalar -> {'^NSEI': px}."""
    be = state.get("benchmark_entry")
    if isinstance(be, dict):
        return {k: v for k, v in be.items() if v}
    bm = state.get("benchmark")
    return {bm: be} if (bm and be) else {}


def _is_shares_book(book) -> bool:
    return any(r.get("shares") for r in book)


def _symbol_map(books) -> dict[str, str]:
    """ticker -> price symbol (India stores 'IOC' + 'IOC.NS'; US books have ticker == symbol)."""
    return {r["ticker"]: r.get("symbol", r["ticker"]) for b in books.values() for r in b}


# ----------------------------- return math (matches paper_portfolio / india book) ----------------
def _book_return(book, cur, cash: float = 0.0, capital: float | None = None):
    if _is_shares_book(book):
        base = capital or (sum((r.get("shares") or 0) * (r.get("entry_price") or 0) for r in book) + cash)
        if not base:
            return None
        # unpriced names are held at entry so the NAV stays defined (conservative, no fake move)
        cur_mv = sum((r.get("shares") or 0) * (cur.get(r["ticker"]) or r.get("entry_price") or 0)
                     for r in book)
        return (cur_mv + cash) / base - 1.0
    rets = [cur[r["ticker"]] / r["entry_price"] - 1.0 for r in book
            if r.get("entry_price") and cur.get(r["ticker"]) and r["entry_price"] > 0]
    return sum(rets) / len(rets) if rets else None


def _bench_returns(bench_entry: dict[str, float], px: dict[str, float]):
    return {bm: (px[bm] / e - 1.0) if (px.get(bm) and e) else None for bm, e in bench_entry.items()}


def _holding_rows(book, cur, intr, capital, cash: float = 0.0):
    """Per-name detail rows. Entry/current price are first-class columns (the questions a PM asks
    first); contributions sum to the book return under either weighting scheme. `entry_value` is
    the position's value AT INCEPTION (shares*entry, or capital/n for equal-weight books) so the
    holdings table can show value-at-entry vs value-now and a total NAV line."""
    intr_h = (intr or {}).get("holdings", {}) if intr else {}
    rows = []
    if _is_shares_book(book):
        base = capital or (sum((r.get("shares") or 0) * (r.get("entry_price") or 0) for r in book) + cash)
        mv = {r["ticker"]: (r.get("shares") or 0) * (cur.get(r["ticker"]) or r.get("entry_price") or 0)
              for r in book}
        tot = sum(mv.values())
        for r in book:
            t, e, c = r["ticker"], r.get("entry_price"), cur.get(r["ticker"])
            ci = intr_h.get(t) or {}
            rows.append({
                "ticker": t, "sector": r.get("sector"), "entry_price": e, "price": c,
                "entry_value": ((r.get("shares") or 0) * e) if e else None,
                "weight": (mv[t] / tot) if tot else None, "market_value": mv[t] or None,
                "return": (c / e - 1.0) if (e and c) else None,
                "contribution": ((r.get("shares") or 0) * ((c or e or 0) - (e or 0)) / base) if base else None,
                "inception_discount": r.get("discount_to_intrinsic", r.get("discount")),
                "current_discount": ci.get("discount"),
                "current_intrinsic": ci.get("intrinsic", r.get("intrinsic")),
                "conviction": r.get("conf", r.get("selection_rank")),
            })
    else:
        priced = [r for r in book if r.get("entry_price") and cur.get(r["ticker"]) and r["entry_price"] > 0]
        n_valid = len(priced)
        growth = {r["ticker"]: cur[r["ticker"]] / r["entry_price"] for r in priced}  # drifted weights
        denom = sum(growth.values())
        for r in book:
            t, e, c = r["ticker"], r.get("entry_price"), cur.get(r["ticker"])
            ret = (c / e - 1.0) if (e and c and e > 0) else None
            w_now = (growth[t] / denom) if (t in growth and denom) else None
            ci = intr_h.get(t) or {}
            rows.append({
                "ticker": t, "sector": r.get("sector"), "entry_price": e, "price": c,
                "entry_value": (capital / len(book)) if (capital and len(book)) else None,
                "weight": w_now,
                "market_value": (capital / len(book) * (1 + ret)) if (capital and len(book) and ret is not None) else None,
                "return": ret, "contribution": (ret / n_valid) if (ret is not None and n_valid) else None,
                "inception_discount": r.get("discount"), "current_discount": ci.get("discount"),
                "current_intrinsic": ci.get("intrinsic"), "conviction": r.get("conf"),
            })
    rows.sort(key=lambda x: (x["inception_discount"] is not None, x["inception_discount"]), reverse=True)
    return rows


def _sector_weights(rows):
    agg = {}
    for r in rows:
        if r.get("weight"):
            agg[r["sector"]] = agg.get(r["sector"], 0.0) + r["weight"]
    return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))


def _max_dd(navs):
    navs = [v for v in navs if v is not None]
    if len(navs) < 2:
        return None
    peak, mdd = navs[0], 0.0
    for v in navs:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def _equity_curve(track, rebal_log, book_id, benchmarks):
    """NAV points {date, port_nav, bench: {bm: nav}} chained across rebalances."""
    pts = []
    nav = 1.0
    bnav = {bm: 1.0 for bm in benchmarks}
    for row in (rebal_log or {}).get("rows", []):
        nav = float(row.get("cumulative_nav", nav))
        br = row.get("benchmark_returns_since_last", {}) or {}
        for bm in benchmarks:
            bnav[bm] *= (1 + br[bm]) if br.get(bm) is not None else 1.0
        pts.append({"date": row["date"], "port_nav": nav, "bench": dict(bnav)})
    anchor_nav, anchor_b = nav, dict(bnav)
    for row in (track or {}).get("rows", []):
        r = (row.get("books") or {}).get(book_id)
        b = row.get("bench", {}) or {}
        pts.append({"date": row["date"],
                    "port_nav": anchor_nav * (1 + r) if r is not None else None,
                    "bench": {bm: anchor_b[bm] * (1 + b[bm]) if b.get(bm) is not None else None
                              for bm in benchmarks}})
    return pts


def _cum_anchor(rebal_log):
    rows = (rebal_log or {}).get("rows", [])
    if rows:
        return float(rows[-1].get("cumulative_nav", 1.0)), rows[-1].get("date")
    return 1.0, None


def _load(p, default=None):
    return json.load(open(p)) if Path(p).exists() else default


def _update_track(folder, state, bench, book_rets, today, days):
    tf = folder / "track.json"
    hist = _load(tf, {"inception": state.get("inception"), "rows": []})
    hist.setdefault("inception", state.get("inception"))
    hist["rows"] = [r for r in hist.get("rows", []) if r.get("date") != today]
    hist["rows"].append({"date": today, "days": days, "bench": bench, "books": book_rets})
    tf.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    return hist


def _metrics_from_eq(eq, port, bench, primary):
    m = {"excess": {bm: (port - r) if (port is not None and r is not None) else None
                    for bm, r in bench.items()},
         "primary_benchmark": primary, "alpha_basis": "simple_excess_return",
         "max_drawdown": _max_dd([p.get("port_nav") for p in eq]),
         "sharpe": None, "volatility": None, "beta": None, "turnover": None, "hhi": None}
    navs = [p["port_nav"] for p in eq if p.get("port_nav") is not None]
    if len(navs) >= 3:
        import numpy as np
        rets = np.diff(navs) / np.array(navs[:-1])
        if rets.std() > 0:
            m["volatility"] = float(rets.std() * (252 ** 0.5))
            m["sharpe"] = float(rets.mean() / rets.std() * (252 ** 0.5))
        pnav = [p["bench"].get(primary) for p in eq
                if p.get("port_nav") is not None and p.get("bench", {}).get(primary) is not None]
        if len(pnav) == len(navs) and len(pnav) >= 3:
            sret = np.diff(pnav) / np.array(pnav[:-1])
            if sret.var() > 0:
                m["beta"] = float(np.cov(rets, sret)[0, 1] / sret.var())
    return m


# ----------------------------- comparables / status / news -----------------------------
def _comparable_rows(name, state, port, days):
    out = []
    for c in COMPARABLES.get(name, []):
        row = {"name": c["name"], "kind": c["kind"], "note": c.get("note"),
               "symbol": c.get("symbol")}
        try:
            if c["kind"] == "run_rate":
                row["return"] = (1 + c["annual"]) ** (days / 365.0) - 1 if days else 0.0
            elif c["kind"] == "index":
                first, last = signals.window_prices(c["symbol"], state["inception"])
                row["return"] = (last / first - 1.0) if (first and last) else None
        except Exception as exc:  # noqa: BLE001 -- comparators are best-effort, never block update
            print(f"  comparable '{c['name']}' failed: {exc}")
            row["return"] = None
        row["active"] = (port - row["return"]) if (port is not None and row.get("return") is not None) else None
        out.append(row)
    return out


def _write_status(folder, name, state, book_id, rows, sectors, port, bench_entry, bench, px,
                  metrics, comparables, today, days, cap=None, cap_basis=None):
    ccy = state.get("currency", "USD")
    cap = cap if cap is not None else state.get("capital")
    status = {
        "as_of": today, "book": name, "book_id": book_id, "currency": ccy,
        "inception": state.get("inception"), "days": days,
        "capital": cap, "capital_basis": cap_basis, "cash": state.get("cash", 0.0),
        "value_at_inception": cap,
        "value": (cap * (1 + port)) if (cap and port is not None) else None,
        "return_since_inception": port,
        "benchmarks": {bm: {"entry": e, "last": px.get(bm), "return": bench.get(bm),
                            "active": (port - bench[bm]) if (port is not None and bench.get(bm) is not None) else None}
                       for bm, e in bench_entry.items()},
        "comparables": comparables,
        "holdings": rows, "sector_weights": sectors, "metrics": metrics,
        "honesty": state.get("honesty"),
    }
    (folder / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def _write_news(folder, books, today, per_name: int = 3):
    syms = sorted({r.get("symbol", r["ticker"]) for b in books.values() for r in b})
    items = signals.fetch_news(syms, per_name=per_name)
    (folder / "news.json").write_text(json.dumps({"as_of": today, "items": items}, indent=2),
                                      encoding="utf-8")
    return items


# ----------------------------- dashboard workbook -----------------------------
def _build_dashboard(name, state, book_id, rows, sectors, book_rets, bench, metrics, eq, days, today, folder):
    figs = folder / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    cap = state.get("capital")
    ccy = state.get("currency", "USD")
    fmt_big, fmt_ps = CCY_FMT.get(ccy, CCY_FMT["USD"])
    sym = CCY_SYM.get(ccy, "$")
    benchmarks = list(bench.keys())
    primary = metrics.get("primary_benchmark") or (benchmarks[0] if benchmarks else "SPY")
    excess = metrics.get("excess", {})
    port = book_rets.get(book_id)
    m = ReportModel(
        title=f"{name} -- Managed Portfolio",
        subtitle=f"Live out-of-sample record on {book_id}; inception {state['inception']} ({days}d)",
        persona="pm",
        meta={"Portfolio": name, "Book": book_id, "As of": today, "Inception": state["inception"],
              "Currency": ccy, "Capital": (f"{sym}{cap:,.0f}" if cap else "notional"),
              "Rebalances": state.get("rebalance_count", 0)},
    )
    # 1) Executive Summary
    s = m.section("Executive Summary", "Executive Summary")
    cur_val = (cap * (1 + port)) if (cap and port is not None) else None
    s.metrics = [
        Metric("Capital", cap, fmt_big) if cap else Metric("Holdings", len(rows), "num0"),
        Metric("Current value", cur_val, fmt_big) if cap else Metric("Portfolio return", port, "pct", good=(port or 0) >= 0),
        Metric("Portfolio return", port, "pct", good=(port or 0) >= 0),
        *[Metric(f"{bm} return", bench.get(bm), "pct") for bm in benchmarks],
        Metric(f"Alpha vs {primary}", excess.get(primary), "pct", good=(excess.get(primary) or 0) >= 0),
        Metric("Sharpe", metrics.get("sharpe"), "num"),
        Metric("Max drawdown", metrics.get("max_drawdown"), "pct", good=False),
        Metric("Turnover (cum)", metrics.get("turnover"), "pct"),
    ]
    s.narrative = [f"Alpha basis: {metrics.get('alpha_basis')}.",
                   "Live, frozen between rebalances; rebalance only on explicit command."]
    # 2) Current Holdings
    h = m.section("Current Holdings", "Current Holdings")
    h.tables.append(Table(None,
        ["Ticker", "Sector", "Entry", "Price", "Weight", "Market value", "Return", "Contribution",
         "Cur discount", "Cur intrinsic", "Conviction"],
        [[r["ticker"], sector_label(r["sector"]), r.get("entry_price"), r.get("price"), r["weight"],
          r["market_value"], r["return"], r["contribution"], r["current_discount"],
          r["current_intrinsic"], r["conviction"]] for r in rows],
        formats=["text", "text", fmt_ps, fmt_ps, "pct", fmt_big, "pct", "pct", "pct", fmt_ps, "num"]))
    # 3) Performance
    p = m.section("Performance", "Performance")
    p.tables.append(Table("Book vs benchmarks (all tracked books)",
        ["Book", "Return"] + [f"vs {bm}" for bm in benchmarks],
        [[bk, br] + [(br - bench[bm]) if (br is not None and bench.get(bm) is not None) else None
                     for bm in benchmarks]
         for bk, br in book_rets.items()],
        formats=["text"] + ["pct"] * (1 + len(benchmarks))))
    ec = viz3d.equity_curve([q["date"] for q in eq],
                            {"portfolio": [q["port_nav"] for q in eq],
                             **{bm: [q["bench"].get(bm) for q in eq] for bm in benchmarks}},
                            figs / "equity_curve.png")
    if ec:
        p.images.append(Image("Equity curve", ec))
    # 4) Attribution
    at = m.section("Attribution", "Attribution")
    parts = {f"Market ({primary})": bench.get(primary) or 0.0, "Selection": (excess.get(primary) or 0.0),
             "Sector allocation": 0.0}
    wf = viz3d.alpha_decomposition(parts, figs / "alpha.png", title="Return attribution")
    if wf:
        at.images.append(Image("Alpha decomposition", wf))
    at.tables.append(Table("Top contributors", ["Ticker", "Contribution"],
        [[r["ticker"], r["contribution"]] for r in sorted(rows, key=lambda x: (x["contribution"] is not None, x["contribution"] or 0), reverse=True)[:10]],
        formats=["text", "pct"]))
    # 5) Risk Dashboard
    rk = m.section("Risk Dashboard", "Risk Dashboard")
    rk.metrics = [Metric("Volatility (ann.)", metrics.get("volatility"), "pct"),
                  Metric(f"Beta vs {primary}", metrics.get("beta"), "num"),
                  Metric("Max drawdown", metrics.get("max_drawdown"), "pct", good=False),
                  Metric("Sector HHI", metrics.get("hhi"), "num")]
    sb = viz3d.sector_bar({sector_label(k): v for k, v in sectors.items()}, figs / "sector.png")
    if sb:
        rk.images.append(Image("Sector exposure", sb))
    dd = viz3d.drawdown([q["date"] for q in eq], [q["port_nav"] for q in eq if q.get("port_nav") is not None] or [1.0], figs / "drawdown.png")
    if dd:
        rk.images.append(Image("Drawdown", dd))
    cn = viz3d.concentration([r["weight"] for r in rows if r.get("weight")], figs / "conc.png")
    if cn:
        rk.images.append(Image("Concentration", cn))
    # 6) Rebalance Center
    rc = m.section("Rebalance Center", "Rebalance Center")
    if registry.get(name) and registry.get(name).get("rebalance") == "manual":
        rc.narrative = ["Manual conviction book: holdings change only by explicit re-lock of the "
                        "book state; no auto-rebalance signal is wired."]
    else:
        rc.narrative = ["Run `portfolio manage preview --book {0}` for a dry-run, "
                        "or `portfolio manage rebalance --book {0}` to execute.".format(name)]
        prev = _load(folder / "rebalance_preview.json")
        if prev:
            rc.metrics = [Metric("Adds", len(prev.get("adds", [])), "num0"), Metric("Drops", len(prev.get("drops", [])), "num0"),
                          Metric("Turnover", prev.get("turnover_1sided"), "pct")]
            rc.tables.append(Table("Latest preview", ["Field", "Value"],
                [["Adds", ", ".join(prev.get("adds", []))], ["Drops", ", ".join(prev.get("drops", []))],
                 ["Realized return since last", prev.get("realized_return_since_last")]], formats=["text", "text"]))
    # 7) History
    hi = m.section("History", "History")
    log = _load(folder / "rebalance_log.json", {"rows": []})
    if log.get("rows"):
        hi.tables.append(Table("Rebalance history", ["Date", "Adds", "Drops", "Turnover", "Realized return", "Cumulative NAV"],
            [[r["date"], len(r.get("adds", [])), len(r.get("drops", [])), r.get("turnover_1sided"),
              r.get("realized_return_since_last"), r.get("cumulative_nav")] for r in log["rows"]],
            formats=["text", "num0", "num0", "pct", "pct", "num"]))
    else:
        hi.narrative = ["No rebalances yet -- this is the inception book."]
    if name in MODEL_OWNER_PITCH:
        from dcf_project.reporting_kit.persist import DASHBOARD_MODEL_FILE
        out = xlsx.render(m, folder / f"{name}_dashboard.xlsx")
        save_model(m, folder, stem=DASHBOARD_MODEL_FILE)
    else:
        out = xlsx.render(m, folder / f"{name}.xlsx")
        save_model(m, folder)
    return out


def run_manage(book: str, action: str = "update") -> None:
    p = registry.get(book)
    if not p:
        raise SystemExit(f"No managed portfolio '{book}'. See `portfolio list`.")
    folder = MANAGED / book
    book_id = p["book_id"]
    state = _load(folder / "state.json")
    if not state:
        raise SystemExit(f"{folder/'state.json'} missing.")
    if action in ("preview", "rebalance"):
        if p.get("rebalance") == "manual":
            raise SystemExit(f"'{book}' is a manual conviction book -- no auto-rebalance signal is "
                             "wired. Re-lock the book state explicitly to change holdings.")
        return _rebalance(book, folder, book_id, state, dry=(action == "preview"))
    # ---- update (monitor) ----
    books = state.get("books") or {book_id: state.get("holdings", [])}
    if book_id not in books:  # standalone books store holdings under a single implicit id
        book_id = next(iter(books))
    bench_entry = _benchmarks_of(state)
    syms = _symbol_map(books)
    fetch = sorted(set(syms.values()) | set(bench_entry))
    print(f"fetching prices for {len(fetch)} names...")
    lp = signals.live_last_prev(fetch)
    px = {s: v[0] for s, v in lp.items()}
    prev = {s: v[1] for s, v in lp.items()}
    cur = {tk: px.get(s) for tk, s in syms.items()}
    day_chg = {tk: (px[s] / prev[s] - 1.0) if (px.get(s) and prev.get(s)) else None
               for tk, s in syms.items()}
    today = dt.date.today().isoformat()
    days = (dt.date.fromisoformat(today) - dt.date.fromisoformat(state["inception"])).days
    bench = _bench_returns(bench_entry, px)
    intr = _load(folder / "intrinsics.json")
    cash = state.get("cash", 0.0)
    # effective capital: booked (India) or the registry's presentation notional (US paper book);
    # the frozen state.json is never mutated -- notional is display/NAV basis only.
    cap_eff = state.get("capital") or p.get("display_capital")
    cap_basis = "booked" if state.get("capital") else ("notional" if cap_eff else None)
    book_rets = {bk: _book_return(b, cur, cash=cash, capital=state.get("capital")) for bk, b in books.items()}
    rows = _holding_rows(books[book_id], cur, intr, cap_eff, cash=cash)
    try:  # full company names (cached after first fetch) -- hover/tooltip + table subline
        names = signals.fetch_names(sorted(set(syms.values())), folder / "figures" / ".names.json")
    except Exception as exc:  # noqa: BLE001
        print(f"  name fetch failed: {exc}")
        names = {}
    for r in rows:  # 1-day move per name + weighted book day change (heatmap fields)
        r["day_change"] = day_chg.get(r["ticker"])
        r["company"] = names.get(syms.get(r["ticker"], r["ticker"]))
    wsum = sum(r["weight"] for r in rows if r.get("weight") and r.get("day_change") is not None)
    book_day = (sum(r["weight"] * r["day_change"] for r in rows
                    if r.get("weight") and r.get("day_change") is not None) / wsum) if wsum else None
    sectors = _sector_weights(rows)
    _update_track(folder, state, bench, book_rets, today, days)
    eq = _equity_curve(_load(folder / "track.json"), _load(folder / "rebalance_log.json"),
                       book_id, list(bench_entry))
    primary = next(iter(bench_entry), None)
    metrics = _metrics_from_eq(eq, book_rets.get(book_id), bench, primary)
    metrics["hhi"] = sum(v * v for v in sectors.values()) if sectors else None
    comparables = _comparable_rows(book, state, book_rets.get(book_id), days)
    metrics["day_change"] = book_day
    metrics["bench_day_change"] = {bm: (px[bm] / prev[bm] - 1.0) if (px.get(bm) and prev.get(bm)) else None
                                   for bm in bench_entry}
    _write_status(folder, book, state, book_id, rows, sectors, book_rets.get(book_id),
                  bench_entry, bench, px, metrics, comparables, today, days,
                  cap=cap_eff, cap_basis=cap_basis)
    try:
        news_n = len(_write_news(folder, {book_id: books[book_id]}, today))
    except Exception as exc:  # noqa: BLE001 -- news is best-effort, never blocks the update
        print(f"  news fetch failed: {exc}")
        news_n = 0
    try:  # logos (cached; only first run downloads) -- tiles, table, news wire, compare panel
        logo_syms = sorted((set(syms.values()) | set(bench_entry)
                            | {c.get("symbol") for c in COMPARABLES.get(book, []) if c.get("symbol")})
                           - {s for s in bench_entry if s.startswith("^")})
        logo_n = signals.fetch_logos(logo_syms, folder / "figures" / "logos")
        print(f"  logos: {logo_n}/{len(logo_syms)} available")
    except Exception as exc:  # noqa: BLE001
        print(f"  logo fetch failed: {exc}")
    state_view = dict(state)  # presentation copy only; disk state.json untouched
    state_view["capital"] = cap_eff
    out = _build_dashboard(book, state_view, book_id, rows, sectors, book_rets, bench, metrics, eq, days, today, folder)
    port = book_rets.get(book_id)
    if port is not None and primary is not None and bench.get(primary) is not None:
        print(f"{book}: portfolio {port:.2%}  {primary} {bench[primary]:.2%}  "
              f"alpha {metrics['excess'].get(primary):+.2%}  ({news_n} news items)")
    else:
        print(f"{book}: priced (no return yet)")
    print(f"wrote {out}, status.json, news.json")


def run_manage_all() -> None:
    """`manage update-all`: refresh every active managed book (prices, track, status, dashboard)."""
    reg = registry.load()
    active = [p["name"] for p in reg["portfolios"] if p.get("status") == "active"]
    if not active:
        raise SystemExit("no active managed portfolios in the registry.")
    for name in active:
        print(f"=== {name} ===")
        try:
            run_manage(name, action="update")
        except SystemExit as exc:  # keep going; one book's failure must not strand the others
            print(f"  {name}: {exc}")


def _rebalance(book, folder, book_id, state, dry: bool):
    if not signals.SNAP.exists():
        raise SystemExit(f"{signals.SNAP} missing; run snapshot first.")
    snap_asof = dt.datetime.fromtimestamp(signals.SNAP.stat().st_mtime).date().isoformat()
    new_book = signals.rank_books()[book_id]
    cur_book = state["books"][book_id]
    prior = [r["ticker"] for r in cur_book]
    newt = [r["ticker"] for r in new_book]
    adds = [t for t in newt if t not in set(prior)]
    drops = [t for t in prior if t not in set(newt)]
    holds = [t for t in newt if t in set(prior)]
    w_old, w_new = {t: 1 / len(prior) for t in prior}, {t: 1 / len(newt) for t in newt}
    turnover = 0.5 * sum(abs(w_new.get(t, 0) - w_old.get(t, 0)) for t in set(prior) | set(newt))
    cur = signals.live_prices(sorted(set(prior) | set(BENCHMARKS)))
    realized = _book_return(cur_book, cur)
    bench = _bench_returns(_benchmarks_of(state), cur)
    cum0, _ = _cum_anchor(_load(folder / "rebalance_log.json"))
    new_cum = cum0 * (1 + realized) if realized is not None else cum0
    print(f"Rebalance {'PREVIEW' if dry else 'EXECUTE'} -- {book} ({book_id}, snapshot {snap_asof})")
    print(f"  realized return since last: {realized:.2%}" if realized is not None else "  realized: n/a")
    print(f"  turnover {turnover:.1%}  adds {len(adds)} {adds}  drops {len(drops)} {drops}  holds {len(holds)}")
    if dry:
        (folder / "rebalance_preview.json").write_text(json.dumps(
            {"asof": dt.date.today().isoformat(), "snapshot_asof": snap_asof, "adds": adds, "drops": drops,
             "holds": holds, "turnover_1sided": round(turnover, 4), "realized_return_since_last": realized,
             "would_be_cumulative_nav": new_cum, "benchmark_returns_since_last": bench}, indent=2), encoding="utf-8")
        print("  [dry-run] wrote rebalance_preview.json; no live state changed. Re-run `manage update` to refresh the dashboard.")
        return
    # execute
    (folder / "books").mkdir(exist_ok=True)
    stamp = state.get("last_rebalance") or state["inception"]
    (folder / "books" / f"book_{stamp}.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    fresh = signals.live_prices(sorted(set(newt) | set(BENCHMARKS)))
    for r in new_book:
        r["entry_price"] = fresh.get(r["ticker"])
    state.update({"last_rebalance": dt.date.today().isoformat(),
                  "rebalance_count": int(state.get("rebalance_count", 0)) + 1, "snapshot_asof": snap_asof,
                  "benchmark_entry": {b: fresh.get(b) for b in BENCHMARKS}, "books": {book_id: new_book}})
    (folder / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    log = _load(folder / "rebalance_log.json", {"inception": state["inception"], "rows": []})
    log["rows"].append({"date": dt.date.today().isoformat(), "adds": adds, "drops": drops,
                        "turnover_1sided": round(turnover, 4), "realized_return_since_last": realized,
                        "cumulative_nav": new_cum, "benchmark_returns_since_last": bench})
    (folder / "rebalance_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (folder / "track.json").write_text(json.dumps({"inception": state["inception"],
                                       "last_rebalance": state["last_rebalance"], "rows": []}, indent=2), encoding="utf-8")
    print(f"  rebalanced: cumulative NAV {cum0:.4f} -> {new_cum:.4f}; archived prior book. Run `manage update` to refresh.")
