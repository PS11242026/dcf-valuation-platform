"""PRODUCT 2 -- Portfolio constructor (PROPOSAL only).

Validated signal -> equal-weight top-N book + capital allocation + expected alpha/risk/capacity
(cited from the validated backtest) -> an investment-committee workbook in reports/portfolio_builds/.
HARD GUARD: writes ONLY under portfolio_builds/. Never touches the live book, registry, or
managed_portfolios/. Promotion to live is a separate, explicit `portfolio promote`.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from dcf_project.paths import PORTFOLIO_BUILDS
from dcf_project.portfolio import analytics, signals
from dcf_project.reporting_kit import (
    Image, Metric, ReportModel, Table, save_model, viz3d, xlsx,
)
from dcf_project.portfolio.scoring import sector_label


def _quarter_id(d: dt.date | None = None) -> str:
    d = d or dt.date.today()
    return f"portfolio_{d.year}Q{(d.month - 1) // 3 + 1}"


def _managed_name(signal: str) -> str:
    return signal.replace("discount_x_conf", "conf").replace("discount", "disc")


def construct(signal: str = "discount_x_conf_top10", capital: float | None = None) -> dict:
    holdings = signals.select(signal)
    n = len(holdings)
    w = 1.0 / n if n else 0.0
    prices = signals.live_prices([h["ticker"] for h in holdings])
    recs = []
    for h in holdings:
        px = prices.get(h["ticker"]) or h.get("snap_price")
        shares = (capital * w / px) if (capital and px) else None
        recs.append({
            "ticker": h["ticker"], "sector": h["sector"], "weight": w, "price": px,
            "intrinsic": h.get("intrinsic"), "discount": h.get("discount"), "conviction": h.get("conf"),
            "shares": round(shares, 2) if shares else None,
            "market_value": round(shares * px, 2) if (shares and px) else None,
        })
    return {
        "build_id": _quarter_id(), "managed_name": _managed_name(signal), "signal": signal,
        "capital": capital, "holdings": recs, "profile": analytics.expected_profile(),
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def _sector_weights(recs) -> dict:
    agg: dict[str, float] = {}
    for r in recs:
        agg[r["sector"]] = agg.get(r["sector"], 0.0) + r["weight"]
    return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))


def _build_model(build: dict, folder: Path) -> ReportModel:
    figs = folder / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    recs = build["holdings"]
    prof = build["profile"]
    capital = build["capital"]
    sw = _sector_weights(recs)
    m = ReportModel(
        title=f"Portfolio Proposal -- {build['signal']}",
        subtitle=f"Equal-weight top-{len(recs)} on the validated DCF discount signal ({build['build_id']})",
        persona="ic",
        meta={"Build": build["build_id"], "Signal": build["signal"], "Suggested name": build["managed_name"],
              "Capital": (f"${capital:,.0f}" if capital else "weights only"), "Created": build["created_utc"],
              "Status": "PROPOSAL (not live until promoted)"},
    )
    # 1) Summary
    s = m.section("Portfolio Summary", "Portfolio Summary")
    s.metrics = [
        Metric("Capital", capital, "usd0") if capital else Metric("Holdings", len(recs), "num0"),
        Metric("Expected return (CAGR)", prof.get("expected_return_cagr"), "pct"),
        Metric("Expected alpha", prof.get("expected_alpha"), "pct", good=True),
        Metric("Expected volatility", prof.get("expected_volatility"), "pct"),
        Metric("Expected Sharpe", prof.get("expected_sharpe"), "num"),
        Metric("Expected max DD", prof.get("expected_max_drawdown"), "pct", good=False),
        Metric("Annual turnover", prof.get("expected_turnover_annual"), "pct"),
        Metric("Holdings", len(recs), "num0"),
    ]
    s.narrative = [
        f"Top-{len(recs)} names by {build['signal']} from the live valuation snapshot; equal-weighted.",
        "Expected return/alpha/risk are the VALIDATED backtest figures for this signal (not promises).",
        "This is a PROPOSAL. It does not become a live portfolio until explicitly promoted.",
    ]
    # 2) Holdings
    hsec = m.section("Holdings", "Holdings")
    hsec.tables.append(Table(None, ["Ticker", "Sector", "Weight", "Price", "Shares", "Market value", "Discount", "Conviction"],
        [[r["ticker"], sector_label(r["sector"]), r["weight"], r["price"], r["shares"], r["market_value"],
          r["discount"], r["conviction"]] for r in recs],
        formats=["text", "text", "pct", "usd", "num", "usd0", "pct", "num"]))
    # 3) Weighting logic
    wsec = m.section("Weighting Logic", "Weighting Logic")
    wsec.narrative = [
        "Equal weight (1/N). The allocation study found no scheme reliably beats equal-weight top-N on alpha;",
        "vol-weighting significantly LOSES (the discount return is carried by higher-vol names).",
        "Sector caps are available as a risk control but are not applied to this base proposal.",
    ]
    # 4) Sector exposure
    secx = m.section("Sector Exposure", "Sector Exposure")
    secx.tables.append(Table("Sector weights", ["Sector", "Weight"],
                             [[sector_label(k), v] for k, v in sw.items()], formats=["text", "pct"]))
    img = viz3d.sector_bar({sector_label(k): v for k, v in sw.items()}, figs / "sector.png", title="Sector exposure")
    if img:
        secx.images.append(Image("Sector allocation", img))
    a3d = viz3d.bar3d_allocation([r["sector"] for r in recs], [r["discount"] or 0 for r in recs],
                                 [r["weight"] for r in recs], figs / "alloc3d.png")
    if a3d:
        secx.images.append(Image("3D allocation (sector x discount x weight)", a3d))
    # 5) Expected alpha attribution (validated: alpha is selection-driven)
    asec = m.section("Alpha Attribution", "Expected Alpha Attribution")
    cu = (prof.get("expected_return_cagr") or 0) - (prof.get("expected_alpha") or 0)
    parts = {"Market (universe)": cu, "Selection": (prof.get("expected_alpha") or 0), "Sector allocation": 0.0}
    wf = viz3d.alpha_decomposition(parts, figs / "alpha.png", title="Expected return decomposition")
    if wf:
        asec.images.append(Image("Alpha decomposition", wf))
    asec.narrative = ["Validated attribution: the edge is broad stock SELECTION across sectors, not sector timing."]
    # 6) Risk analysis
    rsec = m.section("Risk Analysis", "Risk Analysis")
    rsec.metrics = [Metric("Expected volatility", prof.get("expected_volatility"), "pct"),
                    Metric("Expected max DD", prof.get("expected_max_drawdown"), "pct", good=False),
                    Metric("3Y IC", prof.get("ic_3y"), "num"), Metric("3Y long-short", prof.get("long_short_3y"), "pct")]
    pts = [{"name": r["ticker"], "vol": max(0.02, 1.0 - (r["conviction"] or 0.5)), "ret": (r["discount"] or 0),
            "weight": r["weight"], "sector": sector_label(r["sector"])} for r in recs]
    bub = viz3d.risk_return_bubble(pts, figs / "bubble.png",
                                   title="Holdings: discount (y) vs 1-conviction (x), size=weight")
    if bub:
        rsec.images.append(Image("Risk / return map", bub))
    conc = viz3d.concentration([r["weight"] for r in recs], figs / "conc.png")
    if conc:
        rsec.images.append(Image("Concentration", conc))
    # 7) Capacity
    csec = m.section("Capacity Analysis", "Capacity Analysis")
    ceil = prof.get("capacity_ceilings")
    csec.narrative = [f"Validated capacity ceilings: {json.dumps(ceil)[:300]}" if ceil
                      else "Capacity study not available in this build."]
    # 8) Trade list
    tsec = m.section("Trade List", "Trade List")
    tsec.tables.append(Table("Trades to establish the book", ["Ticker", "Action", "Shares", "Price", "Market value", "Weight"],
        [[r["ticker"], "BUY", r["shares"], r["price"], r["market_value"], r["weight"]] for r in recs],
        formats=["text", "text", "num", "usd", "usd0", "pct"], total_row=False))
    return m


def run_construct(signal: str = "discount_x_conf_top10", capital: float | None = None) -> Path:
    build = construct(signal=signal, capital=capital)
    folder = PORTFOLIO_BUILDS / build["build_id"]
    folder.mkdir(parents=True, exist_ok=True)
    model = _build_model(build, folder)
    out = xlsx.render(model, folder / f"{build['build_id']}.xlsx")
    save_model(model, folder)
    (folder / "build.json").write_text(json.dumps(build, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    print(f"PROPOSAL only -- to make it live: python -m dcf_project portfolio promote {build['build_id']}")
    return out
