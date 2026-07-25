"""Portfolio registry -- the Proposal->Managed audit trail.

reports/registry/portfolios.json tracks every managed (live) portfolio with its inception,
starting capital, strategy + signal version, and source build. `promote` is the ONLY path from a
proposal build to a managed portfolio, and it is explicit + recorded. The pre-existing live paper
book is seeded as `conf_top10` without altering its data.
"""
from __future__ import annotations

import datetime as dt
import json

from dcf_project.paths import MANAGED, PORTFOLIO_BUILDS, REGISTRY, REPO_ROOT
from dcf_project.portfolio.signals import BENCHMARKS, live_prices

REGISTRY_FILE = REGISTRY / "portfolios.json"
SIGNAL_VERSION = "discount_x_conf@paper-frozen-2026-06-05"
STRATEGY_VERSION = "ew-topN-v1"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _rel(p) -> str:
    return str(p.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def save(reg: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def load() -> dict:
    reg = json.load(open(REGISTRY_FILE)) if REGISTRY_FILE.exists() else {"portfolios": [], "events": []}
    _ensure_seed(reg)
    return reg


def _ensure_seed(reg: dict) -> None:
    names = {p["name"] for p in reg["portfolios"]}
    dirty = False
    for p in reg["portfolios"]:
        # Presentation notional for value/NAV display on books with no booked capital
        # (the frozen paper book state is never mutated; this lives in the registry).
        if p["name"] == "conf_top10" and "display_capital" not in p:
            p["display_capital"] = 1_000_000.0
            dirty = True
    conf = MANAGED / "conf_top10"
    if "conf_top10" not in names and (conf / "state.json").exists():
        state = json.load(open(conf / "state.json"))
        reg["portfolios"].append({
            "name": "conf_top10", "book_id": "discount_x_conf_top10",
            "source_build": "(migrated live paper book)",
            "inception_date": state.get("inception", "2026-06-04"), "starting_capital": None,
            "display_capital": 1_000_000.0,
            "strategy_version": "ew-top10-v1", "signal_version": SIGNAL_VERSION,
            "folder": _rel(conf), "status": "active", "created_utc": _now(),
        })
        reg["events"].append({"ts": _now(), "event": "seed", "name": "conf_top10",
                              "note": "seeded from migrated live paper book (data unchanged)"})
        dirty = True
    ind = MANAGED / "india_v3_aggressive"
    if "india_v3_aggressive" not in names and (ind / "state.json").exists():
        state = json.load(open(ind / "state.json"))
        reg["portfolios"].append({
            "name": "india_v3_aggressive", "book_id": "holdings",
            "source_build": "(seeded standalone India v3 book)",
            "inception_date": state.get("inception"), "starting_capital": state.get("capital"),
            "strategy_version": state.get("strategy", "conviction-topN"),
            "signal_version": state.get("signal", "engine_v3_discount@india_phase_b"),
            "folder": _rel(ind), "status": "active", "rebalance": "manual",
            "currency": state.get("currency", "INR"), "created_utc": _now(),
        })
        reg["events"].append({"ts": _now(), "event": "seed", "name": "india_v3_aggressive",
                              "note": "seeded standalone India v3 book into the manager (data unchanged)"})
        dirty = True
    if dirty:
        save(reg)


def get(name: str) -> dict | None:
    for p in load()["portfolios"]:
        if p["name"] == name:
            return p
    return None


def available_builds() -> list[str]:
    if not PORTFOLIO_BUILDS.exists():
        return []
    return sorted(d.name for d in PORTFOLIO_BUILDS.iterdir() if (d / "build.json").exists())


def run_list() -> None:
    reg = load()
    print("Managed portfolios (registry):")
    if not reg["portfolios"]:
        print("  (none)")
    for p in reg["portfolios"]:
        cap = f"${p['starting_capital']:,.0f}" if p.get("starting_capital") else "notional"
        print(f"  {p['name']:<16} book={p['book_id']:<22} inception={p['inception_date']} "
              f"capital={cap} status={p['status']} signal={p['signal_version']}")
    builds = available_builds()
    print(f"\nProposal builds available to promote ({len(builds)}):")
    for b in builds:
        print(f"  {b}")
    if not builds:
        print("  (none -- run `portfolio construct` first)")


def run_promote(build_id: str) -> dict:
    build_dir = PORTFOLIO_BUILDS / build_id
    bf = build_dir / "build.json"
    if not bf.exists():
        raise SystemExit(f"No build '{build_id}' at {bf}. Run `portfolio construct` first; see `portfolio list`.")
    build = json.load(open(bf))
    name = build.get("managed_name") or build_id
    reg = load()
    if any(p["name"] == name and p["status"] == "active" for p in reg["portfolios"]):
        raise SystemExit(f"Managed portfolio '{name}' already exists and is active -- refusing to clobber a live book.")

    folder = MANAGED / name
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    holdings = build["holdings"]
    book_id = build["signal"]
    tickers = sorted({h["ticker"] for h in holdings} | set(BENCHMARKS))
    print(f"locking entry prices for {len(tickers)} names...")
    prices = live_prices(tickers)
    today = dt.date.today().isoformat()
    book = []
    for h in holdings:
        book.append({"ticker": h["ticker"], "discount": h.get("discount"), "sector": h.get("sector"),
                     "intrinsic": h.get("intrinsic"), "snap_price": h.get("price"), "conf": h.get("conviction"),
                     "entry_price": prices.get(h["ticker"])})
    state = {
        "inception": today, "snapshot_asof": build.get("created_utc", today)[:10],
        "benchmark_entry": {b: prices.get(b) for b in BENCHMARKS},
        "books": {book_id: book}, "capital": build.get("capital"),
        "source_build": build_id, "strategy_version": STRATEGY_VERSION, "signal_version": SIGNAL_VERSION,
        "note": f"Promoted from proposal {build_id}. Live out-of-sample record; rebalance only on explicit command.",
    }
    (folder / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (folder / "track.json").write_text(json.dumps({"inception": today, "rows": []}, indent=2), encoding="utf-8")

    reg["portfolios"].append({
        "name": name, "book_id": book_id, "source_build": build_id, "inception_date": today,
        "starting_capital": build.get("capital"), "strategy_version": STRATEGY_VERSION,
        "signal_version": SIGNAL_VERSION, "folder": _rel(folder), "status": "active", "created_utc": _now(),
    })
    reg["events"].append({"ts": _now(), "event": "promote", "name": name, "source_build": build_id,
                          "capital": build.get("capital")})
    save(reg)
    print(f"promoted '{build_id}' -> managed portfolio '{name}' ({folder}). "
          f"Run `portfolio manage update --book {name}` to track it.")
    return state
