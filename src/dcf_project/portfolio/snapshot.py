"""Rebuild the live universe snapshot -- ported from scripts/build_live_snapshot.py (logic unchanged).

Values each universe ticker with today's data; caches {discount, sector, intrinsic, price, regime}
to reports/_live_universe_snapshot.json. ~15-25 min for the full ~506-name universe.
"""
from __future__ import annotations

import json
from dataclasses import replace

from dcf_project.core import DCFPipeline, get_settings
from dcf_project.paths import REPORTS


def build_snapshot() -> dict:
    universe = json.load(open(REPORTS / "_universe_tickers.json"))
    tickers = universe["tickers"]
    family_of = universe["family_of"]
    settings = replace(get_settings(), run_sensitivity=False, monte_carlo_samples=0, scenario_monte_carlo_samples=0)
    snapshot: dict[str, dict] = {}
    skips: dict[str, str] = {}
    n = len(tickers)
    out_path = REPORTS / "_live_universe_snapshot.json"
    for i, ticker in enumerate(tickers, 1):
        try:
            res = DCFPipeline(settings).run(ticker)
            diag = res.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
            iv = float(res.intrinsic_value_per_share)
            iv_dcf = float(diag) if isinstance(diag, (int, float)) else iv
            price = float(res.market_price) if res.market_price else None
            if not price or price <= 0:
                skips[ticker] = "no_price"
                continue
            snapshot[ticker] = {"discount": iv_dcf / price - 1.0, "sector": family_of.get(ticker, "unknown"),
                                "intrinsic": round(iv, 2), "price": round(price, 2), "regime": res.valuation_regime}
        except Exception as exc:  # noqa: BLE001
            skips[ticker] = f"{type(exc).__name__}"
        if i % 50 == 0:
            print(f"[{i}/{n}] valued={len(snapshot)} skipped={len(skips)}", flush=True)
            out_path.write_text(json.dumps({"snapshot": snapshot, "skips": skips, "partial": True}, indent=2), encoding="utf-8")
    out_path.write_text(json.dumps({"snapshot": snapshot, "skips": skips, "partial": False,
                                    "n_valued": len(snapshot), "n_universe": n}, indent=2), encoding="utf-8")
    print(f"done: valued {len(snapshot)}/{n}; skipped {len(skips)}")
    return {"n_valued": len(snapshot), "n_universe": n, "skips": len(skips)}
