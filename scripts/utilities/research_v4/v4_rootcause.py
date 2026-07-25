"""Research_V4 Phase C -- root-cause quantification (analysis only; NO engine changes).

For each suspected economic cause of the systematic downward valuation gap, re-price every
clean name under a SINGLE counterfactual driver value (using the engine's own frame+value
functions via v4_reverse.EngineContext) and measure the median change in the valuation gap.
This delivers the plan's "quantify its contribution / estimate expected impact BEFORE
implementation" requirement -- one lever at a time, nothing tuned, nothing committed.

Levers probed (each a future opt-in Settings flag, default off -- see v4_recommended_changes):
  growth_anchor      g_base -> max(g_base, consensus_revenue_growth)   (forward-looking anchor)
  margin_no_compress margin_target -> margin_current                    (drop sector-anchor pull)
  moat_premium       moat cap 2%% -> 4%% (excess_terminal doubled, capped) (longer-lived ROIC)
  terminal_growth    g_terminal + 50 bps                               (less terminal conservatism)

Output: reports/research_v4/v4_rootcause.json. The KEEP/REJECT of any of these is decided
NEXT round on the corrected forward-return PIT panel, not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "src", str(Path(__file__).resolve().parent)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import universe  # noqa: E402
import v4_reverse  # noqa: E402
from v4_analysis import add_gaps, is_pathological, load_rows  # noqa: E402
from v4_snapshot import _settings  # noqa: E402

from dcf_project.engine_v3.drivers import DriverExtractionError  # noqa: E402
from dcf_project.engine_v3.pipeline import EconomicDCFPipeline  # noqa: E402
from dcf_project.valuation.terminal import TerminalValueError  # noqa: E402

REPORT_DIR = ROOT / "reports" / "research_v4"
MOAT_CAP_DOUBLED = 0.04  # cfg.moat_premium_cap default is 0.02


def _safe(fn: Callable[[], float]) -> float | None:
    try:
        return fn()
    except (TerminalValueError, ValueError, ZeroDivisionError):
        return None
    except Exception:  # noqa: BLE001
        return None


def lever_deltas(ctx: v4_reverse.EngineContext, price: float, snap_row: dict) -> dict[str, Any]:
    """Counterfactual gap change (counterfactual_iv/price - base_iv/price) per lever."""
    base_iv = _safe(ctx.price_per_share)
    if not base_iv or not price:
        return {}
    base_gap = base_iv / price - 1.0
    out: dict[str, Any] = {"base_gap": base_gap}

    # 1. Forward-looking growth anchor: lift g_base to consensus growth when it is higher.
    cons_g = snap_row.get("consensus_revenue_growth")
    if isinstance(cons_g, (int, float)) and cons_g > ctx.g_base_effective:
        iv = _safe(lambda: ctx.price_per_share(g_base=float(cons_g)))
        out["growth_anchor"] = (iv / price - 1.0 - base_gap) if iv else None
    else:
        out["growth_anchor"] = 0.0

    # 2. Drop margin compression: revert margin toward current TTM level, not the sector anchor.
    mc = snap_row.get("margin_current")
    if isinstance(mc, (int, float)):
        iv = _safe(lambda: ctx.price_per_share(margin_target=float(mc)))
        out["margin_no_compress"] = (iv / price - 1.0 - base_gap) if iv else None

    # 3. Moat premium: double the surviving-excess cap (RONIC stays elevated longer).
    excess_now = ctx.ronic_terminal - ctx.wacc
    new_excess = min(max(excess_now, 0.0) * 2.0 or 0.01, MOAT_CAP_DOUBLED)
    iv = _safe(lambda: ctx.price_per_share(ronic_terminal=ctx.wacc + new_excess))
    out["moat_premium"] = (iv / price - 1.0 - base_gap) if iv else None

    # 4. Terminal growth +50bps (still band-clipped inside the engine on the live path).
    iv = _safe(lambda: ctx.price_per_share(terminal_growth=ctx.g_terminal + 0.005))
    out["terminal_growth"] = (iv / price - 1.0 - base_gap) if iv else None

    # 5. Slower growth fade ("fade rates too aggressive"): double the growth half-life,
    #    capped at the engine's 8y ceiling -- the near-term growth premium persists longer.
    from dataclasses import replace as _dc_replace

    hl = ctx.half_lives
    hl_slow = _dc_replace(hl, h_growth_years=min(hl.h_growth_years * 2.0, ctx.cfg.half_life_max_years))
    iv = _safe(lambda: ctx.price_per_share(half_lives=hl_slow))
    out["growth_persistence"] = (iv / price - 1.0 - base_gap) if iv else None
    return out


def main() -> None:
    settings = _settings()
    snap = {r["ticker"]: r for r in add_gaps(load_rows())}
    clean = [t for t, r in snap.items() if not is_pathological(r)]
    sectors = {t: lbl for t, lbl in universe.flat_sample(13)}

    per_name: list[dict[str, Any]] = []
    for i, ticker in enumerate(clean, 1):
        row = snap[ticker]
        price = row.get("market_price")
        try:
            base = EconomicDCFPipeline(settings)
            bundle = base.load_data(ticker)
            ctx = v4_reverse.build_context(settings, bundle, ticker)
        except (DriverExtractionError, TerminalValueError, ValueError):
            continue
        deltas = lever_deltas(ctx, price, row)
        if deltas:
            deltas.update(ticker=ticker, sector=sectors.get(ticker))
            per_name.append(deltas)
        print(f"[{i:>2}/{len(clean)}] {ticker}", flush=True)

    levers = ["growth_anchor", "growth_persistence", "margin_no_compress", "moat_premium", "terminal_growth"]
    summary = {"n": len(per_name), "median_gap_closed_pp": {}, "by_sector": {}}
    for lv in levers:
        vals = [d[lv] for d in per_name if isinstance(d.get(lv), (int, float))]
        summary["median_gap_closed_pp"][lv] = (median(vals) if vals else None)

    by_sec: dict[str, dict[str, Any]] = {}
    for d in per_name:
        by_sec.setdefault(d["sector"], {"names": []})["names"].append(d["ticker"])
    for sec, blob in by_sec.items():
        srows = [d for d in per_name if d["sector"] == sec]
        blob["n"] = len(srows)
        for lv in levers:
            vals = [d[lv] for d in srows if isinstance(d.get(lv), (int, float))]
            blob[lv] = (median(vals) if vals else None)
    summary["by_sector"] = by_sec

    out = {
        "design": "single-lever counterfactual re-pricing of the engine_v3 frame; median change "
                  "in valuation gap (positive = lifts intrinsic toward market). Quantification "
                  "only -- KEEP/REJECT decided next round on the forward-return PIT panel.",
        "levers": {
            "growth_anchor": "g_base -> max(g_base, consensus_revenue_growth)",
            "growth_persistence": "growth half-life doubled (<= 8y ceiling): slower fade",
            "margin_no_compress": "margin_target -> margin_current (drop sector-anchor pull)",
            "moat_premium": f"surviving excess ROIC cap doubled, <= {MOAT_CAP_DOUBLED}",
            "terminal_growth": "g_terminal + 50bps",
        },
        "summary": summary,
        "per_name": per_name,
    }
    (REPORT_DIR / "v4_rootcause.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nMedian gap closed (pp) by lever, overall:")
    for lv in levers:
        v = summary["median_gap_closed_pp"][lv]
        print(f"  {lv:20} {v:+.1%}" if v is not None else f"  {lv:20} n/a")


if __name__ == "__main__":
    main()
