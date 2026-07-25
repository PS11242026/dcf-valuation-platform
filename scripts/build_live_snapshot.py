"""Value today's investable universe -> cache current discounts for live cross-sectional ranking.

The scorecard's Expected Return Rank should answer "how attractive is this name vs the other
stocks I can buy TODAY", not "vs all opportunities since 2015". That requires a current
cross-section of discount_to_intrinsic. This script values each universe ticker with today's
data and caches {ticker: discount, sector, intrinsic, price} to reports/_live_universe_snapshot.json.

The discount (intrinsic/price - 1) is deterministic and independent of Monte-Carlo / sensitivity,
so we disable those for speed -- the cached discount matches what the scorecard's score_ticker
produces. No production valuation logic is modified. Run in the background (~15-25 min).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcf_project.config import get_settings  # noqa: E402
from dcf_project.pipeline import DCFPipeline  # noqa: E402

REPORT_DIR = ROOT / "reports"


def main() -> None:
    universe = json.load(open(REPORT_DIR / "_universe_tickers.json"))
    tickers = universe["tickers"]
    family_of = universe["family_of"]
    settings = replace(get_settings(), run_sensitivity=False, monte_carlo_samples=0,
                       scenario_monte_carlo_samples=0)
    snapshot: dict[str, dict] = {}
    skips: dict[str, str] = {}
    n = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        try:
            res = DCFPipeline(settings).run(ticker)
            diag = res.diagnostics.get("state_space_dcf_diagnostic_intrinsic")
            iv = float(res.intrinsic_value_per_share)
            iv_dcf = float(diag) if isinstance(diag, (int, float)) else iv  # distortion-free DCF intrinsic
            price = float(res.market_price) if res.market_price else None
            if not price or price <= 0:
                skips[ticker] = "no_price"
                continue
            snapshot[ticker] = {
                "discount": iv_dcf / price - 1.0,
                "sector": family_of.get(ticker, "unknown"),
                "intrinsic": round(iv, 2),
                "price": round(price, 2),
                "regime": res.valuation_regime,
            }
        except Exception as exc:
            skips[ticker] = f"{type(exc).__name__}"
        if i % 50 == 0:
            print(f"[{i}/{n}] valued={len(snapshot)} skipped={len(skips)}", flush=True)
            (REPORT_DIR / "_live_universe_snapshot.json").write_text(
                json.dumps({"snapshot": snapshot, "skips": skips, "partial": True}, indent=2), encoding="utf-8")
    (REPORT_DIR / "_live_universe_snapshot.json").write_text(
        json.dumps({"snapshot": snapshot, "skips": skips, "partial": False,
                    "n_valued": len(snapshot), "n_universe": n}, indent=2), encoding="utf-8")
    print(f"done: valued {len(snapshot)}/{n}; skipped {len(skips)}")


if __name__ == "__main__":
    main()
