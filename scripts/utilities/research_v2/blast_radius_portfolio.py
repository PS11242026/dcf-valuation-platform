"""Blast-radius re-validation: the conf_top10 / discount portfolio alpha under split correction.

Re-runs the quarterly top-10 portfolio simulator on the ORIGINAL vs SPLIT-CORRECTED discount panel
(same selection logic, |discount|>5 clip, costs). Reports CAGR / Sharpe / annualized active return vs
SPY for signal='discount' and 'discount_x_conf'. Analytic re-run -- patches the signal loader; no MCMC;
frozen artifacts untouched.

Note: discount_x_conf's per-sector 'confidence' weight is loaded from the (uncorrected)
final_institutional_validation.json; a full correction would also recompute sector reliability, so the
discount_x_conf row is an upper bound on what survives.

Usage: PYTHONPATH=src python scripts/utilities/research_v2/blast_radius_portfolio.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scripts.portfolio.portfolio_sim as PS  # noqa: E402

CLIP = PS.DISCOUNT_CLIP
ORIG = json.loads((ROOT / "reports" / "final_universe_backtest.json").read_text())["observations"]
CORR = json.loads((ROOT / "reports" / "research_v2" / "final_universe_backtest_split_corrected.json"
                   ).read_text())["observations"]


def _panel(obs):
    panel: dict[str, dict[str, dict]] = {}
    for o in obs:
        d = o.get("discount_to_intrinsic")
        if not isinstance(d, (int, float)) or abs(d) > CLIP:
            continue
        panel.setdefault(o["date"], {})[o["ticker"]] = {"discount": d, "sector": o["sector"]}
    return sorted(panel), panel


def _stats(r, spy):
    r = r.dropna()
    n = len(r)
    if n < 4:
        return None
    cagr = float(np.prod(1.0 + r.values) ** (4.0 / n) - 1.0)
    sharpe = float(r.mean() / r.std() * np.sqrt(4)) if r.std() > 0 else None
    spy = spy.reindex(r.index).dropna()
    act = (r.reindex(spy.index) - spy).dropna()
    spy_cagr = float(np.prod(1.0 + spy.values) ** (4.0 / len(spy)) - 1.0) if len(spy) else None
    ann_active = float(act.mean() * 4.0) if len(act) else None
    # CAPM (beta-adjusted) annualized alpha: port_q = alpha_q + beta*spy_q
    capm_alpha = capm_beta = None
    common = r.index.intersection(spy.index)
    if len(common) >= 8:
        beta, alpha_q = np.polyfit(spy.reindex(common).values, r.reindex(common).values, 1)
        capm_alpha = float(alpha_q * 4.0)
        capm_beta = float(beta)
    return {"n_q": n, "cagr": cagr, "sharpe": sharpe, "spy_cagr": spy_cagr,
            "cagr_minus_spy": (cagr - spy_cagr) if spy_cagr is not None else None,
            "ann_active_return": ann_active, "capm_alpha_ann": capm_alpha, "capm_beta": capm_beta}


def _run(obs, signal):
    dates, panel = _panel(obs)
    PS.load_signal_panel = lambda: (dates, panel)     # patch the signal source
    res = PS.simulate(PS.SimConfig(signal=signal, top_n=10, cost_bps=10.0))
    spy = res.bench.get("SPY")
    return _stats(res.long_net, spy)


def main() -> None:
    out = {}
    print(f"{'signal':>16} {'variant':>10} {'nq':>4} {'CAGR':>8} {'Sharpe':>7} {'CAGR-SPY':>9} "
          f"{'CAPMalpha':>10} {'beta':>6}")
    for signal in ("discount", "discount_x_conf"):
        for label, obs in (("original", ORIG), ("corrected", CORR)):
            s = _run(obs, signal)
            out[f"{signal}:{label}"] = s
            if s is None:
                print(f"{signal:>16} {label:>10}  insufficient")
                continue
            def f(x, p=4):
                return f"{x:+.{p}f}" if isinstance(x, (int, float)) else "n/a"
            print(f"{signal:>16} {label:>10} {s['n_q']:>4} {f(s['cagr']):>8} {f(s['sharpe'],2):>7} "
                  f"{f(s['cagr_minus_spy']):>9} {f(s['capm_alpha_ann']):>10} {f(s['capm_beta'],2):>6}")
    p = ROOT / "reports" / "research_v2" / "blast_radius_portfolio.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
