"""Live progress dashboard for the tri-engine PIT backtest (read-only).

Reads the per-ticker checkpoints that ``tri_engine_backtest.py`` writes as it runs
(``reports/research_v3/_checkpoints/tri_{scope}/{TICKER}.json``) and renders a snapshot:
how far along, throughput + ETA, outcome/skip breakdown, observation + guard tallies,
and per-family coverage. Pure stdlib, opens nothing the live run holds a lock on, and
never writes -- safe to run any number of times while the backtest is in flight.

Usage:
    python scripts/utilities/research_v3/tri_progress.py                 # one snapshot (broad)
    python scripts/utilities/research_v3/tri_progress.py --watch 30      # refresh every 30s
    python scripts/utilities/research_v3/tri_progress.py --scope n50     # a smaller run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "research_v3"
HORIZONS = (1, 3, 5)


def _fmt_dur(secs: float) -> str:
    secs = max(int(secs), 0)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _load_universe(scope: str) -> tuple[list[str], dict[str, str]]:
    """Ordered ticker list + family map. Broad reads the frozen universe; other scopes
    fall back to the checkpoint set itself (denominator = whatever has been processed)."""
    meta = json.loads((ROOT / "reports" / "_universe_tickers.json").read_text(encoding="utf-8"))
    if scope == "broad":
        return list(meta["tickers"]), meta["family_of"]
    return [], meta["family_of"]


def snapshot(scope: str) -> str:
    ckpt_dir = REPORT_DIR / "_checkpoints" / f"tri_{scope}"
    if not ckpt_dir.exists():
        return f"no checkpoint dir yet: {ckpt_dir}"
    universe, family_of = _load_universe(scope)
    files = sorted(ckpt_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    done_set = {p.stem for p in files}
    total = len(universe) if universe else len(files)

    # --- per-ticker rollup ----------------------------------------------------
    obs_total = guard_total = 0
    ok_with_obs = ok_zero = 0
    out_failed: dict[str, int] = {}
    out_skipped: dict[str, int] = {}
    skips: dict[str, int] = {}
    guard_by_ticker: dict[str, int] = {}
    fam_obs: dict[str, int] = {}
    fam_tickers: dict[str, set] = {}
    horizon_cov = {y: 0 for y in HORIZONS}

    for p in files:
        try:
            pay = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # mid-write; will be valid next refresh
        rows = pay.get("rows", [])
        obs_total += len(rows)
        outcome = pay.get("outcome", "")
        if outcome.startswith("ok"):
            (ok_with_obs := ok_with_obs + 1) if rows else (ok_zero := ok_zero + 1)
        elif outcome.startswith("failed"):
            out_failed[outcome] = out_failed.get(outcome, 0) + 1
        elif outcome.startswith("skipped"):
            out_skipped[outcome] = out_skipped.get(outcome, 0) + 1
        for k, v in pay.get("skips", {}).items():
            skips[k] = skips.get(k, 0) + v
        gp = pay.get("pathological", [])
        guard_total += len(gp)
        if gp:
            guard_by_ticker[pay.get("ticker", p.stem)] = len(gp)
        for r in rows:
            fam = r.get("sector") or family_of.get(pay.get("ticker", ""), "unknown")
            fam_obs[fam] = fam_obs.get(fam, 0) + 1
            fam_tickers.setdefault(fam, set()).add(r["ticker"])
            for y in HORIZONS:
                if f"fwd_{y}y" in r:
                    horizon_cov[y] += 1

    # --- throughput / ETA -----------------------------------------------------
    now = time.time()
    if len(files) >= 2:
        span = files[-1].stat().st_mtime - files[0].stat().st_mtime
        per = span / max(len(files) - 1, 1)
    else:
        span = per = 0.0
    remaining = max(total - len(files), 0)
    eta = remaining * per if per else 0.0
    last_age = now - files[-1].stat().st_mtime if files else 0.0
    next_name = next((t for t in universe if t not in done_set), None) if universe else None

    # --- render ---------------------------------------------------------------
    pct = 100.0 * len(files) / total if total else 0.0
    bar_w = 30
    filled = int(bar_w * len(files) / total) if total else 0
    bar = "#" * filled + "-" * (bar_w - filled)
    L: list[str] = []
    L.append(f"=== tri-engine backtest progress (scope={scope}) @ {datetime.now():%H:%M:%S} ===")
    L.append(f"  [{bar}] {len(files)}/{total}  ({pct:4.1f}%)")
    L.append(f"  rate: {per:5.1f}s/name over {_fmt_dur(span)}   "
             f"ETA: {_fmt_dur(eta)} for {remaining} left   "
             f"last ckpt: {_fmt_dur(last_age)} ago"
             + (f"  -> next: {next_name}" if next_name else ""))
    if last_age > 600:
        L.append(f"  ** WARNING: no new checkpoint in {_fmt_dur(last_age)} -- run may be stalled/finished **")
    L.append("")
    L.append(f"  observations: {obs_total:6d}     guarded blow-ups: {guard_total}")
    L.append(f"  outcomes: ok+data={ok_with_obs}  ok:0(no obs)={ok_zero}  "
             f"failed={sum(out_failed.values())}  skipped={sum(out_skipped.values())}")
    L.append("  fwd-return coverage: " + "  ".join(f"{y}y={horizon_cov[y]}" for y in HORIZONS))

    if fam_obs:
        L.append("")
        L.append("  per-family coverage (obs / tickers-with-data):")
        for fam in sorted(fam_obs, key=lambda f: -fam_obs[f]):
            L.append(f"    {fam:34} {fam_obs[fam]:6d} obs / {len(fam_tickers[fam]):3d} names")

    top_skips = sorted(skips.items(), key=lambda kv: -kv[1])[:8]
    if top_skips:
        L.append("")
        L.append("  top skip reasons: " + ", ".join(f"{k}={v}" for k, v in top_skips))
    if guard_by_ticker:
        worst = sorted(guard_by_ticker.items(), key=lambda kv: -kv[1])[:6]
        L.append("  worst guard names: " + ", ".join(f"{k}({v})" for k, v in worst))
    if out_failed:
        L.append("  failures: " + ", ".join(f"{k.split(':',1)[-1]}={v}" for k, v in
                                             sorted(out_failed.items(), key=lambda kv: -kv[1])))

    done_flag = REPORT_DIR / f"tri_engine_summary_{scope}.json"
    if done_flag.exists() and last_age > 120:
        L.append("")
        L.append(f"  ** summary written: {done_flag.name} -- run appears COMPLETE **")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", default="broad", help="run scope (broad, n50, ...)")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="refresh every N seconds (0 = single snapshot)")
    args = ap.parse_args()

    if args.watch <= 0:
        print(snapshot(args.scope))
        return 0
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(snapshot(args.scope))
            print(f"\n  (watching every {args.watch:.0f}s -- Ctrl-C to stop)")
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
