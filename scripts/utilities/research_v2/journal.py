"""Append-only research journal for the engine_v2 (V2) program.

Two canonical, append-only artifacts under reports/research_v2/:
  - research_journal.md    human-readable, one dated section per event
  - research_journal.jsonl one JSON object per line, same schema keys

Nothing here is ever rewritten -- every tranche / run / conclusion is APPENDED so the
full decision history can be reconstructed from the files alone. Use `append_entry(...)`.

Schema (all keys recorded for every entry):
    datetime, hypothesis, sample_size, engine, command, output_artifact,
    metrics, interpretation, decision, next_action

Run directly to (re)seed the known P2.5 history (idempotent guard via a sentinel marker
so it will not double-append on re-run):

    PYTHONPATH=src python scripts/utilities/research_v2/journal.py --seed
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_V2 = ROOT / "reports" / "research_v2"
MD_PATH = RESEARCH_V2 / "research_journal.md"
JSONL_PATH = RESEARCH_V2 / "research_journal.jsonl"

_FIELDS = ["datetime", "hypothesis", "sample_size", "engine", "command",
           "output_artifact", "metrics", "interpretation", "decision", "next_action"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _md_section(entry: dict[str, Any]) -> str:
    title = entry.get("title") or f"{entry['sample_size']} -- {entry['decision']}"
    lines = [f"## {entry['datetime']} -- {title}", ""]
    for f in _FIELDS:
        v = entry.get(f)
        if isinstance(v, dict):
            lines.append(f"- **{f}**:")
            for k, sub in v.items():
                lines.append(f"    - {k}: {sub}")
        else:
            lines.append(f"- **{f}**: {v}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def append_entry(entry: dict[str, Any]) -> None:
    """Append one event to both the .md and .jsonl journals (append-only)."""
    entry = dict(entry)
    entry.setdefault("datetime", _now_iso())
    for f in _FIELDS:
        entry.setdefault(f, None)

    RESEARCH_V2.mkdir(parents=True, exist_ok=True)
    if not MD_PATH.exists():
        MD_PATH.write_text(
            "# engine_v2 (V2) Research Journal\n\n"
            "Append-only. Each entry records one tranche / run / conclusion. "
            "Earlier entries are never edited or reinterpreted.\n\n---\n\n",
            encoding="utf-8",
        )
    with MD_PATH.open("a", encoding="utf-8") as fh:
        fh.write(_md_section(entry))
    with JSONL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _already_seeded(marker: str) -> bool:
    if not JSONL_PATH.exists():
        return False
    for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line).get("seed_marker") == marker:
                return True
        except json.JSONDecodeError:
            continue
    return False


def seed_known_history() -> None:
    """Seed the three known P2.5 events (n=10, n=50, invalid n=100). Idempotent."""
    marker = "P2.5-seed-v1"
    if _already_seeded(marker):
        print("Journal already seeded (marker present); nothing appended.")
        return

    entries = [
        {
            "seed_marker": marker,
            "datetime": "2026-06-06T00:00:00 (backfilled)",
            "title": "n=10 tranche -- INCONCLUSIVE",
            "hypothesis": "Posterior-coherent V2 ranks stocks better than frozen canonical V1 "
                          "(Rank-IC / long-short / decile monotonicity).",
            "sample_size": "n=10 tickers, 312 obs (3y n=206, 5y n=149)",
            "engine": "v1 (frozen control) vs v2 (engine_version=v2, reduced MCMC: draws=300/chains=2/warmup=200/paths=400)",
            "command": "PYTHONPATH=src python scripts/utilities/research_v2/forecasting_compare.py --n 10",
            "output_artifact": "reports/research_v2/forecasting_compare_10.json",
            "metrics": {
                "3y": "IC v1 0.1357 -> v2 0.1232 (v1 higher); Spearman v1 0.2403 -> v2 0.2606 (v2 higher); L/S v1 0.2794 -> v2 0.2271 (v1 higher)",
                "5y": "IC v1 0.3908 -> v2 0.3765 (v1 higher); Spearman v1 0.3190 -> v2 0.3836 (v2 higher); L/S v1 0.3049 -> v2 0.2782 (v1 higher)",
            },
            "interpretation": "Mixed / tied. v2 better on Spearman, v1 better on IC and long-short. "
                              "Small-cap-survivor noise at n=10; NOT statistically meaningful and NOT "
                              "sufficient to advance.",
            "decision": "INCONCLUSIVE. Recorded as such permanently -- not relabeled as a win.",
            "next_action": "Run a larger tranche (n=50).",
        },
        {
            "seed_marker": marker,
            "datetime": "2026-06-06T00:00:00 (backfilled)",
            "title": "n=50 tranche -- FAVORABLE to V2 (first meaningful evidence)",
            "hypothesis": "Posterior-coherent V2 ranks stocks better than frozen canonical V1.",
            "sample_size": "n=50 tickers, 1809 obs (3y n=1242, 5y n=893)",
            "engine": "v1 (frozen control) vs v2 (reduced MCMC: draws=300/chains=2/warmup=200/paths=400)",
            "command": "PYTHONPATH=src python scripts/utilities/research_v2/forecasting_compare.py --n 50",
            "output_artifact": "reports/research_v2/forecasting_compare_50.json",
            "metrics": {
                "3y": "IC v1 0.1023 (t=3.51) -> v2 0.1177 (t=4.56); Spearman 0.1764 -> 0.1868; L/S 0.1105 -> 0.1382; decile-monotonicity 0.661 -> 0.891",
                "5y": "IC v1 0.0940 (t=2.70) -> v2 0.1116 (t=4.08); Spearman 0.1510 -> 0.1718; L/S 0.0874 -> 0.1025; decile-monotonicity 0.370 -> 0.515",
            },
            "interpretation": "v2 is strictly better on EVERY metric (IC, IC t-stat, Spearman, long-short, "
                              "decile monotonicity) at BOTH horizons. Statistically meaningful and the first "
                              "real evidence posterior-coherent v2 may improve over v1. Promising, but a "
                              "confirmation tranche is still required before advancing.",
            "decision": "FAVORABLE to v2 and meaningful. v2 stays non-default; run n=100 as confirmation.",
            "next_action": "Run n=100 confirmation tranche.",
        },
        {
            "seed_marker": marker,
            "datetime": _now_iso(),
            "title": "n=100 (on-disk artifact) -- INVALID EVIDENCE, quarantined",
            "hypothesis": "Confirm the n=50 favorable result at n=100.",
            "sample_size": "INTENDED n=100; ACTUAL n=6 (216 obs) -- AAPL, MSFT, GOOGL, META, AMZN, NVDA only",
            "engine": "v1 vs v2 (reduced MCMC)",
            "command": "PYTHONPATH=src python scripts/utilities/research_v2/forecasting_compare.py --n 100",
            "output_artifact": "reports/research_v2/forecasting_compare_100_INVALID_sec-throttled.json "
                               "(renamed from forecasting_compare_100.json; preserved, not deleted)",
            "metrics": {
                "note": "Not reported as evidence. 94/100 tickers hit SEC ':sec' throttle skips; only the "
                        "first 6 mega-caps produced observations. Sample is mega-cap biased and tiny.",
            },
            "interpretation": "INVALID EVIDENCE. SEC throttling / incomplete execution collapsed the run to "
                              "6 mega-cap names. Cannot confirm or contradict n=50. Excluded from all "
                              "scientific conclusions.",
            "decision": "INVALID -- excluded from confirmation analysis. File quarantined (renamed) and preserved.",
            "next_action": "Bring the run tracker online, then relaunch a CLEAN, resumable, tracker-monitored "
                           "n=100 run with gentler SEC pacing + exponential backoff + per-ticker checkpoints.",
        },
    ]
    for e in entries:
        append_entry(e)
    print(f"Seeded {len(entries)} journal entries -> {MD_PATH.name}, {JSONL_PATH.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="engine_v2 append-only research journal")
    ap.add_argument("--seed", action="store_true", help="seed the known P2.5 history (idempotent)")
    args = ap.parse_args()
    if args.seed:
        seed_known_history()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
