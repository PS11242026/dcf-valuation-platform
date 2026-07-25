"""File-based progress tracker for bulky research_v2 shell/MCMC/backtest runs.

The tracker writes a single current-state JSON (`run_tracker.json`) plus an append-only
heartbeat history (`run_tracker.log`) under reports/research_v2/. Writes are atomic
(temp file + os.replace) so the state survives a killed shell or lost stdout. Pure
stdlib, PowerShell-friendly, no web UI.

A long run instantiates `RunTracker.start(...)`, calls `heartbeat()` / `mark_completed()` /
`mark_failed()` as it progresses, and `finish()` at the end. A separate process reads the
state with `read_status()` (used by `python -m dcf_project research status` / `... watch`).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dcf_project.paths import REPORTS

RESEARCH_V2 = REPORTS / "research_v2"
STATE_PATH = RESEARCH_V2 / "run_tracker.json"
LOG_PATH = RESEARCH_V2 / "run_tracker.log"


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class RunTracker:
    """Mutable run state, persisted to disk on every update."""

    run_id: str
    command: str
    phase: str
    tranche_size: int
    output_path: str = ""
    pid: int = field(default_factory=os.getpid)
    status: str = "running"  # running | paused | completed | failed
    start_ts: float = field(default_factory=_now)
    last_heartbeat: float = field(default_factory=_now)
    current_ticker: str | None = None
    current_detail: str | None = None  # e.g. "cutoff 2019-03-31"
    completed_names: list[str] = field(default_factory=list)
    failed_names: list[str] = field(default_factory=list)
    sec_retry_count: int = 0
    exit_code: int | None = None
    state_path: Path = STATE_PATH
    log_path: Path = LOG_PATH

    # ---- lifecycle ---------------------------------------------------------
    @classmethod
    def start(cls, run_id: str, command: str, phase: str, tranche_size: int,
              output_path: str = "", state_path: Path = STATE_PATH,
              log_path: Path = LOG_PATH) -> "RunTracker":
        t = cls(run_id=run_id, command=command, phase=phase, tranche_size=tranche_size,
                output_path=output_path, state_path=state_path, log_path=log_path)
        t._flush(event="start")
        return t

    def heartbeat(self, current_ticker: str | None = None, current_detail: str | None = None,
                  sec_retries: int | None = None) -> None:
        if current_ticker is not None:
            self.current_ticker = current_ticker
        if current_detail is not None:
            self.current_detail = current_detail
        if sec_retries is not None:
            self.sec_retry_count = sec_retries
        self.last_heartbeat = _now()
        self._flush(event="heartbeat")

    def mark_completed(self, ticker: str) -> None:
        if ticker not in self.completed_names:
            self.completed_names.append(ticker)
        self.last_heartbeat = _now()
        self._flush(event="completed")

    def mark_failed(self, ticker: str, reason: str = "") -> None:
        tag = f"{ticker}:{reason}" if reason else ticker
        if tag not in self.failed_names:
            self.failed_names.append(tag)
        self.last_heartbeat = _now()
        self._flush(event="failed")

    def pause(self) -> None:
        self.status = "paused"
        self._flush(event="pause")

    def finish(self, status: str = "completed", exit_code: int = 0) -> None:
        self.status = status
        self.exit_code = exit_code
        self.last_heartbeat = _now()
        self._flush(event="finish")

    # ---- derived -----------------------------------------------------------
    @property
    def n_completed(self) -> int:
        return len(self.completed_names)

    @property
    def n_total(self) -> int:
        return self.tranche_size

    @property
    def progress_pct(self) -> float:
        return round(100.0 * self.n_completed / self.n_total, 1) if self.n_total else 0.0

    @property
    def elapsed_s(self) -> float:
        return round(self.last_heartbeat - self.start_ts, 1)

    @property
    def eta_s(self) -> float | None:
        done = self.n_completed
        remaining = self.n_total - done
        if done <= 0 or remaining <= 0:
            return None
        rate = (_now() - self.start_ts) / done  # seconds per completed name
        return round(rate * remaining, 1)

    def snapshot(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("state_path", "log_path")}
        d.update({
            "n_completed": self.n_completed,
            "n_total": self.n_total,
            "progress_pct": self.progress_pct,
            "elapsed_s": self.elapsed_s,
            "eta_s": self.eta_s,
            "start_iso": _iso(self.start_ts),
            "last_heartbeat_iso": _iso(self.last_heartbeat),
            "updated_iso": _iso(_now()),
        })
        return d

    # ---- persistence -------------------------------------------------------
    def _flush(self, event: str) -> None:
        snap = self.snapshot()
        _atomic_write(self.state_path, json.dumps(snap, indent=2))
        line = json.dumps({
            "ts": _iso(_now()), "event": event, "status": self.status,
            "current_ticker": self.current_ticker, "detail": self.current_detail,
            "n_completed": self.n_completed, "n_total": self.n_total,
            "progress_pct": self.progress_pct, "sec_retry_count": self.sec_retry_count,
        })
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def read_status(state_path: Path = STATE_PATH) -> dict[str, Any] | None:
    """Return the last persisted tracker snapshot, or None if no run is recorded."""
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def render_status(snap: dict[str, Any] | None) -> str:
    """Human-readable one-shot render of a tracker snapshot (used by the CLI)."""
    if snap is None:
        return "No active research run recorded (reports/research_v2/run_tracker.json not found)."
    pct = snap.get("progress_pct", 0.0)
    bar_w = 30
    filled = int(round(bar_w * pct / 100.0))
    bar = "#" * filled + "-" * (bar_w - filled)
    lines = [
        f"run_id     : {snap.get('run_id')}   [{snap.get('status', '?').upper()}]",
        f"phase      : {snap.get('phase')}",
        f"command    : {snap.get('command')}",
        f"pid        : {snap.get('pid')}    exit_code: {snap.get('exit_code')}",
        f"progress   : [{bar}] {pct:.1f}%  ({snap.get('n_completed')}/{snap.get('n_total')})",
        f"current    : {snap.get('current_ticker')}  {snap.get('current_detail') or ''}".rstrip(),
        f"failed     : {len(snap.get('failed_names') or [])}   sec_retries: {snap.get('sec_retry_count')}",
        f"elapsed    : {_fmt_dur(snap.get('elapsed_s'))}    eta: {_fmt_dur(snap.get('eta_s'))}",
        f"started    : {snap.get('start_iso')}",
        f"heartbeat  : {snap.get('last_heartbeat_iso')}",
        f"output     : {snap.get('output_path')}",
    ]
    failed = snap.get("failed_names") or []
    if failed:
        lines.append(f"failed_names: {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}")
    return "\n".join(lines)
