"""HTTP API for on-demand analysis (the Phoenix Capital "Analyze any ticker" page).

A full canonical valuation takes ~20-60 s (SEC + market data + Kalman forecast + renders), so
the API is a small job queue rather than one long request:

    GET /api/analyze/{ticker}   -> {"status": "queued"|"running"|"done"|"error", ...}
                                   starts a job when there is no fresh cached report; poll it.
    GET /api/files/{ticker}/{name}  -> the Excel workbook / PDF produced by the job
    GET /api/health

Reports are cached on disk per ticker for DCF_API_TTL_HOURS (default 12). Jobs run on a
bounded worker pool (DCF_API_WORKERS, default 1: the engine is CPU- and rate-limit-bound),
and each client IP may start at most DCF_API_RATE_PER_HOUR new jobs (default 15).

Run:  uvicorn dcf_project.api.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from dcf_project.paths import VALUATION

from . import service

TTL = float(os.environ.get("DCF_API_TTL_HOURS", "12")) * 3600
WORKERS = int(os.environ.get("DCF_API_WORKERS", "1"))
RATE = int(os.environ.get("DCF_API_RATE_PER_HOUR", "15"))
ORIGINS = [o.strip() for o in os.environ.get(
    "DCF_API_ORIGINS",
    "https://ps11242026.github.io,http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
OUT_ROOT = Path(os.environ.get("DCF_API_OUT") or VALUATION)
ALLOWED_FILES = {"canonical_dcf.xlsx", "fade_dcf.xlsx"}

app = FastAPI(title="DCF Valuation Platform API", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET"], allow_headers=["*"])

_pool = ThreadPoolExecutor(max_workers=WORKERS)
_lock = threading.Lock()
_jobs: dict[str, dict] = {}            # ticker -> {"status", "started", "error", "queued_at"}
_hits: dict[str, deque] = {}           # client ip -> start timestamps (rate limit)


def _cached(t: str) -> dict | None:
    p = OUT_ROOT / t / "report.json"
    if p.exists() and (time.time() - p.stat().st_mtime) < TTL:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _run(t: str) -> None:
    with _lock:
        _jobs[t].update(status="running", started=time.time())
    try:
        service.analyze(t, OUT_ROOT)
        with _lock:
            _jobs.pop(t, None)
    except service.AnalysisError as e:
        with _lock:
            _jobs[t].update(status="error", error=str(e), finished=time.time())
    except Exception as e:
        with _lock:
            _jobs[t].update(status="error", error=f"Unexpected failure: {e.__class__.__name__}",
                            finished=time.time())


def _client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")


def _allow(ip: str) -> bool:
    now = time.time()
    q = _hits.setdefault(ip, deque())
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE:
        return False
    q.append(now)
    return True


def _position(t: str) -> int:
    waiting = sorted((j["queued_at"], k) for k, j in _jobs.items() if j["status"] == "queued")
    return next((i + 1 for i, (_, k) in enumerate(waiting) if k == t), 0)


@app.get("/api/health")
def health():
    return {"ok": True, "workers": WORKERS, "jobs": len(_jobs)}


@app.get("/api/analyze/{ticker}")
def analyze(ticker: str, request: Request, refresh: int = 0):
    try:
        t = service.normalize_ticker(ticker)
    except service.AnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    with _lock:
        job = _jobs.get(t)
        if job and job["status"] in ("queued", "running"):
            el = time.time() - (job.get("started") or job["queued_at"])
            return {"status": job["status"], "ticker": t, "elapsed": round(el, 1),
                    "position": _position(t)}
        if job and job["status"] == "error" and not refresh and time.time() - job["finished"] < 600:
            return {"status": "error", "ticker": t, "error": job["error"]}
    if not refresh:
        rep = _cached(t)
        if rep:
            return {"status": "done", "ticker": t, "report": rep}
    with _lock:
        if t in _jobs and _jobs[t]["status"] in ("queued", "running"):
            return {"status": _jobs[t]["status"], "ticker": t, "elapsed": 0, "position": _position(t)}
        if not _allow(_client_ip(request)):
            return JSONResponse(status_code=429, content={
                "status": "error", "ticker": t,
                "error": f"Rate limit: at most {RATE} new analyses per hour from one address."})
        _jobs[t] = {"status": "queued", "queued_at": time.time()}
    _pool.submit(_run, t)
    return {"status": "queued", "ticker": t, "elapsed": 0, "position": _position(t)}


@app.get("/api/files/{ticker}/{name}")
def files(ticker: str, name: str):
    try:
        t = service.normalize_ticker(ticker)
    except service.AnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if name not in ALLOWED_FILES and name != f"{t}.pdf":
        raise HTTPException(status_code=404, detail="unknown file")
    p = OUT_ROOT / t / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="run the analysis first")
    fn = f"{t}_{name}" if name.endswith(".xlsx") else name
    return FileResponse(p, filename=fn)


@app.get("/")
def root():
    return {"service": "DCF Valuation Platform API", "docs": "/api/docs",
            "try": "/api/analyze/KO"}
