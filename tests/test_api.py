"""On-demand analysis API: job lifecycle, caching, validation, file serving, rate limit."""
import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from dcf_project.api import app as api
from dcf_project.api import service


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(api, "_jobs", {})
    monkeypatch.setattr(api, "_hits", {})

    def fake_analyze(t, out_root):
        if t == "BAD":
            raise service.AnalysisError("no SEC filings for BAD")
        d = out_root / t
        d.mkdir(parents=True, exist_ok=True)
        (d / "canonical_dcf.xlsx").write_bytes(b"PK")
        rep = {"ticker": t, "intrinsic_value": 100.0, "files": {"xlsx": "canonical_dcf.xlsx"}}
        (d / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        return rep

    monkeypatch.setattr(service, "analyze", fake_analyze)
    return TestClient(api.app)


def _wait(client, t, n=50):
    for _ in range(n):
        r = client.get(f"/api/analyze/{t}").json()
        if r["status"] in ("done", "error"):
            return r
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_job_runs_then_serves_cache(client):
    first = client.get("/api/analyze/ko").json()
    assert first["status"] in ("queued", "running", "done") and first["ticker"] == "KO"
    done = _wait(client, "KO")
    assert done["status"] == "done" and done["report"]["intrinsic_value"] == 100.0
    assert client.get("/api/analyze/KO").json()["status"] == "done"  # cached, no new job


def test_engine_failure_is_reported(client):
    r = _wait(client, "BAD")
    assert r["status"] == "error" and "SEC" in r["error"]


@pytest.mark.parametrize("bad", ["123", "TOOLONGTICKER", "A$B", "ko;rm"])
def test_invalid_ticker_rejected(client, bad):
    assert client.get(f"/api/analyze/{bad}").status_code == 400


def test_file_download_whitelist(client):
    _wait(client, "KO")
    assert client.get("/api/files/KO/canonical_dcf.xlsx").status_code == 200
    assert client.get("/api/files/KO/report.json").status_code == 404
    assert client.get("/api/files/KO/..%2Fsecret").status_code == 404


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(api, "RATE", 2)
    assert client.get("/api/analyze/AAA").status_code == 200
    assert client.get("/api/analyze/BBB").status_code == 200
    assert client.get("/api/analyze/CCC").status_code == 429


def test_normalize_ticker():
    assert service.normalize_ticker(" brk.b ") == "BRK.B"
    assert service.normalize_ticker("RELIANCE.NS") == "RELIANCE.NS"
    with pytest.raises(service.AnalysisError):
        service.normalize_ticker("")


def test_clean_handles_nan_and_numpy():
    np = pytest.importorskip("numpy")
    out = service._clean({"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(2), float("inf")]})
    assert out == {"a": None, "b": 1.5, "c": [2, None]}


def test_absolute_rank_buckets():
    from dcf_project.portfolio.scoring import _label, absolute_discount_rank
    assert _label(absolute_discount_rank(0.45)) == "Very Attractive"
    assert _label(absolute_discount_rank(0.15)) == "Attractive"
    assert _label(absolute_discount_rank(0.0)) == "Neutral"
    assert _label(absolute_discount_rank(-0.2)) == "Unattractive"
    assert _label(absolute_discount_rank(-0.5)) == "Very Unattractive"
    assert _label(absolute_discount_rank(None)) == "Neutral"


def test_reference_unavailable_without_files(tmp_path, monkeypatch):
    from dcf_project.portfolio import scoring
    monkeypatch.setattr(scoring, "REPORT_DIR", tmp_path)
    ref = scoring.Reference()
    assert ref.available is False and ref.rank_basis == "absolute_discount"
