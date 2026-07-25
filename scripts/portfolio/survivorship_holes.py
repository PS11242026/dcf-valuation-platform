"""Stage A.1 -- size the survivorship holes via EDGAR.

The backtest universe is 365 *current-membership* names. To measure how much is missing, enumerate
every company that filed a 10-K in the 7 sector SIC families during a base window (default 2015-2018)
and then determine which ones STOPPED filing (delisted / acquired / bankrupt / went private). Those are
the survivorship holes the current-membership panel can never see.

Mechanism (all free EDGAR data; fundamentals of dead companies persist forever):
  1. Pull pipe-delimited master.idx for each quarter in the base window -> distinct CIKs that filed
     any 10-K* form.
  2. Per CIK, fetch data.sec.gov submissions JSON -> SIC (classify into the 7 families via the
     coherence classifier), current tickers, and the most-recent 10-K filing date.
  3. SURVIVOR if it still files (last 10-K year >= --alive-year, default 2024) OR a current ticker is
     in the 365-name panel; otherwise a HOLE, bucketed by its last-10-K year (the delisting cohort).
  4. Aggregate holes vs survivors per family + hole-rate + cohort-by-year.

Resumable: per-CIK records cached under reports/portfolio/_surv_subs/. Re-runs skip cached CIKs.

Usage:
    python -m scripts.portfolio.survivorship_holes                 # full run (background recommended)
    python -m scripts.portfolio.survivorship_holes --limit 300     # smoke test
    python -m scripts.portfolio.survivorship_holes --years 2015 2016 2017 2018
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coherence.build_sector_universe import classify_sic  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dcf_project.config import get_settings  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"
PORT_DIR = REPORT_DIR / "portfolio"
SUBS_CACHE = PORT_DIR / "_surv_subs"
MASTER_IDX = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"


def _panel_tickers() -> set[str]:
    obs = json.load(open(REPORT_DIR / "final_universe_backtest.json"))["observations"]
    return {o["ticker"].upper() for o in obs}


# SEC fair-access limit is 10 req/s; serialize request *starts* to ~8/s across all workers.
_RATE_LOCK = threading.Lock()
_LAST_REQ = [0.0]
_MIN_INTERVAL = 0.125


def _throttle() -> None:
    with _RATE_LOCK:
        dt = time.monotonic() - _LAST_REQ[0]
        if dt < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - dt)
        _LAST_REQ[0] = time.monotonic()


def _get(url: str, ua: str, timeout: int = 60, retries: int = 4) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(retries):
        _throttle()
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            return raw
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (403, 429, 500, 502, 503):
                time.sleep(0.5 * (2 ** attempt))  # backoff on rate-limit / transient
                continue
            if exc.code == 404:
                raise  # genuinely gone
            time.sleep(0.3 * (2 ** attempt))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.3 * (2 ** attempt))
    raise last_exc if last_exc else RuntimeError("fetch failed")


def collect_10k_ciks(years: list[int], ua: str) -> dict[str, str]:
    """Return {cik10: company_name} for every distinct 10-K* filer in the given years.

    Cached to reports/portfolio/_surv_ciks_<years>.json so re-runs skip the big master.idx downloads.
    """
    cache = PORT_DIR / f"_surv_ciks_{'_'.join(map(str, years))}.json"
    if cache.exists():
        print(f"  using cached CIK list {cache.name}", flush=True)
        return json.load(open(cache))
    ciks: dict[str, str] = {}
    for y in years:
        for q in (1, 2, 3, 4):
            url = MASTER_IDX.format(year=y, q=q)
            print(f"  downloading master.idx {y}Q{q} ...", flush=True)
            try:
                text = _get(url, ua).decode("latin-1", errors="replace")
            except Exception as exc:  # noqa: BLE001
                print(f"  master.idx {y}Q{q} failed: {exc}", flush=True)
                continue
            n0 = len(ciks)
            for line in text.splitlines():
                parts = line.split("|")
                if len(parts) != 5:
                    continue
                cik, name, form, date, _ = parts
                if not form.startswith("10-K"):
                    continue
                if not cik.strip().isdigit():
                    continue
                ciks[f"{int(cik):010d}"] = name.strip()
            print(f"  {y}Q{q}: +{len(ciks) - n0} new CIKs (total {len(ciks)})", flush=True)
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ciks), encoding="utf-8")
    return ciks


def _fetch_record(cik10: str, name: str, ua: str) -> dict | None:
    cache = SUBS_CACHE / f"{cik10}.json"
    if cache.exists():
        try:
            return json.load(open(cache))
        except Exception:  # noqa: BLE001
            pass
    try:
        d = json.loads(_get(SUBMISSIONS.format(cik=cik10), ua, timeout=30).decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None
    sic = d.get("sic")
    try:
        sic = int(sic) if sic not in (None, "") else None
    except (ValueError, TypeError):
        sic = None
    family = classify_sic(sic) if sic is not None else None
    tickers = [t.upper() for t in (d.get("tickers") or [])]
    # most-recent 10-K date from filings.recent
    rec = (d.get("filings") or {}).get("recent") or {}
    forms = rec.get("form") or []
    dates = rec.get("filingDate") or []
    last_10k = None
    for f, dt in zip(forms, dates):
        if f.startswith("10-K"):
            last_10k = dt  # recent[] is newest-first, so first hit is the latest
            break
    out = {
        "cik": cik10, "name": d.get("name") or name, "sic": sic,
        "sicDescription": d.get("sicDescription"), "family": family,
        "tickers": tickers, "last_10k": last_10k,
    }
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def build(years: list[int], alive_year: int, limit: int, workers: int) -> dict:
    ua = get_settings().sec_user_agent
    SUBS_CACHE.mkdir(parents=True, exist_ok=True)
    panel = _panel_tickers()

    print(f"collecting 10-K filers for {years} ...", flush=True)
    ciks = collect_10k_ciks(years, ua)
    items = list(ciks.items())
    if limit:
        items = items[:limit]
    print(f"classifying {len(items)} CIKs (submissions fetch, resumable cache) ...", flush=True)

    records: list[dict] = []
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_record, c, n, ua): c for c, n in items}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r is not None:
                records.append(r)
            else:
                failed += 1
            if done % 250 == 0:
                print(f"  {done}/{len(items)} fetched ({failed} failed, "
                      f"{len(records)} ok)", flush=True)

    in_family = [r for r in records if r["family"]]
    print(f"{len(records)} classified, {failed} fetch failures; "
          f"{len(in_family)} CIKs fall in the 7 families")

    # survivor vs hole
    fam_stats: dict[str, dict] = {}
    cohort: dict[str, dict[str, int]] = {}
    holes_detail: list[dict] = []
    for r in in_family:
        fam = r["family"]
        last_year = int(r["last_10k"][:4]) if r["last_10k"] else 0
        still_in_panel = any(t in panel for t in r["tickers"])
        survivor = still_in_panel or last_year >= alive_year
        fs = fam_stats.setdefault(fam, {"survivors": 0, "holes": 0, "in_panel": 0})
        if still_in_panel:
            fs["in_panel"] += 1
        if survivor:
            fs["survivors"] += 1
        else:
            fs["holes"] += 1
            cohort.setdefault(fam, {})
            yk = str(last_year) if last_year else "unknown"
            cohort[fam][yk] = cohort[fam].get(yk, 0) + 1
            holes_detail.append({"cik": r["cik"], "name": r["name"], "family": fam,
                                 "tickers": r["tickers"], "last_10k": r["last_10k"]})

    for fam, fs in fam_stats.items():
        tot = fs["survivors"] + fs["holes"]
        fs["total"] = tot
        fs["hole_rate"] = round(fs["holes"] / tot, 3) if tot else None

    totals = {
        "survivors": sum(f["survivors"] for f in fam_stats.values()),
        "holes": sum(f["holes"] for f in fam_stats.values()),
    }
    totals["total"] = totals["survivors"] + totals["holes"]
    totals["hole_rate"] = round(totals["holes"] / totals["total"], 3) if totals["total"] else None

    result = {
        "base_window_years": years, "alive_year_threshold": alive_year,
        "n_panel_tickers": len(panel),
        "n_10k_filers_window": len(ciks), "n_classified": len(records),
        "n_in_family": len(in_family),
        "by_family": fam_stats, "delisting_cohort_by_year": cohort,
        "totals": totals, "holes_detail": holes_detail,
    }
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    (PORT_DIR / "survivorship_holes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _print_summary(res: dict) -> None:
    print("\n=== Survivorship holes (EDGAR 10-K filers, base window "
          f"{res['base_window_years']}) ===")
    print(f"panel survivors universe = {res['n_panel_tickers']} names")
    print(f"{res['n_in_family']} in-family historical filers; "
          f"total hole rate = {res['totals']['hole_rate'] * 100:.1f}% "
          f"({res['totals']['holes']} holes / {res['totals']['total']})")
    print(f"\n{'family':<36} {'total':>6} {'surv':>6} {'hole':>6} {'hole%':>7} {'inpanel':>8}")
    for fam, fs in sorted(res["by_family"].items(), key=lambda kv: -kv[1]["total"]):
        print(f"{fam:<36} {fs['total']:>6} {fs['survivors']:>6} {fs['holes']:>6} "
              f"{fs['hole_rate'] * 100:>6.1f}% {fs['in_panel']:>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Size survivorship holes via EDGAR 10-K enumeration")
    ap.add_argument("--years", type=int, nargs="+", default=[2015, 2016, 2017, 2018])
    ap.add_argument("--alive-year", type=int, default=2024,
                    help="last-10-K year >= this => survivor (still filing)")
    ap.add_argument("--limit", type=int, default=0, help="cap CIKs (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    res = build(args.years, args.alive_year, args.limit, args.workers)
    _print_summary(res)


if __name__ == "__main__":
    main()
