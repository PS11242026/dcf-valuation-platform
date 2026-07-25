"""Offline build of the static site from synthetic book folders (no network).

Pins: pages render with the numbers from status.json (no second computation path), the data/
payloads are copied for hosted auto-refresh, research markdown renders with frontmatter, the
compliance disclaimer is on every page, and authored markdown cannot inject HTML.
"""
from __future__ import annotations

import json

import pytest

from dcf_project.site import md as mdlib
from dcf_project.site.builder import DEFAULT_CONFIG, build, load_config

TWO_BOOK_CFG = dict(
    DEFAULT_CONFIG,
    books=[
        {"key": "us", "name": "conf_top10", "label": "US Deep-Discount 10",
         "nav": "US Book", "flag": "us", "status_badge": "live"},
        {"key": "india", "name": "india_v3_aggressive", "label": "India Energy PSU 5",
         "nav": "India Book", "flag": "in", "status_badge": "prelim"},
    ])

US_STATUS = {
    "as_of": "2026-07-11", "book": "conf_top10", "book_id": "discount_x_conf_top10",
    "currency": "USD", "inception": "2026-06-04", "days": 37,
    "capital": 1_000_000.0, "capital_basis": "notional", "cash": 0.0,
    "value_at_inception": 1_000_000.0, "value": 1_072_800.0, "return_since_inception": 0.0728,
    "benchmarks": {"SPY": {"entry": 757.26, "last": 754.95, "return": -0.003, "active": 0.0758}},
    "comparables": [{"name": "Titan Flagship (modeled)", "kind": "run_rate",
                     "note": "modeled run-rate", "return": 0.0104, "active": 0.0624}],
    "holdings": [{"ticker": "QNST", "sector": "internet_platforms_marketplaces",
                  "entry_price": 11.83, "price": 16.79, "weight": 0.132,
                  "entry_value": 100_000.0, "market_value": 141_927.0,
                  "return": 0.419, "day_change": 0.021, "contribution": 0.04,
                  "inception_discount": 2.49, "current_discount": None,
                  "current_intrinsic": None, "conviction": 0.993}],
    "sector_weights": {"internet_platforms_marketplaces": 1.0},
    "metrics": {"excess": {"SPY": 0.0758}, "primary_benchmark": "SPY", "day_change": 0.019},
    "honesty": None,
}
IN_STATUS = {
    "as_of": "2026-07-11", "book": "india_v3_aggressive", "book_id": "holdings",
    "currency": "INR", "inception": "2026-06-15", "days": 26, "capital": 1_000_000.0,
    "cash": 569.85, "value": 997622.0, "return_since_inception": -0.0024,
    "benchmarks": {"^NSEI": {"entry": 23622.9, "last": 24206.9, "return": 0.0247, "active": -0.0271}},
    "comparables": [{"name": "NIFTY Energy", "kind": "index", "note": "sector-fair",
                     "return": -0.008, "active": 0.0056}],
    "holdings": [{"ticker": "IOC", "sector": "energy", "entry_price": 140.94, "price": 138.96,
                  "weight": 0.201, "entry_value": 203658.0, "market_value": 200797.0,
                  "return": -0.014, "day_change": -0.004, "contribution": -0.003,
                  "inception_discount": -0.2248, "current_discount": None,
                  "current_intrinsic": 112.01, "conviction": 88.0}],
    "sector_weights": {"energy": 1.0},
    "metrics": {"excess": {"^NSEI": -0.0271}, "primary_benchmark": "^NSEI", "day_change": 0.004},
    "honesty": {"validation_status": "PRELIMINARY; NOT validated alpha."},
}


@pytest.fixture()
def managed(tmp_path):
    for name, status in (("conf_top10", US_STATUS), ("india_v3_aggressive", IN_STATUS)):
        d = tmp_path / "managed" / name
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps(status), encoding="utf-8")
        (d / "news.json").write_text(json.dumps({"as_of": "2026-07-11", "items": [
            {"symbol": "QNST", "title": "Sample <headline> & entities", "publisher": "Wire",
             "link": "https://example.com", "published": 1760150000}]}), encoding="utf-8")
        (d / "track.json").write_text(json.dumps({"inception": status["inception"], "rows": [
            {"date": "2026-07-11", "days": status["days"],
             "bench": {k: v["return"] for k, v in status["benchmarks"].items()},
             "books": {status["book_id"]: status["return_since_inception"]}}]}), encoding="utf-8")
        (d / "risk_stats.json").write_text(json.dumps({"portfolio": {
            "beta": 0.51, "vol": 0.178, "tracking_error": 0.178, "div_yield": 0.039,
            "diversification_ratio": 1.2, "effective_bets": 1.44, "realized_alpha": -0.17,
            "max_drawdown": -0.42}}), encoding="utf-8")
    return tmp_path / "managed"


def test_site_builds_all_pages_two_book_config(managed, tmp_path):
    out = build(managed_dir=managed, out_dir=tmp_path / "site", config=TWO_BOOK_CFG)
    for page in ("index.html", "us.html", "india.html", "research.html", "pricing.html", "about.html"):
        assert (out / page).exists(), page
    assert (out / "assets" / "site.css").exists() and (out / "assets" / "site.js").exists()
    assert (out / "data" / "us_status.json").exists() and (out / "data" / "india_track.json").exists()


def test_us_only_config_drops_india_everywhere(managed, tmp_path):
    out = build(managed_dir=managed, out_dir=tmp_path / "site", config=TWO_BOOK_CFG)  # stale run
    (out / "research" / "2000-01-01-retired-post.html").write_text("old", encoding="utf-8")
    out = build(managed_dir=managed, out_dir=tmp_path / "site", config=DEFAULT_CONFIG)
    assert not (out / "india.html").exists()             # stale page removed
    assert not (out / "research" / "2000-01-01-retired-post.html").exists()  # stale post removed
    assert not (out / "data" / "india_status.json").exists()
    idx = (out / "index.html").read_text(encoding="utf-8")
    assert "india.html" not in idx                       # no nav entry / links to an India page
    assert "India Energy PSU" not in idx                 # no India book section
    assert "Phoenix Capital" in idx                      # brand from config
    assert 'id="cmp-us"' in idx                          # comparative benchmark section


def test_pages_carry_status_numbers_and_disclaimer(managed, tmp_path):
    out = build(managed_dir=managed, out_dir=tmp_path / "site", config=TWO_BOOK_CFG)
    us = (out / "us.html").read_text(encoding="utf-8")
    assert "+7.3%" in us            # return_since_inception from status.json
    assert "+7.6" in us and " pp" in us  # active vs SPY in percentage points (no double unit)
    assert "modeled*" in us         # comparator labeled as modeled, never bare
    assert "not investment advice" in us.lower()
    assert 'id="cmp-us"' in us      # comparative benchmark section on the book page
    # holdings table: inception-vs-now framing with a Total NAV line
    assert "Entry @ inception" in us and "Value @ inception" in us and "Value now" in us
    assert "Total NAV" in us and "$1,000,000" in us and "$141,927" in us
    assert "notional base" in us    # notional display capital is labeled, never implied as real
    india = (out / "india.html").read_text(encoding="utf-8")
    assert "₹997,622" in india      # INR value straight from status.json
    assert "PRELIMINARY" in india   # honesty block rendered
    assert "Cash" in india and "₹570" in india  # cash row so the NAV total reconciles
    idx = (out / "index.html").read_text(encoding="utf-8")
    assert "window.SITE_DATA" in idx and '"us"' in idx and '"india"' in idx


def test_news_titles_are_escaped(managed, tmp_path):
    out = build(managed_dir=managed, out_dir=tmp_path / "site", config=TWO_BOOK_CFG)
    idx = (out / "index.html").read_text(encoding="utf-8")
    assert "Sample &lt;headline&gt; &amp; entities" in idx
    assert "Sample <headline>" not in idx


def test_markdown_renderer_basics_and_escaping():
    html_out = mdlib.render("# H\n\n**b** and *i* and `c` and [l](https://x.y)\n\n- a\n- b\n\n> q\n\n<script>alert(1)</script>")
    assert "<h2>H</h2>" in html_out and "<strong>b</strong>" in html_out
    assert "<ul>" in html_out and "<blockquote>" in html_out
    assert "<script>" not in html_out and "&lt;script&gt;" in html_out


def test_frontmatter_parsing():
    meta, body = mdlib.parse_post("---\ntitle: T\ndate: 2026-07-11\ntags: a, b\n---\n\nBody.")
    assert meta["title"] == "T" and meta["date"] == "2026-07-11"
    assert body.strip() == "Body."


def test_config_defaults_load():
    cfg = load_config()
    assert cfg["brand"] and cfg["books"] and cfg["plans"]


def test_custom_domain_writes_cname(managed, tmp_path):
    cfg = dict(DEFAULT_CONFIG, custom_domain="phoenixcapitaldesk.com")
    out = build(managed_dir=managed, out_dir=tmp_path / "site", config=cfg)
    assert (out / "CNAME").read_text(encoding="utf-8").strip() == "phoenixcapitaldesk.com"
    out2 = build(managed_dir=managed, out_dir=tmp_path / "site2", config=DEFAULT_CONFIG)
    assert not (out2 / "CNAME").exists()  # no domain configured -> no CNAME
