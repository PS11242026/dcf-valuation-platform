"""Static-site generator: canonical book JSONs + authored research -> deployable site/.

Every number on the site comes from the SAME files the desk tooling writes (status.json,
news.json, risk_stats.json, track.json) -- there is no second computation path to drift.
Pages embed their data snapshot (so the site works from file:// and any static host) and,
when hosted, the client polls site/data/*.json between deploys; with a feed key configured
the dashboard upgrades to live in-browser quotes and a live news wire.

Honesty is layout, not fine print: MODELED / survivorship badges render next to the numbers
they qualify, and the compliance footer is on every page. Premium tone, honest claims.

CLI:
    python -m dcf_project site build
    python -m dcf_project site post "Why refiners, not upstream" --tags energy
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import shutil
from pathlib import Path

from dcf_project.paths import MANAGED, SITE_CONTENT, SITE_OUT

from . import md as mdlib

ASSETS = Path(__file__).parent / "assets"

# Phoenix monogram: three ascending wing-strokes + head, strictly monochrome (currentColor so
# it renders black on light surfaces and white on the ink hero). Inline; the header never fetches.
def _mark(size: int = 20, cls: str = "mark") -> str:
    return (f"<svg class='{cls}' viewBox='0 0 32 32' width='{size}' height='{size}' aria-hidden='true' "
            "fill='none' stroke='currentColor' stroke-linecap='round' stroke-linejoin='round'>"
            "<path d='M4.5 21.5 Q8.5 8.5 22.6 6.2' stroke-width='2.5'/>"
            "<path d='M8 25.5 Q12.5 13.5 25 11.4' stroke-width='2.3'/>"
            "<path d='M12 29 Q16.5 19 26.6 16.6' stroke-width='2.1'/>"
            "<circle cx='24.9' cy='5.1' r='1.9' fill='currentColor' stroke='none'/>"
            "<path d='M26.9 4.5 L29.6 5.3 L27.2 6.7' stroke-width='1.5'/>"
            "</svg>")


MARK = _mark(20)


def _process_brand_logo(out: Path) -> bool:
    """site_content/logo.png (the authored brand mark: white phoenix on black) -> derived
    assets: logo-chip.png (as-is, for header chip + favicon) and logo-white.png (luminance ->
    alpha, white fill: the bird floats on the ink hero). Cached by mtime; Pillow guarded."""
    src = SITE_CONTENT / "logo.png"
    if not src.exists():
        return False
    chip, white = out / "assets" / "logo-chip.png", out / "assets" / "logo-white.png"
    if chip.exists() and white.exists() and chip.stat().st_mtime >= src.stat().st_mtime:
        return True
    try:
        from PIL import Image
    except ImportError:
        shutil.copy2(src, chip)
        return chip.exists()
    im = Image.open(src).convert("RGB")
    im.resize((256, 256), Image.LANCZOS).save(chip)
    lum = im.convert("L").resize((512, 512), Image.LANCZOS)
    w = Image.new("RGBA", lum.size, (255, 255, 255, 0))
    w.putalpha(lum)
    w.save(white)
    return True

DEFAULT_CONFIG = {
    "brand": "Phoenix Capital",
    "tagline": "Institutional-grade equity research. One disciplined signal.",
    "contact_email": "research@example.com",          # EDIT
    "refresh_seconds": 60,
    "quote_seconds": 60,                               # free feed tier: 60 calls/min shared by viewers
    "news_seconds": 300,
    "finnhub_key": "",                                 # feed key: live quotes + live news wire
    "custom_domain": "",                               # e.g. phoenixcapitaldesk.com (writes CNAME)
    "code_url": "https://github.com/PS11242026/dcf-valuation-platform",
    "api_url": "",                                     # on-demand analysis API (deploy/hf-space)
    "books": [
        {"key": "us", "name": "conf_top10", "label": "US Deep-Discount 10",
         "nav": "Portfolio", "flag": "us", "status_badge": "live"},
    ],
    "plans": [
        {"name": "Observer", "price": "Free", "per": "", "cta": "Follow the portfolio",
         "items": ["Dashboard with end-of-day refresh", "Full holdings, full history",
                   "Methodology and evidence pages"]},
        {"name": "Member", "price": "₹4,999", "per": "per month — EDIT before launch", "hi": True,
         "ribbon": "Most popular", "cta": "Become a member",
         "items": ["Daily research notes the morning they publish",
                   "Rebalance rationale on the day it happens",
                   "Live dashboard feed + news wire",
                   "Full risk statistics and fund comparisons",
                   "Email on every portfolio action"]},
        {"name": "Partner", "price": "Custom", "per": "annual", "cta": "Talk to the desk",
         "items": ["Methodology deep-dive and replication pack",
                   "Backtest ledgers and validation suites",
                   "Direct access for diligence"]},
    ],
}

DISCLAIMER = (
    "<b>This is research, not investment advice.</b> The portfolios shown are model (paper) "
    "books published for research and education; nothing here is a solicitation to buy or sell "
    "any security, and no client assets are managed through this site. Phoenix Capital is a "
    "research publication: the publisher is not a SEBI-registered Portfolio Manager / Investment "
    "Adviser and not a US SEC-registered investment adviser; no subscriptions or paid research are "
    "offered; this is a portfolio project. Backtested figures are labeled and survivorship-caveated where "
    "applicable; live records are short. Past performance, simulated or live, does not predict "
    "future returns. Markets carry risk of loss, including total loss."
)


# ----------------------------- tiny fmt helpers -----------------------------
def _pct(x, dp=1, signed=True):
    if x is None:
        return "–"
    s = "+" if (signed and x > 0) else ""
    return f"{s}{x * 100:.{dp}f}%"


def _pp(x, dp=1):
    return "–" if x is None else f"{x * 100:+.{dp}f} pp"


def _money(x, ccy="USD", dp=0):
    if x is None:
        return "–"
    sym = "₹" if ccy == "INR" else "$"
    return f"{sym}{x:,.{dp}f}"


def _cls(x):
    return "pos" if (x or 0) >= 0 else "neg"


def _esc(s):
    return html.escape(str(s or ""), quote=True)


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _sector(s):
    return (s or "").replace("_", " ").title().replace("Assetlight", "Asset-Light")


def _news_when(v):
    if isinstance(v, (int, float)):
        return dt.datetime.fromtimestamp(v).strftime("%b %d")
    if isinstance(v, str) and len(v) >= 10:
        return v[5:10].replace("-", "/")
    return ""


def _bm_label(bm):
    return {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq 100 (QQQ)", "^NSEI": "NIFTY 50"}.get(bm, bm)


# ----------------------------- data loading -----------------------------
def _load(p, default=None):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else default


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    user = _load(SITE_CONTENT / "site.json", {})
    cfg.update(user or {})
    if os.environ.get("DCF_FINNHUB_KEY"):  # CI secret beats the file, so the key never sits in git
        cfg["finnhub_key"] = os.environ["DCF_FINNHUB_KEY"].strip()
    if os.environ.get("DCF_API_URL"):      # on-demand analysis backend (Analyze page)
        cfg["api_url"] = os.environ["DCF_API_URL"].strip()
    return cfg


def load_books(cfg, managed_dir=None):
    base = Path(managed_dir) if managed_dir else MANAGED
    books = {}
    for b in cfg["books"]:
        folder = base / b["name"]
        status = _load(folder / "status.json")
        if not status:
            print(f"  [skip] {b['name']}: no status.json (run `portfolio manage update-all`)")
            continue
        logo_dir = folder / "figures" / "logos"
        books[b["key"]] = {
            "cfg": b, "status": status, "folder": folder,
            "news": (_load(folder / "news.json", {}) or {}).get("items", []),
            "track": _load(folder / "track.json", {"rows": []}),
            "risk": _load(folder / "risk_stats.json", {}),
            "logos": ({p.stem: f"logos/{p.name}" for p in logo_dir.glob("*.png") if _is_image(p)}
                      if logo_dir.exists() else {}),
        }
    return books


def _is_image(p: Path) -> bool:
    """Magic-byte check so a saved error page can never ship as a broken <img>."""
    try:
        b = p.open("rb").read(12)
    except OSError:
        return False
    return (b[:8] == b"\x89PNG\r\n\x1a\n" or b[:2] == b"\xff\xd8" or b[:3] == b"GIF"
            or (b[:4] == b"RIFF" and b[8:12] == b"WEBP"))


def load_posts():
    posts = []
    pdir = SITE_CONTENT / "research"
    if pdir.exists():
        for f in sorted(pdir.glob("*.md"), reverse=True):
            meta, body = mdlib.parse_post(f.read_text(encoding="utf-8"))
            date = meta.get("date") or f.name[:10]
            title = meta.get("title") or f.stem[11:].replace("-", " ").title()
            posts.append({
                "slug": _slug(f.stem), "date": date, "title": title,
                "tags": [t.strip() for t in (meta.get("tags") or "").split(",") if t.strip()],
                "summary": meta.get("summary", ""), "html": mdlib.render(body),
            })
    posts.sort(key=lambda p: p["date"], reverse=True)
    return posts


# ----------------------------- layout -----------------------------
def _page(cfg, books, *, title, active, body, embed=None, depth=0, asof=""):
    pre = "../" * depth
    items = [("home", "index.html", "Dashboard")]
    items += [(b["cfg"]["key"], f'{b["cfg"]["key"]}.html', b["cfg"].get("nav", b["cfg"]["label"]))
              for b in books.values()]
    items += [("analyze", "analyze.html", "Analyze"),
              ("basket", "basket.html", 'Basket <span class="bcount" hidden></span>'),
              ("mine", "portfolios.html", "My portfolios"),
              ("research", "research.html", "Research"),
              ("about", "about.html", "Methodology")]
    nav = "".join(f'<a href="{pre}{href}" class="{"on" if active == key else ""}">{label}</a>'
                  for key, href, label in items)
    desk = {"api_url": (cfg.get("api_url") or "").rstrip("/"), "finnhub_key": cfg.get("finnhub_key", ""),
            "prefix": pre, "code_url": cfg.get("code_url", "")}
    data = ""
    if embed is not None:
        embed = dict(embed)
        embed.setdefault("config", {})
        embed["config"]["data_prefix"] = pre + "data/"
        embed["config"]["asset_prefix"] = pre + "assets/"
        data = f'<script>window.SITE_DATA = {json.dumps(embed)};</script>'
    year = dt.date.today().year
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} · {_esc(cfg["brand"])}</title>
<meta name="description" content="{_esc(cfg["tagline"])}">
<link rel="stylesheet" href="{pre}assets/site.css">
<link rel="icon" href="{pre}assets/logo-chip.png" type="image/png">
</head><body>
<header class="site-head"><div class="wrap">
  <a class="brand" href="{pre}index.html">{(f'<img class="logo-chip" src="{pre}assets/logo-chip.png" alt="">' if cfg.get("_logo") else MARK)}<span class="wordmark">{_esc(cfg["brand"])}</span>
    <small>research</small></a>
  <nav class="main">{nav}</nav>
  <span class="asof"><span class="feedpill pill" id="feedpill">EOD</span><span class="txt">{_esc(asof)}</span></span>
</div></header>
<div class="demo-banner" style="background:#7a5a00;color:#fff;text-align:center;padding:8px 16px;font-size:13.5px;line-height:1.4"><b>Portfolio project.</b> A model (paper) portfolio built with my <a href="{_esc(cfg["code_url"])}" style="color:#fff;text-decoration:underline">open-source DCF engine</a>. Not a commercial offering and not investment advice.</div>
<main><div class="wrap">
{body}
</div></main>
<footer class="site-foot"><div class="wrap">
  <p class="disc">{DISCLAIMER}</p>
  <div class="foot-meta"><span>© {year} {_esc(cfg["brand"])}</span>
    <span>{_esc(cfg["contact_email"])}</span>
    <span>data: scheduled cloud pipeline{" + live intraday feed" if cfg.get("finnhub_key") else ""}</span>
    <a href="{pre}pricing.html">About this project</a></div>
</div></footer>
{data}<script>window.DESK = {json.dumps(desk)};</script>
<script src="{pre}assets/site.js"></script><script src="{pre}assets/desk.js"></script>
</body></html>"""


# ----------------------------- shared fragments -----------------------------
def _book_kpis(bk):
    st, bcfg = bk["status"], bk["cfg"]
    key = bcfg["key"]
    ret = st.get("return_since_inception")
    day = (st.get("metrics") or {}).get("day_change")
    prim = next(iter(st.get("benchmarks") or {}), None)
    active = (st["benchmarks"][prim]["active"] if prim else None)
    val = st.get("value")
    notional = " notional" if st.get("capital_basis") == "notional" else ""
    val_tile = (
        f'<div class="tile"><div class="lab">NAV (current)</div>'
        f'<div class="val tab">{_money(val, st["currency"])}</div>'
        f'<div class="sub">at inception {_money(st.get("value_at_inception") or st.get("capital"), st["currency"])}{notional}</div></div>'
        if val is not None else
        f'<div class="tile"><div class="lab">Holdings</div>'
        f'<div class="val tab">{len(st["holdings"])}</div>'
        f'<div class="sub">equal-weight model book</div></div>')
    return f"""<div class="kpis">
  {val_tile}
  <div class="tile"><div class="lab">Since inception</div>
    <div class="val tab {_cls(ret)}">{_pct(ret)}</div>
    <div class="sub">{st["inception"]} · {st["days"]}d</div></div>
  <div class="tile"><div class="lab">Active vs {_esc(_bm_label(prim or ""))}</div>
    <div class="val tab {_cls(active)}">{("–" if active is None else f"{active * 100:+.1f}")}<span style="font-size:14px"> pp</span></div>
    <div class="sub">benchmark {_pct(st["benchmarks"][prim]["return"] if prim else None)}</div></div>
  <div class="tile"><div class="lab">1-day move</div>
    <div class="val tab {_cls(day)}" id="day-{key}">{_pct(day, 2)}</div>
    <div class="sub"><span class="badge {bcfg["status_badge"]}">{"live record" if bcfg["status_badge"] == "live" else "preliminary"}</span></div></div>
</div>"""


def _heatmap_block(key):
    return f"""<div class="card"><div class="hm-wrap" id="hm-{key}" data-h="340"></div>
  <div class="hm-legend">
    <span class="k"><span class="sw" style="background:#0B7A5B"></span>▲ up</span>
    <span class="k"><span class="sw" style="background:#D97706"></span>— flat at close</span>
    <span class="k"><span class="sw" style="background:#C0453B"></span>▼ down</span>
    <span class="k" style="margin-left:8px">tile area = portfolio weight</span>
    <span class="hm-note">vs last close · hover for entry / price / weight</span></div></div>"""


def _news_list(items, key, limit=10, logos=None):
    inner = _news_items_html(items, limit, logos)
    return f'<div class="news" id="news-{key}">{inner}</div>'


def _news_items_html(items, limit=10, logos=None):
    if not items:
        return '<p class="lede">No headlines fetched yet.</p>'
    logos = logos or {}
    out = []
    for n in items[:limit]:
        link = _esc(n.get("link") or "#")
        sym = (n.get("symbol") or "").replace(".NS", "")
        chip = (f'<img class="nlogo" src="assets/{logos[sym]}" alt="" loading="lazy" '
                f'onerror="this.remove()">'
                if sym in logos else "")
        out.append(
            f'<a class="item" href="{link}" target="_blank" rel="noopener">'
            f'<span class="sym">{chip}{_esc(sym)}</span>'
            f'<span class="ttl">{_esc(n.get("title"))}</span>'
            f'<span class="src">{_esc(n.get("publisher") or "")} {_news_when(n.get("published"))}</span></a>')
    return "".join(out)


def _holdings_table(bk):
    st = bk["status"]
    ccy = st["currency"]
    logos = bk.get("logos") or {}
    rows = []
    for r in st["holdings"]:
        sym = r.get("symbol", r["ticker"])
        logo = (f'<img class="tlogo" src="assets/{logos[sym]}" alt="" loading="lazy" '
                f'onerror="this.classList.add(\'ph\');this.removeAttribute(\'src\')">'
                if sym in logos else '<span class="tlogo ph"></span>')
        name_cell = (f'<div class="tk-cell">{logo}<div><b>{_esc(r["ticker"])}</b>'
                     f'<span class="co">{_esc(r.get("company") or "")}</span></div></div>')
        save = _esc(json.dumps({"ticker": sym, "company": r.get("company") or "",
                                "intrinsic": r.get("current_intrinsic"), "price": r.get("price"),
                                "discount": r.get("current_discount"), "source": "model portfolio"}))
        rows.append(
            f'<tr data-save="{save}"><td>{name_cell}</td><td><span class="chip">{_esc(_sector(r.get("sector")))}</span></td>'
            f'<td>{_money(r.get("entry_price"), ccy, 2)}</td>'
            f'<td>{_money(r.get("price"), ccy, 2)}</td>'
            f'<td>{_pct(r.get("weight"), 1, False)}</td>'
            f'<td class="{_cls(r.get("day_change"))}">{_pct(r.get("day_change"), 2)}</td>'
            f'<td class="{_cls(r.get("return"))}">{_pct(r.get("return"))}</td>'
            f'<td>{_money(r.get("entry_value"), ccy)}</td>'
            f'<td>{_money(r.get("market_value"), ccy)}</td></tr>')
    cash = st.get("cash") or 0.0
    if cash > 0:
        rows.append('<tr><td><div class="tk-cell"><span class="tlogo ph"></span><div><b>Cash</b>'
                    '<span class="co">uninvested balance</span></div></div></td>'
                    f'<td></td><td></td><td></td><td></td><td></td><td></td>'
                    f'<td>{_money(cash, ccy)}</td><td>{_money(cash, ccy)}</td></tr>')
    ev = [r.get("entry_value") for r in st["holdings"]]
    mv = [r.get("market_value") for r in st["holdings"]]
    tot_e = (sum(v for v in ev if v) + cash) if any(ev) else st.get("value_at_inception")
    tot_m = (sum(v for v in mv if v) + cash) if any(mv) else st.get("value")
    ret = st.get("return_since_inception")
    notional = ' <span class="chip">notional base</span>' if st.get("capital_basis") == "notional" else ""
    total = (f'<tr class="total"><td><b>Total NAV</b>{notional}</td><td></td><td></td><td></td>'
             f'<td>{_pct(1.0, 0, False) if any(ev) else ""}</td><td></td>'
             f'<td class="{_cls(ret)}"><b>{_pct(ret)}</b></td>'
             f'<td><b>{_money(tot_e, ccy)}</b></td><td><b>{_money(tot_m, ccy)}</b></td></tr>')
    return ('<div class="tbl"><table><thead><tr><th>Name</th><th>Sector</th>'
            '<th>Entry @ inception</th><th>Last close</th><th>Weight</th><th>1-day</th>'
            '<th>Since entry</th><th>Value @ inception</th><th>Value now</th></tr>'
            f'</thead><tbody>{"".join(rows)}{total}</tbody></table></div>')


def _compare_section(key, title="How the portfolio stacks up"):
    return f"""<section><h2 class="sec">{title}
  <span class="sub">since inception · same window for every line · * modeled run-rate</span></h2>
<div class="card chart" id="cmp-{key}"></div></section>"""


def _comparables_table(bk):
    st = bk["status"]
    rows = []
    prim = next(iter(st.get("benchmarks") or {}), None)
    for bm, v in (st.get("benchmarks") or {}).items():
        rows.append(f'<tr><td>{_esc(_bm_label(bm))}{" <span class=chip>primary</span>" if bm == prim else ""}</td>'
                    f'<td><span class="badge live">index</span></td>'
                    f'<td class="{_cls(v["return"])}">{_pct(v["return"])}</td>'
                    f'<td class="{_cls(v["active"])}">{_pp(v["active"])}</td></tr>')
    for c in st.get("comparables") or []:
        kind = "model" if c["kind"] == "run_rate" else "live"
        klabel = "modeled*" if c["kind"] == "run_rate" else "index"
        rows.append(f'<tr><td>{_esc(c["name"].replace(" (modeled)", ""))}</td>'
                    f'<td><span class="badge {kind}">{klabel}</span></td>'
                    f'<td class="{_cls(c.get("return"))}">{_pct(c.get("return"))}</td>'
                    f'<td class="{_cls(c.get("active"))}">{_pp(c.get("active"))}</td></tr>')
    note = "".join(f'<p class="lede" style="font-size:12px;margin-top:8px">* {_esc(c["note"])}</p>'
                   for c in st.get("comparables") or [] if c["kind"] == "run_rate" and c.get("note"))
    return ('<div class="tbl"><table><thead><tr><th>Versus</th><th>Type</th><th>Window return</th>'
            f'<th>Book active</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>{note}')


def _risk_block(bk):
    p = (bk["risk"] or {}).get("portfolio") or {}
    if not p:
        return '<p class="lede">Run `portfolio manage riskstats` to populate.</p>'
    cells = [("Beta", p.get("beta"), "num"), ("Volatility (ann.)", p.get("vol"), "pct"),
             ("Tracking error", p.get("tracking_error"), "pct"),
             ("Dividend yield", p.get("div_yield"), "pct"),
             ("Diversification ratio", p.get("diversification_ratio"), "num"),
             ("Effective bets", p.get("effective_bets"), "num"),
             ("Realized alpha (3y)", p.get("realized_alpha"), "pct"),
             ("Max drawdown (3y)", p.get("max_drawdown"), "pct")]
    tiles = "".join(
        f'<div class="tile"><div class="lab">{lab}</div><div class="val tab" style="font-size:19px">'
        f'{(_pct(v) if kind == "pct" else ("–" if v is None else f"{v:.2f}"))}</div></div>'
        for lab, v, kind in cells)
    ev = (bk["risk"] or {}).get("evidence") or {}
    evline = ""
    if ev.get("caveat"):
        evline = f'<p class="lede" style="font-size:12px;margin-top:10px"><b>Evidence note:</b> {_esc(ev.get("source", ""))} — {_esc(ev["caveat"])}</p>'
    return f'<div class="kpis" style="grid-template-columns:repeat(4,1fr)">{tiles}</div>{evline}'


def _honesty_block(bk):
    h = bk["status"].get("honesty")
    if not h:
        return ""
    items = "".join(f"<li><b>{_esc(k.replace('_', ' '))}:</b> {_esc(v)}</li>" for k, v in h.items())
    return f'<section><div class="callout"><b>Read before the numbers</b><ul style="margin:8px 0 0;padding-left:18px">{items}</ul></div></section>'


def _embed_for(cfg, books, keys):
    return {"as_of": max((books[k]["status"]["as_of"] for k in keys), default=""),
            "config": {"finnhub_key": cfg.get("finnhub_key", ""),
                       "refresh_seconds": cfg.get("refresh_seconds", 60),
                       "quote_seconds": cfg.get("quote_seconds", 20),
                       "news_seconds": cfg.get("news_seconds", 300)},
            "books": {k: {"status": books[k]["status"], "track": books[k]["track"],
                          "logos": books[k].get("logos") or {},
                          "cfg": {"label": books[k]["cfg"]["label"]}} for k in keys}}


# ----------------------------- pages -----------------------------
def page_index(cfg, books, posts):
    us = books.get("us")
    ret = us["status"].get("return_since_inception") if us else None
    prim = next(iter(us["status"].get("benchmarks") or {}), None) if us else None
    active = us["status"]["benchmarks"][prim]["active"] if (us and prim) else None
    ev = ((us or {}).get("risk") or {}).get("evidence") or {}
    watermark = ('<img src="assets/logo-white.png" alt="">' if cfg.get("_logo") else _mark(360, "hm-svg"))
    hero = f"""<div class="hero">
  <div class="h-mark" aria-hidden="true">{watermark}</div>
  <div class="h-eyebrow">{_esc(cfg["brand"])} · model portfolio research</div>
  <h1 class="h-title">Own the gap between price and&nbsp;value.</h1>
  <p class="h-lede">A concentrated model portfolio built on one disciplined signal — full DCF
  intrinsic values, ranked, validated, and published with every caveat that institutional
  diligence would demand. No stories. Evidence.</p>
  <div class="h-chips">
    <span class="h-chip"><b class="{_cls(ret)}">{_pct(ret)}</b> since inception</span>
    <span class="h-chip"><b class="{_cls(active)}">{_pp(active)}</b> vs S&amp;P 500</span>
    <span class="h-chip"><b>{_pct(ev.get("backtest_alpha_spy"))}</b>/yr backtest alpha<sup>†</sup></span>
  </div>
  <div class="h-ctas"><a class="cta gold" href="{_esc(cfg["code_url"])}">View the code</a>
  <a class="cta ghost-inv" href="about.html">Read the methodology</a></div>
  <p class="h-foot">† 42-quarter backtest, net of costs, survivorship-caveated — details on the methodology page.</p>
</div>"""
    secs = [hero]
    for key, bk in books.items():
        label = _esc(bk["cfg"]["label"])
        secs.append(f"""<section>
<h2 class="sec">{label} <span class="sub">{_esc(bk["cfg"]["name"])} · inception {bk["status"]["inception"]}</span>
  <a href="{key}.html" style="margin-left:auto;font-size:12.5px">Full portfolio →</a></h2>
{_book_kpis(bk)}
<div style="height:14px"></div>
{_heatmap_block(key)}
</section>""")
        secs.append(_compare_section(key))
    latest = "".join(
        f'<a class="p-item" href="research/{p["slug"]}.html"><span class="d">{p["date"]}</span>'
        f'<h3>{_esc(p["title"])}</h3><p>{_esc(p["summary"])}</p></a>' for p in posts[:3])
    all_news = sorted((n for bk in books.values() for n in bk["news"]),
                      key=lambda n: str(n.get("published") or ""), reverse=True)
    first_key = next(iter(books), "us")
    secs.append(f"""<section class="grid2">
<div><h2 class="sec">Latest research</h2><div class="post-list">{latest or '<p class="lede">First note coming soon.</p>'}</div></div>
<div><h2 class="sec">News wire <span class="sub">constituent coverage · <span id="wire-src">EOD digest</span></span></h2>
<div class="card">{_news_list(all_news, first_key, 10, books[first_key].get("logos") if first_key in books else None)}</div></div>
</section>""")
    return _page(cfg, books, title="Dashboard", active="home", body="\n".join(secs),
                 embed=_embed_for(cfg, books, list(books)),
                 asof=max(b["status"]["as_of"] for b in books.values()))


def page_book(cfg, books, key):
    bk = books[key]
    st = bk["status"]
    label = _esc(bk["cfg"]["label"])
    prim = next(iter(st.get("benchmarks") or {}), None)
    ledes = {
        "us": "Ten US names, equal-weight, selected where the gap between our DCF intrinsic value "
              "and the market price is widest — and where that gap has historically been paid. "
              "Quarterly rebalance discipline; every action published with its rationale.",
    }
    body = f"""<p class="eyebrow">{_esc(st["book"])} · inception {st["inception"]} · {st["days"]} days</p>
<h1>{label}</h1>
<p class="lede">{ledes.get(key, "Concentrated model portfolio.")}</p>
{_book_kpis(bk)}
<section><h2 class="sec">Holdings heatmap <span class="sub">tile area = weight · ▲ green up / — amber flat / ▼ red down vs last close</span></h2>
{_heatmap_block(key)}</section>
{_compare_section(key)}
<section><h2 class="sec">Track record <span class="sub">cumulative since inception · benchmark overlays</span></h2>
<div class="card chart" id="tr-{key}"></div>
<div class="legend"><span class="it"><span class="sw" style="border-color:#0B7A5B"></span>Portfolio</span>
<span class="it"><span class="sw" style="border-color:#2F6DB3"></span>{_esc(_bm_label(prim or ""))}</span>
<span class="it"><span class="sw" style="border-color:#B9832A;border-top-style:dashed"></span>Modeled comparator*</span></div></section>
<section><h2 class="sec">Holdings</h2><div class="card">{_holdings_table(bk)}</div></section>
<section class="grid2">
<div><h2 class="sec">Versus funds &amp; indices</h2><div class="card">{_comparables_table(bk)}</div></div>
<div><h2 class="sec">Risk profile <span class="sub">3y weekly, total-return</span></h2>{_risk_block(bk)}</div>
</section>
{_honesty_block(bk)}
<section><h2 class="sec">News wire <span class="sub">constituent coverage</span></h2>
<div class="card">{_news_list(bk["news"], key, 12, bk.get("logos"))}</div></section>"""
    return _page(cfg, books, title=bk["cfg"]["label"], active=key, body=body,
                 embed=_embed_for(cfg, books, [key]), asof=st["as_of"])


def page_research(cfg, books, posts):
    items = "".join(
        f'<a class="p-item" href="research/{p["slug"]}.html"><span class="d">{p["date"]}</span>'
        f'<h3>{_esc(p["title"])}</h3><p>{_esc(p["summary"])}</p>'
        f'<div class="tags">{"".join(f"<span class=chip>{_esc(t)}</span>" for t in p["tags"])}</div></a>'
        for p in posts)
    body = f"""<p class="eyebrow">Daily notes</p><h1>Research</h1>
<p class="lede">Market views, sector work and portfolio rationale — written by the desk, dated,
and left standing (posts are never edited after publication; corrections get their own post).</p>
<div class="post-list" style="margin-top:22px">{items or '<p class="lede">First note coming soon.</p>'}</div>"""
    return _page(cfg, books, title="Research", active="research", body=body,
                 asof=posts[0]["date"] if posts else "")


def page_post(cfg, books, p):
    tags = "".join(f'<span class="chip">{_esc(t)}</span>' for t in p["tags"])
    body = f"""<article class="post">
<h1>{_esc(p["title"])}</h1>
<div class="meta">{p["date"]} · {_esc(cfg["brand"])}{(" · " if tags else "")}{tags}</div>
{p["html"]}
</article>
<p style="margin-top:18px"><a href="../research.html">← All research</a></p>"""
    return _page(cfg, books, title=p["title"], active="research", body=body, depth=1, asof=p["date"])


def page_pricing(cfg, books):
    """'Project' page (kept at pricing.html so old links resolve): what this site is and is not."""
    code = _esc(cfg["code_url"])
    body = f"""<p class="eyebrow">About this project</p><h1>A portfolio project, not a business</h1>
<p class="lede">{_esc(cfg["brand"])} is a <b>demonstration site</b> for my
<a href="{code}">DCF Valuation Platform</a>: a Kalman-filtered, state-space DCF engine whose
intrinsic values drive a concentrated <b>model (paper) portfolio</b>. Nothing here is for sale,
no subscriptions exist, and no client money is managed.</p>
<div class="plans" style="margin-top:26px"><div class="plan ">
<div class="nm">Explore the model</div><ul><li>Dashboard and full holdings history</li><li>Research notes and rebalance rationale</li><li>Methodology and validation evidence</li></ul>
<a class="cta ghost" href="us.html">See the portfolio</a></div><div class="plan hi">
<div class="nm">Read the code</div><ul><li>Valuation engine, portfolio and research layers</li><li>Backtests, ledgers and validation suites</li><li>Excel workbooks as the primary deliverable</li></ul>
<a class="cta gold" href="{code}">View on GitHub</a></div></div>
<section><div class="callout"><b>Regulatory note.</b> The publisher is not registered with SEBI
(as a Research Analyst, Investment Adviser or Portfolio Manager) or with the US SEC. This site is
published for education and as a portfolio piece only; it offers no paid research or advice.
</div></section>"""
    return _page(cfg, books, title="About this project", active="pricing", body=body)


def page_about(cfg, books):
    us = books.get("us", {})
    ev = ((us.get("risk") or {}).get("evidence") or {})
    body = f"""<p class="eyebrow">Methodology &amp; evidence</p><h1>How the portfolio is built</h1>
<p class="lede">One signal, applied with discipline: a full DCF on every name in a validated
universe, ranked by discount to intrinsic value and sized by how reliably that discount has
predicted returns in that sector family. Everything below is measured by the same code that
runs the portfolio.</p>
<section class="grid2">
<div class="card"><h3>The signal</h3><p style="color:var(--muted);font-size:13.5px">
Point-in-time fundamentals → conservative DCF → discount = intrinsic / price − 1. Discounts are
clipped, ranked within validated sector families, and multiplied by a per-sector reliability
score. Top-10 equal-weight; quarterly rebalance discipline; positions never leave the book
between rebalances.</p></div>
<div class="card"><h3>The evidence — stated honestly</h3><p style="color:var(--muted);font-size:13.5px">
US backtest (42 quarters, net of costs): alpha {_pct(ev.get("backtest_alpha_spy"))} /yr vs the
S&amp;P 500 at beta {ev.get("backtest_beta_spy", "–")}, {_pct(ev.get("backtest_alpha_qqq"))} vs the
Nasdaq 100. <b>Caveat carried on every page:</b> {_esc(ev.get("caveat", "survivorship-caveated"))}.</p></div>
</section>
<section class="grid2">
<div class="card"><h3>Two betas, both true</h3><p style="color:var(--muted);font-size:13.5px">
The <b>strategy's</b> 11-year backtest beta is ~1.43 — the average book it held. The <b>current
holdings'</b> trailing-3y beta is ~0.51: a defensive, high-yield basket down ~40% from its peaks.
That gap is the strategy: the signal buys after the damage. We label which beta every figure uses.</p></div>
<div class="card"><h3>Operations</h3><p style="color:var(--muted);font-size:13.5px">
One command refreshes everything (prices → track → risk → site). News is fetched per holding from
financial-news aggregation and labeled with its original outlet. Dashboard data is end-of-day
pipeline output and upgrades to a direct live feed when a market-data key is configured. Nothing
on this site is computed twice — pages render the same JSON the desk uses.</p></div>
</section>
<section><div class="callout"><b>What would change our mind.</b> The claim is scoped to the
consumer/staples-family sectors where it survived corrected validation; if the live record decays
past its historical drawdown envelope, the book gets cut, and the post explaining why stays up.
</div></section>"""
    return _page(cfg, books, title="Methodology", active="about", body=body,
                 asof=max((b["status"]["as_of"] for b in books.values()), default=""))


def _universe(books):
    """Names researched on the platform (model-book holdings) for the basket / portfolio builder."""
    out = []
    for bk in books.values():
        for h in bk["status"].get("holdings", []):
            out.append({"ticker": h.get("symbol", h["ticker"]), "company": h.get("company") or "",
                        "intrinsic": h.get("current_intrinsic"), "price": h.get("price"),
                        "discount": h.get("current_discount"), "sector": _sector(h.get("sector")),
                        "source": bk["cfg"]["label"]})
    return out


def _desk_embed(books):
    return f"<script>window.DESK_UNIVERSE = {json.dumps(_universe(books))};</script>"


def page_analyze(cfg, books):
    picks = "".join(f'<button type="button" class="qpick" data-tk="{_esc(u["ticker"])}">{_esc(u["ticker"])}</button>'
                    for u in _universe(books)[:10])
    offline = "" if cfg.get("api_url") else (
        '<div class="callout" style="margin-top:14px"><b>Analysis server not configured.</b> '
        'Set <code>api_url</code> in site.json (or DCF_API_URL) to the deployed API.</div>')
    body = f"""<p class="eyebrow">On-demand research</p><h1>Analyze any company</h1>
<p class="lede">Type a US ticker. The platform pulls the company's SEC filings and market data,
runs the <b>Kalman-filtered state-space DCF</b> live, and returns the full report (valuation, forecast,
sensitivity, drivers and thesis), with the Excel workbook and PDF to download.</p>
<form class="az-form" id="az-form" autocomplete="off">
  <input id="az-input" class="az-input" placeholder="Ticker, e.g. KO, MSFT, COST" maxlength="14"
         aria-label="Ticker symbol" spellcheck="false">
  <button class="cta az-go" type="submit">Run DCF</button>
</form>
<div class="qpicks"><span>From the model portfolio:</span>{picks}</div>
{offline}
<div id="az-status"></div>
<div id="az-report"></div>
{_desk_embed(books)}"""
    return _page(cfg, books, title="Analyze", active="analyze", body=body)


def page_basket(cfg, books):
    body = f"""<p class="eyebrow">Saved in this browser</p><h1>Your basket</h1>
<p class="lede">Companies you've saved from an analysis or from the model portfolio. The basket is
kept in this browser, so it's here when you come back. Prices update live; select names to build a portfolio.</p>
<div id="bk-root"></div>
{_desk_embed(books)}"""
    return _page(cfg, books, title="Basket", active="basket", body=body)


def page_portfolios(cfg, books):
    body = f"""<p class="eyebrow">Your research, your book</p><h1>My portfolios</h1>
<p class="lede">Build paper portfolios from your basket and the platform's research. Each one
records entry prices on the day you create it and tracks live value and return from then on.
Everything stays in this browser; export a file to move it to another device.</p>
<div id="pf-root"></div>
{_desk_embed(books)}"""
    return _page(cfg, books, title="My portfolios", active="mine", body=body)


# ----------------------------- build -----------------------------
def build(managed_dir=None, out_dir=None, config=None) -> Path:
    out = Path(out_dir) if out_dir else SITE_OUT
    cfg = dict(config) if config else load_config()
    books = load_books(cfg, managed_dir)
    if not books:
        raise SystemExit("no books with status.json found -- run `portfolio manage update-all` first.")
    posts = load_posts()

    (out / "research").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    for a in ASSETS.iterdir():
        shutil.copy2(a, out / "assets" / a.name)
    cfg["_logo"] = _process_brand_logo(out)
    if cfg.get("custom_domain"):  # GitHub Pages custom domain: CNAME travels with every deploy
        (out / "CNAME").write_text(cfg["custom_domain"].strip() + "\n", encoding="utf-8")
    for key, bk in books.items():
        (out / "data" / f"{key}_status.json").write_text(json.dumps(bk["status"]), encoding="utf-8")
        (out / "data" / f"{key}_track.json").write_text(json.dumps(bk["track"]), encoding="utf-8")
        if bk.get("logos"):  # self-host company logos next to the pages
            (out / "assets" / "logos").mkdir(exist_ok=True)
            for rel in bk["logos"].values():
                src = bk["folder"] / "figures" / rel
                if src.exists():
                    shutil.copy2(src, out / "assets" / rel)

    pages = {"index.html": page_index(cfg, books, posts),
             "research.html": page_research(cfg, books, posts),
             "pricing.html": page_pricing(cfg, books),
             "about.html": page_about(cfg, books),
             "analyze.html": page_analyze(cfg, books),
             "basket.html": page_basket(cfg, books),
             "portfolios.html": page_portfolios(cfg, books)}
    for key in books:
        pages[f"{key}.html"] = page_book(cfg, books, key)
    for p in posts:
        pages[f"research/{p['slug']}.html"] = page_post(cfg, books, p)
    # drop stale pages (a book removed from config, a research post renamed/retired)
    for old in list(out.glob("*.html")) + list(out.glob("research/*.html")):
        rel = old.relative_to(out).as_posix()
        if rel not in pages:
            old.unlink()
    for rel, htmltext in pages.items():
        f = out / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(htmltext, encoding="utf-8")
    for stale in out.glob("data/*_status.json"):
        if stale.stem.replace("_status", "") not in books:
            stale.unlink()
            tr = stale.with_name(stale.name.replace("_status", "_track"))
            if tr.exists():
                tr.unlink()
    print(f"site: {len(pages)} pages, {len(books)} books, {len(posts)} posts -> {out}")
    if not (SITE_CONTENT / "site.json").exists():
        print(f"  note: create {SITE_CONTENT / 'site.json'} to set brand/pricing/feed key (defaults used).")
    return out


def deploy(remote: str | None = None) -> None:
    """Push site/ to the public Pages repo (config `deploy_repo`), via a cached clone.

    Flow: build is assumed done; clone-or-pull CACHE/site_deploy, mirror site/ into it,
    commit if anything changed, push. GitHub Pages redeploys automatically on push."""
    import subprocess

    from dcf_project.paths import CACHE

    cfg = load_config()
    remote = remote or cfg.get("deploy_repo")
    if not remote:
        raise SystemExit('no deploy repo configured -- set "deploy_repo" in site_content/site.json')
    clone = CACHE / "site_deploy"

    def git(*args, cwd=clone):
        return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)

    if not (clone / ".git").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", remote, str(clone)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"clone failed: {r.stderr.strip()}")
    else:
        git("pull", "--ff-only")
    # mirror site/ -> clone (keep .git)
    for item in clone.iterdir():
        if item.name == ".git":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    shutil.copytree(SITE_OUT, clone, dirs_exist_ok=True)
    (clone / ".nojekyll").touch()
    git("add", "-A")
    if not git("status", "--porcelain").stdout.strip():
        print("deploy: no changes to publish.")
        return
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    c = git("commit", "-m", f"site update {stamp}")
    if c.returncode:
        raise SystemExit(f"commit failed: {c.stderr.strip() or c.stdout.strip()}")
    p = git("push")
    if p.returncode:
        raise SystemExit(f"push failed: {p.stderr.strip()}")
    print(f"deployed -> {remote} (Pages redeploys automatically, ~1 min)")


def new_post(title: str, tags: str = "", summary: str = "") -> Path:
    today = dt.date.today().isoformat()
    pdir = SITE_CONTENT / "research"
    pdir.mkdir(parents=True, exist_ok=True)
    f = pdir / f"{today}-{_slug(title)}.md"
    if f.exists():
        raise SystemExit(f"post already exists: {f}")
    f.write_text(f"""---
title: {title}
date: {today}
tags: {tags}
summary: {summary or "One-line summary shown on the research index."}
---

Write the note here. Markdown: **bold**, *italic*, `code`, [links](https://example.com),
lists with `-`, quotes with `>`. Keep the claim, the evidence, and the caveat together.
""", encoding="utf-8")
    print(f"created {f}\n  edit it, then run: python -m dcf_project site build")
    return f
