"""Phoenix Capital analyzer: any US ticker -> live canonical Kalman/state-space DCF -> full report.

Runs on Streamlit Community Cloud (free) and is embedded in the site's Analyze page
(``?embed=true&t=KO``). It calls the SAME code path as ``python -m dcf_project valuation canonical``
via ``dcf_project.api.service.analyze``; this file is presentation only.

Secrets (Streamlit Cloud -> app -> Settings -> Secrets):
    DCF_SEC_USER_AGENT = "Your Name your-email@example.com"   # SEC EDGAR contact (required)
    SITE_URL = "https://ps11242026.github.io/phoenix-capital/"   # optional
"""
from __future__ import annotations

import base64
import html
import json
import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Phoenix Capital · Analyze", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

for k in ("DCF_SEC_USER_AGENT", "SITE_URL"):
    try:
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])
    except Exception:  # noqa: BLE001 -- no secrets file locally
        pass
_TMP = Path(tempfile.gettempdir()) / "phoenix-dcf"
os.environ.setdefault("DCF_REPORTS_DIR", str(_TMP / "reports"))
os.environ.setdefault("DCF_CACHE_DIR", str(_TMP / "cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMP / "mpl"))
SITE = os.environ.get("SITE_URL", "https://ps11242026.github.io/phoenix-capital/").rstrip("/") + "/"
SITE_ORIGIN = "/".join(SITE.split("/")[:3])
OUT = _TMP / "valuation"

from dcf_project.api import service  # noqa: E402

# ---------------------------------------------------------------- style
st.markdown("""<style>
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;height:0}
.block-container{padding-top:1.2rem;max-width:1080px}
html, body, [class*="css"]{font-family:"Segoe UI Variable Text","Segoe UI",system-ui,sans-serif}
.rp-head{background:radial-gradient(800px 260px at 0% -40%,rgba(11,122,91,.30),transparent 60%),#0F141B;
  color:#fff;border-radius:14px;padding:24px 26px;margin:6px 0 14px}
.rp-tk{font:700 12px Consolas,monospace;letter-spacing:.14em;color:#C9A227;background:rgba(201,162,39,.12);
  border:1px solid rgba(201,162,39,.35);border-radius:6px;padding:3px 8px}
.rp-head h2{font-family:Georgia,serif;font-size:30px;font-weight:600;margin:10px 0 4px;color:#fff}
.rp-sub{font-size:12.5px;color:#9FA9B5;text-transform:capitalize}
.verdict{display:inline-block;font-size:13px;font-weight:650;border-radius:99px;padding:4px 12px}
.v-pos{background:#EAF5F0;color:#075C44}.v-neg{background:#F9ECEA;color:#8E2F27}.v-mid{background:#F0F2F5;color:#5B6675}
.rb{position:relative;height:10px;border-radius:99px;background:#EDEFF3;margin:34px 10px 8px}
.rb .band{position:absolute;top:0;bottom:0;border-radius:99px;
  background:linear-gradient(90deg,rgba(11,122,91,.25),rgba(11,122,91,.55),rgba(11,122,91,.25))}
.rb .iv{position:absolute;top:-5px;width:3px;height:20px;margin-left:-1.5px;background:#075C44;border-radius:2px}
.rb .px{position:absolute;top:-12px;margin-left:-6px;border:6px solid transparent;border-top:9px solid #0F141B}
.rb .px i{position:absolute;top:-22px;left:-14px;font:600 10px Consolas,monospace;font-style:normal}
.rbl{position:relative;height:20px;margin:0 10px 6px;font:11px Consolas,monospace;color:#5B6675}
.rbl span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.fine{font-size:12px;color:#8A94A2}
div[data-testid="stMetricValue"]{font-family:Consolas,monospace}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- formatting (mirrors the site)
def fmt(v, f=None) -> str:
    if v is None or v == "":
        return "–"
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "yes" if v else "no"
    cur = "₹" if (f or "").startswith("inr") else "$"
    a, s = abs(v), "-" if v < 0 else ""
    if f in ("usd", "inr"):
        return f"{s}{cur}{a:,.2f}"
    if f in ("usd0", "inr0"):
        return f"{s}{cur}{a:,.0f}"
    if f in ("usdM", "inrM"):
        return f"{s}{cur}{a / 1e6:,.0f}M"
    if f in ("usdB", "inrB"):
        return f"{s}{cur}{a / 1e9:,.1f}B"
    if f in ("pct", "pct1"):
        return f"{v * 100:.1f}%"
    if f == "pct2":
        return f"{v * 100:.2f}%"
    if f == "num0":
        return f"{v:,.0f}"
    if f == "x":
        return f"{v:.1f}x"
    if f == "yrs":
        return f"{v:.1f} yrs"
    if f == "bp":
        return f"{v * 1e4:,.0f} bp"
    return f"{v:,.0f}" if a >= 1000 else f"{v:,.2f}"


def pct(x, dp=1):
    return "–" if x is None else f"{'+' if x > 0 else ''}{x * 100:.{dp}f}%"


def vcls(v: str) -> str:
    v = (v or "").lower()
    return "v-neg" if "unattractive" in v else "v-pos" if "attractive" in v else "v-mid"


# ---------------------------------------------------------------- engine (cached per ticker)
@st.cache_data(ttl=12 * 3600, show_spinner=False, max_entries=200)
def run(ticker: str) -> dict:
    rep = service.analyze(ticker, OUT)
    files = {}
    for kind, name in (rep.get("files") or {}).items():
        p = OUT / rep["ticker"] / name
        if p.exists():
            files[kind] = (f"{rep['ticker']}_{name}" if kind == "xlsx" else name, p.read_bytes())
    rep["_bytes"] = files
    return rep


# ---------------------------------------------------------------- UI
qp = st.query_params
t0 = (qp.get("t") or "").upper()
embedded = qp.get("from") == "site"   # set by the site's Analyze page (Streamlit hides `embed`)

if not embedded:
    st.markdown("### Analyze any company")
    st.caption("Live Kalman-filtered state-space DCF from SEC filings. Part of "
               f"[Phoenix Capital]({SITE}), a portfolio project; not investment advice.")
if embedded:          # the site's own search box drives the embedded analyzer
    raw, go = t0, False
else:
    with st.form("az", border=False):
        c1, c2 = st.columns([4, 1])
        raw = c1.text_input("Ticker", value=t0, placeholder="Ticker, e.g. KO, MSFT, COST",
                            label_visibility="collapsed", max_chars=14)
        go = c2.form_submit_button("Run DCF", type="primary", width="stretch")

ticker = (raw or "").strip().upper()
if go and ticker:
    st.query_params["t"] = ticker
if not ticker:
    st.info("Type a US ticker and press **Run DCF**. A full report takes about 20–60 seconds "
            "(SEC filings → Kalman filter → DCF → sensitivity → report).")
    st.stop()

try:
    service.normalize_ticker(ticker)
except service.AnalysisError as e:
    st.error(str(e))
    st.stop()

if not os.environ.get("DCF_SEC_USER_AGENT"):
    st.warning("Server setup: add the DCF_SEC_USER_AGENT secret (your name and email) so SEC EDGAR "
               "accepts requests.")

try:
    with st.status(f"Running the canonical DCF on {ticker}…", expanded=False) as status:
        st.write("Fetching SEC filings and market data · fitting the Kalman state-space FCF model · "
                 "discounting cash flows · building the report")
        r = run(ticker)
        status.update(label=f"{ticker} analyzed", state="complete")
except service.AnalysisError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Unexpected failure while analyzing {ticker}: {e.__class__.__name__}")
    st.stop()

iv, px, d = r.get("intrinsic_value"), r.get("price"), r.get("discount")
rng = r.get("range_p10_p90") or [None, None]

st.markdown(f"""<div class="rp-head"><span class="rp-tk">{html.escape(r['ticker'])}</span>
<h2>{html.escape(r.get('company') or r['ticker'])}</h2>
<div class="rp-sub">{html.escape((r.get('sector') or '').replace('_', ' '))} ·
{html.escape((r.get('regime') or '').replace('_', ' '))} · analyzed {html.escape(r.get('as_of') or '')}</div></div>""",
            unsafe_allow_html=True)

# actions: save to the site's basket (postMessage to the embedding page, or a link when standalone)
item = {"ticker": r["ticker"], "company": r.get("company"), "intrinsic": iv, "price": px, "discount": d,
        "verdict": r.get("verdict"), "p10": rng[0], "p90": rng[1], "analyzed_at": r.get("as_of"),
        "source": "analysis"}
b64 = base64.urlsafe_b64encode(json.dumps(item).encode()).decode()
a1, a2, a3 = st.columns([1.3, 1, 1])
with a1:
    st.components.v1.html(f"""
<button id="s" style="width:100%;height:40px;border:0;border-radius:8px;background:#0B7A5B;color:#fff;
 font:600 14px 'Segoe UI',system-ui,sans-serif;cursor:pointer">☆ Save to basket</button>
<script>
const item = {json.dumps(item)};
const embedded = {json.dumps(embedded)};
document.getElementById("s").onclick = function () {{
  if (embedded) {{
    window.top.postMessage({{type: "phx-save", item: item}}, "{SITE_ORIGIN}");
    this.textContent = "★ Saved to your basket";
  }} else {{
    window.open("{SITE}basket.html?add={b64}", "_blank", "noopener");
    this.textContent = "★ Opened your basket";
  }}
}};
</script>""", height=44)
xl, pdf = r["_bytes"].get("xlsx"), r["_bytes"].get("pdf")
if xl:
    a2.download_button("Excel workbook", xl[1], file_name=xl[0], width="stretch",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
if pdf:
    a3.download_button("PDF report", pdf[1], file_name=pdf[0], width="stretch",
                       mime="application/pdf")

k1, k2, k3, k4 = st.columns(4)
k1.metric("DCF intrinsic / share", fmt(iv, "usd"), help=f"P50 {fmt(r.get('p50'), 'usd')} · engine {r.get('engine')}")
k2.metric("Price at analysis", fmt(px, "usd"))
k3.metric("Upside to intrinsic", pct(d), delta=pct(d), delta_color="normal" if d is not None else "off")
k4.markdown(f"<div style='font-size:14px;color:#5B6675'>Verdict</div><div style='margin-top:6px'>"
            f"<span class='verdict {vcls(r.get('verdict'))}'>{html.escape(r.get('verdict') or '–')}</span></div>"
            f"<div class='fine' style='margin-top:6px'>{html.escape(r.get('confidence') or '')}</div>",
            unsafe_allow_html=True)

if rng[0] is not None and rng[1] is not None and iv:
    lo = min(rng[0], px or rng[0], iv) * 0.9
    hi = max(rng[1], px or rng[1], iv) * 1.05

    def P(v):
        return (v - lo) / (hi - lo) * 100

    st.markdown(f"""<div style="margin-top:14px;font-weight:650;font-size:14px">Where the price sits in the model's
value range <span class="fine">P10 – P90 intrinsic value</span></div>
<div class="rb"><span class="band" style="left:{P(rng[0]):.2f}%;width:{P(rng[1]) - P(rng[0]):.2f}%"></span>
<span class="iv" style="left:{P(iv):.2f}%"></span>{f'<span class="px" style="left:{P(px):.2f}%"><i>price</i></span>' if px else ''}</div>
<div class="rbl"><span style="left:{P(rng[0]):.2f}%">P10 {fmt(rng[0], 'usd')}</span><span style="left:{P(iv):.2f}%"><b>DCF {fmt(iv, 'usd')}</b></span>
<span style="left:{P(rng[1]):.2f}%">P90 {fmt(rng[1], 'usd')}</span></div>""", unsafe_allow_html=True)

# history + forecast chart ($M)
try:
    import altair as alt
    import pandas as pd

    rows = []
    h, f = r.get("history") or {}, r.get("forecast") or {}
    hp = [str(p)[:7] for p in (h.get("period") or [])]
    for i, p in enumerate(hp):
        for key, name in (("revenue", "Revenue"), ("fcf", "FCF")):
            v = (h.get(key) or [None] * len(hp))[i]
            if v is not None:
                rows.append({"i": i, "label": p, "series": f"{name} (reported)", "value": v / 1e6})
    n = len(hp)
    for j in range(len(f.get("revenue") or [])):
        for key, name in (("revenue", "Revenue"), ("fcf", "FCF")):
            v = (f.get(key) or [None])[j] if j < len(f.get(key) or []) else None
            if v is not None:
                rows.append({"i": n + j, "label": f"F{j + 1}", "series": f"{name} (forecast)", "value": v / 1e6})
    if rows:
        df = pd.DataFrame(rows)
        colors = alt.Scale(domain=["Revenue (reported)", "Revenue (forecast)", "FCF (reported)", "FCF (forecast)"],
                           range=["#0F141B", "#0B7A5B", "#8A94A2", "#2F6DB3"])
        ch = alt.Chart(df).mark_line(strokeWidth=2.2).encode(
            x=alt.X("i:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("value:Q", title="$ millions"),
            color=alt.Color("series:N", scale=colors, legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.condition("indexof(datum.series, 'forecast') > 0", alt.value([6, 5]), alt.value([1, 0])),
            tooltip=["label", "series", alt.Tooltip("value:Q", format=",.0f", title="$M")])
        rule = alt.Chart(pd.DataFrame({"i": [n - 0.5]})).mark_rule(strokeDash=[3, 4], color="#C4CBD4").encode(x="i:Q")
        st.markdown("**Revenue & free cash flow** <span class='fine'>quarterly · reported history, then the "
                    "Kalman-filtered forecast (dashed)</span>", unsafe_allow_html=True)
        st.altair_chart((ch + rule).properties(height=280), width="stretch")
except Exception:  # noqa: BLE001 -- the chart is decorative; never lose the report over it
    pass

# full report sections
secs = (r.get("model") or {}).get("sections") or []
if secs:
    tabs = st.tabs([s["title"] for s in secs])
    for tab, s in zip(tabs, secs):
        with tab:
            if s["metrics"]:
                cols = st.columns(min(4, len(s["metrics"])))
                for i, m in enumerate(s["metrics"]):
                    cols[i % len(cols)].metric(m["label"], fmt(m["value"], m["fmt"]), help=m.get("note"))
            for c in s.get("charts") or []:
                try:
                    import pandas as pd
                    df = pd.DataFrame(c["rows"], columns=c["columns"])
                    ys = [c["columns"][k] for k in c["y_cols"]]
                    df = df.set_index(c["columns"][c["x_col"]])[ys].apply(lambda col: col / 1e6)
                    st.caption(f"{c['title']} ($ millions)")
                    st.line_chart(df, height=240)
                except Exception:  # noqa: BLE001
                    pass
            for tb in s["tables"]:
                import pandas as pd
                fm = tb.get("formats") or []
                data = [[fmt(v, fm[i] if i < len(fm) else None) for i, v in enumerate(row)] for row in tb["rows"]]
                if tb.get("title"):
                    st.markdown(f"**{tb['title']}**")
                st.dataframe(pd.DataFrame(data, columns=tb["columns"]), hide_index=True, width="stretch")
            for im in s["images"]:
                if im.get("src", "").startswith("data:image"):
                    st.image(base64.b64decode(im["src"].split(",", 1)[1]), caption=im["title"])
            nar = [x for x in s["narrative"] if x and x.strip()]
            if nar:
                st.markdown("\n".join(f"**{x}**" if x.endswith(":") else f"- {x}" for x in nar))

st.markdown(f"<p class='fine' style='margin-top:18px'>Generated live by the open-source canonical engine "
            f"({html.escape(str(r.get('engine')))}) from SEC EDGAR filings and market data. Research and education "
            "only; not investment advice.</p>", unsafe_allow_html=True)
