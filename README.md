# DCF Valuation Platform

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Tests](https://img.shields.io/badge/tests-293%20passing-brightgreen)

A validated discounted-cash-flow research platform organized as **four products** behind a single
CLI, with **Excel workbooks as the primary deliverable**. The canonical DCF engine — a
Kalman-filtered, state-space free-cash-flow forecaster with evidence-calibrated priors — is treated
as a **frozen contract**; the valuation, portfolio, research and reporting layers are all built on
top of it without changing the math.

It runs entirely on **free public data**: SEC CompanyFacts/submissions (fundamentals), Yahoo Finance
via `yfinance` (price / shares / beta with fallbacks), and the U.S. Treasury par-yield curve
(risk-free fallback).

> This repository is a **curated snapshot**: the full source, the test suite, and a small set of
> **sample outputs** you can open directly (see [Example outputs](#example-outputs)). The large
> regenerable report trees are produced on demand by the CLI and are intentionally not versioned.

## Highlights

- **State-space FCF forecasting** — a Kalman-filtered latent-growth model (`src/dcf_project/models/`,
  `engine_v2/`, `engine_v3/`) rather than a hand-typed growth ramp.
- **Economic free cash flow** — decomposes reported capex into maintenance vs. growth so terminal
  value isn't contaminated by a peak expansion cycle (see [Methodology](#methodology-economic-fcf-and-hyperscaler-capex)).
- **Sell-side-quality output** — 7-section initiation workbooks (Executive Summary → Thesis), plus
  on-demand PDF and PowerPoint renderers from a single write-once `ReportModel`.
- **A full research layer** — survivorship-robust, capacity- and cost-aware backtests of the
  discount-to-intrinsic signal, packaged into a reproducible FAJ-style paper.
- **293 tests, all green** — `python -m pytest`.

## Quick start

```bash
git clone https://github.com/<your-user>/dcf-valuation-platform.git
cd dcf-valuation-platform
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
python -m pip install -e .
export DCF_SEC_USER_AGENT="Your Name your-email@example.com"   # required for live SEC calls
python -m dcf_project --help
```

## The four products

### 1. Valuation engine — company → DCF → Excel
```bash
python -m dcf_project valuation generic   GOOGL          # textbook baseline DCF
python -m dcf_project valuation canonical GOOGL --forecast-mode consensus_blend --wacc-source valueinvesting
python -m dcf_project valuation screen    basket.csv     # rank a basket (CSV with a 'ticker' column)
```
Outputs → `reports/valuation/<TICKER>/`: `generic_dcf.xlsx` / `canonical_dcf.xlsx` (7-section
sell-side initiation report: Executive Summary, Historical, Forecast, DCF, Sensitivity, Drivers,
Thesis), `forecast.csv`, `assumptions.json`.

### 2. Portfolio constructor — validated signal → PROPOSAL
```bash
python -m dcf_project portfolio construct --signal discount_x_conf_top10 --capital 1000000
```
Output → `reports/portfolio_builds/portfolio_YYYYQX/` (investment-committee workbook + `build.json`).
**A proposal only — it never becomes a live portfolio automatically.**

### 3. Portfolio manager — manage a LIVE portfolio
```bash
python -m dcf_project portfolio promote portfolio_2026Q3            # PROPOSAL -> MANAGED (explicit, audited)
python -m dcf_project portfolio manage update    --book conf_top10  # refresh PM dashboard (no mutation)
python -m dcf_project portfolio manage rebalance --book conf_top10  # EXECUTE (only command that mutates)
```
Each managed portfolio is a self-contained folder `reports/managed_portfolios/<book>/` (7-section PM
dashboard, `state.json`, `track.json`, `rebalance_log.json`, `figures/`). The registry records
inception, capital, strategy and signal version.

### 4. Research package — publication evidence
```bash
python -m dcf_project research build-paper                       # -> reports/paper/
```

### 5. On-demand analysis API + research desk
```bash
pip install -e ".[api]"
uvicorn dcf_project.api.app:app --port 7860      # GET /api/analyze/{ticker} (job + poll), /api/docs
```
Any US ticker -> the same canonical Kalman/state-space valuation as `valuation canonical`, returned
as a JSON report (every ReportModel section) plus the Excel workbook and PDF. The static site's
**Analyze** page calls it (`api_url` in `site_content/site.json` or `DCF_API_URL`); **Basket** and
**My portfolios** let visitors save names and build paper portfolios, stored in their own browser.
The free deployment is a **Streamlit Community Cloud** app ([`deploy/streamlit/`](deploy/streamlit/), main file `deploy/streamlit/streamlit_app.py`) embedded in the Analyze page via `analyzer_url`; the JSON API can also be self-hosted (`deploy/hf-space*/`). Without the
validation research files the verdict is *absolute* (discount to the name's own intrinsic value)
and says so; no relative rank is invented.

### Export layer (on-demand; Excel stays primary)
```bash
python -m dcf_project export pdf GOOGL
python -m dcf_project export ppt conf_top10
```

## Example outputs

Open these directly from the repo — no install required:

| Artifact | Path |
| --- | --- |
| GOOGL canonical DCF workbook (7-section initiation) | [`examples/valuations/GOOGL/canonical_dcf.xlsx`](examples/valuations/GOOGL/canonical_dcf.xlsx) |
| GOOGL generic (textbook) DCF workbook | [`examples/valuations/GOOGL/generic_dcf.xlsx`](examples/valuations/GOOGL/generic_dcf.xlsx) |
| GOOGL initiation report (PDF render) | [`examples/valuations/GOOGL/GOOGL.pdf`](examples/valuations/GOOGL/GOOGL.pdf) |
| Forecast path & assumptions | [`examples/valuations/GOOGL/forecast.csv`](examples/valuations/GOOGL/forecast.csv), [`assumptions.json`](examples/valuations/GOOGL/assumptions.json) |
| Reproducible research paper (figures + tables) | [`reports/paper/master_research_report.md`](reports/paper/master_research_report.md) |
| Static research site (generated from the managed book) | [`site/index.html`](site/index.html) |

## Repository layout
```
src/dcf_project/
  models/          state-space FCF model (Kalman filter, priors, regimes, overlays)
  engine_v2/  engine_v3/   posterior forecasting pipelines (sector priors, fade, terminal, MC)
  data_sources/    SEC, market, pricing, WACC (+ India / Canada extensions)
  valuation/       generic / canonical / screening + DCF math (dcf.py, baseline.py, roic.py, ...)
  portfolio/       scoring, signals, snapshot, analytics, registry, constructor, manager
  research/        paper builder + validation evidence
  reporting_kit/   ReportModel + xlsx / pdf / pptx renderers + 2D/3D viz (write once, render many)
  site/            static-site generator
  cli.py  __main__.py  paths.py  config.py
scripts/           paper / portfolio / utilities (the research suite)
tests/             293 tests
docs/              methodology.md, RESEARCH_POLICY.md
examples/          sample valuation outputs (open directly)
reports/paper/     the reproducible research package (source md + figures + tables)
site/              generated static research site
```

## Engineering notes
- **Frozen engine:** the canonical DCF math, forecast engine, ranking and validation logic are
  treated as a frozen contract — the portfolio, reporting and research layers build on top without
  modifying them.
- **Tests green:** `python -m pytest` → 281 passing.
- **Config over code:** modeling assumptions (ERP, tax, terminal-growth cap, capex normalization,
  SBC add-back) are environment-driven; see [`.env.example`](.env.example).

## Methodology: economic FCF and hyperscaler capex

Accounting free cash flow is not always the same as valuation economic cash flow. For
high-reinvestment compounders (hyperscalers, platforms), reported capex blends maintenance and
growth infrastructure; treating all of it as permanent FCF destruction understates steady-state
earning power and contaminates terminal value with a peak expansion cycle. The engine decomposes:

```text
total capex   = maintenance capex + growth capex
reported FCF  = NOPAT + D&A - total capex       - delta NWC
economic FCF  = NOPAT + D&A - maintenance capex - delta NWC
```

Maintenance capex defaults to a conservative ensemble (depreciation anchor, growth-adjusted capex,
historical lower-band capex/revenue, asset-turnover proxy), selected as `max(median(estimates),
sector floor)` capped at reported capex. `--company-type hyperscaler|platform` defaults to economic
FCF unless `--valuation-mode reported`. Owner-earnings mode (normalized SBC + lease treatment) is
available for models where accounting FCF is a poor proxy. `roic_dcf` (the default) forecasts
operating economics — `growth = reinvestment_rate * ROIC`, `FCF = NOPAT * (1 - reinvestment_rate)` —
with a stable-ROIC terminal value. See [`docs/methodology.md`](docs/methodology.md) for the full
methodology and validation.

## Notes
This is a valuation research tool, **not investment advice**. Free data sources can have missing or
restated facts; review output before use. The "Phoenix Capital" branding in `site/` is a demo
research-site theme generated by the tool, not a real firm or an offer of services.

## License
© 2026 Pulkit Sharma. All rights reserved. This repository is published for review and
demonstration; it is not licensed for reuse, redistribution, or derivative works.
