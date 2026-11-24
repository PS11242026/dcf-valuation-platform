---
title: Phoenix DCF API
emoji: 📈
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: On-demand Kalman-filtered state-space DCF for any US ticker
---

# DCF Valuation Platform API

Backend for the **Analyze** page of [Phoenix Capital](https://ps11242026.github.io/phoenix-capital/).
Type a ticker on the site; this Space runs the canonical Kalman-filtered, state-space DCF from the
open-source [DCF Valuation Platform](https://github.com/PS11242026/dcf-valuation-platform) and
returns the full report (plus the Excel workbook and PDF).

- `GET /api/analyze/{ticker}`: starts or polls a job (`queued` / `running` / `done` / `error`)
- `GET /api/files/{ticker}/canonical_dcf.xlsx` and `/api/files/{ticker}/{ticker}.pdf`
- `GET /api/docs`: interactive docs

Free public data only (SEC EDGAR, Yahoo Finance, US Treasury). Research and education, not
investment advice.
