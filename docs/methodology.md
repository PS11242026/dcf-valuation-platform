# DCF Methodology

## Purpose

This model is a single-company intrinsic valuation engine. It is designed for
portfolio research, not price targeting. The core output is a range of
discounted operating cash-flow values, with diagnostics that make the accounting
and economic assumptions visible.

## Architecture

The engine ingests SEC CompanyFacts and market data, normalizes quarterly
financial periods, builds operating features, forecasts a 20-quarter explicit
cash-flow path, and applies a terminal value. Enterprise value is bridged to
equity value using cash, debt, and normalized lease obligations when applicable.

The model supports four valuation bases:

- reported FCF: `NOPAT + D&A - total capex - delta NWC`
- economic FCF: `NOPAT + D&A - maintenance capex - delta NWC`
- owner earnings: `NOPAT + D&A + SBC addback - maintenance capex - delta NWC`
- ROIC DCF: derives FCF from operating return and reinvestment economics

## ROIC Framework

`roic_dcf` is the default primary framework. Instead of forecasting free cash
flow directly, it forecasts operating economics:

- revenue growth
- operating margin
- reinvestment rate
- ROIC
- competitive fade

The core identity is:

```text
growth = reinvestment rate * ROIC
FCF = NOPAT * (1 - reinvestment rate)
```

Invested capital is normalized from working capital, net PPE, intangibles, and
capitalized leases where available. If accounting invested-capital fields are
missing or unusable, the model falls back to a company-type capital-turnover
proxy. Excess cash is excluded from the operating capital base.

The explicit 5-year period models state-space operating economics. Years 6-15
then use a competitive fade schedule. Durable compounders, platforms, and
consumer franchises fade more slowly than industrial or telecom businesses.
Terminal value is framed from stable ROIC and stable reinvestment:

```text
terminal growth = stable reinvestment * stable ROIC
terminal FCF = NOPAT * (1 - stable reinvestment)
TV = terminal FCF * (1 + g) / (WACC - g)
```

This avoids forcing long-duration compounders through abrupt terminal FCF
compression disconnected from their reinvestment runway.

## Maintenance Capex

Maintenance capex is estimated with a conservative ensemble:

- depreciation anchor
- growth-adjusted capex
- 25th to 40th percentile capex/revenue lower band
- asset-turnover sustaining reinvestment proxy

The default estimate is `max(median(estimates), sector floor)`, capped at
reported capex. This avoids using a low estimate to mechanically inflate
economic FCF.

## SBC

Stock-based compensation is handled through an economic haircut. Ignoring SBC
adds back the full non-cash charge. Full-expense treatment adds back none of it.
The default partial method adds back only the non-haircut portion. Company types
with material dilution risk use higher haircuts.

## Leases

Operating lease liabilities are treated as economic obligations when available.
For lease-heavy models such as marketplace hybrids, lease liabilities are added
to debt in the enterprise-to-equity bridge.

## Policy Engine

Company type controls default valuation mode and adjustments. Consumer and
industrial businesses generally use reported FCF. Hyperscalers and platforms
use economic FCF. Marketplace hybrids use owner earnings, lease normalization,
and lower reliance on naive accounting FCF.

## Reverse DCF

The reverse DCF module solves market-implied explicit FCF growth, terminal
growth, maintenance-capex delta, and margin assumptions from the current market
price. These are cross-checks for expectation realism, not assertions that the
market is correct.

## Confidence Scoring

The valuation confidence score penalizes weak data coverage, forecast clamps,
fallback models, warning flags, unstable forecast collapse, and business-model
complexity. Stable staples can score high; banks, insurers, marketplace hybrids,
and pre-profit or highly acquisitive businesses should usually score lower.

## Validation

Validation uses unit tests, deterministic baselines, Monte Carlo ranges,
sensitivity surfaces, reverse DCF checks, and walk-forward forecast consistency.
The benchmark goal is institutional plausibility, not matching market prices.

## Limitations

The engine will still struggle with banks, insurers, pre-profit hypergrowth,
highly acquisitive rollups, and companies where segment economics dominate
consolidated accounting. Segment-aware valuation hooks are intentionally left
extensible for manual AWS/Retail/Advertising, Search/Cloud/YouTube, or
Ads/Reality Labs cases.
