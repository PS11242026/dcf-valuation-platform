"""Live India macro provider for the country bolt-on.

The US engine bakes US macro anchors into ``config.Settings`` (ERP 4.5%, tax 21%, US 10Y
Treasury risk-free, ~4% nominal GDP, 2.5% inflation). For an Indian valuation those are all
wrong: India runs a much higher cost of capital, a higher growth/inflation regime, and a
different statutory tax. This module assembles the India macro profile **live** with a
documented static fallback for every field, so the bolt-on degrades gracefully offline.

What flows where (the flagship engine_v3 reads all of these straight from ``Settings``):
- ``risk_free_override`` -> ``MarketClient.risk_free_rate`` (India 10Y G-Sec)
- ``equity_risk_premium`` + ``default_cost_of_debt`` -> ``market.build_wacc`` (internal CAPM)
- ``long_run_nominal_gdp`` + ``inflation_anchor`` + terminal band -> ``engine_v3.terminal.gk_terminal_growth``

Live sources (all key-free; each wrapped so a failure falls back, never raises):
- World Bank API: CPI inflation (``FP.CPI.TOTL.ZG``), real GDP growth (``NY.GDP.MKTP.KD.ZG``),
  lending rate as a cost-of-debt proxy (``FR.INR.LEND``).
- Damodaran country-premium workbook for the India total ERP.
- A best-effort India 10Y G-Sec yield (endpoint-fragile; falls back to the static anchor).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import requests

# --- Static fallbacks (decimal). Coarse long-run India anchors; only used when the matching
# live fetch is unavailable. Documented here so an offline run is explainable, not magic. ---
FALLBACK_GSEC_10Y = 0.068          # India 10Y G-Sec ~6.8% (2024-25 level)
FALLBACK_ERP = 0.080               # Damodaran India total ERP ~8.0% (mature ERP + India CRP, 2024-25)
FALLBACK_CPI = 0.050               # India CPI ~5% (RBI mid-band)
FALLBACK_REAL_GDP = 0.055          # long-run real GDP growth
FALLBACK_COST_OF_DEBT = 0.090      # India corporate borrowing ~9%
STATUTORY_TAX_RATE = 0.2517        # new-regime corporate tax (incl. surcharge + cess)
TERMINAL_GROWTH_FLOOR = 0.040
TERMINAL_GROWTH_CAP = 0.070

#: Sanity band for the derived nominal-GDP growth anchor (real + CPI), to keep one noisy
#: live reading from producing an absurd terminal anchor.
_NOMINAL_GDP_BAND = (0.060, 0.140)

_WB_URL = "https://api.worldbank.org/v2/country/IND/indicator/{code}?format=json&per_page=60"
_DAMODARAN_CTRYPREM_URL = "https://www.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx"


@dataclass(frozen=True)
class IndiaMacro:
    """Resolved India macro profile + per-field provenance (``live:<src>`` / ``fallback``)."""

    risk_free: float
    equity_risk_premium: float
    inflation_anchor: float
    long_run_nominal_gdp: float
    default_cost_of_debt: float
    default_tax_rate: float = STATUTORY_TAX_RATE
    terminal_growth_floor: float = TERMINAL_GROWTH_FLOOR
    terminal_growth_cap: float = TERMINAL_GROWTH_CAP
    default_terminal_growth: float = 0.05
    sources: dict[str, str] = field(default_factory=dict)

    def settings_overrides(self) -> dict[str, object]:
        """The keyword overrides to ``dataclasses.replace`` onto a US ``Settings``.

        ``wacc_source`` is deliberately NOT forced here: valueinvesting.io covers Indian
        listings under their ``<symbol>.ns`` page, so India honours the same provider-first
        WACC path as the US (``--wacc-source``, default ``auto``). The India risk-free + ERP
        below still drive the internal-CAPM fallback and the provider WACC's risk-free floor.
        """
        return {
            "country": "IN",
            "risk_free_override": self.risk_free,
            "equity_risk_premium": self.equity_risk_premium,
            "inflation_anchor": self.inflation_anchor,
            "long_run_nominal_gdp": self.long_run_nominal_gdp,
            "default_cost_of_debt": self.default_cost_of_debt,
            "default_tax_rate": self.default_tax_rate,
            "terminal_growth_floor": self.terminal_growth_floor,
            "terminal_growth_cap": self.terminal_growth_cap,
            "default_terminal_growth": self.default_terminal_growth,
        }


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _fetch_wb_indicator(code: str, timeout: int) -> float | None:
    """Most recent non-null annual value of a World Bank indicator, as a decimal.

    World Bank returns ``[metadata, [rows...]]`` newest-first with values in percent.
    """
    try:
        response = requests.get(_WB_URL.format(code=code), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        for row in rows:
            value = row.get("value")
            if value is not None:
                return float(value) / 100.0
    except Exception:
        return None
    return None


def _fetch_gsec_10y(timeout: int) -> float | None:
    """Best-effort India 10Y G-Sec yield (decimal). Endpoint-fragile -> often None offline."""
    try:
        response = requests.get(
            "https://www.worldgovernmentbonds.com/bond-historical-data/india/10-years/",
            headers={"User-Agent": "Mozilla/5.0 (dcf-project India macro)"},
            timeout=timeout,
        )
        response.raise_for_status()
        import re

        match = re.search(r"([0-9]\.[0-9]{2,3})\s*%", response.text)
        if match:
            return float(match.group(1)) / 100.0
    except Exception:
        return None
    return None


def _fetch_damodaran_india_erp(timeout: int) -> float | None:
    """India total equity-risk-premium from Damodaran's country-premium workbook (decimal)."""
    try:
        from io import BytesIO

        from openpyxl import load_workbook

        response = requests.get(_DAMODARAN_CTRYPREM_URL, timeout=timeout)
        response.raise_for_status()
        workbook = load_workbook(BytesIO(response.content), read_only=True, data_only=True)
        sheet = workbook.active
        header = [str(c.value).strip().lower() if c.value is not None else "" for c in next(sheet.iter_rows())]
        try:
            erp_col = next(i for i, h in enumerate(header) if "total equity risk premium" in h)
        except StopIteration:
            erp_col = next((i for i, h in enumerate(header) if "equity risk premium" in h), None)
        if erp_col is None:
            return None
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if row and str(row[0]).strip().lower() == "india":
                raw = row[erp_col]
                if raw is None:
                    return None
                value = float(raw)
                return value / 100.0 if value > 1.0 else value
    except Exception:
        return None
    return None


def live_india_macro(timeout: int = 20) -> IndiaMacro:
    """Assemble the India macro profile, preferring live reads and falling back per field."""
    sources: dict[str, str] = {}

    gsec = _fetch_gsec_10y(timeout)
    sources["risk_free"] = "live:worldgovernmentbonds" if gsec is not None else "fallback"
    gsec = gsec if gsec is not None else FALLBACK_GSEC_10Y

    erp = _fetch_damodaran_india_erp(timeout)
    sources["equity_risk_premium"] = "live:damodaran" if erp is not None else "fallback"
    erp = erp if erp is not None else FALLBACK_ERP

    cpi = _fetch_wb_indicator("FP.CPI.TOTL.ZG", timeout)
    sources["inflation_anchor"] = "live:worldbank" if cpi is not None else "fallback"
    cpi = cpi if cpi is not None else FALLBACK_CPI

    real_gdp = _fetch_wb_indicator("NY.GDP.MKTP.KD.ZG", timeout)
    sources["real_gdp"] = "live:worldbank" if real_gdp is not None else "fallback"
    real_gdp = real_gdp if real_gdp is not None else FALLBACK_REAL_GDP

    cost_of_debt = _fetch_wb_indicator("FR.INR.LEND", timeout)
    sources["default_cost_of_debt"] = "live:worldbank" if cost_of_debt is not None else "fallback"
    cost_of_debt = cost_of_debt if cost_of_debt is not None else FALLBACK_COST_OF_DEBT

    nominal_gdp = _clamp(real_gdp + cpi, *_NOMINAL_GDP_BAND)
    sources["long_run_nominal_gdp"] = "derived:real_gdp+cpi"
    sources["default_tax_rate"] = "statutory"

    # Terminal-growth band tied to the LIVE India macro rather than hardcoded, so the GK anchor
    # stays internally consistent with the (nominal) discount rate as rates move:
    #   ceiling = the 10Y G-Sec  -- Damodaran's stable-growth discipline (g <= risk-free, since
    #             the risk-free rate proxies the economy's long-run nominal growth in that currency).
    #             India's ~10-11% nominal GDP would otherwise let the GK-raw anchor run hot.
    #   floor   = CPI inflation  -- a going concern should at least hold its nominal value; kept
    #             strictly below the ceiling. Real terminal growth is then ceiling - floor >= 0.
    terminal_cap = _clamp(gsec, 0.05, 0.09)
    terminal_floor = min(cpi, terminal_cap - 0.005)
    sources["terminal_growth_band"] = "derived:cpi..gsec"

    return IndiaMacro(
        risk_free=float(gsec),
        equity_risk_premium=float(erp),
        inflation_anchor=float(cpi),
        long_run_nominal_gdp=float(nominal_gdp),
        default_cost_of_debt=float(cost_of_debt),
        terminal_growth_floor=float(terminal_floor),
        terminal_growth_cap=float(terminal_cap),
        default_terminal_growth=float(_clamp(0.5 * (terminal_floor + terminal_cap), terminal_floor, terminal_cap)),
        sources=sources,
    )
