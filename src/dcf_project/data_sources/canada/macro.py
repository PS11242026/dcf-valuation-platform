"""Canada macro profile for the country bolt-on.

The US engine bakes US macro anchors into ``config.Settings`` (ERP 4.5%, tax 21%, US 10Y
risk-free). Canada differs on three axes that matter for an internal-CAPM DCF: a lower
government risk-free (Government of Canada 10Y), a comparable-but-distinct equity risk
premium (Canada is AAA, so its Damodaran country-risk premium is ~0 and the total ERP is
close to the mature-market ERP), and a higher combined federal+provincial statutory
corporate tax. This module assembles the Canadian profile as **documented, dated static
anchors** (parity with India's live profile in methodology — current anchors applied across
all PIT cutoffs, never historical-look-ahead provider WACC — without India's live-fetch
fragility), and exposes the same ``settings_overrides()`` contract India uses.

Sources (mid-2026, decimals):
- Government of Canada 10Y benchmark yield ~3.2% (Bank of Canada).
- Total ERP ~5.0%: Damodaran mature-market ERP + Canada country-risk premium (~0, AAA).
- Inflation anchor 2.0% (Bank of Canada target midpoint).
- Long-run nominal GDP ~3.7% (≈ real ~1.7% + 2.0% inflation; IMF/StatCan).
- Combined federal + provincial statutory corporate tax ~26.5%.
- Investment-grade corporate cost of debt ~5.0%.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- Documented Canadian anchors (decimal), mid-2026. ---
GOC_10Y = 0.032               # Government of Canada 10Y benchmark yield (Bank of Canada)
TOTAL_ERP = 0.050             # Damodaran mature ERP + Canada CRP (~0, AAA) -> ~5.0%
INFLATION_ANCHOR = 0.020      # Bank of Canada target midpoint
LONG_RUN_NOMINAL_GDP = 0.037  # ~real 1.7% + 2.0% inflation
COST_OF_DEBT = 0.050          # Canadian IG corporate borrowing ~5%
STATUTORY_TAX_RATE = 0.265    # combined federal + provincial corporate tax
TERMINAL_GROWTH_FLOOR = 0.015
TERMINAL_GROWTH_CAP = 0.045   # ~ long-run nominal GDP ceiling
DEFAULT_TERMINAL_GROWTH = 0.030


@dataclass(frozen=True)
class CanadaMacro:
    """Resolved Canada macro profile + per-field provenance (here: ``static:<src>``)."""

    risk_free: float = GOC_10Y
    equity_risk_premium: float = TOTAL_ERP
    inflation_anchor: float = INFLATION_ANCHOR
    long_run_nominal_gdp: float = LONG_RUN_NOMINAL_GDP
    default_cost_of_debt: float = COST_OF_DEBT
    default_tax_rate: float = STATUTORY_TAX_RATE
    terminal_growth_floor: float = TERMINAL_GROWTH_FLOOR
    terminal_growth_cap: float = TERMINAL_GROWTH_CAP
    default_terminal_growth: float = DEFAULT_TERMINAL_GROWTH
    sources: dict[str, str] = field(default_factory=lambda: {
        "risk_free": "static:bank_of_canada_goc10y",
        "equity_risk_premium": "static:damodaran_mature+ca_crp",
        "default_tax_rate": "static:combined_federal_provincial",
        "long_run_nominal_gdp": "static:imf_statcan",
    })

    def settings_overrides(self, *, currency_is_cad: bool = True) -> dict[str, object]:
        """Keyword overrides to ``dataclasses.replace`` onto a US ``Settings``.

        ``currency_is_cad`` controls whether the Canadian risk-free / ERP / terminal band are
        injected. CAD-reporting names take the full Canadian profile (CAD cash flows discounted
        at a CAD WACC). USD-reporting names (global operators that report in USD) keep the US
        risk-free + ERP + terminal band (USD cash flows discounted at a USD WACC) and only adopt
        the Canadian statutory tax fallback -- so each name is valued in its own currency.
        """
        common: dict[str, object] = {
            "country": "CA",
            "default_cost_of_debt": self.default_cost_of_debt,
            "default_tax_rate": self.default_tax_rate,
        }
        if not currency_is_cad:
            return common  # USD reporter: US rf/ERP/terminal, Canadian tax fallback only
        common.update({
            "risk_free_override": self.risk_free,
            "equity_risk_premium": self.equity_risk_premium,
            "inflation_anchor": self.inflation_anchor,
            "long_run_nominal_gdp": self.long_run_nominal_gdp,
            "terminal_growth_floor": self.terminal_growth_floor,
            "terminal_growth_cap": self.terminal_growth_cap,
            "default_terminal_growth": self.default_terminal_growth,
        })
        return common


def canada_macro() -> CanadaMacro:
    """The Canadian macro profile (documented static anchors; see module docstring)."""
    return CanadaMacro()
