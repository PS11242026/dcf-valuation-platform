from __future__ import annotations

from dcf_project.contracts import FinancialPeriod


def quarterly_periods(periods: list[FinancialPeriod]) -> list[FinancialPeriod]:
    """Return periods in report-date order.

    SEC CompanyFacts already supplies period-ended facts. This hook keeps cadence
    handling isolated for future annual-to-quarterly conversion.
    """
    return sorted(periods, key=lambda item: item.report_date)

