from __future__ import annotations

from dcf_project.contracts import FinancialPeriod


def require_sorted_periods(periods: list[FinancialPeriod]) -> list[FinancialPeriod]:
    return sorted(periods, key=lambda item: item.report_date)

