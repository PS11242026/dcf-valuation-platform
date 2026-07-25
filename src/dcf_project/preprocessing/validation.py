from __future__ import annotations

import pandas as pd


def validate_quarterly_history(history: pd.DataFrame, minimum_quarters: int = 20) -> list[str]:
    warnings: list[str] = []
    if len(history) < minimum_quarters:
        warnings.append(f"Quarterly history has {len(history)} periods; minimum target is {minimum_quarters}.")
    missing_required = [
        column
        for column in ["revenue", "ebit", "tax_expense", "da", "capex", "fcf"]
        if column not in history or history[column].isna().any()
    ]
    if missing_required:
        warnings.append(f"Missing or incomplete required fields: {', '.join(missing_required)}.")
    return warnings
