from __future__ import annotations


def present_value(cash_flow: float, discount_rate: float, period: int) -> float:
    if period <= 0:
        raise ValueError("period must be positive")
    return cash_flow / ((1.0 + discount_rate) ** period)


def present_value_series(cash_flows: list[float], discount_rate: float) -> float:
    return sum(present_value(cf, discount_rate, idx + 1) for idx, cf in enumerate(cash_flows))

