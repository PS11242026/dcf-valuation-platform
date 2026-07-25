from __future__ import annotations


class TerminalValueError(ValueError):
    pass


def terminal_value(last_fcf: float, wacc: float, terminal_growth: float) -> float:
    spread = wacc - terminal_growth
    if spread <= 0.0025:
        raise TerminalValueError(
            f"Invalid terminal spread: WACC ({wacc:.4f}) must exceed terminal growth ({terminal_growth:.4f}) by at least 0.25%."
        )
    return last_fcf * (1.0 + terminal_growth) / spread
