"""Output-coherence guard for engine_v3 (Research_V4, Modification 1).

A valuation engine should refuse to print a headline number it cannot stand
behind. This module is a *pure robustness check* applied AFTER the deterministic
valuation: it never changes a healthy intrinsic, it only classifies an already
numerically-broken result as ``incomplete`` so downstream surfaces can withhold
it instead of emitting nonsense (a $234M/share value from a corrupt share count,
a negative per-share value for a solvent going concern, or a terminal value that
dwarfs the business).

It is deliberately NOT an economic lever. The thresholds are intentionally wide
so that only *obviously invalid* output trips -- genuine conservatism (e.g. a
positive-but-tiny intrinsic from SBC/low-GAAP-NOPAT names like SNPS) is retained
as a healthy valuation, never flagged. Closing a gap to consensus or market price
is explicitly NOT a goal here. See ``reports/research_v4/report/v4_rootcause.md``
(§B) and ``v4_recommended_changes.md`` (Modification 1) for the diagnostic that
motivated each check and the four names each one catches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- Thresholds (wide by design; documented economic/numeric rationale each) ---

#: Reconstructed diluted shares may diverge from the market provider's share
#: count only within this band. Outside it, one source is corrupt (MCD: a single
#: bad SEC reading gives 713 vs ~713 *million*). A factor of 3 leaves real
#: split/issuance staleness untouched while catching order-of-magnitude breaks.
SHARE_RATIO_MIN: float = 1.0 / 3.0
SHARE_RATIO_MAX: float = 3.0

#: A terminal value is a "blow-up" only when it BOTH dominates enterprise value
#: (PV(TV)/EV above this ceiling) AND implies an absurd EV/revenue multiple. Both
#: must hold, so a legitimately terminal-weighted name (software pvtv≈0.74) or a
#: legitimately high-multiple name (NVDA EV/sales≈6.7) is never flagged; GIS
#: ($866B terminal, pvtv 0.97, EV/sales 21) is.
TERMINAL_PVTV_CEILING: float = 0.95
TERMINAL_EV_SALES_CEILING: float = 15.0

#: Backstop for any remaining "obviously invalid" output: an intrinsic this many
#: times the market price is not a deep-value opinion, it is a broken number
#: (MCD ~822,000x; GIS ~21x). The deepest genuine v3 upside in the V4 panel is
#: ~1.6x, so 10x cannot fire on a healthy name.
IV_PRICE_CEILING: float = 10.0


@dataclass(frozen=True)
class CoherenceVerdict:
    """Result of the output-coherence guard for one valuation."""

    status: str  # "ok" | "incomplete"
    flags: list[str] = field(default_factory=list)
    checks: dict[str, float | None] = field(default_factory=dict)

    @property
    def is_incomplete(self) -> bool:
        return self.status == "incomplete"


def assess_coherence(
    *,
    intrinsic: float,
    equity_value: float,
    enterprise_value: float,
    recon_shares: float | None,
    market_shares: float | None,
    revenue_ttm: float | None,
    pv_terminal_share_of_ev: float | None,
    market_price: float | None,
) -> CoherenceVerdict:
    """Classify a deterministic valuation as ``ok`` or ``incomplete``.

    Pure function: inspects only the already-computed numbers, mutates nothing.
    A result is ``incomplete`` if ANY check fires; ``flags`` records which causes
    and ``checks`` records the diagnostic ratios for the report/audit.
    """
    flags: list[str] = []
    share_ratio: float | None = None
    if recon_shares and market_shares and recon_shares > 0 and market_shares > 0:
        share_ratio = recon_shares / market_shares
    ev_sales: float | None = None
    if revenue_ttm and revenue_ttm > 0:
        ev_sales = enterprise_value / revenue_ttm
    iv_price: float | None = None
    if market_price and market_price > 0:
        iv_price = intrinsic / market_price

    # 1. Negative / zero equity: EV below net debt for a sub-cost-of-capital name
    #    yields a per-share value that is not a meaningful valuation (EL, MRVL).
    if equity_value <= 0 or intrinsic <= 0:
        flags.append("negative_equity")

    # 2. Corrupt share count: reconstructed vs market provider diverge by an
    #    order of magnitude (MCD).
    if share_ratio is not None and (share_ratio < SHARE_RATIO_MIN or share_ratio > SHARE_RATIO_MAX):
        flags.append("corrupt_shares")

    # 3. Terminal blow-up: terminal value both dominates EV and implies an absurd
    #    EV/revenue multiple (GIS).
    if (
        pv_terminal_share_of_ev is not None
        and pv_terminal_share_of_ev > TERMINAL_PVTV_CEILING
        and ev_sales is not None
        and ev_sales > TERMINAL_EV_SALES_CEILING
    ):
        flags.append("terminal_blowup")

    # 4. Backstop: any other obviously invalid intrinsic vs the traded price.
    if iv_price is not None and iv_price > IV_PRICE_CEILING:
        flags.append("implausible_vs_price")

    status = "incomplete" if flags else "ok"
    checks = {
        "share_ratio": share_ratio,
        "ev_to_sales": ev_sales,
        "pv_terminal_share_of_ev": pv_terminal_share_of_ev,
        "iv_to_price": iv_price,
    }
    return CoherenceVerdict(status=status, flags=flags, checks=checks)
