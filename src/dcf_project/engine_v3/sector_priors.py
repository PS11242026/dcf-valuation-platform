"""Static per-sector economic priors for engine_v3.

Documented constants, PIT-safe by construction: these are coarse LONG-RUN U.S. sector
aggregates of the kind tabulated in standard valuation references (Damodaran's sector
margin/ROIC datasets; Koller et al., *Valuation*, on the persistence of sector excess
returns) and stable common knowledge well before the 2015+ backtest window. They are
PRIORS, not estimates: every company series is shrunk toward them with data-driven
weights, and the G2 validation gate ablates them ±25% to confirm conclusions do not
hinge on the exact values.

Fields per sector:
- ``phi``: annual persistence prior of EXCESS ROIC (ROIC − WACC). High-moat sectors
  (software, staples brands, health care franchises) hold excess returns longer;
  commodity/cyclical sectors (energy, materials) mean-revert fast.
- ``margin_anchor``: long-run NOPAT margin the sector reverts toward.
- ``roic_anchor``: long-run annual ROIC level (diagnostic anchor for sanity checks).
- ``anchor_weight``: blend weight of the sector margin anchor against the company's own
  normalized margin when forming the mean-reversion target.
- ``growth_half_life_years`` / ``margin_half_life_years``: prior fade half-lives (years
  for half the gap to the long-run anchor to close) that the per-company AR(1) estimates
  are shrunk toward. Margins are stickier than growth (longer half-lives); moaty/secular
  sectors fade slower than cyclicals (energy/materials revert in ~1.5y, staples ~6y).

Sectors are the 11 canonical labels from ``engine_v2.sector_map.SECTORS``.
"""
from __future__ import annotations

from dataclasses import dataclass

from dcf_project.engine_v2.sector_map import sector_for_ticker  # noqa: F401  (re-export for pipeline use)


@dataclass(frozen=True)
class SectorPrior:
    phi: float
    margin_anchor: float
    roic_anchor: float
    anchor_weight: float = 0.25
    growth_half_life_years: float = 3.0
    margin_half_life_years: float = 4.0


SECTOR_PRIORS: dict[str, SectorPrior] = {
    "technology":             SectorPrior(phi=0.90, margin_anchor=0.18, roic_anchor=0.18, growth_half_life_years=4.0, margin_half_life_years=5.0),
    "communication_services": SectorPrior(phi=0.85, margin_anchor=0.13, roic_anchor=0.10, growth_half_life_years=3.5, margin_half_life_years=4.5),
    "consumer_discretionary": SectorPrior(phi=0.75, margin_anchor=0.07, roic_anchor=0.10, growth_half_life_years=2.5, margin_half_life_years=3.5),
    "consumer_staples":       SectorPrior(phi=0.85, margin_anchor=0.10, roic_anchor=0.12, growth_half_life_years=4.0, margin_half_life_years=6.0),
    "health_care":            SectorPrior(phi=0.85, margin_anchor=0.12, roic_anchor=0.12, growth_half_life_years=4.0, margin_half_life_years=5.5),
    "industrials":            SectorPrior(phi=0.75, margin_anchor=0.09, roic_anchor=0.10, growth_half_life_years=3.0, margin_half_life_years=4.0),
    "financials":             SectorPrior(phi=0.70, margin_anchor=0.15, roic_anchor=0.08, growth_half_life_years=3.0, margin_half_life_years=4.0),
    "energy":                 SectorPrior(phi=0.55, margin_anchor=0.08, roic_anchor=0.07, growth_half_life_years=1.5, margin_half_life_years=2.5),
    "materials":              SectorPrior(phi=0.60, margin_anchor=0.08, roic_anchor=0.08, growth_half_life_years=1.5, margin_half_life_years=2.5),
    "utilities":              SectorPrior(phi=0.80, margin_anchor=0.13, roic_anchor=0.06, growth_half_life_years=3.5, margin_half_life_years=5.0),
    "real_estate":            SectorPrior(phi=0.75, margin_anchor=0.25, roic_anchor=0.05, growth_half_life_years=3.0, margin_half_life_years=4.5),
}

#: India per-sector priors (country bolt-on). Same canonical labels, but the long-run
#: aggregates differ materially from the US -- this is the "same sector behaves differently
#: per country" surface. Coarse India aggregates of the kind Damodaran tabulates for emerging
#: markets / India: FMCG (consumer_staples) and IT-services (technology) run structurally
#: HIGHER ROIC and stickier excess returns than their US peers (brand + offshore-cost moats);
#: financials carry NIM-style margins; cyclicals (energy/materials) mean-revert just as fast.
#: Seeds, not estimates: every company series is shrunk toward them and they are overridable.
#: Any sector absent here falls back to the US prior for that sector (see ``prior_for``).
IN_SECTOR_PRIORS: dict[str, SectorPrior] = {
    "technology":             SectorPrior(phi=0.88, margin_anchor=0.22, roic_anchor=0.30, growth_half_life_years=4.5, margin_half_life_years=5.5),
    "communication_services": SectorPrior(phi=0.75, margin_anchor=0.16, roic_anchor=0.10, growth_half_life_years=3.5, margin_half_life_years=4.5),
    "consumer_discretionary": SectorPrior(phi=0.78, margin_anchor=0.09, roic_anchor=0.14, growth_half_life_years=3.0, margin_half_life_years=4.0),
    "consumer_staples":       SectorPrior(phi=0.88, margin_anchor=0.16, roic_anchor=0.30, growth_half_life_years=4.5, margin_half_life_years=6.0),
    "health_care":            SectorPrior(phi=0.82, margin_anchor=0.16, roic_anchor=0.16, growth_half_life_years=4.0, margin_half_life_years=5.0),
    "industrials":            SectorPrior(phi=0.74, margin_anchor=0.10, roic_anchor=0.13, growth_half_life_years=3.0, margin_half_life_years=4.0),
    "financials":             SectorPrior(phi=0.72, margin_anchor=0.20, roic_anchor=0.13, growth_half_life_years=3.0, margin_half_life_years=4.0),
    "energy":                 SectorPrior(phi=0.55, margin_anchor=0.08, roic_anchor=0.10, growth_half_life_years=1.5, margin_half_life_years=2.5),
    "materials":              SectorPrior(phi=0.55, margin_anchor=0.10, roic_anchor=0.11, growth_half_life_years=1.5, margin_half_life_years=2.5),
    "utilities":              SectorPrior(phi=0.80, margin_anchor=0.16, roic_anchor=0.09, growth_half_life_years=3.5, margin_half_life_years=5.0),
    "real_estate":            SectorPrior(phi=0.70, margin_anchor=0.22, roic_anchor=0.10, growth_half_life_years=3.0, margin_half_life_years=4.5),
}

#: Country -> per-sector prior table. "US" is the frozen default; "IN" is the bolt-on.
# Canada is a developed market with US-like sector economics, so it reuses the US priors
# (this is byte-identical to the unregistered-country fallback in ``prior_for`` -- explicit
# here so the Canada bolt-on's prior source is documented, not implicit).
_PRIORS_BY_COUNTRY: dict[str, dict[str, SectorPrior]] = {
    "US": SECTOR_PRIORS, "IN": IN_SECTOR_PRIORS, "CA": SECTOR_PRIORS,
}

#: Used when the ticker cannot be mapped to a sector (sector_for_ticker -> None).
DEFAULT_PRIOR = SectorPrior(phi=0.75, margin_anchor=0.10, roic_anchor=0.10)


def prior_for_sector(sector: str | None) -> SectorPrior:
    if sector is None:
        return DEFAULT_PRIOR
    return SECTOR_PRIORS.get(sector, DEFAULT_PRIOR)


def prior_for(country: str, sector: str | None) -> SectorPrior:
    """Country-aware sector prior. India falls back to the US prior for any sector it omits.

    ``country="US"`` is byte-identical to :func:`prior_for_sector` (same table, same fallback),
    so the frozen US path is unchanged.
    """
    table = _PRIORS_BY_COUNTRY.get(country, SECTOR_PRIORS)
    if sector is None:
        return DEFAULT_PRIOR
    return table.get(sector) or SECTOR_PRIORS.get(sector, DEFAULT_PRIOR)
