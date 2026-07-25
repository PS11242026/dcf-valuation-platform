"""Single source of truth for the corrected-panel validation verdict (2026-06-11).

Every product surface (the `valuation compare` workbook, the fade Excel, the live
dashboard, and the CFA/FAJ manuscript) must tell the SAME scoped story, derived from the
SAME frozen ledgers rather than hand-copied numbers. This module loads those ledgers and
exposes per-family verdicts plus the one-line headline claim.

The scoped claim (Phase 3 decision, see reports/research_v2/research_journal.md
2026-06-11T11:30 and the engine-v3-program memory):

  v3 is a STRICTLY BETTER FADE than v1 -- it out-ranks v1 paired, survives the post-2021
  out-of-sample holdout, and wins across regimes -- but it is NOT a broad unique alpha
  source: the Book/Market value factor spans its broad edge. v3's genuine incremental-to-
  value ranking information is confined to consumer staples and consumer discretionary;
  in semiconductors the discount is wrong-signed net of Book/Market. v1 remains the default
  research / live-book ranking engine; v3 is the opt-in, most-explainable engine with a
  SCOPED consumer-family alpha claim.

Ledgers consumed (all under reports/research_v3/, produced by the corrected ladder):
  * tri_bm_residual.json        -- v3/v1 raw + B/M-residualised discount IC, overall + family
  * tri_paired_significance.json -- paired v3-v1 IC diff (NW + block bootstrap), overall + family
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dcf_project.paths import REPORTS  # repo reports/ dir

_R3 = REPORTS / "research_v3"
_BM = _R3 / "tri_bm_residual.json"
_PAIRED = _R3 / "tri_paired_significance.json"
_UNIVERSE = REPORTS / "_universe_tickers.json"

T_SIG = 2.0  # |NW t| threshold for "significant" (matches the ladder's convention)

FAMILY_LABEL = {
    "consumer_staples": "Consumer staples",
    "consumer_discretionary": "Consumer discretionary",
    "internet_platforms_marketplaces": "Internet platforms",
    "semiconductors": "Semiconductors",
    "semiconductor_equipment": "Semiconductor equipment",
    "software_saas_infra": "Software / SaaS",
    "industrials_assetlight_services": "Industrials (asset-light)",
}

# Tier -> how the triangulation should be read for a name in that family.
TIER_GUIDANCE = {
    "value_orthogonal": "the fade discount carries ranking information beyond the Book/Market "
                        "value factor in this family -- treat the v3 read as a genuine (scoped) signal.",
    "value_proxy": "the fade discount ranks here, but the edge is explained by the Book/Market "
                   "value factor -- read the triangulation as coherence, not incremental alpha.",
    "wrong_signed": "the discount is wrong-signed against forward returns net of Book/Market in "
                    "this family -- do NOT read the triangulation as a ranking signal; use it only "
                    "as an economic-coherence check.",
    "insufficient": "evidence is inconclusive in this family (small cross-section or not "
                    "significant) -- read the triangulation as coherence, not a ranking signal.",
}

TIER_BADGE = {
    "value_orthogonal": "Value-orthogonal signal (scoped alpha)",
    "value_proxy": "Value-factor proxy (coherence only)",
    "wrong_signed": "Wrong-signed net of value (coherence only)",
    "insufficient": "Inconclusive (coherence only)",
}


@dataclass(frozen=True)
class FamilyVerdict:
    family: str
    label: str
    paired_v3_minus_v1_3y: Optional[float]   # v3 out-ranks v1 by this IC at 3y
    paired_nw_t_3y: Optional[float]
    bm_resid_v3_3y: Optional[float]          # v3 discount IC net of Book/Market, 3y
    bm_resid_v3_t_3y: Optional[float]
    bm_resid_v1_3y: Optional[float]
    tier: str

    @property
    def badge(self) -> str:
        return TIER_BADGE[self.tier]

    @property
    def guidance(self) -> str:
        return TIER_GUIDANCE[self.tier]

    @property
    def v3_beats_v1(self) -> Optional[bool]:
        d, t = self.paired_v3_minus_v1_3y, self.paired_nw_t_3y
        if d is None or t is None:
            return None
        return d > 0 and t >= T_SIG


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _classify(bm_resid_v3: Optional[float], bm_resid_t: Optional[float]) -> str:
    """Tier a family from its v3 B/M-residualised 3y IC and NW t-stat."""
    if bm_resid_v3 is None or bm_resid_t is None:
        return "insufficient"
    if bm_resid_t <= -T_SIG and bm_resid_v3 < 0:
        return "wrong_signed"
    if bm_resid_t >= T_SIG and bm_resid_v3 > 0:
        return "value_orthogonal"
    return "value_proxy"  # ranks (or not) but not separable from Book/Market


@lru_cache(maxsize=1)
def _verdicts() -> dict[str, FamilyVerdict]:
    bm = _load(_BM).get("by_family", {})
    paired = _load(_PAIRED)
    out: dict[str, FamilyVerdict] = {}
    fams = set(bm) | {k for k in paired if k not in ("overall", "join")}
    for fam in fams:
        b3 = (bm.get(fam, {}).get("3y", {}) or {})
        v3b = b3.get("v3", {}).get("bm_resid_ic", {}) if b3 else {}
        v1b = b3.get("v1", {}).get("bm_resid_ic", {}) if b3 else {}
        p3 = (paired.get(fam, {}).get("horizons", {}) or {}).get("3y", {})
        bm_resid_v3 = v3b.get("mean")
        bm_resid_t = v3b.get("t_stat")
        out[fam] = FamilyVerdict(
            family=fam,
            label=FAMILY_LABEL.get(fam, fam),
            paired_v3_minus_v1_3y=p3.get("mean_ic_diff"),
            paired_nw_t_3y=p3.get("nw_tstat"),
            bm_resid_v3_3y=bm_resid_v3,
            bm_resid_v3_t_3y=bm_resid_t,
            bm_resid_v1_3y=v1b.get("mean"),
            tier=_classify(bm_resid_v3, bm_resid_t),
        )
    return out


def available() -> bool:
    """True only when the corrected-ladder ledgers exist (surfaces degrade gracefully)."""
    return _BM.exists() and _PAIRED.exists()


def family_verdict(family: str) -> Optional[FamilyVerdict]:
    return _verdicts().get(family)


@lru_cache(maxsize=1)
def _family_of() -> dict[str, str]:
    return _load(_UNIVERSE).get("family_of", {})


def family_of_ticker(ticker: str) -> Optional[str]:
    return _family_of().get(ticker.upper())


def verdict_for_ticker(ticker: str) -> Optional[FamilyVerdict]:
    fam = family_of_ticker(ticker)
    return family_verdict(fam) if fam else None


def headline_claim() -> str:
    """One-paragraph scoped claim, with the two surviving families named from the ledger."""
    survivors = [v.label for v in _verdicts().values() if v.tier == "value_orthogonal"]
    survivors_s = " and ".join(survivors) if survivors else "select stable-cash-flow families"
    return (
        "The Fundamentals Fade (v3) engine out-ranks the frozen v1 baseline on the corrected "
        "point-in-time panel (paired, out-of-sample, and across regimes), but the Book/Market "
        "value factor spans its broad ranking edge. v3 carries genuine incremental-to-value "
        f"information only in {survivors_s}; the v1 engine remains the default ranking / live-book "
        "model and v3 is the opt-in, most-explainable engine with a scoped alpha claim."
    )


def overall_paired() -> dict:
    """Overall paired v3-v1 diff block ({'3y':..,'5y':..}) for headline exhibits."""
    return _load(_PAIRED).get("overall", {}).get("horizons", {})
