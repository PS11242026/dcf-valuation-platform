"""Persona helpers -- the three institutional report types share the kit but differ in framing."""
from __future__ import annotations

from .model import ReportModel

PERSONAS = {
    "initiation": "Sell-side equity research initiation report (analyst / PM audience).",
    "ic": "Investment-committee portfolio proposal package (IC audience).",
    "pm": "Portfolio-management operational dashboard (PM / ops audience).",
}


def new_report(title: str, subtitle: str, persona: str, **meta) -> ReportModel:
    assert persona in PERSONAS, f"unknown persona {persona}"
    return ReportModel(title=title, subtitle=subtitle, persona=persona, meta=dict(meta))
