"""Builder tests for the corrected research manuscript (scripts/paper/build_paper_v2.py).

Output is redirected to tmp_path; the builder still reads the real frozen artifacts under
reports/research_v2 + research_v3 (absolute paths). Structural assertions (section headers,
the hardcoded gate ledger) hold regardless of which data files are present; numeric
assertions are guarded on the source JSON existing.
"""
from __future__ import annotations

import pandas as pd

from scripts.paper import build_paper_v2, figures_v2


def test_master_manuscript_structure(tmp_path):
    md = build_paper_v2.build_master_v2(tmp_path)
    assert md.exists()
    text = md.read_text(encoding="utf-8")
    for header in ("# Economic Coherence Is Not Broad Predictive Power",
                   "## 3. Valuation Architecture", "## 5. Results I", "## 6. Results II",
                   "## 7. Results III",
                   "## 7A. Results IV -- The Kill, Re-examined",  # corrected-ladder reversal
                   "### Appendix B -- v3 Gate Ledger"):
        assert header in text, header
    # the honest engine verdicts are in the prose regardless of data: the original kill, its
    # supersession, the Book/Market boundary, and the scoped consumer-family survival.
    assert "KILL" in text and "consumer staples" in text.lower()
    assert "margin-seasonality defect" in text and "Book/Market" in text
    assert "scoped" in text.lower()


def test_master_cites_corrected_numbers_when_available(tmp_path):
    md = build_paper_v2.build_master_v2(tmp_path)
    text = md.read_text(encoding="utf-8")
    cv = build_paper_v2._j(build_paper_v2.R2 / "correction_validation.json")
    if cv:  # corrected staples 3y IC must appear verbatim in Section 6
        staples = cv["by_family_3y"]["consumer_staples"]["full"]
        assert f"{staples:+.3f}" in text


def test_gate_ledger_table(tmp_path):
    made = build_paper_v2.build_tables_v2(tmp_path)
    assert "v3_gate_ledger.csv" in made
    ledger = pd.read_csv(tmp_path / "tables" / "v3_gate_ledger.csv")
    assert len(ledger) == 8  # G0..G7
    verdicts = dict(zip(ledger["gate"], ledger["verdict"]))
    # Corrected ladder (2026-06-11): G2 killed then superseded by the corrected panel; G3-G5 run.
    assert verdicts["G2"] == "KILL -> SUPERSEDED"
    assert verdicts["G3"] == "PASS"
    assert verdicts["G4"].startswith("PASS")          # PASS (scoped)
    assert "kappa" in verdicts["G5"].lower()          # sector-state tilt held off
    assert verdicts["G6"] == "KILL"                    # MC calibration kill stands
    # no gate is left unrun any longer
    assert not any("NOT RUN" in str(v) for v in ledger["verdict"])


def test_figures_generate_smoke(tmp_path):
    made = figures_v2.generate(tmp_path)
    assert isinstance(made, list)
    if (build_paper_v2.R2 / "correction_validation.json").exists():
        assert "correction_decomposition.png" in made
        assert (tmp_path / "figures" / "correction_decomposition.png").exists()
