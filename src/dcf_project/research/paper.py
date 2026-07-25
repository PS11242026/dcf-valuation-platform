"""Build the CFA research package.

Thin wrapper over the assemblers under scripts/paper/, which pull numbers from the frozen
reports/*.json + reports/{research_v2,research_v3,portfolio}/*.{json,csv} and write the
master report + tables + figures. No operational functionality lives here.

Two versions:
  v2 (default) -- the corrected manuscript -> reports/paper_v2/ (scripts.paper.build_paper_v2):
      post-split-correction story + the Fundamentals Fade engine as the platform.
  v1           -- the legacy pre-correction package -> reports/paper/ (scripts.paper.build_paper):
      preserved verbatim; cited by the v2 manuscript as the superseded artifact.
"""
from __future__ import annotations

import subprocess
import sys

from dcf_project.paths import PAPER, REPO_ROOT

_MODULES = {"v2": "scripts.paper.build_paper_v2", "v1": "scripts.paper.build_paper"}


def build_paper(version: str = "v2") -> None:
    module = _MODULES.get(version)
    if module is None:
        raise SystemExit(f"unknown paper version '{version}' (choose v1 or v2)")
    print(f"Building research package ({version}) via {module} ...")
    result = subprocess.run([sys.executable, "-m", module], cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"research build-paper failed (exit {result.returncode})")
    dest = (PAPER.parent / "paper_v2") if version == "v2" else PAPER
    print(f"research package refreshed -> {dest}")
