"""Stable repo-root + reports paths, independent of the current working directory."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # src/dcf_project/paths.py -> repo root
CACHE = REPO_ROOT / ".cache"                      # git-ignored fetch caches (SEC, prices)
REPORTS = REPO_ROOT / "reports"
VALUATION = REPORTS / "valuation"
PORTFOLIO_BUILDS = REPORTS / "portfolio_builds"
MANAGED = REPORTS / "managed_portfolios"
REGISTRY = REPORTS / "registry"
PAPER = REPORTS / "paper"
SITE_CONTENT = REPO_ROOT / "site_content"         # research posts + site.json (authored)
SITE_OUT = REPO_ROOT / "site"                     # generated static site (deployable)
