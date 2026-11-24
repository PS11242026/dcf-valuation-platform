"""Stable repo-root + reports paths, independent of the current working directory."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # src/dcf_project/paths.py -> repo root
CACHE = Path(os.environ.get("DCF_CACHE_DIR") or REPO_ROOT / ".cache")  # git-ignored fetch caches (SEC, prices)
# DCF_REPORTS_DIR / DCF_SITE_CONTENT let a scheduled cloud job keep the managed-book state and the
# authored site content outside this repo (e.g. in the Pages repo it publishes to).
REPORTS = Path(os.environ.get("DCF_REPORTS_DIR") or REPO_ROOT / "reports")
VALUATION = REPORTS / "valuation"
PORTFOLIO_BUILDS = REPORTS / "portfolio_builds"
MANAGED = REPORTS / "managed_portfolios"
REGISTRY = REPORTS / "registry"
PAPER = REPORTS / "paper"
SITE_CONTENT = Path(os.environ.get("DCF_SITE_CONTENT") or REPO_ROOT / "site_content")  # posts + site.json
SITE_OUT = REPO_ROOT / "site"                     # generated static site (deployable)
