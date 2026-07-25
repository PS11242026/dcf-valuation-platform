"""PRODUCT 5 -- Static research site: the market-facing publication of the managed books.

Renders reports/managed_portfolios/*/{status,news,risk_stats,track}.json plus authored research
notes (site_content/research/*.md) into a deployable static site (site/). No server, no build
toolchain: every page embeds its data snapshot (works from file:// and static hosts alike) and,
when hosted, polls its own data/*.json to refresh without redeploying the HTML.
"""
