"""Unified command-line interface for the four products.

    python -m dcf_project valuation generic GOOGL
    python -m dcf_project valuation canonical GOOGL --forecast-mode consensus_blend --wacc-source valueinvesting
    python -m dcf_project valuation screen basket.csv
    python -m dcf_project portfolio construct --capital 1000000 --signal discount_x_conf_top10
    python -m dcf_project portfolio list
    python -m dcf_project portfolio promote portfolio_2026Q3
    python -m dcf_project portfolio manage update   --book conf_top10
    python -m dcf_project portfolio manage preview   --book conf_top10
    python -m dcf_project portfolio manage rebalance --book conf_top10
    python -m dcf_project research build-paper
    python -m dcf_project export pdf GOOGL
    python -m dcf_project export ppt conf_top10

Excel workbooks are the primary product; PDF/PPT are on-demand secondary exports of the same
ReportModel. Command bodies lazy-import their product module so `--help` stays fast and the CLI
loads even while individual products are still being built.
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    help="DCF research suite: valuation, portfolio construction/management, research, export.",
    no_args_is_help=True,
    add_completion=False,
)
valuation_app = typer.Typer(help="Product 1 -- company -> DCF valuation -> Excel.", no_args_is_help=True)
portfolio_app = typer.Typer(help="Products 2-3 -- construct (proposal) / manage (live) portfolios.", no_args_is_help=True)
manage_app = typer.Typer(help="Manage an existing (promoted) portfolio.", no_args_is_help=True)
research_app = typer.Typer(help="Product 4 -- build the CFA research paper.", no_args_is_help=True)
export_app = typer.Typer(help="On-demand PDF / PPT export of a product's report.", no_args_is_help=True)
site_app = typer.Typer(help="Product 5 -- static research site (dashboard + research + pricing).",
                       no_args_is_help=True)

app.add_typer(valuation_app, name="valuation")
app.add_typer(portfolio_app, name="portfolio")
portfolio_app.add_typer(manage_app, name="manage")
app.add_typer(research_app, name="research")
app.add_typer(export_app, name="export")
app.add_typer(site_app, name="site")


# ----------------------------- Product 1: Valuation -----------------------------
@valuation_app.command("generic")
def valuation_generic(ticker: str = typer.Argument(..., help="Ticker, e.g. GOOGL.")) -> None:
    """Generic (textbook) DCF -> reports/valuation/<TICKER>/generic_dcf.xlsx."""
    from dcf_project.valuation.generic import run_generic

    run_generic(ticker)


@valuation_app.command("dcf")
def valuation_dcf(
    ticker: str = typer.Argument(..., help="Ticker, e.g. KO."),
    mc_draws: int = typer.Option(0, "--mc-draws",
                                 help="Monte Carlo draws (0 = off; diagnostic-only driver sensitivity, G6-captioned)."),
    wacc_source: str = typer.Option("auto", "--wacc-source", help="auto | valueinvesting | internal"),
    terminal_growth_source: str = typer.Option("auto", "--terminal-growth-source"),
    company_type: str = typer.Option("auto", "--company-type"),
    country: str = typer.Option("auto", "--country",
                                help="auto | US | IN. auto infers IN from a .NS/.BO suffix; IN pulls "
                                     "NSE/BSE filings + live India macro (INR)."),
) -> None:
    """Fundamentals Fade DCF (FLAGSHIP) -> reports/valuation/<TICKER>/fade_dcf.xlsx.

    Economic-driver fade model (engine v3): faded growth/margin/RONIC with identity-derived
    reinvestment, per-driver half-lives, CAP/persistence, and a Grinold-Kroner terminal.
    """
    from dcf_project.valuation.canonical import run_canonical

    run_canonical(ticker, wacc_source=wacc_source, terminal_growth_source=terminal_growth_source,
                  company_type=company_type, engine="v3", mc_draws=mc_draws, country=country)


@valuation_app.command("canonical")
def valuation_canonical(
    ticker: str = typer.Argument(..., help="Ticker, e.g. GOOGL."),
    forecast_mode: str = typer.Option("consensus_blend", "--forecast-mode", help="state_space | consensus_blend"),
    wacc_source: str = typer.Option("auto", "--wacc-source", help="auto | valueinvesting | internal"),
    terminal_growth_source: str = typer.Option("auto", "--terminal-growth-source"),
    company_type: str = typer.Option("auto", "--company-type"),
    engine: str = typer.Option("v1", "--engine", help="v1 (frozen canonical) | v2 (Bayesian) | v3 (economic drivers)"),
) -> None:
    """Canonical frozen v1 DCF (research / live-book reproducibility) -> canonical_dcf.xlsx.

    Meaning is frozen: this is the engine the validation battery and the live paper book run on.
    For the product-facing valuation use `valuation dcf` (the Fundamentals Fade flagship).
    """
    from dcf_project.valuation.canonical import run_canonical

    run_canonical(ticker, forecast_mode=forecast_mode, wacc_source=wacc_source,
                  terminal_growth_source=terminal_growth_source, company_type=company_type,
                  engine=engine)


@valuation_app.command("compare")
def valuation_compare(
    ticker: str = typer.Argument(..., help="Ticker, e.g. KO."),
    mc_draws: int = typer.Option(0, "--mc-draws", help="Fade Monte Carlo draws (0 = off; G6-captioned band)."),
    country: str = typer.Option("auto", "--country",
                                help="auto | US | IN. IN triangulates v1 vs v3 only (generic leg is US-only)."),
) -> None:
    """Triangulate generic / canonical v1 / fade v3 -> reports/valuation/<TICKER>/comparison/comparison.xlsx."""
    from dcf_project.valuation.compare import run_compare

    run_compare(ticker, mc_draws=mc_draws, country=country)


@valuation_app.command("screen")
def valuation_screen(basket: Path = typer.Argument(..., help="CSV of tickers (one column 'ticker').")) -> None:
    """Basket screening -> ranked valuation workbook."""
    from dcf_project.valuation.screening import run_screen

    run_screen(basket)


# ----------------------------- Product 2: Constructor (proposal) -----------------------------
@portfolio_app.command("construct")
def portfolio_construct(
    signal: str = typer.Option("discount_x_conf_top10", "--signal", help="Validated signal / book id."),
    capital: float | None = typer.Option(None, "--capital", help="Capital to allocate, e.g. 1000000."),
) -> None:
    """Build a PROPOSAL portfolio -> reports/portfolio_builds/portfolio_YYYYQX/ (never live)."""
    from dcf_project.portfolio.constructor import run_construct

    run_construct(signal=signal, capital=capital)


@portfolio_app.command("list")
def portfolio_list() -> None:
    """List managed portfolios (registry) and available proposal builds."""
    from dcf_project.portfolio.registry import run_list

    run_list()


@portfolio_app.command("promote")
def portfolio_promote(build_id: str = typer.Argument(..., help="Build id, e.g. portfolio_2026Q3.")) -> None:
    """Promote a proposal build into a managed (live) portfolio -- explicit, audited."""
    from dcf_project.portfolio.registry import run_promote

    run_promote(build_id)


# ----------------------------- Product 3: Manager (live) -----------------------------
@manage_app.command("update")
def manage_update(book: str = typer.Option("conf_top10", "--book", help="Managed portfolio name.")) -> None:
    """Refresh prices/returns and regenerate the PM dashboard workbook (no live mutation)."""
    from dcf_project.portfolio.manager import run_manage

    run_manage(book, action="update")


@manage_app.command("update-all")
def manage_update_all() -> None:
    """Refresh EVERY active managed book (prices, track, status.json, news, dashboard)."""
    from dcf_project.portfolio.manager import run_manage_all

    run_manage_all()


@manage_app.command("riskstats")
def manage_riskstats(book: str = typer.Option("conf_top10", "--book", help="Managed portfolio name."),
                     years: float = typer.Option(3.0, "--years", help="Lookback window (years).")) -> None:
    """3y-weekly risk statistics (beta/vol/TE/correlations/div yield) -> <book>/risk_stats.json."""
    from dcf_project.portfolio.riskstats import run_riskstats

    run_riskstats(book, years=years)


@manage_app.command("preview")
def manage_preview(book: str = typer.Option("conf_top10", "--book")) -> None:
    """Preview the next rebalance (adds/drops/turnover); changes nothing."""
    from dcf_project.portfolio.manager import run_manage

    run_manage(book, action="preview")


@manage_app.command("rebalance")
def manage_rebalance(book: str = typer.Option("conf_top10", "--book")) -> None:
    """EXECUTE the rebalance -- the only command that mutates the live book."""
    from dcf_project.portfolio.manager import run_manage

    run_manage(book, action="rebalance")


# ----------------------------- Product 4: Research -----------------------------
@research_app.command("build-paper")
def research_build_paper(
    version: str = typer.Option("v2", "--version", help="v2 (corrected manuscript, default) | v1 (legacy)"),
) -> None:
    """Regenerate the CFA research package (v2 -> reports/paper_v2/, v1 -> reports/paper/)."""
    from dcf_project.research.paper import build_paper

    build_paper(version=version)


@research_app.command("status")
def research_status() -> None:
    """One-shot snapshot of the current bulky research_v2 run (from run_tracker.json)."""
    from dcf_project.research.run_tracker import read_status, render_status

    typer.echo(render_status(read_status()))


@research_app.command("watch")
def research_watch(
    interval: float = typer.Option(5.0, "--interval", help="Refresh seconds."),
) -> None:
    """Live-watch the current research_v2 run by polling run_tracker.json (Ctrl-C to stop)."""
    import os
    import time

    from dcf_project.research.run_tracker import read_status, render_status

    try:
        while True:
            snap = read_status()
            os.system("cls" if os.name == "nt" else "clear")
            typer.echo(render_status(snap))
            if snap is not None and snap.get("status") in ("completed", "failed"):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("\n(watch stopped)")


@research_app.command("tail")
def research_tail(
    n: int = typer.Option(8, "--n", help="Show the last N journal entries first."),
    follow: bool = typer.Option(True, "--follow/--no-follow", help="Keep streaming new entries."),
    interval: float = typer.Option(3.0, "--interval", help="Poll seconds when following."),
) -> None:
    """Stream the append-only research journal (latest milestones / decisions). Ctrl-C to stop."""
    import json
    import time

    from dcf_project.paths import REPORTS

    jsonl = REPORTS / "research_v2" / "research_journal.jsonl"
    if not jsonl.exists():
        typer.echo("no research journal yet (reports/research_v2/research_journal.jsonl)")
        return

    def _fmt(line: str) -> str | None:
        try:
            e = json.loads(line)
        except Exception:  # noqa: BLE001
            return None
        title = e.get("title") or (e.get("hypothesis") or "")[:80]
        dec = e.get("decision") or ""
        return f"[{e.get('datetime', '?')}] {title}\n    -> {dec}"

    lines = jsonl.read_text(encoding="utf-8").splitlines()
    for ln in lines[-n:]:
        msg = _fmt(ln)
        if msg:
            typer.echo(msg)
    seen = len(lines)
    if not follow:
        return
    typer.echo("--- following (Ctrl-C to stop) ---")
    try:
        while True:
            time.sleep(interval)
            lines = jsonl.read_text(encoding="utf-8").splitlines()
            for ln in lines[seen:]:
                msg = _fmt(ln)
                if msg:
                    typer.echo(msg)
            seen = len(lines)
    except KeyboardInterrupt:
        typer.echo("\n(tail stopped)")


# ----------------------------- Product 5: Site -----------------------------
@site_app.command("build")
def site_build() -> None:
    """Regenerate the static site from the books' JSON + research notes -> site/."""
    from dcf_project.site.builder import build

    build()


@site_app.command("deploy")
def site_deploy() -> None:
    """Publish site/ to the configured Pages repo (config: deploy_repo in site.json)."""
    from dcf_project.site.builder import deploy

    deploy()


@site_app.command("post")
def site_post(title: str = typer.Argument(..., help="Post title."),
              tags: str = typer.Option("", "--tags", help="Comma-separated tags."),
              summary: str = typer.Option("", "--summary", help="One-line index summary.")) -> None:
    """Scaffold today's research note in site_content/research/ (then `site build`)."""
    from dcf_project.site.builder import new_post

    new_post(title, tags=tags, summary=summary)


# ----------------------------- Export layer (on-demand) -----------------------------
@export_app.command("pdf")
def export_pdf(target: str = typer.Argument(..., help="Ticker (valuation) or portfolio/book name."),
               dashboard: bool = typer.Option(False, "--dashboard",
                   help="Render the live PM-dashboard model slot instead of the default (pitch) slot.")) -> None:
    """Render the target's existing ReportModel to PDF."""
    from dcf_project.reporting_kit.export import export_target

    export_target(target, fmt="pdf", variant="dashboard" if dashboard else None)


@export_app.command("ppt")
def export_ppt(target: str = typer.Argument(..., help="Ticker (valuation) or portfolio/book name."),
               dashboard: bool = typer.Option(False, "--dashboard",
                   help="Render the live PM-dashboard model slot instead of the default (pitch) slot.")) -> None:
    """Render the target's existing ReportModel to PowerPoint."""
    from dcf_project.reporting_kit.export import export_target

    export_target(target, fmt="pptx", variant="dashboard" if dashboard else None)


if __name__ == "__main__":
    app()
