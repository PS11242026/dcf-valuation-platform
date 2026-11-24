from __future__ import annotations

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod, MarketSnapshot, SourceMeta, WACCInputs
from dcf_project.data_sources.market import MarketClient, MarketDataError
from dcf_project.data_sources.sec import SECClient
from dcf_project.data_sources.wacc import resolve_wacc


class FundamentalsBundle:
    def __init__(
        self,
        periods: list[FinancialPeriod],
        market: MarketSnapshot,
        wacc: WACCInputs,
        sources: list[SourceMeta],
        warnings: list[str],
    ):
        self.periods = periods
        self.market = market
        self.wacc = wacc
        self.sources = sources
        self.warnings = warnings


def load_fundamentals(ticker: str, settings: Settings) -> FundamentalsBundle:
    warnings: list[str] = []
    # Filings source is country-routed. US (default) uses SEC EDGAR verbatim; India uses the
    # NSE/BSE XBRL bolt-on. Both emit the same FinancialPeriod contract, so everything below
    # (and the entire downstream engine) is identical regardless of source.
    if getattr(settings, "country", "US") == "IN":
        from dcf_project.data_sources.india.filings import IndiaFilingsClient

        periods, filing_source = IndiaFilingsClient(settings).financial_periods(ticker)
    else:
        periods, filing_source = SECClient(settings).financial_periods(ticker)
    # Yahoo needs an exchange suffix for Indian names (a bare NSE symbol like "HINDUNILVR"
    # resolves to nothing -> empty snapshot -> WACC fails). US path is unchanged.
    if getattr(settings, "country", "US") == "IN":
        market = _india_market_snapshot(ticker, settings, warnings)
        # valueinvesting.io indexes Indian listings under <symbol>.ns; use that form so the
        # same provider-first WACC path works for India as for the US.
        wacc_ticker = ticker.upper().removesuffix(".NS").removesuffix(".BO") + ".NS"
    else:
        market = MarketClient(settings).snapshot(ticker)
        wacc_ticker = ticker
    latest = periods[-1]
    # Market value of equity from the SEC share count when the market feed has a price but no
    # shares / market cap (common when Yahoo throttles cloud hosts).
    if market.market_cap is None and market.price:
        sec_shares = market.shares_outstanding or _float(latest.shares_outstanding)
        if sec_shares:
            market = market.model_copy(update={"shares_outstanding": sec_shares,
                                               "market_cap": market.price * sec_shares})
            warnings.append("Market cap derived from price x SEC shares outstanding.")

    # Debt is a sticky balance-sheet stock. Some issuers (e.g. Verizon) disclose
    # total interest-bearing debt only in the 10-K, so the latest quarter carries
    # an incomplete figure (current portion only). Use the most complete recent
    # reading (trailing-4-quarter max) to avoid understating leverage.
    recent_debt = [float(p.debt) for p in periods[-4:] if p.debt is not None]
    debt = max(recent_debt) if recent_debt else market.debt
    latest_tax_rate = None
    if latest.tax_expense is not None and latest.ebit not in (None, 0):
        latest_tax_rate = latest.tax_expense / abs(latest.ebit)
    wacc = resolve_wacc(wacc_ticker, market, latest_tax_rate, debt, settings)

    if market.shares_outstanding and not latest.shares_outstanding:
        warnings.append("Using Yahoo shares outstanding because latest SEC shares are missing.")
    if market.cash and not latest.cash:
        warnings.append("Using Yahoo cash because latest SEC cash is missing.")
    if market.debt and not latest.debt:
        warnings.append("Using Yahoo debt because latest SEC debt is missing.")
    return FundamentalsBundle(
        periods=periods,
        market=market,
        wacc=wacc,
        sources=[filing_source, market.source],
        warnings=warnings,
    )


def _is_populated(snapshot: MarketSnapshot) -> bool:
    return bool(snapshot.price or snapshot.market_cap or snapshot.shares_outstanding)


def _india_market_snapshot(ticker: str, settings: Settings, warnings: list[str]) -> MarketSnapshot:
    """Resolve an Indian market snapshot via Yahoo, normalising the exchange suffix.

    A bare NSE symbol (e.g. ``HINDUNILVR``) does not resolve on Yahoo; ``.NS`` (NSE) does, with
    ``.BO`` (BSE) as the fallback for names listed only on BSE. If the ticker already carries a
    suffix it is used as-is. Returns the first populated snapshot, else the last attempt.
    """
    base = ticker.upper()
    candidates = [ticker] if base.endswith((".NS", ".BO")) else [f"{base}.NS", f"{base}.BO"]
    client = MarketClient(settings)
    last: MarketSnapshot | None = None
    for candidate in candidates:
        try:
            snapshot = client.snapshot(candidate)
        except MarketDataError:
            continue
        last = snapshot
        if _is_populated(snapshot):
            if candidate != candidates[0]:
                warnings.append(f"Resolved India market data via {candidate} (NSE .NS lookup was empty).")
            return snapshot
    if last is not None:
        return last
    raise MarketDataError(f"No Yahoo market data for India ticker {ticker} (tried {', '.join(candidates)}).")


def _float(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None
