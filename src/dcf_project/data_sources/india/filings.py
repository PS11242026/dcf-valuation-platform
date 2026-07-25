"""India NSE XBRL filings client -- the SEBI-side analogue of ``data_sources.sec``.

SEC's ``companyfacts`` is pre-parsed XBRL served as JSON; the faithful India analogue is the
NSE corporate-filings API, which lists every SEBI LODR Reg-33 result with a direct link to its
Ind-AS XBRL instance. This client does the exchange handshake, lists a symbol's quarterly AND
annual filings, downloads each XBRL, and parses them into the SAME :class:`FinancialPeriod`
contract the SEC client emits -- so the whole downstream reconstruction + engine_v3 stack is
reused untouched.

Validated against live NSE data (RELIANCE, 2018-2024). Hard-won facts about the taxonomy:

- Reporting columns are keyed by **context-id prefix on the FACT** (``OneD`` = current period,
  ``FourD`` = year-to-date, ``OneI`` = current period-end instant), NOT by period dates -- every
  context is stamped with the current period. Critically, in 2018-2021 filings the ``OneD``
  context is *referenced by the facts but never defined* in the instance, so selection must match
  the fact's ``contextRef`` string directly (not via the parsed context table), and the report
  date comes from the NSE record's ``toDate`` (authoritative), not the instance.
- Monetary facts are **full rupees** ("Crores" rounding describes the PDF, not the values).
- Consolidated is preferred over standalone.

Data completeness: quarterly results XBRL is income-statement-only (revenue / operating profit /
tax / D&A). **Annual** filings additionally carry the balance sheet (cash, PP&E, current assets &
liabilities, goodwill), which this client merges onto the matching fiscal-year-end quarter so the
reconstruction can carry invested-capital forward. Debt is sourced from the Yahoo snapshot
(``load_fundamentals``). **Capex** is not in the results XBRL and the cash-flow statement is only
inconsistently filed (RELIANCE's, for instance, is P&L + BS only), so ``_derive_capex`` builds it
from ``D&A + YoY ΔPP&E`` (see that function); a real cash-flow capex value is preferred when present.
Pre-2018 "Old format" filings expose only HTML (no XBRL) and are skipped; 2018+ already yields ~28
quarters, well past the engine's 20-quarter minimum.
"""
from __future__ import annotations

import gzip
import hashlib
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from threading import Lock

import requests

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod, SourceMeta
from dcf_project.paths import CACHE

#: FinancialPeriod field -> candidate Ind-AS taxonomy local-names (namespace stripped), in
#: priority order. ``ebit`` is approximated by profit-before-(exceptional/tax) since Ind-AS
#: results carry no single "operating income" tag.
IND_AS_CONCEPT_MAP: dict[str, list[str]] = {
    "revenue": ["RevenueFromOperations", "Revenue", "TotalRevenueFromOperations"],
    "ebit": ["ProfitBeforeExceptionalItemsAndTax", "ProfitLossBeforeExceptionalItemsAndTax",
             "ProfitBeforeTax", "ProfitLossBeforeTax"],
    "tax_expense": ["TaxExpense", "TotalTaxExpense", "IncomeTaxExpenseContinuingOperations"],
    "da": ["DepreciationDepletionAndAmortisationExpense", "DepreciationAndAmortisationExpense"],
    # Real cash-flow capex (only some filers include the cash-flow statement). The Ind-AS taxonomy
    # uses the "...ClassifiedAsInvestingActivities" element names; when absent, ``_derive_capex``
    # synthesises capex from D&A + ΔPP&E instead.
    "capex": ["PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
              "PurchaseOfPropertyPlantAndEquipment", "AcquisitionOfPropertyPlantAndEquipment"],
    "ppe": ["PropertyPlantAndEquipment", "PropertyPlantAndEquipmentNet"],
    "goodwill": ["Goodwill"],
    "intangible_assets": ["IntangibleAssetsOtherThanGoodwill", "OtherIntangibleAssets"],
    "current_assets": ["CurrentAssets", "TotalCurrentAssets"],
    "current_liabilities": ["CurrentLiabilities", "TotalCurrentLiabilities"],
    "cash": ["CashAndCashEquivalents", "CashAndCashEquivalentsCashFlowStatement"],
    "debt_total": ["Borrowings", "TotalBorrowings"],
    "debt_current": ["CurrentBorrowings", "ShorttermBorrowings"],
    "debt_noncurrent": ["NoncurrentBorrowings", "LongtermBorrowings"],
    "shares_outstanding": ["NumberOfSharesOutstanding", "NumberOfEquityShares",
                           "NumberOfFullyPaidEquityShares"],
}

#: Flow (duration-context) fields; the rest are instant (balance-sheet) stocks.
FLOW_FIELDS = {"revenue", "ebit", "tax_expense", "da", "capex"}

#: Sticky balance-sheet stocks sourced from annual filings and carried across the quarters
#: between annual snapshots (invested capital moves slowly; this is the same forward-fill the
#: SEC reconstruction applies to annually-disclosed stocks).
BALANCE_SHEET_CARRY = ["cash", "ppe", "current_assets", "current_liabilities", "goodwill", "intangible_assets"]

#: NSE INDAS entity-level current-period contextRef tokens (validated 2018-2024 live).
ENTITY_DURATION_CTX = "OneD"
ENTITY_INSTANT_CTX = "OneI"

NSE_HOME = "https://www.nseindia.com/"
NSE_RESULTS_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
NSE_RESULTS_API = ("https://www.nseindia.com/api/corporates-financial-results"
                   "?index=equities&symbol={symbol}&period={period}")

INDIA_CACHE_DIR = CACHE / "india"
#: Cap on quarters fetched per name (most-recent first) -- politeness + runtime bound.
MAX_QUARTERS = 40
#: Annual filings fetched (most-recent first) for balance-sheet enrichment.
MAX_ANNUAL = 12


class IndiaFilingsError(RuntimeError):
    pass


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _parse_contexts(root: ET.Element) -> dict[str, dict[str, str]]:
    """contextRef id -> {start, end, instant} for the contexts that ARE defined."""
    key_map = {"startDate": "start", "endDate": "end", "instant": "instant"}
    contexts: dict[str, dict[str, str]] = {}
    for ctx in root.iter():
        if _localname(ctx.tag) != "context" or not ctx.get("id"):
            continue
        info: dict[str, str] = {}
        for node in ctx.iter():
            key = key_map.get(_localname(node.tag))
            if key and node.text:
                info[key] = node.text.strip()
        contexts[ctx.get("id")] = info
    return contexts


def _concept_to_field() -> dict[str, str]:
    return {c: field for field, concepts in IND_AS_CONCEPT_MAP.items() for c in concepts}


def parse_india_xbrl_instance(content: str | bytes) -> list[dict[str, object]]:
    """Parse one Ind-AS instance into a flat list of mapped facts (no scaling: full rupees).

    Each fact: ``{field, concept, ctx, val, start, end, instant}``. Kept general (all contexts)
    for inspection; the per-period selection lives in ``extract_period_values``.
    """
    root = ET.fromstring(content.encode("utf-8") if isinstance(content, str) else content)
    contexts = _parse_contexts(root)
    concept_to_field = _concept_to_field()
    facts: list[dict[str, object]] = []
    for el in root.iter():
        field = concept_to_field.get(_localname(el.tag))
        if field is None or el.text is None:
            continue
        raw = el.text.strip().replace(",", "")
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        ctx_id = el.get("contextRef", "")
        ctx = contexts.get(ctx_id, {})
        facts.append({"field": field, "concept": _localname(el.tag), "ctx": ctx_id, "val": value,
                      "start": ctx.get("start"), "end": ctx.get("end"), "instant": ctx.get("instant")})
    return facts


def _select_entity_values(root: ET.Element, instants_only: bool = False) -> dict[str, float]:
    """Select entity-level current-period facts by contextRef (OneD flows / OneI instants).

    Matches the contextRef STRING directly so it works even when the OneD/OneI context is
    referenced-but-undefined (the 2018-2021 quirk). ``instants_only`` keeps balance-sheet stocks
    only (used for annual enrichment, so a full-year flow never clobbers the quarter's P&L).
    """
    concept_to_field = _concept_to_field()
    values: dict[str, float] = {}
    for el in root.iter():
        field = concept_to_field.get(_localname(el.tag))
        if field is None or el.text is None:
            continue
        is_flow = field in FLOW_FIELDS
        if instants_only and is_flow:
            continue
        want = ENTITY_DURATION_CTX if is_flow else ENTITY_INSTANT_CTX
        if el.get("contextRef") != want:
            continue
        raw = el.text.strip().replace(",", "")
        try:
            values[field] = float(raw)
        except ValueError:
            continue
    return values


def extract_period_values(content: str | bytes, instants_only: bool = False) -> dict[str, float]:
    root = ET.fromstring(content.encode("utf-8") if isinstance(content, str) else content)
    return _select_entity_values(root, instants_only)


def quarter_values_from_instance(content: str | bytes) -> tuple[date, dict[str, float]] | None:
    """Convenience: (report_date, values) using the OneD/OneI context date when it IS defined.

    The client uses the NSE record's ``toDate`` for the date instead (always available); this
    wrapper is for tests/inspection where the instance is self-contained.
    """
    root = ET.fromstring(content.encode("utf-8") if isinstance(content, str) else content)
    values = _select_entity_values(root)
    contexts = _parse_contexts(root)
    end_iso = (contexts.get(ENTITY_DURATION_CTX, {}).get("end")
               or contexts.get(ENTITY_INSTANT_CTX, {}).get("instant"))
    if not values or not end_iso:
        return None
    try:
        return date.fromisoformat(end_iso), values
    except ValueError:
        return None


def _financial_period(ticker: str, report_date: date, v: dict[str, float], consolidated: bool) -> FinancialPeriod:
    debt = v.get("debt_total")
    if debt is None:
        parts = [v.get("debt_current"), v.get("debt_noncurrent")]
        debt = sum(p for p in parts if p is not None) or None
    ca, cl = v.get("current_assets"), v.get("current_liabilities")
    return FinancialPeriod(
        ticker=ticker.upper(), report_date=report_date,
        revenue=v.get("revenue"), ebit=v.get("ebit"), tax_expense=v.get("tax_expense"),
        capex=abs(v["capex"]) if v.get("capex") is not None else None, da=v.get("da"),
        ppe=v.get("ppe"), goodwill=v.get("goodwill"), intangible_assets=v.get("intangible_assets"),
        current_assets=ca, current_liabilities=cl,
        nwc=(ca - cl) if ca is not None and cl is not None else None,
        cash=v.get("cash"), debt=debt, shares_outstanding=v.get("shares_outstanding"),
        source=f"NSE:ind-as:{'consolidated' if consolidated else 'standalone'}",
    )


def _choose_consolidated(records: list[dict]) -> dict[str, dict]:
    """One filing per period end (toDate), preferring the consolidated variant."""
    chosen: dict[str, dict] = {}
    for rec in records:
        xbrl, to_date = rec.get("xbrl"), rec.get("toDate")
        if not xbrl or not str(xbrl).startswith("http") or not to_date:
            continue
        is_cons = str(rec.get("consolidated", "")).strip().lower() == "consolidated"
        prev = chosen.get(to_date)
        if prev is None or (is_cons and not prev["_cons"]):
            chosen[to_date] = {**rec, "_cons": is_cons}
    return chosen


class IndiaFilingsClient:
    """Fetch + parse NSE XBRL results (quarterly P&L + annual balance sheet) into FinancialPeriods."""

    _session: requests.Session | None = None
    _session_lock = Lock()

    def __init__(self, settings: Settings):
        self.settings = settings

    def _nse_session(self) -> requests.Session:
        if IndiaFilingsClient._session is not None:
            return IndiaFilingsClient._session
        with IndiaFilingsClient._session_lock:
            if IndiaFilingsClient._session is None:
                s = requests.Session()
                s.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/124 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                })
                try:
                    s.get(NSE_HOME, timeout=self.settings.request_timeout_seconds)
                    s.get(NSE_RESULTS_REFERER, timeout=self.settings.request_timeout_seconds)
                except Exception as exc:  # noqa: BLE001
                    raise IndiaFilingsError(f"NSE handshake failed: {exc}") from exc
                IndiaFilingsClient._session = s
            return IndiaFilingsClient._session

    def _list_filings(self, symbol: str, period: str = "Quarterly") -> list[dict]:
        s = self._nse_session()
        try:
            r = s.get(NSE_RESULTS_API.format(symbol=symbol, period=period),
                      headers={"Accept": "application/json", "Referer": NSE_RESULTS_REFERER},
                      timeout=self.settings.request_timeout_seconds)
            r.raise_for_status()
            records = r.json()
        except Exception as exc:  # noqa: BLE001
            raise IndiaFilingsError(f"NSE {period} listing failed for {symbol}: {exc}") from exc
        return records if isinstance(records, list) else []

    def _fetch_xbrl(self, url: str) -> bytes:
        cache_path = INDIA_CACHE_DIR / f"{hashlib.sha1(url.encode()).hexdigest()}.xml.gz"
        if cache_path.exists():
            with gzip.open(cache_path, "rb") as fh:
                return fh.read()
        s = self._nse_session()
        r = s.get(url, headers={"Accept": "application/xml,*/*", "Referer": NSE_RESULTS_REFERER},
                  timeout=self.settings.request_timeout_seconds)
        r.raise_for_status()
        content = r.content
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with gzip.open(tmp, "wb") as fh:
            fh.write(content)
        tmp.replace(cache_path)
        return content

    def financial_periods(self, ticker: str) -> tuple[list[FinancialPeriod], SourceMeta]:
        symbol = ticker.upper().removesuffix(".NS").removesuffix(".BO")
        quarterly = _choose_consolidated(self._list_filings(symbol, "Quarterly"))
        try:
            annual = _choose_consolidated(self._list_filings(symbol, "Annual"))
        except IndiaFilingsError:
            annual = {}  # annual enrichment is best-effort; quarterly P&L stands alone

        q_ordered = sorted(quarterly.values(),
                           key=lambda r: _parse_nse_date(r["toDate"]) or date.min)[-MAX_QUARTERS:]
        merged: dict[date, dict[str, float]] = {}
        cons_flag: dict[date, bool] = {}
        for rec in q_ordered:
            report_date = _parse_nse_date(rec["toDate"])
            values = self._values_for(rec)
            if report_date is None or not values:
                continue
            merged.setdefault(report_date, {}).update(values)
            cons_flag[report_date] = rec["_cons"]

        # Annual balance-sheet enrichment: attach instant stocks onto the matching fiscal
        # year-end quarter already in the panel (carry-forward fills the intervening quarters).
        if merged:
            window_start = min(merged)
            for rec in sorted(annual.values(), key=lambda r: _parse_nse_date(r["toDate"]) or date.min)[-MAX_ANNUAL:]:
                report_date = _parse_nse_date(rec["toDate"])
                if report_date is None or report_date < window_start or report_date not in merged:
                    continue
                bs = self._values_for(rec, instants_only=True)
                if bs:
                    merged[report_date].update(bs)

        _carry_balance_sheet(merged)
        _derive_capex(merged)
        rows = [_financial_period(ticker, rd, vals, cons_flag.get(rd, True))
                for rd, vals in sorted(merged.items())]
        if not rows:
            raise IndiaFilingsError(
                f"No usable Ind-AS result facts parsed for NSE:{symbol}. The symbol may not be an "
                f"NSE equity, or its filings use an off-template taxonomy -- reconcile against published results."
            )
        return rows, SourceMeta(provider="NSE", key=symbol,
                                url=NSE_RESULTS_API.format(symbol=symbol, period="Quarterly"),
                                retrieved_at=datetime.now(timezone.utc))

    def _values_for(self, rec: dict, instants_only: bool = False) -> dict[str, float]:
        """Fetch + parse one filing; returns {} on any fetch/parse failure (e.g. HTML old-format)."""
        try:
            xml = self._fetch_xbrl(rec["xbrl"])
            values = extract_period_values(xml, instants_only=instants_only)
        except Exception:  # noqa: BLE001  -- one bad filing must not sink the series
            return {}
        time.sleep(0.3)  # politeness between archive downloads (cached on re-run)
        return values


def _carry_balance_sheet(merged: dict[date, dict[str, float]]) -> None:
    """Forward- then back-fill the sticky balance-sheet stocks across the quarter panel.

    Annual filings supply invested-capital components only at fiscal year-ends; carrying them
    to the intervening quarters (and back-filling quarters before the first annual snapshot)
    lets the reconstruction compute ROIC/invested capital for every quarter instead of dropping
    the ~80% that lack a quarter-level balance sheet.
    """
    dates = sorted(merged)
    last: dict[str, float] = {}
    for d in dates:
        for field in BALANCE_SHEET_CARRY:
            if merged[d].get(field) is not None:
                last[field] = merged[d][field]
            elif field in last:
                merged[d][field] = last[field]
    nxt: dict[str, float] = {}
    for d in reversed(dates):
        for field in BALANCE_SHEET_CARRY:
            if merged[d].get(field) is not None:
                nxt[field] = merged[d][field]
            elif field in nxt:
                merged[d][field] = nxt[field]


def _derive_capex(merged: dict[date, dict[str, float]]) -> None:
    """Derive a per-quarter capex where the cash-flow statement is absent (the common case for
    NSE results XBRL, which carries only the P&L + balance sheet).

        capex_q = D&A_q + max(0, PPE_q - PPE_{q-4}) / 4

    i.e. maintenance capex anchored to depreciation PLUS the year-over-year growth in (carried)
    net PPE, spread across the four quarters of the fiscal year. When PPE is flat this reduces to
    capex == D&A (FCF = NOPAT - ΔNWC, maintenance only); when PPE grows it adds the net new
    investment, so reconstruction's ``fcf = nopat + da - capex - Δnwc`` correctly nets out growth
    capital instead of treating missing capex as zero (which over-states FCF for capital-intensive
    names -- exactly the India gap). A real cash-flow capex value, if ever present, is preserved.
    """
    dates = sorted(merged)
    for i, d in enumerate(dates):
        if merged[d].get("capex") is not None:        # real cash-flow capex present -> keep it
            continue
        da = merged[d].get("da")
        if da is None:
            continue
        capex = float(da)
        ppe = merged[d].get("ppe")
        if i >= 4 and ppe is not None:
            ppe_prev = merged[dates[i - 4]].get("ppe")
            if ppe_prev is not None:
                growth = float(ppe) - float(ppe_prev)
                if growth > 0:
                    capex += growth / 4.0
        merged[d]["capex"] = capex


def _parse_nse_date(text: str | None) -> date | None:
    """Parse NSE's '31-Dec-2024' date format."""
    if not text:
        return None
    try:
        return datetime.strptime(text.strip(), "%d-%b-%Y").date()
    except ValueError:
        return None
