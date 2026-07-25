from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from threading import Lock
from typing import Any

import requests

from dcf_project.config import Settings
from dcf_project.contracts import FinancialPeriod, SourceMeta
from dcf_project.data_sources import sec_cache


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# A period's source + fiscal metadata is owned by the financial-statement taxonomy
# (us-gaap / ifrs-full); dei (cover-page shares) and srt (supplementary) never override it.
# Without a stable precedence the metadata was set by whichever concept happened to be
# processed last, which is nondeterministic across processes (hash-randomized set iteration
# over concept names) -- it would flip a period's source between e.g. ifrs-full and dei.
_TAXONOMY_RANK = {"dei": 0, "srt": 1, "us-gaap": 3, "ifrs-full": 3}


def _taxonomy_rank(taxonomy: str | None) -> int:
    return _TAXONOMY_RANK.get(taxonomy or "", 2)


class SECDataError(RuntimeError):
    pass


class SECClient:
    _shared_ticker_cache: dict[str, str] | None = None
    _shared_ticker_cache_lock = Lock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": settings.sec_user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def cik_for_ticker(self, ticker: str) -> str:
        if SECClient._shared_ticker_cache is None:
            with SECClient._shared_ticker_cache_lock:
                if SECClient._shared_ticker_cache is None:
                    mapping: dict[str, str] | None = None
                    if self.settings.sec_cache_enabled:
                        mapping = sec_cache.load_ticker_map(max_age_days=self.settings.sec_cache_max_age_days)
                    if mapping is None:
                        response = requests.get(
                            SEC_COMPANY_TICKERS_URL,
                            headers={
                                "User-Agent": self.settings.sec_user_agent,
                                "Accept": "application/json",
                            },
                            timeout=self.settings.request_timeout_seconds,
                        )
                        response.raise_for_status()
                        mapping = {
                            item.get("ticker", "").upper(): f"{int(item['cik_str']):010d}"
                            for item in response.json().values()
                            if item.get("ticker") and item.get("cik_str")
                        }
                        if self.settings.sec_cache_enabled:
                            sec_cache.store_ticker_map(mapping, SEC_COMPANY_TICKERS_URL)
                    SECClient._shared_ticker_cache = mapping
        ticker_upper = ticker.upper()
        if ticker_upper in SECClient._shared_ticker_cache:
            return SECClient._shared_ticker_cache[ticker_upper]
        raise SECDataError(f"No SEC CIK found for ticker {ticker_upper}")

    def company_facts(self, ticker: str) -> tuple[dict[str, Any], SourceMeta]:
        cik = self.cik_for_ticker(ticker)
        url = SEC_COMPANY_FACTS_URL.format(cik=cik)
        if self.settings.sec_cache_enabled:
            envelope = sec_cache.load_cached(cik, max_age_days=self.settings.sec_cache_max_age_days)
            if envelope is not None:
                meta = SourceMeta(
                    provider="SEC", key=f"CIK{cik}", url=url,
                    retrieved_at=datetime.fromisoformat(envelope["retrieved_at"]),
                )
                return envelope["payload"], meta
        response = self.session.get(url, timeout=self.settings.request_timeout_seconds)
        response.raise_for_status()
        facts = response.json()
        if self.settings.sec_cache_enabled:
            sec_cache.store(cik, facts, url)
        return facts, SourceMeta(provider="SEC", key=f"CIK{cik}", url=url)

    def financial_periods(self, ticker: str) -> tuple[list[FinancialPeriod], SourceMeta]:
        facts, source = self.company_facts(ticker)
        taxonomies = _reporting_taxonomies(facts)
        periods: dict[date, dict[str, Any]] = defaultdict(dict)

        for field in _all_concept_fields():
            facts_for_field = []
            for taxonomy_name, taxonomy_facts in taxonomies.items():
                for concept in _concept_names_for_field(field, taxonomy_name):
                    facts_for_field.extend(_facts_for_concept(taxonomy_facts, concept, taxonomy_name, concept))
            if field in FLOW_FIELDS:
                selected = _select_quarterly_facts(facts_for_field)
                selected = _quarterize_flow_facts(selected)
            else:
                selected = _select_instant_facts(facts_for_field)
            for fact in selected:
                report_date = date.fromisoformat(fact["end"])
                _merge_fact_value(periods[report_date], field, float(fact["val"]))
                # Values always merge (above); period metadata is taken only from the
                # highest-ranked taxonomy seen for the date, so the source/fiscal labels are
                # deterministic and never downgraded to dei. (>= keeps "last financial fact
                # wins" within a taxonomy, matching the prior single-taxonomy behaviour.)
                new_tax = fact.get("_taxonomy")
                cur_tax = periods[report_date].get("taxonomy")
                if cur_tax is None or _taxonomy_rank(new_tax) >= _taxonomy_rank(cur_tax):
                    periods[report_date]["fiscal_year"] = fact.get("fy")
                    periods[report_date]["fiscal_period"] = fact.get("fp")
                    periods[report_date]["taxonomy"] = new_tax
                    periods[report_date]["form"] = fact.get("form")

        if _needs_annual_expansion(periods):
            periods = _expand_annual_periods(periods)

        rows = []
        for report_date, values in sorted(periods.items()):
            if not _is_real_reporting_period(values):
                # Drop phantom rows that carry no income-statement data. These are
                # created by instant-only facts whose `end` dates do not align with a
                # fiscal period end -- most commonly the dei EntityCommonStockShares-
                # Outstanding fact (dated at the filing cover page, ~3 weeks after the
                # period end) and stale comparative balance-sheet instants. Emitting
                # them as standalone periods interleaves NaN rows between real quarters
                # and destroys revenue.pct_change() (revenue_growth) contiguity.
                continue
            ca = values.get("current_assets")
            cl = values.get("current_liabilities")
            nwc = (ca - cl) if ca is not None and cl is not None else None
            debt = values.get("debt_total")
            if debt is None:
                debt_parts = [values.get("debt_current"), values.get("debt_noncurrent")]
                debt = sum(item for item in debt_parts if item is not None) or None
            rows.append(
                FinancialPeriod(
                    ticker=ticker.upper(),
                    report_date=report_date,
                    fiscal_year=values.get("fiscal_year"),
                    fiscal_period=values.get("fiscal_period"),
                    revenue=values.get("revenue"),
                    ebit=values.get("ebit"),
                    tax_expense=values.get("tax_expense"),
                    capex=abs(values["capex"]) if values.get("capex") is not None else None,
                    da=values.get("da"),
                    r_and_d=values.get("r_and_d"),
                    sbc=values.get("sbc"),
                    operating_lease_liability=values.get("operating_lease_liability"),
                    operating_lease_cost=values.get("operating_lease_cost"),
                    ppe=values.get("ppe"),
                    goodwill=values.get("goodwill"),
                    intangible_assets=values.get("intangible_assets"),
                    current_assets=ca,
                    current_liabilities=cl,
                    nwc=nwc,
                    cash=values.get("cash"),
                    debt=debt,
                    shares_outstanding=values.get("shares_outstanding"),
                    source=values.get("source", f"SEC:{values.get('taxonomy', 'unknown')}"),
                )
            )
        if not rows:
            concepts = set().union(*(_extract_concepts(items) for items in taxonomies.values())) if taxonomies else set()
            available = ", ".join(sorted(concepts)[:20])
            available_taxonomies = ", ".join(sorted(taxonomies)) or "none"
            raise SECDataError(
                f"No usable SEC facts found for supported GAAP/IFRS concepts. "
                f"Taxonomies={available_taxonomies}. Available sample: {available}"
            )
        return rows, source


CONCEPT_MAP = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "ebit": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "r_and_d": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "sbc": [
        "ShareBasedCompensation",
        "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodTotal",
        "AllocatedShareBasedCompensationExpense",
    ],
    "operating_lease_liability": [
        "OperatingLeaseLiability",
        "OperatingLeaseLiabilities",
        "OperatingLeaseLiabilityCurrent",
        "OperatingLeaseLiabilityNoncurrent",
    ],
    "operating_lease_cost": [
        "OperatingLeaseCost",
        "OperatingLeasesRentExpenseNet",
        "LeaseCost",
    ],
    "ppe": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],
    "goodwill": ["Goodwill"],
    "intangible_assets": [
        "FiniteLivedIntangibleAssetsNet",
        "IntangibleAssetsNetExcludingGoodwill",
        "GoodwillAndIntangibleAssetsNet",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "debt_total": [
        "DebtAndFinanceLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
        # Total (current + long-term) interest-bearing debt reported as a single
        # line by some issuers (e.g. Verizon). Without these, only DebtCurrent was
        # captured and total debt was understated by ~80% ($28B vs $158B for VZ).
        "DebtLongtermAndShorttermCombinedAmount",
        "DebtAndCapitalLeaseObligations",
    ],
    "debt_current": [
        "DebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
    ],
    "debt_noncurrent": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    ],
    "shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
}

IFRS_CONCEPT_MAP = {
    "revenue": ["Revenue", "RevenueFromContractsWithCustomers"],
    "ebit": ["ProfitLossFromOperatingActivities", "AccountingProfit", "ProfitLossBeforeTax"],
    "tax_expense": ["IncomeTaxExpenseContinuingOperations", "CurrentTaxExpenseIncome", "TaxExpenseIncome"],
    "capex": [
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PurchaseOfPropertyPlantAndEquipment",
    ],
    "da": ["DepreciationAndAmortisationExpense", "AdjustmentsForDepreciationAndAmortisationExpense"],
    "r_and_d": ["ResearchAndDevelopmentExpense"],
    "sbc": ["ExpenseFromSharebasedPaymentTransactionsWithEmployees", "AdjustmentsForSharebasedPayments"],
    "operating_lease_liability": ["LeaseLiabilities", "CurrentLeaseLiabilities", "NoncurrentLeaseLiabilities"],
    "operating_lease_cost": ["MinimumOperatingLeasePayments", "CashOutflowForLeases"],
    "ppe": ["PropertyPlantAndEquipment"],
    "goodwill": ["Goodwill"],
    "intangible_assets": ["IntangibleAssetsOtherThanGoodwill"],
    "current_assets": ["CurrentAssets", "CurrentAssetsOtherThanAssetsOrDisposalGroupsClassifiedAsHeldForSaleOrAsHeldForDistributionToOwners"],
    "current_liabilities": ["CurrentLiabilities", "CurrentLiabilitiesOtherThanLiabilitiesIncludedInDisposalGroupsClassifiedAsHeldForSale"],
    "cash": ["CashAndCashEquivalents", "CashAndCashEquivalentsIfDifferentFromStatementOfFinancialPosition", "Cash"],
    "debt_total": ["Borrowings", "FinancialLiabilities"],
    "debt_current": ["CurrentBorrowings", "CurrentFinancialLiabilities"],
    "debt_noncurrent": ["NoncurrentBorrowings", "NoncurrentFinancialLiabilities"],
    "shares_outstanding": ["AdjustedWeightedAverageShares"],
}

DEI_CONCEPT_MAP = {
    "shares_outstanding": ["EntityCommonStockSharesOutstanding"],
}

FLOW_FIELDS = {"revenue", "ebit", "tax_expense", "capex", "da", "r_and_d", "sbc", "operating_lease_cost"}
FP_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4}
VALID_FORMS = {"10-Q", "10-K", "20-F", "40-F", "6-K"}


def _is_real_reporting_period(values: dict[str, Any]) -> bool:
    """A usable fiscal period must carry income-statement flow data.

    Phantom rows produced by cover-page share counts (dei) or stale comparative
    balance-sheet instants have neither revenue nor operating income and must not
    be emitted as standalone periods.
    """
    return values.get("revenue") is not None or values.get("ebit") is not None


def _reporting_taxonomies(facts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    available = facts.get("facts", {})
    return {
        name: available.get(name, {})
        for name in ("us-gaap", "ifrs-full", "dei")
        if isinstance(available.get(name), dict) and available.get(name)
    }


def _all_concept_fields() -> set[str]:
    return set(CONCEPT_MAP) | set(IFRS_CONCEPT_MAP) | set(DEI_CONCEPT_MAP)


def _concept_names_for_field(field: str, taxonomy_name: str) -> list[str]:
    if taxonomy_name == "ifrs-full":
        return IFRS_CONCEPT_MAP.get(field, [])
    if taxonomy_name == "dei":
        return DEI_CONCEPT_MAP.get(field, [])
    return CONCEPT_MAP.get(field, [])


def _extract_concepts(us_gaap: dict[str, Any]) -> set[str]:
    return set(us_gaap.keys())


def _facts_for_concept(taxonomy: dict[str, Any], concept: str, taxonomy_name: str, source_concept: str) -> list[dict[str, Any]]:
    units = taxonomy.get(concept, {}).get("units", {})
    facts = []
    for unit_values in units.values():
        for fact in unit_values:
            item = dict(fact)
            item["_taxonomy"] = taxonomy_name
            item["_concept"] = source_concept
            facts.append(item)
    return facts


def _select_quarterly_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        if "val" not in fact or "end" not in fact:
            continue
        form = fact.get("form")
        if form not in VALID_FORMS:
            continue
        key = (fact["end"], fact.get("accn", ""))
        if key in seen:
            continue
        seen.add(key)
        selected.append(fact)
    selected.sort(key=lambda item: item["end"])
    return selected


def _select_instant_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_end: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if "val" not in fact or "end" not in fact:
            continue
        if fact.get("form") not in VALID_FORMS:
            continue
        end = fact["end"]
        existing = by_end.get(end)
        candidate = dict(fact)
        candidate["_restatement_indicator"] = False
        if existing is None or _instant_fact_priority(candidate) < _instant_fact_priority(existing):
            if existing is not None:
                candidate["_restatement_indicator"] = True
            by_end[end] = candidate
    return [by_end[end] for end in sorted(by_end)]


def _instant_fact_priority(fact: dict[str, Any]) -> tuple[int, str, str]:
    form_penalty = 0 if fact.get("form") == "10-Q" else 1
    return (0, str(fact.get("filed", "")), f"{form_penalty}:{fact.get('accn', '')}")


def _quarterize_flow_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_year: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    passthrough = []
    for fact in facts:
        fy = fact.get("fy")
        fp = fact.get("fp")
        if not isinstance(fy, int) or fp not in FP_ORDER:
            passthrough.append(fact)
            continue
        existing = by_year[fy].get(fp)
        if existing is None or fact.get("filed", "") >= existing.get("filed", ""):
            by_year[fy][fp] = fact

    quarterized = []
    for fy in sorted(by_year):
        previous_ytd = 0.0
        for fp in sorted(by_year[fy], key=lambda item: FP_ORDER[item]):
            fact = dict(by_year[fy][fp])
            value = float(fact["val"])
            if _looks_cumulative(fact, fp):
                fact["val"] = value - previous_ytd
                previous_ytd = value
            else:
                fact["val"] = value
                previous_ytd += value
            quarterized.append(fact)
    quarterized.extend(passthrough)
    quarterized.sort(key=lambda item: item["end"])
    return quarterized


def _looks_cumulative(fact: dict[str, Any], fp: str) -> bool:
    if fp == "Q1":
        return False
    start = fact.get("start")
    end = fact.get("end")
    if not start or not end:
        return fp in {"Q2", "Q3", "FY"}
    try:
        duration = (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:
        return fp in {"Q2", "Q3", "FY"}
    return duration > 120


def _merge_fact_value(period: dict[str, Any], field: str, value: float) -> None:
    if field in {"debt_current", "debt_noncurrent"}:
        period[field] = value
    elif field == "debt_total":
        period[field] = max(value, period.get(field, 0.0))
    elif field == "cash":
        period[field] = max(value, period.get(field, 0.0))
    elif field == "shares_outstanding":
        if value > 0:
            period[field] = value
    elif field == "da":
        period[field] = max(value, period.get(field, 0.0))
    else:
        period[field] = value


def _needs_annual_expansion(periods: dict[date, dict[str, Any]]) -> bool:
    if not periods:
        return False
    rows = list(periods.values())
    has_foreign_annual = any(row.get("form") in {"20-F", "40-F"} for row in rows)
    has_quarters = any(row.get("fiscal_period") in {"Q1", "Q2", "Q3"} for row in rows)
    return has_foreign_annual and (len(periods) < 20 or not has_quarters)


def _expand_annual_periods(periods: dict[date, dict[str, Any]]) -> dict[date, dict[str, Any]]:
    expanded: dict[date, dict[str, Any]] = defaultdict(dict)
    for annual_end, values in sorted(periods.items()):
        if values.get("fiscal_period") != "FY":
            expanded[annual_end].update(values)
            continue
        quarter_ends = [_shift_months(annual_end, -9), _shift_months(annual_end, -6), _shift_months(annual_end, -3), annual_end]
        for index, quarter_end in enumerate(quarter_ends, start=1):
            row = dict(values)
            for field in FLOW_FIELDS:
                if row.get(field) is not None:
                    row[field] = float(row[field]) / 4.0
            row["fiscal_period"] = f"Q{index}" if index < 4 else "FY"
            row["source"] = f"SEC:{values.get('taxonomy', 'unknown')}:annual_expanded"
            row["annual_expansion"] = True
            expanded[quarter_end].update(row)
    return expanded


def _shift_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    if month in {4, 6, 9, 11}:
        return 30
    return 31
