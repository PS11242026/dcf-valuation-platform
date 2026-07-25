"""Construct the final backtest universe from the supplied master list.

Removes banks/insurance/REITs/telecom/utilities/biotech/dev-stage pharma (none are
present in the supplied list -- the data-analytics names SPGI/MCO/FICO/VRSK/EFX are
asset-light software/services, kept), dedupes, and expands within the same sector
families until the unique count exceeds 500.

Writes final_universe_membership.json and sector_counts.json (no network).
"""

from __future__ import annotations

import json
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

# family -> {"supplied": [...], "added": [...]}  (US-listed common-stock tickers)
FAMILIES: dict[str, dict[str, list[str]]] = {
    "software_saas_infra": {
        "supplied": ["CRM", "NOW", "ADBE", "INTU", "WDAY", "TEAM", "HUBS", "SNOW", "DDOG", "CRWD",
                     "PLTR", "MDB", "OKTA", "DOCU", "ZM", "GTLB", "DT", "ESTC", "CFLT", "NET",
                     "SMAR", "MNDY", "ASAN", "S", "PATH", "BRZE", "BILL", "BL", "QLYS", "RPD",
                     "FRSH", "SPT", "JAMF", "RNG", "FIVN", "PCTY", "PCOR", "NCNO", "APPF", "VEEV",
                     "GWRE", "BSY", "TYL", "MANH", "PRGS", "CWAN", "ZS", "SAIL", "TENB", "PEGA",
                     "VRNT", "TDC", "AI", "YEXT", "DOCN", "BOX", "DBX", "U", "ADSK", "SNPS",
                     "CDNS", "ANSS", "FICO", "VRSK", "SPGI", "MCO"],
        "added": ["PANW", "FTNT", "PAYC", "DAY", "ZI", "PD", "FROG", "APPN", "AYX", "WK",
                  "ZUO", "AMPL", "BIGC", "DV", "QTWO", "INTA", "SEMR", "KVYO", "TOST", "GLBE",
                  "ALRM", "EGHT", "CXM", "BASE", "DSGX", "SPSC", "BAND", "CALX", "NABL", "RXT",
                  "OS", "ENFN", "ALKT", "EVCM", "MITK", "PRO", "MODN", "FORG", "ENV", "EPAM",
                  "GLOB", "DSP", "APPS", "CRTO", "MGNI", "PUBM", "TTGT", "CCCS", "ESTC", "WIX",
                  "MSTR", "NICE", "CYBR", "VRNS", "OTEX", "VERX", "INFA", "BLKB", "ACIW", "AGYS",
                  "INST", "PWSC", "RAMP", "SPNS", "CDLX", "OSPN", "ALTR", "BLZE", "EXFY", "SUMO"],
    },
    "internet_platforms_marketplaces": {
        "supplied": ["AMZN", "GOOGL", "META", "ABNB", "BKNG", "EXPE", "MELI", "DASH", "UBER", "LYFT",
                     "EBAY", "ETSY", "APP", "PINS", "RDDT", "SSTK", "MTCH", "ZG", "CSGP", "CARG",
                     "CARS", "W", "TRIP", "FVRR", "UPWK", "GRPN", "SE", "CPNG", "SHOP", "GDDY",
                     "TTD", "ROKU", "SNAP", "IAC", "EB", "YELP"],
        "added": ["GOOG", "PYPL", "Z", "ANGI", "QNST", "CHGG", "BMBL", "DJT", "RUM", "OPRA",
                  "DUOL", "TASK", "SOUN", "PERI", "CART", "RBLX", "TDUP", "REAL", "EVER", "SFIX",
                  "VMEO", "CVNA", "OPEN", "WBD", "TTWO", "EA", "RBLX", "PTON", "CHWY", "DLO"],
    },
    "semiconductors": {
        "supplied": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "ADI", "MU", "MRVL", "MCHP",
                     "NXPI", "ON", "MPWR", "LSCC", "SWKS", "QRVO", "RMBS", "SLAB", "CRUS", "MTSI",
                     "POWI", "AMBA", "AOSL", "CEVA", "HIMX", "TSEM", "TSM", "UMC", "GFS", "ASX",
                     "ALGM", "WOLF", "CRDO", "ALAB", "ARM"],
        "added": ["STM", "INDI", "SITM", "SMTC", "DIOD", "VSH", "NVTS", "AOSL", "MX", "LITE",
                  "AMKR", "PI", "SGH", "QUIK", "GSIT", "MXL", "SYNA"],
    },
    "semiconductor_equipment": {
        "supplied": ["AMAT", "LRCX", "KLAC", "ASML", "TER", "ENTG", "ACLS", "COHR", "NVMI", "FORM",
                     "KLIC", "VECO", "COHU", "ONTO", "UCTT", "PLAB", "ICHR"],
        "added": ["AEIS", "MKSI", "BRKS", "AMBA", "CAMT", "ACMR", "AEHR", "KE", "II", "LASR"],
    },
    "consumer_staples": {
        "supplied": ["KO", "PEP", "COST", "PG", "MDLZ", "CL", "KMB", "CHD", "HSY", "GIS",
                     "K", "KHC", "CAG", "CPB", "MKC", "CLX", "TSN", "HRL", "LW", "MNST",
                     "CELH", "BJ", "DG", "DLTR", "WMT", "TGT"],
        "added": ["KVUE", "KDP", "STZ", "TAP", "SJM", "CAG", "POST", "FLO", "BGS", "UTZ",
                  "SAM", "KOF", "FMX", "BF.B", "EL", "KR", "ACI", "SFM", "GO", "PRMB",
                  "CASY", "PFGC", "USFD", "SYY", "CHEF", "COKE", "FIZZ", "KLG", "THS", "CENTA"],
    },
    "consumer_discretionary": {
        "supplied": ["NKE", "SBUX", "HD", "LOW", "MCD", "CMG", "TJX", "ROST", "LULU", "WSM",
                     "TSCO", "ORLY", "AZO", "ULTA", "DECK", "CROX", "YETI", "BBY", "RH", "POOL",
                     "DPZ", "DRI", "MAR", "HLT", "YUM", "HAS", "MAT"],
        "added": ["ELF", "ONON", "BIRK", "SKX", "COLM", "VFC", "RL", "TPR", "CPRI", "GPS",
                  "ANF", "URBN", "BURL", "DKS", "FND", "LEVI", "CROX", "WING", "CAVA", "SHAK",
                  "TXRH", "EAT", "BLMN", "QSR", "DENN", "PLAY", "GRMN", "WHR", "LEG", "MHK",
                  "GIII", "BOOT", "FIVE", "OLLI", "DG", "CASY", "WGO", "THO", "PVH", "KTB",
                  "CRI", "FL", "GES", "MOV", "SIG", "WOOF", "PETS", "CHDN", "MTN", "PLNT"],
    },
    "industrials_assetlight_services": {
        "supplied": ["ADP", "CTAS", "CPRT", "WM", "RSG", "EFX", "ROL", "ROP", "TDG", "HEI",
                     "AXON", "ODFL", "UNP", "CSX", "NSC", "CNI", "CP", "XYL", "VLTO", "J",
                     "ACM", "PWR", "EME", "FAST", "GWW", "WCC", "HUBB", "PH", "ITW", "DOV",
                     "FTV", "ROK", "ETN", "TT", "CARR", "DE", "CAT", "CMI", "OSK", "JBHT",
                     "EXPD", "CHRW", "GXO"],
        "added": ["PAYX", "BR", "JKHY", "G", "WEX", "FLT", "EEFT", "MMS", "ENV", "EXLS",
                  "ICFI", "FORR", "NSP", "TRU", "CSGS", "DNB", "FIS", "FI", "GPN", "TTEK",
                  "EXP", "PWR", "STRL", "MTZ", "DY", "ROAD", "FIX", "POWL", "NVT", "AYI",
                  "GGG", "NDSN", "IEX", "AME", "EMR", "PNR", "WAB", "GNRC", "ALLE", "MAS",
                  "SNA", "SWK", "LECO", "GTLS", "MIDD", "JBT", "CR", "FLS", "XPO", "SAIA",
                  "ARCB", "LSTR", "WERN", "KNX", "RXO", "FWRD", "HUBG", "MRTN", "CHE", "VRSK"],
    },
}

REMOVED_SECTORS = ["banks", "insurance", "REITs", "telecom", "utilities", "biotech", "development_stage_pharma"]


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    seen: set[str] = set()
    members = []
    for family, groups in FAMILIES.items():
        for source in ("supplied", "added"):
            for tkr in groups.get(source, []):
                if tkr in seen:
                    continue
                seen.add(tkr)
                members.append({"ticker": tkr, "sector_family": family, "source": source})

    sector_counts = {}
    for m in members:
        sector_counts[m["sector_family"]] = sector_counts.get(m["sector_family"], 0) + 1
    source_counts = {"supplied": sum(1 for m in members if m["source"] == "supplied"),
                     "added": sum(1 for m in members if m["source"] == "added")}

    membership = {
        "removed_sectors": REMOVED_SECTORS,
        "removal_note": "None present in supplied list to remove; SPGI/MCO/FICO/VRSK/EFX/financial-data are asset-light software/services (kept).",
        "total_unique_tickers": len(members),
        "exceeds_500": len(members) > 500,
        "source_counts": source_counts,
        "members": members,
    }
    (REPORT_DIR / "final_universe_membership.json").write_text(json.dumps(membership, indent=2), encoding="utf-8")
    (REPORT_DIR / "sector_counts.json").write_text(
        json.dumps({"sector_counts": sector_counts, "total": len(members), "source_counts": source_counts}, indent=2),
        encoding="utf-8",
    )

    print(f"TOTAL UNIQUE: {len(members)}  (exceeds 500: {len(members) > 500})")
    print(f"sources: {source_counts}")
    for fam, n in sector_counts.items():
        print(f"  {fam:38} {n}")
    # write the plain ticker list for the backtest harness
    (REPORT_DIR / "_universe_tickers.json").write_text(
        json.dumps({"tickers": [m["ticker"] for m in members], "family_of": {m["ticker"]: m["sector_family"] for m in members}}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
