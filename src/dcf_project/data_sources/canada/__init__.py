"""Canada market bolt-on (additive, like ``india``).

Unlike India (live NSE XBRL), Canadian large caps are reachable through the **existing** SEC
EDGAR pipeline: most TSX names are SEC filers (full 10-K/10-Q domestic filers, or MJDS 40-F /
6-K, or 20-F), and EDGAR ``companyfacts`` carries deep point-in-time quarterly history for them.
So the Canada bolt-on reuses ``SECClient`` for fundamentals and adds only (i) a Canadian macro
profile for the internal-CAPM WACC + terminal band and (ii) a TSX universe / sector map.

Reporting currency is per-name (Barrick reports USD; Canadian Natural / CN Rail / Suncor report
CAD) and is detected from the EDGAR unit tag, so each name's intrinsic value and price share a
currency. CAD reporters are valued with the Canadian macro profile; USD reporters (global
operators that report in USD) are valued like a US name (USD risk-free + US ERP) with the
Canadian statutory tax fallback.
"""
