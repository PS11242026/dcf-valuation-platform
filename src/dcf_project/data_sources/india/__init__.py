"""India market bolt-on (NSE/BSE filings + live India macro).

Additive and isolated: nothing here is imported on the US path. Selected per ticker via
``data_sources.markets.resolve_country`` returning ``"IN"`` (a ``.NS``/``.BO`` suffix or an
explicit ``--country IN``). Everything degrades to documented static fallbacks offline so a
valuation never hard-fails on a flaky exchange/macro endpoint.
"""
from __future__ import annotations
