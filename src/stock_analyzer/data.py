"""Market data access behind a Protocol, so tests and other layers never touch the network.

Ports the validation intent of the legacy `_load_all_data` (ticker with no usable
price data is unusable) but exposes it as an `is_valid` flag / `DataUnavailableError`
for callers to act on, instead of printing.
"""

from functools import cached_property
from typing import Protocol, cast

import pandas as pd
import yfinance as yf


class DataUnavailableError(Exception):
    """Raised by callers when a ticker's market data is not usable (see `is_valid`)."""


class MarketData(Protocol):
    @property
    def info(self) -> dict[str, object]: ...

    @property
    def history(self) -> pd.DataFrame: ...

    @property
    def financials(self) -> pd.DataFrame: ...

    @property
    def balance_sheet(self) -> pd.DataFrame: ...

    @property
    def cashflow(self) -> pd.DataFrame: ...


class YFinanceData:  # implements MarketData
    """MarketData backed by a single yfinance.Ticker; each property is fetched once."""

    def __init__(self, ticker: str, period: str = "2y") -> None:
        self.ticker_symbol = ticker
        self.period = period
        self._ticker = yf.Ticker(ticker)

    @cached_property
    def info(self) -> dict[str, object]:
        info = cast(dict[str, object] | None, self._ticker.info)
        return dict(info) if info else {}

    @cached_property
    def history(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._ticker.history(period=self.period))

    @cached_property
    def financials(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._ticker.financials)

    @cached_property
    def balance_sheet(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._ticker.balance_sheet)

    @cached_property
    def cashflow(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._ticker.cashflow)

    @property
    def is_valid(self) -> bool:
        has_price = self.info.get("regularMarketPrice") is not None or (
            self.info.get("currentPrice") is not None
        )
        return not self.history.empty and has_price
