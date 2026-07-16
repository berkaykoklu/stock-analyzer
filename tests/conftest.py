from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def financials() -> pd.DataFrame:
    # columns: [2025, 2024] — newest first, matching yfinance.
    # 2024 Total Revenue is 850.0 (not 900.0) so that asset-turnover
    # improvement (period 0 vs period 1) is a strict inequality rather
    # than 1000/2000 == 900/1800.
    # Operating Income (2025: 320.0) pairs with balance_sheet's
    # Stockholders Equity (2025: 900.0) and Total Debt (2025: 300.0) to make
    # ROIC hand-checkable: nopat = 320 * (1 - 0.25) = 240;
    # invested_capital = 900 + 300 = 1200; roic = 240 / 1200 = 0.20 exactly.
    return pd.DataFrame(
        {"2025": [100.0, 400.0, 1000.0, 320.0], "2024": [80.0, 350.0, 850.0, 280.0]},
        index=["Net Income", "Gross Profit", "Total Revenue", "Operating Income"],
    )


@pytest.fixture
def balance_sheet() -> pd.DataFrame:
    # Stockholders Equity (2025: 900.0) makes ROIC computable/hand-checkable
    # together with the financials fixture's Operating Income — see comment there.
    return pd.DataFrame(
        {
            "2025": [2000.0, 300.0, 500.0, 250.0, 900.0],
            "2024": [1800.0, 320.0, 450.0, 300.0, 850.0],
        },
        index=[
            "Total Assets",
            "Total Debt",
            "Current Assets",
            "Current Liabilities",
            "Stockholders Equity",
        ],
    )


@pytest.fixture
def cashflow() -> pd.DataFrame:
    return pd.DataFrame(
        {"2025": [150.0], "2024": [110.0]},
        index=["Operating Cash Flow"],
    )


@pytest.fixture
def history() -> pd.DataFrame:
    # 250 days of a gentle uptrend with noise, deterministic.
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.1, 1.0, 250))
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, 250).astype(float),
        }
    )


@dataclass
class FakeMarketData:
    """MarketData Protocol implementation for tests: all fields passed in directly."""

    info: dict[str, object]
    history: pd.DataFrame
    financials: pd.DataFrame
    balance_sheet: pd.DataFrame
    cashflow: pd.DataFrame
