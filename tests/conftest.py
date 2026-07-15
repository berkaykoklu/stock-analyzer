import pandas as pd
import pytest


@pytest.fixture
def financials() -> pd.DataFrame:
    # columns: [2025, 2024] — newest first, matching yfinance.
    # 2024 Total Revenue is 850.0 (not 900.0) so that asset-turnover
    # improvement (period 0 vs period 1) is a strict inequality rather
    # than 1000/2000 == 900/1800.
    return pd.DataFrame(
        {"2025": [100.0, 400.0, 1000.0], "2024": [80.0, 350.0, 850.0]},
        index=["Net Income", "Gross Profit", "Total Revenue"],
    )


@pytest.fixture
def balance_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {"2025": [2000.0, 300.0, 500.0, 250.0], "2024": [1800.0, 320.0, 450.0, 300.0]},
        index=["Total Assets", "Total Debt", "Current Assets", "Current Liabilities"],
    )


@pytest.fixture
def cashflow() -> pd.DataFrame:
    return pd.DataFrame(
        {"2025": [150.0], "2024": [110.0]},
        index=["Operating Cash Flow"],
    )
