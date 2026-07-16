"""Small internal helpers shared across analysis modules."""

import pandas as pd


def as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def line(df: pd.DataFrame, item: str, period: int = 0) -> float | None:
    """Look up a single line item from a yfinance-shaped statement DataFrame
    (row index = line items, columns = periods, newest first)."""
    try:
        value = df.loc[item].iloc[period]
    except (KeyError, IndexError):
        return None
    return None if pd.isna(value) else float(value)
