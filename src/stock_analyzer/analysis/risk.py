"""Risk metrics over a price-history DataFrame: annualized volatility, Sharpe
ratio, maximum drawdown, and (when benchmark data is supplied) beta.

Legacy `risk.py` computed historical/parametric Value-at-Risk and canned
stress-test scenarios but had no volatility, Sharpe, or beta calculation at
all; those don't map onto the target interface and are not ported. Only
`max_drawdown`'s formula (expanding-peak drawdown) is a direct port of the
legacy `calculate_maximum_drawdown` math; `volatility` and `sharpe` are new,
standard implementations added to satisfy the target `RiskResult` shape.

DataFrames use standard OHLCV orientation: row index = trading days
(oldest first), columns include "Close".
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class RiskResult:
    volatility: float
    sharpe: float | None
    max_drawdown: float
    beta: float | None


def metrics(history: pd.DataFrame, risk_free_rate: float = 0.02) -> RiskResult:
    returns = history["Close"].pct_change().dropna()

    daily_std = float(returns.std())
    volatility = daily_std * float(np.sqrt(TRADING_DAYS_PER_YEAR))

    sharpe: float | None = None
    if daily_std > 0:
        annualized_return = float(returns.mean()) * TRADING_DAYS_PER_YEAR
        sharpe = (annualized_return - risk_free_rate) / volatility

    prices = history["Close"]
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    max_drawdown = float(drawdown.min())

    return RiskResult(
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        beta=None,  # no benchmark series in this signature
    )
