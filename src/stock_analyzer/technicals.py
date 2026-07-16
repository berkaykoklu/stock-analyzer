"""Technical indicator computation and interpretation over OHLCV history.

DataFrames use standard OHLCV orientation: row index = trading days
(oldest first), columns = Open, High, Low, Close, Volume.
"""

import math

import pandas as pd
import ta


def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    close = df["Close"]

    rsi = ta.momentum.RSIIndicator(close, window=14)
    macd = ta.trend.MACD(close)
    sma_50 = ta.trend.SMAIndicator(close, window=50)
    sma_200 = ta.trend.SMAIndicator(close, window=200)
    bb = ta.volatility.BollingerBands(close)

    df["rsi"] = rsi.rsi()
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["sma_50"] = sma_50.sma_indicator()
    df["sma_200"] = sma_200.sma_indicator()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    return df


def interpret_rsi(rsi: float) -> str:
    if math.isnan(rsi):
        return "N/A"
    if rsi >= 70:
        return "Overbought"
    elif rsi <= 30:
        return "Oversold"
    elif 40 <= rsi <= 60:
        return "Neutral"
    elif rsi > 60:
        return "Bullish"
    else:
        return "Bearish"


def support_resistance(history: pd.DataFrame, lookback: int = 60) -> tuple[float, float] | None:
    if len(history) < lookback:
        return None

    df = history.tail(lookback)
    highs = df["High"].rolling(window=5, center=True).max()
    lows = df["Low"].rolling(window=5, center=True).min()

    support = lows.tail(20).min()
    resistance = highs.tail(20).max()
    return float(support), float(resistance)


def volume_trend(history: pd.DataFrame) -> str:
    df = history.tail(20)
    recent_avg_volume = df["Volume"].tail(5).mean()
    historical_avg_volume = df["Volume"].head(15).mean()

    if recent_avg_volume > historical_avg_volume * 1.5:
        return "high"
    elif recent_avg_volume < historical_avg_volume * 0.7:
        return "low"
    else:
        return "normal"
