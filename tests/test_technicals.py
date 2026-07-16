import numpy as np
import pandas as pd
import pytest

from stock_analyzer.technicals import (
    compute_indicators,
    interpret_rsi,
    support_resistance,
    volume_trend,
)


@pytest.fixture
def history() -> pd.DataFrame:
    # 250 days of a gentle uptrend with noise, deterministic
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


def test_compute_indicators_adds_columns(history):
    out = compute_indicators(history)
    for col in ["rsi", "macd", "sma_50", "sma_200", "bb_high", "bb_low"]:
        assert col in out.columns
    assert not out["rsi"].tail(10).isna().any()


@pytest.mark.parametrize(
    ("rsi", "label"),
    [
        (75.0, "Overbought"),
        (25.0, "Oversold"),
        (50.0, "Neutral"),
        (65.0, "Bullish"),
        (35.0, "Bearish"),
    ],
)
def test_interpret_rsi(rsi, label):
    assert interpret_rsi(rsi) == label


def test_support_resistance_bounds(history):
    result = support_resistance(history)
    assert result is not None
    support, resistance = result
    assert support < resistance
    assert history["Low"].tail(60).min() <= support
    assert resistance <= history["High"].tail(60).max()


def test_volume_trend_detects_spike(history):
    spiked = history.copy()
    spiked.loc[spiked.index[-5:], "Volume"] = 10_000_000.0
    assert volume_trend(spiked) == "high"
    assert volume_trend(history) == "normal"
