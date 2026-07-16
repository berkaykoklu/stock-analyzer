import math

import pandas as pd
import pytest

from stock_analyzer.analyzer import AnalysisReport, StockAnalyzer, analyze_ticker
from stock_analyzer.data import DataUnavailableError
from tests.conftest import FakeMarketData


def _info() -> dict[str, object]:
    return {
        "symbol": "TEST",
        "longName": "Test Corp",
        "sector": "Technology",
        "currentPrice": 100.0,
        "trailingEps": 10.0,
        "trailingPE": 12.0,
        "returnOnEquity": 0.22,
        # Deliberately no marketCap/freeCashflow: forces the trailing-P/E
        # valuation fallback instead of the multistage DCF.
    }


def test_analyze_full_fixture_report(financials, balance_sheet, cashflow, history):
    data = FakeMarketData(
        info=_info(),
        history=history,
        financials=financials,
        balance_sheet=balance_sheet,
        cashflow=cashflow,
    )

    report = StockAnalyzer(data).analyze()

    assert isinstance(report, AnalysisReport)

    # Identity
    assert report.ticker == "TEST"
    assert report.company_name == "Test Corp"
    assert report.price == pytest.approx(100.0)

    # Fundamentals: matches the tightened Task 3 fixture expectation.
    assert report.fundamentals.score == 7
    assert report.fundamentals.available == 8

    # Ratios: fully computable from the fixture, none missing.
    assert report.ratios["roa"] == pytest.approx(100.0 / 2000.0)
    assert report.ratios["debt_ratio"] == pytest.approx(300.0 / 2000.0)
    assert report.ratios["current_ratio"] == pytest.approx(500.0 / 250.0)
    assert report.ratios["gross_margin"] == pytest.approx(0.4)
    assert report.ratios["asset_turnover"] == pytest.approx(0.5)
    assert all(v is not None for v in report.ratios.values())

    # Technicals: 250-day history is long enough for every indicator.
    for col in ("rsi", "macd", "sma_50", "sma_200"):
        assert math.isfinite(report.technicals[col])
    assert report.rsi_label in {"Overbought", "Oversold", "Neutral", "Bullish", "Bearish"}
    assert report.support_resistance is not None
    support, resistance = report.support_resistance
    assert support < resistance

    # Valuation: falls back to trailing P/E given the minimal info dict.
    assert report.valuation.method == "trailing_pe"
    assert report.valuation.fair_value is not None
    assert report.valuation.upside_pct is not None

    # Quality: ROIC + ROE + gross margin + revenue growth factors are
    # computable (fixture's Operating Income/Stockholders Equity make ROIC
    # hand-checkable at exactly 0.20 — see tests/conftest.py comment).
    assert report.quality.moat_score == pytest.approx(8.0)
    assert report.quality.factors

    # Risk: deterministic uptrend fixture always yields a Sharpe ratio.
    assert report.risk.sharpe is not None

    # Macro: sector lookup is direct from info.
    assert report.macro.sector == "Technology"
    assert report.macro.cyclical is True

    # Composite: sentiment is always deferred/unavailable; everything else
    # in this fixture is available, so coverage is positive but < 1.
    assert report.composite.components["sentiment"].available is False
    assert report.composite.components["sentiment"].score == 0.0
    assert report.composite.coverage > 0
    assert report.composite.coverage < 1.0


def test_analyze_technical_component_unavailable_on_short_history(
    financials, balance_sheet, cashflow
):
    # Only 30 rows: too short for the 200-day SMA to produce a value.
    short_history = pd.DataFrame(
        {
            "Open": [100.0] * 30,
            "High": [101.0] * 30,
            "Low": [99.0] * 30,
            "Close": [100.0] * 30,
            "Volume": [1_000_000.0] * 30,
        }
    )
    data = FakeMarketData(
        info=_info(),
        history=short_history,
        financials=financials,
        balance_sheet=balance_sheet,
        cashflow=cashflow,
    )

    report = StockAnalyzer(data).analyze()

    assert report.composite.components["technical"].available is False
    assert report.composite.components["technical"].score == 0.0


def test_analyze_ticker_raises_on_invalid_data(monkeypatch):
    import stock_analyzer.data as data_module

    class _InvalidStubTicker:
        info: dict[str, object] = {}
        financials = pd.DataFrame()
        balance_sheet = pd.DataFrame()
        cashflow = pd.DataFrame()

        def history(self, period: str = "2y") -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(data_module.yf, "Ticker", lambda ticker: _InvalidStubTicker())

    with pytest.raises(DataUnavailableError):
        analyze_ticker("FAKE")


def test_analyze_ticker_falls_back_to_requested_ticker_when_symbol_missing(
    monkeypatch, financials, balance_sheet, cashflow, history
):
    # No network: yf.Ticker is monkeypatched with a stub carrying valid,
    # already-in-memory data (mirrors tests/test_data.py's _StubTicker).
    import stock_analyzer.data as data_module

    class _ValidStubTicker:
        def __init__(self) -> None:
            self.info: dict[str, object] = {"currentPrice": 100.0}  # no "symbol" key
            self.financials = financials
            self.balance_sheet = balance_sheet
            self.cashflow = cashflow

        def history(self, period: str = "2y") -> pd.DataFrame:
            return history

    monkeypatch.setattr(data_module.yf, "Ticker", lambda ticker: _ValidStubTicker())

    report = analyze_ticker("XYZ")

    assert report.ticker == "XYZ"
