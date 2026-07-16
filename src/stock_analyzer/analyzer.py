"""Orchestrates all analysis modules into a single AnalysisReport.

Thin wiring layer: no I/O, no printing. Pure composition of `ratios`,
`technicals`, `scoring`, and `analysis.*` over an already-fetched
`MarketData` instance. Replaces the legacy
`analysis_engine.py:154-179` + `687-816` orchestration.
"""

from dataclasses import dataclass, replace

import pandas as pd

from stock_analyzer._util import as_float
from stock_analyzer.analysis.macro import MacroResult
from stock_analyzer.analysis.macro import context as macro_context
from stock_analyzer.analysis.quality import QualityResult, moat_score
from stock_analyzer.analysis.risk import RiskResult
from stock_analyzer.analysis.risk import metrics as risk_metrics
from stock_analyzer.analysis.valuation import ValuationResult
from stock_analyzer.analysis.valuation import estimate as valuation_estimate
from stock_analyzer.data import DataUnavailableError, MarketData, YFinanceData
from stock_analyzer.ratios import (
    PiotroskiResult,
    asset_turnover,
    current_ratio,
    debt_ratio,
    gross_margin,
    piotroski_score,
    roa,
)
from stock_analyzer.scoring import (
    ComponentScore,
    CompositeResult,
    composite_score,
    quality_component,
    technical_component,
    valuation_component,
)
from stock_analyzer.technicals import compute_indicators, interpret_rsi, support_resistance

_TECHNICAL_COLS = ("rsi", "macd", "sma_50", "sma_200")


@dataclass(frozen=True)
class AnalysisReport:
    ticker: str
    company_name: str
    price: float | None
    fundamentals: PiotroskiResult
    ratios: dict[str, float | None]
    technicals: dict[str, float]
    rsi_label: str
    support_resistance: tuple[float, float] | None
    valuation: ValuationResult
    quality: QualityResult
    risk: RiskResult
    macro: MacroResult
    composite: CompositeResult


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _technical_readings(
    history: pd.DataFrame,
) -> tuple[dict[str, float], bool, str, tuple[float, float] | None, float]:
    """Latest technical readings, a 0-3 trend score, and availability.

    Available iff the history is long enough for every indicator (notably
    the 200-day SMA) to have a non-NaN value on the most recent bar.
    """
    if history.empty:
        values = {col: float("nan") for col in _TECHNICAL_COLS}
        return values, False, "N/A", None, 0.0

    df = compute_indicators(history)
    latest = df.iloc[-1]
    values = {col: float(latest[col]) for col in _TECHNICAL_COLS}
    available = all(pd.notna(latest[col]) for col in _TECHNICAL_COLS)

    latest_close = float(latest["Close"])
    trend_score = float(
        (latest_close > values["sma_50"])
        + (values["sma_50"] > values["sma_200"])
        + (values["macd"] > 0)
    )
    rsi_label = interpret_rsi(values["rsi"])
    sr = support_resistance(history)
    return values, available, rsi_label, sr, trend_score


class StockAnalyzer:
    """Ties fundamentals, technicals, valuation, quality, risk, and macro
    modules together into a single AnalysisReport.

    Pure computation only: the caller is responsible for supplying an
    already-fetched `MarketData` instance.
    """

    def __init__(self, data: MarketData) -> None:
        self._data = data

    def analyze(self) -> AnalysisReport:
        info = self._data.info
        history = self._data.history
        financials = self._data.financials
        balance_sheet = self._data.balance_sheet
        cashflow = self._data.cashflow

        ticker = _as_str(info.get("symbol"))
        company_name = _as_str(info.get("longName")) or _as_str(info.get("shortName")) or ticker
        price = as_float(info.get("currentPrice")) or as_float(info.get("regularMarketPrice"))

        fundamentals = piotroski_score(financials, balance_sheet, cashflow)
        ratios: dict[str, float | None] = {
            "roa": roa(financials, balance_sheet),
            "debt_ratio": debt_ratio(balance_sheet),
            "current_ratio": current_ratio(balance_sheet),
            "gross_margin": gross_margin(financials),
            "asset_turnover": asset_turnover(financials, balance_sheet),
        }

        technicals, technical_available, rsi_label, sr, trend_score = _technical_readings(history)

        valuation = valuation_estimate(info, financials)
        quality = moat_score(info, financials, balance_sheet)
        risk = risk_metrics(history)
        macro = macro_context(info)

        upside_pct = valuation.upside_pct
        quality_factors = quality.factors
        sharpe = risk.sharpe

        components: dict[str, ComponentScore] = {
            "fundamental": ComponentScore(
                score=(fundamentals.score / max(fundamentals.available, 1)) * 100,
                available=fundamentals.available > 0,
            ),
            "technical": (
                ComponentScore(
                    technical_component(technicals["rsi"], trend_score, technicals["macd"]),
                    available=True,
                )
                if technical_available
                else ComponentScore(0.0, available=False)
            ),
            "valuation": (
                ComponentScore(valuation_component(upside_pct), available=True)
                if upside_pct is not None
                else ComponentScore(0.0, available=False)
            ),
            "quality": (
                ComponentScore(quality_component(quality.moat_score), available=True)
                if quality_factors
                else ComponentScore(0.0, available=False)
            ),
            # Net-new mapping (no legacy equivalent; legacy folded risk in ad
            # hoc): simple linear band from Sharpe to 0-100,
            # sharpe -1 -> 0, sharpe 2 -> 100.
            "risk": (
                ComponentScore(min(max((sharpe + 1) / 3, 0.0), 1.0) * 100, available=True)
                if sharpe is not None
                else ComponentScore(0.0, available=False)
            ),
            # Sentiment module deferred (see README roadmap): always unavailable.
            "sentiment": ComponentScore(0.0, available=False),
        }
        composite = composite_score(components)

        return AnalysisReport(
            ticker=ticker,
            company_name=company_name,
            price=price,
            fundamentals=fundamentals,
            ratios=ratios,
            technicals=technicals,
            rsi_label=rsi_label,
            support_resistance=sr,
            valuation=valuation,
            quality=quality,
            risk=risk,
            macro=macro,
            composite=composite,
        )


def analyze_ticker(ticker: str) -> AnalysisReport:
    """Convenience wrapper: fetch `ticker` via yfinance and analyze it.

    Raises `DataUnavailableError` if the ticker has no usable market data
    (see `YFinanceData.is_valid`).
    """
    data = YFinanceData(ticker)
    if not data.is_valid:
        raise DataUnavailableError(f"No usable market data for ticker '{ticker}'")
    report = StockAnalyzer(data).analyze()
    # info["symbol"] is usually present, but fall back to the requested
    # ticker rather than silently returning an empty ticker field.
    return report if report.ticker else replace(report, ticker=ticker)
