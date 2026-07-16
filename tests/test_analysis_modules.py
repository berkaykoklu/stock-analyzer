import math

import pandas as pd

from stock_analyzer.analysis.macro import context
from stock_analyzer.analysis.quality import moat_score
from stock_analyzer.analysis.risk import metrics
from stock_analyzer.analysis.valuation import estimate


def test_estimate_falls_back_to_trailing_pe_with_minimal_info():
    info: dict[str, object] = {"trailingPE": 10.0, "trailingEps": 5.0, "currentPrice": 50.0}
    result = estimate(info, pd.DataFrame())

    assert result.method == "trailing_pe"
    assert result.fair_value is not None
    assert math.isfinite(result.fair_value)
    assert result.upside_pct is not None
    assert math.isfinite(result.upside_pct)


def test_estimate_insufficient_data_returns_none_fields():
    result = estimate({}, pd.DataFrame())
    assert result.fair_value is None
    assert result.upside_pct is None
    assert result.method == "insufficient_data"


def test_moat_score_uses_available_factors(financials, balance_sheet):
    # fixture financials/balance_sheet have no Operating Income or Total
    # Stockholder Equity, so ROIC is not computable -> excluded from factors.
    info: dict[str, object] = {"returnOnEquity": 0.22}
    result = moat_score(info, financials, balance_sheet)

    assert result.factors["roe"] == 0.22
    assert result.factors["gross_margin"] == 0.4  # 400 / 1000
    assert math.isclose(result.factors["revenue_growth"], (1000 - 850) / 850)
    assert "roic" not in result.factors
    assert result.moat_score == 5.0  # +2 roe, +1 gross_margin, +2 revenue_growth
    assert 0.0 <= result.moat_score <= 10.0


def test_risk_metrics_on_deterministic_history(history):
    result = metrics(history)

    assert result.volatility > 0
    assert result.max_drawdown <= 0
    assert result.sharpe is not None
    assert result.beta is None


def test_macro_context_classifies_sector():
    tech = context({"sector": "Technology"})
    assert tech.sector == "Technology"
    assert tech.cyclical is True
    assert "sensitiv" in tech.notes.lower()

    defensive = context({"sector": "Healthcare"})
    assert defensive.cyclical is False

    unknown = context({})
    assert unknown.sector == ""
    assert unknown.cyclical is False
