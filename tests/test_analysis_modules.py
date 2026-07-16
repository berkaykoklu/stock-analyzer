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
    # fixture financials has Operating Income (320.0) and balance_sheet has
    # Stockholders Equity (900.0) + Total Debt (300.0), so ROIC is
    # computable: nopat = 320 * (1 - 0.25) = 240; invested_capital =
    # 900 + 300 = 1200; roic = 240 / 1200 = 0.20 exactly.
    info: dict[str, object] = {"returnOnEquity": 0.22}
    result = moat_score(info, financials, balance_sheet)

    assert math.isclose(result.factors["roic"], 0.20)
    assert result.factors["roe"] == 0.22
    assert result.factors["gross_margin"] == 0.4  # 400 / 1000
    assert math.isclose(result.factors["revenue_growth"], (1000 - 850) / 850)
    # +3 roic (0.20 > 0.15), +2 roe, +1 gross_margin, +2 revenue_growth
    assert result.moat_score == 8.0
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


def test_macro_context_matches_real_yfinance_sector_name():
    # Real yfinance `info["sector"]` values use "Financial Services", not
    # the legacy "Financial" key — regression guard for the key modernization.
    financials = context({"sector": "Financial Services"})
    assert financials.cyclical is True
    assert "+0.6" in financials.notes
    assert "benefits from rate increases" in financials.notes.lower()

    consumer_cyclical = context({"sector": "Consumer Cyclical"})
    assert consumer_cyclical.cyclical is True
    assert "-0.4" in consumer_cyclical.notes

    consumer_defensive = context({"sector": "Consumer Defensive"})
    assert consumer_defensive.cyclical is False
    assert "-0.4" in consumer_defensive.notes


def test_estimate_beta_zero_affects_wacc():
    """Regression: beta=0.0 should produce different WACC/fair_value than beta absent,
    not be replaced by DEFAULT_BETA."""
    # Minimal DCF-eligible info dict
    base_info = {
        "marketCap": 1_000_000_000.0,
        "totalDebt": 100_000_000.0,
        "freeCashflow": 50_000_000.0,
        "sharesOutstanding": 100_000_000.0,
        "currentPrice": 10.0,
        "trailingEps": 1.0,
        "trailingPE": 10.0,
    }

    # Minimal financials with revenue growth and tax data for DCF.
    # yfinance orientation: row index = line items, columns = periods
    # (newest first), matching every other fixture in this suite.
    financials = pd.DataFrame(
        {"2025": [1_000.0, 100.0, 21.0, 5.0], "2024": [950.0, 95.0, 19.95, 4.75]},
        index=["Total Revenue", "Pretax Income", "Tax Provision", "Interest Expense"],
    )

    # Test with beta=0.0 (explicit zero)
    info_with_beta_zero = {**base_info, "beta": 0.0}
    result_beta_zero = estimate(info_with_beta_zero, financials)

    # Test with beta absent (should fall back to DEFAULT_BETA=1.0)
    result_no_beta = estimate(base_info, financials)

    # Both should use DCF (have enough data)
    assert result_beta_zero.method == "multistage_dcf"
    assert result_no_beta.method == "multistage_dcf"

    # With beta=0.0, cost_of_equity is lower, so WACC is lower, so fair_value is higher
    # (lower discount rate means higher PV). They should differ.
    assert result_beta_zero.fair_value is not None
    assert result_no_beta.fair_value is not None
    assert result_beta_zero.fair_value != result_no_beta.fair_value
