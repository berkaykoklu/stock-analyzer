import pandas as pd
import pytest

from stock_analyzer.ratios import (
    asset_turnover,
    current_ratio,
    debt_ratio,
    gross_margin,
    piotroski_score,
    roa,
)


def test_roa(financials, balance_sheet):
    assert roa(financials, balance_sheet) == pytest.approx(100.0 / 2000.0)


def test_roa_missing_row_returns_none(balance_sheet):
    empty = pd.DataFrame()
    assert roa(empty, balance_sheet) is None


def test_roa_zero_denominator_returns_none(financials):
    zero_assets = pd.DataFrame(
        {"2025": [0.0]},
        index=["Total Assets"],
    )
    assert roa(financials, zero_assets) is None


def test_debt_ratio(balance_sheet):
    assert debt_ratio(balance_sheet) == pytest.approx(300.0 / 2000.0)


def test_debt_ratio_falls_back_to_long_term_debt():
    bs = pd.DataFrame(
        {"2025": [2000.0, 400.0]},
        index=["Total Assets", "Long Term Debt"],
    )
    assert debt_ratio(bs) == pytest.approx(400.0 / 2000.0)


def test_debt_ratio_zero_total_debt_does_not_fall_back():
    bs = pd.DataFrame(
        {"2025": [2000.0, 0.0, 400.0]},
        index=["Total Assets", "Total Debt", "Long Term Debt"],
    )
    assert debt_ratio(bs) == pytest.approx(0.0)


def test_current_ratio(balance_sheet):
    assert current_ratio(balance_sheet) == pytest.approx(500.0 / 250.0)


def test_gross_margin(financials):
    assert gross_margin(financials) == pytest.approx(400.0 / 1000.0)


def test_asset_turnover(financials, balance_sheet):
    assert asset_turnover(financials, balance_sheet) == pytest.approx(1000.0 / 2000.0)


def test_period_selects_prior_column(financials, balance_sheet):
    assert roa(financials, balance_sheet, period=1) == pytest.approx(80.0 / 1800.0)
    assert gross_margin(financials, period=1) == pytest.approx(350.0 / 850.0)


def test_piotroski_mixed_fixture_deterministic(financials, balance_sheet, cashflow):
    result = piotroski_score(financials, balance_sheet, cashflow)
    # Fixtures give: positive NI, positive OCF, ROA up, OCF > NI,
    # debt ratio down, current ratio up, asset turnover up (strictly,
    # thanks to the 850.0 2024 revenue fixture value).
    # With the fixture's 2024 revenue of 850, the gross-margin check is
    # genuinely False and the dilution check is unavailable.
    assert result.score == 7
    assert result.available == 8
    assert "no_dilution" not in result.checks


def test_piotroski_missing_statements_yields_no_checks():
    empty = pd.DataFrame()
    result = piotroski_score(empty, empty, empty)
    assert result.score == 0
    assert result.available == 0
    assert result.checks == {}
