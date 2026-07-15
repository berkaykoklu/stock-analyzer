"""Pure fundamental-analysis ratio calculations over yfinance-shaped DataFrames.

DataFrames use yfinance orientation: row index = line items
(e.g. "Net Income", "Total Assets"), columns = periods, newest first.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class PiotroskiResult:
    score: int
    checks: dict[str, bool]
    available: int


def _line(df: pd.DataFrame, item: str, period: int) -> float | None:
    try:
        value = df.loc[item].iloc[period]
    except (KeyError, IndexError):
        return None
    return None if pd.isna(value) else float(value)


def roa(financials: pd.DataFrame, balance_sheet: pd.DataFrame, period: int = 0) -> float | None:
    net_income = _line(financials, "Net Income", period)
    total_assets = _line(balance_sheet, "Total Assets", period)
    if net_income is None or total_assets is None or total_assets == 0:
        return None
    return net_income / total_assets


def debt_ratio(balance_sheet: pd.DataFrame, period: int = 0) -> float | None:
    total_debt = _line(balance_sheet, "Total Debt", period)
    if total_debt is None:
        total_debt = _line(balance_sheet, "Long Term Debt", period)
    total_assets = _line(balance_sheet, "Total Assets", period)
    if total_debt is None or total_assets is None or total_assets == 0:
        return None
    return total_debt / total_assets


def current_ratio(balance_sheet: pd.DataFrame, period: int = 0) -> float | None:
    current_assets = _line(balance_sheet, "Current Assets", period)
    current_liabilities = _line(balance_sheet, "Current Liabilities", period)
    if current_assets is None or current_liabilities is None or current_liabilities == 0:
        return None
    return current_assets / current_liabilities


def gross_margin(financials: pd.DataFrame, period: int = 0) -> float | None:
    gross_profit = _line(financials, "Gross Profit", period)
    total_revenue = _line(financials, "Total Revenue", period)
    if gross_profit is None or total_revenue is None or total_revenue == 0:
        return None
    return gross_profit / total_revenue


def asset_turnover(
    financials: pd.DataFrame, balance_sheet: pd.DataFrame, period: int = 0
) -> float | None:
    total_revenue = _line(financials, "Total Revenue", period)
    total_assets = _line(balance_sheet, "Total Assets", period)
    if total_revenue is None or total_assets is None or total_assets == 0:
        return None
    return total_revenue / total_assets


def piotroski_score(
    financials: pd.DataFrame, balance_sheet: pd.DataFrame, cashflow: pd.DataFrame
) -> PiotroskiResult:
    """Enhanced Piotroski F-Score (9 checks), honest about missing data.

    Unlike the legacy implementation, a check that cannot be computed from
    the supplied statements (missing row, missing prior period, etc.) is
    excluded from both `score` and `available` rather than silently
    defaulted to failing or passing.
    """
    checks: dict[str, bool] = {}

    net_income = _line(financials, "Net Income", 0)
    if net_income is not None:
        checks["positive_net_income"] = net_income > 0

    operating_cf = _line(cashflow, "Operating Cash Flow", 0)
    if operating_cf is not None:
        checks["positive_operating_cash_flow"] = operating_cf > 0

    current_roa = roa(financials, balance_sheet, 0)
    previous_roa = roa(financials, balance_sheet, 1)
    if current_roa is not None and previous_roa is not None:
        checks["roa_improved"] = current_roa > previous_roa

    if net_income is not None and operating_cf is not None:
        checks["earnings_quality"] = operating_cf > net_income

    current_debt_ratio = debt_ratio(balance_sheet, 0)
    previous_debt_ratio = debt_ratio(balance_sheet, 1)
    if current_debt_ratio is not None and previous_debt_ratio is not None:
        checks["debt_ratio_decreased"] = current_debt_ratio < previous_debt_ratio

    current_current_ratio = current_ratio(balance_sheet, 0)
    previous_current_ratio = current_ratio(balance_sheet, 1)
    if current_current_ratio is not None and previous_current_ratio is not None:
        checks["current_ratio_increased"] = current_current_ratio > previous_current_ratio

    # Deviation from legacy: compares Share Issued across periods;
    # legacy only checked sharesOutstanding presence, not a dilution signal.
    current_shares = _line(balance_sheet, "Share Issued", 0)
    previous_shares = _line(balance_sheet, "Share Issued", 1)
    if current_shares is not None and previous_shares is not None:
        checks["no_dilution"] = current_shares <= previous_shares

    current_margin = gross_margin(financials, 0)
    previous_margin = gross_margin(financials, 1)
    if current_margin is not None and previous_margin is not None:
        checks["gross_margin_increased"] = current_margin > previous_margin

    current_turnover = asset_turnover(financials, balance_sheet, 0)
    previous_turnover = asset_turnover(financials, balance_sheet, 1)
    if current_turnover is not None and previous_turnover is not None:
        checks["asset_turnover_increased"] = current_turnover > previous_turnover

    score = sum(1 for passed in checks.values() if passed)
    return PiotroskiResult(score=score, checks=checks, available=len(checks))
