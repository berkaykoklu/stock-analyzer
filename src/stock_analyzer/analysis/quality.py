"""Business-quality and moat scoring over yfinance-shaped fundamentals.

DataFrames use yfinance orientation: row index = line items, columns =
periods, newest first.
"""

from dataclasses import dataclass

import pandas as pd

from stock_analyzer._util import as_float

DEFAULT_TAX_RATE = 0.25


@dataclass
class QualityResult:
    moat_score: float
    factors: dict[str, float]


def _line(df: pd.DataFrame, item: str, period: int = 0) -> float | None:
    try:
        value = df.loc[item].iloc[period]
    except (KeyError, IndexError):
        return None
    return None if pd.isna(value) else float(value)


# Deviation from legacy: legacy's .loc.get bug meant the default rate was ALWAYS used;
# this computes the real effective rate when Pretax Income / Tax Provision exist.
def _tax_rate(financials: pd.DataFrame) -> float:
    pretax_income = _line(financials, "Pretax Income")
    if pretax_income is None or pretax_income <= 0:
        return DEFAULT_TAX_RATE
    tax_expense = _line(financials, "Tax Provision") or 0.0
    return abs(tax_expense / pretax_income)


def _roic(financials: pd.DataFrame, balance_sheet: pd.DataFrame) -> float | None:
    operating_income = _line(financials, "Operating Income")
    total_equity = _line(balance_sheet, "Total Stockholder Equity")
    if operating_income is None or total_equity is None:
        return None

    total_debt = _line(balance_sheet, "Total Debt")
    if total_debt is None:
        total_debt = _line(balance_sheet, "Long Term Debt") or 0.0
    cash = _line(balance_sheet, "Cash And Cash Equivalents") or 0.0

    invested_capital = total_equity + total_debt - cash
    if invested_capital <= 0:
        return 0.0

    nopat = operating_income * (1 - _tax_rate(financials))
    return nopat / invested_capital


def _gross_margin(financials: pd.DataFrame) -> float | None:
    total_revenue = _line(financials, "Total Revenue")
    gross_profit = _line(financials, "Gross Profit")
    if total_revenue is None or gross_profit is None or total_revenue <= 0:
        return None
    return gross_profit / total_revenue


def _revenue_growth(financials: pd.DataFrame) -> float | None:
    if financials.shape[1] < 2:
        return None
    current = _line(financials, "Total Revenue", 0)
    previous = _line(financials, "Total Revenue", 1)
    if current is None or previous is None or previous <= 0:
        return None
    return (current - previous) / previous


def moat_score(
    info: dict[str, object], financials: pd.DataFrame, balance_sheet: pd.DataFrame
) -> QualityResult:
    """Competitive-moat score (0-10): ROIC, ROE, gross margin, revenue growth brackets."""
    roic = _roic(financials, balance_sheet)
    roe = as_float(info.get("returnOnEquity"))
    gross_margin = _gross_margin(financials)
    revenue_growth = _revenue_growth(financials)

    score = 0.0

    if roic is not None and roic > 0.15:
        score += 3
    elif roic is not None and roic > 0.10:
        score += 2
    elif roic is not None and roic > 0.05:
        score += 1

    if roe is not None and roe > 0.20:
        score += 2
    elif roe is not None and roe > 0.15:
        score += 1

    if gross_margin is not None and gross_margin > 0.60:
        score += 3
    elif gross_margin is not None and gross_margin > 0.40:
        score += 2
    elif gross_margin is not None and gross_margin > 0.25:
        score += 1

    if revenue_growth is not None and revenue_growth > 0.10:
        score += 2
    elif revenue_growth is not None and revenue_growth > 0.05:
        score += 1

    factors: dict[str, float] = {}
    if roic is not None:
        factors["roic"] = roic
    if roe is not None:
        factors["roe"] = roe
    if gross_margin is not None:
        factors["gross_margin"] = gross_margin
    if revenue_growth is not None:
        factors["revenue_growth"] = revenue_growth

    return QualityResult(moat_score=score, factors=factors)
