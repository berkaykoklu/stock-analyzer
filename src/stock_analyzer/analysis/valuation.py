"""Valuation: multi-stage DCF with dynamic WACC, falling back to a trailing
P/E identity when the info dict/financials statement don't carry enough to
drive the DCF.

DataFrames use yfinance orientation: row index = line items, columns =
periods, newest first. `info` is the yfinance-style ticker info dict.
"""

from dataclasses import dataclass

import pandas as pd

EQUITY_RISK_PREMIUM = 0.055
LONG_TERM_GROWTH = 0.025
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_TAX_RATE = 0.21
DEFAULT_BETA = 1.0
DEFAULT_COST_OF_DEBT = 0.05
PROJECTION_YEARS = 10
SHORT_TERM_YEARS = 5
MAX_HISTORICAL_GROWTH = 0.20
DEFAULT_GROWTH = 0.05


@dataclass
class ValuationResult:
    fair_value: float | None
    upside_pct: float | None
    method: str


def _line(df: pd.DataFrame, item: str, period: int = 0) -> float | None:
    try:
        value = df.loc[item].iloc[period]
    except (KeyError, IndexError):
        return None
    return None if pd.isna(value) else float(value)


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _current_price(info: dict[str, object]) -> float | None:
    return _as_float(info.get("currentPrice")) or _as_float(info.get("regularMarketPrice"))


def _tax_rate(financials: pd.DataFrame) -> float:
    pretax_income = _line(financials, "Pretax Income")
    if pretax_income is None or pretax_income <= 0:
        return DEFAULT_TAX_RATE
    tax_provision = _line(financials, "Tax Provision") or 0.0
    return abs(tax_provision / pretax_income)


def _wacc(info: dict[str, object], financials: pd.DataFrame) -> tuple[float | None, float]:
    """Returns (wacc, total_debt); wacc is None if it can't be computed."""
    market_cap = _as_float(info.get("marketCap"))
    if not market_cap:
        return None, 0.0

    total_debt = _as_float(info.get("totalDebt")) or 0.0
    total_value = market_cap + total_debt
    if total_value == 0:
        return None, total_debt

    beta = _as_float(info.get("beta")) or DEFAULT_BETA
    cost_of_equity = DEFAULT_RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

    interest_expense = abs(_line(financials, "Interest Expense") or 0.0)
    cost_of_debt = (interest_expense / total_debt) if total_debt > 0 else DEFAULT_COST_OF_DEBT

    tax_rate = _tax_rate(financials)

    wacc = ((market_cap / total_value) * cost_of_equity) + (
        (total_debt / total_value) * cost_of_debt * (1 - tax_rate)
    )
    if pd.isna(wacc):
        return None, total_debt
    return wacc, total_debt


def _historical_growth(financials: pd.DataFrame) -> float | None:
    current = _line(financials, "Total Revenue", 0)
    previous = _line(financials, "Total Revenue", 1)
    if current is None or previous is None or previous <= 0:
        return None
    return (current - previous) / previous


def _multistage_dcf(info: dict[str, object], financials: pd.DataFrame) -> ValuationResult | None:
    wacc, total_debt = _wacc(info, financials)
    if wacc is None or wacc <= LONG_TERM_GROWTH:
        return None

    last_fcf = _as_float(info.get("freeCashflow"))
    if last_fcf is None or last_fcf <= 0:
        return None

    historical_growth = _historical_growth(financials)
    short_term_growth = (
        min(historical_growth, MAX_HISTORICAL_GROWTH)
        if historical_growth is not None
        else DEFAULT_GROWTH
    )

    fcf = last_fcf
    projected_fcf: list[float] = []
    for year in range(1, PROJECTION_YEARS + 1):
        growth_rate = (
            short_term_growth
            if year <= SHORT_TERM_YEARS
            else LONG_TERM_GROWTH
            + (short_term_growth - LONG_TERM_GROWTH) * ((PROJECTION_YEARS - year) / 5.0)
        )
        fcf *= 1 + growth_rate
        projected_fcf.append(fcf)

    terminal_value = (projected_fcf[-1] * (1 + LONG_TERM_GROWTH)) / (wacc - LONG_TERM_GROWTH)
    pv_fcf = sum(cf / ((1 + wacc) ** (i + 1)) for i, cf in enumerate(projected_fcf))
    pv_terminal = terminal_value / ((1 + wacc) ** PROJECTION_YEARS)
    enterprise_value = pv_fcf + pv_terminal

    cash = _as_float(info.get("totalCash")) or 0.0
    equity_value = enterprise_value + cash - total_debt

    shares_outstanding = _as_float(info.get("sharesOutstanding"))
    if not shares_outstanding:
        return None

    fair_value = equity_value / shares_outstanding
    current_price = _current_price(info)
    if current_price is None or current_price == 0:
        return None

    upside_pct = ((fair_value - current_price) / current_price) * 100
    return ValuationResult(fair_value=fair_value, upside_pct=upside_pct, method="multistage_dcf")


def _trailing_pe_fallback(info: dict[str, object]) -> ValuationResult:
    eps = _as_float(info.get("trailingEps"))
    pe = _as_float(info.get("trailingPE"))
    current_price = _current_price(info)

    if eps is None or pe is None or current_price is None or current_price == 0:
        return ValuationResult(fair_value=None, upside_pct=None, method="insufficient_data")

    fair_value = eps * pe
    upside_pct = ((fair_value - current_price) / current_price) * 100
    return ValuationResult(fair_value=fair_value, upside_pct=upside_pct, method="trailing_pe")


def estimate(info: dict[str, object], financials: pd.DataFrame) -> ValuationResult:
    dcf_result = _multistage_dcf(info, financials)
    if dcf_result is not None:
        return dcf_result
    return _trailing_pe_fallback(info)
