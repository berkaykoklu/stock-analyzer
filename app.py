"""Streamlit UI for stock-analyzer: single-stock deep dive and a multi-ticker screener.

Renders `AnalysisReport` fields exclusively (see `stock_analyzer.analyzer`) — no
analysis is computed in this file. The one exception is the candlestick chart,
which pulls raw OHLCV via `YFinanceData` directly since that's data, not analysis.
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_analyzer import AnalysisReport, analyze_ticker
from stock_analyzer.data import DataUnavailableError, YFinanceData
from stock_analyzer.formatting import format_large_number
from stock_analyzer.technicals import compute_indicators

# Legacy screener list (LEGACY/config.py:SCREENER_TICKERS) had 2 duplicate
# tickers (UNH, JNJ) across categories; dedupe while preserving first-seen
# order since these become multiselect options.
_RAW_DEFAULT_TICKERS = [
    # Large Cap Tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "CRM",
    "NFLX",
    "ADBE",
    # Large Cap Traditional
    "BRK-B",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "MA",
    "UNH",
    "HD",
    "BAC",
    "KO",
    "DIS",
    "XOM",
    # Mid Cap Growth
    "ROKU",
    "TWLO",
    "ZM",
    "DDOG",
    "SNOW",
    "CRWD",
    "ZS",
    "OKTA",
    "NET",
    # Value & Dividend
    "WMT",
    "CVX",
    "PEP",
    "MCD",
    "COST",
    "T",
    "VZ",
    "IBM",
    "MMM",
    "CAT",
    # Healthcare & Biotech
    "PFE",
    "ABT",
    "TMO",
    "UNH",
    "JNJ",
    "MRNA",
    "GILD",
    "AMGN",
    "BMY",
    "LLY",
    # Financial
    "GS",
    "MS",
    "C",
    "WFC",
    "AXP",
    "BLK",
    "SCHW",
    "SPGI",
    "MCO",
    "ICE",
    # Industrial & Materials
    "BA",
    "HON",
    "UPS",
    "LMT",
    "RTX",
    "DE",
    "FCX",
    "NEM",
    "AA",
]
DEFAULT_TICKERS = list(dict.fromkeys(_RAW_DEFAULT_TICKERS))

_PERCENT_RATIOS = {"roa", "debt_ratio", "gross_margin", "roic", "roe", "revenue_growth"}


def _format_ratio(name: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    if name in _PERCENT_RATIOS:
        return f"{value:.1%}"
    return f"{value:.2f}x"


def _labeled_table(values: dict[str, float | None]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": name.replace("_", " ").title(), "Value": _format_ratio(name, value)}
            for name, value in values.items()
        ]
    )


def _render_candlestick(ticker: str) -> None:
    try:
        history = YFinanceData(ticker).history
    except Exception as exc:  # yfinance can raise assorted network/parsing errors
        st.warning(f"Could not load price history for chart: {exc}")
        return
    if history.empty:
        st.info("No price history available for chart.")
        return

    df = compute_indicators(history)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
            ),
            go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50", line={"width": 1}),
            go.Scatter(x=df.index, y=df["sma_200"], name="SMA 200", line={"width": 1}),
        ]
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend={"orientation": "h"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_fundamentals(report: AnalysisReport) -> None:
    st.subheader("Fundamentals")
    f = report.fundamentals
    st.metric(
        "Piotroski F-Score",
        f"{f.score}/{f.available}",
        help="Checks that could be computed from the available statements",
    )
    if f.checks:
        checks_df = pd.DataFrame(
            [{"Check": k.replace("_", " ").title(), "Passed": v} for k, v in f.checks.items()]
        )
        st.dataframe(checks_df, width="stretch", hide_index=True)
    st.dataframe(_labeled_table(report.ratios), width="stretch", hide_index=True)


def _render_technicals(report: AnalysisReport) -> None:
    st.subheader("Technicals")
    t = report.technicals
    col1, col2, col3 = st.columns(3)
    rsi_value = "N/A" if math.isnan(t["rsi"]) else f"{t['rsi']:.1f}"
    rsi_label = None if math.isnan(t["rsi"]) else report.rsi_label
    col1.metric("RSI", rsi_value, rsi_label)
    col2.metric("SMA 50", format_large_number(t["sma_50"]))
    col3.metric("SMA 200", format_large_number(t["sma_200"]))

    if report.support_resistance is not None:
        support, resistance = report.support_resistance
        support_str = format_large_number(support)
        resistance_str = format_large_number(resistance)
        st.write(f"**Support:** {support_str}  |  **Resistance:** {resistance_str}")
    else:
        st.caption("Support/resistance unavailable (insufficient price history).")


def _render_valuation(report: AnalysisReport) -> None:
    st.subheader("Valuation")
    v = report.valuation
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Fair Value", format_large_number(v.fair_value) if v.fair_value is not None else "N/A"
    )
    col2.metric("Upside", f"{v.upside_pct:+.1f}%" if v.upside_pct is not None else "N/A")
    col3.metric("Method", v.method)


def _render_quality(report: AnalysisReport) -> None:
    st.subheader("Quality / Moat")
    q = report.quality
    st.metric("Moat Score", f"{q.moat_score:.1f}/10")
    if q.factors:
        st.dataframe(_labeled_table(q.factors), width="stretch", hide_index=True)
    else:
        st.caption("No quality factors available for this ticker.")


def _render_risk(report: AnalysisReport) -> None:
    st.subheader("Risk")
    r = report.risk
    col1, col2, col3 = st.columns(3)
    col1.metric("Volatility (annualized)", f"{r.volatility:.1%}")
    col2.metric("Sharpe Ratio", f"{r.sharpe:.2f}" if r.sharpe is not None else "N/A")
    col3.metric("Max Drawdown", f"{r.max_drawdown:.1%}")


def _render_macro(report: AnalysisReport) -> None:
    st.subheader("Macro")
    m = report.macro
    col1, col2 = st.columns(2)
    col1.metric("Sector", m.sector or "Unknown")
    col2.metric("Cyclical", "Yes" if m.cyclical else "No")
    st.caption(m.notes)


def render_single_stock(report: AnalysisReport) -> None:
    st.header(f"{report.company_name} ({report.ticker})")
    if report.price is not None:
        st.caption(f"Price: {format_large_number(report.price)}")

    col1, col2 = st.columns(2)
    col1.metric("Composite Score", f"{report.composite.score:.0f}/100")
    col2.metric("Data Coverage", f"{report.composite.coverage:.0%}")

    _render_candlestick(report.ticker)
    _render_fundamentals(report)
    _render_technicals(report)
    _render_valuation(report)
    _render_quality(report)
    _render_risk(report)
    _render_macro(report)


def _screen_one(ticker: str) -> tuple[str, AnalysisReport | None, str | None]:
    try:
        return ticker, analyze_ticker(ticker), None
    except DataUnavailableError as exc:
        return ticker, None, str(exc)
    except Exception as exc:  # network/yfinance failures shouldn't abort the whole screen
        return ticker, None, str(exc)


def render_single_stock_tab() -> None:
    if "single_report" not in st.session_state:
        st.session_state.single_report = None

    ticker_input = st.text_input("Enter a stock ticker", value="AAPL").strip().upper()
    if st.button("Analyze", key="analyze_single") and ticker_input:
        with st.spinner(f"Analyzing {ticker_input}..."):
            try:
                st.session_state.single_report = analyze_ticker(ticker_input)
            except DataUnavailableError as exc:
                st.error(str(exc))
                st.session_state.single_report = None

    if st.session_state.single_report is not None:
        render_single_stock(st.session_state.single_report)


def render_screener_tab() -> None:
    selected = st.multiselect("Tickers to screen", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:10])
    if st.button("Run Screener", key="run_screener") and selected:
        rows: list[dict[str, object]] = []
        errors: list[tuple[str, str]] = []
        progress = st.progress(0.0)
        status = st.empty()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_screen_one, ticker): ticker for ticker in selected}
            for i, future in enumerate(as_completed(futures), start=1):
                ticker, report, error = future.result()
                if report is not None:
                    rows.append(
                        {
                            "Ticker": report.ticker,
                            "Name": report.company_name,
                            "Composite Score": round(report.composite.score, 1),
                            "Coverage": round(report.composite.coverage, 2),
                        }
                    )
                else:
                    errors.append((ticker, error or "Unknown error"))
                status.text(f"Screened {i}/{len(selected)}: {ticker}")
                progress.progress(i / len(selected))

        status.text(f"Done: {len(rows)}/{len(selected)} succeeded")

        if rows:
            results_df = (
                pd.DataFrame(rows)
                .sort_values("Composite Score", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(results_df, width="stretch")

        if errors:
            st.warning(f"{len(errors)} ticker(s) failed:")
            for ticker, error in errors:
                st.write(f"- **{ticker}**: {error}")


st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analyzer")

tab1, tab2 = st.tabs(["Single Stock", "Screener"])
with tab1:
    render_single_stock_tab()
with tab2:
    render_screener_tab()
