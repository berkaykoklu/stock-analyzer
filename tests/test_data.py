import pandas as pd

import stock_analyzer.data as data_module
from stock_analyzer.data import DataUnavailableError, YFinanceData


class _StubTicker:
    """Mimics the yfinance.Ticker surface YFinanceData relies on."""

    def __init__(
        self,
        *,
        history_df: pd.DataFrame,
        info: dict[str, object],
        financials: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> None:
        self._history_df = history_df
        self.info = info
        self.financials = financials
        self.balance_sheet = balance_sheet
        self.cashflow = cashflow

    def history(self, period: str = "2y") -> pd.DataFrame:
        return self._history_df


def test_is_valid_false_on_empty_history(monkeypatch):
    stub = _StubTicker(
        history_df=pd.DataFrame(),
        info={"regularMarketPrice": 100.0},
        financials=pd.DataFrame(),
        balance_sheet=pd.DataFrame(),
        cashflow=pd.DataFrame(),
    )
    monkeypatch.setattr(data_module.yf, "Ticker", lambda ticker: stub)

    data = YFinanceData("FAKE")

    assert data.is_valid is False


def test_properties_delegate_to_ticker(monkeypatch, history, financials, balance_sheet, cashflow):
    info: dict[str, object] = {"regularMarketPrice": 123.45}
    stub = _StubTicker(
        history_df=history,
        info=info,
        financials=financials,
        balance_sheet=balance_sheet,
        cashflow=cashflow,
    )
    monkeypatch.setattr(data_module.yf, "Ticker", lambda ticker: stub)

    data = YFinanceData("FAKE")

    pd.testing.assert_frame_equal(data.financials, financials)
    pd.testing.assert_frame_equal(data.balance_sheet, balance_sheet)
    pd.testing.assert_frame_equal(data.cashflow, cashflow)
    pd.testing.assert_frame_equal(data.history, history)
    assert data.info == info
    assert data.is_valid is True


def test_is_valid_true_with_current_price_only(monkeypatch, history):
    stub = _StubTicker(
        history_df=history,
        info={"currentPrice": 50.0},
        financials=pd.DataFrame(),
        balance_sheet=pd.DataFrame(),
        cashflow=pd.DataFrame(),
    )
    monkeypatch.setattr(data_module.yf, "Ticker", lambda ticker: stub)

    data = YFinanceData("FAKE")

    assert data.is_valid is True


def test_is_valid_false_when_info_missing_price_fields(monkeypatch, history):
    stub = _StubTicker(
        history_df=history,
        info={"shortName": "Fake Corp"},
        financials=pd.DataFrame(),
        balance_sheet=pd.DataFrame(),
        cashflow=pd.DataFrame(),
    )
    monkeypatch.setattr(data_module.yf, "Ticker", lambda ticker: stub)

    data = YFinanceData("FAKE")

    assert data.is_valid is False


def test_data_unavailable_error_is_exception():
    assert issubclass(DataUnavailableError, Exception)


def test_default_period_is_two_years(monkeypatch):
    captured: dict[str, str] = {}

    class _RecordingStub(_StubTicker):
        def history(self, period: str = "2y") -> pd.DataFrame:
            captured["period"] = period
            return self._history_df

    stub = _RecordingStub(
        history_df=pd.DataFrame({"Close": [1.0]}),
        info={},
        financials=pd.DataFrame(),
        balance_sheet=pd.DataFrame(),
        cashflow=pd.DataFrame(),
    )
    monkeypatch.setattr(data_module.yf, "Ticker", lambda ticker: stub)

    data = YFinanceData("FAKE")
    _ = data.history

    assert captured["period"] == "2y"
