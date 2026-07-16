import math

import pandas as pd

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
