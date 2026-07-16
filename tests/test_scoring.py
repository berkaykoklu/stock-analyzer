import pytest

from stock_analyzer.scoring import (
    WEIGHTS,
    ComponentScore,
    composite_score,
    quality_component,
    technical_component,
    valuation_component,
)


def _c(score: float, available: bool = True) -> ComponentScore:
    return ComponentScore(score=score, available=available)


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_all_available_weighted_mean():
    components = {name: _c(80.0) for name in WEIGHTS}
    result = composite_score(components)
    assert result.score == pytest.approx(80.0)
    assert result.coverage == pytest.approx(1.0)


def test_missing_components_renormalize():
    components = {name: _c(0.0, available=False) for name in WEIGHTS}
    components["fundamental"] = _c(90.0)
    components["technical"] = _c(60.0)
    result = composite_score(components)
    # renormalized: (90*.30 + 60*.20) / (.30+.20) = 78.0
    assert result.score == pytest.approx(78.0)
    assert result.coverage == pytest.approx(0.50)


def test_nothing_available_scores_zero_coverage():
    components = {name: _c(0.0, available=False) for name in WEIGHTS}
    result = composite_score(components)
    assert result.coverage == 0.0


@pytest.mark.parametrize(
    ("rsi", "trend_score", "macd", "expected"),
    [
        # RSI in [30, 70] -> 30, else 15; trend contributes (trend_score/3)*40; macd>0 -> 30 else 10
        (30.0, 0.0, 1.0, 30 + 0.0 + 30),  # rsi lower boundary in-range
        (70.0, 0.0, 1.0, 30 + 0.0 + 30),  # rsi upper boundary in-range
        (29.99, 0.0, 1.0, 15 + 0.0 + 30),  # just below in-range
        (70.01, 0.0, 1.0, 15 + 0.0 + 30),  # just above in-range
        (50.0, 3.0, 1.0, 30 + 40.0 + 30),  # max trend_score
        (50.0, 0.0, 0.0, 30 + 0.0 + 10),  # macd boundary (not > 0) -> 10
        (50.0, 0.0, 0.01, 30 + 0.0 + 30),  # macd just above 0 -> 30
    ],
)
def test_technical_component(rsi, trend_score, macd, expected):
    assert technical_component(rsi, trend_score, macd) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("upside_pct", "expected"),
    [
        (25.01, 95),  # just above 25
        (25.0, 85),  # boundary: not > 25, falls to next bracket
        (15.01, 85),  # just above 15
        (15.0, 75),  # boundary: not > 15
        (5.01, 75),  # just above 5
        (5.0, 60),  # boundary: not > 5
        (-4.99, 60),  # just above -5
        (-5.0, 40),  # boundary: not > -5
        (-14.99, 40),  # just above -15
        (-15.0, 20),  # boundary: not > -15
        (-50.0, 20),  # well below
    ],
)
def test_valuation_component(upside_pct, expected):
    assert valuation_component(upside_pct) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("moat_score", "expected"),
    [
        (0.0, 0.0),
        (5.0, 50.0),
        (10.0, 100.0),
    ],
)
def test_quality_component(moat_score, expected):
    assert quality_component(moat_score) == pytest.approx(expected)
