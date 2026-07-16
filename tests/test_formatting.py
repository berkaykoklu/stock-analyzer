import pytest

from stock_analyzer.formatting import format_large_number


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "N/A"),
        ("not-a-number", "N/A"),
        (float("nan"), "N/A"),
        (float("inf"), "N/A"),
        (float("-inf"), "N/A"),
        (2_500_000_000_000, "$2.50T"),
        (3_400_000_000, "$3.40B"),
        (-3_400_000_000, "-$3.40B"),
        (7_250_000, "$7.25M"),
        (12_500, "$12.50K"),
        (999.4, "$999.40"),
        (-42.5, "-$42.50"),
    ],
)
def test_format_large_number(value, expected):
    assert format_large_number(value) == expected
