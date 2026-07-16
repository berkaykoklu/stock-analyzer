"""Sector/macro context: interest-rate sensitivity classification and a
cyclical-vs-defensive flag.

The rate-sensitivity map and interpretation thresholds are ported unchanged
from legacy `macro.py`. `cyclical` has no legacy analog (legacy only
classified interest-rate sensitivity, a different axis than business-cycle
sensitivity); it is a new, disclosed addition using a standard cyclical
sector set, since the target `MacroResult` requires it.
"""

from dataclasses import dataclass

RATE_SENSITIVITY: dict[str, float] = {
    "Technology": -0.5,  # Growth stocks sensitive to rates
    "Real Estate": -1.2,  # REITs very sensitive
    "Utilities": -0.8,  # Dividend stocks sensitive
    "Financial": 0.6,  # Banks benefit from higher rates
    "Energy": -0.2,  # Less sensitive
    "Healthcare": -0.3,  # Defensive sector
    "Consumer": -0.4,  # Mixed sensitivity
}
DEFAULT_RATE_SENSITIVITY = -0.3

CYCLICAL_SECTORS = {
    "Technology",
    "Financial",
    "Financial Services",
    "Consumer Cyclical",
    "Industrials",
    "Basic Materials",
    "Energy",
    "Real Estate",
}


@dataclass
class MacroResult:
    sector: str
    cyclical: bool
    notes: str


def _interpret_rate_sensitivity(sensitivity: float) -> str:
    if sensitivity < -0.8:
        return "Very sensitive to rate increases (negative impact)"
    elif sensitivity < -0.4:
        return "Moderately sensitive to rate increases"
    elif sensitivity < 0.2:
        return "Low sensitivity to rate changes"
    else:
        return "Benefits from rate increases"


def context(info: dict[str, object]) -> MacroResult:
    sector_raw = info.get("sector", "")
    sector = sector_raw if isinstance(sector_raw, str) else ""

    sensitivity = RATE_SENSITIVITY.get(sector, DEFAULT_RATE_SENSITIVITY)
    interpretation = _interpret_rate_sensitivity(sensitivity)
    notes = f"Rate sensitivity {sensitivity:+.1f}: {interpretation}"

    return MacroResult(
        sector=sector,
        cyclical=sector in CYCLICAL_SECTORS,
        notes=notes,
    )
