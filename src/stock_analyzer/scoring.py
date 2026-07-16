"""Composite investment scoring with weight renormalization.

Legacy substitutes 50 for missing components; this port instead renormalizes
weights over the components that are actually available and reports
`coverage` (the fraction of total weight backed by real data), so a score
built on 2 of 6 factors says so rather than pretending it's whole.
"""

from dataclasses import dataclass

WEIGHTS: dict[str, float] = {
    "fundamental": 0.30,
    "valuation": 0.25,
    "technical": 0.20,
    "quality": 0.15,
    "risk": 0.05,
    "sentiment": 0.05,
}


@dataclass(frozen=True)
class ComponentScore:
    score: float
    available: bool


@dataclass(frozen=True)
class CompositeResult:
    score: float
    components: dict[str, ComponentScore]
    coverage: float


def composite_score(components: dict[str, ComponentScore]) -> CompositeResult:
    live_weight = sum(WEIGHTS[k] for k, c in components.items() if c.available and k in WEIGHTS)
    if live_weight == 0:
        return CompositeResult(score=0.0, components=components, coverage=0.0)
    weighted = sum(
        c.score * WEIGHTS[k] for k, c in components.items() if c.available and k in WEIGHTS
    )
    return CompositeResult(
        score=weighted / live_weight, components=components, coverage=live_weight
    )


def technical_component(rsi: float, trend_score: float, macd: float) -> float:
    score = 30.0 if 30 <= rsi <= 70 else 15.0
    score += (trend_score / 3) * 40
    score += 30.0 if macd > 0 else 10.0
    return score


def valuation_component(upside_pct: float) -> float:
    if upside_pct > 25:
        return 95.0
    if upside_pct > 15:
        return 85.0
    if upside_pct > 5:
        return 75.0
    if upside_pct > -5:
        return 60.0
    if upside_pct > -15:
        return 40.0
    return 20.0


def quality_component(moat_score: float) -> float:
    return (moat_score / 10) * 100
