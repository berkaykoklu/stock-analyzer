# Roadmap

Where this project is heading, staged so each step is cheap, shippable, and
produces the evidence needed for the next decision. Written 2026-07-16;
statuses will be updated as milestones land.

**Positioning:** the second opinion that shows its work. One defensible
composite score, an ML signal with its backtest record beside it, and an
AI-written research note in which every number is traceable to real data.
Honesty about uncertainty and data coverage is the product's core
differentiator — every milestone below inherits that constraint.

## Stage 1 — The grounded research note (current repo, Streamlit)

### M2: ML signal, honestly reported
- Leakage-safe feature pipeline over OHLCV + fundamentals; walk-forward
  (time-series split) evaluation only — no shuffled CV on time series.
- Output is a direction probability over a stated horizon (e.g. "58%
  probability of beating SPY over 30 trading days"), never a buy/sell call.
- The signal is deliberately **separate from the composite score**: the
  composite is explainable fundamentals; the predictor is a statistical bet
  with different epistemics. A "Signal" panel always displays the model's
  backtest record (accuracy, calibration, sample size) beside the prediction.
- Dependencies stay light: sklearn first; a boosted-tree library is added
  only if it measurably beats it in the walk-forward eval.

### M3: AI narrative with a grounding contract
- An LLM turns the `AnalysisReport` + ML signal into a readable research
  note.
- **Grounding contract:** the prompt receives only the serialized report;
  every numeric claim in the output must be traceable to it. Enforced by an
  eval harness (number-faithfulness checks against golden reports) that runs
  in CI and ships with the feature — not optional polish.
- Notes are cached per ticker/day and generated on demand only, so LLM cost
  scales with real usage.

### M4: Ship and measure
- Deploy free on Streamlit Community Cloud with a prominent
  "educational tool, not investment advice" disclaimer, a feedback link, and
  minimal anonymous usage counting.
- Distribute deliberately: Show HN, relevant subreddit tool threads, X.
- **Gate to Stage 2:** evidence of repeat usage by strangers — on the order
  of 100+ distinct users with a meaningful return rate within about a month
  of sharing. Absent that, iterate on Stage 1 cheaply rather than building
  infrastructure nobody asked for.

## Stage 2 — Productionization (triggered only by the Stage 1 gate)

- **Architecture:** Next.js (App Router) on Vercel as the product frontend;
  this Python package wrapped in a thin FastAPI service (the analysis core
  stays Python — it is tested and correct). Streaming narrative via the
  AI SDK; Upstash Redis for report caching.
- **Data provider swap (hard prerequisite):** yfinance scrapes Yahoo and is
  not licensed for a public product. Stage 2 requires a licensed provider
  (Financial Modeling Prep / Polygon / Alpha Vantage — pick by price and
  fundamentals coverage at decision time). The `MarketData` Protocol exists
  for exactly this: a new provider is one new class; analysis code does not
  change.
- Production trimmings: rate limiting, error tracking, provider-cost
  monitoring. Accounts only when a feature genuinely needs them
  (e.g. watchlists), not before.

## Stage 3 — Monetization optionality (decide with data, not now)

If Stage 2 shows sustained growth: the free tier stays genuinely useful;
candidate paid features are the expensive ones (watchlist monitoring with
alerts, portfolio-level reports, deeper analysis). No commitment is made
here — this section exists so infrastructure choices don't foreclose the
option.

## Non-negotiables (every stage)

1. **Never imply advice.** Signals carry probabilities and backtest records;
   disclaimers stay prominent; no buy/sell language anywhere.
2. **Data licensing.** yfinance is acceptable for the free educational
   Stage 1; it is blocking for Stage 2 (see provider swap above).
3. **LLM grounding.** A hallucinated number in a financial note is the worst
   possible failure mode; the faithfulness eval harness ships with M3.
4. **Maintenance reality.** yfinance's schema drifts (this repo already hit
   renamed balance-sheet line items). CI plus a scheduled live-ticker smoke
   run catches drift early.
