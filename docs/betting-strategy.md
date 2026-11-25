# Betting, Recommendations and Validation

This is an operating reference, not a claim that the system is profitable.
There is no established positive expected-value strategy. Keep recommendations
as decision support and validate with forward data before changing production
rules or stakes.

## Probability and Rank

- The scoring pipeline combines the model's independent opinion with the
  market, applies calibration, then normalizes race probabilities. See
  `core/scorer.py` for the current blend and calibrator behavior.
- `rank` is betting rank and follows descending `calibrated_prob`, with an
  explicit deterministic tie-break. Do not promote a market favourite or
  BANKER after calibration.
- Edge calculations use the model-side `pure_prob` against normalized implied
  market probabilities. This is a signal for analysis, not proof of a
  profitable price; stale odds and takeout can erase apparent value.

## Current Ticket Policy

The frozen ticket is created in `dashboard/generator.py::build_bet_ticket`.
That function and the associated tests are authoritative for exact gates.

- Top-three QIN PLACE box coverage is generated for each unresolved race unless
  the bankroll halt is active.
- F4 saver and F4 BOX-5 are limited to EDGE/HIGH and apply field/price gates.
- BOX TRIO-5 is limited to EDGE/HIGH/MED and a sufficient field. BOX TRIO-6 is
  paper-only.
- Primary QIN/WIN/PLACE/TRIO recommendations and QIN protection follow the
  recommender and ticket gates; a `SKIP` label does not remove separately
  configured ticket coverage.
- A suggested stake is not confirmation that a wager was placed. The database
  records generated recommendations; bankroll reporting therefore measures
  recommended exposure unless actual purchases are separately recorded.
- A realized daily-loss threshold is currently a warning by user policy
  (`BANKROLL_LOSS_BLOCKS_BETS=False`); it does not remove suggested bets.
  Card-level exposure above the configured cap is disclosed for manual review
  and does not itself delete recommendations. If either policy changes, update
  this reference and its dashboard/ticket tests together.

Never describe all listed ticket items as bets actually purchased. Check
`paper`, `bankroll_halt`, unit stake, combo count and allocation in the frozen
ticket.

## Evaluation

```bash
python3 -m scripts.validate_hk --report --replay
python3 -m scripts.calibration_report
python3 -m scripts.clv_report
```

- Frozen-ticket replay evaluates the saved ticket and captured price basis.
- Synthetic top-N backtests are counterfactual diagnostics, not evidence of
  actual ticket ROI.
- Closing-odds backtests can overstate attainable returns. Use timestamped
  pre-off odds and CLV to evaluate price quality.
- Report hit rate alongside ROI, sample size, meeting-level uncertainty and
  calibration. A hit-rate increase alone does not imply positive ROI.
- Keep new ranking objectives and features shadow-only until grouped
  walk-forward evaluation improves top-four capture and frozen-ticket results
  across multiple meetings.

Avoid repeated threshold sweeps on a previously inspected holdout. Record
candidate definitions before evaluating them and reserve future meetings for
confirmation. Dated result tables in `docs/audits/` are historical measurements,
not current production estimates.
