# Top-Four Shadow Research and Calibration Tracking

## Intent and Scope

Diagnose whether training for first-four capture improves the frozen betting
shortlist, and monitor win calibration by probability band and conviction tier.
Production models, calibrators, rankings, thresholds and tickets are unchanged.
This is research, not evidence of profitable deployment.

## Delivery Boundaries

1. Calibration reporting extends the existing forward/replay reporting flow.
2. A separately invoked walk-forward experiment compares a top-four-aware
   ranker against an identically trained win classifier. Expensive training
   must not run automatically during `raceday live` or `raceday finish`.
3. Prospective shadow scoring can be wired only after the historical diagnostic
   passes data-integrity checks. Historical reconstructed bets are never logged
   as actual frozen production tickets.

## Calibration Records

Read pre-race probability, runner number, race tier and model version from
frozen prediction records and join official outcomes by meeting/race/runner.
The saved prediction timestamp must precede scheduled HKT off-time. Exclude
explicitly post-off-unverified races and races without verifiable timestamps,
logging exclusion counts rather than silently accepting them.

Use all eligible runners and a separate rank-one view. Validate finite
probabilities in [0,1]; do not renormalize malformed records into validity.
Duplicate records have one deterministic canonical prediction per runner.
Exclude withdrawn/non-starting runners from binary calibration. Require
complete official outcomes rather than treating absent runners as losers.
For winner dead heats, use fractional win credit across tied winners and label
the reporting convention; do not silently count multiple full wins.

Fixed probability bands: [0,.05), [.05,.10), [.10,.20), [.20,.30),
[.30,.50), [.50,1], crossed with recorded conviction tier.
Report meetings, races, runners, mean prediction, observed win frequency,
calibration gap, Brier score and multiclass race log loss on eligible complete
fields. Report log loss only with explicit normalization validation and zero
probability clipping for numerical evaluation, not probability correction.
Meeting-block bootstrap intervals use a fixed seed. Sparse groups are marked
insufficient evidence; a meeting count alone is not a promotion gate.

Persist an idempotent research log and an atomic report under tracking_data.
Refresh as a nonfatal reporting stage after results ingestion in post_race.
Record exclusions and provenance in each report. No automatic blend changes.

## Ranking Diagnostic

Reuse production feature definitions and race eligibility rules. Extend or
wrap the training builder with explicit runner IDs and finish positions;
never align labels through incidental database row order. Existing callers
retain their current return contract.

Use the installed XGBoost ranking implementation with a fixed
`rank:ndcg` objective and binary relevance for finish_position <= 4.
This is a top-four-aware ranking surrogate, not a direct set likelihood or
calibrated win probability. Rank scores must never enter Kelly or value math.
Use the same feature set and conservative fixed hyperparameters for the
classifier comparison; do not tune on evaluation folds.

Split by whole meeting date, strictly chronological expanding windows.
Training, early stopping and any calibration data precede test meetings.
Race groups remain intact. Recency weights are computed relative to training
data only. Features must use information available at the historical bet
moment; closing odds, current-race outcomes and retrospectively updated
statistics cannot enter the input vector. Audit feature availability before
running folds; fail the diagnostic if the contract cannot be established.

Historical results provide capture comparisons and clearly labeled
counterfactual returns using official dividends, not executable historical
price claims. Compare against the retrained fold classifier, not today's
production pickle, which has seen historical outcomes.

For dates with eligible frozen tickets, report the actual production replay
separately. Counterfactual comparison retains each ticket's pool, box size,
unit stake and paper designation, replacing only applicable ranked legs.
If a bet cannot be mapped without reconstructing unsupported gate decisions,
exclude it from both paired arms and report why. Gate thresholds stay fixed.

Measure winner top-one accuracy, winner-in-five, complete top-three-in-five,
complete top-four-in-five and paired TRIO/F4 cost, payout, P&L and ROI.
Use the existing official dividend normalization and dead-heat settlement
helpers. Withdrawn-runner combinations need correct refund settlement; until
available, exclude affected paired races rather than report inflated losses.
Missing dividends remain unpriced, never silently zero-priced wins.

## Evidence and Outputs

Write per-race provenance, fold dates, model parameters, sample counts,
eligible/excluded populations and paired results to research-only files.
Meeting-block bootstrap paired differences for capture, P&L and ROI.
Clearly distinguish historical research, frozen production replay and future
pre-off shadow tickets. Do not emit SHIP from cumulative meeting win counts.
At least six prospective meetings are a minimum review floor, not proof of
edge; follow the project's broader forward-validation policy before promotion.

## Verification

- Train/test meeting separation, intact race groups and feature/label ID alignment.
- No current-race outcomes in historical features; fold-local recency weights.
- Production artifacts and ticket rows unchanged by research execution.
- Calibration band boundaries, missing/nonfinite probabilities, unresolved
  runners, duplicate records, withdrawals, dead heats and HKT time checks.
- Idempotent report updates and atomic persistence.
- Pairing excludes unsupported/missing settlement symmetrically.
- Synthetic cases with known top-four sets and dividends establish expected
  capture and returns before a real-data diagnostic.
- Focused tests plus the full existing suite before completion.

## Execution Sequence

First deliver calibration reporting and tests. Then implement the runner-keyed
research data contract and leakage audit. Finally run the walk-forward ranker
and report evidence; keep or kill the candidate without modifying production.
