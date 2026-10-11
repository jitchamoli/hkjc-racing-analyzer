# Model Comparison Baseline Correction

## Confirmed Comparison Problems

The original five-race comparison contrasted two retrospective model bundles:
20261011-202622 (before population filtering) and 20261011-205325 (after).
Those results remain a valid identical-input comparison of those bundles,
but neither is the model recorded in the actual pre-race tickets.

All October 11 tickets record model_version=20261008-192348. The ticket
timestamps are 12:22:13 for R1 and 12:57:31 for R2-R11. The before-filtering
artifact was trained at 20:26, after the races. HEAD's preceding committed
artifact is 20261004-185504, which also does not match the recorded version.
No matching October 8 artifact was found among the local models-directory
backups inspected. Other archives have not been exhaustively searched.

Maximum absolute errors between the reconstructed before-filtering model and
saved probabilities are 0.149779 raw ML, 0.070528 calibrated ML, and 0.055718
final probability. The October 4 bundle also fails replay parity. Missing
inputs or database-history revisions are possible additional explanations,
but have not been established as the cause: using the wrong model version
alone prevents an exact replay claim.

## Correct First-Five Baselines

Order-independent actual first-five overlap, October 11 R7-R11:

| Baseline | R7 | R8 | R9 | R10 | R11 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Saved race-day predictions | 3 | 3 | 2 | 2 | 5 | 15/25 (60%) |
| Before-filtering retrospective retrain | 3 | 3 | 3 | 3 | 5 | 17/25 (68%) |
| Corrected retrospective retrain | 3 | 3 | 3 | 2 | 5 | 16/25 (64%) |

The saved R10 top five are [2,10,3,7,5]. The corrected retrain has the same
five horses. The before-filtering retrain substitutes #9 for #5. Therefore
the #9-to-#5 swap was not a change from the original race-day predictions.
Do not interpret 60% versus 64% as proof of a deployable improvement either:
this is one meeting, post-race research, and reconstructed inputs.

## Attribution Limits

Population filtering changed fold boundaries as well as sample eligibility:
the training endpoint moved from 2024/10/23 R2 to 2024/10/13 R6; calibration
ended 2026/01/07 R7 before versus 2026/01/18 R11 after. Model parameters,
ML isotonic calibration and blend calibration were refitted. This experiment
cannot isolate overseas-row removal from incomplete-field quarantine or
changed temporal boundaries.

Historical feature helpers still query the full horse-history database.
In particular, the running-position helper derives a historical field size
with date/race-number COUNT(*) without a venue filter (ml/features.py).
This remains a residual integrity risk for colliding or incomplete historical
fields; it is not established as the cause of the R10 difference. Both arms
used the same reconstructed vectors in this comparison.

## Next Verification

- Recover the exact October 8 model and matching calibrators, then require
  saved raw/calibrated probability parity before calling a replay exact.
- Use identical calendar cutoffs for a larger controlled ablation: original
  eligibility, local-only eligibility, and local complete-field eligibility.
- Keep field-level source snapshots and artifact hashes for future replay.
- Audit historical field-size helpers against local race identity, with
  regression tests before changing historical feature behavior.

No model, database, predictions or ticket records were changed by this audit.
