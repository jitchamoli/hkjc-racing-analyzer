# Fixed-Cutoff Eligibility Ablation

## Method

Three isolated ML classifiers used identical calendar boundaries:
training through 2024/10/13, early-stopping validation through 2025/03/30,
calibration through 2026/01/18, and common eligible test races thereafter.
All arms scored the same 7,434 clean runner vectors across 600 races and
60 meetings. Production XGBoost parameters were retained; uniform training
weights removed population-dependent recency weighting as a confound.
ML isotonic calibration was refit per arm on the fixed calibration period.
No rule/market blending or recommender gates were used. This is an ML-only
research ablation, not a production dashboard backtest or frozen-ticket replay.

The original arm reproduces HEAD's training eligibility. The local-only arm
excludes overseas rows before field aggregates but retains the old incomplete
field policy. The local-complete arm uses the repaired training eligibility.
Historical horse-form queries are unchanged in all three arms.

## Results

| Metric | Original | Local Only | Complete Local |
|---|---:|---:|---:|
| First-five overlap, ignoring order | 56.67% | 56.77% | 56.90% |
| Complete first-five set | 2.33% | 2.17% | 1.50% |
| Actual first four inside predicted five | 9.67% | 10.17% | 9.67% |
| First-four overlap within predicted four | 50.58% | 50.04% | 50.12% |
| Winner inside predicted five | 66.67% | 67.33% | 66.83% |
| Hypothetical TRIO-5 ROI | -31.41% | -32.83% | -42.46% |
| Hypothetical F4-5 ROI | -12.95% | +8.88% | -11.42% |

ROI uses declared dividends, HK$10 per combination, and identical fixed boxes.
It is not ROI of original recommended tickets. Zero/null-position races are
excluded from settlement where identifiable; completely missing withdrawals
cannot be detected by that database check. All 600 evaluated races had declared
TRIO and FIRST 4 dividends. No price-impact or recommender stake model is used.

Meeting-block bootstrap 95% intervals for first-five-overlap improvement over
original include zero for both alternatives. This experiment does not establish
a reliable first-five accuracy improvement or degradation. Payout outcomes are
volatile; the local-only positive F4 ROI is not evidence for production promotion.
Historical test races have previously been researched, so this is not pristine
forward validation or a newly untouched holdout.

## Artifact Recovery

October 11 frozen tickets record model version 20261008-192348. It was not found
in inspected local model backups, repository artifact history or the searched
temporary paths. The trainer removes its .bak after successful replacement
(train_ml_model.py:194). That is a reproducibility gap, not evidence that the
race-day model itself was corrupt. Recovery from external backups remains open.
Without that artifact and matching calibrators, exact race-day replay is blocked.

## Conclusion

Exclude overseas rows from local race fields: verified race identity requires it.
Complete-field quarantine is defensible for malformed field aggregates, but this
ablation does not prove it increases returns. Backfill missing local finishers to
recover legitimate training information rather than restoring malformed groups.
Do not promote a candidate or change betting thresholds from this diagnostic.
Versioned model/calibrator archives and saved feature inputs are needed for
future exact replay. Historical field-size feature helpers still need a separate
venue-aware audit and regression tests.

## Verification

- Production .pkl/.sha256 artifacts and October 11 prediction JSON hashes unchanged.
- No trained research model saved; no database or frozen tickets rewritten.
- Focused research/integrity tests: 22 passed.
- Full per-race rankings, cutoffs, sample counts and intervals are in the companion JSON.
- Research runner: /tmp/hkjc_fixed_cutoff_ablation.py (temporary, not production code).
