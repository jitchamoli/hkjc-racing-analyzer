# ML Model

This is a current operating reference. The code is authoritative: the feature
names and order come from `ml/feature_schema.py`; training and inference
behavior come from `ml/training.py`, `ml/features.py`, `ml/folds.py` and
`ml/model.py`. Historical experiment outcomes belong in dated audit documents.

## Production Contract

- The current schema has 23 features:
  `prior_win_rate`, `recent_form`, `jockey_win_rate`, `trainer_win_rate`,
  `field_size`, `field_rel_draw`, `cd_form_prior`, `class_level`, `freshness`,
  `going_performance`, `weight_vs_field`, `jockey_horse_coop`,
  `last_start_margin`, `class_change`, `horse_weight_change`, `horse_age`,
  `rating_change`, `jockey_form_30d`, `jt_pair_win_rate`, `early_speed`,
  `late_kick`, `mid_race_pace`, `sire_top3_rate_on_going`.
- Feature order, defaults and ranges are validated. Change the shared schema and
  train/inference extraction together; do not manually maintain a second
  feature list.
- Features must use only information available before the target race. Group
  all runners from a meeting/race together when splitting data. Fit calibrators
  only on their designated calibration data, separate from final evaluation.
- Production inference fails closed when model integrity checks fail. Preserve
  the model metadata and matching `.sha256` sidecar when publishing artifacts.
- Model probabilities and betting ranks are different contracts: final betting
  rank follows `calibrated_prob`; do not alter production ranks to test a new
  ranking objective. Evaluate such objectives in shadow/replay first.

## Training and Validation

Use the repository entry point and check its output before replacing a
production artifact:

```bash
python3 train_ml_model.py
python3 -m scripts.validate_hk --report --replay
```

The exact cutoff, grouped-fold implementation, artifact metadata and calibration
lifecycle can change. Inspect `ml/folds.py`, `ml/training.py`, `ml/model.py`,
`ml/integrity.py`, and the relevant training script rather than relying on old
split percentages or model-version notes.

For production comparisons, prefer timestamped forward predictions and frozen
`race_bet_ticket` replay. A historical top-N capture diagnostic is not frozen
ticket ROI. Backtests using closing odds should be treated as optimistic unless
the wager price was recorded at the actual decision time.

## Adding or Removing a Feature

1. Establish the signal and time-availability from source data.
2. Verify train/inference parity, null/default behavior and range constraints.
3. Use grouped race-level folds and preserve a genuinely untouched evaluation
   window.
4. Compare against the current model with walk-forward top-four capture,
   probability calibration and frozen-ticket replay.
5. Keep a candidate shadow-only until evidence across multiple meetings supports
   deployment. Do not tune and report on the same holdout repeatedly.

Do not restore removed features solely because an older audit reported a
promising correlation or small-sample ROI. See `docs/audits/` for dated evidence
and `docs/top4-shadow-research.md` for the current shadow-objective research.
