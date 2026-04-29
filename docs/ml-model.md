# ML Model — Detailed Reference

Companion to `CLAUDE.md`. Holds the feature inventory, removed-features changelog, class-extraction helper, training pipeline detail, and deprecated-component notes. The 8 ML Critical Rules and the high-level scoring summary stay in `CLAUDE.md`.

## Current 19 ML Features (v6.5 — all prior-race only)

Features 1–8: `prior_win_rate, recent_form, draw_advantage, jockey_win_rate, trainer_win_rate, field_size, field_rel_draw, cd_form_prior`

Features 9–16: `class_level` (Group=1.0, Class1=0.8, Class5=0.0), `freshness, going_performance, weight_vs_field, jockey_horse_coop, last_start_margin, class_change, horse_weight_change`

Features 17–19 (v6.5 additions): `horse_age` (3yo→1.0, 9yo+→0.0), `rating_change` (+10pts→1.0, 0→0.5, -10→0.0), `jockey_form_30d` (Bayesian-smoothed 30-day strike rate)

Feature importance on the v6.5 50K-row training (none zero):
- `last_start_margin` 15.8%, `jockey_win_rate` 8.4%, `recent_form` 8.3%, `field_rel_draw` 5.7%, `class_level` 5.0%, **`rating_change` 4.6%**, `prior_win_rate` 4.6%, **`jockey_form_30d` 4.5%**, `going_performance` 4.3%, `field_size` 4.2%, `trainer_win_rate` 4.2%, `horse_weight_change` 4.2%, `weight_vs_field` 4.1%, `jockey_horse_coop` 4.0%, `cd_form_prior` 3.9%, `freshness` 3.9%, `draw_advantage` 3.6%, `class_change` 3.5%, **`horse_age` 3.2%**

**Caveat (worth tracking)**: `jockey_win_rate × jockey_form_30d` correlation = 0.750 (just over Wave 2c's 0.72 drop threshold). Both jockey signals; XGBoost handles correlated trees fine. Worth revisiting in a v6.6 cleanup pass.

## Removed-features changelog

- **v6.4 (2026-04-25)**: `prior_pace, adjusted_speed, distance_pb_ratio, speed_consistency` — all 0.0 importance after the v6.3 → v6.4 bug fixes; redundant with `last_start_margin`.
- **Wave 2c**: `field_rel_speed` (r=0.72 with `recent_form`), `days_since_last_win` (r=0.74 with `recent_form`).
- **Earlier audits**: `odds_implied_prob`, `venue_win_rate` (r=0.984), `weight_carried_norm` (r=0.877), `draw_field_interact` (r=0.934), `prior_avg_position`, `prior_speed` (replaced by `adjusted_speed` before `adjusted_speed` itself was dropped), `late_money_signal` (constant 0.5 — train/predict distribution mismatch), `jockey_trainer_combo`.

## Class extraction helper (`ml/features.py:extract_class_rank`)

`re.search(r'(\d)', class_type)` was buggy — turned `'G1'` into class 1, but Group races sit ABOVE the digit ladder. v6.4 introduces `extract_class_rank()`:

| Class string | Rank |
|---|---|
| Group 1 / Group 2 | 0 (highest) |
| Group 3 | 1 (≈ top of class 1 in difficulty) |
| Class 1–5 | 1–5 |
| GRIFFIN, MAIDEN | 4 (restricted, lower-tier) |
| 4YO / age-restricted | 3 |
| Empty/unparseable | None (caller defaults to 3) |

Both training and inference share this helper. The `class_level` feature uses it for current-race rank; `class_change` uses it for current-vs-prior rank comparison.

## Training Pipeline

- **5-year window** (default, configurable via `TRAINING_YEARS_BACK` env var; set 0 to disable). Empirically: training on 49y of legacy data dropped ROI ~30% vs the recent 5y window. Older eras have different jockey rosters, rule sets, and pool dynamics that confuse the model. Retraining: `TRAINING_YEARS_BACK=5 SKIP_FETCH=1 python3 train_ml_model.py` (SKIP_FETCH bypasses the slow Playwright meeting fetch when DB already has what we need).
- **60/20/20 temporal split**: Train on first 60%, validate (early stopping + isotonic calibration) on next 20%, evaluate on final 20%. Test metrics are computed through the calibrator but the calibrator was never fitted on test data — metrics are unbiased.
- **Exponential recency weighting** (v6.4 — was linear): `w(i) = 0.25 + 0.75·exp((i-(n-1))/(n/3))`. Newest row weight 1.0, oldest ~0.29. With 50% jockey turnover and 94% horse turnover since 2021, older rows describe a different field; exponential decay halves their influence twice as fast as the prior linear schedule.
- **Field clipping** (v6.4 fix): training-side `field_size` and `field_rel_draw` are now clipped to [0,1]. Inference (features.py) already clipped via F-16; training (training.py) was missing the clip, so the model trained on values up to 15.0 from oversize legacy fields.
- **Freshness fallback** (v6.4 fix): when DB column `days_since_last_run` is null/zero (87.7% of legacy rows), training now falls back to `(race_date - prior race_date)` math — matching the inference-side fallback. Without this, freshness was a near-constant 0.5 in training and got 0.0 importance in v6.3.
- **Isotonic calibration** (Wave 2c switch from Platt): `IsotonicRegression(out_of_bounds='clip')` fitted on validation-set predictions (logit-input) → actual outcomes. Recalibrates XGBoost's overconfident probabilities. Fitted on validation (not test) so test metrics remain unbiased. The val slice currently spans 2024-04 → 2025-04 (1y stale on field strength) — known structural trade-off, not addressed in v6.4.
- **Place model**: Separate XGBoost trained on `is_top3` labels alongside the win model. Used for place bet probability and quinella placer-boost pricing.
- **scale_pos_weight**: Computed as `n_neg / n_pos` (≈ 11 with 8.3% positive class) and passed to XGBClassifier. Without this, the model would bias toward predicting "loss" everywhere (92% accuracy from doing nothing).
- **experiment_log integrity** (v6.4 fix): `log_experiment` is only called when `save=True`. Tests passing `save=False` were polluting `models/experiment_log.json` with n_samples=300/2000 dummy entries (399 polluted rows cleaned out, 5 real training runs retained).

## Speed Figures (`analytics/speed_figures.py`) — DEPRECATED in v6.4

The Beyer/Timeform-style adjusted speed figures are still computed and exposed via `analytics.speed_figures` for analyst use, but no longer fed into the ML model. The `adjusted_speed` and `speed_consistency` features were dropped in v6.4 because XGBoost ignored them in favor of `last_start_margin`. The `compute_par_times` precompute was removed from training.
