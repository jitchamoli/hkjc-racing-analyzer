# ML Model — Detailed Reference

Companion to `CLAUDE.md`. Holds the feature inventory, removed-features changelog, class-extraction helper, training pipeline detail, and deprecated-component notes. The 8 ML Critical Rules and the high-level scoring summary stay in `CLAUDE.md`.

## Current 23 ML Features (v6.11 — all prior-race only)

Source of truth is `ml/feature_schema.py` — the list below mirrors that order.

Features 1–7: `prior_win_rate, recent_form, jockey_win_rate, trainer_win_rate, field_size, field_rel_draw, cd_form_prior`

Features 8–15: `class_level` (Group=1.0, Class1=0.8, Class5=0.0), `freshness, going_performance, weight_vs_field, jockey_horse_coop, last_start_margin, class_change, horse_weight_change`

Features 16–19: `horse_age` (3yo→1.0, 9yo+→0.0), `rating_change` (+10pts→1.0, 0→0.5, -10→0.0), `jockey_form_30d` (Bayesian-smoothed 30-day strike rate), `jt_pair_win_rate` (v6.6, jockey×trainer pairing)

Features 20–23: `early_speed` (v6.8 Path A, 1st-call rank), `late_kick` (v6.8 Path A, closing rate), `mid_race_pace` (v6.9, race-mean-normalised 800-400 minus last-400 split), `sire_top3_rate_on_going` (v6.10, Beta(2,8)-smoothed sire's top-3 rate on TODAY's going)

Feature importance on the v6.11 50K-row training (top → bottom):
- `last_start_margin` 14.5%, `jockey_win_rate` 7.8%, `recent_form` 6.8%, `field_rel_draw` 4.2%, `horse_age` 4.1%, `class_change` 4.0%, `class_level` 3.9%, `early_speed` 3.8%, `prior_win_rate` 3.6%, `jockey_horse_coop` 3.6%, `going_performance` 3.6%, `cd_form_prior` 3.6%, `trainer_win_rate` 3.6%, `field_size` 3.5%, `jt_pair_win_rate` 3.4%, `jockey_form_30d` 3.4%, `weight_vs_field` 3.4%, `sire_top3_rate_on_going` 3.3%, `late_kick` 3.2%, `horse_weight_change` 3.2%, `mid_race_pace` 3.2%, `freshness` 3.2%, `rating_change` 3.2%.

**Known correlations (XGBoost handles internally)**:
- `jockey_win_rate × jockey_form_30d` = 0.740 (both jockey skill at different windows)
- `jockey_win_rate × jt_pair_win_rate` = 0.736 (jt_pair is Bayesian-smoothed jockey rate)
- `last_start_margin × rating_change` = 0.563 (winning bumps rating)
- These were flagged but kept; the 3-feature Tier 1 prune (see below) hit the more obviously-redundant ones.

## Removed-features changelog

- **v6.11 (2026-05-24, Tier 1 PRUNE from 3-expert review)**: `draw_advantage` (r=−0.0095 with `is_winner` on 50k rows — literal zero), `speed_fig_norm` (HK paceless; r=+0.53 with `last_start_margin`), `running_style` (r=+0.914 with `early_speed` — coarsened duplicate). Burned-A/B mixed: WIN flat, QIN +3, TOP-4 −3, TOP-5 −1. Backup: `models/hkjc_racing_model.pkl.pre-prune-tier1`. See memory: `project_v6_11_prune_and_gate_revert_2026-05-24.md`.
- **v6.4 (2026-04-25)**: `prior_pace, adjusted_speed, distance_pb_ratio, speed_consistency` — all 0.0 importance after the v6.3 → v6.4 bug fixes; redundant with `last_start_margin`.
- **Wave 2c**: `field_rel_speed` (r=0.72 with `recent_form`), `days_since_last_win` (r=0.74 with `recent_form`).
- **Earlier audits**: `odds_implied_prob`, `venue_win_rate` (r=0.984), `weight_carried_norm` (r=0.877), `draw_field_interact` (r=0.934), `prior_avg_position`, `prior_speed` (replaced by `adjusted_speed` before `adjusted_speed` itself was dropped), `late_money_signal` (constant 0.5 — train/predict distribution mismatch), `jockey_trainer_combo`.

## Added-features changelog (v6.5 → v6.11)

- **v6.6 (2026-05-09)**: `jt_pair_win_rate` (Bayesian-smoothed jockey×trainer pairing, 305 pairs ≥50 joint runs), `speed_fig_norm` (par-time-normalised; revived again 2026-05-23 after par-times bug fix, then re-dropped in v6.11).
- **v6.8 (2026-05-14, Path A)**: `early_speed`, `late_kick`, `running_style` (running_style dropped in v6.11). Derived from `race_results.running_position` (99.4% backfill).
- **v6.9 (2026-05-24)**: `mid_race_pace` — race-mean-normalised (800-400)-(last-400) split, mean of last 3 prior sectional runs. Source: `race_results.sectional_times`.
- **v6.10 (2026-05-24)**: `sire_top3_rate_on_going` — sire's Beta(2,8)-smoothed top-3 rate on today's going bucket. Source: `renavon_horses` (frozen 2026/04/22).

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
- **4-fold disjoint temporal split (current)**: rows[0:60%] train, [60:70%] early-stopping, [70:85%] post-blend calibrator, [85:100%] backtest holdout. The previous 60/20/20 split was replaced when `scripts/calibrate_blend.py` was added (post-blend isotonic recalibrator needs its own disjoint fold).
- **Exponential recency weighting** (v6.4 — was linear): `w(i) = 0.25 + 0.75·exp((i-(n-1))/(n/3))`. Newest row weight 1.0, oldest ~0.29. With 50% jockey turnover and 94% horse turnover since 2021, older rows describe a different field; exponential decay halves their influence twice as fast as the prior linear schedule.
- **Field clipping** (v6.4 fix): training-side `field_size` and `field_rel_draw` are now clipped to [0,1]. Inference (features.py) already clipped via F-16; training (training.py) was missing the clip, so the model trained on values up to 15.0 from oversize legacy fields.
- **Freshness fallback** (v6.4 fix): when DB column `days_since_last_run` is null/zero (87.7% of legacy rows), training now falls back to `(race_date - prior race_date)` math — matching the inference-side fallback. Without this, freshness was a near-constant 0.5 in training and got 0.0 importance in v6.3.
- **Isotonic calibration** (Wave 2c switch from Platt): `IsotonicRegression(out_of_bounds='clip')` fitted on validation-set predictions (logit-input) → actual outcomes. Recalibrates XGBoost's overconfident probabilities. Fitted on validation (not test) so test metrics remain unbiased. The val slice currently spans 2024-04 → 2025-04 (1y stale on field strength) — known structural trade-off, not addressed in v6.4.
- **Place model**: Separate XGBoost trained on `is_top3` labels alongside the win model. Used for place bet probability and quinella placer-boost pricing.
- **scale_pos_weight**: Computed as `n_neg / n_pos` (≈ 11 with 8.3% positive class) and passed to XGBClassifier. Without this, the model would bias toward predicting "loss" everywhere (92% accuracy from doing nothing).
- **experiment_log integrity** (v6.4 fix): `log_experiment` is only called when `save=True`. Tests passing `save=False` were polluting `models/experiment_log.json` with n_samples=300/2000 dummy entries (399 polluted rows cleaned out, 5 real training runs retained).

## Speed Figures (`analytics/speed_figures.py`) — out of ML schema again as of v6.11

History:
- **v6.4 (2026-04-25)**: dropped from ML schema (0.0 importance — XGBoost ignored in favour of `last_start_margin`).
- **v6.6 (2026-05-09)**: added back as `speed_fig_norm` (par-time + weight normalised, distinct from the v6.4 `adjusted_speed`).
- **2026-05-23**: par-times bug fix (`compute_par_times` was silently dropping 85% of rows) revived `speed_fig_norm` importance from 0.000 → 0.034; memory `project_speed_fig_starved_not_dead.md` documents that calibration regressed (ECE 0.053→0.080) and hit-rates didn't move.
- **v6.11 (2026-05-24)**: dropped again in Tier 1 PRUNE. r=+0.53 redundancy with `last_start_margin` + HK paceless context + prior NEUTRAL-for-betting verdict.

`analytics.speed_figures` module still computes adjusted_speed for analyst use; just not consumed by the ML schema. `analytics.speed_figures` is used by the rule engine `speed_fig` factor (5% weight, separate channel).
