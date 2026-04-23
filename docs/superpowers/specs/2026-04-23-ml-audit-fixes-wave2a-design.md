# ML Audit Fixes — Wave 2a Design Spec

**Date:** 2026-04-23
**Upstream:** `docs/audits/2026-04-23/final-report.md`, `docs/audits/2026-04-23/post-fix-baseline.md`, Wave 1 implementation complete (commits `aad8229` through `2ee6910`).
**Scope:** Wave 2a — six measurement-critical fixes.
**Status:** Design approved, pending user spec review before plan-writing.

---

## Goal

Produce a corrected baseline that removes the `rating=0` drag on backtest metrics, eliminates a redundant correlated feature, fixes a train/predict feature mismatch, empirically grounds the market-blend weights, and re-evaluates the box-4 strategic default against the corrected numbers. After Wave 2a lands, every ROI figure is measured on a faithful production pipeline with a validated feature schema and justified blend weights.

## Non-Goals

- Hygiene fixes (F-08, F-12 through F-20) — Wave 2b
- Correlated-finish dividend multiplier (F-10) — conditional on Wave 2a revealing EV is still broken
- Historical data expansion (F-21) — parallel track, weeks of lead time
- Isotonic regression Platt alternative — only if calibration curve still has tail miscalibration post-F-09
- Dashboard redesign — third workstream, queued

## In-Scope Fixes

| ID | Title | Severity | Tractability | Stage |
|----|-------|----------|--------------|-------|
| C1 | `rating=0` drag in `core/scorer.py` ALWAYS_INCLUDED | High | <1h | A |
| F-06 | Fix `weight_vs_field` train/predict mismatch (DS-03) | High | <1h | A |
| F-15 | Drop `jockey_trainer_combo` (r=0.79 with jockey_win_rate) | Medium | <1h | B |
| F-11 | Tighten feature parity test (<0.02 per-feature tolerance) | Medium | <1h | C |
| F-09 | Grid-sweep and justify market-blend weights | High | <1d | C |
| F-22 | Re-evaluate box-4 pivot on corrected baseline | High | needs-data | C |

Hard-gated order: Stage A (any order within) → Stage B → Stage C.

## Stage A — Code-only fixes (no retrain)

### C1 — rating=0 drag in `core/scorer.py`

**Problem:** `core/scorer.py`'s `ALWAYS_INCLUDED` set contains `rating` and `gear`. When a horse's rating is missing from the DB (86% of holdout races), `build_production_race` in `scripts/backtest.py` substitutes `rating=0.0`. The scorer then includes this in the weighted average with weight ~0.08, dragging every horse's rule_score down by ~8 points. Relative rankings in a race are preserved (all horses get the same drag), but absolute probability scales are deflated, explaining why holdout Brier rose 0.126 → 0.168 post-F-03 despite training-set Brier improving 0.098 → 0.074.

**Fix approach:** In `core/scorer.py::compute_weighted_score` (or wherever `ALWAYS_INCLUDED` is honored), treat `rating == 0` and `gear == ''` as "missing" — exclude from the weighted average via the same dynamic-weight renormalization used for other missing factors. This means the ALWAYS_INCLUDED set loses its absolute guarantee for these two factors specifically when the value is unavailable.

**Live predictions are unaffected** — race cards always populate rating. Only backtest (historical races with missing inputs) changes behavior.

**Files touched:** `core/scorer.py`; possibly a new test under `tests/`.

**Acceptance:** After C1 lands, re-run `python3 -m scripts.backtest`. Holdout Brier must improve (current 0.168 → expected < 0.150). Unit test added: feeding a 12-horse race where rating is missing for all horses produces the same rule_score ranking as feeding the same race with rating=50 for all horses (because both are equivalent under renormalization — the rating factor drops out).

### F-06 — `weight_vs_field` train/predict mismatch

**Problem:** DS-03 measured that 68% of horses show |Δ| > 0.1 on `weight_vs_field` between `extract_features()` (inference) and `build_training_data_from_db()` (training). Data-scientist persona flagged this as High-severity — the model was trained on one feature distribution and predicts on another.

**Fix approach:** Read both functions. Identify whose `weight_vs_field` computation is wrong (likely one uses field median weight, the other uses per-horse weight, or one uses kg and the other lbs, or one normalizes and the other doesn't). Pick the semantically correct one and make both compute it identically.

**Files touched:** `ml/features.py` and/or `ml/training.py`.

**Acceptance:** Parity probe — for 50 random holdout horses, run both functions; assert <5% with >0.1 delta on `weight_vs_field` (down from 68%). Probe lives in a test: `tests/test_feature_parity_weight.py`.

### Stage A Gates

- `pytest tests/` 100% passing
- `python3 -m scripts.backtest` runs clean
- Holdout Brier < 0.150 (primary gate — confirms C1 took effect)
- Parity probe: weight_vs_field <5% with >0.1 delta
- Production pickle sha256 unchanged (Stage A does not retrain)

Checkpoint: `git tag wave2-pre-retrain`, backup `models/hkjc_racing_model.pkl` → `models/hkjc_racing_model.pkl.wave1-backup`.

## Stage B — Schema change + retrain

### F-15 — Drop `jockey_trainer_combo`

**Problem:** Post-Wave-1 retrain, feature correlation report flags `jockey_win_rate × jockey_trainer_combo = 0.79`. DS-07 classified as Medium. Drop the redundant feature.

**Fix approach:**
1. Edit `ml/feature_schema.py`: remove `jockey_trainer_combo` from `FEATURE_NAMES`; update `NUM_FEATURES = 22`; if there's a `FEATURE_DEFAULTS` dict, remove the entry too.
2. Edit `ml/features.py::extract_features()`: stop computing and returning `jockey_trainer_combo`.
3. Edit `ml/training.py::build_training_data_from_db()`: same — stop producing `jockey_trainer_combo`.
4. Edit `ml/model.py` only if the schema is hardcoded anywhere (should not be — it reads from `FEATURE_NAMES`).
5. Retrain: `python3 -m scripts.train`.

The pickle's `_load_model()` feature-count guard in `ml/model.py` will refuse to load the Wave 1 (23-feature) pickle after this schema change. The retrain produces a new 22-feature pickle that loads cleanly. This is correct.

**Files touched:** `ml/feature_schema.py`, `ml/features.py`, `ml/training.py`. Possibly minor touch-ups elsewhere if `jockey_trainer_combo` is referenced by name.

**Stage B Gates:** (Wave 1 gate set + two new)
- n_samples ≥ 3000 (unchanged)
- trained_date ≥ 2026-04-23 (will be a fresh date)
- Calibrator span ≥ 0.10 (should stay ~0.26)
- Brier ≤ 0.080 on training test set
- BSS ≥ 0
- **NEW gate:** |Brier(22-feature) − Brier(23-feature Wave 1)| ≤ 0.01 on training test set. If Brier rises by >0.01, `jockey_trainer_combo` was carrying non-redundant signal — revert F-15.
- **NEW gate:** Feature correlation report from training shows zero pairs with |r| > 0.7.
- F-05 integrity check passes against the new pickle.

Rollback trigger: any gate fails → restore `wave2-pre-retrain` checkpoint (`models/hkjc_racing_model.pkl.wave1-backup`), revert F-15 commits. Escalate.

Checkpoint: `git tag wave2-post-retrain`.

## Stage C — Post-retrain validation + tuning + strategic re-evaluation

### F-11 — Tighten feature parity test

**Problem:** `tests/test_data_accuracy.py::test_training_and_predict_features_same_distribution` currently allows up to 2 "wildly-off" features per horse. That's how DS-03 slipped through — weight_vs_field was "wildly-off" for 68% of horses but only counted as 1 feature so the test passed.

**Fix approach:** Replace the test with a per-horse, per-feature check: for 20 random DB rows, call both `extract_features()` and the corresponding training row; assert `|Δ| < 0.02` element-wise for every feature (no wildly-off tolerance). Test passes on the Stage B retrained model if F-06 is correct and F-15 is correct.

**Files touched:** `tests/test_data_accuracy.py` (replace the existing test).

**Acceptance:** Test passes on Wave 2 retrained model. Verify by temporarily reverting F-06 — test fails. Then re-apply F-06 — test passes. (Quick sanity check, then re-apply.)

### F-09 — Grid-sweep market-blend weights

**Problem:** Current blend weights (35-65% by field_size × distance) are heuristic per CLAUDE.md. P-07 classified as High but "gated on other fixes." Other fixes are now done.

**Fix approach:**
1. Confirm `scripts/backtest.py --sweep` still works (it retained ML-only path per Wave 1). If it's ML-only, adapt to exercise production pipeline with configurable `_dynamic_blend_weights()` override via a CLI flag or env var.
2. Sweep over `model_w ∈ {0.25, 0.35, 0.45, 0.55, 0.65}` × `field_size_threshold ∈ {8, 10, 12}` × `distance_threshold ∈ {1200, 1400, 1600}`. That's 5 × 3 × 3 = 45 grid points. Each runs the full 73-race backtest.
3. Primary objective: BSS on training test set (from accuracy report run after each sweep point). Secondary: box-3 ROI lower-bound bootstrap CI.
4. Pick the grid point with best primary; if multiple points are statistically indistinguishable (within ±0.005 BSS), prefer the simplest (constant weight).
5. Hardcode the chosen grid point in `core/scorer.py::_dynamic_blend_weights()` (or replace with constant). Document the choice in CLAUDE.md with a one-paragraph rationale and the specific numbers.

**Files touched:** `core/scorer.py` (and possibly `scripts/backtest.py` if sweep path needs adapting), `CLAUDE.md` (document choice).

**Acceptance:** The `--sweep` log documents all 45 grid points with their BSS. The chosen point is hardcoded. Backtest with the chosen point shows BSS ≥ the pre-sweep default.

### F-22 — Re-evaluate box-4 pivot

**Problem:** Wave 1 post-fix numbers showed box-3 unconditional +1798% > box-4 unconditional +1228%, suggesting the box-4 default shipped yesterday may not be supported. But Wave 1 numbers had the rating=0 drag affecting 86% of races. Now that C1 removes the drag and F-15 retrain gives a cleaner model, re-evaluate.

**Fix approach:**
1. Run `python3 -m scripts.backtest` with the F-09-chosen blend weights.
2. Extract from output: box-4 hit rate, box-3 hit rate, straight-Q hit rate.
3. Compute Wilson 95% CI on each hit rate (`statsmodels.stats.proportion.proportion_confint` with `method='wilson'` or manual Wilson formula).
4. Compute bootstrap 95% CI on each ROI (resample the 73-race hit-and-dividend vector 10,000 times, compute ROI per resample, take 2.5% and 97.5% percentiles).
5. Decision rule: **if `box4_CI_lower > box3_CI_upper` AND `box4_ROI_CI_lower > box3_ROI_CI_upper`, keep box-4 default. Else, switch to whichever of {straight-Q, box-3} has the highest conservative lower bound on ROI.**
6. Update `core/recommender.py::recommend_bets` to reflect the decision. If box-4 is kept, no change. If demoting, change the default fallback logic.
7. Update CLAUDE.md's "Betting Strategy" section with the decision and the CI numbers.

**Files touched:** `core/recommender.py`, `CLAUDE.md`, `docs/audits/2026-04-23/wave2-baseline.md` (new final doc).

**Acceptance:** The recommender's default is backed by specific CI numbers documented in the new baseline doc. If the CI evidence is ambiguous (box-4 and box-3 CIs overlap), the default stays as current with an explicit "statistically indistinguishable at n=73" caveat and F-21 data-collection is queued.

### Stage C Gates

- F-11 tightened parity test PASSES on Stage B retrained model
- `scripts/backtest.py --sweep` completes; 45 grid points documented
- Recommender default backed by explicit CI numbers
- Final end-to-end: `pytest tests/` passing, backtest clean, accuracy report BSS ≥ 0.03 (matching Wave 1), `wave2-baseline.md` written

Checkpoint: `git tag wave2-complete`.

## Post-Wave-2a Deliverables

1. All six fixes landed on `develop`
2. Tags: `wave2-pre`, `wave2-pre-retrain`, `wave2-post-retrain`, `wave2-complete`
3. Backup: `models/hkjc_racing_model.pkl.wave1-backup`
4. `docs/audits/2026-04-23/wave2-baseline.md` replaces `post-fix-baseline.md` as authoritative baseline
5. CLAUDE.md updated: blend-weight decision, recommender default decision, 22-feature schema
6. User review gate: user reviews the new baseline doc and decides whether to proceed to Wave 2b (hygiene) or queue something else

## Known Risks

1. **Stage A Brier improvement gate may fail.** If C1 doesn't move holdout Brier, the rating=0 drag hypothesis was incomplete. Rollback via tag, investigate.
2. **F-15 may hurt Brier.** The 0.79 correlation is strong but XGBoost may have been using interactions that jockey_win_rate alone can't replicate. New gate (|ΔBrier| ≤ 0.01) catches this.
3. **F-09 grid sweep may find no improvement.** Null result is valid — document and keep current weights. Not a failure.
4. **F-22 may be ambiguous.** If CIs overlap, keep current default with caveat; don't force a decision on insufficient data.
5. **F-22 may contradict Wave 1's takeaway.** Corrected baseline (no drag) could restore box-4's relative position. Follow the CI rule regardless of priors.
6. **Commit count** — Wave 2a may produce 8-12 commits. Mitigation: same per-fix discipline as Wave 1, tags for rollback.

## Process After Wave 2a

1. User reviews `wave2-baseline.md` and decides:
   - **Wave 2b** — brainstorm hygiene batch (F-08, F-12 through F-20)
   - **F-10 dividend multiplier** — if EV% is still broken post-2a
   - **F-21 data collection** — parallel track, weeks lead time
   - **Dashboard redesign** — third workstream, starts after Wave 2b

## Open Questions (resolved during brainstorm)

- Q: Wave 2 scope — narrow strategic, measurement-critical, everything, user-chosen? → **A:** Measurement-critical (A).
- Q: Stage breakdown? → **A:** Three stages mirroring Wave 1 pattern (A code-only, B retrain, C validation + tuning + re-eval).
- Q: F-22 decision rule? → **A:** CI lower/upper bound comparison per stage.

## Known Limitations

- 73-race sample means all CI bands are wide. F-22 may be ambiguous. This is a feature of the evidence, not a spec failure.
- The blend-weight grid sweep (F-09) is restricted to hyperparameters of the existing schedule. Doesn't consider alternative blend structures (e.g., learned per-race weights).
- F-06 assumes one-sided fix (either training or inference is correct). If both are wrong in different ways, the parity gate catches it but the decision on which is "right" needs a judgment call — default to whatever matches the documented intent in CLAUDE.md.
