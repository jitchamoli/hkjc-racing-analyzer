# ML Audit Fixes — Wave 1 Design Spec

**Date:** 2026-04-23
**Upstream:** `docs/audits/2026-04-23/final-report.md` (23-fix prioritized list)
**Scope:** Wave 1 — the four Critical fixes plus two High fixes that validate this batch
**Status:** Design approved, pending user review before plan-writing

---

## Goal

Restore correctness to the prediction pipeline by shipping the minimum set of fixes that make the production model, its calibrator, and the backtest measurement valid. After this batch lands, race-day predictions come from a real-data-trained model with a non-destroyed Platt calibrator, the backtest measures what is actually bet, and a pre-bet integrity check prevents regression. Further hygiene, strategic re-validation, and dashboard work are explicit follow-up waves.

## Non-Goals

- Implement remaining audit fixes (F-06 through F-22) — separate wave
- Replace Platt with isotonic regression — try the simpler logit fix first
- Tune softmax temperature, 60/40 blend, or market-blend weights
- Dashboard redesign (still queued as the third workstream)
- Decide whether the betting strategy is correct on a strategic level — that discussion is triggered by Stage C results, but only if the post-fix ROI comes back genuinely negative

## In-Scope Fixes

| ID | Title | Severity | Tractability | Stage |
|----|-------|----------|--------------|-------|
| F-04 | pytest-overwrite guard | Critical | <1h | A |
| F-01 | Platt logit-input fix | Critical | <1h | A |
| F-05 | Pre-bet integrity check | High | <1h | A |
| F-02 | Retrain on real DB data | Critical | <1d | B |
| F-03 | Backtest matches production pipeline | Critical | <1d | C |
| F-07 partial | Tests for calibration, integrity, parity | High | <1d | A/C |

Six fixes total. Hard-gated order: F-04 → F-01 → F-05 (Stage A, any order within stage) → F-02 (Stage B) → F-03 (Stage C) → parity tests (Stage C).

## Stage A — Code-only fixes (no retrain)

Three commits on `develop`. Each commit is independently reversible, but together they leave the repo in a **temporarily-unusable-for-predictions** state until Stage B lands. Acknowledged limitation; document in commit messages.

### F-04 — pytest-overwrite guard

**Problem:** The pytest fixture `generate_synthetic_training_data(500)` calls `train()`, which calls `_save_model()` unconditionally, overwriting the production pickle with a synthetic-trained artifact. This is how the current `models/hkjc_racing_model.pkl` (version `20260423-012955`, n_samples=500) got there.

**Fix approach:** Add a `save: bool = True` kwarg to `HKJCRacingMLModel.train()`. The synthetic-data pytest fixture passes `save=False`. Alternatively or additionally, `_save_model()` refuses to write if `len(training_data) < MIN_REAL_SAMPLES` (e.g., 1000) — a belt-and-suspenders guard in case future tests forget the kwarg.

**Expected files touched:** `ml/model.py`, `tests/**` (wherever the synthetic fixture trains).

**Acceptance:** `pytest tests/` runs to completion AND `models/hkjc_racing_model.pkl` sha256 is unchanged after the test run. New test `test_synthetic_training_does_not_save()` asserts this directly.

### F-01 — Platt logit-input fix

**Problem:** Calibrator is fit on raw probability values (scale [0,1]) rather than their logits. LogisticRegression's linear decision function + the ~0.5-clustered input → slope of ~0.137 mechanically bounds output to a ~0.012 span → 88× compression observed in Stage 1 forensic.

**Fix approach:** Where `calibrator.fit(val_proba, y_val)` is called in `ml/model.py` (likely in `train()` right after XGBoost produces validation probs), change to `calibrator.fit(logit(val_proba), y_val)`. Mirror at inference: `calibrator.predict_proba(raw_probs)` becomes `calibrator.predict_proba(logit(raw_probs))`. Clip inputs to `[1e-6, 1-1e-6]` before `logit` to avoid infinities. Use `scipy.special.logit` or equivalent.

**Expected files touched:** `ml/model.py` (train + predict paths).

**Acceptance:** Unit test `test_platt_calibrator_logit_span()` trains a synthetic calibrator on logit-transformed data, asserts the resulting calibrator's span on a test grid is ≥ 0.1 (versus the pre-fix span of ~0.012). Code compiles, `pytest tests/` still passes, existing synthetic-trained pickle becomes temporarily unusable — this is expected and documented in the commit message.

### F-05 — Pre-bet integrity check

**Problem:** No defensive layer prevents predictions from running against a synthetic-trained or otherwise-compromised pickle. F-04 prevents the overwrite going forward, but we need a runtime check that blocks predictions if the pickle is already bad.

**Fix approach:** In the prediction entry point (`generate_predictions.py` or `core/analyzer.py`, whichever loads the model and runs `predict`), assert before any race processing:
- Loaded pickle metadata has `n_samples >= 1000` (rejects synthetic 500-sample artifacts)
- `trained_date` is parseable as a recent date
- Feature importance signature does not match the synthetic fingerprint (optional; if feature importances on synthetic data are known to cluster in a particular pattern, detect it)

Fail loudly with a clear error message pointing to this fix (e.g., "Integrity check failed: model trained on <1000 samples. Retrain via `python3 -m scripts.train` before predictions can run. See F-05.").

**Expected files touched:** `generate_predictions.py` and/or `core/analyzer.py`, possibly a new helper `ml/integrity.py`.

**Acceptance:** Unit test `test_integrity_rejects_synthetic_pickle()` loads a fixture pickle with n_samples=500 and asserts the integrity check raises. Test `test_integrity_accepts_real_pickle()` with n_samples=3000+ asserts it does not raise. Current production pickle (synthetic) will fail the check — correct and intended behaviour.

### Stage A Gates

- All three commits landed in `develop`, each independently reversible
- `pytest tests/` 100% passing (new tests + existing, none skipped)
- `models/hkjc_racing_model.pkl` sha256 unchanged (guard works)
- Current pickle is knowingly unusable for predictions — must retrain before live use

Checkpoint: `git tag pre-retrain-2026-04-23` and back up `models/hkjc_racing_model.pkl` to `models/hkjc_racing_model.pkl.synthetic-backup` before Stage B.

## Stage B — Retrain on real data (F-02)

**Problem:** Production model was trained on 500 synthetic samples. Retrain on the 4403+ real race results in `tracking_data/horse_history.db`.

**Fix approach:** Run `python3 scripts/train.py` (or `python3 -m scripts.train`). Verify the training pipeline uses DB data when ≥200 samples exist (per CLAUDE.md). Because Stage A landed F-01, the retrain automatically uses the logit-calibrator fix.

**Expected files touched:** `models/hkjc_racing_model.pkl`, `models/hkjc_racing_model.sha256`, `models/experiment_log.json` (all outputs of training, not code).

**Stage B Gates (each is a hard requirement):**
- `n_samples >= 3000` in new experiment_log entry
- `trained_date` is 2026-04-23 or later
- Calibrator span on validation set ≥ 0.1 (validates F-01 took effect; pre-fix was ~0.012)
- Brier score ≤ 0.080 (down from 0.098; if Brier stays ≥ 0.098, the retrain did not improve predictive signal and we investigate before proceeding)
- BSS ≥ 0 (model at least matches constant predictor; if negative, retrain did not fix underlying issue)
- F-05 integrity check passes against the new pickle
- `python3 -m scripts.accuracy --all` completes; 5-bin calibration curve tightens (bins closer to diagonal vs pre-fix)

**Rollback trigger:** Any gate fails → restore `models/hkjc_racing_model.pkl.synthetic-backup` (though note this is itself a bad pickle; more useful is tag `pre-retrain-2026-04-23` which lets us revert the code changes too). Escalate to user with failure details.

## Stage C — Backtest matches production (F-03 + tests)

**Problem:** `scripts/backtest.py` runs ML → softmax → market blend, bypassing the rule engine, the 60/40 blend, and the recommender. The +999%/+1064% ROI numbers that justified yesterday's box-4 pivot describe a pipeline that is not what the dashboard produces on race day.

**Fix approach:** Modify `scripts/backtest.py` so each per-race prediction runs through the production path:
1. Build race card from DB row + live odds (from `race_dividends` or equivalent historical snapshot if available)
2. Invoke `CombinedRaceAnalyzer.analyze()` or its equivalent — this runs the 24-factor rule engine, ML model, 60/40 blend, softmax, market blend
3. Invoke `core/recommender.py` to apply conviction gates, BANKER/market-favourite swaps, and produce the actual bet recommendations
4. Simulate bets based on recommender output, not unconditional boxes

**Match-production scope:** "Match production" means the backtest produces the same `HorseAnalysis.combined_probability` and `bet_recommendation` fields per horse as `generate_predictions.py` would on the same inputs. If historical rule-engine inputs (tips, live-odds-at-race-time snapshots) are not available for a holdout race, use documented neutral fallbacks for those factors and print a per-race warning. If more than 50% of holdout races lack adequate inputs, surface as an audit follow-up and proceed with what's available.

**Expected files touched:** `scripts/backtest.py`. Possibly `scripts/backtest.py` grows helper functions that delegate to `core/` modules; keep backward-compatible CLI flags.

**Tests added in Stage C (F-07 subset):**
- `tests/test_backtest_production_parity.py` — integration test. Picks 5 holdout races, runs each through both `scripts/backtest.py`'s per-race prediction function AND `generate_predictions.py`'s race-prediction code path (both loaded via import). Asserts: per-horse `combined_probability` matches within tolerance 0.001, and `bet_recommendation` matches exactly.
- `tests/test_platt_calibrator_logit_span.py` — unit test from F-01 acceptance (add under Stage C for consolidation)
- `tests/test_synthetic_training_does_not_save.py` — from F-04 acceptance
- `tests/test_integrity_rejects_synthetic_pickle.py` — from F-05 acceptance

**Stage C Gates:**
- `pytest tests/` 100% passing, including new tests above
- `scripts/backtest.py` runs to completion under the new production-parity pipeline
- Backtest output is substantially different from pre-F-03 — this is expected (likely lower ROI percentage under conviction-gated bets, lower bet count). **The gate is NOT "ROI goes up"; the gate is "backtest runs cleanly and parity test passes"**
- `docs/audits/2026-04-23/post-fix-baseline.md` created, capturing new backtest numbers as the authoritative baseline for all future ROI claims

**Rollback trigger:** Parity test fails systematically (e.g., combined_probability mismatch >0.01 on >20% of races) → do NOT land F-03 commit. Document the blocker, leave backtest in pre-F-03 state with its known bias, escalate to user.

## Post-Batch Deliverables

1. `develop` branch contains F-01/02/03/04/05 + tests, all gates passed
2. `docs/audits/2026-04-23/post-fix-baseline.md` with corrected backtest + accuracy numbers, superseding `raw/backtest-output.txt`
3. Tags: `pre-retrain-2026-04-23`, `pre-backtest-parity-2026-04-23`
4. Model pickle in real-data-trained, logit-calibrator state, integrity-checkable
5. User review gate: user reviews post-fix baseline and decides whether to keep box-4 strategy, revert, or open separate strategic conversation (this seeds the next brainstorm)

## Explicitly Out of Scope

- All other audit fixes (F-06 re-validation, F-07 full test coverage, F-08 through F-22). These go to Wave 2.
- Isotonic regression as a Platt alternative (try the simpler logit fix first)
- Blend-weight tuning, softmax-temperature tuning, Harville replacement
- Dashboard redesign (still queued as the third workstream — not starting until fixes settle)
- Any decision about whether to bet at all — that conversation is triggered only if post-Stage-C ROI is genuinely negative

## Known Risks

1. **Repo temporarily unusable for predictions during Stage A→B transition.** Mitigation: run stages same-day, F-05 integrity check prevents accidental use.
2. **Stage B may reveal deeper issues.** If Brier stays ≥ 0.098 after retrain on real data, the problem isn't just Platt. Rollback keeps us safe; user escalation required.
3. **Stage C parity may be unachievable for races missing inputs.** If tips/live-odds snapshots aren't in the DB for most historical races, the production-parity backtest becomes sparse. Separate conversation required if this manifests.
4. **Post-fix ROI may be negative.** With recommender gating bets instead of unconditional boxes, the true strategy ROI may be negative. This is a feature of the fix (we're now measuring the right thing), not a regression — but it kicks the strategic question back to brainstorming.
5. **Race-meeting during implementation:** if a meeting happens mid-fix, F-05 prevents prediction misuse; document this in a CLAUDE.md note while the batch is in flight.

## Process After This Batch

1. User reviews `post-fix-baseline.md`. Decides: keep current strategy | revert to box-3 | open strategic re-design | other.
2. Brainstorm Wave 2 (remaining High/Medium/Low fixes from audit) based on the new baseline.
3. Eventually: brainstorm the dashboard redesign (third workstream, deferred).

## Open Questions (resolved during brainstorm)

- Q: Order of fixes — audit said F-02 first, I recommended F-04 → F-01 → F-02 → F-03 so retrain picks up the logit fix in a single pass. → **A:** Approved, my ordering.
- Q: Scope — strict F-01/02/03/04 or include adjacent Highs? → **A:** Include F-05 (cheap defensive layer) + F-07 subset (tests required by validation gates).
- Q: Sequencing approach — monolithic batch, per-fix checkpoint, or staged with gates? → **A:** Staged with gates (Approach 3).
- Q: "Match production" scope in F-03 — literal code-path sharing, or functional-equivalence with documented fallbacks for missing inputs? → **A:** Functional-equivalence with documented fallbacks; flag if >50% of holdout races lack adequate inputs.
