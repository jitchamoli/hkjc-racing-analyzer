# HKJC Racing Analyzer — ML + Probability Audit Final Report

**Date:** 2026-04-23
**Scope:** Full ML + probability + betting-strategy pipeline
**Evidence:** 73-race holdout (72 with dividend data), Apr 22 race card, code inspection, production-model pickle inspection, feature-parity probes
**Authors:** Main agent (forensic + strategic) + probability-expert subagent (Stage 2 Platt/softmax/Harville/calibration) + data-scientist subagent (Stage 2 features/leakage/backtest/training)

---

## Executive Summary

The audit surfaces **four Critical production issues**, any one of which would be sufficient cause to stop betting off the current model. Together they constitute the finding that **the production pipeline is not doing what its architecture claims, and the backtest that supposedly validated the recent Quinella-first + box-4 pivot was not measuring the production pipeline**.

The three headline problems:

1. **The shipped model (`20260423-012955`) was trained by the pytest fixture on 500 synthetic samples**, not by `scripts/train.py` on the 4,403-row real DB. Confirmed independently by probability-expert (via `experiment_log.json` Brier anomaly and calibrator coefficient inspection — P-04) and data-scientist (via seed-42 reproduction of the exact high-correlation pairs logged — DS-01). This is live in production right now. `models/hkjc_racing_model.pkl` is a pytest artifact.
2. **The Platt calibrator is fit on raw probability instead of logit** (`ml/model.py:232-235`), producing a mathematically capped output span of 0.012. Every XGBoost probability — regardless of strength — collapses into a ~88× compressed band around the base rate. Measured across 10 severe-miss traces + 9 Apr 22 races: compression ratio 86.6×–89.1× (P-01). This erases all ranking signal from the ML model. The market blend (heuristic, ungrounded weights — P-07) becomes the sole ranking signal in practice.
3. **The backtest (`scripts/backtest.py`) runs an ML-only pipeline with zero rule engine and zero recommender logic** (DS-02). The +999% straight-Q / +849% box-3 / +1064% box-4 ROI numbers in `raw/backtest-output.txt` describe unconditional box bets under a synthetic-trained ML-only → softmax → market-blend pipeline. Production emits conditional bets through `core/recommender.py`'s conviction gates + BANKER / market-favourite swaps, which the backtest does not exercise. The numbers used to justify the box-4 pivot do not describe production behaviour.

The compound statistical indictment: the current production Brier score (0.098) is **worse** than always predicting the base rate (0.075). Brier Skill Score = **−0.31**. The model is actively adding noise (P-02).

**What's at stake:** Real-money bets are currently being placed against a synthetic-trained model whose ML signal is mechanically destroyed, informed by a backtest that measures a different pipeline. The Apr 22 "rough day" that triggered this audit is exactly what this architecture predicts: 88× compression leaves the market blend in control, the market loses to takeout, so do we.

**Top three fixes that restore correctness:**

- **F-01** — Fit Platt on `logit(val_proba)`, not `val_proba`. One-line change. Unlocks ML ranking.
- **F-02** — Retrain on the 4,403-row DB via `python3 -m scripts.train`. Overwrites the synthetic pickle that shipped by accident.
- **F-03** — Make `scripts/backtest.py` run the full production pipeline (rule engine + `assign_bet_recommendations` + conviction gates), so reported ROI describes what we actually bet.

Until all three land, **no strategic claim derived from the current backtest should be treated as decision-grade**, including the box-4 pivot itself.

---

## Top Findings (by severity)

### Critical

**F-root / P-04 ∩ DS-01 — Production model is synthetic-trained on 500 samples, not the real 4,403-row DB.**
Evidence is independent and convergent. Probability-expert inferred it from a latest `experiment_log.json` entry recording `n_samples=500` attached to the shipped pickle version `20260423-012955`. Data-scientist reproduced the exact high-correlation pairs logged in that entry (`draw_advantage × field_rel_draw r=+0.716`, `trainer_win_rate × jockey_trainer_combo r=+0.725`) by running `generate_synthetic_training_data(500)` with seed 42. `tests/conftest.py:140-152` calls `HKJCRacingMLModel()` (which uses `config.MODEL_PATH`) → `model.train(...)` → `_save_model()` — unconditionally overwriting the production pickle every time the fixture runs. Blast radius: every prediction, every bet, every calibration curve number, every backtest — all derived from a toy model trained on fabricated data.

**P-01 — Platt calibrator fit on raw probability, not logit → 88× signal compression.**
`ml/model.py:234-235` passes `val_proba` (scale [0,1]) as the Logistic Regression feature. Correct Platt fits `logit(val_proba)`. Learned slope `coef_=[[0.137]]`, intercept `-2.33` mathematically caps output span at 0.012 across the entire input range [0,1]. Measured Stage-1 output: 0.089–0.095 for raw inputs 0.07–0.54 — exactly what `expit(0.137·raw + -2.33)` predicts. All 19 traced races (10 severe-miss + 9 Apr 22) show compression 86.6×–89.1× — it is a property of the calibrator coefficients, not race difficulty. Blast radius: softmax input is uniform, market blend (55%+) is the sole ranking signal.

**P-02 — Production Brier (0.098) is worse than always-predict-base-rate (0.075). BSS = −0.31.**
Baseline Brier for always predicting `p=0.0815` (DB winner rate) is `p(1-p) = 0.0749`. Model's reported test Brier is 0.0981. The XGBoost + Platt pipeline is adding noise, not information. Blast radius: any claim that the ML model "adds signal over market" is currently false — statistically, we'd be better off replacing ML with a constant.

**DS-02 — Backtest pipeline is architecturally disjoint from production recommender.**
`grep -n "recommender\|assign_bet\|create_combined_prediction" scripts/backtest.py` returns zero matches. Backtest pick generation (line 174-176): sort by blended prob, take top-1..top-4, check `issubset(actual_top2, our_top_K)`. Production (`core/recommender.py`): conviction gates (`MIN_TOP3_CONVICTION=0.30`, `MIN_TOP4_CONVICTION=0.38`), `_rank_has_strong_history` filter, market-favourite swap (`market_prob ≥ 0.30`), BANKER promotion. The two code paths produce different picks on an unknown fraction of races (Stage 1 estimates 10-20% via BANKER/market-fav swap alone). The +999%/+849%/+1064% are ROI of an unconditional box on ML-only probs, not the conditional strategy the dashboard actually emits. Blast radius: the entire box-4 pivot rested on backtest numbers that describe a pipeline we don't run.

### High

- **DS-03** — `weight_vs_field` train/predict mismatch. Training uses `0.5 - (weight - field_median)/20`; inference falls back to `1 - (weight-108)/27` because `horse_data['field_weights']` is never populated in `core/analyzer.py`. Mean `|Δ|=0.182` across 107 Apr 22 horses; 68.2% have `|Δ|>0.1`. Feature importance 4.66% — ~5% of ML signal is on a distribution the trees never saw. <1h fix.
- **DS-11** — Zero tests for `core/recommender.py`, `core/scorer.py::calibrate_probabilities`, Platt calibrator, `scripts/backtest.py` ROI math, and all 13 `analytics/` modules. DS-01..DS-04 are exactly the failures a minimal test harness would have caught. <1d to close the most important gaps.
- **P-07 ∩ DS-02** — Dynamic market-blend weights (±0.15 heuristic shifts at field-size 10 and distance 1400/1800) have no empirical validation, AND the backtest that could validate them doesn't call the recommender. With ML signal near-zero (P-01/P-04), these ungrounded weights are the de facto model.
- **P-08** — Harville independence model underestimates real HKJC quinella dividends. CLAUDE.md itself acknowledges this. No correlated-finish correction exists beyond an ad-hoc +15% placer boost in `exotic_calculator.py`. EV signal is structurally biased. Redesign tractability.
- **DS-04** — Wilson 95% CI on box-4 hit rate is [34.8%, 57.3%] (23pp wide). Parametric-bootstrap ROI 95% CIs: straight-Q [+385%, +1742%], box-3 [+588%, +1529%], box-4 [+462%, +984%] — heavily overlapping. "Box-4 +1064 vs box-3 +849" is statistically indistinguishable at n=73. Even if the backtest were measuring the right pipeline, the sample can't support the pivot.

### Medium

- **P-05** — Calibration curve bins in `scripts/accuracy.py` (max bin 0.25–1.0) are unreachable under current Platt max output 0.100. The "calibrated" label from this report is currently vacuous.
- **P-06** — Rule engine's base-rate-anchored sigmoid uses a magic slope `0.06` with no empirical fit. Opaque in backtest (backtest doesn't call rule engine — DS-02).
- **P-09** — `qin_bet_ev` mixes model-Harville hit probability with market-Harville dividend. Internally consistent in principle but given P-01/P-04 produce `model ≈ market` post-blend, EV always degenerates to −takeout × n_combos. Matches CLAUDE.md's observation that EV% is typically −30% to −60%.
- **DS-05** — `ingest_race_day()` calls `rebuild_stats()` which unconditionally deletes HKJC-rankings-scraped jockey/trainer stats without a source tag. Known-in-CLAUDE.md; no code guard.
- **DS-06** — `class_change` has std=0.038 (5-10× lower than every other feature). Mostly wasted XGBoost capacity, though still gets 5.64% importance on tail cases.
- **DS-07** — `experiment_log.high_correlations` reflects synthetic, not DB. After DB retrain, real correlations worth noting: `jockey_win_rate × jockey_trainer_combo r=+0.793` (combo feature mostly redundant), `recent_form × days_since_last_win r=+0.743` (intuitive, both informative).
- **DS-08** — Train/predict parity test uses `z_score > 5` threshold and allows 2 wildly-off features. Too loose — missed DS-03 entirely (z≈0.7 systematic bias).
- **DS-12** — Recency weights key on distinct dates, not rows. Busy Saturday ST meetings and quiet Wednesday HV meetings contribute equally per-date — imbalanced vs their row counts. 0.3 floor arbitrary.

### Low

- **P-10** — BANKER / market-fav swaps in `core/recommender.py` mutate `combined_probability` post-softmax+blend. Displayed probability no longer matches the model's internal probability. Documented in CLAUDE.md but violates the calibration invariant — `scripts/accuracy.py` measures the human override, not the model.
- **P-11** — `calibrate_probabilities` in `core/scorer.py:147-154` mixes model-only and model+market formulas in the same race when some horses have odds and others don't. Renormalize hides the issue in the sum but under-weights no-odds horses relatively.
- **DS-09** — `field_rel_draw` can exceed 1.0 when original draw > post-scratch field size (29/4403 rows, max observed 1.200). Out of convention; XGBoost learned on the out-of-range values but it's fragile.
- **DS-10** — `predict()` silently falls back to `1/field_size` when `is_trained==False`. Any silent-fail on pickle load produces uniform ML probs with no warning. Today benign (pickle loads); on regression becomes High.

### Informational

- **P-03** — Softmax temp=1.5 on Platt output with 0.006 span produces exponent diffs ≤0.009 → essentially uniform. Non-issue once P-01 is fixed. No direct action.
- **P-12** — Softmax uses `max(scores)` reference (standard, no issue). Noted for completeness.
- **DS-13** — Training set is 4,403 rows but only 359 winners / 36 dates; Platt sees ~74 positive validation examples. Winner count is the binding statistical constraint. Collecting one more season would roughly 2× positives.

---

## Strategic Critique

### Is the Quinella-First + Box-4 Default Defensible?

**No, on two independent grounds.**

**First:** the backtest that justified the pivot (`scripts/backtest.py` yielding +999% / +849% / +1064% for straight-Q / box-3 / box-4) does not exercise the production pipeline (DS-02). It runs ML-only → softmax → market-blend, takes top-K by blended probability unconditionally, and computes `issubset(actual_top2, top_K)`. Production emits bets through `core/recommender.py` which applies conviction gates, `_rank_has_strong_history`, BANKER promotion, and market-favourite swaps — any of which can change which horses end up in the box or SKIP the race entirely. The pivot was made on evidence that wasn't measuring the right thing. This is the core DS-02 finding and alone is sufficient to invalidate the pivot.

**Second:** even granting that the backtest *were* measuring production correctly, the sample cannot support the pivot statistically.

- n=72 evaluable races, with 33 box-4 hits (45.8%), 23 box-3 hits (31.9%), 11 straight-Q hits (15.3%).
- Wilson 95% CI on box-4 hit rate: **[34.8%, 57.3%]** — a 22.5pp band.
- Parametric-bootstrap 95% ROI CIs (DS-04):
  - Straight Q: **[+385%, +1742%]**
  - Box-3: **[+588%, +1529%]**
  - Box-4: **[+462%, +984%]**

The CIs overlap heavily. "Box-4 +1064 vs box-3 +849" is not statistically distinguishable at this sample size. Picking box-4 over box-3 is within noise.

**Third ground (interactive):** the rank-2 calibration narrative (`scripts/analyze_rank2.py`) that motivated the pivot — "33% of actual 2nd-placers rank 4+ in our model" — was computed against the same broken pipeline. With the calibrator (P-01) compressing all probabilities into an 0.012 band, within-race "rank-2 probability" is a near-uniform slice of noise. The observed empirical pattern (box-4 catches more rank-2s than box-3) might still be directionally true as a brute-force hit-rate fact, but the causal mechanism attributed to it — "rank-2 calibration issue" — is diagnostically unsupported. The finding sits on top of a compromised foundation.

**Recommendation:** roll the default recommendation back to the pre-pivot behaviour (straight Q or box-3 with conviction gate) until F-01, F-02, and F-03 land AND the backtest is re-run against the production pipeline AND the sample grows. Do not advertise a "+1064% box-4 ROI" in user-facing docs.

### Is "Relative EV" Ranking Actionable?

CLAUDE.md: *"usually negative (typically -30% to -60%)... The RELATIVE ranking across races (BEST EV vs LOW EV) is what identifies allocation priority."*

**No, not currently.** EV = model_P(hit) × dividend − stake. With Platt compressing all model probabilities into an 0.012-wide band (P-01), `model_P(hit)` ≈ constant across horses within a race and nearly constant across races. EV differences become driven almost entirely by dividend differences, which are driven by market probabilities — i.e. "relative EV" ranking degenerates to "rank races by which favourite the market dislikes most." That is not a model-derived allocation signal; it is a market-derived allocation signal, computed with extra steps.

`apr22-trace.json` shows EV% ranges of [−40%, −60%] across races with BEST-vs-LOW EV spreads ≤5pp — consistent with noise around a common market-driven value. Until P-01 is fixed (restoring model-market divergence), relative EV is not actionable. After fixing P-01, the *spread* between `model_Harville(i,j)` and `market_Harville(i,j)` per pair may become a real signal — but this requires re-measurement post-fix.

### Rank-2 Calibration Pivot — Sound Diagnosis?

The `scripts/analyze_rank2.py` output that drove the box-4 default concluded that 33% of actual 2nd-place finishers rank 4+ in our model, and therefore box-4 captures cases box-3 misses. Two issues:

1. **The probabilities that analysis was based on are compressed by 88×** (P-01). Within-race rank-2 probability is not a meaningful quantity — it's a near-uniform slice. "Why does our rank-2 probability miss the real rank-2 finisher?" presupposes rank-2 means something, which post-Platt it does not.
2. **The analysis was computed on the same ML-only backtest pipeline** (DS-02) that does not include rule engine, recommender, or BANKER swaps. Whatever pattern the analysis identified is a pattern in the backtest pipeline, not in production ranking.

The empirical observation (winners finish at ranks 5, 6, 8 on Apr 22) is real — Apr 22 had 3/9 winners at rank 5+. But the explanation "our rank-2 is uncalibrated" misattributes the mechanism. A more probable mechanism is: "the model has no signal on these winners" (maiden winners with zero DB history, e.g. CASA OF HONOR rank-8 @ $22 with `prior_win_rate=0.000`), which no reasonable box size will ever capture unless the model gains new information.

**Recommendation:** revisit the rank-2 analysis AFTER F-01/F-02 land, using the post-fix probability distribution. The diagnosis may look entirely different once ranking differentiation is restored.

### Sample Size Reality Check

Approximate binding: for a hit rate ~35% and dividend CV (ratio of std to mean) ~1.5–2, narrowing ROI 95% CI to ±5pp requires on the order of **500–1500 successful hits**, which at a 35% hit rate implies **~1500–4500 races**. We have 73. Even with every fix in place and a re-run backtest that measures production honestly, any strategic claim about ROI on this sample is a 10–50× under-sampled extrapolation.

This is not a reason to stop — it is a reason to (a) collect aggressively via `scrape_history.py --days 365` across prior seasons if possible, (b) footnote every ROI claim with its 95% CI, and (c) defer strategy pivots to evidence thresholds that the data can meet (e.g., "accept box-4 as default if after N≥300 races, box-4 CI lower bound exceeds box-3 CI upper bound").

---

## Fix List

Sort is by **severity × tractability**, with Critical + <1h first because those are the smoking guns that unlock re-measuring everything else. Fix-implementation uses the Validation Protocol at the end of this report.

### F-01 — Fit Platt on logit(val_proba), not val_proba
- **Finding IDs:** P-01 (Critical)
- **Severity:** Critical
- **Tractability:** <1h
- **Description:** In `ml/model.py:232-235`, change `self.calibrator.fit(val_proba.reshape(-1, 1), y_val, sample_weight=sw_val)` to fit on `logit(np.clip(val_proba, 1e-6, 1-1e-6))`. Mirror the transformation at inference in `ml/model.py:307-310`. Consider the alternative of switching to `sklearn.isotonic.IsotonicRegression` (non-parametric, handles non-linear miscalibration).
- **Success criterion:** After retrain, calibrator `coef_[0][0]` approximately 1.0 (pass-through for already-calibrated XGBoost). Measured post-Platt probability span on holdout ≥ 0.15 (vs current 0.012). Raw-vs-calibrated Spearman rank correlation within-race ≥ 0.95 (Platt preserves order).
- **Validation gate:**
  - **Unit tests:** Add `tests/test_calibration.py` — (a) test calibrator coef close to 1.0 after fit on well-calibrated inputs; (b) test post-calibration output spread ≥ 0.1 on a 12-horse race with raw probs spanning [0.05, 0.5]; (c) test rank-preservation (Spearman = 1.0). Currently NO tests exist for the calibrator (DS-11).
  - **Backtest regression:** Run `python3 -m scripts.backtest` before/after. Expected direction: rank-1 hit rate should rise (currently 16.4%); straight-Q hit rate should rise (currently 15.1%); Brier score should fall (currently 0.126 for top-pick). Accept the fix if **rank-1 hit rate Δ ≥ +0pp** AND **Brier Δ ≤ +0.000** (no regression) AND straight-Q hit rate non-negative delta.
  - **Accuracy report:** `python3 -m scripts.accuracy --all` — BSS must move from −0.31 toward 0 or positive. Any regression in BSS rejects the fix.
  - **What "regression" means here:** This fix is *expected* to move metrics (it's restoring signal). Negative movement on rank-1 hit rate or BSS is a hard reject. ROI metrics are noisy on 73 races (±5pp band); rank-1 hit rate is the primary gate because it's the least-noisy bounded metric.
- **Depends on:** F-02 should land first or simultaneously (otherwise F-01's retrain runs on synthetic again).

### F-02 — Retrain model on real DB data (overwrite synthetic pickle)
- **Finding IDs:** P-04, DS-01 (both Critical, convergent)
- **Severity:** Critical — production-blocker
- **Tractability:** <1h to run retrain; <1d to add guard
- **Description:** Run `python3 -m scripts.train` to regenerate `models/hkjc_racing_model.pkl` on the 4,403-row DB. Verify `experiment_log.json` latest entry shows `n_samples ≥ 4000`. Back up the current (synthetic) pickle to `models/hkjc_racing_model.pkl.synthetic-backup` before overwrite. Separately (part of this fix): add a guard in `ml/model.py:_save_model()` that refuses to save when `len(training_features) < 1000` OR when `os.environ.get('PYTEST_CURRENT_TEST')` is set. Point `tests/conftest.py:149-151` at `tmp_path / 'test_model.pkl'` instead of the default.
- **Success criterion:** Shipped pickle's experiment entry shows `n_samples=4403` (or current DB row count); `tests/conftest.py` fixture no longer overwrites production pickle (verify: run pytest, inspect `models/hkjc_racing_model.pkl` mtime/sha256 before and after — must be unchanged).
- **Validation gate:**
  - **Unit tests:** Add `tests/test_model_save_guard.py` — (a) calling `_save_model()` with `n_features < 1000` raises; (b) calling within pytest context raises or writes to test path; (c) production pickle sha256 is unchanged after `pytest tests/` full run.
  - **Backtest regression:** Run `python3 -m scripts.backtest` before/after. Pre-fix baseline is **not meaningful** (it measures the synthetic model). After fix, capture new baseline. Expected direction: rank-1 hit rate may rise materially once real signal is present; Brier should improve. The post-fix backtest IS the new baseline for all subsequent fixes.
  - **Accuracy report:** `python3 -m scripts.accuracy --all` — BSS must be > −0.10 (i.e., model is at worst slightly worse than base rate — ideally BSS > 0 meaning real skill). If post-retrain BSS is still −0.31, the Platt issue (F-01) dominates and F-01 must land before F-02 is re-evaluated.
  - **What "regression" means here:** N/A to the pre-fix baseline because that baseline is a pytest artifact. Post-fix metrics define the real baseline.
- **Depends on:** None. Should land first of all fixes — it defines the meaningful baseline.

### F-03 — Make backtest exercise production recommender pipeline
- **Finding IDs:** DS-02 (Critical)
- **Severity:** Critical
- **Tractability:** <1d
- **Description:** Extend `scripts/backtest.py` to (a) instantiate `CombinedRaceAnalyzer` and run the full rule engine + ML blend (60/40) for each held-out race, (b) call `core/recommender.py::assign_bet_recommendations` on the analyses, (c) filter ROI measurement to only races where production emits a non-SKIP bet with a specific bet_type (WIN/QIN/BOX Q/BOX Q4/TRIO/BANKER), (d) measure straight-Q / box-3 / box-4 ROI conditional on the actual bet_type produced (e.g., box-4 ROI is measured only on races where production said "BOX Q4"). Keep the existing unconditional-issubset path alongside as a sanity reference ("theoretical upper bound") but rename it and promote the recommender-conditional path to primary.
- **Success criterion:** Backtest output includes, per bet type, (a) number of races production would have bet, (b) hit rate, (c) ROI, (d) Wilson 95% CI on hit rate, (e) bootstrap 95% CI on ROI. The headline "box-4 ROI +1064%" figure either disappears (replaced with conditional number) or is clearly labeled as "unconditional box-4 upper bound; production bets on K races for ROI X%".
- **Validation gate:**
  - **Unit tests:** Add `tests/test_backtest.py` — (a) ROI math closes (sum of `hits × dividend − n_bets × stake` matches reported ROI to $0.01); (b) at least one race per bet_type is correctly classified in a fixture; (c) SKIP races are excluded from ROI denominators.
  - **Backtest regression:** N/A to the "backtest produces same numbers" sense — by design this fix changes what the backtest measures. Run before/after to capture the delta between "unconditional box-4" (pre) and "conditional recommender box-4" (post). Accept if the conditional number has a sensible relationship to the unconditional (typically lower hit rate on fewer races, but per-race ROI can go either direction).
  - **Accuracy report:** Unchanged by this fix (accuracy report doesn't depend on backtest). Not regressed.
  - **What "regression" means here:** This fix is *expected* to change every ROI number reported. "Regression" here means the math is wrong (e.g., ROI doesn't match `sum(wins) / sum(stakes) − 1`) or the conditional filter is dropping too many races. Reject if fewer than 40% of 73 races produce a non-SKIP recommendation (suggests the conviction gates are over-tight and must be re-tuned separately).
- **Depends on:** F-02 (otherwise the rule engine + ML blend feeds on synthetic model).

### F-04 — Prevent test code from overwriting production pickle
- **Finding IDs:** DS-01 (ancillary)
- **Severity:** Critical (production hygiene)
- **Tractability:** <1h
- **Description:** In `tests/conftest.py:140-152`, change the `trained_model` fixture to instantiate the model with an explicit `model_path=tmp_path / 'test_model.pkl'`. Requires `HKJCRacingMLModel.__init__` to accept an optional path override — add the parameter if absent, default to `config.MODEL_PATH` for production callers. Alternatively (belt-and-braces): add the guard in F-02 that refuses to save inside pytest.
- **Success criterion:** After change, running `pytest tests/` does NOT modify `models/hkjc_racing_model.pkl` — verified by sha256 before/after.
- **Validation gate:**
  - **Unit tests:** Already covered by F-02's `tests/test_model_save_guard.py`.
  - **Backtest regression:** Not applicable (this is pure test isolation).
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** This is pure hygiene — no metric movement expected. Regression = production pickle changes during `pytest` run.
- **Depends on:** F-02 (guard should be added as part of F-02).

### F-05 — Add pre-bet pipeline integrity check
- **Finding IDs:** P-04 (defensive), DS-10 (related)
- **Severity:** High
- **Tractability:** <1h
- **Description:** In `generate_predictions.py` (or `dashboard/generator.py`), before emitting any bet recommendation: (a) assert `ml_model.is_trained` is True; (b) assert the loaded pickle's `n_samples` (or equivalent recorded in `experiment_log.json` indexed by model version) is ≥ 1000; (c) assert the pickle's training correlations match the DB fingerprint, not the synthetic-seed-42 fingerprint. If any check fails, refuse to emit predictions and surface a banner in the dashboard: "MODEL INTEGRITY CHECK FAILED — DO NOT BET".
- **Success criterion:** Running `python3 generate_predictions.py` against the current synthetic pickle raises and refuses to emit. Running against a real-DB-trained pickle succeeds.
- **Validation gate:**
  - **Unit tests:** Add `tests/test_integrity_check.py` — (a) integrity check fails on synthetic pickle; (b) passes on real-DB pickle; (c) banner appears in HTML output when check fails.
  - **Backtest regression:** Not applicable (pre-bet check, not a metric).
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** Pure defensive — regression = legitimate predictions blocked or illegitimate predictions emitted.
- **Depends on:** F-02.

### F-06 — Fix weight_vs_field train/predict mismatch
- **Finding IDs:** DS-03 (High)
- **Severity:** High
- **Tractability:** <1h
- **Description:** In `core/analyzer.py:1031-1057`, populate `horse_data['field_weights']` with `[int(h.weight) for h in race.horses if h.weight]` before calling `extract_features`. Remove or deprecate the `1 - (weight - 108)/27` absolute-range fallback branch in `ml/features.py:383-391` — or keep it only for true edge cases when field_weights is empty, and loudly warn.
- **Success criterion:** Per-horse parity: for a sample of 100 real DB rows, `extract_features(horse_data, ...)` output for `weight_vs_field` matches `build_training_data_from_db()` output for the same row within `|Δ| ≤ 0.02`. Feature distribution at inference matches the training distribution on KS test p-value ≥ 0.05.
- **Validation gate:**
  - **Unit tests:** Add to `tests/test_data_accuracy.py` — stricter per-feature parity test (DS-08 also asks for this). Assert `weight_vs_field` delta < 0.02 for 20 random DB rows. Tighten the existing `test_training_and_predict_features_same_distribution` z-score threshold from 5 to 2 AND remove the `<=2` wildly-off tolerance.
  - **Backtest regression:** Expected direction: this fix affects ~5% of ML signal. Rank-1 hit rate and Brier may move slightly in either direction. Accept if neither regresses by more than 1pp (hit rate) or 0.005 (Brier).
  - **Accuracy report:** BSS movement ≤ 0.03 in either direction is acceptable.
  - **What "regression" means here:** This is correcting a bug, not adding signal — the feature distribution at inference simply matches training. Effect should be small.
- **Depends on:** F-02 (otherwise delta measures against synthetic baseline).

### F-07 — Test coverage for recommender / calibrator / blend / backtest
- **Finding IDs:** DS-11 (High)
- **Severity:** High
- **Tractability:** <1d
- **Description:** Add minimal but sufficient tests for the zero-coverage areas. Specifically:
  - `tests/test_recommender.py`: exercise `assign_bet_recommendations` with fixture analyses covering each bet_type (WIN, QIN, BOX Q, BOX Q4, TRIO, BANKER, SKIP). Assert conviction gates, market-fav swap, BANKER promotion trigger correctly.
  - `tests/test_calibration.py`: specified in F-01 above.
  - `tests/test_scorer_blend.py`: assert `_dynamic_blend_weights` returns correct (model_w, market_w) for each branch (field ≥10 + sprint, field <8 or long, default).
  - `tests/test_backtest.py`: specified in F-03 above.
- **Success criterion:** `pytest tests/` includes ≥15 new tests covering the previously-zero-coverage modules. Test files pass.
- **Validation gate:**
  - **Unit tests:** These ARE the unit tests. Gate: all new tests pass on main; passing new tests fail on an intentionally broken version of each module (tests have real teeth).
  - **Backtest regression:** Not applicable.
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** New tests must fail on broken code. If they all pass regardless of what I break, they're decorative. Reject.
- **Depends on:** None (adds only, doesn't change prod logic).

### F-08 — Empirically fit rule-engine sigmoid slope
- **Finding IDs:** P-06 (Medium)
- **Severity:** Medium
- **Tractability:** <1d
- **Description:** Build a `(rule_score, won)` table by running the rule engine over the DB race cards we have (where horse data was logged at prediction time). Fit `logit(win_rate_in_decile) = a + b × score` via Logistic Regression with binned data. Replace the magic `0.06` in `ml/model.py:353` with the fitted `b`. Document the fit and the sample.
- **Success criterion:** New slope is within a plausible range (0.03–0.10). Calibration of rule_score-to-win-prob improves — decile-binned predicted matches actual win rate within 3pp on at minimum 70% of deciles.
- **Validation gate:**
  - **Unit tests:** Add `tests/test_rule_sigmoid.py` — (a) slope is fitted (not hardcoded) from DB; (b) rule_score=100 produces prob substantially above base rate; (c) rule_score=0 produces prob substantially below base rate.
  - **Backtest regression:** Run before/after. Rule engine only enters production through 60/40 blend in `CombinedRaceAnalyzer`. Expected: marginal effect on ROI (rule engine is 60% weight but rule_score is the dashboard's main output; backtest doesn't currently call it — will once F-03 lands). Accept if BSS improves or stays flat.
  - **Accuracy report:** Monitor BSS.
  - **What "regression" means here:** This is a known-arbitrary choice being grounded. Should improve calibration. Hard reject if BSS drops by more than 0.02.
- **Depends on:** F-03 (so the backtest can actually measure this).

### F-09 — Grid-sweep and justify market-blend weights
- **Finding IDs:** P-07 (High)
- **Severity:** High — but gated on other fixes
- **Tractability:** <1d
- **Description:** After F-01/F-02/F-03 land, use `python3 -m scripts.backtest --sweep` to grid over `model_w ∈ [0.25, 0.35, 0.45, 0.55, 0.65]` × `field_size_threshold ∈ [8, 10, 12]` × `distance_threshold ∈ [1200, 1400, 1600]`. Pick the grid point that maximises a primary objective (BSS or median box-3 ROI lower-bound CI). Document the choice and rationale in CLAUDE.md. If multiple grid points are statistically indistinguishable, prefer the simpler (constant) weight schedule.
- **Success criterion:** Market-blend weights have a documented empirical basis. The ±0.15 shifts are either justified or replaced with a constant.
- **Validation gate:**
  - **Unit tests:** Add `tests/test_blend_weights.py` — (a) blend weights sum to 1; (b) chosen grid point beats default by a measured metric on a test fold.
  - **Backtest regression:** Run before/after on the full holdout. Accept if chosen point beats current heuristic on BSS OR box-3 ROI lower-bound CI.
  - **Accuracy report:** BSS improves or stays flat.
  - **What "regression" means here:** This fix REPLACES the current heuristic. Gate: new weights must beat or match current on primary metric (BSS).
- **Depends on:** F-01, F-02, F-03.

### F-10 — Correlated-finish / empirically-fitted dividend multiplier
- **Finding IDs:** P-08 (High)
- **Severity:** High
- **Tractability:** redesign (full correlated-finish model) or <1d (empirical multiplier)
- **Description:** Pragmatic path (<1d): using the `race_dividends` table, fit `dividend_multiplier = actual_divid / harville_estimate` as a function of `top_2_share` and `field_size`. Apply the multiplier in `betting/dividends.py::estimate_quinella_dividend` after the Harville step. Rigorous path (redesign): replace Harville with a Stern (1990) or Lo & Bacon-Shone permutation model. Start with pragmatic; Stern is for a later cycle.
- **Success criterion:** Predicted quinella dividend has median absolute error ≤ 40% of actual HKJC dividend on the holdout (vs current "grossly underestimated" per CLAUDE.md). EV% no longer degenerates to always-negative — at least 10% of races show positive model-edge EV.
- **Validation gate:**
  - **Unit tests:** Add `tests/test_dividend_estimator.py` — (a) predicted dividend within 40% MAPE of actual on held-out dividends; (b) positive-EV case exists on a crafted fixture where model assigns substantially higher P(hit) than market.
  - **Backtest regression:** Run before/after. Box-Q ROI should rise or stay flat. Reject if box-3 or box-4 ROI drops by >5pp (noise).
  - **Accuracy report:** Not directly applicable to this fix.
  - **What "regression" means here:** This is fixing a known underestimate. Should make EV calc actionable. Reject if EV% post-fix is still always-negative.
- **Depends on:** F-01, F-02 (EV signal requires restored model probs).

### F-11 — Tighten feature parity test
- **Finding IDs:** DS-08 (Medium)
- **Severity:** Medium
- **Tractability:** <1h
- **Description:** Replace `tests/test_data_accuracy.py:259-297::test_training_and_predict_features_same_distribution` with a per-horse parity test: for N=20 random DB rows, call both `extract_features(horse_data, ...)` and the corresponding row from `build_training_data_from_db()`; assert `|Δ| < 0.02` element-wise for every feature. Remove the `<=2 wildly-off features` tolerance.
- **Success criterion:** Test catches DS-03 when the F-06 fix is reverted. Test catches any future train/predict divergence of magnitude ≥0.02 on any feature.
- **Validation gate:**
  - **Unit tests:** The replacement test IS the gate. It must fail on the pre-F-06 code and pass on the post-F-06 code.
  - **Backtest regression:** Not applicable.
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** This tightens a test. Regression = test is too strict and flags legitimate intentional differences. Tune threshold if true positives show up.
- **Depends on:** F-06 (otherwise the tightened test fails on current code).

### F-12 — Recalibration curve bins (equal-count) in accuracy report
- **Finding IDs:** P-05 (Medium)
- **Severity:** Medium
- **Tractability:** <1h
- **Description:** In `scripts/accuracy.py:112`, replace fixed-edge bins `[(0,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.25),(0.25,1.0)]` with equal-count deciles computed from observed `calibrated_prob` distribution. Document whether the report uses post-Platt, post-softmax, or post-blend prob (currently ambiguous).
- **Success criterion:** Calibration curve has at least 5 bins populated (not all concentrated in one range). All bins have n ≥ 20 samples.
- **Validation gate:**
  - **Unit tests:** Add to `tests/test_accuracy.py` — deciles are computed per-run; no bin with n < 20 is reported unless the dataset is <100 samples.
  - **Backtest regression:** Not applicable (accuracy report is read-only analytics).
  - **Accuracy report:** This fix changes what the report shows. Run before/after to document the change. Not a regression check — this is a display fix.
  - **What "regression" means here:** Pure display — regression = report format is broken or ambiguous.
- **Depends on:** None (can land in parallel with anything).

### F-13 — Add source tag to jockey_stats / trainer_stats
- **Finding IDs:** DS-05 (Medium)
- **Severity:** Medium
- **Tractability:** <1h
- **Description:** Alter `jockey_stats` and `trainer_stats` schemas to include a `source` column ('local' or 'hkjc_rankings'). Have `rebuild_stats()` skip rows where `source='hkjc_rankings'`. Have `scrape_history.py --rankings` write with `source='hkjc_rankings'`.
- **Success criterion:** Running `ingest_race_day()` no longer wipes HKJC rankings. User no longer needs to re-run `--rankings` after every ingestion.
- **Validation gate:**
  - **Unit tests:** Add `tests/test_rebuild_stats.py` — ingest a race, verify HKJC-rankings rows survive.
  - **Backtest regression:** Not applicable.
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** Pure data integrity — regression = HKJC rankings still wiped.
- **Depends on:** None.

### F-14 — Demote class_change or widen its scale
- **Finding IDs:** DS-06 (Medium)
- **Severity:** Medium
- **Tractability:** <1h
- **Description:** Current `class_change` has std=0.038. Either (a) rescale via `0.5 + diff × 0.2` clipped to [0,1], or (b) convert to a ternary indicator `{dropping: 0, same: 0.5, rising: 1}`. Document choice in CLAUDE.md.
- **Success criterion:** Feature std ≥ 0.1 post-change. Feature importance measured via retrain — tolerable drop.
- **Validation gate:**
  - **Unit tests:** Test feature std ≥ 0.1 on DB training sample.
  - **Backtest regression:** Run before/after retrain. Accept if Brier neutral-to-positive.
  - **Accuracy report:** BSS neutral-to-positive.
  - **What "regression" means here:** This rescales a feature. If the trees already learned the tail signal well, rescaling may hurt slightly. Accept ±0.01 BSS change.
- **Depends on:** F-02 (retrain cycle).

### F-15 — Drop jockey_trainer_combo (redundant with jockey_win_rate)
- **Finding IDs:** DS-07 (Medium)
- **Severity:** Medium
- **Tractability:** <1h
- **Description:** Post-retrain on DB, `jockey_win_rate × jockey_trainer_combo r=+0.793`. Remove `jockey_trainer_combo` from the 23-feature schema. Update `ml/feature_schema.py`, `extract_features()`, and `build_training_data_from_db()`. Bump `n_features` to 22.
- **Success criterion:** Model retrains with 22 features. BSS does not drop by more than 0.01.
- **Validation gate:**
  - **Unit tests:** Existing `test_no_constant_training_features` plus the tightened `test_training_and_predict_features_same_distribution` must still pass.
  - **Backtest regression:** Run before/after. Accept if rank-1 hit rate Δ ≥ −2pp and BSS Δ ≥ −0.01.
  - **Accuracy report:** BSS neutral-to-positive.
  - **What "regression" means here:** Feature removal. Expected neutral since DS-07 shows the feature is near-duplicate. Hard reject on BSS drop > 0.02.
- **Depends on:** F-02, F-01 (clean baseline needed to measure).

### F-16 — Clip field_rel_draw to [0,1]
- **Finding IDs:** DS-09 (Low)
- **Severity:** Low
- **Tractability:** <1h
- **Description:** In `ml/features.py:281` and `ml/training.py:245`, clip: `min(1.0, draw / max(1, field_size))`. Or reformulate: `(draw - 1) / max(1, field_size - 1)`.
- **Success criterion:** Feature max value ≤ 1.0 across full DB.
- **Validation gate:**
  - **Unit tests:** Test feature ≤ 1.0 on 100 random DB rows.
  - **Backtest regression:** Negligible — 29/4403 rows affected.
  - **Accuracy report:** Negligible.
  - **What "regression" means here:** Pure convention fix.
- **Depends on:** F-02 (retrain cycle).

### F-17 — Raise exception on untrained model in predict()
- **Finding IDs:** DS-10 (Low today / High on regression)
- **Severity:** Low currently, High on any load failure
- **Tractability:** <1h
- **Description:** In `ml/model.py:299-302`, change silent `1/field_size` fallback to raise `RuntimeError("Model not trained — refuse to predict")`. Caller in `core/analyzer.py` catches and surfaces as a dashboard banner.
- **Success criterion:** Any model-load failure produces a visible error, not silent uniform probs.
- **Validation gate:**
  - **Unit tests:** Test `predict()` on untrained model raises.
  - **Backtest regression:** Not applicable (backtest assumes loaded model).
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** Pure safety. Regression = silent fallback preserved.
- **Depends on:** None.

### F-18 — Per-row recency weights instead of per-date
- **Finding IDs:** DS-12 (Medium)
- **Severity:** Medium
- **Tractability:** <1h
- **Description:** In `ml/training.py:137-144`, change recency weights from per-date-rank linear interpolation to per-row linear interpolation. Busy race days contribute proportional to their row count. Optionally: exponential decay `exp(-λ × days_before_test)` with λ fit via cross-validated logloss.
- **Success criterion:** Weights are proportional to row index, not date. Documented.
- **Validation gate:**
  - **Unit tests:** Test weight is monotonically increasing with row index.
  - **Backtest regression:** Run before/after. Accept if BSS neutral-to-positive.
  - **Accuracy report:** Monitor BSS.
  - **What "regression" means here:** Weight curve change. Effect modest. Accept ±0.01 BSS band.
- **Depends on:** F-02.

### F-19 — Preserve display_probability vs model_probability distinction
- **Finding IDs:** P-10 (Low)
- **Severity:** Low
- **Tractability:** <1h
- **Description:** In `core/recommender.py:50-81` and `_resync_ranks_and_probs`, keep both a `model_probability` (pre-swap) and `display_probability` (post-swap). `scripts/accuracy.py` uses `model_probability` for calibration; dashboard shows `display_probability`.
- **Success criterion:** Calibration curve measures the model, not human overrides.
- **Validation gate:**
  - **Unit tests:** Test both values exist and differ after a BANKER swap fixture.
  - **Backtest regression:** Not applicable.
  - **Accuracy report:** Calibration curve may change — document the change.
  - **What "regression" means here:** Display separation.
- **Depends on:** None.

### F-20 — Fix mixed-odds blend formula
- **Finding IDs:** P-11 (Low)
- **Severity:** Low
- **Tractability:** <1h
- **Description:** In `core/scorer.py:147-154`, for horses without `market_prob > 0`, scale consistently with odds-having horses. Simplest fix: upstream in `core/recommender.py:111`, SKIP the entire race if any horse lacks odds (already done for pre-race). For current mixed case: use `b = cal_probs[i]` but apply the same post-normalize so relative weights are preserved.
- **Success criterion:** Mixed-odds races produce distribution where no-odds horses are weighted consistently with others.
- **Validation gate:**
  - **Unit tests:** Test mixed-odds fixture produces coherent distribution.
  - **Backtest regression:** Not applicable (backtest uses races with odds).
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** Edge case correctness.
- **Depends on:** None.

### F-21 — Collect additional historical data to grow holdout
- **Finding IDs:** DS-04, DS-13 (High + Informational, sample-size constraint)
- **Severity:** High (ceiling on every ROI claim)
- **Tractability:** needs-data
- **Description:** Run `python3 scrape_history.py --days 365` (or further) to capture at least one full prior season (~80 additional meetings × ~100 rows = ~8000 more rows). Re-run `python3 -m scripts.train`. Re-run `python3 -m scripts.backfill_dividends --force` on the new dates. Re-run backtest — holdout grows proportionally.
- **Success criterion:** DB winner count ≥ 700 (doubled). Holdout grows to ≥ 150 races. ROI CI band narrows to roughly ±8pp or better on box-4.
- **Validation gate:**
  - **Unit tests:** Not applicable (data collection).
  - **Backtest regression:** Larger holdout yields tighter CIs; per-race ROI may move in either direction as the sample becomes representative. Accept any direction — this is a measurement improvement, not a strategy change.
  - **Accuracy report:** BSS recomputed on larger sample.
  - **What "regression" means here:** More data is always directionally better; but if BSS drops substantially on the larger sample, the current "good" BSS on 73 races was a lucky stratum. Either way, inform future decisions.
- **Depends on:** None but pairs best with F-02 (retrain on enlarged data).

### F-22 — Re-evaluate box-4 pivot after F-01/F-02/F-03/F-21
- **Finding IDs:** Strategic (post-fix evaluation)
- **Severity:** High (strategic, decision-gating)
- **Tractability:** needs-data (post-fix backtest)
- **Description:** After F-01/F-02/F-03 land (and ideally F-21 for more data), re-run `python3 -m scripts.backtest`. Compare straight-Q vs box-3 vs box-4 on (a) hit rate with Wilson CI, (b) ROI with bootstrap CI. If box-4's CI lower bound exceeds box-3's CI upper bound, the pivot is supported. If not, default to box-3 or straight-Q conditional on conviction gate. Update CLAUDE.md to reflect the new evidence.
- **Success criterion:** The default recommendation in `core/recommender.py` is re-chosen based on post-fix measurement, not pre-fix measurement. Documentation reflects the evidence.
- **Validation gate:**
  - **Unit tests:** Whatever default is chosen, `tests/test_recommender.py` asserts it.
  - **Backtest regression:** THIS fix IS the backtest re-run. No additional gate.
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** This is a strategic re-evaluation — outcome may be "revert box-4 → box-3" or "confirm box-4." Either is acceptable.
- **Depends on:** F-01, F-02, F-03 (plus ideally F-21).

### F-99 — Do NOT advertise new ROI / strategy improvements from post-fix backtest until sample grows to 150+ races
- **Finding IDs:** DS-04, DS-13 (strategic restraint)
- **Severity:** Informational
- **Tractability:** N/A
- **Description:** Explicit non-change: even after F-01/F-02/F-03 land and backtest shows improved numbers, do not publish new "+X%% ROI" claims in CLAUDE.md, dashboard banners, or WhatsApp copy until holdout ≥ 150 races (roughly 2× current). All ROI numbers should carry their Wilson + bootstrap CIs. Keep F-22's decision conditional on CI-separation, not point estimates.
- **Success criterion:** No user-facing "ROI +X%" claims in docs or UI without accompanying CI and sample-size footnote.
- **Validation gate:**
  - **Unit tests:** Not applicable.
  - **Backtest regression:** Not applicable.
  - **Accuracy report:** Not applicable.
  - **What "regression" means here:** Process discipline only.
- **Depends on:** N/A.

---

## Acknowledged Limitations

- **73-race sample:** Wilson CIs on any single ROI claim are 20pp+ wide. Bootstrap ROI CIs are ±500pp+. No strategy choice is statistically well-supported at this sample size. F-21 addresses this; until it lands, every ROI number in this audit and elsewhere carries that caveat.
- **Audit is one-shot:** Systemic fixes change the system; periodic re-audit (quarterly or per-50-new-races) is appropriate once the initial fix-pass lands.
- **Persona overlap ~20% by finding count:** Two strong convergence points (P-04 ∩ DS-01, P-01 ∩ DS-02) and three partial overlaps out of 25 findings. The parallel read produced complementary rather than duplicate findings — approach validated for this audit class; no recommendation to fold to single-threaded review next time.
- **Stage 3 strategic conclusions are caveated by DS-02:** The backtest measures a different pipeline than production. Every strategic claim in this report (box-4 pivot, relative EV, rank-2 analysis) is conditional on that caveat until F-03 lands.
- **Probability-expert and data-scientist subagents each listed things they did NOT review** — notably Kelly sizing (`betting/optimizer.py`), exotic-boost factors in `exotic_calculator.py`, rule-engine factor weights, live-odds scraper reliability, and a full pytest run. These are candidates for a future audit cycle or scoped spot-checks.
- **Sample for per-race feature parity (DS-03):** measured on a single date (2026/04/22, 107 horses). The root cause is deterministic (missing `field_weights` in `horse_data`) so the pattern applies everywhere, but the specific magnitude distribution is from one slice.
- **DS-04 ROI CIs approximate hit-dividend distributions from summary stats** (min/median/mean/max in `raw/backtest-output.txt`) because raw per-hit dividends are not preserved in that file. The true ROI CI could be modestly narrower if the actual dividend distribution is tighter than the approximation.

---

## Next Steps for User

1. Review this report and mark each fix as: **accepted | deferred | rejected**, with optional priority reordering.
2. For accepted fixes, open a new brainstorm / writing-plan cycle scoped to fix implementation. Every accepted fix enters the fix-implementation plan with its Validation Gate attached. The fix-plan's execution requires running `pytest tests/` + `python3 -m scripts.backtest` + `python3 -m scripts.accuracy --all` before and after each change, and rejecting any change that regresses the primary metric for that fix's finding.
3. **Strongly recommend F-02 and F-04 are landed first and together** — otherwise every subsequent fix is measured against a synthetic baseline.
4. **Strongly recommend F-01 lands second** — it's a one-line change that restores ML signal and unlocks meaningful measurement of every downstream fix.
5. **F-03 third** — once backtest measures production pipeline, the remaining fix set can be evaluated honestly.
6. Dashboard redesign is still queued as a third workstream, explicitly outside this audit's scope.

---

## Fix-Implementation Validation Protocol (for the downstream plan)

When the separate fix-implementation plan is written and executed:

- Each fix is implemented on its own commit.
- Before committing: run `pytest tests/` and `python3 -m scripts.backtest` and `python3 -m scripts.accuracy --all` against the pre-fix state, capture metrics.
- After implementing: re-run all three, capture metrics, compute deltas on:
  - Rank-1 hit rate (primary for signal fixes)
  - Top-pick Brier score (primary for calibration fixes)
  - Straight-Q / box-3 / box-4 ROI (monitoring, wide noise band on 73 races)
  - BSS (primary for overall skill)
  - Calibration curve bin alignment (for calibration fixes)
- If any primary metric for this fix's target regresses beyond the ±3pp ROI band / ±0.01 BSS band and the fix wasn't expected to move that metric, the fix is reverted or re-scoped.
- Multi-fix batches are re-run end-to-end at the end to confirm interactions don't regress the overall strategy.

**IMPORTANT:** Given DS-01 (synthetic-trained pickle) and DS-02 (backtest uses wrong pipeline), **the pre-fix backtest numbers themselves should be treated as a broken baseline**. The first fix to land must be **F-02 (retrain on real data)** simultaneous with **F-04 (prevent test overwrite)**, immediately followed by **F-01 (Platt logit)** and **F-03 (backtest matches production)**. Only after those four land does the backtest become a meaningful baseline for measuring all subsequent fixes. Landing any other fix first risks chasing metrics that don't measure what we think they measure.
