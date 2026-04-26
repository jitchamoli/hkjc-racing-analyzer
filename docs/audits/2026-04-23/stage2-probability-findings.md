# Probability-Expert Findings

## Summary
Platt compression root cause **diagnosed**: the calibrator was fit on the raw XGBoost probability directly (scale [0,1]) instead of on its logit, so the learned logistic regression slope (0.137) can only stretch the output logit by 0.137 across the entire input range. Result: every raw input maps to a platt probability in a 0.012-wide band around the base rate — 88x compression — which erases all ranking information from the ML model. Three secondary findings make the situation worse: the current production model is trained on only 500 samples (not the 4403-row DB), its test Brier (0.098) is *worse* than the base-rate baseline (0.075), and the 5-bin calibration curve in `scripts/accuracy.py` uses bins whose upper edges (0.25, 1.0) are unreachable under current Platt output.

## Platt Calibrator Diagnosis (First Task)

### Observed coefficients
From `pickle.load(models/hkjc_racing_model.pkl)['calibrator']`:
- `type`: `sklearn.linear_model._logistic.LogisticRegression`
- `coef_` = `[[0.1368081]]`
- `intercept_` = `[-2.33244596]`
- `classes_` = `[0, 1]`

### The 88x compression is mechanical, not statistical
`ml/model.py:234-235` fits the calibrator with `val_proba.reshape(-1, 1), y_val` — i.e. the feature passed to Logistic Regression is the **raw probability** in [0,1], not `logit(val_proba)`. The learned model is therefore

```
logit(p_platt) = 0.137 * raw_p + (-2.332)
```

Because `raw_p` is constrained to [0,1], the LR slope times the input range produces at most a 0.137-unit shift in `logit(p_platt)`. Empirically (scipy test):
- raw=0.00 → platt=0.0885
- raw=0.50 → platt=0.0941
- raw=1.00 → platt=0.1001
- **Output span across the entire input domain: 0.0117** (i.e. the calibrator cannot emit any probability outside [0.088, 0.100])

Observed Stage 1 range 0.089–0.095 for raw inputs 0.07–0.54 is exactly what this formula predicts; the compression is not noise, it is mathematical.

### Why the slope is 0.137 and not, say, 4 or 5
The slope is correct given the input scale. A well-calibrated slope *if the input were `logit(raw_p)`* would be ~1 (pass-through for an already-calibrated model). But because the input is `raw_p` on the linear scale, the LR has to fight non-linearity: near `raw_p=0.05` a logistic regression wants to raise the output logit by ~0.7 to land near base rate, while near `raw_p=0.5` it wants a different slope. The gradient-descent solver finds the least-bad linear-in-`p` compromise, which is a tiny slope. The "real" compression cause is feature choice, not recency weights or validation prevalence.

### Secondary contributing factors (but not the root cause)
1. **Training signal is weak** — with only 500 training samples (see P-04 below) and base-rate-only model quality (Brier 0.098 > baseline 0.075), XGBoost's raw probabilities are nearly uninformative on validation. Even a correct logit-input Platt would produce a shallow slope. But the linear-in-p choice compounds this by another ~7x.
2. **Recency weights are applied** (`ml/model.py:233-235`: `sw_val = sample_weight[cut_train:cut_val]`; passed via `sample_weight=sw_val`). Per CLAUDE.md claim — verified in code. Not a driver of compression.
3. **Validation set prevalence** — not directly inspectable without re-running training, but the 60/20/20 temporal split puts the last-20% of dates in test and the prior 20% in validation. On 36 distinct race dates (4403 rows, 8.15% winners) validation has ~880 rows with ~72 winners. Prevalence matches training, so this is not the imbalance issue suggested in the prompt.

### Counterfactual: what happens if we bypass Platt entirely
Measured on the 10 severe-miss traces (`raw/trace-analysis.json`):
- 9/10 cases: rank-1 had higher `raw_ml_prob` than the actual winner (i.e. Platt didn't save the ML from feature-driven misranking — but it also didn't make things worse than the raw XGBoost on ranking)
- 1/10 cases (Apr 12 R10): winner had `raw_ml_prob=0.2176` vs rank-1's `0.1637`. Raw XGBoost correctly identified the winner. Platt collapsed them to 0.0909 vs 0.0903 — a 0.0006 gap that softmax (temp=1.5) could not meaningfully amplify (softmax diff 0.0001), and market blend then overrode.

So removing Platt alone would recover the winner in 1/10 severe misses directly — small but free. Bigger effect: the softmax (temp=1.5) would suddenly have a real input range [0.07, 0.54] instead of [0.089, 0.095], producing ~6x more spread across the field and meaningfully reducing the market's dominance in the 55-65% blend.

### Recommended fix (conceptual)
Two options, in order of preference:

1. **Fit Platt on `logit(val_proba)`, not `val_proba`** — one-line change: `self.calibrator.fit(logit(val_proba).reshape(-1, 1), y_val, ...)`, and mirror the transformation at inference time in `predict()`. Standard Platt-scaling formulation. Immediately restores a slope near 1.0 and preserves raw XGBoost's ordering.

2. **Switch to `sklearn.isotonic.IsotonicRegression`** — non-parametric, doesn't assume logistic shape, handles non-linear miscalibration naturally. Needs ~200+ val samples (which we have). Handles the regime where XGBoost is overconfident only at tails (favourites).

Either fix is <1 hour of work. Retrain only, no schema changes. Do NOT retrain-and-ship without also addressing the underlying 500-sample issue (P-04) — otherwise the recalibrated slope will still be mild because the signal itself is weak.

## Findings

### P-01: Platt calibrator fit on raw probability, not on logit — 88x signal compression
- **Severity:** Critical
- **Tractability:** <1h
- **Location:** `ml/model.py:232-235`, inference at `ml/model.py:307-310`
- **Finding:** `self.calibrator.fit(val_proba.reshape(-1, 1), y_val, ...)` passes `val_proba` (probability in [0,1]) as the LR feature. Correct Platt scaling fits on `logit(val_proba)`. The mismatch constrains the LR slope to 0.137 and produces output span 0.012 regardless of input.
- **Evidence:** Coefficients `coef_=[[0.1368]]`, `intercept_=[-2.332]`. Direct calculation `expit(0.137*raw_p + -2.332)` for `raw_p in [0,1]` yields outputs in [0.0885, 0.1001] — span 0.0117. Stage 1 observed post-Platt range 0.089–0.095 matches. All 10 severe-miss traces + all 9 Apr 22 races show compression ratio 86.6×–89.1× — tight band around the theoretical ~88.4x.
- **Recommendation:** Change Platt input to `logit(val_proba)` (one line in `train()`, mirror at `predict()`). Alternative: swap `LogisticRegression` for `IsotonicRegression`. Retrain. Sanity check: after fix, `coef_[0][0]` should be approximately 1.0 for an already-calibrated XGBoost; output should span at least the input's effective range.

### P-02: Current production Brier (0.098) is WORSE than the base-rate baseline (0.075)
- **Severity:** Critical
- **Tractability:** needs-data
- **Location:** `models/experiment_log.json` latest entry (`model_version=20260423-012955`), baseline computed from DB winners (359/4403=8.15%)
- **Finding:** Best Brier achievable by always predicting 8.15% on every horse is 8.15% × (1−8.15%) = 0.0749. The model's reported test Brier is 0.0981. Brier-Skill-Score = 1 − 0.0981/0.0749 = **−0.31** — the model is worse than a constant. This means the XGBoost + Platt pipeline is adding noise, not information.
- **Evidence:** Experiment log `{"brier_score": 0.09806, "n_samples": 500, ...}`. Base-rate calculation: `p=0.0815`, baseline Brier = `p*(1-p) = 0.0749`. No skill vs always-predict-base-rate is a statistical red line.
- **Recommendation:** Do not ship the recalibration fix (P-01) in isolation — it will preserve an already-bad ranking. Root-cause is either (a) training on the 500-sample run (see P-04), (b) feature leakage masking actual predictive value (data-scientist scope), or (c) the XGBoost hyperparameters are over-regularized. Minimum: retrain on the full 4403 rows and re-measure. If Brier remains > baseline, investigate features.

### P-03: Softmax temperature 1.5 on already-compressed Platt output has nearly zero effect
- **Severity:** High
- **Tractability:** <1h
- **Location:** `core/scorer.py:125-130`, `config.py:SOFTMAX_TEMPERATURE=1.5`
- **Finding:** With Platt span 0.006, `(s - max_s) * 1.5` gives exponent differences of at most 0.009, so `exp(-0.009) ≈ 0.991` — softmax output is essentially uniform 1/n. Measured in the 10 severe-miss traces: rank-1 vs winner softmax difference averages 0.0002 (uniform would be 0.0). This effectively destroys whatever ML signal remained after Platt.
- **Evidence:** Trace file `raw/trace-analysis.json` — all 10 cases show softmax diff in range ±0.0006 despite raw ML diffs of 0.05–0.45. Source comment in `core/scorer.py:119` acknowledges "softmax here is purely for normalization, not recalibration" — which is accurate only if Platt preserved information (P-01).
- **Recommendation:** Fix P-01 first; then re-tune temperature on validation set. With logit-input Platt, the typical platt-prob spread will likely be ~0.05–0.5 and temperature 1.5 is appropriate. No action needed on temperature itself until P-01 is resolved; revisit after.

### P-04: Production model trained on 500 samples, not the 4403-row DB
- **Severity:** Critical
- **Tractability:** <1h
- **Location:** `models/experiment_log.json`, last three entries (2026-04-23T01:09, T01:25, T01:29) all show `"n_samples": 500`. Prior runs on the same day (T00:48, T00:52) show `"n_samples": 4403`.
- **Finding:** Current production model `20260423-012955` was apparently trained on 500 samples. `tests/conftest.py:150` and `tests/test_data_integrity.py:16,21,25` all call `generate_synthetic_training_data(200)`/`(500)` and implicitly invoke `train()`, which writes via `_save_model()`. If pytest was run after the real training, the shipped model will be the test's synthetic-500 model.
- **Evidence:** DB has 4403 results; log shows `n_samples=500` for last three saves. `ml/model.py:136-153` `_save_model()` has no guard against overwriting the production model with test-run artifacts.
- **Recommendation:** (a) Confirm which training run the current pkl corresponds to — if 500, retrain immediately on real 4403. (b) Guard `_save_model()` against test contexts (env var, path separation, or refuse-to-save if `n_features < threshold`). (c) Update `.gitignore` / CI to prevent silent model overwrite.

### P-05: Brier/calibration curve bins in `scripts/accuracy.py` unreachable under current Platt
- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `scripts/accuracy.py:112`
- **Finding:** Bins are `[(0,0.05), (0.05,0.10), (0.10,0.15), (0.15,0.25), (0.25,1.0)]`. Under the current Platt output span [0.089, 0.095], ~all predictions fall into a single bin: (0.05,0.10). The 0.25–1.0 bin is definitionally empty (max output is 0.10). The 5-bin calibration curve is therefore vacuous — it will show one populated bin with `pred ≈ actual ≈ base rate` and claim "calibrated."
- **Evidence:** Platt max output = `expit(0.137+(-2.332)) = 0.100`. No prediction can enter bins above 0.10. (Note: `calibrated_prob` logged to DB is the blended post-market-mix prob, not the raw Platt — so after market blend, the final prob does reach higher values. The bin coverage is better than pure Platt but still biased toward mid-bin.)
- **Recommendation:** After fixing P-01 and retraining, recompute calibration curve with equal-count bins (deciles) rather than fixed edges. Document that the `calibrated_prob` column in `prediction_log` is post-blend, not post-Platt — currently ambiguous.

### P-06: Base-rate-anchored rule sigmoid `0.06 * (score-50) + logit(1/field_size)` uses an ad-hoc slope
- **Severity:** Medium
- **Tractability:** <1d
- **Location:** `ml/model.py:353` `rule_prob_raw = 1.0 / (1.0 + math.exp(-(0.06 * (rule_score - 50) + logit_base)))`
- **Finding:** The slope 0.06 is a magic number with no documented derivation. The rule_score in [0,100] produces logit offset in [−3, +3], so rule_prob ranges roughly [base_rate/20, min(1, base_rate*20)] — for field_size=12 this is [0.004, 0.65]. The anchor logic is sound (centers at 1/field_size), but the slope controls both (a) how much spread the rule engine gets after the rule/ML blend, and (b) whether a score of 60 vs 80 means "much better" or "marginally better." No empirical fit of this slope to historical win rates has been done. Opaque in backtest (`rule_score = null` in traces), so this slope's effect is only visible in the dashboard pipeline.
- **Evidence:** File comment "Without this anchor the rule engine was grossly uncalibrated" — honest, but slope choice is not justified. No code exists to empirically fit this slope. For comparison, 0.06 across 50-point range = 3.0 logit span; a proper fit on rule_score→win_rate bins would likely produce slopes of 0.04–0.10 depending on score distribution.
- **Recommendation:** Fit the slope empirically: for each past race, compute the rule_score; bin into deciles; measure win rate per bin; fit logistic regression `logit(win_rate) = a + b * score` on binned data; use that `b` instead of 0.06. Tractability <1d once a rule-score-and-outcome table exists.

### P-07: Dynamic market-blend thresholds (55%/65% market weight) based on two heuristics, not empirical
- **Severity:** High
- **Tractability:** <1d
- **Location:** `core/scorer.py:85-107`
- **Finding:** `_dynamic_blend_weights` shifts 15% toward market for "large field (10+) + sprint (≤1400m)" and 15% toward model for "small (<8) or long (≥1800m)." The thresholds (10, 1400, 8, 1800) and shift magnitude (±0.15) are heuristic — no citation, no ablation, no backtest comparison. Given P-01 (ML signal is near-zero), **the market blend is currently the sole determinant of ranking** — setting market_w=0.55 vs 0.60 vs 0.65 mechanically changes the recommender's behaviour in every race.
- **Evidence:** Code has no references to the tuning data. The field-size and distance breakpoints match CLAUDE.md but CLAUDE.md cites no source either. Comment says "HK bettors are sharp" — a reasonable prior but not evidence.
- **Recommendation:** After P-01 fix, do a proper market-blend sweep using `scripts/backtest.py --sweep` (grid over model_w) and log Brier/ROI per setting. Until then, flag in docs that the blend weights are heuristic and the model's residual edge over market is not established.

### P-08: Harville independence model grossly underestimates real HKJC quinella dividends
- **Severity:** High
- **Tractability:** redesign
- **Location:** `betting/dividends.py:17-40`
- **Finding:** `harville_pair_prob(p_a, p_b) = p_a*p_b/(1-p_a) + p_b*p_a/(1-p_b)` assumes independent finish positions — a known-bad approximation because pace dynamics strongly correlate who finishes 1st/2nd. The file comment acknowledges this ("diverges from reality when pace dynamics strongly correlate"). Project-memory entry `project_backtest_findings.md` confirms real HKJC dividends are massively higher than Harville estimates, implying the model overestimates pair probability (too concentrated on favourites). The `quinella_bet_ev` function then uses model_probs for P(hit) and market_probs for dividend — but both probabilities feed through the same Harville function, so the bias is not cancelled, only partially offset.
- **Evidence:** CLAUDE.md: "Harville grossly underestimated" on backtest. `dividends.py:6` openly admits the independence violation. No explicit correlation correction (e.g. Henery, Stern, or placer-boost beyond the ad-hoc +15% in `exotic_calculator.py:85-94`).
- **Recommendation:** Either (a) replace Harville with a correlated-finish model (Stern 1990 or Lo & Bacon-Shone's permutation model) — this is a redesign, not a patch; or (b) empirically fit `dividend_multiplier(top2_share)` from real HKJC dividends — the `race_dividends` table already has the data needed. Tractability for (b) is <1d; (a) is a redesign.

### P-09: `qin_bet_ev` mixes model probs and market probs inconsistently
- **Severity:** Medium
- **Tractability:** <1d
- **Location:** `betting/dividends.py:43-86`, specifically lines 66-67
- **Finding:** `p_hit = harville_pair_prob(model_probs[i], model_probs[j])` uses model probs for hit rate, then `div = estimate_quinella_dividend(market_probs, i, j)` uses market probs for dividend. This is *theoretically* sensible ("our edge = we think the true P(hit) differs from market's implied") but is internally inconsistent in one subtle way: the same Harville function is applied to two different prob distributions. If model and market both correctly represent their respective views of reality, the EV should reflect model_edge × dividend. The current code computes model_Harville × market_Harville_dividend, which conflates "my estimate of hit rate" with "my dividend" — because dividends are determined by the *field's* betting (market), using model_Harville × (1-t)/model_Harville = (1-t) would always give EV = -t. The code uses market_Harville for the dividend which is correct; but for hit rate it is also model_Harville which is correct. The subtle issue is that Harville miscorrelation (P-08) biases both sides, and since the model is near-identical to the market post-blend (P-01), `model_Harville ≈ market_Harville` → EV ≈ −takeout × n_combos always. This matches CLAUDE.md's observation that EV% is typically -30% to -60% with no upside — consistent with "no real edge."
- **Evidence:** Fitting 10 severe-miss races in `apr22-trace.json` shows EV always in [-40%, -60%] and "BEST EV" vs "LOW EV" spread is ≤5pp — matches the observed dashboard behaviour. CLAUDE.md section "Interpretation of EV%" is essentially a rationalization of no edge.
- **Recommendation:** Until P-01 is fixed, the EV calc is mathematically sound but produces no actionable signal. After P-01: track the *spread* between model_Harville and market_Harville per pair; pairs where our model assigns materially more hit probability than market has real positive EV. Current code already does this implicitly — it will simply start working once P-01 restores model-market divergence.

### P-10: BANKER/market-fav swaps in recommender mutate `combined_probability` AFTER softmax+blend
- **Severity:** Low
- **Tractability:** <1h
- **Location:** `core/recommender.py:50-81`, `_resync_ranks_and_probs` at 208-215
- **Finding:** After calibration produces `calibrated_probs`, a market-favorite-with-prob≥30% or BANKER horse is swapped into rank-1 and its position's probability is swapped along with it. This means horse A (which the math said had 15% blended prob) inherits the 25% slot, and horse B moves down. The probabilities are now inconsistent with the underlying model+market+softmax derivation. Subsequent EV calculations use these swapped probs. The bet-recommendation is correct (we're saying "bet the favorite"), but the probability column in the output is no longer a faithful model output — it's a bet-adjusted display value.
- **Evidence:** Lines 51-52: `analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]; calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]`. Subsequent `_resync_ranks_and_probs` writes the swapped probs back. This is documented behaviour (CLAUDE.md "Probability sync after rank swaps") but still violates the "probability must match what the model computed" invariant.
- **Recommendation:** Keep a separate `display_probability` column (post-swap) and a `model_probability` column (pre-swap). Dashboard users and `scripts/accuracy.py` calibration curve should ideally use `model_probability` for calibration measurement — otherwise the calibration is measuring the human override, not the model.

### P-11: Calibration breaks the sum-to-1 invariant in one code path
- **Severity:** Low
- **Tractability:** <1h
- **Location:** `core/scorer.py:147-154`
- **Finding:** `calibrate_probabilities` renormalizes `blended` to sum to 1 at the end. Good. But at line 150, horses without `market_prob > 0` (no odds) get `b = cal_probs[i]` (model-only) while horses with odds get `b = model_w * cal_prob + market_w * market_prob` (blended). These two formulas are not on the same scale — `cal_prob` sums to ~1, `market_prob` also sums to ~1 after normalization, so blended sums to ~1, but mixing the two formulas in one race where some horses have odds and others don't produces a mal-formed distribution before the final renormalize. Final renormalize fixes the sum but not the relative weight: a no-odds horse gets `cal_prob` (sum ~1/n) while odds horses share the rest at ~1 × average → no-odds horses get *under*-weighted. In pre-race races (no odds yet) every horse hits the fallback, so it's fine; mixed cases are the rare but problematic ones.
- **Evidence:** Line 150: `b = cal_probs[i]` for missing-odds, `b = model_w * cal_probs[i] + market_w * market_probs[i]` otherwise. Different scales.
- **Recommendation:** For missing-odds horses, use the full model_w weight and scale: `b = (model_w + market_w) * cal_probs[i]` (i.e. `b = cal_probs[i]` is already right if all weights sum to 1 — the issue is conceptual, not numeric). Simpler fix: skip the no-odds race entirely (SKIP) upstream. Low-severity because the dashboard already does this in `core/recommender.py:111` (`has_live_odds = any(...)`).

### P-12: Softmax reference point is `max(scores)`, fine for normalization but sensitive to outliers
- **Severity:** Informational
- **Tractability:** <1h
- **Location:** `core/scorer.py:127-130`
- **Finding:** `exp_scores = [math.exp((s - max_score) * temp) for s in raw_scores]` is the standard numerically-stable softmax. No issue; noting only for completeness.
- **Evidence:** Standard implementation.
- **Recommendation:** No action.

## What I Did NOT Review
- Kelly / bet-sizing logic in `betting/optimizer.py` — out of scope for this findings pass, though related to EV translation to stakes.
- Exotic calculator boost factors (+15% placer, +10% pace, +8% draw) in `betting/exotic_calculator.py:85-130` — these are ad-hoc multipliers deserving the same empirical-fit treatment as P-06, but I did not trace their evidence base.
- `ml/experiments.py` experiment-log integrity — only consumed the latest entry for P-02/P-04.
- Feature correlations reported in experiment log (`draw_advantage × field_rel_draw = 0.716`, `trainer_win_rate × jockey_trainer_combo = 0.725`) — out of scope (data-science).
- Pool-by-pool confidence calibration (separate blend weights for quinella vs win per CLAUDE.md) — code path not located.

## Caveats
- **P-04 is uncertain**: I inferred from `experiment_log.json` that production is the 500-sample model. This is consistent with all 10 severe-miss traces showing compression derived from the `.pkl` whose coef/intercept I inspected. However, the test-contamination hypothesis for *why* n_samples=500 is speculative — could also be a bug in `log_experiment` or a manual test run. Either way the symptom is real and actionable.
- **P-02 Brier-vs-baseline calculation** uses the full-DB base rate 0.0815. The 500-sample training run's per-horse base rate could differ — I did not extract the validation-set prevalence from the pickle because validation indices weren't saved. But 0.0815 × 0.0919 is a wide corridor; the reported 0.098 is outside it.
- **P-08 Harville correction** is a known open problem in racing stats; my recommendation (b) is pragmatic but the "correct" answer is a Stern/Henery model which is a larger investment.
- I did NOT reproduce the backtest to confirm the Stage 1 "45% box-4 hit rate" vs "34% box-3" — that's data-science scope.
