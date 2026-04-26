# Stage 2: Systematic Review (Merged)

## Summary

Two independent review lenses — a probability expert and a data scientist — collectively surfaced 25 findings across the ML pipeline. The picture is severe: there are 4 Critical findings, two of which independently confirmed each other (the synthetic-trained production model, DS-01 and P-04), and one pair that together fully explain the Stage 1 forensic anomaly (P-01 Platt mis-fit + DS-02 backtest pipeline divergence). The dominant pattern is that the current production system is not doing what its architecture claims: the ML model in production was trained on 500 synthetic samples by pytest (not the 4,403-row DB), the Platt calibrator mechanically crushes all ML signal to an 0.012-wide band, and the backtest that generated the +999%/+1064% ROI headlines runs an entirely different pipeline than the dashboard. Every quantitative claim about model performance and betting strategy must be re-evaluated after the 4 Critical findings are remediated.

## Top-Priority Findings (Convergent or Critical)

- **P-04 ∩ DS-01 (Critical):** Production model is trained on 500 synthetic samples from pytest, not the real 4,403-row DB. Both personas identified this independently from different angles (P: Brier score and calibrator coefficients; DS: experiment_log correlations reproducing with seed 42). Strongest signal in the entire audit — this is a live production issue.
- **P-01 ∩ DS-02 (Critical):** Platt calibrator fit on raw probability (not logit) produces 88× signal compression, AND the backtest that generated headline ROI numbers runs zero rule engine and zero recommender logic. Together these two findings explain the entire Stage 1 forensic: the ML is essentially uninformative in production, and the backtest that "validated" the strategy wasn't measuring the production strategy.
- **P-02 (Critical):** Production Brier score (0.098) is worse than always-predicting-base-rate (0.075). Brier Skill Score = −0.31. The XGBoost + Platt pipeline is currently adding noise, not information.
- **DS-03 (High):** Systematic train/predict mismatch on `weight_vs_field`: training uses field-median-relative formula, inference uses absolute-range fallback. Mean `|Δ|=0.182` across 107 horses; 68.2% of horses have `|Δ|>0.1`. Feature importance is ~5% — this mismatch actively corrupts a meaningful slice of model signal.
- **DS-11 (High):** Zero test coverage for `core/recommender.py`, `core/scorer.py::calibrate_probabilities`, the Platt calibrator, backtest ROI math, and all 13 analytics modules. The findings DS-01 through DS-04 are production failures that a minimal test harness would have caught.
- **P-07 ∩ DS-02 (High):** Dynamic market-blend weights are pure heuristic (no empirical fit, no ablation), AND the backtest that would have validated them doesn't call the blend logic. With ML signal near-zero (P-01/P-04), the market blend is currently the sole ranking signal — making the ungrounded weights the de facto model.
- **P-08 (High):** Harville independence model grossly underestimates real HKJC quinella dividends. CLAUDE.md itself notes "Harville grossly underestimated." No correlated-finish correction exists beyond an ad-hoc +15% placer boost. The EV calculator's signal is structurally non-actionable until this and P-01/P-04 are fixed.

---

## Probability-Expert Findings

### Summary
Platt compression root cause **diagnosed**: the calibrator was fit on the raw XGBoost probability directly (scale [0,1]) instead of on its logit, so the learned logistic regression slope (0.137) can only stretch the output logit by 0.137 across the entire input range. Result: every raw input maps to a platt probability in a 0.012-wide band around the base rate — 88x compression — which erases all ranking information from the ML model. Three secondary findings make the situation worse: the current production model is trained on only 500 samples (not the 4403-row DB), its test Brier (0.098) is *worse* than the base-rate baseline (0.075), and the 5-bin calibration curve in `scripts/accuracy.py` uses bins whose upper edges (0.25, 1.0) are unreachable under current Platt output.

### Platt Calibrator Diagnosis (First Task)

#### Observed coefficients
From `pickle.load(models/hkjc_racing_model.pkl)['calibrator']`:
- `type`: `sklearn.linear_model._logistic.LogisticRegression`
- `coef_` = `[[0.1368081]]`
- `intercept_` = `[-2.33244596]`
- `classes_` = `[0, 1]`

#### The 88x compression is mechanical, not statistical
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

#### Why the slope is 0.137 and not, say, 4 or 5
The slope is correct given the input scale. A well-calibrated slope *if the input were `logit(raw_p)`* would be ~1 (pass-through for an already-calibrated model). But because the input is `raw_p` on the linear scale, the LR has to fight non-linearity: near `raw_p=0.05` a logistic regression wants to raise the output logit by ~0.7 to land near base rate, while near `raw_p=0.5` it wants a different slope. The gradient-descent solver finds the least-bad linear-in-`p` compromise, which is a tiny slope. The "real" compression cause is feature choice, not recency weights or validation prevalence.

#### Secondary contributing factors (but not the root cause)
1. **Training signal is weak** — with only 500 training samples (see P-04 below) and base-rate-only model quality (Brier 0.098 > baseline 0.075), XGBoost's raw probabilities are nearly uninformative on validation. Even a correct logit-input Platt would produce a shallow slope. But the linear-in-p choice compounds this by another ~7x.
2. **Recency weights are applied** (`ml/model.py:233-235`: `sw_val = sample_weight[cut_train:cut_val]`; passed via `sample_weight=sw_val`). Per CLAUDE.md claim — verified in code. Not a driver of compression.
3. **Validation set prevalence** — not directly inspectable without re-running training, but the 60/20/20 temporal split puts the last-20% of dates in test and the prior 20% in validation. On 36 distinct race dates (4403 rows, 8.15% winners) validation has ~880 rows with ~72 winners. Prevalence matches training, so this is not the imbalance issue suggested in the prompt.

#### Counterfactual: what happens if we bypass Platt entirely
Measured on the 10 severe-miss traces (`raw/trace-analysis.json`):
- 9/10 cases: rank-1 had higher `raw_ml_prob` than the actual winner (i.e. Platt didn't save the ML from feature-driven misranking — but it also didn't make things worse than the raw XGBoost on ranking)
- 1/10 cases (Apr 12 R10): winner had `raw_ml_prob=0.2176` vs rank-1's `0.1637`. Raw XGBoost correctly identified the winner. Platt collapsed them to 0.0909 vs 0.0903 — a 0.0006 gap that softmax (temp=1.5) could not meaningfully amplify (softmax diff 0.0001), and market blend then overrode.

So removing Platt alone would recover the winner in 1/10 severe misses directly — small but free. Bigger effect: the softmax (temp=1.5) would suddenly have a real input range [0.07, 0.54] instead of [0.089, 0.095], producing ~6x more spread across the field and meaningfully reducing the market's dominance in the 55-65% blend.

#### Recommended fix (conceptual)
Two options, in order of preference:

1. **Fit Platt on `logit(val_proba)`, not `val_proba`** — one-line change: `self.calibrator.fit(logit(val_proba).reshape(-1, 1), y_val, ...)`, and mirror the transformation at inference time in `predict()`. Standard Platt-scaling formulation. Immediately restores a slope near 1.0 and preserves raw XGBoost's ordering.

2. **Switch to `sklearn.isotonic.IsotonicRegression`** — non-parametric, doesn't assume logistic shape, handles non-linear miscalibration naturally. Needs ~200+ val samples (which we have). Handles the regime where XGBoost is overconfident only at tails (favourites).

Either fix is <1 hour of work. Retrain only, no schema changes. Do NOT retrain-and-ship without also addressing the underlying 500-sample issue (P-04) — otherwise the recalibrated slope will still be mild because the signal itself is weak.

### Findings

#### P-01: Platt calibrator fit on raw probability, not on logit — 88x signal compression
- **Severity:** Critical
- **Tractability:** <1h
- **Location:** `ml/model.py:232-235`, inference at `ml/model.py:307-310`
- **Finding:** `self.calibrator.fit(val_proba.reshape(-1, 1), y_val, ...)` passes `val_proba` (probability in [0,1]) as the LR feature. Correct Platt scaling fits on `logit(val_proba)`. The mismatch constrains the LR slope to 0.137 and produces output span 0.012 regardless of input.
- **Evidence:** Coefficients `coef_=[[0.1368]]`, `intercept_=[-2.332]`. Direct calculation `expit(0.137*raw_p + -2.332)` for `raw_p in [0,1]` yields outputs in [0.0885, 0.1001] — span 0.0117. Stage 1 observed post-Platt range 0.089–0.095 matches. All 10 severe-miss traces + all 9 Apr 22 races show compression ratio 86.6×–89.1× — tight band around the theoretical ~88.4x.
- **Recommendation:** Change Platt input to `logit(val_proba)` (one line in `train()`, mirror at `predict()`). Alternative: swap `LogisticRegression` for `IsotonicRegression`. Retrain. Sanity check: after fix, `coef_[0][0]` should be approximately 1.0 for an already-calibrated XGBoost; output should span at least the input's effective range.

#### P-02: Current production Brier (0.098) is WORSE than the base-rate baseline (0.075)
- **Severity:** Critical
- **Tractability:** needs-data
- **Location:** `models/experiment_log.json` latest entry (`model_version=20260423-012955`), baseline computed from DB winners (359/4403=8.15%)
- **Finding:** Best Brier achievable by always predicting 8.15% on every horse is 8.15% × (1−8.15%) = 0.0749. The model's reported test Brier is 0.0981. Brier-Skill-Score = 1 − 0.0981/0.0749 = **−0.31** — the model is worse than a constant. This means the XGBoost + Platt pipeline is adding noise, not information.
- **Evidence:** Experiment log `{"brier_score": 0.09806, "n_samples": 500, ...}`. Base-rate calculation: `p=0.0815`, baseline Brier = `p*(1-p) = 0.0749`. No skill vs always-predict-base-rate is a statistical red line.
- **Recommendation:** Do not ship the recalibration fix (P-01) in isolation — it will preserve an already-bad ranking. Root-cause is either (a) training on the 500-sample run (see P-04), (b) feature leakage masking actual predictive value (data-scientist scope), or (c) the XGBoost hyperparameters are over-regularized. Minimum: retrain on the full 4403 rows and re-measure. If Brier remains > baseline, investigate features.

#### P-03: Softmax temperature 1.5 on already-compressed Platt output has nearly zero effect
- **Severity:** High
- **Tractability:** <1h
- **Location:** `core/scorer.py:125-130`, `config.py:SOFTMAX_TEMPERATURE=1.5`
- **Finding:** With Platt span 0.006, `(s - max_s) * 1.5` gives exponent differences of at most 0.009, so `exp(-0.009) ≈ 0.991` — softmax output is essentially uniform 1/n. Measured in the 10 severe-miss traces: rank-1 vs winner softmax difference averages 0.0002 (uniform would be 0.0). This effectively destroys whatever ML signal remained after Platt.
- **Evidence:** Trace file `raw/trace-analysis.json` — all 10 cases show softmax diff in range ±0.0006 despite raw ML diffs of 0.05–0.45. Source comment in `core/scorer.py:119` acknowledges "softmax here is purely for normalization, not recalibration" — which is accurate only if Platt preserved information (P-01).
- **Recommendation:** Fix P-01 first; then re-tune temperature on validation set. With logit-input Platt, the typical platt-prob spread will likely be ~0.05–0.5 and temperature 1.5 is appropriate. No action needed on temperature itself until P-01 is resolved; revisit after.

#### P-04: Production model trained on 500 samples, not the 4403-row DB
- **Severity:** Critical
- **Tractability:** <1h
- **Location:** `models/experiment_log.json`, last three entries (2026-04-23T01:09, T01:25, T01:29) all show `"n_samples": 500`. Prior runs on the same day (T00:48, T00:52) show `"n_samples": 4403`.
- **Finding:** Current production model `20260423-012955` was apparently trained on 500 samples. `tests/conftest.py:150` and `tests/test_data_integrity.py:16,21,25` all call `generate_synthetic_training_data(200)`/`(500)` and implicitly invoke `train()`, which writes via `_save_model()`. If pytest was run after the real training, the shipped model will be the test's synthetic-500 model.
- **Evidence:** DB has 4403 results; log shows `n_samples=500` for last three saves. `ml/model.py:136-153` `_save_model()` has no guard against overwriting the production model with test-run artifacts.
- **Recommendation:** (a) Confirm which training run the current pkl corresponds to — if 500, retrain immediately on real 4403. (b) Guard `_save_model()` against test contexts (env var, path separation, or refuse-to-save if `n_features < threshold`). (c) Update `.gitignore` / CI to prevent silent model overwrite.

#### P-05: Brier/calibration curve bins in `scripts/accuracy.py` unreachable under current Platt
- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `scripts/accuracy.py:112`
- **Finding:** Bins are `[(0,0.05), (0.05,0.10), (0.10,0.15), (0.15,0.25), (0.25,1.0)]`. Under the current Platt output span [0.089, 0.095], ~all predictions fall into a single bin: (0.05,0.10). The 0.25–1.0 bin is definitionally empty (max output is 0.10). The 5-bin calibration curve is therefore vacuous — it will show one populated bin with `pred ≈ actual ≈ base rate` and claim "calibrated."
- **Evidence:** Platt max output = `expit(0.137+(-2.332)) = 0.100`. No prediction can enter bins above 0.10. (Note: `calibrated_prob` logged to DB is the blended post-market-mix prob, not the raw Platt — so after market blend, the final prob does reach higher values. The bin coverage is better than pure Platt but still biased toward mid-bin.)
- **Recommendation:** After fixing P-01 and retraining, recompute calibration curve with equal-count bins (deciles) rather than fixed edges. Document that the `calibrated_prob` column in `prediction_log` is post-blend, not post-Platt — currently ambiguous.

#### P-06: Base-rate-anchored rule sigmoid `0.06 * (score-50) + logit(1/field_size)` uses an ad-hoc slope
- **Severity:** Medium
- **Tractability:** <1d
- **Location:** `ml/model.py:353` `rule_prob_raw = 1.0 / (1.0 + math.exp(-(0.06 * (rule_score - 50) + logit_base)))`
- **Finding:** The slope 0.06 is a magic number with no documented derivation. The rule_score in [0,100] produces logit offset in [−3, +3], so rule_prob ranges roughly [base_rate/20, min(1, base_rate*20)] — for field_size=12 this is [0.004, 0.65]. The anchor logic is sound (centers at 1/field_size), but the slope controls both (a) how much spread the rule engine gets after the rule/ML blend, and (b) whether a score of 60 vs 80 means "much better" or "marginally better." No empirical fit of this slope to historical win rates has been done. Opaque in backtest (`rule_score = null` in traces), so this slope's effect is only visible in the dashboard pipeline.
- **Evidence:** File comment "Without this anchor the rule engine was grossly uncalibrated" — honest, but slope choice is not justified. No code exists to empirically fit this slope. For comparison, 0.06 across 50-point range = 3.0 logit span; a proper fit on rule_score→win_rate bins would likely produce slopes of 0.04–0.10 depending on score distribution.
- **Recommendation:** Fit the slope empirically: for each past race, compute the rule_score; bin into deciles; measure win rate per bin; fit logistic regression `logit(win_rate) = a + b * score` on binned data; use that `b` instead of 0.06. Tractability <1d once a rule-score-and-outcome table exists.

#### P-07: Dynamic market-blend thresholds (55%/65% market weight) based on two heuristics, not empirical
- **Severity:** High
- **Tractability:** <1d
- **Location:** `core/scorer.py:85-107`
- **Finding:** `_dynamic_blend_weights` shifts 15% toward market for "large field (10+) + sprint (≤1400m)" and 15% toward model for "small (<8) or long (≥1800m)." The thresholds (10, 1400, 8, 1800) and shift magnitude (±0.15) are heuristic — no citation, no ablation, no backtest comparison. Given P-01 (ML signal is near-zero), **the market blend is currently the sole determinant of ranking** — setting market_w=0.55 vs 0.60 vs 0.65 mechanically changes the recommender's behaviour in every race.
- **Evidence:** Code has no references to the tuning data. The field-size and distance breakpoints match CLAUDE.md but CLAUDE.md cites no source either. Comment says "HK bettors are sharp" — a reasonable prior but not evidence.
- **Recommendation:** After P-01 fix, do a proper market-blend sweep using `scripts/backtest.py --sweep` (grid over model_w) and log Brier/ROI per setting. Until then, flag in docs that the blend weights are heuristic and the model's residual edge over market is not established.

#### P-08: Harville independence model grossly underestimates real HKJC quinella dividends
- **Severity:** High
- **Tractability:** redesign
- **Location:** `betting/dividends.py:17-40`
- **Finding:** `harville_pair_prob(p_a, p_b) = p_a*p_b/(1-p_a) + p_b*p_a/(1-p_b)` assumes independent finish positions — a known-bad approximation because pace dynamics strongly correlate who finishes 1st/2nd. The file comment acknowledges this ("diverges from reality when pace dynamics strongly correlate"). Project-memory entry `project_backtest_findings.md` confirms real HKJC dividends are massively higher than Harville estimates, implying the model overestimates pair probability (too concentrated on favourites). The `quinella_bet_ev` function then uses model_probs for P(hit) and market_probs for dividend — but both probabilities feed through the same Harville function, so the bias is not cancelled, only partially offset.
- **Evidence:** CLAUDE.md: "Harville grossly underestimated" on backtest. `dividends.py:6` openly admits the independence violation. No explicit correlation correction (e.g. Henery, Stern, or placer-boost beyond the ad-hoc +15% in `exotic_calculator.py:85-94`).
- **Recommendation:** Either (a) replace Harville with a correlated-finish model (Stern 1990 or Lo & Bacon-Shone's permutation model) — this is a redesign, not a patch; or (b) empirically fit `dividend_multiplier(top2_share)` from real HKJC dividends — the `race_dividends` table already has the data needed. Tractability for (b) is <1d; (a) is a redesign.

#### P-09: `qin_bet_ev` mixes model probs and market probs inconsistently
- **Severity:** Medium
- **Tractability:** <1d
- **Location:** `betting/dividends.py:43-86`, specifically lines 66-67
- **Finding:** `p_hit = harville_pair_prob(model_probs[i], model_probs[j])` uses model probs for hit rate, then `div = estimate_quinella_dividend(market_probs, i, j)` uses market probs for dividend. This is *theoretically* sensible ("our edge = we think the true P(hit) differs from market's implied") but is internally inconsistent in one subtle way: the same Harville function is applied to two different prob distributions. If model and market both correctly represent their respective views of reality, the EV should reflect model_edge × dividend. The current code computes model_Harville × market_Harville_dividend, which conflates "my estimate of hit rate" with "my dividend" — because dividends are determined by the *field's* betting (market), using model_Harville × (1-t)/model_Harville = (1-t) would always give EV = -t. The code uses market_Harville for the dividend which is correct; but for hit rate it is also model_Harville which is correct. The subtle issue is that Harville miscorrelation (P-08) biases both sides, and since the model is near-identical to the market post-blend (P-01), `model_Harville ≈ market_Harville` → EV ≈ −takeout × n_combos always. This matches CLAUDE.md's observation that EV% is typically -30% to -60% with no upside — consistent with "no real edge."
- **Evidence:** Fitting 10 severe-miss races in `apr22-trace.json` shows EV always in [-40%, -60%] and "BEST EV" vs "LOW EV" spread is ≤5pp — matches the observed dashboard behaviour. CLAUDE.md section "Interpretation of EV%" is essentially a rationalization of no edge.
- **Recommendation:** Until P-01 is fixed, the EV calc is mathematically sound but produces no actionable signal. After P-01: track the *spread* between model_Harville and market_Harville per pair; pairs where our model assigns materially more hit probability than market has real positive EV. Current code already does this implicitly — it will simply start working once P-01 restores model-market divergence.

#### P-10: BANKER/market-fav swaps in recommender mutate `combined_probability` AFTER softmax+blend
- **Severity:** Low
- **Tractability:** <1h
- **Location:** `core/recommender.py:50-81`, `_resync_ranks_and_probs` at 208-215
- **Finding:** After calibration produces `calibrated_probs`, a market-favorite-with-prob≥30% or BANKER horse is swapped into rank-1 and its position's probability is swapped along with it. This means horse A (which the math said had 15% blended prob) inherits the 25% slot, and horse B moves down. The probabilities are now inconsistent with the underlying model+market+softmax derivation. Subsequent EV calculations use these swapped probs. The bet-recommendation is correct (we're saying "bet the favorite"), but the probability column in the output is no longer a faithful model output — it's a bet-adjusted display value.
- **Evidence:** Lines 51-52: `analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]; calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]`. Subsequent `_resync_ranks_and_probs` writes the swapped probs back. This is documented behaviour (CLAUDE.md "Probability sync after rank swaps") but still violates the "probability must match what the model computed" invariant.
- **Recommendation:** Keep a separate `display_probability` column (post-swap) and a `model_probability` column (pre-swap). Dashboard users and `scripts/accuracy.py` calibration curve should ideally use `model_probability` for calibration measurement — otherwise the calibration is measuring the human override, not the model.

#### P-11: Calibration breaks the sum-to-1 invariant in one code path
- **Severity:** Low
- **Tractability:** <1h
- **Location:** `core/scorer.py:147-154`
- **Finding:** `calibrate_probabilities` renormalizes `blended` to sum to 1 at the end. Good. But at line 150, horses without `market_prob > 0` (no odds) get `b = cal_probs[i]` (model-only) while horses with odds get `b = model_w * cal_prob + market_w * market_prob` (blended). These two formulas are not on the same scale — `cal_prob` sums to ~1, `market_prob` also sums to ~1 after normalization, so blended sums to ~1, but mixing the two formulas in one race where some horses have odds and others don't produces a mal-formed distribution before the final renormalize. Final renormalize fixes the sum but not the relative weight: a no-odds horse gets `cal_prob` (sum ~1/n) while odds horses share the rest at ~1 × average → no-odds horses get *under*-weighted. In pre-race races (no odds yet) every horse hits the fallback, so it's fine; mixed cases are the rare but problematic ones.
- **Evidence:** Line 150: `b = cal_probs[i]` for missing-odds, `b = model_w * cal_probs[i] + market_w * market_probs[i]` otherwise. Different scales.
- **Recommendation:** For missing-odds horses, use the full model_w weight and scale: `b = (model_w + market_w) * cal_probs[i]` (i.e. `b = cal_probs[i]` is already right if all weights sum to 1 — the issue is conceptual, not numeric). Simpler fix: skip the no-odds race entirely (SKIP) upstream. Low-severity because the dashboard already does this in `core/recommender.py:111` (`has_live_odds = any(...)`).

#### P-12: Softmax reference point is `max(scores)`, fine for normalization but sensitive to outliers
- **Severity:** Informational
- **Tractability:** <1h
- **Location:** `core/scorer.py:127-130`
- **Finding:** `exp_scores = [math.exp((s - max_score) * temp) for s in raw_scores]` is the standard numerically-stable softmax. No issue; noting only for completeness.
- **Evidence:** Standard implementation.
- **Recommendation:** No action.

### What I Did NOT Review
- Kelly / bet-sizing logic in `betting/optimizer.py` — out of scope for this findings pass, though related to EV translation to stakes.
- Exotic calculator boost factors (+15% placer, +10% pace, +8% draw) in `betting/exotic_calculator.py:85-130` — these are ad-hoc multipliers deserving the same empirical-fit treatment as P-06, but I did not trace their evidence base.
- `ml/experiments.py` experiment-log integrity — only consumed the latest entry for P-02/P-04.
- Feature correlations reported in experiment log (`draw_advantage × field_rel_draw = 0.716`, `trainer_win_rate × jockey_trainer_combo = 0.725`) — out of scope (data-science).
- Pool-by-pool confidence calibration (separate blend weights for quinella vs win per CLAUDE.md) — code path not located.

### Caveats
- **P-04 is uncertain**: I inferred from `experiment_log.json` that production is the 500-sample model. This is consistent with all 10 severe-miss traces showing compression derived from the `.pkl` whose coef/intercept I inspected. However, the test-contamination hypothesis for *why* n_samples=500 is speculative — could also be a bug in `log_experiment` or a manual test run. Either way the symptom is real and actionable.
- **P-02 Brier-vs-baseline calculation** uses the full-DB base rate 0.0815. The 500-sample training run's per-horse base rate could differ — I did not extract the validation-set prevalence from the pickle because validation indices weren't saved. But 0.0815 × 0.0919 is a wide corridor; the reported 0.098 is outside it.
- **P-08 Harville correction** is a known open problem in racing stats; my recommendation (b) is pragmatic but the "correct" answer is a Stern/Henery model which is a larger investment.
- I did NOT reproduce the backtest to confirm the Stage 1 "45% box-4 hit rate" vs "34% box-3" — that's data-science scope.

---

## Data-Scientist Findings

### Summary

The backtest does **NOT** match the production pipeline. `scripts/backtest.py` runs an ML-only → softmax → market-blend pipeline with zero rule engine and zero recommender involvement; it also applies a simple `top-K set issubset actual_top2` proxy for box-quinella ROI that skips the rank-2 BANKER/market-favourite promotion logic used on race day. Compounding this, the currently-shipped production model pickle (version `20260423-012955`) was **trained by the pytest fixture on 500 synthetic samples**, not by `scripts/train.py` on the 4,403 real DB samples — the last three entries in `experiment_log.json` trace to `tests/conftest.py:151`. The +999% / +1064% / +849% ROI claims from `raw/backtest-output.txt` therefore describe behaviour of a synthetic-trained model under an ML-only pipeline, and are not statistically reliable evidence about today's race-day strategy.

### Backtest Code-Path Verification (First Task)

**Q1 — Which model pickle does it load?**
`models/hkjc_racing_model.pkl`, metadata `version=20260423-012955`, `trained_date=2026-04-23T01:29:55.559167`, `n_features=23`.
Loader: `ml/model.py:99-134` (`HKJCRacingMLModel._load_model()`), invoked from `scripts/backtest.py:536` via `HKJCRacingMLModel()`.
**Critical:** this pickle was written by the pytest fixture — see DS-01 for evidence.

**Q2 — Which recommender logic does it use for pick generation?**
**None.** `scripts/backtest.py` has zero references to `assign_bet_recommendations`, `CombinedRaceAnalyzer`, `create_combined_prediction`, `_rank_has_strong_history`, conviction gates, or BANKER/WIN/TRIO/BOX-Q selection logic.
Pick generation in backtest: `scripts/backtest.py:174-176` — `sorted_by_prob = sorted(..., key=lambda i: preds[i][2], reverse=True)` → top-1..top-4 by blended probability. Exotic "hits" use `actual_top2.issubset(our_top3_names)` / `our_top4_names` (lines 222-225). There is no `_rank_has_strong_history` gate, no `MIN_TOP3_CONVICTION=0.30` / `MIN_TOP4_CONVICTION=0.38` check, no TRIO branch, no BANKER or market-favourite promotion (`core/recommender.py:44-81`).

**Q3 — Do the reported ROI numbers reflect today's `recommender.py` logic?**
No. The backtest reports ROI as "bet box-3 on every race where top-2 actual ⊂ top-3 model" (`raw/backtest-output.txt:35-36,54-55`). Production `core/recommender.py:111-125` SKIPS many of those races via conviction gates and the `_rank_has_strong_history` filter. Production also swaps rank-1 ↔ market-favourite when `market_prob ≥ 0.30` and `rank>1` (`core/recommender.py:44-61`), changing which top-2/top-3/top-4 set is bet. The measured `+999%` / `+849%` / `+1064%` are therefore the ROI of an **unconditional** box strategy on the final blended probabilities, not the conditional strategy `core/recommender.py` actually emits.

**Q4 — Does the backtest use the 60/40 rule+ML blend?**
No. `scripts/backtest.py:132-139` computes `ml_probs = model.predict(features, ...)` for each horse, then applies softmax and the dynamic market blend directly. No call to `compute_weighted_score` (rule engine), no `create_combined_prediction` (60/40 blend in `ml/model.py:333-382`). As Stage-1 forensic already documents (`stage1-forensic.md:368-374`), `rule_score` and `combined_prob` are null in every trace — the backtest's `final_prob` is architecturally different from the dashboard's `combined_probability`.

**Verdict: backtest-to-production pipeline match = NO.** This is the dominant Stage-2 data-science finding and it caveats every quantitative claim derived from `raw/backtest-output.txt`.

---

### Findings

#### DS-01: Production model pickle was trained by pytest fixture on 500 synthetic samples, not by the real training pipeline

- **Severity:** Critical
- **Tractability:** <1h (retrain via `python3 -m scripts.train`) + <1d (add checksum guard)
- **Location:** `tests/conftest.py:140-152`, `ml/model.py:136-153` (`_save_model`), `models/experiment_log.json` entries 85-87
- **Finding:** The last three `experiment_log.json` entries log `n_samples=500`. Reproducing `generate_synthetic_training_data(500)` (seed 42) yields exactly the two correlations logged: `draw_advantage × field_rel_draw r=+0.716` and `trainer_win_rate × jockey_trainer_combo r=+0.725`. The pytest fixture `trained_model` at `tests/conftest.py:149-151` calls `HKJCRacingMLModel()` (which loads the global `models/` path via `config.MODEL_PATH`), then `model.train(X, y)` with 500 synthetic samples. `ml/model.py:136-153` unconditionally calls `_save_model()` inside `train()`, overwriting the production pickle every time the fixture runs.
- **Evidence:**
  - `pickle.load(open('models/hkjc_racing_model.pkl')) → version='20260423-012955'` matches the `n_samples=500` log entry.
  - Running `generate_synthetic_training_data(500)` then computing `|corrcoef|>0.7`:
    ```
    draw_advantage   × field_rel_draw         r=+0.716
    trainer_win_rate × jockey_trainer_combo   r=+0.725
    ```
    These are the exact two high-corr pairs stored in the production pickle's experiment entry.
  - Real DB training data (4,403 rows, 359 winners) has *different* high correlations — see DS-07 — proving the log entry is not from a DB-trained run.
  - Calibrator coef=0.1368, intercept=-2.332. Input raw probs [0.05, 0.10, 0.20, 0.30, 0.40, 0.50] map to [0.0890, 0.0896, 0.0907, 0.0918, 0.0930, 0.0941] — compression matches Stage-1's observed 88× figure. This calibrator is what ships in production and what all Stage-1 traces measured.
- **Recommendation:** Regenerate the production pickle via `python3 -m scripts.train` on DB data. Add a guard in `_save_model` (or the fixture) that refuses to overwrite `config.MODEL_PATH` when the training sample count is `<MIN_DB_SAMPLES` or when the caller is a pytest test. Point the fixture at a `tmp_path` model file instead of `config.MODEL_PATH`.

#### DS-02: Backtest pipeline is architecturally disjoint from production recommender

- **Severity:** Critical
- **Tractability:** <1d (rewrite backtest to call `assign_bet_recommendations`) or redesign (add an integration-backtest harness)
- **Location:** `scripts/backtest.py:131-300` (no recommender), vs `core/recommender.py:13-170` + `core/analyzer.py:1031-1122`
- **Finding:** Backtest applies its own pick-generation and exotic logic: it sorts by blended prob, takes top-1..top-4, and checks `issubset`. Production's `core/recommender.py` applies conviction gates (`MIN_TOP3_CONVICTION=0.30`, `MIN_TOP4_CONVICTION=0.38`), a `_rank_has_strong_history` filter (lines 173-201) requiring specific reasoning-string matches derived from the rule engine, and market-favourite / BANKER swaps that can change which horses land in the box.
- **Evidence:**
  - `grep -n "recommender\|assign_bet\|create_combined_prediction" scripts/backtest.py` returns no matches.
  - Backtest emits a bet on every race (`simulate_bets:167` skips only when no odds). Production's recommender can return `SKIP` when conviction gates aren't met (`core/recommender.py:124`). The 73-race sample with 72 evaluable thus overstates the race count that production would actually bet.
  - Rank-1 market-favourite swap (`core/recommender.py:47-61`) fires whenever `market_prob ≥ 0.30` and rank>1. Stage-1 forensic note (`stage1-forensic.md:10`) estimates this disagrees on ~10-20% of races — exactly where the backtest's blended rank-1 differs from the recommender's rank-1.
- **Recommendation:** Extend `scripts/backtest.py` to (a) run the full `CombinedRaceAnalyzer` including rule engine on each held-out race, (b) call `assign_bet_recommendations`, (c) measure ROI only on races where production would have issued a non-SKIP recommendation with the exact bet_type and top-K set. Until this exists, treat current ROI numbers as upper-bound / rank-order sanity only.

#### DS-03: Train/predict mismatch on `weight_vs_field` feature

- **Severity:** High
- **Tractability:** <1h
- **Location:** `core/analyzer.py:1031-1057` (builds `horse_data` without `field_weights`), `ml/features.py:377-394` (inference path falls back to 108-135 formula), `ml/training.py:326-338` (training uses `field_median_weights`)
- **Finding:** `core/analyzer.py` never populates `horse_data['field_weights']`. The inference code path in `ml/features.py:383-391` therefore never reaches the `if field_weights:` branch and always computes the absolute-range fallback `1 - (weight - 108) / 27`. Training (`ml/training.py:332-336`) computes the field-median-relative formula `0.5 - (weight - median_wt)/20`. Same horse, same race, different feature value.
- **Evidence:** For BLAZING BEAM in 2026/04/22 R5, inference returns `weight_vs_field=0.5000` while training returns `0.7500` — delta 0.2500. Bulk probe over all 107 horses on 2026/04/22 shows mean `|Δ|=0.182`, max `|Δ|=0.519`, 68.2% have `|Δ|>0.1`, 42.1% have `|Δ|>0.2`. Feature importance for `weight_vs_field` is 4.66% (5th-9th ranked per `experiment_log.json`), so this mismatch puts ~5% of the model's signal on a distribution the XGBoost trees never learned.
- **Recommendation:** Populate `horse_data['field_weights']` in `core/analyzer.py:1031-1057` from `[int(h.weight) for h in race.horses if h.weight]`. Alternatively, remove the fallback branch entirely so inference fails loudly when training-parity data is missing.

#### DS-04: `backtest.py` ROI CIs are too wide to support a strategy pivot

- **Severity:** High
- **Tractability:** needs-data
- **Location:** `scripts/backtest.py:348-421` (exotic_hits + ROI reporting)
- **Finding:** Wilson 95% CIs on the reported hit rates (n=72 races with dividends):
  - Straight Q: 11/72 = 15.3% CI [8.8%, 25.3%]
  - Box-3: 23/72 = 31.9% CI [22.3%, 43.4%]
  - Box-4: 33/72 = 45.8% CI [34.8%, 57.3%]
  Parametric-bootstrap ROI intervals (resample hits from Wilson width + resample HKJC-dividend draws from summary stats; 10,000 sims):
  - Straight Q 95% ROI CI ≈ [+385%, +1742%]
  - Box-3 Q 95% ROI CI ≈ [+588%, +1529%]
  - Box-4 Q 95% ROI CI ≈ [+462%, +984%]
  The box-3 CI overlaps heavily with the straight-Q and box-4 CIs — the sample size cannot distinguish them.
- **Evidence:** See Bash probe output in this audit session (Wilson + Monte Carlo blocks). Raw summary stats pulled directly from `raw/backtest-output.txt:46-47`.
- **Recommendation:** Minimum sample for ±5pp ROI CI narrowing at current variance is on the order of 500-700 races. Do not treat the measured +999% / +1064% as decision-grade until the holdout grows by ~7×. In the interim, the project memo `project_backtest_findings.md` should be footnoted with these CIs.

#### DS-05: `ingest_race_day` silently wipes HKJC-rankings-scraped jockey/trainer stats

- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `data/database.py:329-330` (`ingest_race_day` → `rebuild_stats`), `data/database.py:842-888` (`rebuild_stats` does `DELETE FROM jockey_stats` / `trainer_stats`)
- **Finding:** `rebuild_stats()` unconditionally deletes and repopulates both tables from local `race_results` only. If the user previously ran `python3 scrape_history.py --rankings` to pull the full-season HKJC figures (~80 meets), the next `ingest_race_day` call reverts both tables to the ~10-meeting local view without logging the source change. CLAUDE.md documents this as a "known fix" but the data pipeline does not guard against it.
- **Evidence:** `data/database.py:852` — `c.execute("DELETE FROM jockey_stats")`. No conditional, no source tagging. `data/database.py:330` fires on every ingestion.
- **Recommendation:** Add a `source` column to `jockey_stats`/`trainer_stats` (`'local'` vs `'hkjc_rankings'`). Have `rebuild_stats()` skip rows where source='hkjc_rankings' (or write to separate table and have the feature query prefer rankings). At minimum, emit a warning when the existing table has substantially more records than rebuild would produce (indicating rankings were present).

#### DS-06: Near-constant feature `class_change` (std=0.038) — effectively wasted capacity

- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `ml/features.py:517-528`, `ml/training.py:404-410`
- **Finding:** On the 4,403-row DB training set, `class_change` has std=0.038 and 60% of values fall in the `[0.375, 0.625]` middle (`class_rank - last_class_rank ∈ [-1, 0, +1]`). CLAUDE.md's rule is "std < 0.01 = wasted"; std=0.038 is above that floor but still 5-10× lower than every other feature except `horse_weight_change` (0.078) and `jockey_trainer_combo` (0.066). `test_no_constant_training_features` uses the same `<0.01` threshold and misses this.
- **Evidence:** Full std distribution: `{prior_win_rate:0.170, recent_form:0.258, draw_advantage:0.200, jockey_win_rate:0.042, trainer_win_rate:0.029, field_size:0.104, field_rel_draw:0.291, prior_pace:0.206, adjusted_speed:0.217, cd_form_prior:0.187, class_level:0.156, freshness:0.257, going_performance:0.302, weight_vs_field:0.260, jockey_horse_coop:0.144, distance_pb_ratio:0.224, field_rel_speed:0.223, last_start_margin:0.222, class_change:0.038, days_since_last_win:0.251, jockey_trainer_combo:0.066, horse_weight_change:0.078, speed_consistency:0.206}`. XGBoost still assigns `class_change` 5.64% importance — most of that signal is concentrated at `0.125`/`0.875` tail cases (big jumps).
- **Recommendation:** Either widen the feature scale (e.g. `0.5 + diff * 0.2` clipped to [0,1]) so common `±1`-step moves spread more, or demote to a binary "dropping / rising / same" indicator. Tighten the test threshold from `std<0.01` to `std<0.05` given this feature's proximity.

#### DS-07: `experiment_log.high_correlations` does not reflect DB-trained model

- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `ml/model.py:284-295` (correlation computation), `models/experiment_log.json`
- **Finding:** When the production pipeline runs on the DB's 4,403 rows, the correlations above 0.7 are `jockey_win_rate × jockey_trainer_combo r=+0.793` and `recent_form × days_since_last_win r=+0.743` (plus `recent_form × field_rel_speed r=+0.718` and `field_rel_speed × last_start_margin r=+0.710`). The currently-shipped experiment_log entry logs the two synthetic-data correlations instead (DS-01). The DS-relevant real-data findings are:
  - `jockey_win_rate × jockey_trainer_combo = 0.793`: the combo feature is nearly redundant with jockey_win_rate. `jockey_trainer_combo` importance is 4.2% vs `jockey_win_rate` 7.3%; dropping `jockey_trainer_combo` would barely hurt and would reduce the effective-dimension inflation.
  - `recent_form × days_since_last_win = 0.743`: intuitive (horses that won recently have small days-since-last-win). Both remain informative because one is a rate, the other a time decay.
- **Evidence:** Probe:
  ```
  jockey_win_rate        × jockey_trainer_combo  r=+0.793
  recent_form            × days_since_last_win   r=+0.743
  recent_form            × field_rel_speed       r=+0.718
  field_rel_speed        × last_start_margin     r=+0.710
  ```
  from full DB-training feature matrix.
- **Recommendation:** After retraining on DB data, drop `jockey_trainer_combo` (or replace with a non-multiplicative interaction like `jockey_win_rate > 0.15 AND trainer_win_rate > 0.12` indicator). Keep `recent_form` and `days_since_last_win` — they capture different temporal signals.

#### DS-08: Feature test thresholds too loose to catch systematic inference/training offsets

- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `tests/test_data_accuracy.py:259-297` (`test_training_and_predict_features_same_distribution`)
- **Finding:** The test uses `z_score = |X_pred - train_mean| / train_std > 5` to flag mismatches. For `weight_vs_field` (DS-03), the inference-vs-training systematic offset is ~0.18 mean / 0.26 std = z≈0.7, well below 5. The test only catches catastrophic dispersion, not systematic bias, and allows up to 2 features to be wildly off before failing.
- **Evidence:** DS-03 shows a 68.2% frequency of `|Δ|>0.1` on `weight_vs_field` that the current test does not flag. The test also tolerates `<=2` wildly-off features, weakening it further.
- **Recommendation:** Replace distributional z-score test with a per-horse parity test: for each of N sampled real DB rows, compute both `extract_features(horse_data, ...)` (with all fields the production pipeline would populate) and the corresponding row from `build_training_data_from_db()`; assert element-wise `|Δ| < 0.02` on every feature. Drop the `<=2 features` slack.

#### DS-09: `field_rel_draw` can exceed 1.0 (no clip) — late-scratch edge case

- **Severity:** Low
- **Tractability:** <1h
- **Location:** `ml/features.py:281` (`features.append(draw / max(1, field_size) if draw > 0 else 0.5)`), `ml/training.py:245`
- **Finding:** When a horse's original draw exceeds the post-scratch field size, `draw/field_size > 1`. DB has 29/4403 (0.7%) such rows. Max `field_rel_draw` observed in training is 1.200. Feature distribution thus has out-of-range values XGBoost has learned on, but the range violates the 0..1 convention every other feature follows.
- **Evidence:** `field_rel_draw max=1.200` in DB-training stats; 29 rows with draw > field_size (see probe output).
- **Recommendation:** Clip to `min(1.0, draw / field_size)` in both files, or use `(draw - 1) / max(1, field_size - 1)` which naturally bounds to [0, 1] when draw≤field_size.

#### DS-10: `predict()` silent `1/field_size` fallback on untrained model

- **Severity:** Low (today) / High (on regression)
- **Tractability:** <1h
- **Location:** `ml/model.py:299-302`
- **Finding:** When `self.is_trained == False` (e.g., model load failed due to missing checksum file, or pickle corruption), `predict()` returns `(1/field_size, "MODEL_NOT_TRAINED")` silently for every horse. The caller in `core/analyzer.py:1059-1068` checks `ml_available` but does not verify `ml_model.is_trained` post-load. Any production code path that reached this branch would generate uniform ML probs, letting the market blend drive 100% of the ranking.
- **Evidence:** `ml/model.py:300-302`:
  ```python
  if not self.is_trained or self.model is None:
      base_prob = 1.0 / max(field_size, 1)
      return base_prob, "MODEL_NOT_TRAINED"
  ```
  Stage-1 forensic shows current Platt compression already reduces ML to ~0.006 spread (near-uniform); this is an *additional* silent-failure trapdoor on top.
- **Recommendation:** Raise an exception in `predict()` when `self.is_trained == False`, or propagate the "MODEL_NOT_TRAINED" string to the dashboard as a banner. Add a healthcheck in `scripts/generate.py` that refuses to produce predictions when any horse returned `MODEL_NOT_TRAINED`.

#### DS-11: Test coverage gap — no tests for recommender, calibrator, blend weights, backtest methodology

- **Severity:** High
- **Tractability:** <1d
- **Location:** `tests/`
- **Finding:** Out of 78 test functions (counted via `grep -c "def test_" tests/*.py`):
  - `core/recommender.py` (170 lines, BANKER/BOX Q4/BOX Q/QIN/TRIO/WIN/SKIP logic): **0 tests**. No test exercises `assign_bet_recommendations`, `_rank_has_strong_history`, conviction gates, market-favourite swap, or BANKER promotion.
  - `core/scorer.py::calibrate_probabilities` (full softmax + market-blend pipeline): **0 tests**. `_dynamic_blend_weights` (field-size/distance branches): **0 tests**.
  - `ml/model.py` Platt calibrator fit + predict-with-calibration (lines 231-243, 307-311): **0 tests**. No test checks calibrator compression, coefficients, or monotonicity.
  - `scripts/backtest.py` (dividend lookup, ROI math, Harville fallback): **0 tests**.
  - `analytics/` (13 specialist modules, including `speed_figures.py` par-time leakage risk): **0 tests**.
  - `ml/training.py::build_training_data_from_db` (the 23-feature training builder, 280 LOC): **0 direct tests** (only indirect via `test_training_and_predict_features_same_distribution`, which DS-08 shows is too loose).
  Covered areas: feature schema (19 tests), data accuracy / DB invariants (29 tests), data integrity / synthetic data (9 tests), scoring-rule weights + sigmoid anchor (9 tests), validation (12 tests).
- **Evidence:** `grep -l "recommender\|assign_bet\|calibrate_prob\|Platt\|calibrator" tests/*.py` returns only `tests/test_scoring.py`, which tests `create_combined_prediction` (60/40 blend math) but not the recommender or calibrator. `grep -rn "core/recommender\|calibrate_probabilities\|_dynamic_blend_weights\|scripts/backtest" tests/` returns no matches.
- **Recommendation:** Before any DS-01..DS-04 fixes, add (a) a recommender integration test using a fixture race card + known rule-engine reasoning strings, asserting the expected `bet_recommendation` per horse; (b) a calibrator smoke test that asserts `calibrator.predict_proba` is monotonic and spread ≥ some minimum on a realistic raw-prob range; (c) a backtest self-test that runs on ≤10 held-out races and asserts the reported ROI math (hits × dividend − stake) closes to within $0.01.

#### DS-12: Recency-weight sample curve is `0.3 + 0.7*rank/max_rank`, linearly over **dates not races**

- **Severity:** Low
- **Tractability:** <1h
- **Location:** `ml/training.py:137-144, 466-467`
- **Finding:** Recency weights are keyed by **distinct date rank**, not by row index. With 36 distinct dates and 4,403 rows, every horse on a given race date gets the same weight, and a single quiet Wednesday HV meeting (e.g., 8-10 races) contributes the same to recency as a full Saturday ST meeting (11 races × ~12 horses = ~130 rows). Newer dates correctly weight higher (1.0 vs 0.3), but the 0.3 floor is arbitrary ("3.3× recent vs oldest" per CLAUDE.md) with no justification in commit messages or docstrings. The shape is strictly linear in date-rank — no empirical fit.
- **Evidence:** `ml/training.py:137-144`:
  ```python
  all_dates = sorted(set(dict(row).get('race_date', '') for row in rows))
  ...
  date_rank = {d: i for i, d in enumerate(all_dates)}
  max_rank = len(all_dates) - 1
  ...
  weights.append(0.3 + 0.7 * (rank / max(1, max_rank)))
  ```
  36 dates → weight values are in `{0.300, 0.320, 0.340, ..., 1.000}`. With imbalanced horses-per-date this under-weights busy race days relative to their race count.
- **Recommendation:** Either (a) replace with per-row recency `0.3 + 0.7 * i / max_i` over sorted rows (so busy dates contribute proportionally), or (b) use an exponential-decay kernel `exp(-λ * days_before_test)` with λ chosen via cross-validated logloss. Document whichever curve you pick.

#### DS-13: Training sample = 4,403 rows / **359 winners** / 36 dates — winner count is the binding constraint

- **Severity:** Informational
- **Tractability:** needs-data
- **Location:** `ml/training.py:180-468`
- **Finding:** Naming this explicitly because CLAUDE.md and the `n_samples` metric obscure it. The 60/20/20 split gives:
  - Train: 2,641 rows / 213 winners / ~20 dates
  - Val: 881 rows / 74 winners / ~8 dates
  - Test: 881 rows / 72 winners / ~8 dates
  Platt scaling is fitted on 74 positive examples against 807 negatives — tight. The model's `recall=0.18` reported in the latest (synthetic) run is about what you'd expect with this label imbalance, but when running on real DB data the same recall is a concrete statement that the model correctly flags only 18% of actual winners in the top-1 slot.
- **Evidence:** `SELECT COUNT(*), SUM(is_winner), COUNT(DISTINCT race_date) FROM race_results WHERE finish_position > 0` → 4403 / 359 / 36.
- **Recommendation:** Informational only. Collecting 1 additional season (~80 meetings × ~100 rows) would roughly 2× the training winners and substantially improve Platt fit stability.

### What I Did NOT Review

- **Probability math in depth:** Platt slope/intercept derivation, softmax temperature justification, dynamic blend coefficient choice, Harville-independence validity — all assigned to the parallel probability-expert subagent per spec.
- **Kelly / bet-sizing:** `betting/exotic_calculator.py` and `betting/dividends.py` pricing logic internals — out of scope.
- **Rule-engine factor weights:** `BettingWeights` in `core/scorer.py:13-60` — scorecard weightings are out of ML scope.
- **Dashboard UX / HTML:** Explicitly non-goal.
- **Live-odds scraper reliability:** `data/odds_scraper.py` — not a modeling concern.
- **Full tests pytest run:** I enumerated test coverage but did not execute the suite.
- **Exhaustive Stage-1 overlap checks:** I cross-referenced the 10 severe-miss traces but did not re-run the probes.

### Caveats

- **ROI CI assumptions:** The Monte Carlo in DS-04 approximates the 11 straight-Q and 22 box-3 hit-dividend samples from summary stats (min/median/mean/max), because `raw/backtest-output.txt` does not preserve the raw per-hit dividends. The true CI could be modestly narrower if the real dividend distribution is tighter than my approximation.
- **DS-01 causality:** I establish the production pickle matches synthetic-trained correlations exactly — this is strong circumstantial evidence, but I did not observe the pytest run that wrote it. An alternative explanation (someone manually trained on 500 samples) seems implausible given `generate_synthetic_training_data(500)` reproduces the exact correlations with seed 42.
- **DS-03 magnitude:** The 0.182 mean absolute delta on `weight_vs_field` is measured on a single race day (2026/04/22, 107 horses). I did not bulk-probe all 4,403 rows, but the discrepancy root-cause (missing `field_weights` in `horse_data`) is deterministic and applies everywhere.
- **DS-02 ROI impact:** I cannot quantify how much of the +1064% box-4 ROI survives under the real recommender's conviction gates without re-running the holdout through `assign_bet_recommendations`. The impact is directional ("production bets fewer races on higher conviction") — magnitude depends on how often the gates fire.
- **Calibration compression (DS-01-adjacent):** Stage-1 attributes 88× compression to "the calibrator." If the pickle is truly from the 500-sample synthetic run, the calibrator was fit on a tiny validation slice with a different winner prevalence than real racing — which would naturally produce extreme compression. Retraining on 4,403 rows should substantially reduce compression independently of any Platt-algorithm fixes the probability-expert subagent recommends.

---

## Merge Notes

### Disagreements

No disagreements on severity or recommendation direction were observed between the two personas for findings that covered overlapping areas. The one area where both touched calibration quality (P-02 Brier vs DS-13 winner-count constraint) is complementary rather than contradictory: P-02 flags the current model as statistically harmful (BSS = −0.31), while DS-13 contextualizes why (only 74 positive validation examples make Platt fit inherently unstable). Both point toward the same remediation: retrain on full DB data and measure again. No cases were found where one persona said "Critical" and the other said "Low" for the same issue, or where recommendations conflict.

### Convergence (same issue, both lenses)

- **P-04 ∩ DS-01: Synthetic-trained production pickle.** The probability expert identified this from Brier score anomaly and calibrator coefficient inspection; the data scientist identified it from experiment_log correlation fingerprinting (seed-42 synthetic data reproduces the logged correlations exactly). Two completely independent evidence chains reach identical conclusions. Strongest signal in the audit — independent confirmation makes this unambiguous. Both agree it is Critical and fixable in <1h.

- **P-01 ∩ DS-02: ML signal destroyed before the backtest, and the backtest doesn't run production logic.** P-01 explains WHY the ML is uninformative in production (logit mis-fit → 88× compression). DS-02 explains WHY the backtest ROI numbers don't describe production behaviour (no rule engine, no recommender). Together they form the complete explanation of the Stage 1 forensic: the model can't rank horses meaningfully AND the number that supposedly validated the strategy was measuring a different strategy. Neither finding alone is the full story — their union is.

- **P-03 ∩ DS-02 (partial): Softmax is ineffective / backtest bypasses softmax's context.** P-03 shows softmax temperature 1.5 is mathematically inert given Platt's 0.006 span. DS-02 notes the backtest does apply softmax but does so on ML-only probs without the rule engine's combined_prob — so even a properly-functioning softmax in the backtest measures a different distribution than production. Both point to the same structural gap: post-Platt probability transformations are either inert (production) or measuring the wrong thing (backtest).

- **P-07 ∩ DS-02 (partial): Market-blend weights are ungrounded, and no backtest validates them.** P-07 flags the ±0.15 heuristic market-blend shifts as empirically unjustified. DS-02 establishes that the backtest does apply the dynamic market blend (Q4) but not the recommender conviction gates that sit downstream — so the blend's effect on actual strategy output has never been measured end-to-end. Both personas converge on "the market blend is the de facto ranking signal with no validation."

- **P-02 ∩ DS-13 (partial): Model quality is structurally limited by winner count.** P-02 establishes that Brier Skill Score = −0.31 (adding noise). DS-13 contextualizes: Platt scaling fitted on only 74 positive validation examples, and 213 winners in training — so a negative Brier Skill Score is an expected outcome of fitting on insufficient positive signal, independent of the logit-vs-raw-prob choice. These findings reinforce each other: the Platt mis-fit (P-01) is the immediate cause; low winner count (DS-13) is a structural amplifier.

### Single-Lens Findings

**Probability-expert only** (5 findings): P-05 (calibration bins unreachable), P-06 (rule sigmoid slope ad-hoc), P-08 (Harville independence violation), P-09 (EV formula mixing model/market probs inconsistently), P-10 (BANKER swaps mutate probability post-softmax), P-11 (sum-to-1 break in mixed-odds races), P-12 (softmax reference point, informational).

**Data-scientist only** (8 findings): DS-03 (weight_vs_field train/predict mismatch), DS-04 (ROI CIs too wide), DS-05 (ingest_race_day wipes jockey/trainer rankings), DS-06 (class_change near-constant), DS-07 (experiment_log reflects synthetic not DB correlations), DS-08 (feature test thresholds too loose), DS-09 (field_rel_draw > 1.0), DS-10 (silent MODEL_NOT_TRAINED fallback), DS-11 (zero test coverage for recommender/calibrator/backtest), DS-12 (recency weights by date not by row).

These single-lens findings are not inherently weaker — DS-03 (weight_vs_field mismatch) and DS-11 (coverage gap) are High-severity findings that the data-scientist lens was specifically positioned to find. P-08 (Harville) and P-06 (rule sigmoid slope) are findings the probability expert was positioned to find. The audit approach correctly separated concerns.

### Process Signal

Rough overlap calculation: 25 total findings across both personas. Strong convergence on 2 pairs (P-04/DS-01, P-01/DS-02) plus 3 partial overlaps (P-03/DS-02, P-07/DS-02, P-02/DS-13). That is approximately 5 findings touching shared territory out of 25 total — **~20% overlap by finding count, ~25-30% by thematic coverage**.

**Conclusion: overlap is at the low end of the 20-80% "complementary" range. Parallel dispatch produced largely complementary findings — the two lenses genuinely covered different failure modes.** The probability expert went deep on calibration math, EV formula correctness, and probability pipeline integrity. The data scientist went deep on backtest code-path fidelity, feature train/predict consistency, test coverage, and DB data quality. The 2 strong convergence points (synthetic model + Platt/backtest pairing) validate that both independently caught the most important issues, while the large non-overlapping tails confirm the parallelism was warranted rather than duplicative. Single-threaded review would likely have caught the Critical convergence points but would have missed the depth of the single-lens findings on both sides.
