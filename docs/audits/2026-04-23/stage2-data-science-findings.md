# Data-Scientist Findings

## Summary

The backtest does **NOT** match the production pipeline. `scripts/backtest.py` runs an ML-only → softmax → market-blend pipeline with zero rule engine and zero recommender involvement; it also applies a simple `top-K set issubset actual_top2` proxy for box-quinella ROI that skips the rank-2 BANKER/market-favourite promotion logic used on race day. Compounding this, the currently-shipped production model pickle (version `20260423-012955`) was **trained by the pytest fixture on 500 synthetic samples**, not by `scripts/train.py` on the 4,403 real DB samples — the last three entries in `experiment_log.json` trace to `tests/conftest.py:151`. The +999% / +1064% / +849% ROI claims from `raw/backtest-output.txt` therefore describe behaviour of a synthetic-trained model under an ML-only pipeline, and are not statistically reliable evidence about today's race-day strategy.

## Backtest Code-Path Verification (First Task)

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

## Findings

### DS-01: Production model pickle was trained by pytest fixture on 500 synthetic samples, not by the real training pipeline

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

### DS-02: Backtest pipeline is architecturally disjoint from production recommender

- **Severity:** Critical
- **Tractability:** <1d (rewrite backtest to call `assign_bet_recommendations`) or redesign (add an integration-backtest harness)
- **Location:** `scripts/backtest.py:131-300` (no recommender), vs `core/recommender.py:13-170` + `core/analyzer.py:1031-1122`
- **Finding:** Backtest applies its own pick-generation and exotic logic: it sorts by blended prob, takes top-1..top-4, and checks `issubset`. Production's `core/recommender.py` applies conviction gates (`MIN_TOP3_CONVICTION=0.30`, `MIN_TOP4_CONVICTION=0.38`), a `_rank_has_strong_history` filter (lines 173-201) requiring specific reasoning-string matches derived from the rule engine, and market-favourite / BANKER swaps that can change which horses land in the box.
- **Evidence:**
  - `grep -n "recommender\|assign_bet\|create_combined_prediction" scripts/backtest.py` returns no matches.
  - Backtest emits a bet on every race (`simulate_bets:167` skips only when no odds). Production's recommender can return `SKIP` when conviction gates aren't met (`core/recommender.py:124`). The 73-race sample with 72 evaluable thus overstates the race count that production would actually bet.
  - Rank-1 market-favourite swap (`core/recommender.py:47-61`) fires whenever `market_prob ≥ 0.30` and rank>1. Stage-1 forensic note (`stage1-forensic.md:10`) estimates this disagrees on ~10-20% of races — exactly where the backtest's blended rank-1 differs from the recommender's rank-1.
- **Recommendation:** Extend `scripts/backtest.py` to (a) run the full `CombinedRaceAnalyzer` including rule engine on each held-out race, (b) call `assign_bet_recommendations`, (c) measure ROI only on races where production would have issued a non-SKIP recommendation with the exact bet_type and top-K set. Until this exists, treat current ROI numbers as upper-bound / rank-order sanity only.

### DS-03: Train/predict mismatch on `weight_vs_field` feature

- **Severity:** High
- **Tractability:** <1h
- **Location:** `core/analyzer.py:1031-1057` (builds `horse_data` without `field_weights`), `ml/features.py:377-394` (inference path falls back to 108-135 formula), `ml/training.py:326-338` (training uses `field_median_weights`)
- **Finding:** `core/analyzer.py` never populates `horse_data['field_weights']`. The inference code path in `ml/features.py:383-391` therefore never reaches the `if field_weights:` branch and always computes the absolute-range fallback `1 - (weight - 108) / 27`. Training (`ml/training.py:332-336`) computes the field-median-relative formula `0.5 - (weight - median_wt)/20`. Same horse, same race, different feature value.
- **Evidence:** For BLAZING BEAM in 2026/04/22 R5, inference returns `weight_vs_field=0.5000` while training returns `0.7500` — delta 0.2500. Bulk probe over all 107 horses on 2026/04/22 shows mean `|Δ|=0.182`, max `|Δ|=0.519`, 68.2% have `|Δ|>0.1`, 42.1% have `|Δ|>0.2`. Feature importance for `weight_vs_field` is 4.66% (5th-9th ranked per `experiment_log.json`), so this mismatch puts ~5% of the model's signal on a distribution the XGBoost trees never learned.
- **Recommendation:** Populate `horse_data['field_weights']` in `core/analyzer.py:1031-1057` from `[int(h.weight) for h in race.horses if h.weight]`. Alternatively, remove the fallback branch entirely so inference fails loudly when training-parity data is missing.

### DS-04: `backtest.py` ROI CIs are too wide to support a strategy pivot

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

### DS-05: `ingest_race_day` silently wipes HKJC-rankings-scraped jockey/trainer stats

- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `data/database.py:329-330` (`ingest_race_day` → `rebuild_stats`), `data/database.py:842-888` (`rebuild_stats` does `DELETE FROM jockey_stats` / `trainer_stats`)
- **Finding:** `rebuild_stats()` unconditionally deletes and repopulates both tables from local `race_results` only. If the user previously ran `python3 scrape_history.py --rankings` to pull the full-season HKJC figures (~80 meets), the next `ingest_race_day` call reverts both tables to the ~10-meeting local view without logging the source change. CLAUDE.md documents this as a "known fix" but the data pipeline does not guard against it.
- **Evidence:** `data/database.py:852` — `c.execute("DELETE FROM jockey_stats")`. No conditional, no source tagging. `data/database.py:330` fires on every ingestion.
- **Recommendation:** Add a `source` column to `jockey_stats`/`trainer_stats` (`'local'` vs `'hkjc_rankings'`). Have `rebuild_stats()` skip rows where source='hkjc_rankings' (or write to separate table and have the feature query prefer rankings). At minimum, emit a warning when the existing table has substantially more records than rebuild would produce (indicating rankings were present).

### DS-06: Near-constant feature `class_change` (std=0.038) — effectively wasted capacity

- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `ml/features.py:517-528`, `ml/training.py:404-410`
- **Finding:** On the 4,403-row DB training set, `class_change` has std=0.038 and 60% of values fall in the `[0.375, 0.625]` middle (`class_rank - last_class_rank ∈ [-1, 0, +1]`). CLAUDE.md's rule is "std < 0.01 = wasted"; std=0.038 is above that floor but still 5-10× lower than every other feature except `horse_weight_change` (0.078) and `jockey_trainer_combo` (0.066). `test_no_constant_training_features` uses the same `<0.01` threshold and misses this.
- **Evidence:** Full std distribution: `{prior_win_rate:0.170, recent_form:0.258, draw_advantage:0.200, jockey_win_rate:0.042, trainer_win_rate:0.029, field_size:0.104, field_rel_draw:0.291, prior_pace:0.206, adjusted_speed:0.217, cd_form_prior:0.187, class_level:0.156, freshness:0.257, going_performance:0.302, weight_vs_field:0.260, jockey_horse_coop:0.144, distance_pb_ratio:0.224, field_rel_speed:0.223, last_start_margin:0.222, class_change:0.038, days_since_last_win:0.251, jockey_trainer_combo:0.066, horse_weight_change:0.078, speed_consistency:0.206}`. XGBoost still assigns `class_change` 5.64% importance — most of that signal is concentrated at `0.125`/`0.875` tail cases (big jumps).
- **Recommendation:** Either widen the feature scale (e.g. `0.5 + diff * 0.2` clipped to [0,1]) so common `±1`-step moves spread more, or demote to a binary "dropping / rising / same" indicator. Tighten the test threshold from `std<0.01` to `std<0.05` given this feature's proximity.

### DS-07: `experiment_log.high_correlations` does not reflect DB-trained model

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

### DS-08: Feature test thresholds too loose to catch systematic inference/training offsets

- **Severity:** Medium
- **Tractability:** <1h
- **Location:** `tests/test_data_accuracy.py:259-297` (`test_training_and_predict_features_same_distribution`)
- **Finding:** The test uses `z_score = |X_pred - train_mean| / train_std > 5` to flag mismatches. For `weight_vs_field` (DS-03), the inference-vs-training systematic offset is ~0.18 mean / 0.26 std = z≈0.7, well below 5. The test only catches catastrophic dispersion, not systematic bias, and allows up to 2 features to be wildly off before failing.
- **Evidence:** DS-03 shows a 68.2% frequency of `|Δ|>0.1` on `weight_vs_field` that the current test does not flag. The test also tolerates `<=2` wildly-off features, weakening it further.
- **Recommendation:** Replace distributional z-score test with a per-horse parity test: for each of N sampled real DB rows, compute both `extract_features(horse_data, ...)` (with all fields the production pipeline would populate) and the corresponding row from `build_training_data_from_db()`; assert element-wise `|Δ| < 0.02` on every feature. Drop the `<=2 features` slack.

### DS-09: `field_rel_draw` can exceed 1.0 (no clip) — late-scratch edge case

- **Severity:** Low
- **Tractability:** <1h
- **Location:** `ml/features.py:281` (`features.append(draw / max(1, field_size) if draw > 0 else 0.5)`), `ml/training.py:245`
- **Finding:** When a horse's original draw exceeds the post-scratch field size, `draw/field_size > 1`. DB has 29/4403 (0.7%) such rows. Max `field_rel_draw` observed in training is 1.200. Feature distribution thus has out-of-range values XGBoost has learned on, but the range violates the 0..1 convention every other feature follows.
- **Evidence:** `field_rel_draw max=1.200` in DB-training stats; 29 rows with draw > field_size (see probe output).
- **Recommendation:** Clip to `min(1.0, draw / field_size)` in both files, or use `(draw - 1) / max(1, field_size - 1)` which naturally bounds to [0, 1] when draw≤field_size.

### DS-10: `predict()` silent `1/field_size` fallback on untrained model

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

### DS-11: Test coverage gap — no tests for recommender, calibrator, blend weights, backtest methodology

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

### DS-12: Recency-weight sample curve is `0.3 + 0.7*rank/max_rank`, linearly over **dates not races**

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

### DS-13: Training sample = 4,403 rows / **359 winners** / 36 dates — winner count is the binding constraint

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

---

## What I Did NOT Review

- **Probability math in depth:** Platt slope/intercept derivation, softmax temperature justification, dynamic blend coefficient choice, Harville-independence validity — all assigned to the parallel probability-expert subagent per spec.
- **Kelly / bet-sizing:** `betting/exotic_calculator.py` and `betting/dividends.py` pricing logic internals — out of scope.
- **Rule-engine factor weights:** `BettingWeights` in `core/scorer.py:13-60` — scorecard weightings are out of ML scope.
- **Dashboard UX / HTML:** Explicitly non-goal.
- **Live-odds scraper reliability:** `data/odds_scraper.py` — not a modeling concern.
- **Full tests pytest run:** I enumerated test coverage but did not execute the suite.
- **Exhaustive Stage-1 overlap checks:** I cross-referenced the 10 severe-miss traces but did not re-run the probes.

## Caveats

- **ROI CI assumptions:** The Monte Carlo in DS-04 approximates the 11 straight-Q and 22 box-3 hit-dividend samples from summary stats (min/median/mean/max), because `raw/backtest-output.txt` does not preserve the raw per-hit dividends. The true CI could be modestly narrower if the real dividend distribution is tighter than my approximation.
- **DS-01 causality:** I establish the production pickle matches synthetic-trained correlations exactly — this is strong circumstantial evidence, but I did not observe the pytest run that wrote it. An alternative explanation (someone manually trained on 500 samples) seems implausible given `generate_synthetic_training_data(500)` reproduces the exact correlations with seed 42.
- **DS-03 magnitude:** The 0.182 mean absolute delta on `weight_vs_field` is measured on a single race day (2026/04/22, 107 horses). I did not bulk-probe all 4,403 rows, but the discrepancy root-cause (missing `field_weights` in `horse_data`) is deterministic and applies everywhere.
- **DS-02 ROI impact:** I cannot quantify how much of the +1064% box-4 ROI survives under the real recommender's conviction gates without re-running the holdout through `assign_bet_recommendations`. The impact is directional ("production bets fewer races on higher conviction") — magnitude depends on how often the gates fire.
- **Calibration compression (DS-01-adjacent):** Stage-1 attributes 88× compression to "the calibrator." If the pickle is truly from the 500-sample synthetic run, the calibrator was fit on a tiny validation slice with a different winner prevalence than real racing — which would naturally produce extreme compression. Retraining on 4,403 rows should substantially reduce compression independently of any Platt-algorithm fixes the probability-expert subagent recommends.
