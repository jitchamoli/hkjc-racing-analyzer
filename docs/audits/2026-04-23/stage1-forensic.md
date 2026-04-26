# Stage 1: Forensic Findings — 2026-04-23

## Evidence Base
- Backtest run: `raw/backtest-output.txt` (73-race holdout, model version 20260423-012955, trained 2026-04-23)
- Holdout enumeration: `raw/holdout-failures.json` (73 races, 1 data-incomplete)
- Failure flag counts: rank1_miss=31, top2_miss=12, winner_deep=24, qin_box3_miss=50, qin_box4_miss=40
- Severe miss selection: threshold top_2_share ≥ 0.35 (fallback from 0.40, which yielded only 1 case), 10 cases selected

## Note on Rank Definitions
The probe uses pre-recommender rank-1 (final blended probability, no BANKER/market-favourite override). Production's rank-1 may differ on ~10-20% of races due to recommender swaps. Traces here reflect the model's probability assessment, not the bet actually placed.

## Critical System-Level Finding (Pre-Case Summary)

Before the per-case breakdown, one finding is so pervasive it must be stated up front:

**Platt calibration compresses XGBoost probabilities by ~88x across the board.** The raw XGBoost output spans a meaningful range (0.07–0.54 across the 10 cases), but after Platt scaling, ALL horses in a race cluster into a band of 0.089–0.095 — a spread of just 0.006. This makes the softmax step nearly uniform (all horses get ~1/field_size), which means **the market blend (55% weight for large sprint fields) is the primary determinant of final rank**. The model has effectively been reduced to a thin perturbation on top of the market odds. This is an architectural finding, not a per-race anomaly. See "Notes for Stage 2" for recommended follow-up.

---

## Severe Miss Cases

### Case 1: 2026/04/06 R5 — SUPERB SPIRIT (top_2_share=0.401)

**Model's rank-1:** SUPERB SPIRIT, platt_prob=0.0924, blended_prob=0.2193, finish=4
**Actual winner:** THOUSAND SPIRIT, our rank=5, platt_prob=0.0906, blended_prob=0.0621
**Market view:** rank-1 market_implied_prob=0.3402, winner market_implied_prob=0.0544
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| prior_win_rate | 0.500 (fallback) | 0.133 | −0.367 |
| recent_form | 0.500 (fallback) | 0.280 | −0.220 |
| prior_pace | 0.500 (fallback) | 0.908 | +0.408 |
| draw_advantage | 0.600 | 0.800 | +0.200 |
| adjusted_speed | 0.500 (fallback) | 0.286 | −0.214 |
| cd_form_prior | 0.500 (fallback) | 0.200 | −0.300 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null (opaque — rule engine not in backtest) | null |
| raw_ml_prob | 0.3455 | 0.1930 |
| platt_prob | 0.0924 | 0.0906 |
| combined_prob | null (opaque — rule+ML blend not in backtest) | null |
| softmax_prob | 0.0716 | 0.0714 |
| blended_prob | 0.2193 | 0.0621 |

**Suspected failure layer:** feature-driven
**Suspect features:** prior_win_rate, recent_form, adjusted_speed, cd_form_prior, jockey_horse_coop, going_performance, field_rel_speed, last_start_margin, distance_pb_ratio, speed_consistency, prior_pace, days_since_last_win
**Notes:** SUPERB SPIRIT had zero prior races in DB — all 12 history-dependent features defaulted to 0.5. The model had no information about this horse. Despite this, it ranked #1 because its market odds gave it 34% implied probability, and the 55% market blend elevated it. THOUSAND SPIRIT's actual pace advantage (0.908 prior_pace) and some DB history were invisible to the model for rank-1.

---

### Case 2: 2026/04/06 R4 — GALLANT EPOCH (top_2_share=0.399)

**Model's rank-1:** GALLANT EPOCH, platt_prob=0.0912, blended_prob=0.2150, finish=6
**Actual winner:** LUCKY MAN, our rank=4, platt_prob=0.0893, blended_prob=0.0951
**Market view:** rank-1 market_implied_prob=0.3279, winner market_implied_prob=0.1101
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| prior_win_rate | 0.556 | 0.000 | −0.556 |
| recent_form | 0.800 | 0.050 | −0.750 |
| adjusted_speed | 0.701 | 0.000 | −0.701 |
| freshness | 0.167 | 0.450 | +0.283 |
| weight_vs_field | 0.370 | 0.630 | +0.260 |
| cd_form_prior | 0.683 | 0.200 | −0.483 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.2442 | 0.0715 |
| platt_prob | 0.0912 | 0.0893 |
| combined_prob | null | null |
| softmax_prob | 0.0770 | 0.0768 |
| blended_prob | 0.2150 | 0.0951 |

**Suspected failure layer:** feature-driven
**Suspect features:** freshness, weight_vs_field, jockey_horse_coop
**Notes:** GALLANT EPOCH looked strong on form (recent_form=0.800), speed (0.701), and C&D history (0.683) — all legitimate data-driven signals. LUCKY MAN won despite having no wins in DB (prior_win_rate=0.000, recent_form=0.050) and no speed data, but had a significant freshness advantage (0.450 vs 0.167 — GALLANT EPOCH was running back quickly on 8 days rest) and lighter weight allocation. This is a genuine miss where the Platt compression (~89x) destroyed the model's conviction in rank-1: raw_ml_prob of 0.2442 vs winner's 0.0715 — a 3.4x raw difference became 0.002 after Platt.

---

### Case 3: 2026/04/12 R10 — QUANTUM LEGEND (top_2_share=0.383)

**Model's rank-1:** QUANTUM LEGEND, platt_prob=0.0903, blended_prob=0.2273, finish=9
**Actual winner:** MY MARS, our rank=8, platt_prob=0.0909, blended_prob=0.0564
**Market view:** rank-1 market_implied_prob=0.3390, winner market_implied_prob=0.0281
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| prior_win_rate | 0.500 (fallback) | 0.500 (fallback) | 0.000 |
| recent_form | 0.500 (fallback) | 0.500 (fallback) | 0.000 |
| draw_advantage | 1.000 | 0.800 | −0.200 |
| adjusted_speed | 0.500 (fallback) | 0.500 (fallback) | 0.000 |
| prior_pace | 0.500 (fallback) | 0.500 (fallback) | 0.000 |
| jockey_win_rate | 0.112 | 0.133 | +0.021 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.1637 | 0.2176 |
| platt_prob | 0.0903 | 0.0909 |
| combined_prob | null | null |
| softmax_prob | 0.0909 | 0.0910 |
| blended_prob | 0.2273 | 0.0564 |

**Suspected failure layer:** calibration-driven
**Suspect features:** prior_win_rate, recent_form, adjusted_speed, cd_form_prior, jockey_horse_coop, going_performance, field_rel_speed, last_start_margin, distance_pb_ratio, speed_consistency, prior_pace, days_since_last_win
**Notes:** Both QUANTUM LEGEND and MY MARS had zero DB history — all 12 history features defaulted to 0.5. The raw XGBoost correctly ranked MY MARS higher (0.2176 vs 0.1637) — winner was identified at the raw layer. Platt then compressed both to ~0.090, erasing the 0.054 raw difference to 0.0006 (compression factor ~87x). After softmax the two were indistinguishable. The 55% market blend then amplified the damage: MY MARS had only 2.8% market share (outsider) vs QUANTUM LEGEND's 33.9%, resulting in blended probs of 0.057 vs 0.227. Root cause: Platt calibration destruction of the correct raw ML signal. A longshot winner invisible to both DB history and market, but the model got it right before calibration ruined it.

---

### Case 4: 2026/04/12 R4 — INVICTUS DRAGON (top_2_share=0.377)

**Model's rank-1:** INVICTUS DRAGON, platt_prob=0.0936, blended_prob=0.2436, finish=5
**Actual winner:** HAPPY BOSS, our rank=3, platt_prob=0.0898, blended_prob=0.0955
**Market view:** rank-1 market_implied_prob=0.3842, winner market_implied_prob=0.1153
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| prior_win_rate | 0.200 | 0.444 | +0.244 |
| recent_form | 0.340 | 0.633 | +0.293 |
| cd_form_prior | 0.200 | 0.683 | +0.483 |
| prior_pace | 0.431 | 0.641 | +0.210 |
| field_rel_draw | 0.571 | 0.929 | +0.357 |
| draw_advantage | 0.800 | 0.300 | −0.500 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.4529 | 0.1238 |
| platt_prob | 0.0936 | 0.0898 |
| combined_prob | null | null |
| softmax_prob | 0.0718 | 0.0714 |
| blended_prob | 0.2436 | 0.0955 |

**Suspected failure layer:** feature-driven
**Suspect features:** prior_win_rate, recent_form, cd_form_prior, going_performance, field_rel_speed, prior_pace, days_since_last_win, field_rel_draw
**Notes:** HAPPY BOSS dominated on all meaningful history features — better form (0.633 vs 0.340), stronger C&D record (0.683 vs 0.200), better prior pace (0.641 vs 0.431). INVICTUS DRAGON's raw XGBoost score was 0.4529 — very high — but Platt crushed this to 0.0936, barely above HAPPY BOSS's 0.0898. Market then pushed INVICTUS DRAGON to the top at 38.4% implied. The raw ML signal might have captured legitimate features (INVICTUS DRAGON had better draw advantage 0.800 vs 0.300), but the market override at 55% weight overrode what HAPPY BOSS's feature profile warranted.

---

### Case 5: 2026/03/29 R9 — MAGNIFIQUE (top_2_share=0.376)

**Model's rank-1:** MAGNIFIQUE, platt_prob=0.0919, blended_prob=0.1951, finish=9
**Actual winner:** YOUNG CHAMPION, our rank=4, platt_prob=0.0906, blended_prob=0.1075
**Market view:** rank-1 market_implied_prob=0.2727, winner market_implied_prob=0.1136
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| prior_win_rate | 0.778 | 0.167 | −0.611 |
| recent_form | 0.833 | 0.317 | −0.517 |
| adjusted_speed | 0.976 | 0.478 | −0.498 |
| prior_pace | 0.852 | 0.333 | −0.519 |
| draw_advantage | 1.000 | 0.500 | −0.500 |
| going_performance | 0.500 (fallback) | 0.500 (fallback) | 0.000 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.3026 | 0.1899 |
| platt_prob | 0.0919 | 0.0906 |
| combined_prob | null | null |
| softmax_prob | 0.1001 | 0.1000 |
| blended_prob | 0.1951 | 0.1075 |

**Suspected failure layer:** feature-driven
**Suspect features:** going_performance
**Notes:** This is arguably the "cleanest" miss — MAGNIFIQUE had legitimately strong features across the board: 77.8% prior win rate, 83.3% recent form, 97.6% adjusted speed (top speed figure in field), excellent pace position (85.2%), and ideal draw. The feature signal was real. But MAGNIFIQUE finished 9th — a genuine on-the-day underperformance (likely injury, track, or mid-race incident). The going_performance feature for both horses was at fallback (0.5), suggesting the going condition on the day may have been a factor neither horse had experience in. This is the one case where feature quality is not the culprit — the race outcome was simply unpredictable from any available pre-race signal.

---

### Case 6: 2026/04/22 R6 — BEAUTY SHOW (top_2_share=0.372)

**Model's rank-1:** BEAUTY SHOW, platt_prob=0.0917, blended_prob=0.2053, finish=4
**Actual winner:** KING PROFIT, our rank=3, platt_prob=0.0906, blended_prob=0.1041
**Market view:** rank-1 market_implied_prob=0.3050, winner market_implied_prob=0.1211
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| recent_form | 0.700 | 0.533 | −0.167 |
| draw_advantage | 0.600 | 0.300 | −0.300 |
| field_rel_draw | 0.667 | 1.000 | +0.333 |
| cd_form_prior | 0.200 (fallback-like) | 0.500 (fallback) | +0.300 |
| going_performance | 0.500 (fallback) | 0.500 (fallback) | 0.000 |
| jockey_horse_coop | 0.500 (fallback) | 0.222 | −0.278 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.2848 | 0.1916 |
| platt_prob | 0.0917 | 0.0906 |
| combined_prob | null | null |
| softmax_prob | 0.0835 | 0.0834 |
| blended_prob | 0.2053 | 0.1041 |

**Suspected failure layer:** feature-driven
**Suspect features:** cd_form_prior, jockey_horse_coop, going_performance, speed_consistency, field_rel_draw, horse_weight_change
**Notes:** BEAUTY SHOW was ranked #1 by both the model (raw 0.285) and the market (30.5%), a case of both signals agreeing on the wrong horse. KING PROFIT's field_rel_draw was superior (1.000 — widest draw in field, beneficial at this distance), suggesting the draw bias module underweighted wide draws at this particular course configuration. Both horses lacked going_performance data (fallback), masking any going preference signal. BEAUTY SHOW's jockey_horse_coop at fallback (0.5) means the model assigned average rider-horse familiarity despite the pair's actual history being unknown.

---

### Case 7: 2026/04/06 R6 — LOOKING BRIGHT (top_2_share=0.367)

**Model's rank-1:** LOOKING BRIGHT, platt_prob=0.0922, blended_prob=0.2172, finish=8
**Actual winner:** FAST RESPONDER, our rank=3, platt_prob=0.0913, blended_prob=0.1273
**Market view:** rank-1 market_implied_prob=0.3265, winner market_implied_prob=0.1632
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| prior_pace | 0.909 | 0.432 | −0.477 |
| adjusted_speed | 0.553 | 0.726 | +0.173 |
| jockey_horse_coop | 0.167 | 0.500 (fallback) | +0.333 |
| going_performance | 0.500 (fallback) | 0.400 | −0.100 |
| speed_consistency | 0.500 (fallback) | 0.500 (fallback) | 0.000 |
| draw_advantage | 0.500 | 0.600 | +0.100 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.3280 | 0.2524 |
| platt_prob | 0.0922 | 0.0913 |
| combined_prob | null | null |
| softmax_prob | 0.0836 | 0.0835 |
| blended_prob | 0.2172 | 0.1273 |

**Suspected failure layer:** feature-driven
**Suspect features:** jockey_horse_coop, going_performance, speed_consistency
**Notes:** LOOKING BRIGHT had a very high prior_pace (0.909 — consistently led early), but FAST RESPONDER had better adjusted_speed (0.726 vs 0.553). In a pace setup where an early leader holds on, LOOKING BRIGHT would be favoured, but FAST RESPONDER's superior finishing speed proved decisive. The pace feature overweighted front-running style without accounting for whether the pace scenario would collapse. LOOKING BRIGHT's jockey_horse_coop was very low (0.167 — poor history with this jockey), a negative signal the model partially captured but not enough to demote it.

---

### Case 8: 2026/04/12 R7 — SHOTGUN (top_2_share=0.364)

**Model's rank-1:** SHOTGUN, platt_prob=0.0918, blended_prob=0.2376, finish=4
**Actual winner:** ALABAMA SONG, our rank=6, platt_prob=0.0899, blended_prob=0.0644
**Market view:** rank-1 market_implied_prob=0.3734, winner market_implied_prob=0.0587
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| prior_win_rate | 0.429 | 0.048 | −0.381 |
| recent_form | 0.650 | 0.117 | −0.533 |
| adjusted_speed | 0.210 | 0.582 | +0.372 |
| field_rel_draw | 0.357 | 0.643 | +0.286 |
| jockey_horse_coop | 0.500 (fallback) | 0.500 (fallback) | 0.000 |
| weight_vs_field | 0.481 | 0.333 | −0.148 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.3004 | 0.1276 |
| platt_prob | 0.0918 | 0.0899 |
| combined_prob | null | null |
| softmax_prob | 0.0716 | 0.0714 |
| blended_prob | 0.2376 | 0.0644 |

**Suspected failure layer:** feature-driven
**Suspect features:** adjusted_speed, jockey_horse_coop, speed_consistency, field_rel_draw, weight_vs_field
**Notes:** SHOTGUN had strong form history (43% win rate, 65% recent form) but very poor adjusted_speed (0.210 — slowest speed figure in sample). ALABAMA SONG had weak form stats but excellent speed (0.582) and a favourable field-relative draw (0.643). The model's raw XGBoost weighted form/win_rate too heavily over speed, placing SHOTGUN 2.4x above ALABAMA SONG. The market also backed SHOTGUN (37.3% implied) — both model and market failed on this longshot winner.

---

### Case 9: 2026/04/08 R9 — MIGHTY COMMANDER (top_2_share=0.364)

**Model's rank-1:** MIGHTY COMMANDER, platt_prob=0.0937, blended_prob=0.1934, finish=8
**Actual winner:** SIGHT HERMOSO, our rank=2, platt_prob=0.0923, blended_prob=0.1704
**Market view:** rank-1 market_implied_prob=0.2831, winner market_implied_prob=0.2415
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| adjusted_speed | 1.000 | 0.616 | −0.384 |
| prior_pace | 0.879 | 0.600 | −0.279 |
| freshness | 0.417 | 0.717 | +0.300 |
| weight_vs_field | 0.259 | 0.074 | −0.185 |
| jockey_horse_coop | 0.500 (fallback) | 0.500 (fallback) | 0.000 |
| speed_consistency | 0.500 (fallback) | 0.500 (fallback) | 0.000 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.4648 | 0.3370 |
| platt_prob | 0.0937 | 0.0923 |
| combined_prob | null | null |
| softmax_prob | 0.0836 | 0.0834 |
| blended_prob | 0.1934 | 0.1704 |

**Suspected failure layer:** feature-driven
**Suspect features:** jockey_horse_coop, speed_consistency, freshness
**Notes:** This is the narrowest miss in the set — SIGHT HERMOSO was ranked #2 with blended_prob=0.1704 vs MIGHTY COMMANDER's 0.1934, and both had market support (28.3% vs 24.2%). MIGHTY COMMANDER's adjusted_speed was the highest in the field (1.000) and prior_pace excellent (0.879), but its freshness was poor (0.417 — short turnaround). SIGHT HERMOSO was considerably fresher (0.717). The model weighted speed/pace signals over freshness, missing the fitness deficit. With less Platt compression the raw difference (0.4648 vs 0.3370) would have correctly favoured MIGHTY COMMANDER, suggesting a legitimate speed vs fitness tradeoff that both model and market overrode based on form.

---

### Case 10: 2026/04/08 R5 — MONARCH COUNTY (top_2_share=0.364)

**Model's rank-1:** MONARCH COUNTY, platt_prob=0.0946, blended_prob=0.2153, finish=7
**Actual winner:** SMART FIGHTER, our rank=2, platt_prob=0.0894, blended_prob=0.1484
**Market view:** rank-1 market_implied_prob=0.3228, winner market_implied_prob=0.2018
**Blend weights used:** model_w=0.45, market_w=0.55

**Feature comparison (key deltas):**

| Feature | Rank-1 | Winner | Delta |
|---|---|---|---|
| adjusted_speed | 0.629 | 0.126 | −0.503 |
| freshness | 0.133 | 0.950 | +0.817 |
| weight_vs_field | 0.037 | 0.370 | +0.333 |
| going_performance | 0.400 | 0.000 | −0.400 |
| prior_win_rate | 0.350 | 0.067 | −0.283 |
| cd_form_prior | 0.500 (fallback) | 0.200 | −0.300 |

**Probability trace:**
| Layer | Rank-1 | Winner |
|---|---|---|
| rule_score | null | null |
| raw_ml_prob | 0.5358 | 0.0863 |
| platt_prob | 0.0946 | 0.0894 |
| combined_prob | null | null |
| softmax_prob | 0.0838 | 0.0832 |
| blended_prob | 0.2153 | 0.1484 |

**Suspected failure layer:** feature-driven
**Suspect features:** cd_form_prior, jockey_horse_coop, distance_pb_ratio, freshness, weight_vs_field
**Notes:** The most extreme Platt compression case: MONARCH COUNTY's raw_ml_prob was 0.5358 — 6.2x higher than SMART FIGHTER's 0.0863. After Platt: 0.0946 vs 0.0894 — a difference of 0.0052 (compression 87.5x). SMART FIGHTER had a massive freshness advantage (0.950 — ideal spacing, likely first start after a spell) vs MONARCH COUNTY's 0.133 (very short turnaround) and was significantly lighter in weight (weight_vs_field 0.370 vs 0.037 — MONARCH COUNTY was carrying near the maximum weight in the field). These two fitness signals — freshness and weight — were systematically underweighted at the raw ML level. SMART FIGHTER's going_performance (0.000 — no wins on this going) is a red herring since it still won.

---

## Failure Layer Summary
- Feature-driven: 9 cases — top suspect features (counted across cases): jockey_horse_coop (8 cases), going_performance (6), speed_consistency (6), cd_form_prior (5), freshness (3), adjusted_speed (3), prior_win_rate (3), field_rel_speed (3), distance_pb_ratio (3), prior_pace (3), weight_vs_field (3)
- Calibration-driven: 1 case (Apr 12 R10) — raw ML correctly ranked winner higher (0.2176 vs 0.1637), but Platt calibration compressed both probabilities to ~0.090, erasing the 0.054 raw signal and allowing market blend to override
- Blend-driven: 0 cases in isolation — market consistently boosted rank-1 but this interacted with feature-driven misranking
- Mixed: 0 cases

## Opaque Layers Encountered (Audit Findings)

Two layers are not cleanly traceable without modifying production code, and are recorded as `null` in trace-analysis.json:

1. **rule_score (0-100 from rule engine):** The backtest pipeline (`scripts/backtest.py`) uses only the ML model — it does not instantiate `CombinedRaceAnalyzer` or call the rule engine. Obtaining rule scores would require fetching live race cards (not stored in DB) or running `core/analyzer.py` separately, which would require live HKJC data. Rule scores are opaque in this audit.

2. **combined_prob (60/40 rule+ML blend):** Depends on rule_score (opaque, above). The backtest's `final_prob` is NOT equivalent to the dashboard's `combined_probability` — the backtest uses ML-only signal (Platt-calibrated) → softmax → market blend, while the dashboard does rule_score → base-rate sigmoid × 0.60 + ml_prob × 0.40 → softmax → market blend. These are architecturally different pipelines. Any comparison of backtest results to dashboard ranking must account for this.

**Additional finding — Platt over-compression (auditable, not opaque):** The Platt calibrator is compressing all XGBoost probabilities by approximately 88x (measured across all 10 cases). Post-Platt, all horses in a 12-horse field cluster into a 0.006-wide band (0.089–0.095), making softmax essentially uniform. This is measurable and reproducible — it is not an opaque layer but an architectural problem: the calibrator appears to have been fitted on a prior where winners are extremely rare, or the validation set had very few winners, causing the calibrator to compress everything toward the base rate (1/field_size ≈ 0.083). The functional consequence is that the ML model provides almost no ranking signal — the market blend determines rank-1 in ~100% of cases.

## Apr 22 Probe

### Context
- Apr 22 2026 (Sha Tin — confirmed from DB; task description said Happy Valley but DB is ground truth) — the most recent race day and one of the "rough days" that triggered this audit.
- In holdout: **YES** — all 9 races are in the 73-race holdout (the 20% held-out tail). Apr 22 is the final date in the DB, so its 107 rows are all in the holdout slice.
- Probed against production model version `20260423-012955`.
- Script: `docs/audits/2026-04-23/raw/_probe_apr22.py`
- Output: `docs/audits/2026-04-23/raw/apr22-trace.json`

### Race-by-race summary

| Race | Field | Top-1 prob | Top-2 share | Rank-1 finish | Winner our-rank | Failure flags | Failure layer | Platt compression |
|------|-------|------------|-------------|---------------|-----------------|---------------|---------------|-------------------|
| R1   | 12    | 0.1549     | 0.287       | 2             | 2 (NEBRASKAN)   | —             | feature-driven | 87.91×           |
| R2   | 11    | 0.1750     | 0.318       | 2             | 3 (PODIUM)      | —             | calibration-driven | 87.94×       |
| R3   | 12    | 0.1132     | 0.223       | 5             | 2 (SHOOTING TO TOP) | rank1_miss, qin_box3_miss, qin_box4_miss | calibration-driven | 87.27× |
| R4   | 12    | 0.2324     | 0.369       | **1** ✓       | 1 (VIGOR EYE)   | qin_box3_miss, qin_box4_miss | **no-miss** | 88.98× |
| R5   | 12    | 0.2419     | 0.366       | **1** ✓       | 1 (THUNDER PRINCE) | —          | **no-miss**   | 89.13×           |
| R6   | 12    | 0.2053     | 0.372       | 4             | 3 (KING PROFIT) | rank1_miss, top2_miss, qin_box3_miss | feature-driven | 89.05× |
| R7   | 12    | 0.2112     | 0.334       | 3             | **8** (CASA OF HONOR) | winner_deep, qin_box3_miss, qin_box4_miss | feature-driven | 88.45× |
| R8   | 12    | 0.1398     | 0.258       | 2             | **6** (THE AUSPICIOUS) | winner_deep, qin_box3_miss, qin_box4_miss | feature-driven | 88.62× |
| R9   | 12    | 0.1733     | 0.343       | 4             | **5** (JUBILANT WINNER) | rank1_miss, winner_deep, qin_box3_miss, qin_box4_miss | calibration-driven | 88.10× |

Notes on individual races:
- **R4**: Rank-1 correct (VIGOR EYE won) but quinella missed because runner-up was CAPTAIN LINK at $48 (rank 9 in our model — a massive longshot the market and model both dismissed).
- **R7**: CASA OF HONOR won at $22 — our rank 8. The horse had prior_win_rate=0.000, recent_form=0.000 (all fallback signals); the market also disliked it (22.0 odds). No model or market signal supported it.
- **R8**: THE AUSPICIOUS won at exactly $10.0 — marginal longshot territory; our rank 6.
- **R9**: JUBILANT WINNER at $14.0 — our rank 5; R9 is the only race with 4 active failure flags.

### Platt compression on Apr 22

Across all 9 Apr 22 races: **average raw_ml_range = 0.3710**, **average platt_range = 0.004207**, **average compression ratio = 88.4×**.

Range spans 87.27×–89.13× — consistent within ±1× across all 9 races.

Comparison to Task 3's 10 severe-miss cases (measured pair-wise as |raw_diff| / |platt_diff|): avg **88.1×**, range 86.6×–89.0×. The two measurements — range-based across full field (Apr 22) and pair-wise rank1-vs-winner (Task 3) — converge to the same figure within 0.3×.

**Conclusion: Platt compression is not isolated to severe-miss cases.** It is a constant of the model's calibration — indistinguishable between the 2 races VIGOR EYE and THUNDER PRINCE won correctly and the 7 races with at least one miss. The compression is structural, not race-specific.

### Failure pattern

Flag counts across 9 Apr 22 races:
- `qin_box3_miss`: **6/9** (67%) — the dominant failure mode on Apr 22
- `qin_box4_miss`: **5/9** (56%) — box-4 captured 1 additional hit vs box-3 (R1: winner rank-2, runner-up captured by box-4 but not box-3)
- `rank1_miss`: **3/9** (33%) — R3, R6, R9
- `winner_deep`: **3/9** (33%) — R7, R8, R9 (winners at rank 8, 6, 5 respectively)
- `top2_miss`: **1/9** (11%) — R6 only (KING PROFIT finished 3rd, outside even our top-2)

Failure layer distribution: feature-driven **4/9**, calibration-driven **3/9**, no-miss **2/9**.

The split between feature-driven and calibration-driven is notably different from the Task 3 severe-miss sample (9/10 feature-driven, 1/10 calibration-driven). Apr 22 shows a higher calibration-driven share (33% vs 10%). This makes sense: the Task 3 filter selected top_2_share ≥ 0.35 (high-conviction races), which naturally biases toward cases where a highly-featured horse failed. Apr 22 includes lower-conviction races (R1: top_2_share=0.287, R3: 0.223) where Platt destruction of a correct raw ML signal is more likely to be the root cause.

The two "correct" races (R4, R5) are the highest top_2_share races on the day (0.369 and 0.366). This is consistent with the hypothesis that when the model and market agree on a clear favourite, the blend produces a correct pick — not because the ML model is informative, but because both sources converge.

### Cross-reference to Apr 19 learnings

The Apr 19 learnings (from project memory) reported: longshot-heavy day, 5/11 winners at $10+, maiden winners, rank-2 sweet spot.

**Maiden winners** (prior_win_rate=0.000 in DB): NEBRASKAN (R1) and CASA OF HONOR (R7) — **2 of 9 winners** (22%). Not as extreme as Apr 19 but still notable. A horse with zero prior wins is invisible to all win-rate-based features.

**Rank-2 finishes**: Winners ranked #2 by our model in **2 of 9 races** (R1 NEBRASKAN, R3 SHOOTING TO TOP). An additional winner was rank-3 (R2 PODIUM, R6 KING PROFIT). No winner was exactly at rank-2 in the "sweet spot" sense — the pattern is more spread: 2×rank-2, 2×rank-3, 1×rank-5, 1×rank-6, 1×rank-8.

**Longshot winners** (>$10 odds): CASA OF HONOR ($22.0), JUBILANT WINNER ($14.0), THE AUSPICIOUS ($10.0 exactly) — **3 of 9 winners** (33%). Apr 19 had 5/11 (45%) — Apr 22 shows a similar but less extreme distribution. The $22 CASA OF HONOR winner is Apr 22's most egregious longshot: rank-8 in our model, zero prior history in DB.

### Implications for Stage 2

1. **Platt compression is universal, not a severe-miss artifact.** The 88.4× compression on Apr 22's full day — including 2 correct calls — proves that compression is not caused by difficult races or feature failures. It is a fixed property of the calibrator coefficients. Stage 2's probability-expert subagent should measure the calibrator's decision boundary (coefficients) and determine what validation-set winner prevalence would produce this level of compression.

2. **Calibration-driven failures are more common in average races than severe-miss selection implies.** Task 3 showed 1/10 (10%) calibration-driven misses; Apr 22 shows 3/9 (33%). The difference is entirely explained by Task 3's selection bias toward high top_2_share races. In R2 and R3 (the cleanest calibration-driven Apr 22 cases), the raw XGBoost correctly identified the winner (R2: raw_ml_prob=0.4866 for winner PODIUM vs 0.1221 for rank-1 SOARING BRONCO — a 4:1 raw signal that Platt compressed to a 0.004 gap), but Platt destroyed the signal. This is an additional point of evidence that fixing Platt calibration alone (Stage 2's first target) could immediately recover 2-3 additional correct picks per 9-race day.

3. **Box-4 provides marginal benefit over box-3 on Apr 22.** Box-3 missed 6/9; box-4 missed 5/9 — only a 1-race improvement. On Apr 22 the winners who escaped box-4 were at ranks 5, 6, and 8 — far outside any reasonable box expansion. The strategic recommendation to default to box-4 is confirmed as directionally correct but not sufficient to capture the R7/R8/R9 winners.

## Notes for Stage 2

1. **Platt calibrator is the most urgent investigation target.** All 10 severe misses show Platt compressing by ~88x, reducing the ML model to a ~0.006-range perturbation on top of the market. Stage 2's probability-expert subagent should inspect the calibrator's coefficients, the validation set's winner prevalence, and whether re-fitting Platt on a properly balanced set would restore ranking differentiation.

2. **`jockey_horse_coop` is flagged in 8/10 cases as a suspect feature** — specifically, it defaults to 0.5 (fallback) for most horses because fewer than 2 shared rides exist in the local DB. The feature is defined but receives no signal in practice. Stage 2 should check whether this feature is providing any discriminative value in training or whether it should be removed/replaced with a jockey form score from the jockey_stats table.

3. **`going_performance` at fallback (0.5) recurs in 6/10 cases.** The DB has limited going-specific history per horse, meaning this feature fires for very few horses. The model learned on mostly-fallback going_performance values — the feature may be adding noise rather than signal. Stage 2 should measure its training data coverage rate and consider whether folding going_performance into the jockey_win_rate or trainer_win_rate calculation would be more robust.
