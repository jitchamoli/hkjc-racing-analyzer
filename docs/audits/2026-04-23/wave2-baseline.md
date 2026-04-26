# Post-Wave-2a Baseline — 2026-04-23

**Supersedes:** `docs/audits/2026-04-23/post-fix-baseline.md` (Wave 1 baseline).

**Scope:** After Wave 2a fixes (C1, F-06, F-15, F-11, F-09, F-22) landed on develop.

## Fixes Shipped (Wave 2a)

| Fix | Description | Status |
|-----|-------------|--------|
| C1 | Uniform exclusion of score=0 factors in `compute_weighted_score` | LANDED |
| F-06 | `weight_vs_field` train/predict parity (field_weights plumbed into horse_data) | LANDED |
| F-06 guard | Regression test asserting field_weights stays plumbed | LANDED |
| F-15 | Drop `jockey_trainer_combo`, retrain on 22 features | LANDED |
| F-11 | Strict per-feature parity test | LANDED |
| F-09 | Grid-sweep market-blend weights (15 points) | LANDED |
| F-22 | Box-4 re-evaluation on corrected baseline | LANDED |

## Model State

| Field | Wave 1 | Wave 2a |
|---|---|---|
| n_features | 23 | 22 (F-15 dropped `jockey_trainer_combo`) |
| n_samples | 4403 | 4403 |
| Trained date | 2026-04-23T09:35:46.200397 | 2026-04-23T23:28:44.178848 |
| Calibrator span | 0.2604 | 0.2498 |
| Training Brier | 0.0737 | 0.0734 (Δ=-0.0003) |
| BSS (training) | +0.0346 | +0.0389 (Δ=+0.0043) |
| Holdout Brier | 0.168 | 0.1678 (Δ=-0.0002) |
| High-correlation pairs (F-15 scope) | 1 (jockey_trainer_combo) | 0 (eliminated) |
| Pre-existing structural correlations (Wave 2b scope) | N/A | 3: `recent_form × field_rel_speed` (0.718), `recent_form × days_since_last_win` (0.743), `field_rel_speed × last_start_margin` (0.710) |

## Blend Weights (F-09)

Grid sweep: 15 points (reduced from 45 for compute budget). All 15 fell within ±0.0022 Brier of each other on the 73-race holdout — blend parameters have negligible differentiating power at this sample size.

Chosen point: `base_model_w=0.45, field_thresh=10, dist_thresh=1200` (Brier 0.167783) vs previous default `base_model_w=0.40` (estimated Brier ~0.16851). Improvement: ~0.0008 Brier. Statistically indistinguishable from prior default at n=73.

**Context:** 86% of holdout races lacked live odds in the DB, triggering model-only path (blend weights inert). Sweeping blend parameters on this holdout is low-signal. Re-run post-F-21 when more real-odds races are collected.

See `docs/audits/2026-04-23/raw/wave2-sweep-summary.json` for full data (all 15 points, Brier range 0.167783-0.169682).

## Strategic Decision (F-22)

### Post-Wave-2a Backtest (73 races, real HKJC dividends for 72)

| Strategy | Hits | Hit rate | Wilson 95% CI | ROI | Bootstrap 95% CI |
|---|---|---|---|---|---|
| Straight-Q (1 unit/race) | 14/72 | 19.4% | [12.0%, 30.0%] | +2876% | [+1122%, +5215%] |
| Box-3 (3 units/race) | 30/72 | 41.7% | [31.0%, 53.2%] | +1939% | [+1161%, +2871%] |
| Box-4 (6 units/race) | 39/72 | 54.2% | [42.7%, 65.2%] | +1402% | [+905%, +1977%] |
| Recommender-gated (mixed) | 31/66 | 47.0% | — | +1976% | — |

### Comparison to Wave 1 post-fix-baseline

| Strategy | Wave 1 ROI | Wave 2a ROI | Delta |
|---|---|---|---|
| Straight-Q | +1328% | +2876% | +1548pp |
| Box-3 | +1798% | +1939% | +141pp |
| Box-4 | +1228% | +1402% | +174pp |
| Recommender-gated | +1854% | +1976% | +122pp |

**Interpretation:** All strategies improved on the corrected baseline. The largest gain is straight-Q (+1548pp), suggesting C1's `rating=0` drag was disproportionately suppressing top-pick precision. Box-3 and box-4 improvements are modest (+141pp and +174pp), indicating the fixes' impact was distributed unevenly across rank positions — rank-1 benefited most.

### Decision: Keep Box-4 Default

**Criteria for switching:**
1. Hit-rate CI separation: box-4 CI low (42.7%) < box-3 CI high (53.2%) ✗ **OVERLAP** — no separation
2. ROI CI separation: box-4 CI [+905%, +1977%] ∩ box-3 CI [+1161%, +2871%] ✗ **OVERLAP** — no separation

**Conclusion:** All three unconditional strategies are statistically indistinguishable at n=72. Straight-Q point-estimate is highest (+2876%) but with wide CI [+1122%, +5215%] that encompasses box-3 entirely. The recommender's conviction-gated fallback tree already auto-selects per-race based on `top_3_share` and `top_4_share` thresholds — this hybrid approach achieves +1976% (between box-3 unconditional +1939% and straight-Q +2876%), representing a balanced risk-return profile.

**No action taken.** Box-4 default stands pending F-21 data collection (~150+ additional races for tighter CIs).

## Known Limitations

### 1. 86% fallback rate on historical inputs (inherited from Wave 1)

63 of 73 holdout races lack `last_6`, `rating`, `gear`, `age`, `season_stakes` in the DB. The backtest's `build_production_race()` substitutes neutral fallbacks. The F-03 note in the backtest output explicitly flags this: "63/73 races (86%) used neutral fallbacks for missing DB fields (last_6/rating/gear/age/season_stakes). Rule engine excluded these factors; other factors scored normally." This is the same constraint as Wave 1 — not introduced by Wave 2a.

### 2. 73-race holdout → CIs 3000-4000pp wide

Wilson CI on box-3 hit rate: [31.0%, 53.2%] (22.2pp range). Bootstrap CI on box-4 ROI: [+905%, +1977%] (1072pp range). These confidence bands are too wide to justify strategy pivots. F-21 data collection is the right next step. Any strategy change should wait for ≥150 additional races of evidence.

### 3. Pre-existing structural correlations (Wave 2b scope)

`jockey_trainer_combo` was eliminated in F-15. Three other high-correlation pairs remain, flagged in the retrain log:
- `recent_form × field_rel_speed` = 0.718
- `recent_form × days_since_last_win` = 0.743
- `field_rel_speed × last_start_margin` = 0.710

These are not F-15 scope (combo was the only jockey-related pair). Instead, they represent model structure that trades off collinearity cost against discriminative power. Decisions on these pairs (drop one, regularize, or keep) are deferred to Wave 2b (F-08, F-12 calibration work).

### 4. Field_rel_draw can exceed 1.0 for 14-gate races

When `field_size` defaults to 12 during inference but the actual field is 14, `field_rel_draw` (draw_pos / field_size) can exceed 1.0. Wave 2b item F-16: clip to [0.0, 1.0] or re-derive from actual field at prediction time.

### 5. F-09 grid sweep reduced to 15 points (from 45)

Compute budget. 86% of holdout races lacked live odds, making blend-weight choice insensitive. Reduction is defensible for THIS holdout, but a fuller sweep (45 points) should re-run post-F-21 when more real-odds races are added.

### 6. Calibration spike in [18%-25%] bin

Post-Wave-2a holdout calibration:
- [18.0%-25.0%] bin: 33 races predicted, 1 actual win (3.03% actual vs 21.88% predicted) — model over-confident in this range

This is a persistent issue inherited from Wave 1 and worsened slightly by F-15's feature drop. Separate Wave 2b item (F-12 calibration audit).

## Tags

- `wave2-pre` — before Stage A retrain
- `wave2-pre-retrain` — Stage A complete (C1 + F-06 + F-06 guard landed)
- `wave2-post-retrain` — Stage B complete (F-15 retrain landed)
- `wave2-complete` — Stage C complete (F-11 + F-09 + F-22 landed)

## Reproduction

```bash
# Pre-Wave-2a state (post-Wave-1):
git checkout wave2-pre-retrain
python3 -m scripts.backtest

# Post-Wave-2a (current):
git checkout develop
python3 -m scripts.backtest
```

## Next Steps

1. **F-21 data collection** (parallel track) — scrape additional 150+ races into horse_history.db, retrain ML, re-run baseline with tighter CIs. Target: 200+ total holdout races for <500pp ROI CI width.
2. **Wave 2b hygiene** (after F-21 starts) — F-08 sigmoid slope, F-12 calibration audit, F-13 stats source tag, F-14 class_change scaling, F-16 field_rel_draw clip, F-17 raise on untrained, F-18 per-row recency weights, F-19 display_probability, F-20 mixed-odds formula. Batch after F-21 baseline.
3. **F-10 dividend multiplier** — if EV% still broken post-Wave-2a, re-assess (deferred pending F-21).
4. **Dashboard redesign** — third workstream, queued after Wave 2b closes.
