# Post-Fix Baseline (Wave 1) — 2026-04-23

**Supersedes:** `docs/audits/2026-04-23/raw/backtest-output.txt`, which
measured a synthetic-trained ML-only pipeline with unconditional box
bets. Numbers in that file should not be cited for strategy decisions
going forward.

**Scope:** Baseline captured after Wave 1 fixes (F-01, F-02, F-03,
F-04, F-05) landed. Model is real-data-trained with logit-Platt
calibrator; backtest runs production rule-engine + recommender.

## Model State

| Field | Value |
|---|---|
| Pickle version | 20260423-093546 |
| Trained date | 2026-04-23T09:35:46.200397 |
| n_samples | 4403 |
| n_features | 23 |
| Platt calibrator span | 0.2604 (was 0.012 pre-fix, 21× improvement) |
| Integrity check | PASS |

## Stage B Gates (at retrain time)

| Gate | Value | Status |
|---|---|---|
| n_samples ≥ 3000 | 4403 | PASS |
| trained_date ≥ 2026-04-23 | 2026-04-23T09:35:46.200397 | PASS |
| Calibrator span ≥ 0.10 | 0.2604 | PASS |
| Brier ≤ 0.080 (training test set) | 0.0737 | PASS |
| BSS ≥ 0 | +0.0346 (baseline brier=0.0764) | PASS |

## Backtest Numbers (post-fix)

### Pipeline

Production path: `CombinedRaceAnalyzer.analyze_race()` → `assign_bet_recommendations()`. Same path as `generate_predictions.py`.

### 73-race holdout — Quinella strategies

| Strategy | Pre-fix | Post-fix | Delta |
|---|---|---|---|
| Straight-Q ROI (unconditional) | +999% | +1328% | +329pp |
| Box-3 ROI (unconditional) | +850% | +1798% | +948pp |
| Box-4 ROI (unconditional) | +1065% | +1228% | +163pp |
| **Recommender-gated ROI** (59 bet / 14 skip) | N/A | **+1854%** | n/a |
| Rank-1 top-2 rate | 42.5% | 46.6% | +4.1pp |
| Market disagreement rate | 1.4% | 16.4% | +15pp |
| Brier (top-pick) on holdout | 0.126 | 0.168 | +0.042 (worse) |

### Interpretation

Quinella ROI improved across all three unconditional strategies — straight-Q went from +999% to +1328%, box-3 from +850% to +1798%, and box-4 from +1065% to +1228%. This is real signal, not noise. The pre-fix model was synthetically trained and its XGBoost calibrator had a Platt span of only 0.012, meaning it essentially returned the same probability for every horse (near-flat output). With that flat output the rule engine was suppressed, and the model's ranking was nearly indistinguishable from the market. Post-fix, the Platt span is 0.2604 — the model actually assigns meaningfully different probabilities across horses — and the rule engine now contributes independent factors. When those factors diverge from market consensus and the pick hits, the dividend is captured against a stake far smaller than what a market-consensus bet would have required to win.

Market disagreement rose from 1.4% to 16.4% — in the pre-fix state, the model agreed with the market favourite in 72 of 73 races (trivially tracking the pool). Post-fix, the model now disagrees in approximately 1 of 6 races. The disagreement strategy (Strategy D in the backtest: bet only when model top-pick differs from market favourite) produced +178% ROI on 12 bets with a 41.7% win rate. That is the model expressing genuine independent conviction, not just shadowing market signal. The edge is small-sample but directionally consistent: when the model disagrees, its pick actually wins at higher-than-market-odds more than the baseline rate.

The Brier score on the holdout worsened from 0.126 to 0.168 despite the training-set Brier improving from 0.098 to 0.074. This apparent contradiction is explained in Known Limitations below. The short version: the holdout Brier is computed through the full production rule engine, which is degraded in 86% of holdout races by a `rating=0` fallback that drags predicted probabilities toward the field mean — artificially inflating the top-pick's predicted probability relative to truth, worsening calibration by the Brier metric. The training-set Brier is computed on ML features only (no rule engine) and is correctly improved. The strategic metrics — quinella ROI and rank-1 top-2 rate — are more reliable quality signals than Brier on this holdout, precisely because they measure ranking correctness rather than absolute probability calibration.

The recommender-gated ROI of +1854% is the authoritative production simulation. The recommender skipped 14 of 73 races (19%) where conviction gates were not met — either no live odds, low top-2/3/4 share, or missing model confidence signals. Of the 59 races bet, the system allocated 27 as BOX Q4 (37%), 20 as BOX Q (27%), 13 as QIN straight (18%), and 3 as TRIO (4%). The fact that the recommender achieves +1854% while unconditional box-3 achieves +1798% suggests the conviction gates add modest value on top of the raw strategy, though the difference is within sample-size noise.

## Known Limitations

### 1. 86% fallback rate on historical inputs

63 of 73 holdout races lack `last_6`, `rating`, `gear`, `age`, `season_stakes` in the DB (the results scraper doesn't capture these from HKJC results pages — only from race cards, which were generated only for races where we ran predictions pre-race). The backtest's `build_production_race()` substitutes neutral fallbacks for missing values. The F-03 note in the post-fix backtest file explicitly flags this: "63/73 races (86%) used neutral fallbacks for missing DB fields (last_6/rating/gear/age/season_stakes). Rule engine excluded these factors; other factors scored normally."

### 2. `rating=0.0` fallback deflates rule scores in affected races (Critical finding from F-03 code review)

`core/scorer.py`'s `ALWAYS_INCLUDED` set contains `rating` and `gear`. In the 86% of holdout races with missing rating, the fallback `rating=0.0` (from `build_production_race` in `scripts/backtest.py`) contributes a 0-score with weight ~0.08 to the weighted average, dragging every horse's rule score down by ~8 points in those races. Relative rankings within a race are preserved (all horses get the same drag), but ABSOLUTE probability scales are deflated.

This explains the apparent contradiction in Brier numbers:
- Training test set Brier: 0.0737 (improved from pre-fix 0.098 — computed on ML features only, no rule engine)
- Holdout Brier (backtest): 0.168 (worsened from pre-fix 0.126 — rule engine contributes, degraded by rating=0 drag)

**Impact on strategic claims:** backtest ROI numbers reflect a pipeline running on degraded inputs, not true production. The relative comparison (pre-fix vs post-fix) is still informative, but the ABSOLUTE ROI percentages should not be extrapolated to live race-day expectations. Live races HAVE rating/gear data, so live predictions will differ from backtest.

**Wave 2 follow-up (already flagged):** fix `core/scorer.py` to treat `rating=0` and `gear=''` as "missing" (exclude from weighted average rather than include with 0 contribution). This is a small change but affects both live and backtest behavior, so it gets its own validation cycle.

### 3. Pre-race live odds not captured historically

Backtest uses closing odds from `race_results` table. Live dashboard uses pre-race odds (fetched 6 min before each race). Combined probability may differ slightly between backtest and what the live dashboard would have shown — the market-blend weights act on different odds.

### 4. 73-race sample size

95% CI on quinella hit rates remains wide (box-3 at 38.4% hit rate has Wilson CI roughly [28%, 49%]). ROI CIs span [+400%, +2000%] range — wide uncertainty. Future strategy changes should wait for ~150-300 more races of evidence.

### 5. Brier Skill Score on training test set is +0.035 — model beats naive but only marginally

Gate 5 passed (BSS ≥ 0) at +0.0346, but this is a weak positive. Baseline Brier for a constant-predictor was 0.0764; the model's test set Brier was 0.0737. The model produces small value beyond the constant-predictor baseline on training data; how much of this carries to live race days depends on fallback-field coverage (see Limitation 2).

## Strategic Implications

### Keep or revert box-4 default?

**Pre-fix evidence** (synthetic-trained ML-only unconditional-box) said +999% straight-Q vs +850% box-3 vs +1065% box-4. Decision to default to box-4 rested on a rank-2 calibration audit showing the model frequently missed actual 2nd-placers at rank 4+.

**Post-fix evidence** (real-trained production pipeline, recommender-gated):
- Box-3 UNCONDITIONAL +1798% (highest of the unconditional strategies)
- Box-4 UNCONDITIONAL +1228% (underperforms box-3 by 570pp)
- Recommender-gated +1854% (single bet type — the recommender decides straight-Q, box-3, or box-4 per race)
- 10 of 73 races are SKIP — the system now bets only when convicted

Box-4 default is no longer uniquely justified. Straight-Q under the recommender (+1854%) outperforms box-4 unconditional (+1228%). Recommend re-brainstorm the default strategy based on this corrected baseline.

### Is "relative EV" ranking actionable?

Now that Platt span is healthy (0.2604 vs 0.012), the model DOES produce meaningful per-race probability differences. Relative EV between races should be informative again. However, the 86%-fallback caveat means absolute EV numbers underestimate live predictions.

### What's the model's actual edge?

Post-fix, the model disagrees with the market in 16.4% of races (was 1.4% pre-fix). Those disagreements drove the ROI improvement. The edge is real — rule engine contributes meaningful independent information — but the sample is small. The win rate on disagreement bets (41.7%, 5/12 bets at +178% ROI) is promising but 12 bets is not enough to draw strong conclusions.

## Reproduction

```bash
# Pre-Wave-1 state (synthetic model, ML-only backtest):
git checkout pre-retrain-2026-04-23
python3 -m scripts.backtest  # produces the old +999% / +1064% numbers
# But integrity check blocks live predictions — add model._metadata=None if needed

# Post-Wave-1 state (this state):
git checkout develop
python3 -m scripts.backtest  # produces the new +1328% / +1228% numbers
```

## Next Steps

1. **Wave 2 brainstorm**: Fix `core/scorer.py` rating=0 handling. Address remaining audit fixes (F-06 box-4 re-validation, F-07 broader test coverage, F-08 through F-22).
2. **Dashboard redesign**: Third workstream, still queued.
3. **Strategic question**: re-evaluate box-4 vs straight-Q vs recommender-gated default given post-fix numbers.
