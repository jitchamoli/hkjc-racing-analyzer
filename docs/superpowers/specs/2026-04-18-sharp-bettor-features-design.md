# Sharp Bettor Features — Design Spec

**Date**: 2026-04-18
**Status**: Approved
**Scope**: Three features — adjusted speed figures, exotic bet calculator, late odds movement signal

---

## 1. Adjusted Speed Figures

### Problem
Raw `prior_speed` feature (m/s normalized to 0-1) conflates performance across different conditions. 1:09.5 in Class 5 on Good ground is not equivalent to 1:09.5 in Class 2 on Yielding.

### Design

**Par time computation**: For each (distance, going, class) combination, compute the median finish time from race_results. Require ≥5 races per bucket; fall back to (distance, going) if class bucket is too small, then (distance) only.

**Weight adjustment**: Benter's HK approximation — +0.1 seconds per pound above 120 lbs carried weight. Applied before comparing to par.

**Adjusted speed formula**:
```
weight_adjusted_time = actual_time + 0.1 * max(0, carried_weight - 120)
adjusted_speed = (par_time - weight_adjusted_time) / par_time * 100
```

Positive = faster than par, negative = slower. Normalized to 0-1 for ML via `(adjusted + 5) / 10` clamped.

**Speed consistency**: Standard deviation of a horse's last 5 adjusted speed figures. Low std = reliable horse (important for exotic bets where order matters). Normalized via `1.0 - min(1.0, std / 3.0)`.

### ML Schema Changes
- Feature 9: `prior_speed` → `adjusted_speed` (replaces)
- Feature 23: `speed_consistency` (new)

### Implementation
- `analytics/speed_figures.py` — new module: `compute_par_times(conn)`, `adjusted_speed_figure(time, distance, going, class_type, weight, par_times)`
- Precompute par times once per training run (in `build_training_data_from_db`)
- Precompute once per prediction run (in `CombinedRacingAnalyzer.__init__`)
- Dashboard: show adjusted speed figure per horse in detail view

---

## 2. Exotic Bet Calculator

### Problem
WIN/PLACE bets compete against an ~85% efficient market. Exotic bets (exacta, trifecta, first4) require pricing N! combinations — the crowd is systematically worse here, creating exploitable mispricings.

### Math

Given calibrated win probabilities P₁..Pₙ:

```
Exacta(A,B) = P(A) × P(B) / (1 - P(A))
Quinella(A,B) = Exacta(A,B) + Exacta(B,A)
Trifecta(A,B,C) = P(A) × [P(B)/(1-P(A))] × [P(C)/(1-P(A)-P(B))]
First4(A,B,C,D) = extends trifecta formula to 4th position
```

### Value Detection

For each combination, compare model probability against implied market probability:
```
edge = model_prob / market_implied_prob - 1
```

Since HKJC doesn't publish pre-race exotic pool dividends, estimate fair odds from win odds:
```
market_exacta_prob ≈ market_prob(A) × market_prob(B) / (1 - market_prob(A))
```

Flag combinations where `model_prob > market_prob × 1.10` (10%+ edge).

### Output Per Race
- Top 5 value exactas with: horses, model prob, min required odds, edge %
- Top 5 value trifectas (same)
- Top 3 value quinellas (same)
- Top 3 value first4s (same)
- Kelly-optimal stake per combination (from existing Kelly optimizer)

### Implementation
- `betting/exotic_calculator.py` — new module
  - `ExoticCalculator` class
  - `price_exactas(probs, horse_numbers)` → list of (combo, prob) sorted by prob
  - `price_trifectas(probs, horse_numbers)` → same
  - `price_quinellas(probs, horse_numbers)` → same
  - `price_first4s(probs, horse_numbers)` → same
  - `find_value_exotics(model_probs, market_probs, horse_numbers)` → value combos with edge
- Dashboard: "Exotic Value" panel per race in detail view
- WhatsApp copy: include top exotic picks

### Performance
- Exacta: N×(N-1) = 182 combinations for 14-horse field (fast)
- Trifecta: N×(N-1)×(N-2) = 2,184 (fast)
- First4: N×(N-1)×(N-2)×(N-3) = 24,024 (still <1s with numpy)

---

## 3. Late Odds Movement Signal

### Problem
Sharp money (professional bettors) moves the market. Currently tracked in dashboard (steam/drift badges) but excluded from ML features.

### Phase 1 (immediate)
Use opening vs current win odds from race card data:
```
change_pct = (current_odds - opening_odds) / opening_odds
late_money_signal:
  steam (change < -20%): 0.8-1.0 (money coming in = positive signal)
  mild steam (-10% to -20%): 0.6-0.8
  stable (-10% to +10%): 0.5
  mild drift (+10% to +30%): 0.3-0.4
  drift (> +30%): 0.0-0.2 (money leaving = negative signal)
```

### Phase 2 (future, when odds_history.db has data)
Rate of change in last 30 minutes, weighted by implied volume shift. Not in this spec.

### ML Schema Change
- Feature 24: `late_money_signal` (new)

### Implementation
- Add to `extract_features()` — compute from `horse_data.opening_odds` and `horse_data.win_odds`
- Add to `build_training_data_from_db()` — compute from `race_results.win_odds` (opening odds not in DB, so use 0.5 default for training; real signal comes at prediction time)
- Training data note: this feature will have low variance in training (mostly 0.5 default) but high signal at prediction time when live odds are available. XGBoost handles this gracefully — it learns to use the feature when available.

---

## Schema Summary

| Slot | Old | New |
|------|-----|-----|
| 9 | prior_speed | **adjusted_speed** |
| 23 | — | **speed_consistency** (new) |
| 24 | — | **late_money_signal** (new) |

Total: 22 → 24 features.

### Files Modified
- `ml/feature_schema.py` — add 2 features, rename 1
- `ml/features.py` — replace prior_speed, add 2 new features
- `ml/training.py` — same changes for training builder + synthetic
- `analytics/speed_figures.py` — new module
- `betting/exotic_calculator.py` — new module
- `dashboard/template.py` — exotic panel + speed figure display
- `dashboard/generator.py` — include exotic data in JSON
- `tests/test_features.py` — update count 22 → 24

### Files NOT Modified
- `core/analyzer.py` — no changes to rule engine
- `core/scorer.py` — no changes to scoring weights
- `config.py` — no new config needed
