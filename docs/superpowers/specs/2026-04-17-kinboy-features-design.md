# Enhanced Analytics & Race Day Monitor — Design Spec

**Date:** 2026-04-17
**Status:** Draft
**Approach:** Incremental (A) with package refactor first

## Overview

Add enhanced analytics features to the HKJC racing analyzer: going-filtered form, jockey-horse cooperation stats, distance personal bests, odds movement tracking, steam/cold horse detection, speed maps, and a race day monitor with auto-refresh dashboard. Two phases: structural refactor, then feature additions. No external data sources needed — features 15-17 use existing DB data; feature 18 builds up via new race day monitor.

## Phase 1: Package Refactor

Reorganize flat file structure into packages. Zero behavior changes — move files, update imports, verify everything works identically.

### Target Structure

```
hkjc_racing_analyzer/
├── core/
│   ├── __init__.py
│   ├── analyzer.py          ← hkjc_racing_analyzer.py
│   ├── fetcher.py           ← hkjc_data_fetcher.py
│   └── predictor.py         ← ml_predictor.py
├─��� analytics/
│   ├── __init__.py
│   ├── draw_bias.py         ← DrawBiasAnalyzer from advanced_analytics.py
│   ├── pace.py              ← PaceAnalyzer
│   ├── sectionals.py        ← SectionalAnalyzer
│   ├── barrier_trials.py    ← BarrierTrialAnalyzer
│   ├── course_distance.py   �� CourseDistanceAnalyzer
│   ├── class_movement.py    ← ClassMovementAnalyzer
│   ├── gear_change.py       ← GearChangeAnalyzer
│   ├── weight.py            ← WeightAnalyzer
│   ├── going_pref.py        ← GoingPreferenceAnalyzer
│   ├── rolling_form.py      ← RollingFormAnalyzer
│   ├── beaten_margins.py    ← BeatenMarginsAnalyzer
│   └── jockey_change.py     ← JockeyChangeAnalyzer
├── data/
│   ├── __init__.py
│   ├── database.py          ← horse_database.py
│   ├── odds_tracker.py      ← odds_tracker.py
│   ├── odds_scraper.py      ← odds_scraper.py
│   └── scraper.py           ← scrape_history.py
├── monitor/
│   ├── __init__.py
│   └── race_day.py          ← NEW
├── dashboard/
│   ├── __init__.py
│   ├── generator.py         ← generate_predictions.py (prediction + orchestration logic)
│   └── template.py          ← HTML template string extracted from generate_predictions.py
├── betting/
│   ├── __init__.py
│   └── optimizer.py         ← betting_optimizer.py
├── models/                  ← unchanged (pkl files)
├── tracking_data/           ← unchanged (SQLite DBs)
├── results/                 ← unchanged (output JSON/HTML)
├── training_data/           ← unchanged
├── scripts/
│   ├── generate.py          ← CLI entry: python3 -m scripts.generate '2026/04/16'
│   ├── scrape.py            ← CLI entry: python3 -m scripts.scrape
│   ├── train.py             ← CLI entry: python3 -m scripts.train
│   ├── monitor.py           ← CLI entry: python3 -m scripts.monitor '2026/04/16'
│   └── fetch_results.py     ← CLI entry: results_fetcher.py
├── results_tracker.py       ← stays (standalone utility)
├── update_results.py        ← stays (standalone utility)
├── interactive_analyzer.py  ← stays (standalone utility)
├── racing_scheduler.py      ← stays (standalone utility)
├── mark6_analyzer.py        ��� stays (separate domain)
└── screenshot_capture.py    ← stays (standalone utility)
```

### Refactor Rules

- Each moved file keeps its classes/functions unchanged
- `advanced_analytics.py` splits into 12 individual files — one class per file, plus `parse_form_positions()` goes into `analytics/__init__.py`
- `generate_predictions.py` splits: `main()` + prediction orchestration → `dashboard/generator.py`, `generate_html()` → `dashboard/template.py`
- Scripts in `scripts/` are thin wrappers (~10 lines) that import and call main()
- Old file names at root become thin re-export shims temporarily (for any external references), removed after verification
- All existing CLI commands documented in CLAUDE.md continue to work

### Verification

After refactor, run:
1. `python3 -m scripts.generate '2026/04/15'` — must produce identical JSON + HTML
2. `python3 -m scripts.train` — must train model identically
3. `python3 -m scripts.scrape --date 2026/04/09` — must scrape correctly
4. Compare output JSON diff against pre-refactor output — should be identical

## Phase 2: Enhanced Analytics Features

### 2A. Going-Filtered Horse Form

**What:** Query a horse's past performance filtered by the current race's going condition.

**Where:** `data/database.py` — new method `get_horse_going_form(horse_name, going, before_date)`

**Query:**
```sql
SELECT COUNT(*) as runs,
       SUM(is_winner) as wins,
       SUM(is_top3) as places
FROM race_results
WHERE horse_name = ? AND going = ? AND race_date < ?
```

**ML Feature 15 (`going_form_rate`):** `wins / runs` if runs >= 2, else 0.5 fallback. Added to both `extract_features()` and `build_training_data_from_db()`.

**Dashboard:** Show "Good: 2W/5R" in horse detail view.

### 2B. Jockey-Horse Cooperation Stats

**What:** How a specific jockey performs on a specific horse.

**Where:** `data/database.py` — new method `get_jockey_horse_stats(jockey, horse_name, before_date)`

**Query:**
```sql
SELECT COUNT(*) as rides,
       SUM(is_winner) as wins,
       SUM(is_top3) as places,
       AVG(finish_position) as avg_pos
FROM race_results
WHERE jockey = ? AND horse_name = ? AND race_date < ?
```

**ML Feature 16 (`jockey_horse_coop`):** `(wins * 2 + places) / (rides * 3)` if rides >= 2, else 0.5 fallback.

**Dashboard:** Show "Z Purton on GOLDEN SIXTY: 6W 2P / 8 rides" in horse detail.

### 2C. Same-Distance Personal Bests

**What:** Fastest finish time per horse at the race distance, compared to track record.

**Where:** `data/database.py` — new methods:
- `get_distance_best_time(horse_name, distance, before_date)` — horse's PB
- `get_distance_record(venue, distance)` — track record (best time by any horse)

**Query (PB):**
```sql
SELECT MIN(finish_time) FROM race_results
WHERE horse_name = ? AND distance = ? AND race_date < ?
AND finish_time != '' AND finish_time IS NOT NULL
```

**ML Feature 17 (`distance_pb_ratio`):** Parse time to seconds, compute `track_record / horse_pb`. Closer to 1.0 = faster horse. Fallback 0.5 if no times.

**Dashboard:** Show "PB 1:09.2 (record 1:08.1)" in horse detail.

### 2D. Odds Movement ML Feature

**What:** Percentage change from opening to current odds as an ML signal.

**Where:** `core/predictor.py` — feature 18 in `extract_features()`

**Source:** `data/odds_tracker.py` — `OddsDatabase.get_opening_odds()` vs current odds from race card.

**ML Feature 18 (`odds_movement`):** `(opening - current) / opening`, clamped to [-1, 1], scaled to [0, 1]. Positive = shortening (money coming in). Fallback 0.5 when no opening odds recorded.

**Note:** This feature will be inactive (returning 0.5) until the race day monitor has collected opening odds for enough race days. Expected to become useful after ~10 meetings of monitoring.

### 2E. Steam Move / Drift / Cold Horse Detection

**What:** Enhance `OddsMovementAnalyzer` with specific detection for market signals.

**Where:** `data/odds_tracker.py`

**New methods on `OddsMovementAnalyzer`:**

- `detect_steam_move(opening, current, threshold=0.20)` — returns True + severity if odds shortened >= threshold. Levels: STEAM (>=30%), STRONG SUPPORT (>=20%).
- `detect_cold_horse(opening, current, rule_score, threshold=0.25)` — returns True if odds drifted >= threshold BUT horse has rule_score >= 65 (strong fundamentals). Potentially undervalued by market.

**Dashboard:** Badges on horse cards:
- Green "STEAM -18%" for shortening
- Red "DRIFT +22%" for lengthening
- Gold "COLD HORSE" for cold horse detection (drift + strong form)

### 2F. Speed Map Visualization

**What:** Visual predicted running positions for each race.

**Where:** `analytics/pace.py` — new method `generate_speed_map(horses, distance, venue)`

**Logic:**
1. For each horse, determine running style from DB history (or race card `running_style` field)
2. Combine with draw position to predict inside/outside lane
3. Group into predicted positions: Leaders (1-3), Stalkers (4-6), Midfield (7-9), Closers (10+)
4. Return structured data for dashboard rendering

**Dashboard:** New "Speed Map" section per race — simple track graphic with horse numbers placed in predicted positions. CSS-only, no external libs.

### 2G. Race Day Monitor

**What:** Standalone script that monitors odds before each race, detects market moves, and auto-regenerates the dashboard.

**Where:** `monitor/race_day.py`

**CLI:** `python3 -m scripts.monitor '2026/04/16'`

**Lifecycle:**
1. Fetch race card → extract race times for all races
2. Sleep until 6 minutes before Race 1 post time
3. Fetch odds (win + place) via `HKJCDataFetcher`
4. Record snapshot in `OddsDatabase`
5. Run steam/drift/cold horse detection
6. Regenerate dashboard HTML with fresh odds + market move badges
7. Sleep until 6 minutes before next race
8. Repeat for all races, then exit

**Dashboard auto-refresh:** When generated by the monitor, HTML includes `<meta http-equiv="refresh" content="60">` so the browser reloads automatically. Manual `generate_predictions.py` runs do NOT include this tag.

**Edge cases:**
- Race times not yet published: fall back to polling every 30 minutes until times appear
- Scratched horses: odds = 0, excluded from movement analysis
- Network errors: log warning, retry next cycle, don't crash
- After last race: clean exit with summary log

**Polling frequency:** Every 5 minutes during the 6-minute pre-race window (1-2 snapshots per race).

## ML Model Changes Summary

Current 14 features → 18 features:

| # | Feature | Source | Fallback |
|---|---------|--------|----------|
| 1-14 | Existing features | Unchanged | Unchanged |
| 15 | `going_form_rate` | DB: race_results (going + finish) | 0.5 |
| 16 | `jockey_horse_coop` | DB: race_results (jockey + horse) | 0.5 |
| 17 | `distance_pb_ratio` | DB: race_results (finish_time + distance) | 0.5 |
| 18 | `odds_movement` | OddsDB: odds_snapshots | 0.5 |

**Version bump:** Model version 1.0 → 2.0. Old pickle incompatible — must retrain after feature addition.

**Consistency rules (from CLAUDE.md audit):**
- All features use `race_date < ?` guard — no leakage
- `build_training_data_from_db()` computes features identically to `extract_features()`
- `generate_synthetic_training_data()` updated to match 18-feature schema
- Features with std < 0.01 in training data are logged as warnings

## What Stays Unchanged

- 26-factor rule engine scoring and weights
- 75/25 rule/ML blend ratio
- Bet recommendation logic (BANKER/WIN/QIN/PLACE/SKIP)
- Kelly Criterion value betting
- WhatsApp copy format
- Results fetcher and auto-learn flow
- Historical scraping
- All existing DB tables and schema (no migrations needed)
