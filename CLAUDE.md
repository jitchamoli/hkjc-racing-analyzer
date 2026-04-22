# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Generate predictions + dashboard for a race date (fetches live data from HKJC)
python3 generate_predictions.py '2026/04/15'

# Output: results/race_data_YYYYMMDD.json + results/racing_dashboard.html (always-latest) + results/racing_dashboard_YYYYMMDD.html (archive)

# Scrape historical results into database (Playwright required)
python3 scrape_history.py                    # Last 10 meeting dates
python3 scrape_history.py --date 2026/04/09  # Specific date
python3 scrape_history.py --rankings         # Scrape full-season jockey/trainer stats from HKJC
python3 scrape_history.py --days 90          # Look back 90 days (default: 60)

# Fetch results after races, score predictions, regenerate dashboard with results overlay
python3 results_fetcher.py '2026/04/15'
python3 results_fetcher.py '2026/04/15' --auto-learn  # Also ingest into DB + retrain ML

# Retrain ML model on latest data from horse_database
python3 train_ml_model.py

# Ingest training_data/historical_races.json into horse_database
python3 -c "from horse_database import ingest_from_training_data; ingest_from_training_data()"

# Race day monitor (auto-polls odds 6min before each race, regenerates dashboard)
python3 -m scripts.monitor '2026/04/16'

# New CLI entry points (equivalent to root scripts)
python3 -m scripts.generate '2026/04/16'
python3 -m scripts.scrape --date 2026/04/09
python3 -m scripts.train
python3 -m scripts.fetch_results '2026/04/16'
```

## Architecture

### Package Structure

```
core/       — fetcher, analyzer, scorer, recommender, ML predictor
ml/         — feature schema, feature extraction, model, training, experiments
analytics/  — 12 specialized analyzers (draw bias, pace, sectionals, etc.)
data/       — database, odds tracker, scraper, validation
monitor/    — race day monitor
dashboard/  — prediction generator, HTML template
betting/    — Kelly Criterion optimizer
scripts/    — CLI entry points
tests/      — pytest suite (77 tests)
```

Root-level shims exist for backwards compatibility (e.g., `hkjc_data_fetcher.py` re-exports from `core/fetcher.py`).

### Data Pipeline

```
core/fetcher.py  →  core/analyzer.py  →  dashboard/generator.py  →  results/
(fetch from HKJC)   (26-factor + ML)     (JSON + HTML dashboard)
        ↑                   ↑
        |                   |
data/database.py ←── dashboard/results_fetcher.py (post-race: ingest results, update stats)
(SQLite: tracking_data/horse_history.db)
        ↓
scripts/train.py → models/hkjc_racing_model.pkl
```

### Scoring System

**CombinedRaceAnalyzer** (`hkjc_racing_analyzer.py`) merges two systems:
- **Rule engine (60%)**: 24 factors with dynamic weighting — factors with missing data are excluded and remaining weights renormalize to 1.0. The rule_score (0-100) is converted to a probability via a **base-rate-anchored sigmoid**: `sigmoid(0.06 × (score - 50) + logit(1/field_size))`, so an "average" horse (score=50) outputs ~1/field_size instead of 50%. Note: `odds` and `odds_movement` were removed from the rule engine to prevent triple-counting market signal (market info enters via the dynamic market blend in calibration). Approximate grouping: Core (rating, form, jockey, trainer, draw, tips), Expert (going, rolling form, margins, field size, career stats), Technical (pace, sectionals, C&D, class movement, gear, weight, freshness). Actual percentages vary per race due to renormalization.
- **ML model (40%)**: XGBoost classifier with 23 features (`ml/model.py`, schema in `ml/feature_schema.py`). Includes a dedicated **place model** (is_top3) for place bet recommendations and quinella pricing. Both models use Platt scaling (LogisticRegression) for probability calibration, **weighted by the same recency weights used for XGBoost training** so the calibrator sees the same prior as the model. Trained on 4155+ real race results from DB via `build_training_data_from_db()`; falls back to synthetic data if <50 results. The `scripts/train.py` pipeline prefers DB data over scraped training_data when ≥200 samples exist.

The 13 specialized analyzers live in `analytics/` (draw bias, pace, sectionals, barrier trials, C&D, class movement, gear, weight, going preference, rolling form, beaten margins, jockey change, **speed figures**).

### Supplementary Scripts

- `odds_tracker.py` / `odds_scraper.py` — SQLite odds history, movement tracking, multi-method odds fetching.
- `betting_optimizer.py` — Kelly Criterion value bet optimization.
- `interactive_analyzer.py` — Interactive mode with manual odds input.
- `racing_scheduler.py` — Automated daily race analysis scheduler.
- `results_tracker.py` / `update_results.py` — Track predictions vs actuals, manual result entry.
- `mark6_analyzer.py` — HKJC Mark Six lottery analyzer (separate from racing).

### Key Data Structures

- **HorseEntry / RaceCard** (`hkjc_data_fetcher.py`): Raw data from HKJC. HorseEntry has ~30 fields including tips index, running style, sectionals, weight changes, import category.
- **HorseAnalysis** (`hkjc_racing_analyzer.py`): Analysis output per horse — rule_score (0-100), combined_probability, bet_recommendation (BANKER/WIN/QIN/PLACE/SKIP), reasoning list.

### Database

SQLite at `tracking_data/horse_history.db`. Four tables:
- **race_results** (primary data source): one row per horse per race, keyed by (race_date, race_number, horse_name). Contains finish_position, is_winner, is_top3, margin, odds, horse_weight, running_position, finish_time, days_since_last_run.
- **horses**: master record per horse. Note: `get_horse_record()` computes aggregates from race_results directly (not from this table's stale counters).
- **jockey_stats** / **trainer_stats**: season-level win/place rates. Populated two ways: `rebuild_stats()` computes from local race_results; `scrape_history.py --rankings` fetches full-season stats from HKJC (more accurate, ~80 meets vs our ~10).

**Data integrity**: `ingest_race_day()` calls `rebuild_stats()` after every ingestion, which overwrites jockey/trainer stats with local-only data. If full-season HKJC rankings were previously imported, re-run `python3 scrape_history.py --rankings` after ingestion to restore them. The `horses` table upsert derives `total_starts`/`total_wins`/`total_places` from `race_results` via subquery — safe for re-ingestion (idempotent, no double-counting).

**Auto-computed fields**: `days_since_last_run` is backfilled from race_results on ingestion (uses `julianday(REPLACE(date,'/','-'))` since dates are YYYY/MM/DD format). `rating_change` and `age` come from race cards (not results pages) so are only populated for races where predictions were generated.

### Data Fetching (hkjc_data_fetcher.py)

Three methods tried in order:
1. **requests + BeautifulSoup** (fast, reliable for race cards)
2. **Playwright** (slower, gets live odds but often times out)
3. Tips data fetched separately for expert ratings

Place odds are fetched via `_apply_odds_via_requests()` using `fetch_odds_html(date, race_no, 'PLA')` — but PLA odds are only available near race time (returns None pre-race).

**Venue detection is critical**: `HKJCApiClient.detect_venue(race_date)` auto-detects ST (Sha Tin) vs HV (Happy Valley). Using wrong venue returns empty HTML with no tables. Wednesday meetings are typically Happy Valley; weekends are Sha Tin.

### Dashboard

`generate_predictions.py` generates a self-contained HTML dashboard with embedded race data as JSON. The venue is detected dynamically from race data. `results/racing_dashboard.html` is always overwritten with the latest run (date-stamped archive also saved). Dashboard includes grade legend (A+/A/B+/B/B-), WhatsApp copy with full betting strategy, and race detail views. All scraped data (horse names, jockeys, trainers, reasoning) is HTML-escaped via a JS `esc()` helper before `innerHTML` injection (XSS protection). The JSON embedding sanitizes `</script>` tags to prevent script block injection.

### Race Day Monitor

`python3 -m scripts.monitor '2026/04/16'` — polls odds 6 minutes before each race (every 5 min), snapshots to odds_history.db, detects steam/drift/cold horses, regenerates dashboard with auto-refresh (60s). Dashboard at `results/racing_dashboard.html` reloads automatically in browser.

### Enhanced Dashboard Fields

JSON output now includes per-horse: `going_form`, `jockey_horse_stats`, `distance_pb`, `distance_record`, `odds_movement`, `steam_move`, `cold_horse`. Per-race: `speed_map`. Dashboard renders these as enhanced stats grid, market move badges (steam/drift/cold), and a visual speed map.

### Quinella Dividend & EV Fields (Apr 22 addition)

Per-race JSON additions from `betting/dividends.py` (Harville independence model, HK 17.5% takeout):

- `strategy_confidence`: HIGH / MEDIUM / LOW / NONE based on top_2_share and top_3_share (model's conviction on pair identification)
- `top_2_share`, `top_3_share`: sum of calibrated_prob across top model picks
- `qin_top_pair_dividend`: estimated quinella dividend for rank-1 + rank-2 pair (per $1 stake)
- `qin_bet_ev`: dict with `ev_pct`, `avg_dividend`, `hit_prob`, `total_stake`, `expected_profit` for the actual bet configuration (straight QIN or box-3). Uses model probs for P(hit), market probs for dividend.
- `is_best_ev` / `is_worst_ev`: per-race flag for the highest/lowest EV among active QIN races (only set when ≥2 races have QIN bets with EV spread ≥5pp)

**Interpretation of EV%**: Usually negative (typically -30% to -60%) because our model closely tracks market, so Harville dividend ≈ (1-takeout)/model_prob → EV ≈ -takeout × n_combos. The RELATIVE ranking across races (BEST EV vs LOW EV) is what identifies allocation priority. Absolute positive EV would indicate strong model-market divergence — rare in current setup.

Dashboard renders:
- Strategy caveat banner at top ("validated on 71 held-out races, 95% CI 9-25%")
- Confidence badge per race card next to race number
- EV% + dividend estimate in race card footer
- BEST EV / LOW EV highlight ring + badge
- Detailed EV breakdown (hit prob, avg dividend, EV%) in race-detail "What To Bet" panel

### Historical Scraping (scrape_history.py)

Uses Playwright to scrape HKJC results pages (JS-rendered). Extracts from results table columns: position(0), horse_no(1), name(2), jockey(3), trainer(4), weight(5), horse_weight(6), draw(7), LBW/margin(8), running_position(9), finish_time(10), win_odds(11). The `--rankings` flag scrapes the HKJC Jockey/Trainer Ranking pages for full-season stats.

## Codebase Conventions

- Code is organized into packages (`core/`, `analytics/`, `data/`, `monitor/`, `dashboard/`, `betting/`, `scripts/`). Root-level shims provide backwards compatibility.
- Async/await used in data fetching pipeline (`HKJCDataFetcher.get_race_card` is async).
- Race dates use `YYYY/MM/DD` format in HKJC URLs and data, but `YYYYMMDD` in filenames. SQLite `julianday()` requires `YYYY-MM-DD` — always use `REPLACE(date,'/','-')`.
- The HTML parser (`HKJCParser`) expects specific HKJC table layouts (27 cells per row with documented column indices). HKJC site changes can break parsing — check `_parse_horse_table()` if horses come back empty.
- ML model saved as pickle dict with keys: `model`, `place_model`, `calibrator`, `version`, `n_features`, `feature_importance`, `trained_date`. A `.sha256` sidecar file is required for loading — model refuses to load without valid checksum.
- Dashboard probabilities are stored as decimals (0.178) in JSON but displayed as percentages (17.8%) — multiply by 100 in JavaScript display code.
- `HKJCParser._safe_int()` preserves minus signs (regex `[^\d\-]`) — important for `rating_change` and `weight_change` which can be negative.

## ML Model — Lessons Learned (2026-04-18 Audit)

### Critical Rules for ML Features

1. **No data leakage**: Every feature must use ONLY data available BEFORE the race. Never use finish_position, finish_time, running_position, or margin from the CURRENT race. For horse history features (C&D form, pace, speed), use strictly `prior_date < current_date`. `extract_features()` now accepts `race_date` via `horse_data` and filters DB queries with `AND race_date < ?`. Track records in training are also date-aware (cumulative best-time lookup, not global).

2. **Train/predict consistency**: Features in `build_training_data_from_db()` MUST match `extract_features()` exactly — same computation, same order, same fallback values. `generate_synthetic_training_data()` must also match the same 23-feature schema. The DB (race_results) has different fields than race cards: rating, last_6, gear are ~97% missing in DB but always present on race cards. Use DB-available proxies (prior finishes, prior win rate) instead.

3. **No constant features**: If a feature has std < 0.01 in training data, it's wasted. Common cause: field is mostly null and falls back to 0.5. Check with `np.std()` after building features. Feature correlation matrix is logged after each training run — pairs with |r| > 0.7 are flagged.

4. **value_score must use ML blend**: `value_score` in HorseAnalysis feeds `combined_score` in the JSON. It must be `combined_prob * 100` (the 60/40 rule+ML blend), NOT raw `rule_score`.

5. **Probability sync after rank swaps**: When `_assign_bet_recommendations` swaps horse positions (market-favourite promotion or BANKER override), `combined_probability` and `expected_return` on ALL `HorseAnalysis` objects must be re-written from the swapped `calibrated_probs` list.

6. **ML fallback probability**: `predict()` accepts `field_size` param (default 12). When model is untrained, returns `1/field_size` — not `1/len(features)` which is always 23.

7. **Feature count guard**: Model pickle stores `n_features`. On load, it's checked against the current `NUM_FEATURES` — mismatched models refuse to load with a clear error message.

8. **Model integrity**: `.sha256` sidecar is required for loading. Missing checksum = hard refusal. `train_ml_model.py` backs up the old model before training and restores on failure.

### Current 23 ML Features (all prior-race only)

Features 1-9: `prior_win_rate, recent_form, draw_advantage, jockey_win_rate, trainer_win_rate, field_size, field_rel_draw, prior_pace, adjusted_speed` (Beyer-style par-adjusted, weight-corrected — replaces raw prior_speed)

Features 10-17: `cd_form_prior, class_level, freshness, going_performance` (merged), `weight_vs_field` (relative to field median), `jockey_horse_coop, distance_pb_ratio, field_rel_speed` (percentile within prior fields)

Features 18-23:
- `last_start_margin`: how far behind winner last race (normalized)
- `class_change`: current class vs last-race class (drop = positive signal)
- `days_since_last_win`: recency of last win (normalized decay)
- `jockey_trainer_combo`: geometric mean of jockey × trainer win rates (interaction)
- `horse_weight_change`: body weight % change from last 3 races (fitness signal)
- `speed_consistency`: std of adjusted speed figures — low = reliable horse (important for exotic bets)

Features removed during audit: `odds_implied_prob`, `venue_win_rate` (r=0.984), `weight_carried_norm` (r=0.877), `draw_field_interact` (r=0.934), `prior_avg_position`, `prior_speed` (replaced by adjusted_speed), `late_money_signal` (constant 0.5 in training — DB lacks opening odds — removed to eliminate train/predict distribution mismatch; odds signal enters via market blend instead).

### Training Pipeline

- **60/20/20 temporal split**: Train on first 60%, validate (early stopping + Platt calibration) on next 20%, evaluate on final 20%. Test metrics are computed through the calibrator but the calibrator was never fitted on test data — metrics are unbiased.
- **Recency weighting**: Sample weights decay from 1.0 (newest) to 0.3 (oldest). Recent races get 3.3x weight.
- **Platt scaling**: LogisticRegression fitted on validation set predictions → actual outcomes. Recalibrates XGBoost's overconfident probabilities. Fitted on validation (not test) so test metrics remain unbiased.
- **Place model**: Separate XGBoost trained on `is_top3` labels alongside the win model. Used for place bet probability and quinella placer-boost pricing.
- **Precomputed lookups**: Par times (per distance/going/class), track records, field sizes, field median weights, and per-race finish times are all precomputed before the training loop.

### Speed Figures (`analytics/speed_figures.py`)

Beyer/Timeform-style adjusted speed figures. Computes par times per (distance, going, class) bucket from DB, then measures each horse's performance relative to par with symmetric weight correction (Benter's 0.1s per lb deviation from 120 baseline — heavier horses are credited, lighter debited). Par times in training are date-filtered per month to prevent leakage. Replaces raw `prior_speed` which conflated fast sprinters with slow stayers.

### Exotic Bet Calculator (`betting/exotic_calculator.py`)

Prices all exotic bet combinations using conditional probability elimination:
- **Exacta**: P(A wins) × P(B wins | A removed)
- **Quinella**: Exacta(A,B) + Exacta(B,A), enhanced with place-model boost (consistent placers +15%), pace pairing boost (leader+closer +10%), draw clustering boost (adjacent good draws +8%)
- **Trifecta**: extends to 3-horse permutations (top 8 candidates)
- **First4**: extends to 4-horse permutations (top 6 candidates)
- **Value detection**: compares model probability vs market-implied, flags combinations with 10%+ edge

### Dynamic Market/Model Blend

Market blend weights adjust by race characteristics (HK bettors are sharp — larger/shorter races have sharper pools):
- Large fields (10+) + sprints (≤1400m): 55% market / 45% model
- Small fields (<8) or routes (≥1800m): 35% market / 65% model
- Default: 40% market / 60% model
- Quinella pool: lean 55-65% ML (less sharp money, public ignores consistent placers)

### DB Data Gaps

- `rating`, `last_6`, `gear`, `age`, `rating_change` — only populated for dates where predictions were generated before results ingestion (~3% of rows). The scraper (`scrape_history.py`) doesn't capture these from results pages.
- `running_position` has two formats: space-separated (`"6 5 6"`) and concatenated (`"665"`). Always handle both when parsing.
- `horse_weight` is in lbs (900-1350 range), `weight` is carried weight assignment (113-133 lbs).
- `class_type` has 96% coverage in DB. Missing rows default to class 3 for class_change calculation.

### Betting Strategy — Quinella-First (pivoted Apr 22 2026)

**Validated by out-of-sample backtest** (`scripts/backtest.py`, 71 held-out races):
- WIN pool has NO edge: model top-pick = market favourite 98.6% of time → -37% ROI (identical to betting market fav every race — both lose to takeout)
- Box-3 quinella: **33.8% hit** (3.7× random) — the real edge
- Rank-1 finishes top-2: **45.1%** (banker-leg strength)

**Recommendation logic** (`core/recommender.py`):
- Default: `BOX Q` — all top-3 model picks flagged `QIN`, user boxes them (3 combinations)
- `TRIO` when top 3 are tightly bunched (gap_1_2 < 0.02 AND gap_2_3 < 0.02)
- `WIN` rare, only when rank-1 has BANKER signal AND market consensus (≥35% implied) AND clear top-2 gap (>0.08)
- `SKIP` if no live odds, field < 3, OR top_3_share < 0.30 (conviction gate; uniform 12-horse = 0.25)

**Backtest reproduction**:
```
python3 -m scripts.backtest              # Default
python3 -m scripts.backtest --sweep      # Grid over temperature × model_w
```

- Races without live odds (pre-race or post-race) will show all SKIP — this is correct behaviour.
- `results_fetcher.py --auto-learn` scrapes live results from HKJC via `scrape_meeting_results()` and ingests into DB (no longer reads from `training_data/historical_races.json`). Retraining timeout is caught gracefully; run `python3 train_ml_model.py` separately if it times out.

### Probability Pipeline

```
rule_score (0-100) → sigmoid → ×0.75
ml_prob (Platt-calibrated) → ×0.25
combined_prob = rule + ml (uncalibrated)
→ softmax (temp=1.5, mild normalization)
→ dynamic blend: model × (45-65%) + market × (35-55%)
  (weights depend on field_size + distance)
→ normalize to sum=1.0
→ expected_return = odds × prob - 1
→ exotic pricing: conditional elimination for exacta/trifecta/quinella/first4
```

Market odds enter ONLY through the dynamic market blend — removed from rule engine to prevent triple-counting. Blend weights are heavier on market for large sprint fields (sharp pool), heavier on model for small/long fields and exotics (thinner pool).

### Accuracy Report

`python3 -m scripts.accuracy --all` includes Brier score, Brier Skill Score, and a 5-bin calibration curve (predicted vs actual win rates). Use this to monitor calibration quality after retraining.
