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
```

## Architecture

### Data Pipeline

```
hkjc_data_fetcher.py  →  hkjc_racing_analyzer.py  →  generate_predictions.py  →  results/
(fetch from HKJC)        (26-factor + ML scoring)     (JSON + HTML dashboard)
        ↑                         ↑
        |                         |
horse_database.py ←── results_fetcher.py (post-race: ingest results, update stats)
(SQLite: tracking_data/horse_history.db)
        ↓
train_ml_model.py → models/hkjc_racing_model.pkl
```

### Scoring System

**CombinedRaceAnalyzer** (`hkjc_racing_analyzer.py`) merges two systems:
- **Rule engine (75%)**: 26 factors with dynamic weighting — factors with missing data are excluded and remaining weights renormalize to 1.0. Approximate grouping: Core (rating, form, odds, jockey, trainer, draw, tips), Expert (going, rolling form, margins, field size, career stats), Technical (pace, sectionals, C&D, class movement, gear, weight, freshness). Actual percentages vary per race due to renormalization.
- **ML model (25%)**: XGBoost classifier with 14 features (`ml_predictor.py`). Trained on 4155+ real race results from DB via `build_training_data_from_db()`; falls back to synthetic data if <50 results. The `train_ml_model.py` pipeline prefers DB data over scraped training_data when ≥200 samples exist.

The 12 specialized analyzers live in `advanced_analytics.py` (draw bias, pace, sectionals, barrier trials, C&D, class movement, gear, weight, going preference, rolling form, beaten margins, jockey change).

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

**Data integrity**: `ingest_race_day()` calls `rebuild_stats()` after every ingestion, which overwrites jockey/trainer stats with local-only data. If full-season HKJC rankings were previously imported, re-run `python3 scrape_history.py --rankings` after ingestion to restore them.

**Auto-computed fields**: `days_since_last_run` is backfilled from race_results on ingestion (uses `julianday(REPLACE(date,'/','-'))` since dates are YYYY/MM/DD format). `rating_change` and `age` come from race cards (not results pages) so are only populated for races where predictions were generated.

### Data Fetching (hkjc_data_fetcher.py)

Three methods tried in order:
1. **requests + BeautifulSoup** (fast, reliable for race cards)
2. **Playwright** (slower, gets live odds but often times out)
3. Tips data fetched separately for expert ratings

Place odds are fetched via `_apply_odds_via_requests()` using `fetch_odds_html(date, race_no, 'PLA')` — but PLA odds are only available near race time (returns None pre-race).

**Venue detection is critical**: `HKJCApiClient.detect_venue(race_date)` auto-detects ST (Sha Tin) vs HV (Happy Valley). Using wrong venue returns empty HTML with no tables. Wednesday meetings are typically Happy Valley; weekends are Sha Tin.

### Dashboard

`generate_predictions.py` generates a self-contained HTML dashboard with embedded race data as JSON. The venue is detected dynamically from race data. `results/racing_dashboard.html` is always overwritten with the latest run (date-stamped archive also saved). Dashboard includes grade legend (A+/A/B+/B/B-), WhatsApp copy with full betting strategy, and race detail views.

### Historical Scraping (scrape_history.py)

Uses Playwright to scrape HKJC results pages (JS-rendered). Extracts from results table columns: position(0), horse_no(1), name(2), jockey(3), trainer(4), weight(5), horse_weight(6), draw(7), LBW/margin(8), running_position(9), finish_time(10), win_odds(11). The `--rankings` flag scrapes the HKJC Jockey/Trainer Ranking pages for full-season stats.

## Codebase Conventions

- All Python files are standalone scripts in the project root (no package structure).
- Async/await used in data fetching pipeline (`HKJCDataFetcher.get_race_card` is async).
- Race dates use `YYYY/MM/DD` format in HKJC URLs and data, but `YYYYMMDD` in filenames. SQLite `julianday()` requires `YYYY-MM-DD` — always use `REPLACE(date,'/','-')`.
- The HTML parser (`HKJCParser`) expects specific HKJC table layouts (27 cells per row with documented column indices). HKJC site changes can break parsing — check `_parse_horse_table()` if horses come back empty.
- ML model saved as pickle dict with keys: `model`, `version`, `feature_importance`, `trained_date`.
- Dashboard probabilities are stored as decimals (0.178) in JSON but displayed as percentages (17.8%) — multiply by 100 in JavaScript display code.

## ML Model — Lessons Learned (2026-04-15 Audit)

### Critical Rules for ML Features

1. **No data leakage**: Every feature must use ONLY data available BEFORE the race. Never use finish_position, finish_time, running_position, or margin from the CURRENT race. For horse history features (C&D form, pace, speed), use strictly `prior_date < current_date`.

2. **Train/predict consistency**: Features in `build_training_data_from_db()` MUST match `extract_features()` exactly — same computation, same order, same fallback values. The DB (race_results) has different fields than race cards: rating, last_6, gear are ~97% missing in DB but always present on race cards. Use DB-available proxies (prior finishes, prior win rate) instead.

3. **No constant features**: If a feature has std < 0.01 in training data, it's wasted. Common cause: field is mostly null and falls back to 0.5. Check with `np.std()` after building features.

4. **value_score must use ML blend**: `value_score` in HorseAnalysis feeds `combined_score` in the JSON. It must be `combined_prob * 100` (the 75/25 rule+ML blend), NOT raw `rule_score`.

### Current 14 ML Features (all prior-race only)

`prior_win_rate, recent_form, draw_advantage, jockey_win_rate, trainer_win_rate, field_size, field_rel_draw, prior_pace, prior_speed, cd_form_prior, odds_implied_prob, class_level, freshness, going_pref`

Top signal: `odds_implied_prob` (17.9%) — market odds are the single best predictor.

### DB Data Gaps

- `rating`, `last_6`, `gear`, `age`, `rating_change` — only populated for dates where predictions were generated before results ingestion (~3% of rows). The scraper (`scrape_history.py`) doesn't capture these from results pages.
- `running_position` has two formats: space-separated (`"6 5 6"`) and concatenated (`"665"`). Always handle both when parsing.
- `horse_weight` is in lbs (900-1350 range), `weight` is carried weight assignment (113-133 lbs).

### Betting Strategy

- Race-skip logic: if top pick has no edge over market odds AND expected return < 5%, skip entire race.
- Per-bet EV filter: each bet must have expected return ≥ 5% (except rank 1-2 in QIN which work as a pair).
- Races without live odds (pre-race or post-race) will show all SKIP — this is correct behaviour.
- `results_fetcher.py --auto-learn` retraining timeout is caught gracefully; run `python3 train_ml_model.py` separately if it times out.
