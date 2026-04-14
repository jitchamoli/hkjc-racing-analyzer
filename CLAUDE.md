# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Generate predictions + dashboard for a race date (fetches live data from HKJC)
python3 generate_predictions.py '2026/04/15'

# Output: results/race_data_YYYYMMDD.json + results/racing_dashboard.html (always-latest) + results/racing_dashboard_YYYYMMDD.html (archive)

# Update database with actual results after races complete
python3 update_results.py

# Retrain ML model on latest data from horse_database
python3 train_ml_model.py

# Scrape historical results into database (Playwright required)
python3 scrape_history.py

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
- **Rule engine (75%)**: 26 factors with dynamic weighting — factors with missing data are excluded and remaining weights renormalize to 1.0. Factors grouped into Core (~56%), Expert (~37%), Technical (~7%).
- **ML model (25%)**: XGBoost classifier with 14 features (`ml_predictor.py`). Trained on real race results from DB; falls back to synthetic data if <50 results.

The 12 specialized analyzers live in `advanced_analytics.py` (draw bias, pace, sectionals, barrier trials, C&D, class movement, gear, weight, going preference, rolling form, beaten margins, jockey change).

### Key Data Structures

- **HorseEntry / RaceCard** (`hkjc_data_fetcher.py`): Raw data from HKJC. HorseEntry has ~30 fields including tips index, running style, sectionals, weight changes, import category.
- **HorseAnalysis** (`hkjc_racing_analyzer.py`): Analysis output per horse — rule_score (0-100), combined_probability, bet_recommendation (BANKER/WIN/QIN/PLACE/SKIP), reasoning list.

### Database

SQLite at `tracking_data/horse_history.db`. Four tables:
- **race_results** (primary data source): one row per horse per race, keyed by (race_date, race_number, horse_name). Contains finish_position, is_winner, is_top3, margin, odds.
- **horses**: master record per horse. Note: `get_horse_record()` computes aggregates from race_results directly (not from this table's stale counters).
- **jockey_stats** / **trainer_stats**: season-level win/place rates, auto-updated on ingestion.

### Data Fetching (hkjc_data_fetcher.py)

Three methods tried in order:
1. **requests + BeautifulSoup** (fast, reliable for race cards)
2. **Playwright** (slower, gets live odds but often times out)
3. Tips data fetched separately for expert ratings

**Venue detection is critical**: `HKJCApiClient.detect_venue(race_date)` auto-detects ST (Sha Tin) vs HV (Happy Valley). Using wrong venue returns empty HTML with no tables. Wednesday meetings are typically Happy Valley; weekends are Sha Tin.

### Dashboard

`generate_predictions.py` generates a self-contained HTML dashboard with embedded race data as JSON. The venue is detected dynamically from race data. `results/racing_dashboard.html` is always overwritten with the latest run.

## Codebase Conventions

- All Python files are standalone scripts in the project root (no package structure).
- Async/await used in data fetching pipeline (`HKJCDataFetcher.get_race_card` is async).
- Race dates use `YYYY/MM/DD` format in HKJC URLs and data, but `YYYYMMDD` in filenames.
- The HTML parser (`HKJCParser`) expects specific HKJC table layouts (27 cells per row with documented column indices). HKJC site changes can break parsing — check `_parse_horse_table()` if horses come back empty.
- ML model saved as pickle dict with keys: `model`, `version`, `feature_importance`, `trained_date`.
