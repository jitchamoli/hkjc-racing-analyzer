# HKJC Horse Racing Analyzer v5.0

Real-time horse racing analysis for the Hong Kong Jockey Club (HKJC). Fetches live race data, odds, and expert tips from racing.hkjc.com to generate betting recommendations using a 26-factor rule-based engine + XGBoost ML model trained on real historical race data, persistent horse database, and self-improving data pipeline.

## What's New in v5.0

- **26-factor analysis engine** - expanded from 17 to 26 scoring factors with dynamic weighting
- **Real jockey/trainer stats from DB** - replaced all hardcoded ratings with actual win/place rates computed from 4,155+ historical race results
- **ML model trained on real data** - XGBoost now trains on actual race results from horse_database (was 100% synthetic)
- **Historical data scraper** (`scrape_history.py`) - Playwright-based bulk scraper for past HKJC results, auto-ingests into DB
- **Self-improving pipeline** - each race day adds ~130 results; jockey/trainer stats, ML model, and going/margin data all improve automatically
- **9 new expert-driven factors** - Going preference, Rolling form, Beaten margins, Jockey change detection, Jockey/Trainer hot streaks, Field size impact, Career place rate, Age vs field
- **Research-backed feature design** - informed by Bill Benter methodology, CodeWorks Paris (1,618 features), gmalbert (field_size #1 at 17.41%), pbovard63 (field-relative features), Teddy Koker (value betting layer)

### Previous Versions
- **v4.0**: QIN default strategy, BANKER WIN system, market probability blending, 17 factors
- **v3.0**: 14-factor analytics, rail bias, pace map, sectionals, C&D form, odds movement
- **v2.0**: Tips Index integration, ML prediction, advanced analytics
- **v1.0**: Basic rule-based scoring

## How It Works

### Data Sources (from racing.hkjc.com)

| Source | What It Provides |
|--------|-----------------|
| Race Card page | Horse entries, jockeys, trainers, draw, weight, gear, ratings, last 6 runs |
| Tips Index page | Live odds, Initial Tips Index, Race Day Tips Index, top picks per race |
| Horse Profile page | Running positions, sectional times, class history, C&D record |
| Results page | Finishing positions, margins, odds for completed races |
| Horse Database (local) | 4,155+ race results, jockey/trainer season stats, going records, margins |

### Prediction System

The analyzer combines two systems (75/25 weighted):

**Rule-Based System (75%) - 26 Factors**

| Category | Factors | Total Weight |
|----------|---------|--------------|
| **Core** | Rating (vs field avg), Form, Tips Index, Market Odds, Jockey (DB stats), Trainer (DB stats), Draw + Rail Bias | ~56% |
| **Expert** | Going Preference, Rolling Form, Beaten Margins, Jockey Change, Jockey/Trainer Hot Streaks, Field Size, Career Place Rate, Age vs Field | ~37% |
| **Technical** | Pace Map, C&D Form, Odds Movement, Class Movement, Gear Changes, Weight Analysis, Sectional Speed, Freshness, Rating Trend, Season Stakes | ~7% |

Weights are dynamic - factors with missing data are excluded and remaining weights are re-normalized to sum to 1.0.

**ML-Based System (25%)**
- XGBoost classifier with 18 features trained on **4,155+ real race results** from horse_database
- Auto-retrains on real data when DB has 50+ results; falls back to calibrated synthetic data otherwise
- Top features: odds_implied_prob (14.0%), recent_form (6.9%), cd_form (6.1%), jockey_win_rate (6.0%), weight_carried (4.9%)
- Model improves automatically as more race days are scraped

### Key Improvements Over Hardcoded Approach

| What | Old (v4.0) | New (v5.0) |
|------|-----------|-----------|
| Jockey scores | 26 hardcoded entries (Purton=92) | Real DB stats (Purton=15.6% from 109 rides) |
| Trainer scores | 23 hardcoded entries (Size=92) | Real DB stats (Size=10.2% from 88 runners) |
| ML training data | 5,000 synthetic random samples | 4,155+ real race results |
| Rating scoring | Absolute (rating/100) | Relative to field average (pbovard63) |
| Jockey form | Static tier | Last 10 rides win/place rate, hot streak detection |
| Trainer form | Static tier | Last 10 runners win/place rate |
| Going | Not scored | Horse's record on today's track condition |
| Margins | Not scored | Last-start margin, close-finish frequency |

### Bet Type Logic

| Bet Type | When |
|----------|------|
| **BANKER WIN** | 7+ of 9 conviction signals fire - highest confidence |
| **WIN** | Very clear market standout (35%+ implied) or extreme model confidence |
| **QIN** (Quinella) | **Default** - rank 1 + rank 2 together |
| **PLACE** | Strong horse at short odds, safety play |
| **SKIP** | No positive EV or insufficient data |

### BANKER WIN System (9 Conviction Signals)

| Signal | Threshold |
|--------|-----------|
| High score | Rule score >= 68 |
| Rating surge | Rating change +5 or more |
| Peak fitness | 14-35 days since last run |
| Winning form | Form score >= 60 |
| Pace advantage | Pace score >= 70 |
| Ideal draw | Draw score >= 70 |
| Top jockey | Jockey score >= 82 |
| Proven earner | Season stakes >= $2M |
| Market backed | Implied probability >= 10% |

## Project Structure

```
hkjc_racing_analyzer/
  core/                      # Core modules
    fetcher.py               #   Real-time data layer (HKJC API + parser)
    analyzer.py              #   Main analyzer — 26-factor engine + ML blend
    scorer.py                #   Score combination + probability calibration
    recommender.py           #   Bet recommendation engine (BANKER/WIN/QIN/PLACE/SKIP)
    predictor.py             #   ML predictor shim (delegates to ml/)
  ml/                        # ML pipeline
    feature_schema.py        #   Single source of truth for 18 features
    features.py              #   Feature extraction for prediction
    model.py                 #   XGBoost model — train, predict, save, load
    training.py              #   Training data builders (DB + synthetic)
    experiments.py           #   JSON-based experiment tracking
  analytics/                 # 12 specialized analyzers (draw bias, pace, sectionals,
                             #   barrier trial, C&D, class, gear, weight, going,
                             #   rolling form, beaten margins, jockey change)
  data/                      # Data layer
    database.py              #   SQLite horse/jockey/trainer database
    validation.py            #   Schema validation for parsed HKJC data
  dashboard/                 # Dashboard
    generator.py             #   Prediction JSON + HTML dashboard generator
    results_fetcher.py       #   Fetch results + scorecard + auto-learn
    template.py              #   HTML template
  monitor/                   # Race day auto-polling monitor
  betting/                   # Kelly Criterion optimizer
  scripts/                   # CLI entry points (generate, scrape, train, fetch_results)
  tests/                     # pytest suite — 77 tests (features, scoring, validation,
                             #   data integrity, data accuracy)
  config.py                  # Centralized constants and hyperparameters
  models/                    # Trained ML models (auto-generated)
  results/                   # HTML dashboards and prediction JSON
  tracking_data/             # SQLite databases (horse_history.db with 4,155+ results)
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### Build Historical Database (recommended first run)

```bash
# Scrape past 10 meetings to train ML model
python3 scrape_history.py

# Or scrape a specific date
python3 scrape_history.py --date 2026/04/06

# Import full-season jockey/trainer rankings
python3 scrape_history.py --rankings
```

### Generate Dashboard

```bash
python3 generate_predictions.py '2026/04/19'
# Opens results/racing_dashboard.html with betting recommendations
```

### Race Day Monitor (auto-refresh odds)

```bash
python3 -m scripts.monitor '2026/04/19'
# Auto-polls odds 6min before each race, regenerates dashboard
```

### Fetch Results & Auto-Learn

```bash
python3 results_fetcher.py '2026/04/19' --auto-learn
# Scrapes results, scores predictions, ingests into DB, retrains ML
```

### Run Tests

```bash
python3 -m pytest tests/ -v
```

## Performance (April 12, 2026 - Sha Tin)

| Metric | Result |
|--------|--------|
| #1 Pick Won | 3/11 (27%) |
| #1 or #2 Won | 8/11 (73%) |
| #1 or #2 Top 3 | 14/22 entries placed |
| QIN exact hits | 3/11 |
| Random baseline | ~8% (1/avg field of 12) |

## Requirements

- Python 3.10+
- Internet connection (fetches live data from racing.hkjc.com)
- Playwright + Chromium (for historical scraping and JS-rendered pages)

## Key Dependencies

- `xgboost` - ML prediction engine
- `scikit-learn` - Model training and evaluation
- `numpy` - Numerical computations
- `playwright` - Browser automation for JS-rendered pages
- `beautifulsoup4` - HTML parsing
- `requests` - HTTP requests

## Disclaimer

For entertainment and educational purposes only. Not financial advice. Past performance does not guarantee future results. Gamble responsibly.
