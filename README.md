# HKJC Horse Racing Analyzer v5.1

Real-time horse racing analysis for the Hong Kong Jockey Club (HKJC). Fetches live race data, odds, and expert tips from racing.hkjc.com to generate betting recommendations using a 24-factor rule-based engine + XGBoost ML model (22 features, Platt-calibrated) trained on real historical race data, persistent horse database, and self-improving data pipeline.

## What's New in v5.1

- **22-feature ML schema** - redesigned from data science audit. Removed redundant features (venue_win_rate r=0.984, draw_field_interact r=0.934, weight_carried_norm r=0.877). Added field-relative speed, class change, jockey-trainer interaction, horse weight change
- **Platt scaling** - LogisticRegression calibrator eliminates overconfident probabilities (was predicting 44% for top bin, actual 15%)
- **Dedicated place model** - separate XGBoost trained on is_top3 labels (65.9% accuracy vs 24.3% base rate). Replaces `win_prob * 1.8` hack
- **60/20/20 temporal split** - early stopping on validation set, final metrics on held-out test set (was using test set for both)
- **Odds triple-counting fixed** - removed odds + odds_movement from rule engine (market signal enters only via 30% market blend)
- **Model integrity** - SHA-256 checksum required for loading, feature count assertion, backup-before-train with rollback
- **Brier score + calibration curve** in accuracy report (`python3 -m scripts.accuracy --all`)
- **Feature correlation matrix** logged after each training run, flags pairs with |r| > 0.7
- **Date-aware track records** - training no longer leaks future race times into distance_pb_ratio

### Previous Versions
- **v5.0**: 26-factor engine, real DB stats, 18-feature ML, historical scraper, self-improving pipeline
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

**Rule-Based System (75%) - 24 Factors**

| Category | Factors | Total Weight |
|----------|---------|--------------|
| **Core** | Rating (vs field avg), Form, Tips Index, Jockey (DB stats), Trainer (DB stats), Draw + Rail Bias | ~52% |
| **Expert** | Going Preference, Rolling Form, Beaten Margins, Jockey Change, Jockey/Trainer Hot Streaks, Field Size, Career Place Rate, Age vs Field | ~37% |
| **Technical** | Pace Map, C&D Form, Class Movement, Gear Changes, Weight Analysis, Sectional Speed, Freshness, Rating Trend, Season Stakes | ~11% |

Note: Market odds removed from rule engine — enters only via 30% market blend in calibration (prevents triple-counting). Weights are dynamic — factors with missing data are excluded and remaining weights re-normalize to 1.0.

**ML-Based System (25%)**
- XGBoost classifier with **22 features** + dedicated **place model** (is_top3), both Platt-calibrated
- 60/20/20 temporal split with early stopping on validation set
- Trained on **4,155+ real race results** from horse_database with recency weighting (newest races 3.3x)
- Top features: recent_form (10.4%), jockey_trainer_combo (6.6%), days_since_last_win (6.6%), class_change (6.3%), field_rel_speed (5.3%)
- Feature correlation matrix logged after training — redundant pairs flagged automatically
- Model improves automatically as more race days are scraped

### Key Improvements Over Hardcoded Approach

| What | Old (v5.0) | New (v5.1) |
|------|-----------|-----------|
| ML features | 18 (with odds, redundant pairs) | 22 (no odds, no redundancy, interaction terms) |
| Probability calibration | Softmax temp=2.5 (overconfident) | Platt scaling + softmax temp=1.5 |
| Place prediction | `win_prob * 1.8` hack | Dedicated XGBoost on is_top3 (65.9% accuracy) |
| Train/test split | 80/20 (test used for early stopping) | 60/20/20 (val for stopping, test for metrics) |
| Speed feature | Absolute (conflates sprinters/stayers) | Field-relative percentile within prior fields |
| Weight feature | Absolute 108-135 lbs | Relative to field median (lighter = advantage) |
| Market signal | Triple-counted (rule+rule+blend) | Single entry via 30% market blend |
| Model safety | Pickle loaded unsigned | SHA-256 checksum + feature count guard + backup |

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
    feature_schema.py        #   Single source of truth for 22 features
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
