# HKJC Horse Racing Analyzer v5.3

Real-time horse racing analysis for the Hong Kong Jockey Club (HKJC). Fetches live race data, odds, and expert tips from racing.hkjc.com to generate betting recommendations using a 24-factor rule-based engine + XGBoost ML model (23 features, Platt-calibrated) with exotic bet value calculator, trained on real historical race data, persistent horse database, and self-improving data pipeline.

## What's New in v5.3

- **Data leakage eliminated** - par times date-filtered in training, jockey/trainer win rates computed from prior-only race history (not season aggregates)
- **Speed figure weight correction fixed** - symmetric adjustment around 120lb baseline (heavier credited, lighter debited). Was previously inverted (double-penalizing heavy horses)
- **23-feature ML schema** - removed `late_money_signal` (constant in training, variable at prediction, causing distribution mismatch). Odds signal enters via market blend instead
- **Train/predict consistency** - jockey/trainer win rates now use same prior-only `race_results` queries in both training and prediction
- **Prediction merge logic** - re-running predictions preserves scored results from prior runs; includes bogus-data guard
- **Quinella edge fix** - boosts applied to both model and market probabilities for fair comparison
- **ML place model wired up** - betting optimizer uses dedicated place model instead of crude heuristic
- **value_score sync** - now correctly updated after probability calibration

### Previous Versions
- **v5.2**: Beyer speed figures, exotic calculator, dynamic blend, 24-feature ML, enhanced quinella pricing
- **v5.1**: 22-feature ML, Platt scaling, place model, correlation audit, code review fixes
- **v5.0**: 26-factor engine, real DB stats, 18-feature ML, historical scraper, self-improving pipeline
- **v4.0**: QIN default strategy, BANKER WIN system, market probability blending, 17 factors
- **v3.0**: 14-factor analytics, rail bias, pace map, sectionals, C&D form, odds movement
- **v2.0**: Tips Index integration, ML prediction, advanced analytics
- **v1.0**: Basic rule-based scoring
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

Note: Market odds removed from rule engine — enters only via dynamic market blend (35-55% depending on field size/distance). Weights are dynamic — factors with missing data are excluded and remaining weights re-normalize to 1.0.

**ML-Based System (25%)**
- XGBoost classifier with **23 features** + dedicated **place model** (is_top3), both Platt-calibrated
- 60/20/20 temporal split with early stopping on validation, Platt calibration on validation, unbiased test metrics
- Trained on **4,155+ real race results** from horse_database with recency weighting (newest races 3.3x)
- **Adjusted speed figures**: Beyer-style par-adjusted, symmetric weight correction (±0.1s per lb from 120 baseline)
- **Dynamic market blend**: 55% market for large sprints, 35% for small/long fields
- Top features: recent_form (9.5%), class_change (8.2%), field_rel_speed (5.8%), prior_win_rate (5.2%), freshness (4.4%)
- Feature correlation matrix logged after training — redundant pairs flagged automatically

**Exotic Bet Calculator**
- Prices exacta, trifecta, quinella, first4 using conditional probability elimination
- Quinella enhanced with place-model boost, pace pairing, draw clustering
- Value detection: flags combinations where model probability exceeds market-implied by 10%+
- Dashboard shows exotic value picks per race with edge percentages

### Key Improvements

| What | v5.0 | v5.2 |
|------|------|------|
| ML features | 18 (with odds, redundant pairs) | 24 (par-adjusted speed, interactions, late money) |
| Speed figures | Absolute m/s (conflates sprinters/stayers) | Beyer-style par-adjusted + weight-corrected |
| Probability calibration | Softmax temp=2.5 (overconfident) | Platt scaling + dynamic market blend |
| Place prediction | `win_prob * 1.8` hack | Dedicated XGBoost on is_top3 (67.5% accuracy) |
| Exotic bets | Not supported | Full calculator: exacta, trifecta, quinella, first4 |
| Quinella pricing | Simple probability | Place-model boost + pace pairing + draw clustering |
| Market blend | Static 70/30 | Dynamic 45-65% model / 35-55% market by race type |
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
    feature_schema.py        #   Single source of truth for 23 features
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
