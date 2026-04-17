# Architecture Refactor Design

**Date**: 2026-04-17
**Scope**: Incremental refactor of HKJC Racing Analyzer — config, ML pipeline, analyzer split, validation, imports, experiment tracking, tests

## Problem

The codebase has grown organically and has several structural gaps:
- Constants hardcoded across modules (no centralized config)
- `core/predictor.py` (1,144 lines) mixes feature engineering, model logic, training data builders, and a scraper
- `core/analyzer.py` (1,715 lines) is a god class combining orchestration, scoring, and bet recommendations
- No data validation on parsed HKJC HTML — site changes cause silent failures
- Inconsistent imports (some use root shims, some use package paths)
- No experiment tracking — old model metrics are lost on retrain
- Train/predict feature consistency enforced by convention, not code
- Zero tests

## Approach

Incremental layered refactor. Each step protects the next:

1. Config + Feature Schema
2. ML pipeline split
3. Analyzer split
4. Data validation
5. Import cleanup
6. Experiment tracking
7. Tests

## 1. Config + Feature Schema

### `config.py` (project root)

Single source for all tunable constants:

```python
# Paths
DB_PATH = "tracking_data/horse_history.db"
ODDS_DB_PATH = "tracking_data/odds_history.db"
MODEL_PATH = "models/hkjc_racing_model.pkl"
EXPERIMENT_LOG_PATH = "models/experiment_log.json"
TRAINING_DATA_PATH = "training_data/historical_races.json"

# Scoring weights
RULE_WEIGHT = 0.75
ML_WEIGHT = 0.25

# Betting thresholds
MIN_EXPECTED_RETURN = 0.05
MIN_INVESTMENT = 10
MAX_INVESTMENT = 200

# Monitor
POLL_INTERVAL = 300
PRE_RACE_WINDOW = 360

# ML hyperparameters
XGBOOST_N_ESTIMATORS = 200
XGBOOST_MAX_DEPTH = 5
XGBOOST_LEARNING_RATE = 0.05
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8
XGBOOST_MIN_CHILD_WEIGHT = 3
XGBOOST_GAMMA = 0.1
MIN_DB_SAMPLES = 200
MIN_REAL_SAMPLES = 50
SYNTHETIC_FALLBACK_SIZE = 2000

# HKJC URLs
HKJC_RACE_CARD_URL = "https://racing.hkjc.com/racing/information/English/Racing/RaceCard.aspx"
HKJC_RESULTS_URL = "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"
HKJC_ODDS_URL = "https://racing.hkjc.com/racing/information/English/Odds"
```

### `ml/feature_schema.py`

Shared feature definition used by both training and prediction:

```python
FEATURE_SCHEMA = [
    {"name": "prior_win_rate", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "recent_form", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "draw_advantage", "default": 0.6, "range": (0.0, 1.0)},
    {"name": "jockey_win_rate", "default": 0.08, "range": (0.0, 0.5)},
    {"name": "trainer_win_rate", "default": 0.06, "range": (0.0, 0.4)},
    {"name": "field_size", "default": 0.857, "range": (0.0, 1.0)},
    {"name": "field_rel_draw", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "prior_pace", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "prior_speed", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "cd_form_prior", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "odds_implied_prob", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "class_level", "default": 0.6, "range": (0.0, 1.0)},
    {"name": "freshness", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "going_pref", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "going_form_rate", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "jockey_horse_coop", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "distance_pb_ratio", "default": 0.5, "range": (0.0, 1.0)},
    {"name": "odds_movement", "default": 0.5, "range": (0.0, 1.0)},
]

FEATURE_NAMES = [f["name"] for f in FEATURE_SCHEMA]
FEATURE_DEFAULTS = {f["name"]: f["default"] for f in FEATURE_SCHEMA}
NUM_FEATURES = len(FEATURE_SCHEMA)

def validate_features(features) -> bool:
    """Assert feature vector matches schema length."""
    assert len(features) == NUM_FEATURES, \
        f"Expected {NUM_FEATURES} features, got {len(features)}"
    return True
```

## 2. ML Pipeline Split

Split `core/predictor.py` (1,144 lines) into:

### `ml/features.py` (~250 lines)
- `FeatureEngineering` class with all static helper methods
- `extract_features(horse_data, all_results, venue)` — builds 18-feature vector using `FEATURE_SCHEMA` order
- DB lookups for prior race history
- Imports `FEATURE_SCHEMA` from `ml/feature_schema.py`

### `ml/model.py` (~150 lines)
- `HKJCRacingMLModel` class — train, predict, save, load
- `RaceResult`, `HorseFeatures`, `MLPrediction` dataclasses
- `create_combined_prediction()` function
- `CombinedRacingAnalyzer` class (thin wrapper)
- Imports hyperparams from `config.py`
- Imports `FEATURE_NAMES` from `ml/feature_schema.py`

### `ml/training.py` (~300 lines)
- `build_training_data_from_db()` — uses `FEATURE_SCHEMA` order
- `generate_synthetic_training_data()` — fallback
- Imports `FEATURE_SCHEMA` from `ml/feature_schema.py`

### `ml/experiments.py` (~60 lines)
- `log_experiment(metrics, features_used, data_source, n_samples)` — appends to JSON
- See Section 6.

### `ml/__init__.py`
Re-exports for backwards compatibility:
```python
from ml.features import FeatureEngineering
from ml.model import HKJCRacingMLModel, RaceResult, HorseFeatures, MLPrediction, create_combined_prediction, CombinedRacingAnalyzer
from ml.training import build_training_data_from_db, generate_synthetic_training_data
from ml.feature_schema import FEATURE_NAMES, FEATURE_SCHEMA, NUM_FEATURES, validate_features
```

Root shim `ml_predictor.py` unchanged: `from ml import *`

### `HKJCHistoricalScraper`
Moves from `core/predictor.py` to `data/scraper.py` (where it belongs).

## 3. Analyzer Split

Split `core/analyzer.py` (1,715 lines) into:

### `core/analyzer.py` (~400 lines) — orchestrator
- `CombinedRaceAnalyzer` class, slimmed down
- `analyze_race(race_card)` — calls scorer, then recommender
- Imports and runs all 12 analytics modules
- Collects raw factor scores per horse

### `core/scorer.py` (~500 lines) — score logic
- `combine_scores(factor_scores, ml_prob)` — 75/25 blend
- `calibrate_probabilities(raw_scores, field_size)` — normalize to sum ~1.0
- Dynamic weight renormalization for missing factors
- `create_combined_prediction()` may live here instead of ml/model.py if it's more scoring than ML

### `core/recommender.py` (~400 lines) — bet recommendations
- `assign_bet_recommendations(ranked_horses, odds)`
- BANKER/WIN/QIN/PLACE/SKIP logic
- Market-favourite promotion and BANKER override rank swaps
- Probability sync after rank swaps
- Expected return calculations
- Race-skip logic

26-factor scoring functions (~400 lines) remain in `core/analyzer.py` — tightly coupled to analysis.

Root shim `hkjc_racing_analyzer.py` unchanged via `core/__init__.py` re-exports.

## 4. Data Validation

### `data/validation.py` (~100 lines)

```python
def validate_horse_entry(entry) -> list[str]:
    """Returns list of warnings. Empty = valid."""
    # Check: name non-empty, draw 1-14, rating 0-140,
    # weight reasonable, odds > 0, jockey/trainer non-empty

def validate_race_card(card) -> list[str]:
    """Validate full race card after parsing."""
    # Check: >= 2 horses, distance 800-2400, delegate to validate_horse_entry
```

Called in `core/fetcher.py` after `HKJCParser.parse_race_card()`. Warnings logged, not fatal.

## 5. Import Cleanup

Rule: all internal imports use package paths.

Files to update:
- `dashboard/generator.py` — `from core.fetcher import ...`
- `dashboard/results_fetcher.py` — same
- `monitor/race_day.py` — same
- `train_ml_model.py` — `from ml.training import ...`, `from ml.model import ...`
- `interactive_analyzer.py` — package imports

Root shims stay for CLI usage but no internal module imports them.

## 6. Experiment Tracking

### `ml/experiments.py` (~60 lines)

Appends one JSON entry per training run to `models/experiment_log.json`:

```python
{
    "timestamp": "2026-04-17T14:30:00",
    "model_version": "20260417-143000",
    "accuracy": 0.89,
    "precision": 0.34,
    "recall": 0.28,
    "logloss": 0.31,
    "n_samples": 4155,
    "data_source": "db",
    "feature_importance": {...}
}
```

Called from `ml/model.py` after `train()` succeeds.

## 7. Tests

### `tests/conftest.py` — shared fixtures
- `sample_horse_data` — dict with all expected fields
- `sample_race_card` — minimal valid race card
- `tmp_db` — temporary SQLite DB with ~20 fake race results
- `trained_model` — model trained on small synthetic dataset

### `tests/test_features.py`
- Feature vector length == 18
- Features match `FEATURE_SCHEMA` order
- Values within declared ranges
- Missing data uses defaults (no NaN)
- `extract_features()` and `build_training_data_from_db()` produce same shape
- No data leakage with `race_date` filter

### `tests/test_scoring.py`
- 75/25 blend for known inputs
- Probabilities sum to ~1.0 after calibration
- Rank swap updates all probabilities
- ML unavailable → 100% rule-based
- `value_score == combined_prob * 100`

### `tests/test_validation.py`
- Valid entry → no warnings
- Missing name, bad draw/rating → warnings
- Empty race card → warning
- Normal card → clean

### `tests/test_data_integrity.py`
- No NaN/Inf in feature vectors
- No constant features (std > 0.01)
- Realistic win rate in labels (5-15%)
- Model pickle has expected keys

Run: `pytest tests/ -v`

## Files Created (new)

- `config.py`
- `ml/__init__.py`
- `ml/feature_schema.py`
- `ml/features.py`
- `ml/model.py`
- `ml/training.py`
- `ml/experiments.py`
- `core/scorer.py`
- `core/recommender.py`
- `data/validation.py`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_features.py`
- `tests/test_scoring.py`
- `tests/test_validation.py`
- `tests/test_data_integrity.py`

## Files Modified

- `core/analyzer.py` — slim down, extract scorer + recommender
- `core/predictor.py` — gutted, becomes shim to `ml/`
- `core/fetcher.py` — add validation call after parsing
- `core/__init__.py` — re-exports
- `train_ml_model.py` — use package imports
- `dashboard/generator.py` — use package imports
- `dashboard/results_fetcher.py` — use package imports
- `monitor/race_day.py` — use package imports
- `interactive_analyzer.py` — use package imports
- `ml_predictor.py` (root shim) — `from ml import *`

## Files Unchanged

- All 12 `analytics/` modules
- `data/database.py`, `data/odds_tracker.py`, `data/odds_scraper.py`
- `betting/optimizer.py`
- `dashboard/template.py`
- `scripts/` entry points
- All other root shims except `ml_predictor.py`
