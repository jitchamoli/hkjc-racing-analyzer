#!/usr/bin/env python3
"""Centralized configuration for HKJC Racing Analyzer."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Database paths
DB_PATH = PROJECT_ROOT / "tracking_data" / "horse_history.db"
ODDS_DB_PATH = PROJECT_ROOT / "tracking_data" / "odds_history.db"

# Model paths
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "hkjc_racing_model.pkl"
EXPERIMENT_LOG_PATH = MODEL_DIR / "experiment_log.json"

# Training data
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
TRAINING_DATA_PATH = TRAINING_DATA_DIR / "historical_races.json"

# Scoring weights
RULE_WEIGHT = 0.60
ML_WEIGHT = 0.40

# Betting thresholds
MIN_EXPECTED_RETURN = 0.05
MIN_INVESTMENT = 10
MAX_INVESTMENT = 200

# Bankroll-based stake suggestions (HKD). Used by the per-card "suggested
# stake" line in the dashboard. Adjust to your actual rolling bankroll.
# Conservative default. Per-tier fractions are defined in
# betting/optimizer.compute_suggested_stake.
BANKROLL = 10000

# Monitor
POLL_INTERVAL = 300  # seconds
PRE_RACE_WINDOW = 360  # seconds

# ML hyperparameters
# colsample_bytree=1.0: with only 19 features, column subsampling saves
# nothing and made the jockey_win_rate × jockey_form_30d (r=0.75) pair
# double-count in any tree they both landed in. With all features per
# tree, XGBoost picks the better-correlating one per split.
# tree_method="hist": ~3-5× faster on 50k samples than the default exact.
# reg_lambda explicit: was implicit at 1.0 from xgboost defaults.
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "eval_metric": "logloss",
}

# Training data thresholds
MIN_DB_SAMPLES = 200
MIN_REAL_SAMPLES = 50
SYNTHETIC_FALLBACK_SIZE = 2000

# Softmax temperature for probability normalization
# Used as multiplier: exp((s - max) * T). Lower = flatter distribution.
# Per-horse calibration is handled at the model level by IsotonicRegression
# (Wave 2c switch from Platt). The softmax here is purely for race-level
# normalization on top of already-calibrated probabilities.
# 1.5 = mild differentiation without overconfidence.
SOFTMAX_TEMPERATURE = 1.5

# Market blend ratio (model vs market)
# 40% market anchors probabilities to realistic levels (bookmakers are well-calibrated)
# 60% model preserves our ranking edge from rule engine + ML
MARKET_BLEND_WEIGHT = 0.40
MODEL_BLEND_WEIGHT = 0.60

# HKJC URLs
HKJC_BASE_URL = "https://racing.hkjc.com/racing/information/English"
HKJC_RACE_CARD_URL = f"{HKJC_BASE_URL}/Racing/RaceCard.aspx"
HKJC_RESULTS_URL = f"{HKJC_BASE_URL}/Racing/LocalResults.aspx"
HKJC_ODDS_URL = f"{HKJC_BASE_URL}/Odds"
