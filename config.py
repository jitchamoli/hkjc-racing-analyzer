#!/usr/bin/env python3
"""Centralized configuration for HKJC Racing Analyzer."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Load .env file from project root, if present. python-dotenv is a no-op
# when the file is missing, so this is safe in fresh checkouts. See
# .env.example for the expected variables.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv optional; secrets still readable via plain os.environ

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

# Minimum rides for the jockey_form factor to fire. Set to 15 (2026-06-19) to
# effectively disable the factor — n_rides SQL LIMIT is 10, so >=15 never
# fires. Backtest 2026-05-23 found disabling improves straight QIN +6.24pp and
# top-pick Brier; shadow forward-validation confirmed it: 7 meetings / 71 races,
# +31.9pp ROI delta, 6/7 meetings favourable, Bayesian posterior 0.965 → SHIP.
JOCKEY_FORM_RIDES_MIN = 15

# Betting thresholds
MIN_EXPECTED_RETURN = 0.05
MIN_INVESTMENT = 10
MAX_INVESTMENT = 200

# Bankroll-based stake suggestions (HKD). Used by the per-card "suggested
# stake" line in the dashboard. Adjust to your actual rolling bankroll.
# Conservative default. Per-tier fractions are defined in
# betting/optimizer.compute_suggested_stake.
BANKROLL = 10000

# Card-level exposure cap: per-race stakes are computed independently, so a
# busy card can sum to 15-20% of bankroll at risk at once (and the box bets
# within a race are correlated — they win/lose together). Cap the day's TOTAL
# forward exposure at this fraction of bankroll; the dashboard pro-rates every
# unfinished race's stake down proportionally if the sum exceeds it.
CARD_EXPOSURE_CAP = 0.10

# Bankroll guard (stop-loss / daily caps) — betting/bankroll_guard.py. A
# STATEFUL discipline layer that recomputes the day's realised P&L from
# race_bet_ticket + results and HALTS further betting once a cap is hit (the
# anti-chase mechanism CARD_EXPOSURE_CAP does NOT provide — that only pro-rates
# forward stakes within a card, it never stops you after a losing run). No edge
# is assumed; this is pure wallet protection (tip #5: max daily loss, stop after
# limit, no doubling). Set BANKROLL above to your real rolling bankroll.
MAX_STAKE_PER_RACE_PCT = 0.03   # cap any single race's stake at 3% of bankroll
DAILY_EXPOSURE_CAP_PCT = 0.10   # block NEW bets once cumulative stake >= 10%
DAILY_LOSS_CAP_PCT = 0.08       # hard STOP once realised loss >= 8% of bankroll

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

# The Racing API (https://www.theracingapi.com) — UK + IRE + HK racing data.
# Uses HTTP Basic Auth — both credentials loaded from .env (see .env.example).
# Used by data/uk_racing_api.py for the UK pipeline; HK pipeline still
# scrapes HKJC directly and does NOT consume these credentials.
RACING_API_USERNAME = os.environ.get("RACING_API_USERNAME", "")
RACING_API_PASSWORD = os.environ.get("RACING_API_PASSWORD", "")
RACING_API_BASE_URL = "https://api.theracingapi.com"  # final path TBD after first API call

# UK database path — separate file from HK so schemas/ML models don't mix.
UK_DB_PATH = PROJECT_ROOT / "tracking_data" / "uk_history.db"

# AU database path — separate file from UK/HK. Same theracingapi.com client
# (data/uk_racing_api.py) services both UK and AU; only the region filter +
# DB destination differ.
AU_DB_PATH = PROJECT_ROOT / "tracking_data" / "au_history.db"

# JP database path — same plumbing as UK/AU, region='jpn'. Plan tier returns
# stakes-only JRA racing (Hanshin, Nakayama, Tokyo, etc) without OR/RPR
# ratings — too sparse to model, kept for reference / HKIR raider lookup.
JP_DB_PATH = PROJECT_ROOT / "tracking_data" / "jp_history.db"

# FR database path — same plumbing, region='fr'. Standard plan returns ~2,364
# races/yr across 7 major French tracks (Longchamp, Auteuil, Chantilly,
# Saint-Cloud, Compiegne, Deauville, Toulouse) with OR coverage ~89% — the
# only non-UK jurisdiction with enough data + ratings to plausibly model.
FR_DB_PATH = PROJECT_ROOT / "tracking_data" / "fr_history.db"
