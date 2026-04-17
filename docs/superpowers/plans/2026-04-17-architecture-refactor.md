# Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the HKJC Racing Analyzer into a cleaner architecture with centralized config, split ML pipeline, split analyzer, data validation, consistent imports, experiment tracking, and a pytest test suite.

**Architecture:** Incremental layered refactor — each task builds on the previous. Config + feature schema first (shared definitions), then ML pipeline split, analyzer split, validation, imports, experiments, and finally tests to lock everything in.

**Tech Stack:** Python 3, XGBoost, scikit-learn, numpy, SQLite, pytest

---

### Task 1: Create `config.py` — Centralized Constants

**Files:**
- Create: `config.py`

- [ ] **Step 1: Create `config.py` with all constants extracted from codebase**

```python
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
RULE_WEIGHT = 0.75
ML_WEIGHT = 0.25

# Betting thresholds
MIN_EXPECTED_RETURN = 0.05
MIN_INVESTMENT = 10
MAX_INVESTMENT = 200

# Monitor
POLL_INTERVAL = 300  # seconds
PRE_RACE_WINDOW = 360  # seconds

# ML hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

# Training data thresholds
MIN_DB_SAMPLES = 200
MIN_REAL_SAMPLES = 50
SYNTHETIC_FALLBACK_SIZE = 2000

# Softmax temperature for probability calibration
SOFTMAX_TEMPERATURE = 8.0

# Market blend ratio (model vs market)
MARKET_BLEND_WEIGHT = 0.30
MODEL_BLEND_WEIGHT = 0.70

# HKJC URLs
HKJC_BASE_URL = "https://racing.hkjc.com/racing/information/English"
HKJC_RACE_CARD_URL = f"{HKJC_BASE_URL}/Racing/RaceCard.aspx"
HKJC_RESULTS_URL = f"{HKJC_BASE_URL}/Racing/LocalResults.aspx"
HKJC_ODDS_URL = f"{HKJC_BASE_URL}/Odds"
```

- [ ] **Step 2: Verify the file is importable**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "import config; print(config.DB_PATH)"`
Expected: prints the DB path without errors

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "Add centralized config.py with all tunable constants"
```

---

### Task 2: Create `ml/feature_schema.py` — Shared Feature Definition

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/feature_schema.py`

- [ ] **Step 1: Create the `ml/` package directory**

```bash
mkdir -p ml
```

- [ ] **Step 2: Create `ml/__init__.py`**

```python
"""ML pipeline: feature engineering, model training, and prediction."""
```

- [ ] **Step 3: Create `ml/feature_schema.py`**

```python
"""Shared feature schema — single source of truth for train and predict.

Both extract_features() and build_training_data_from_db() MUST build
feature vectors in FEATURE_SCHEMA order. This prevents train/predict mismatch.
"""

import numpy as np

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
    """Assert feature vector matches schema length and has no NaN/Inf."""
    arr = np.asarray(features, dtype=np.float32)
    assert len(arr) == NUM_FEATURES, (
        f"Expected {NUM_FEATURES} features, got {len(arr)}"
    )
    assert not np.any(np.isnan(arr)), "Feature vector contains NaN"
    assert not np.any(np.isinf(arr)), "Feature vector contains Inf"
    return True
```

- [ ] **Step 4: Verify import**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES, validate_features; print(f'{NUM_FEATURES} features: {FEATURE_NAMES[:3]}...')"` 
Expected: `18 features: ['prior_win_rate', 'recent_form', 'draw_advantage']...`

- [ ] **Step 5: Commit**

```bash
git add ml/__init__.py ml/feature_schema.py
git commit -m "Add ml/feature_schema.py — shared feature definition for train/predict"
```

---

### Task 3: Create `ml/features.py` — Feature Extraction

**Files:**
- Create: `ml/features.py`
- Reference: `core/predictor.py:190-545` (existing `FeatureEngineering` class and `extract_features`)

- [ ] **Step 1: Create `ml/features.py`**

Move `FeatureEngineering` class and `extract_features` from `core/predictor.py` into `ml/features.py`. Replace hardcoded feature count with `FEATURE_SCHEMA` imports and add `validate_features()` call at the end.

```python
#!/usr/bin/env python3
"""Feature engineering for ML predictions.

Extracts 18 features per horse using FEATURE_SCHEMA order.
All features use only data available BEFORE the race (no leakage).
"""

import re
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES, validate_features
import config

logger = logging.getLogger(__name__)


class FeatureEngineering:
    @staticmethod
    def parse_last_6(last_6: str) -> List[int]:
        """Parse form string — handles '1/3/10/2/5' and legacy '13425'."""
        s = str(last_6)
        if "/" in s:
            return [int(p) for p in s.split("/") if p.strip().isdigit()]
        return [int(ch) for ch in s if ch.isdigit()]

    @staticmethod
    def calculate_form_score(last_6: str) -> float:
        positions = FeatureEngineering.parse_last_6(last_6)
        if not positions:
            return 50.0

        total = 0
        for pos in positions:
            if pos == 1:
                total += 100
            elif pos == 2:
                total += 70
            elif pos == 3:
                total += 50
            elif pos == 4:
                total += 30
            elif pos == 5:
                total += 15
            else:
                total += 5

        return total / len(positions)

    @staticmethod
    def calculate_win_rate(history) -> float:
        if not history:
            return 0.08
        wins = sum(1 for r in history if r.is_win)
        return min(0.8, wins / len(history))

    @staticmethod
    def calculate_place_rate(history) -> float:
        if not history:
            return 0.25
        places = sum(1 for r in history if r.is_place)
        return min(0.9, places / len(history))

    @staticmethod
    def calculate_avg_position(history) -> float:
        if not history:
            return 6.0
        return sum(r.position for r in history) / len(history)

    @staticmethod
    def calculate_jockey_win_rate(jockey: str, all_results) -> float:
        jockey_rides = [r for r in all_results if jockey.lower() in r.jockey.lower()]
        if not jockey_rides:
            return 0.12
        wins = sum(1 for r in jockey_rides if r.is_win)
        return min(0.6, wins / len(jockey_rides))

    @staticmethod
    def calculate_trainer_win_rate(trainer: str, all_results) -> float:
        trainer_horses = [r for r in all_results if trainer.lower() in r.trainer.lower()]
        if not trainer_horses:
            return 0.12
        wins = sum(1 for r in trainer_horses if r.is_win)
        return min(0.5, wins / len(trainer_horses))

    @staticmethod
    def calculate_draw_advantage(draw: int, distance) -> float:
        try:
            distance = int(distance)
        except (ValueError, TypeError):
            distance = 1200

        if distance <= 1000:
            ideal = 4
        elif distance <= 1400:
            ideal = 6
        else:
            ideal = 10

        diff = abs(draw - ideal)
        if diff == 0:
            return 1.0
        elif diff <= 2:
            return 0.8
        elif diff <= 4:
            return 0.6
        else:
            return max(0.3, 1 - (diff * 0.1))

    @staticmethod
    def _style_to_float(style: str) -> float:
        return {"Leader": 0.9, "Stalker": 0.7, "Midfield": 0.5, "Closer": 0.3}.get(style, 0.5)

    @staticmethod
    def extract_features(horse_data: dict, all_results, venue: str) -> np.ndarray:
        """Extract 18 features matching FEATURE_SCHEMA order exactly.

        Features:
        1. prior_win_rate  2. recent_form  3. draw_advantage
        4. jockey_win_rate 5. trainer_win_rate 6. field_size
        7. field_rel_draw  8. prior_pace   9. prior_speed
        10. cd_form_prior  11. odds_implied 12. class_level
        13. freshness      14. going_pref  15. going_form_rate
        16. jockey_horse_coop 17. distance_pb_ratio 18. odds_movement
        """
        features = []
        horse_name = horse_data.get('name', '')
        distance = horse_data.get('distance', 1200)
        going = horse_data.get('going', '')

        # Load prior race history from DB (strictly before race_date if provided)
        prior = []
        race_date = horse_data.get('race_date', '')
        try:
            db_path = config.DB_PATH
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                if race_date:
                    rows = conn.execute(
                        "SELECT * FROM race_results WHERE horse_name = ? AND race_date < ? ORDER BY race_date",
                        (horse_name, race_date)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM race_results WHERE horse_name = ? ORDER BY race_date",
                        (horse_name,)
                    ).fetchall()
                prior = [dict(r) for r in rows]
                conn.close()
        except Exception:
            pass

        # Feature 1: prior win rate
        if prior:
            prior_wins = sum(1 for p in prior if p.get('is_winner', 0))
            prior_top3 = sum(1 for p in prior if p.get('is_top3', 0))
            feat_form_proxy = min(1.0, (prior_wins * 2 + prior_top3) / (len(prior) * 3))
        else:
            feat_form_proxy = 0.5
        features.append(feat_form_proxy)

        # Feature 2: recent form from prior finishes
        if prior:
            recent = prior[-6:]
            form_pts = 0
            for p in recent:
                pos = p.get('finish_position', 7)
                if pos == 1: form_pts += 10
                elif pos == 2: form_pts += 7
                elif pos == 3: form_pts += 5
                elif pos <= 5: form_pts += 2
            features.append(form_pts / (len(recent) * 10))
        else:
            last_6 = horse_data.get('last_6', '')
            if last_6:
                features.append(FeatureEngineering.calculate_form_score(last_6) / 100)
            else:
                features.append(0.5)

        # Feature 3: draw advantage
        draw = horse_data.get('draw', 8)
        features.append(FeatureEngineering.calculate_draw_advantage(draw, distance))

        # Feature 4: jockey win rate
        jockey = horse_data.get('jockey', '')
        jockey_rate = FeatureEngineering.calculate_jockey_win_rate(jockey, all_results)
        features.append(min(0.5, jockey_rate) if jockey_rate > 0 else 0.08)

        # Feature 5: trainer win rate
        trainer = horse_data.get('trainer', '')
        trainer_rate = FeatureEngineering.calculate_trainer_win_rate(trainer, all_results)
        features.append(min(0.4, trainer_rate) if trainer_rate > 0 else 0.06)

        # Feature 6: field size
        field_size = horse_data.get('field_size', 12)
        features.append(field_size / 14.0)

        # Feature 7: field-relative draw
        features.append(draw / max(1, field_size) if draw > 0 else 0.5)

        # Feature 8: prior pace
        pace_vals = []
        for p in prior[-5:]:
            rp = p.get('running_position', '') or ''
            if rp:
                if ' ' in rp:
                    positions = [int(x) for x in rp.split() if x.isdigit()]
                else:
                    positions = [int(c) for c in rp if c.isdigit()]
                if positions:
                    early = min(positions[0], 14)
                    pace_vals.append(1.0 - (early - 1) / 13.0)
        features.append(sum(pace_vals) / len(pace_vals) if pace_vals else 0.5)

        # Feature 9: prior finishing speed
        speed_vals = []
        for p in prior[-5:]:
            ft = p.get('finish_time', '') or ''
            d = p.get('distance', 0) or 0
            if ft and ':' in ft and d > 0:
                try:
                    parts = ft.split(':')
                    secs = float(parts[0]) * 60 + float(parts[1])
                    if secs > 0:
                        speed = d / secs
                        speed_vals.append(max(0.0, min(1.0, (speed - 14) / 4)))
                except (ValueError, IndexError):
                    pass
        features.append(sum(speed_vals) / len(speed_vals) if speed_vals else 0.5)

        # Feature 10: C&D form from prior races
        prior_at_dist = [p for p in prior if p.get('distance') == distance]
        if prior_at_dist:
            cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
            features.append(min(1.0, cd_wins / len(prior_at_dist) + 0.2))
        else:
            features.append(0.5)

        # Feature 11: odds-implied probability
        win_odds = horse_data.get('win_odds', 0)
        features.append(min(1.0, 1.0 / win_odds) if win_odds > 0 else 0.5)

        # Feature 12: class level
        class_type = horse_data.get('class_type', '')
        cm = re.search(r'(\d)', class_type)
        class_rank = int(cm.group(1)) if cm else 3
        features.append(max(0.0, min(1.0, class_rank / 5.0)))

        # Feature 13: freshness
        dslr = horse_data.get('days_since_last_run', 0) or 0
        if not dslr and prior:
            last_date = prior[-1].get('race_date', '')
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date.replace('/', '-'), '%Y-%m-%d')
                    dslr = (datetime.now() - last_dt).days
                except Exception:
                    pass
        if dslr > 0:
            features.append(max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60)))
        else:
            features.append(0.5)

        # Feature 14: going preference
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if prior_same_going:
                going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                going_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                features.append(min(1.0, (going_wins * 2 + going_top3) / (len(prior_same_going) * 3) + 0.2))
            else:
                features.append(0.5)
        else:
            features.append(0.5)

        # Feature 15: going form rate
        going_form_rate = 0.5
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if len(prior_same_going) >= 2:
                going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                going_form_rate = going_wins / len(prior_same_going)
        features.append(going_form_rate)

        # Feature 16: jockey-horse cooperation rate
        jockey_horse_coop = 0.5
        jockey = horse_data.get('jockey', '')
        if prior and jockey:
            prior_same_jockey = [p for p in prior if p.get('jockey', '') == jockey]
            if len(prior_same_jockey) >= 2:
                jh_wins = sum(1 for p in prior_same_jockey if p.get('is_winner', 0))
                jh_places = sum(1 for p in prior_same_jockey if p.get('is_top3', 0))
                jockey_horse_coop = min(1.0, (jh_wins * 2 + jh_places) / (len(prior_same_jockey) * 3))
        features.append(jockey_horse_coop)

        # Feature 17: distance personal best ratio
        distance_pb_ratio = 0.5
        if prior:
            prior_at_dist = [p for p in prior
                             if p.get('distance') == distance
                             and p.get('finish_time', '')
                             and ':' in str(p.get('finish_time', ''))]
            if prior_at_dist:
                def _parse_time(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0

                times = [_parse_time(p['finish_time']) for p in prior_at_dist]
                valid_times = [t for t in times if t < 999]
                if valid_times:
                    horse_pb = min(valid_times)
                    try:
                        db_path_pb = config.DB_PATH
                        if db_path_pb.exists():
                            conn_pb = sqlite3.connect(str(db_path_pb))
                            row_pb = conn_pb.execute(
                                "SELECT MIN(finish_time) FROM race_results WHERE distance = ? AND finish_time != '' AND finish_time IS NOT NULL",
                                (distance,)
                            ).fetchone()
                            conn_pb.close()
                            if row_pb and row_pb[0]:
                                track_record = _parse_time(row_pb[0])
                                if track_record < 999 and horse_pb < 999:
                                    distance_pb_ratio = min(1.0, track_record / horse_pb)
                    except Exception:
                        pass
        features.append(distance_pb_ratio)

        # Feature 18: odds movement
        odds_movement = 0.5
        win_odds = horse_data.get('win_odds', 0)
        horse_number = horse_data.get('number', 0)
        race_date_for_odds = horse_data.get('race_date', '')
        race_number = horse_data.get('race_number', 0)
        if win_odds > 0 and horse_number > 0 and race_date_for_odds:
            try:
                odds_db_path = config.ODDS_DB_PATH
                if odds_db_path.exists():
                    conn_odds = sqlite3.connect(str(odds_db_path))
                    row_odds = conn_odds.execute(
                        """SELECT win_odds FROM odds_snapshots
                           WHERE race_date = ? AND race_number = ? AND horse_number = ?
                           ORDER BY timestamp ASC LIMIT 1""",
                        (race_date_for_odds, race_number, horse_number)
                    ).fetchone()
                    conn_odds.close()
                    if row_odds and row_odds[0] > 0:
                        opening = row_odds[0]
                        movement = (opening - win_odds) / opening
                        odds_movement = max(0.0, min(1.0, (movement + 1) / 2))
            except Exception:
                pass
        features.append(odds_movement)

        result = np.array(features, dtype=np.float32)
        validate_features(result)
        return result
```

- [ ] **Step 2: Verify import and basic functionality**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from ml.features import FeatureEngineering; print(FeatureEngineering.calculate_draw_advantage(3, 1200))"`
Expected: `0.8`

- [ ] **Step 3: Commit**

```bash
git add ml/features.py
git commit -m "Add ml/features.py — feature extraction using shared schema"
```

---

### Task 4: Create `ml/model.py` — Model Train/Predict/Save/Load

**Files:**
- Create: `ml/model.py`
- Reference: `core/predictor.py:547-687` (HKJCRacingMLModel class)
- Reference: `core/predictor.py:33-80` (dataclasses)
- Reference: `core/predictor.py:1063-1107` (create_combined_prediction)
- Reference: `core/predictor.py:1108-1145` (CombinedRacingAnalyzer)

- [ ] **Step 1: Create `ml/model.py`**

```python
#!/usr/bin/env python3
"""XGBoost model for HKJC race prediction — train, predict, save, load."""

import pickle
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np

import config
from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RaceResult:
    date: str
    venue: str
    race_number: int
    distance: int
    going: str
    class_type: str
    horse_number: int
    horse_name: str
    draw: int
    jockey: str
    trainer: str
    weight: int
    rating: float
    position: int
    win_odds: float
    place_odds: float
    last_6: str
    is_win: bool
    is_place: bool


@dataclass
class HorseFeatures:
    horse_number: int
    horse_name: str
    rating: float
    draw: int
    weight: int
    recent_form_score: float
    win_rate: float
    place_rate: float
    avg_position: float
    jockey_win_rate: float
    trainer_win_rate: float
    distance_suitability: float
    draw_advantage: float
    going_suitability: float


@dataclass
class MLPrediction:
    horse_number: int
    horse_name: str
    win_probability: float
    place_probability: float
    ml_confidence: str
    feature_importance: Dict[str, float]
    model_version: str


class HKJCRacingMLModel:
    MODEL_DIR = config.MODEL_DIR
    MODEL_FILE = config.MODEL_PATH

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_version = "1.0"
        self.feature_importance = {}

        if ML_AVAILABLE:
            self._load_model()

    def _load_model(self):
        if self.MODEL_FILE.exists():
            try:
                with open(self.MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.model_version = data.get('version', '1.0')
                    self.feature_importance = data.get('feature_importance', {})
                    self.is_trained = self.model is not None
                    logger.info(f"Loaded model version {self.model_version}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def _save_model(self):
        self.MODEL_DIR.mkdir(exist_ok=True)

        data = {
            'model': self.model,
            'version': self.model_version,
            'feature_importance': self.feature_importance,
            'trained_date': datetime.now().isoformat()
        }

        with open(self.MODEL_FILE, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved model version {self.model_version}")

    def train(self, features: np.ndarray, labels: np.ndarray,
              sample_weight: np.ndarray = None) -> Dict:
        if not ML_AVAILABLE:
            return {'error': 'ML libraries not available'}

        logger.info(f"Training with {len(features)} samples "
                    f"({labels.sum():.0f} winners, {labels.mean()*100:.1f}% win rate)")

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42,
            stratify=labels if labels.sum() >= 10 else None
        )

        sw_train = None
        if sample_weight is not None:
            sw_train, _ = train_test_split(
                sample_weight, test_size=0.2, random_state=42,
                stratify=labels if labels.sum() >= 10 else None
            )

        n_pos = max(1, y_train.sum())
        n_neg = max(1, len(y_train) - n_pos)
        scale = n_neg / n_pos

        params = dict(config.XGBOOST_PARAMS)
        params['scale_pos_weight'] = scale

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, sample_weight=sw_train)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        self.feature_importance = dict(zip(
            FEATURE_NAMES,
            self.model.feature_importances_.tolist()
        ))

        self.is_trained = True
        self.model_version = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._save_model()

        # Log experiment
        from ml.experiments import log_experiment
        result = {
            'accuracy': accuracy,
            'logloss': logloss,
            'precision': precision,
            'recall': recall,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version
        }
        log_experiment(result, FEATURE_NAMES, n_samples=len(features))

        return result

    def predict(self, features: np.ndarray, field_size: int = 12) -> Tuple[float, str]:
        if not self.is_trained or self.model is None:
            base_prob = 1.0 / max(field_size, 1)
            return base_prob, "MODEL_NOT_TRAINED"

        proba = self.model.predict_proba(features.reshape(1, -1))[0]

        win_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        if win_prob > 0.30:
            confidence = "HIGH"
        elif win_prob > 0.15:
            confidence = "MEDIUM"
        elif win_prob > 0.08:
            confidence = "LOW"
        else:
            confidence = "VERY_LOW"

        return win_prob, confidence


def create_combined_prediction(
    rule_score: float,
    rule_confidence: str,
    ml_prob: float,
    ml_confidence: str,
    ml_available: bool
) -> Tuple[float, str, Dict]:
    weights = {'rule': config.RULE_WEIGHT, 'ml': config.ML_WEIGHT}

    if not ml_available:
        weights = {'rule': 1.0, 'ml': 0.0}

    ml_prob_adjusted = ml_prob * weights['ml']
    rule_prob = (rule_score / 100) * weights['rule']

    combined_prob = rule_prob + ml_prob_adjusted

    if 'HIGH' in rule_confidence and 'HIGH' in ml_confidence:
        combined_confidence = "VERY_HIGH"
    elif 'HIGH' in rule_confidence or 'HIGH' in ml_confidence:
        combined_confidence = "HIGH"
    elif 'MEDIUM' in rule_confidence or 'MEDIUM' in ml_confidence:
        combined_confidence = "MEDIUM"
    else:
        combined_confidence = "LOW"

    breakdown = {
        'rule_based': {
            'score': rule_score,
            'contribution': rule_prob,
            'confidence': rule_confidence
        },
        'ml_based': {
            'probability': ml_prob,
            'contribution': ml_prob_adjusted,
            'confidence': ml_confidence if ml_available else 'NOT_AVAILABLE'
        },
        'final_probability': combined_prob,
        'weights': weights
    }

    return combined_prob, combined_confidence, breakdown


class CombinedRacingAnalyzer:
    """Thin wrapper that combines ML model with feature engineering."""

    def __init__(self):
        from ml.features import FeatureEngineering
        from ml.training import build_training_data_from_db, generate_synthetic_training_data

        self.ml_model = HKJCRacingMLModel()
        self.feature_eng = FeatureEngineering()

        if ML_AVAILABLE and not self.ml_model.is_trained:
            X_real, y_real, sw_real = build_training_data_from_db()
            if X_real is not None and len(X_real) >= config.MIN_REAL_SAMPLES:
                logger.info(f"Training ML model with {len(X_real)} REAL race results...")
                result = self.ml_model.train(X_real, y_real, sample_weight=sw_real)
                logger.info(f"Model trained on REAL data: accuracy={result['accuracy']:.2%}")
            else:
                logger.info("Insufficient real data — training ML model with synthetic data...")
                X, y = generate_synthetic_training_data()
                result = self.ml_model.train(X, y)
                logger.info(f"Model trained on synthetic data: accuracy={result['accuracy']:.2%}")

    def predict(self, horse_data: dict, all_results, venue: str) -> MLPrediction:
        features = self.feature_eng.extract_features(horse_data, all_results, venue)

        field_size = horse_data.get('field_size', 12)
        ml_prob, ml_confidence = self.ml_model.predict(features, field_size=field_size)

        return MLPrediction(
            horse_number=horse_data.get('number', 0),
            horse_name=horse_data.get('name', ''),
            win_probability=ml_prob,
            place_probability=min(1.0, ml_prob * 2.5),
            ml_confidence=ml_confidence,
            feature_importance=dict(zip(
                self.ml_model.FEATURE_NAMES if hasattr(self.ml_model, 'FEATURE_NAMES') else FEATURE_NAMES,
                (features * self.ml_model.feature_importance.get(f, 0.1) for f in FEATURE_NAMES)
            )),
            model_version=self.ml_model.model_version
        )
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from ml.model import HKJCRacingMLModel, RaceResult, MLPrediction, create_combined_prediction, ML_AVAILABLE; print('ML_AVAILABLE:', ML_AVAILABLE)"`
Expected: `ML_AVAILABLE: True`

- [ ] **Step 3: Commit**

```bash
git add ml/model.py
git commit -m "Add ml/model.py — XGBoost model with config-driven hyperparams"
```

---

### Task 5: Create `ml/training.py` — Training Data Builders

**Files:**
- Create: `ml/training.py`
- Reference: `core/predictor.py:688-1062` (build_training_data_from_db, generate_synthetic_training_data)

- [ ] **Step 1: Create `ml/training.py`**

```python
#!/usr/bin/env python3
"""Training data builders for the ML model.

Three sources in priority order:
1. SQLite DB (build_training_data_from_db) — preferred when >= MIN_DB_SAMPLES
2. Scraped race data (build_training_data) — fallback
3. Synthetic data (generate_synthetic_training_data) — last resort
"""

import re
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np

import config
from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES
from ml.features import FeatureEngineering

logger = logging.getLogger(__name__)


def build_training_data_from_db() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Build training data from real race results in horse_database.

    Returns (features, labels, sample_weights) or (None, None, None) if insufficient data.
    Each row = one horse in one race. Label = 1 if winner, 0 otherwise.
    """
    conn = None
    try:
        db_path = config.DB_PATH
        if not db_path.exists():
            return None, None, None

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT r.*, j.win_rate as jockey_wr, j.place_rate as jockey_pr,
                   t.win_rate as trainer_wr, t.place_rate as trainer_pr
            FROM race_results r
            LEFT JOIN jockey_stats j ON r.jockey = j.name
            LEFT JOIN trainer_stats t ON r.trainer = t.name
            WHERE r.finish_position > 0
            ORDER BY r.race_date, r.race_number
        """).fetchall()

        if len(rows) < config.MIN_REAL_SAMPLES:
            logger.info(f"Only {len(rows)} results in DB — not enough for real training")
            return None, None, None

        logger.info(f"Building training data from {len(rows)} real race results")

        # Precompute field sizes per race
        field_sizes = {}
        for row in rows:
            r = dict(row)
            rk = f"{r.get('race_date', '')}-{r.get('race_number', 0)}"
            field_sizes[rk] = field_sizes.get(rk, 0) + 1

        # Build per-horse chronological race history for PRIOR-only lookups
        horse_history = {}
        for row in rows:
            r = dict(row)
            name = r.get('horse_name', '')
            if name:
                if name not in horse_history:
                    horse_history[name] = []
                horse_history[name].append(r)
        for name in horse_history:
            horse_history[name].sort(key=lambda x: x.get('race_date', ''))

        X, y, weights = [], [], []
        for row in rows:
            r = dict(row)
            draw = r.get('draw', 0) or 0
            distance = r.get('distance', 1200) or 1200
            jockey_wr = r.get('jockey_wr', 0) or 0
            trainer_wr = r.get('trainer_wr', 0) or 0
            win_odds = r.get('win_odds', 0) or 0
            horse_name = r.get('horse_name', '')
            race_date = r.get('race_date', '')
            race_no = r.get('race_number', 0)
            race_key = f"{race_date}-{race_no}"
            field_size = field_sizes.get(race_key, 12)
            going = r.get('going', '') or ''

            prior = [h for h in horse_history.get(horse_name, [])
                     if h.get('race_date', '') < race_date]

            # Feature 1: prior win rate
            if prior:
                prior_wins = sum(1 for p in prior if p.get('is_winner', 0))
                prior_top3 = sum(1 for p in prior if p.get('is_top3', 0))
                feat_form_proxy = min(1.0, (prior_wins * 2 + prior_top3) / (len(prior) * 3))
            else:
                feat_form_proxy = 0.5

            # Feature 2: recent form
            if prior:
                recent = prior[-6:]
                form_pts = 0
                for p in recent:
                    pos = p.get('finish_position', 7)
                    if pos == 1: form_pts += 10
                    elif pos == 2: form_pts += 7
                    elif pos == 3: form_pts += 5
                    elif pos <= 5: form_pts += 2
                feat_form = form_pts / (len(recent) * 10)
            else:
                feat_form = 0.5

            # Feature 3: draw advantage
            feat_draw = FeatureEngineering.calculate_draw_advantage(draw, distance)

            # Feature 4: jockey win rate
            feat_jockey = min(0.5, jockey_wr) if jockey_wr > 0 else 0.08

            # Feature 5: trainer win rate
            feat_trainer = min(0.4, trainer_wr) if trainer_wr > 0 else 0.06

            # Feature 6: field size
            feat_field_size = field_size / 14.0

            # Feature 7: field-relative draw
            feat_field_draw = draw / max(1, field_size) if draw > 0 else 0.5

            # Feature 8: prior pace
            pace_vals = []
            for p in prior[-5:]:
                rp = p.get('running_position', '') or ''
                fs = field_sizes.get(f"{p.get('race_date','')}-{p.get('race_number',0)}", 12)
                if rp:
                    if ' ' in rp:
                        positions = [int(x) for x in rp.split() if x.isdigit()]
                    else:
                        positions = [int(c) for c in rp if c.isdigit()]
                    if positions and fs > 1:
                        early = min(positions[0], fs)
                        pace_vals.append(1.0 - (early - 1) / max(1, fs - 1))
            feat_pace = sum(pace_vals) / len(pace_vals) if pace_vals else 0.5

            # Feature 9: prior finishing speed
            speed_vals = []
            for p in prior[-5:]:
                ft = p.get('finish_time', '') or ''
                d = p.get('distance', 0) or 0
                if ft and ':' in ft and d > 0:
                    try:
                        parts = ft.split(':')
                        secs = float(parts[0]) * 60 + float(parts[1])
                        if secs > 0:
                            speed = d / secs
                            speed_vals.append(max(0.0, min(1.0, (speed - 14) / 4)))
                    except (ValueError, IndexError):
                        pass
            feat_finish = sum(speed_vals) / len(speed_vals) if speed_vals else 0.5

            # Feature 10: C&D form
            prior_at_dist = [p for p in prior if p.get('distance') == distance]
            if prior_at_dist:
                cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
                feat_cd = min(1.0, cd_wins / len(prior_at_dist) + 0.2)
            else:
                feat_cd = 0.5

            # Feature 11: odds-implied probability
            feat_odds = min(1.0, 1.0 / win_odds) if win_odds > 0 else 0.5

            # Feature 12: class level
            class_type = r.get('class_type', '') or ''
            cm = re.search(r'(\d)', class_type)
            class_rank = int(cm.group(1)) if cm else 3
            feat_class = max(0.0, min(1.0, class_rank / 5.0))

            # Feature 13: freshness
            dslr = r.get('days_since_last_run', 0) or 0
            if dslr > 0:
                feat_freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))
            else:
                feat_freshness = 0.5

            # Feature 14: going preference
            if prior and going:
                prior_same_going = [p for p in prior if p.get('going', '') == going]
                if prior_same_going:
                    going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                    going_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                    feat_going = min(1.0, (going_wins * 2 + going_top3) / (len(prior_same_going) * 3) + 0.2)
                else:
                    feat_going = 0.5
            else:
                feat_going = 0.5

            # Feature 15: going form rate
            if prior and going:
                prior_same_going_f = [p for p in prior if p.get('going', '') == going]
                if len(prior_same_going_f) >= 2:
                    gf_wins = sum(1 for p in prior_same_going_f if p.get('is_winner', 0))
                    feat_going_form = gf_wins / len(prior_same_going_f)
                else:
                    feat_going_form = 0.5
            else:
                feat_going_form = 0.5

            # Feature 16: jockey-horse cooperation
            jockey_name = r.get('jockey', '') or ''
            if prior and jockey_name:
                prior_same_jh = [p for p in prior if p.get('jockey', '') == jockey_name]
                if len(prior_same_jh) >= 2:
                    jh_w = sum(1 for p in prior_same_jh if p.get('is_winner', 0))
                    jh_p = sum(1 for p in prior_same_jh if p.get('is_top3', 0))
                    feat_jh_coop = min(1.0, (jh_w * 2 + jh_p) / (len(prior_same_jh) * 3))
                else:
                    feat_jh_coop = 0.5
            else:
                feat_jh_coop = 0.5

            # Feature 17: distance PB ratio
            feat_dist_pb = 0.5
            prior_dist_times = [p for p in prior
                                if p.get('distance') == distance
                                and p.get('finish_time', '')
                                and ':' in str(p.get('finish_time', ''))]
            if prior_dist_times:
                def _pt(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0
                horse_times = [_pt(p['finish_time']) for p in prior_dist_times]
                valid_ht = [t for t in horse_times if t < 999]
                if valid_ht:
                    horse_pb_t = min(valid_ht)
                    try:
                        tr_row = conn.execute(
                            "SELECT MIN(finish_time) FROM race_results WHERE distance = ? AND finish_time != '' AND finish_time IS NOT NULL AND race_date < ?",
                            (distance, race_date)
                        ).fetchone()
                        if tr_row and tr_row[0]:
                            tr_secs = _pt(tr_row[0])
                            if tr_secs < 999 and horse_pb_t < 999:
                                feat_dist_pb = min(1.0, tr_secs / horse_pb_t)
                    except Exception:
                        pass

            # Feature 18: odds movement (not available in historical DB)
            feat_odds_move = 0.5

            features = [
                feat_form_proxy, feat_form, feat_draw,
                feat_jockey, feat_trainer, feat_field_size,
                feat_field_draw, feat_pace, feat_finish, feat_cd, feat_odds,
                feat_class, feat_freshness, feat_going,
                feat_going_form, feat_jh_coop, feat_dist_pb, feat_odds_move,
            ]
            assert len(features) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(features)}"
            X.append(features)
            y.append(1 if r.get('is_winner', 0) else 0)
            weights.append(1.0)

        conn.close()

    except Exception as e:
        logger.warning(f"Error building training data from DB: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return None, None, None

    return np.array(X, dtype=np.float32), np.array(y), np.array(weights)


def generate_synthetic_training_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data calibrated to real HK racing distributions.
    Used as fallback when real DB data is insufficient.
    """
    np.random.seed(42)

    X = []
    y = []

    for _ in range(n_samples):
        rating = np.clip(np.random.normal(60, 18), 15, 100)
        form_score = np.random.beta(2.5, 3) * 100
        draw_adv = np.random.uniform(0.3, 1.0)

        jockey_rate = np.random.choice(
            [np.random.uniform(0.15, 0.25),
             np.random.uniform(0.08, 0.15),
             np.random.uniform(0.03, 0.08)],
            p=[0.15, 0.35, 0.50]
        )
        trainer_rate = np.random.choice(
            [np.random.uniform(0.12, 0.20),
             np.random.uniform(0.06, 0.12),
             np.random.uniform(0.02, 0.06)],
            p=[0.15, 0.40, 0.45]
        )
        field_size = np.random.choice([6, 8, 10, 12, 14], p=[0.05, 0.15, 0.25, 0.35, 0.20])
        feat_field_size = field_size / 14.0
        feat_field_draw = draw_adv * np.random.uniform(0.3, 1.0)
        prior_pace = np.random.uniform(0.3, 0.9)
        prior_speed = np.random.uniform(0.2, 0.9)
        cd_form = np.random.uniform(0.2, 1.0)
        odds_implied = np.random.uniform(0.03, 0.50)
        class_level = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.40, 0.25, 0.15]) / 5.0
        dslr = np.random.choice([7, 14, 21, 28, 45, 90], p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])
        freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))
        going_pref = np.random.uniform(0.3, 0.8)
        going_form_rate = np.random.uniform(0.0, 0.5)
        jockey_horse_coop = np.random.uniform(0.0, 0.5)
        distance_pb_ratio = np.random.uniform(0.7, 1.0)
        odds_movement_feat = np.random.uniform(0.3, 0.7)

        features = [
            rating / 100, form_score / 100, draw_adv,
            jockey_rate, trainer_rate, feat_field_size,
            feat_field_draw, prior_pace, prior_speed, cd_form, odds_implied,
            class_level, freshness, going_pref,
            going_form_rate, jockey_horse_coop, distance_pb_ratio, odds_movement_feat,
        ]
        assert len(features) == NUM_FEATURES
        X.append(features)

        win_logit = (
            1.8 * (rating / 100 - 0.6)
            + 1.2 * (form_score / 100 - 0.4)
            + 0.5 * (draw_adv - 0.6)
            + 3.0 * (jockey_rate - 0.10)
            + 2.0 * (trainer_rate - 0.08)
            + 0.3 * (feat_field_size - 0.8)
            + 0.4 * (feat_field_draw - 0.5)
            + 0.5 * (prior_pace - 0.5)
            + 0.3 * (prior_speed - 0.5)
            + 0.5 * (cd_form - 0.5)
            + 1.5 * (odds_implied - 0.1)
            + 0.3 * (class_level - 0.5)
            + 0.2 * (freshness - 0.5)
            + 0.3 * (going_pref - 0.5)
            + 0.4 * (going_form_rate - 0.25)
            + 0.3 * (jockey_horse_coop - 0.25)
            + 0.2 * (distance_pb_ratio - 0.85)
            + 0.1 * (odds_movement_feat - 0.5)
            - 2.3
        )
        win_prob = 1.0 / (1.0 + np.exp(-win_logit))
        is_win = np.random.random() < win_prob
        y.append(1 if is_win else 0)

    return np.array(X, dtype=np.float32), np.array(y)
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from ml.training import generate_synthetic_training_data; X, y = generate_synthetic_training_data(100); print(f'X shape: {X.shape}, winners: {y.sum()}')"`
Expected: `X shape: (100, 18), winners: <some number>`

- [ ] **Step 3: Commit**

```bash
git add ml/training.py
git commit -m "Add ml/training.py — DB + synthetic training data builders"
```

---

### Task 6: Create `ml/experiments.py` — Experiment Tracking

**Files:**
- Create: `ml/experiments.py`

- [ ] **Step 1: Create `ml/experiments.py`**

```python
#!/usr/bin/env python3
"""Lightweight experiment tracking — appends metrics per training run to JSON."""

import json
import logging
from datetime import datetime
from typing import Dict, List

import config

logger = logging.getLogger(__name__)


def log_experiment(metrics: Dict, features_used: List[str], n_samples: int) -> None:
    """Append one entry per training run to experiment log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_version": metrics.get("model_version", "unknown"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "logloss": metrics.get("logloss"),
        "n_samples": n_samples,
        "n_features": len(features_used),
        "feature_importance": metrics.get("feature_importance"),
    }

    log_path = config.EXPERIMENT_LOG_PATH
    log_path.parent.mkdir(exist_ok=True)

    existing = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    existing.append(entry)

    with open(log_path, 'w') as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Logged experiment: accuracy={entry['accuracy']:.4f}, "
                f"n_samples={n_samples}, version={entry['model_version']}")


def get_experiments() -> List[Dict]:
    """Load all experiment entries."""
    log_path = config.EXPERIMENT_LOG_PATH
    if not log_path.exists():
        return []
    try:
        with open(log_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
```

- [ ] **Step 2: Verify**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from ml.experiments import get_experiments; print(f'{len(get_experiments())} experiments logged')"`
Expected: `0 experiments logged`

- [ ] **Step 3: Commit**

```bash
git add ml/experiments.py
git commit -m "Add ml/experiments.py — JSON-based training run tracking"
```

---

### Task 7: Update `ml/__init__.py` — Re-exports for Backwards Compatibility

**Files:**
- Modify: `ml/__init__.py`

- [ ] **Step 1: Update `ml/__init__.py` with re-exports**

```python
"""ML pipeline: feature engineering, model training, and prediction."""

from ml.feature_schema import (
    FEATURE_SCHEMA, FEATURE_NAMES, FEATURE_DEFAULTS,
    NUM_FEATURES, validate_features,
)
from ml.features import FeatureEngineering
from ml.model import (
    HKJCRacingMLModel, RaceResult, HorseFeatures, MLPrediction,
    CombinedRacingAnalyzer, create_combined_prediction, ML_AVAILABLE,
)
from ml.training import build_training_data_from_db, generate_synthetic_training_data
```

- [ ] **Step 2: Update root shim `ml_predictor.py`**

Replace the content of `ml_predictor.py` with:

```python
"""Backwards compatibility shim — imports from ml/ package."""
from ml import *  # noqa: F401,F403
```

This was previously `from core.predictor import *`.

- [ ] **Step 3: Verify the shim works**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from ml_predictor import FeatureEngineering, HKJCRacingMLModel, build_training_data_from_db, ML_AVAILABLE; print('Shim OK, ML_AVAILABLE:', ML_AVAILABLE)"`
Expected: `Shim OK, ML_AVAILABLE: True`

- [ ] **Step 4: Commit**

```bash
git add ml/__init__.py ml_predictor.py
git commit -m "Update ml/__init__.py re-exports and ml_predictor.py shim"
```

---

### Task 8: Update `core/predictor.py` — Redirect to `ml/` Package

**Files:**
- Modify: `core/predictor.py`

- [ ] **Step 1: Replace `core/predictor.py` with a shim to `ml/`**

Replace the entire content of `core/predictor.py` with:

```python
"""Core ML predictor — re-exports from ml/ package.

All ML logic has moved to the ml/ package:
  ml/feature_schema.py — shared feature definitions
  ml/features.py       — feature extraction
  ml/model.py          — XGBoost model
  ml/training.py       — training data builders
  ml/experiments.py    — experiment tracking
"""

from ml import *  # noqa: F401,F403
```

- [ ] **Step 2: Verify core.predictor still works for existing importers**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from core.predictor import FeatureEngineering, HKJCRacingMLModel, CombinedRacingAnalyzer, create_combined_prediction, ML_AVAILABLE; print('core.predictor shim OK')"`
Expected: `core.predictor shim OK`

- [ ] **Step 3: Verify `core/analyzer.py` still imports correctly**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from core.analyzer import CombinedRaceAnalyzer; print('analyzer OK')"`
Expected: `analyzer OK`

- [ ] **Step 4: Commit**

```bash
git add core/predictor.py
git commit -m "Replace core/predictor.py with shim to ml/ package"
```

---

### Task 9: Create `core/scorer.py` — Score Combination + Calibration

**Files:**
- Create: `core/scorer.py`
- Reference: `core/analyzer.py:1003-1064` (weight map + normalization)
- Reference: `core/analyzer.py:1206-1260` (softmax calibration + market blend)

- [ ] **Step 1: Create `core/scorer.py`**

```python
#!/usr/bin/env python3
"""Score combination, calibration, and probability normalization."""

import math
import logging
from typing import List, Dict, Tuple

import config

logger = logging.getLogger(__name__)


class BettingWeights:
    """Factor weight constants for the 26-factor rule engine."""
    RATING = 0.07
    FORM = 0.09
    ODDS = 0.08
    JOCKEY = 0.04
    TRAINER = 0.02
    DRAW = 0.04
    TIPS = 0.10
    PACE = 0.03
    SECTIONAL = 0.03
    CD_FORM = 0.04
    ODDS_MOVE = 0.04
    CLASS_MOVE = 0.03
    GEAR = 0.02
    WEIGHT = 0.04
    FRESHNESS = 0.04
    RATING_TREND = 0.03
    STAKES = 0.02
    GOING = 0.05
    ROLLING_FORM = 0.06
    MARGIN = 0.05
    JOCKEY_CHANGE = 0.04
    JOCKEY_FORM = 0.04
    TRAINER_FORM = 0.03
    FIELD_SIZE = 0.05
    CAREER_PLACE = 0.03
    AGE = 0.02

    # Mapping factor keys to weights
    WEIGHT_MAP = {
        "rating": RATING, "form": FORM, "odds": ODDS,
        "jockey": JOCKEY, "trainer": TRAINER, "draw": DRAW,
        "tips": TIPS, "pace": PACE, "sectional": SECTIONAL,
        "cd_form": CD_FORM, "odds_movement": ODDS_MOVE,
        "class_movement": CLASS_MOVE, "gear": GEAR,
        "weight_carried": WEIGHT, "freshness": FRESHNESS,
        "rating_trend": RATING_TREND, "season_stakes": STAKES,
        "going_pref": GOING, "rolling_form": ROLLING_FORM,
        "margin": MARGIN, "jockey_change": JOCKEY_CHANGE,
        "jockey_form": JOCKEY_FORM, "trainer_form": TRAINER_FORM,
        "field_size": FIELD_SIZE, "career_place": CAREER_PLACE,
        "age": AGE,
    }


# Factors always included (score of 0 means no data; these use 50 as neutral)
ALWAYS_INCLUDED = {"rating", "form", "odds", "jockey", "trainer", "draw",
                   "class_movement", "gear", "weight_carried"}


def compute_weighted_score(factors: Dict[str, float]) -> float:
    """Compute rule-based score from factor dict with dynamic weight normalization.

    Factors with score == 0 are excluded (except ALWAYS_INCLUDED factors).
    Remaining weights are renormalized to sum to 1.0.
    """
    weight_map = {}
    for key, score in factors.items():
        if key in ALWAYS_INCLUDED or score > 0:
            if key in BettingWeights.WEIGHT_MAP:
                weight_map[key] = BettingWeights.WEIGHT_MAP[key]

    total_weight = sum(weight_map.values())
    if total_weight > 0:
        return sum(factors[k] * (w / total_weight) for k, w in weight_map.items())
    return 50.0


def calibrate_probabilities(analyses, calibrated_probs=None) -> List[float]:
    """Convert raw combined_probability scores into calibrated probabilities.

    Uses softmax with temperature scaling, then blends with market-implied
    probabilities (30% market, 70% model) to prevent severely underrating
    strong favourites.
    """
    if not analyses:
        return []

    temp = config.SOFTMAX_TEMPERATURE
    raw_scores = [a.combined_probability for a in analyses]
    max_score = max(raw_scores) if raw_scores else 0.5
    exp_scores = [math.exp((s - max_score) * temp) for s in raw_scores]
    exp_total = sum(exp_scores) or 1.0
    cal_probs = [e / exp_total for e in exp_scores]

    # Blend with market-implied probabilities
    market_probs = []
    for a in analyses:
        if a.horse.win_odds > 0:
            market_probs.append(1.0 / a.horse.win_odds)
        else:
            market_probs.append(0.0)
    market_total = sum(market_probs) or 1.0
    market_probs = [p / market_total for p in market_probs]

    blended = []
    for i in range(len(cal_probs)):
        if market_probs[i] > 0:
            b = config.MODEL_BLEND_WEIGHT * cal_probs[i] + config.MARKET_BLEND_WEIGHT * market_probs[i]
        else:
            b = cal_probs[i]
        blended.append(b)

    blend_total = sum(blended) or 1.0
    return [p / blend_total for p in blended]


def apply_calibrated_probs(analyses, calibrated_probs: List[float]) -> None:
    """Write calibrated probabilities and expected returns onto analyses."""
    for i, a in enumerate(analyses):
        a.combined_probability = calibrated_probs[i]
        if a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
        else:
            a.expected_return = -1.0
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from core.scorer import compute_weighted_score, BettingWeights; print('scorer OK, keys:', len(BettingWeights.WEIGHT_MAP))"`
Expected: `scorer OK, keys: 26`

- [ ] **Step 3: Commit**

```bash
git add core/scorer.py
git commit -m "Add core/scorer.py — score combination and probability calibration"
```

---

### Task 10: Create `core/recommender.py` — Bet Recommendations

**Files:**
- Create: `core/recommender.py`
- Reference: `core/analyzer.py:1206-1419` (existing `_assign_bet_recommendations`)

- [ ] **Step 1: Create `core/recommender.py`**

```python
#!/usr/bin/env python3
"""Bet recommendation engine — assigns BANKER/WIN/QIN/PLACE/SKIP per horse."""

import logging
from typing import List

import config
from core.scorer import calibrate_probabilities, apply_calibrated_probs

logger = logging.getLogger(__name__)


def assign_bet_recommendations(analyses) -> None:
    """Assign realistic bet recommendations based on field shape.

    Calibrates probabilities via softmax + market blend, handles
    market-favourite promotion and BANKER overrides, then assigns
    bet types per horse.
    """
    if not analyses:
        return

    n = len(analyses)
    calibrated_probs = calibrate_probabilities(analyses)
    apply_calibrated_probs(analyses, calibrated_probs)

    top = analyses[0] if n >= 1 else None
    second = analyses[1] if n >= 2 else None

    top_prob = calibrated_probs[0] if n >= 1 else 0
    second_prob = calibrated_probs[1] if n >= 2 else 0
    third_prob = calibrated_probs[2] if n >= 3 else 0

    gap_1_2 = top_prob - second_prob
    gap_2_3 = second_prob - third_prob
    top_3_share = top_prob + second_prob + third_prob

    market_prob_top = (1 / top.horse.win_odds) if top and top.horse.win_odds > 0 else 0

    top_is_banker = any("BANKER" in r for r in analyses[0].reasoning) if analyses else False

    # Market favourite promotion (skip if BANKER)
    market_fav = min(analyses, key=lambda a: a.horse.win_odds if a.horse.win_odds > 0 else 999)
    market_fav_prob = (1 / market_fav.horse.win_odds) if market_fav.horse.win_odds > 0 else 0
    if (market_fav_prob >= 0.30 and market_fav.rank > 1
            and market_fav.horse.win_odds > 0
            and not top_is_banker):
        old_idx = next(i for i, a in enumerate(analyses) if a is market_fav)
        analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]
        calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]
        _resync_ranks_and_probs(analyses, calibrated_probs)
        top = analyses[0]
        top_prob = calibrated_probs[0]
        second_prob = calibrated_probs[1] if n >= 2 else 0
        third_prob = calibrated_probs[2] if n >= 3 else 0
        gap_1_2 = top_prob - second_prob
        gap_2_3 = second_prob - third_prob
        top_3_share = top_prob + second_prob + third_prob
        market_prob_top = market_fav_prob

    # BANKER promotion
    banker_horse = None
    for a in analyses:
        if any("BANKER" in r for r in a.reasoning):
            if banker_horse is None or a.rule_score > banker_horse.rule_score:
                banker_horse = a

    if banker_horse and banker_horse.rank > 1:
        old_idx = next(i for i, a in enumerate(analyses) if a is banker_horse)
        analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]
        calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]
        _resync_ranks_and_probs(analyses, calibrated_probs)
        top = analyses[0]
        top_prob = calibrated_probs[0]
        second_prob = calibrated_probs[1] if n >= 2 else 0
        gap_1_2 = top_prob - second_prob

    # Determine race-level bet type
    race_bet_type = "QIN"  # default

    if banker_horse and banker_horse.rank == 1:
        race_bet_type = "WIN"
    elif market_prob_top >= 0.35 and gap_1_2 > 0.08:
        race_bet_type = "WIN"
    elif gap_1_2 > 0.06 and top and top.expected_return > 0.15:
        race_bet_type = "WIN"
    elif gap_1_2 < 0.02 and top_3_share > 0.35:
        if gap_2_3 < 0.02 and n >= 3:
            race_bet_type = "TRIO"
        else:
            race_bet_type = "QNL PL"

    # Race-level skip check
    MIN_EV = config.MIN_EXPECTED_RETURN
    top_ev = analyses[0].expected_return if analyses else -1
    top_cal_prob = calibrated_probs[0] if calibrated_probs else 0
    top_market_prob = (1 / analyses[0].horse.win_odds) if analyses and analyses[0].horse.win_odds > 0 else 1
    has_edge = top_cal_prob > top_market_prob
    skip_race = top_ev < MIN_EV and not has_edge and not (banker_horse and banker_horse.rank == 1)

    # Assign per-horse recommendations
    for i, a in enumerate(analyses):
        if skip_race:
            a.bet_recommendation = "SKIP"
            continue

        rank = a.rank
        bet_has_ev = a.expected_return >= MIN_EV
        has_place_value = (a.horse.place_odds > 1.2) if a.horse.place_odds > 0 else (a.horse.win_odds >= 3.0)

        if race_bet_type == "WIN":
            if rank == 1:
                is_banker = any("BANKER" in r for r in a.reasoning)
                a.bet_recommendation = "BANKER WIN" if is_banker else "WIN"
            elif rank == 2 and bet_has_ev and has_place_value:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "QIN":
            if rank <= 2:
                a.bet_recommendation = "QIN"
            elif rank == 3 and bet_has_ev and has_place_value:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "QNL PL":
            if rank <= 2:
                a.bet_recommendation = "QNL PL"
            elif rank == 3 and bet_has_ev and has_place_value:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "TRIO":
            if rank <= 3:
                a.bet_recommendation = "TRIO"
            elif rank == 4 and bet_has_ev and has_place_value:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "PLACE":
            if rank <= 3 and bet_has_ev:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        else:
            a.bet_recommendation = "SKIP"


def _resync_ranks_and_probs(analyses, calibrated_probs: List[float]) -> None:
    """After a rank swap, update all ranks, probabilities, and expected returns."""
    for i, a in enumerate(analyses):
        a.rank = i + 1
        a.combined_probability = calibrated_probs[i]
        if a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from core.recommender import assign_bet_recommendations; print('recommender OK')"`
Expected: `recommender OK`

- [ ] **Step 3: Commit**

```bash
git add core/recommender.py
git commit -m "Add core/recommender.py — bet recommendation engine"
```

---

### Task 11: Update `core/analyzer.py` — Use Scorer + Recommender

**Files:**
- Modify: `core/analyzer.py`

This task wires the analyzer to use `core/scorer.py` and `core/recommender.py` instead of inline logic. The `analyze_horse` method stays in analyzer (it orchestrates factor scoring), but weight computation delegates to `compute_weighted_score()` and `_assign_bet_recommendations` delegates to `recommender.assign_bet_recommendations()`.

- [ ] **Step 1: Update imports at top of `core/analyzer.py`**

Replace lines 19-26 (the predictor imports) with:

```python
from ml.model import (
    CombinedRacingAnalyzer as _MLAnalyzer,
    MLPrediction,
    create_combined_prediction,
    ML_AVAILABLE,
)
```

- [ ] **Step 2: Replace inline BettingStrategy weight constants with import from scorer**

Add after the existing imports:

```python
from core.scorer import compute_weighted_score, BettingWeights
from core.recommender import assign_bet_recommendations
```

- [ ] **Step 3: Update `CombinedRaceAnalyzer.__init__`**

Change line 525 from:
```python
self.ml_analyzer = CombinedRacingAnalyzer()
```
to:
```python
self.ml_analyzer = _MLAnalyzer()
```

- [ ] **Step 4: Replace the weight map block in `analyze_horse`**

Replace lines 1003-1063 (the `weight_map` construction + normalization) with:

```python
        # --- Weighted score calculation (dynamic) ---
        rule_score = compute_weighted_score(factors)
```

- [ ] **Step 5: Replace `_assign_bet_recommendations` static method**

Replace the entire `_assign_bet_recommendations` method (lines 1206-1418) with:

```python
    @staticmethod
    def _assign_bet_recommendations(analyses):
        """Delegate to core.recommender."""
        assign_bet_recommendations(analyses)
```

- [ ] **Step 6: Remove `BettingStrategy` weight constants**

Delete lines 198-225 (the `RATING_WEIGHT`, `FORM_WEIGHT`, etc. constants from `BettingStrategy` class) since they now live in `core/scorer.py:BettingWeights`.

Keep the static methods (`analyze_form`, `analyze_jockey`, `analyze_trainer`, `analyze_draw`, `analyze_tips`, `get_rule_confidence`) — they stay in the analyzer.

- [ ] **Step 7: Also update `import horse_database` to package path**

Change line 542:
```python
from horse_database import HorseDatabase
```
to:
```python
from data.database import HorseDatabase
```

- [ ] **Step 8: Verify the analyzer still works end-to-end**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from core.analyzer import CombinedRaceAnalyzer; a = CombinedRaceAnalyzer(); print('Analyzer init OK')"`
Expected: `Analyzer init OK` (may show model loading logs)

- [ ] **Step 9: Verify the root shim**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from hkjc_racing_analyzer import CombinedRaceAnalyzer; print('Root shim OK')"`
Expected: `Root shim OK`

- [ ] **Step 10: Commit**

```bash
git add core/analyzer.py
git commit -m "Wire analyzer to use scorer + recommender modules"
```

---

### Task 12: Create `data/validation.py` — HKJC Data Validation

**Files:**
- Create: `data/validation.py`

- [ ] **Step 1: Create `data/validation.py`**

```python
#!/usr/bin/env python3
"""Schema validation for parsed HKJC data.

Called after parsing race cards to catch HKJC site layout changes early.
Returns warnings (not fatal) so predictions still run on partial data.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def validate_horse_entry(entry) -> List[str]:
    """Validate a single HorseEntry. Returns list of warning strings."""
    warnings = []
    if not entry.name or len(entry.name) < 2:
        warnings.append(f"Horse #{entry.number}: missing or very short name '{entry.name}'")
    if entry.number < 1 or entry.number > 14:
        warnings.append(f"Horse #{entry.number}: number out of expected range 1-14")
    if entry.draw < 0 or entry.draw > 14:
        warnings.append(f"Horse #{entry.number} {entry.name}: draw {entry.draw} out of range 0-14")
    if entry.rating and (entry.rating < 0 or entry.rating > 140):
        warnings.append(f"Horse #{entry.number} {entry.name}: rating {entry.rating} suspicious (expected 0-140)")
    if entry.jockey and len(entry.jockey) < 2:
        warnings.append(f"Horse #{entry.number} {entry.name}: jockey name too short '{entry.jockey}'")
    if entry.trainer and len(entry.trainer) < 2:
        warnings.append(f"Horse #{entry.number} {entry.name}: trainer name too short '{entry.trainer}'")
    return warnings


def validate_race_card(card) -> List[str]:
    """Validate a full RaceCard after parsing. Returns list of warning strings."""
    warnings = []
    if len(card.horses) < 2:
        warnings.append(f"Race {card.race_number}: only {len(card.horses)} horses parsed (expected 2+)")
    if len(card.horses) > 14:
        warnings.append(f"Race {card.race_number}: {len(card.horses)} horses parsed (expected max 14)")

    distance = 0
    try:
        distance = int(card.distance) if card.distance else 0
    except (ValueError, TypeError):
        pass
    if distance > 0 and (distance < 800 or distance > 2400):
        warnings.append(f"Race {card.race_number}: distance {card.distance}m suspicious (expected 800-2400)")

    for horse in card.horses:
        warnings.extend(validate_horse_entry(horse))

    if warnings:
        for w in warnings:
            logger.warning(f"[Validation] {w}")
    return warnings
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from data.validation import validate_race_card, validate_horse_entry; print('validation OK')"`
Expected: `validation OK`

- [ ] **Step 3: Commit**

```bash
git add data/validation.py
git commit -m "Add data/validation.py — schema checks for parsed HKJC data"
```

---

### Task 13: Wire Validation into `core/fetcher.py`

**Files:**
- Modify: `core/fetcher.py`

- [ ] **Step 1: Add validation import**

Add after existing imports in `core/fetcher.py`:

```python
from data.validation import validate_race_card
```

- [ ] **Step 2: Add validation call after `parse_race_card`**

Find the `parse_race_card` call site in `HKJCDataFetcher.get_race_card()` — after the race card is built and returned, add a validation call. Search for the method that returns a `RaceCard`. Add validation just before returning:

```python
        warnings = validate_race_card(race_card)
        if warnings:
            logger.warning(f"Race {race_number}: {len(warnings)} validation warnings")
```

This should be added right before the `return race_card` statement in `get_race_card()`.

- [ ] **Step 3: Verify fetcher still imports cleanly**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "from core.fetcher import HKJCDataFetcher; print('fetcher OK')"`
Expected: `fetcher OK`

- [ ] **Step 4: Commit**

```bash
git add core/fetcher.py
git commit -m "Wire data validation into fetcher after race card parsing"
```

---

### Task 14: Import Cleanup — Dashboard, Monitor, Interactive

**Files:**
- Modify: `dashboard/generator.py`
- Modify: `dashboard/results_fetcher.py`
- Modify: `interactive_analyzer.py`
- Modify: `train_ml_model.py`

- [ ] **Step 1: Fix `dashboard/generator.py` imports**

Replace:
```python
from hkjc_data_fetcher import HKJCDataFetcher
from hkjc_racing_analyzer import CombinedRaceAnalyzer
```
With:
```python
from core.fetcher import HKJCDataFetcher
from core.analyzer import CombinedRaceAnalyzer
```

- [ ] **Step 2: Fix `dashboard/results_fetcher.py` imports**

Replace:
```python
from hkjc_data_fetcher import HKJCDataFetcher
```
With:
```python
from core.fetcher import HKJCDataFetcher
```

- [ ] **Step 3: Fix `interactive_analyzer.py` imports**

Replace:
```python
from hkjc_data_fetcher import HKJCDataFetcher
from hkjc_racing_analyzer import (
    CombinedRaceAnalyzer,
    ReportGenerator,
    Race,
    Horse
)
from betting_optimizer import BettingOptimizer, BettingReportGenerator
from results_tracker import ResultsTracker
```
With:
```python
from core.fetcher import HKJCDataFetcher
from core.analyzer import (
    CombinedRaceAnalyzer,
    ReportGenerator,
    Race,
    Horse
)
from betting.optimizer import BettingOptimizer, BettingReportGenerator
from results_tracker import ResultsTracker
```

- [ ] **Step 4: Fix `train_ml_model.py` imports**

Replace:
```python
from hkjc_data_fetcher import HKJCApiClient, HKJCParser, HKJCDataFetcher
from ml_predictor import (
    FeatureEngineering, HKJCRacingMLModel, RaceResult,
    generate_synthetic_training_data, build_training_data_from_db, ML_AVAILABLE
)
```
With:
```python
from core.fetcher import HKJCApiClient, HKJCParser, HKJCDataFetcher
from ml.model import HKJCRacingMLModel, RaceResult, ML_AVAILABLE
from ml.features import FeatureEngineering
from ml.training import build_training_data_from_db, generate_synthetic_training_data
```

- [ ] **Step 5: Verify all imports**

Run:
```bash
cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer
python3 -c "import dashboard.generator; print('generator OK')"
python3 -c "import dashboard.results_fetcher; print('results_fetcher OK')"
python3 -c "import interactive_analyzer; print('interactive OK')"
python3 -c "from train_ml_model import main; print('train OK')"
```
Expected: all print OK (may show logging output)

- [ ] **Step 6: Commit**

```bash
git add dashboard/generator.py dashboard/results_fetcher.py interactive_analyzer.py train_ml_model.py
git commit -m "Clean up imports — use package paths everywhere"
```

---

### Task 15: Create `tests/conftest.py` — Shared Test Fixtures

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create test directory and init**

```bash
mkdir -p tests
```

Create `tests/__init__.py`:

```python
"""Test suite for HKJC Racing Analyzer."""
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures for HKJC Racing Analyzer tests."""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

import config


@pytest.fixture
def sample_horse_data():
    """Minimal horse data dict with all fields extract_features expects."""
    return {
        "number": 1,
        "name": "Test Horse",
        "rating": 65.0,
        "draw": 3,
        "jockey": "Z Purton",
        "trainer": "J Size",
        "last_6": "1/3/2/5/4/1",
        "distance": 1200,
        "running_style": "Stalker",
        "last_400m_avg": 23.5,
        "cd_wins": 2,
        "cd_runs": 5,
        "opening_odds": 4.0,
        "win_odds": 3.5,
        "tips_odds": 4.2,
        "course": "A",
        "venue": "Sha Tin",
        "gear": "B",
        "weight": "126",
        "class_type": "Class 3",
        "field_size": 12,
        "going": "Good",
        "days_since_last_run": 21,
        "race_date": "2026/04/17",
        "race_number": 1,
    }


@pytest.fixture
def sample_race_card():
    """Minimal RaceCard-like object for testing."""
    from core.fetcher import RaceCard, HorseEntry

    horses = []
    for i in range(1, 13):
        horses.append(HorseEntry(
            number=i,
            name=f"Horse {i}",
            draw=i,
            jockey=f"Jockey {i}",
            trainer=f"Trainer {i}",
            weight="126",
            rating=50.0 + i * 3,
            win_odds=3.0 + i * 2,
            last_6="3/4/5/2/1/3",
        ))

    return RaceCard(
        race_number=1,
        venue="Sha Tin",
        date="2026/04/17",
        time="13:00",
        distance="1200",
        class_type="Class 3",
        going="Good",
        track="Turf",
        course="A",
        horses=horses,
    )


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite DB with fake race results for testing."""
    db_file = tmp_path / "test_horse_history.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE race_results (
            race_date TEXT, race_number INTEGER, horse_name TEXT,
            horse_number INTEGER, jockey TEXT, trainer TEXT,
            finish_position INTEGER, is_winner INTEGER, is_top3 INTEGER,
            win_odds REAL, draw INTEGER, weight TEXT, horse_weight INTEGER,
            distance INTEGER, going TEXT, class_type TEXT,
            running_position TEXT, finish_time TEXT, margin TEXT,
            days_since_last_run INTEGER, rating REAL, rating_change INTEGER,
            age TEXT, last_6 TEXT, gear TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE jockey_stats (
            name TEXT PRIMARY KEY, rides INTEGER, wins INTEGER,
            places INTEGER, win_rate REAL, place_rate REAL
        )
    """)
    conn.execute("""
        CREATE TABLE trainer_stats (
            name TEXT PRIMARY KEY, runners INTEGER, wins INTEGER,
            places INTEGER, win_rate REAL, place_rate REAL
        )
    """)
    conn.execute("""
        CREATE TABLE horses (
            name TEXT PRIMARY KEY, total_starts INTEGER,
            total_wins INTEGER, total_places INTEGER
        )
    """)

    # Insert 20 fake race results across 2 races
    for race_no in [1, 2]:
        for horse_no in range(1, 11):
            pos = horse_no  # horse 1 wins, horse 2 is second, etc.
            conn.execute("""
                INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "2026/04/10", race_no, f"Horse {horse_no}",
                horse_no, f"Jockey {horse_no}", f"Trainer {horse_no}",
                pos, 1 if pos == 1 else 0, 1 if pos <= 3 else 0,
                3.0 + horse_no, horse_no, "126", 1100,
                1200, "Good", "Class 3",
                f"{horse_no} {horse_no+1}", f"1:{10+horse_no}.{horse_no}0", "",
                21, 60 + horse_no, 0,
                "4", "3/4/5", "",
            ))

    conn.execute("INSERT INTO jockey_stats VALUES (?, ?, ?, ?, ?, ?)",
                 ("Z Purton", 100, 20, 35, 0.20, 0.35))
    conn.execute("INSERT INTO trainer_stats VALUES (?, ?, ?, ?, ?, ?)",
                 ("J Size", 80, 12, 25, 0.15, 0.31))
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture
def trained_model():
    """Small model trained on synthetic data for testing."""
    from ml.model import HKJCRacingMLModel, ML_AVAILABLE
    from ml.training import generate_synthetic_training_data

    if not ML_AVAILABLE:
        pytest.skip("XGBoost not available")

    model = HKJCRacingMLModel()
    X, y = generate_synthetic_training_data(500)
    model.train(X, y)
    return model
```

- [ ] **Step 3: Verify pytest discovers fixtures**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -m pytest tests/ --collect-only 2>&1 | head -5`
Expected: shows `<Module ...>` or "no tests ran" (no errors)

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "Add tests/conftest.py — shared fixtures for test suite"
```

---

### Task 16: Create `tests/test_features.py`

**Files:**
- Create: `tests/test_features.py`

- [ ] **Step 1: Create `tests/test_features.py`**

```python
"""Tests for ML feature extraction."""

import numpy as np
import pytest

from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES, FEATURE_NAMES, validate_features
from ml.features import FeatureEngineering


class TestFeatureSchema:
    def test_schema_has_18_features(self):
        assert NUM_FEATURES == 18

    def test_feature_names_match_schema(self):
        assert len(FEATURE_NAMES) == NUM_FEATURES
        for i, f in enumerate(FEATURE_SCHEMA):
            assert FEATURE_NAMES[i] == f["name"]

    def test_all_defaults_within_range(self):
        for f in FEATURE_SCHEMA:
            lo, hi = f["range"]
            assert lo <= f["default"] <= hi, (
                f"{f['name']}: default {f['default']} not in range {f['range']}"
            )

    def test_validate_features_correct_length(self):
        features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
        assert validate_features(features) is True

    def test_validate_features_wrong_length(self):
        features = np.full(10, 0.5, dtype=np.float32)
        with pytest.raises(AssertionError, match="Expected 18"):
            validate_features(features)

    def test_validate_features_rejects_nan(self):
        features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
        features[0] = np.nan
        with pytest.raises(AssertionError, match="NaN"):
            validate_features(features)

    def test_validate_features_rejects_inf(self):
        features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
        features[5] = np.inf
        with pytest.raises(AssertionError, match="Inf"):
            validate_features(features)


class TestFeatureEngineering:
    def test_parse_last_6_slash_format(self):
        result = FeatureEngineering.parse_last_6("1/3/10/2/5")
        assert result == [1, 3, 10, 2, 5]

    def test_parse_last_6_legacy_format(self):
        result = FeatureEngineering.parse_last_6("13425")
        assert result == [1, 3, 4, 2, 5]

    def test_parse_last_6_empty(self):
        result = FeatureEngineering.parse_last_6("")
        assert result == []

    def test_form_score_all_wins(self):
        score = FeatureEngineering.calculate_form_score("1/1/1")
        assert score == 100.0

    def test_form_score_empty(self):
        score = FeatureEngineering.calculate_form_score("")
        assert score == 50.0

    def test_draw_advantage_ideal_sprint(self):
        # Draw 4 is ideal for <= 1000m
        score = FeatureEngineering.calculate_draw_advantage(4, 1000)
        assert score == 1.0

    def test_draw_advantage_wide(self):
        score = FeatureEngineering.calculate_draw_advantage(14, 1200)
        assert score < 0.5

    def test_extract_features_returns_18(self, sample_horse_data):
        features = FeatureEngineering.extract_features(
            sample_horse_data, [], "Sha Tin"
        )
        assert len(features) == NUM_FEATURES

    def test_extract_features_no_nan(self, sample_horse_data):
        features = FeatureEngineering.extract_features(
            sample_horse_data, [], "Sha Tin"
        )
        assert not np.any(np.isnan(features))

    def test_extract_features_no_inf(self, sample_horse_data):
        features = FeatureEngineering.extract_features(
            sample_horse_data, [], "Sha Tin"
        )
        assert not np.any(np.isinf(features))

    def test_extract_features_within_ranges(self, sample_horse_data):
        features = FeatureEngineering.extract_features(
            sample_horse_data, [], "Sha Tin"
        )
        for i, f in enumerate(FEATURE_SCHEMA):
            lo, hi = f["range"]
            assert lo - 0.01 <= features[i] <= hi + 0.01, (
                f"Feature {f['name']} = {features[i]} outside range {f['range']}"
            )

    def test_extract_features_minimal_data(self):
        """Minimal horse data should still produce 18 features with defaults."""
        minimal = {"name": "Unknown", "distance": 1200, "field_size": 10}
        features = FeatureEngineering.extract_features(minimal, [], "")
        assert len(features) == NUM_FEATURES
        assert not np.any(np.isnan(features))
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -m pytest tests/test_features.py -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "Add tests/test_features.py — feature schema and extraction tests"
```

---

### Task 17: Create `tests/test_scoring.py`

**Files:**
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Create `tests/test_scoring.py`**

```python
"""Tests for score combination and probability calibration."""

import pytest

from ml.model import create_combined_prediction
import config


class TestCombinedPrediction:
    def test_75_25_blend(self):
        """Rule 60/100 + ML 0.20 → 0.60*0.75 + 0.20*0.25 = 0.50"""
        prob, conf, breakdown = create_combined_prediction(
            rule_score=60.0,
            rule_confidence="MEDIUM",
            ml_prob=0.20,
            ml_confidence="MEDIUM",
            ml_available=True,
        )
        expected = 60.0 / 100 * config.RULE_WEIGHT + 0.20 * config.ML_WEIGHT
        assert abs(prob - expected) < 0.001

    def test_ml_unavailable_uses_rule_only(self):
        """When ML unavailable, weight should be 100% rule."""
        prob, conf, breakdown = create_combined_prediction(
            rule_score=80.0,
            rule_confidence="HIGH",
            ml_prob=0.50,
            ml_confidence="HIGH",
            ml_available=False,
        )
        assert abs(prob - 0.80) < 0.001
        assert breakdown['weights']['ml'] == 0.0

    def test_confidence_high_high(self):
        _, conf, _ = create_combined_prediction(
            rule_score=80, rule_confidence="HIGH",
            ml_prob=0.3, ml_confidence="HIGH", ml_available=True,
        )
        assert conf == "VERY_HIGH"

    def test_confidence_low_low(self):
        _, conf, _ = create_combined_prediction(
            rule_score=30, rule_confidence="LOW",
            ml_prob=0.05, ml_confidence="LOW", ml_available=True,
        )
        assert conf == "LOW"

    def test_breakdown_structure(self):
        _, _, breakdown = create_combined_prediction(
            rule_score=50, rule_confidence="MEDIUM",
            ml_prob=0.1, ml_confidence="LOW", ml_available=True,
        )
        assert "rule_based" in breakdown
        assert "ml_based" in breakdown
        assert "final_probability" in breakdown
        assert "weights" in breakdown
        assert breakdown["rule_based"]["score"] == 50


class TestWeightedScore:
    def test_compute_weighted_score_basic(self):
        from core.scorer import compute_weighted_score

        factors = {
            "rating": 80.0,
            "form": 70.0,
            "odds": 60.0,
            "jockey": 75.0,
            "trainer": 65.0,
            "draw": 50.0,
        }
        score = compute_weighted_score(factors)
        assert 50 < score < 80  # should be a weighted average in this range

    def test_zero_factors_excluded(self):
        from core.scorer import compute_weighted_score

        # tips=0 should be excluded from weight normalization
        factors = {
            "rating": 80.0,
            "form": 70.0,
            "odds": 60.0,
            "jockey": 75.0,
            "trainer": 65.0,
            "draw": 50.0,
            "tips": 0.0,  # should be excluded
        }
        score = compute_weighted_score(factors)
        assert 50 < score < 80

    def test_always_included_factors_kept_even_at_zero(self):
        """Class_movement, gear, weight_carried are always included."""
        from core.scorer import compute_weighted_score

        factors = {
            "rating": 80.0,
            "form": 70.0,
            "odds": 60.0,
            "jockey": 75.0,
            "trainer": 65.0,
            "draw": 50.0,
            "class_movement": 0.0,  # always included
            "gear": 0.0,  # always included
            "weight_carried": 0.0,  # always included
        }
        # These zeroes should pull the score down
        score = compute_weighted_score(factors)
        assert score < 70
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -m pytest tests/test_scoring.py -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_scoring.py
git commit -m "Add tests/test_scoring.py — score combination and calibration tests"
```

---

### Task 18: Create `tests/test_validation.py`

**Files:**
- Create: `tests/test_validation.py`

- [ ] **Step 1: Create `tests/test_validation.py`**

```python
"""Tests for HKJC data validation."""

import pytest
from unittest.mock import MagicMock

from data.validation import validate_horse_entry, validate_race_card


def _make_horse(number=1, name="Test Horse", draw=3, rating=65.0,
                jockey="Z Purton", trainer="J Size"):
    """Create a mock HorseEntry."""
    horse = MagicMock()
    horse.number = number
    horse.name = name
    horse.draw = draw
    horse.rating = rating
    horse.jockey = jockey
    horse.trainer = trainer
    return horse


def _make_card(race_number=1, distance="1200", horses=None):
    """Create a mock RaceCard."""
    card = MagicMock()
    card.race_number = race_number
    card.distance = distance
    card.horses = horses or [_make_horse(i) for i in range(1, 11)]
    return card


class TestValidateHorseEntry:
    def test_valid_entry_no_warnings(self):
        horse = _make_horse()
        warnings = validate_horse_entry(horse)
        assert warnings == []

    def test_missing_name(self):
        horse = _make_horse(name="")
        warnings = validate_horse_entry(horse)
        assert any("missing" in w.lower() or "short" in w.lower() for w in warnings)

    def test_short_name(self):
        horse = _make_horse(name="A")
        warnings = validate_horse_entry(horse)
        assert len(warnings) > 0

    def test_draw_out_of_range(self):
        horse = _make_horse(draw=20)
        warnings = validate_horse_entry(horse)
        assert any("draw" in w.lower() for w in warnings)

    def test_rating_too_high(self):
        horse = _make_horse(rating=200)
        warnings = validate_horse_entry(horse)
        assert any("rating" in w.lower() for w in warnings)

    def test_valid_rating_zero(self):
        """Rating of 0 is valid (unrated horse)."""
        horse = _make_horse(rating=0)
        warnings = validate_horse_entry(horse)
        assert not any("rating" in w.lower() for w in warnings)


class TestValidateRaceCard:
    def test_valid_card_no_warnings(self):
        card = _make_card()
        warnings = validate_race_card(card)
        assert warnings == []

    def test_too_few_horses(self):
        card = _make_card(horses=[_make_horse(1)])
        warnings = validate_race_card(card)
        assert any("only" in w.lower() or "horses" in w.lower() for w in warnings)

    def test_empty_card(self):
        card = _make_card(horses=[])
        warnings = validate_race_card(card)
        assert len(warnings) > 0

    def test_suspicious_distance(self):
        card = _make_card(distance="500")
        warnings = validate_race_card(card)
        assert any("distance" in w.lower() for w in warnings)

    def test_normal_distance(self):
        card = _make_card(distance="1600")
        warnings = validate_race_card(card)
        assert not any("distance" in w.lower() for w in warnings)

    def test_too_many_horses(self):
        horses = [_make_horse(i) for i in range(1, 20)]
        card = _make_card(horses=horses)
        warnings = validate_race_card(card)
        assert any("14" in w for w in warnings)
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -m pytest tests/test_validation.py -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_validation.py
git commit -m "Add tests/test_validation.py — HKJC data validation tests"
```

---

### Task 19: Create `tests/test_data_integrity.py`

**Files:**
- Create: `tests/test_data_integrity.py`

- [ ] **Step 1: Create `tests/test_data_integrity.py`**

```python
"""Tests for data integrity — feature vectors, model pickle, training labels."""

import pickle
from pathlib import Path

import numpy as np
import pytest

import config
from ml.feature_schema import NUM_FEATURES, FEATURE_NAMES
from ml.training import generate_synthetic_training_data


class TestSyntheticData:
    def test_shape_is_correct(self):
        X, y = generate_synthetic_training_data(200)
        assert X.shape == (200, NUM_FEATURES)
        assert y.shape == (200,)

    def test_no_nan(self):
        X, y = generate_synthetic_training_data(200)
        assert not np.any(np.isnan(X))

    def test_no_inf(self):
        X, y = generate_synthetic_training_data(200)
        assert not np.any(np.isinf(X))

    def test_no_constant_features(self):
        """Each feature should have std > 0.01 — constant features are wasted."""
        X, _ = generate_synthetic_training_data(1000)
        for i in range(NUM_FEATURES):
            std = np.std(X[:, i])
            assert std > 0.01, (
                f"Feature {FEATURE_NAMES[i]} (idx {i}) is near-constant: std={std:.6f}"
            )

    def test_realistic_win_rate(self):
        """Win rate should be roughly 5-15% (realistic HK field)."""
        _, y = generate_synthetic_training_data(5000)
        win_rate = y.mean()
        assert 0.04 < win_rate < 0.16, f"Win rate {win_rate:.2%} outside expected 5-15%"


class TestModelPickle:
    def test_model_file_has_expected_keys(self):
        """If a trained model exists, verify its pickle structure."""
        model_path = config.MODEL_PATH
        if not model_path.exists():
            pytest.skip("No trained model found")

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        assert 'model' in data, "Pickle missing 'model' key"
        assert 'version' in data, "Pickle missing 'version' key"
        assert 'feature_importance' in data, "Pickle missing 'feature_importance' key"
        assert 'trained_date' in data, "Pickle missing 'trained_date' key"

    def test_feature_importance_matches_schema(self):
        """Feature importance keys should match FEATURE_NAMES."""
        model_path = config.MODEL_PATH
        if not model_path.exists():
            pytest.skip("No trained model found")

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        fi = data.get('feature_importance', {})
        if fi:
            for name in FEATURE_NAMES:
                assert name in fi, f"Feature '{name}' missing from importance dict"


class TestTrainedModel:
    def test_predict_returns_probability(self, trained_model):
        features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
        prob, confidence = trained_model.predict(features, field_size=12)
        assert 0.0 <= prob <= 1.0
        assert isinstance(confidence, str)

    def test_predict_untrained_returns_base_rate(self):
        from ml.model import HKJCRacingMLModel
        model = HKJCRacingMLModel.__new__(HKJCRacingMLModel)
        model.model = None
        model.is_trained = False
        features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
        prob, confidence = model.predict(features, field_size=10)
        assert abs(prob - 0.1) < 0.001  # 1/10
        assert confidence == "MODEL_NOT_TRAINED"
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -m pytest tests/test_data_integrity.py -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_data_integrity.py
git commit -m "Add tests/test_data_integrity.py — data integrity and model tests"
```

---

### Task 20: Final Verification — Full Test Suite + End-to-End

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -m pytest tests/ -v --tb=short`
Expected: all tests PASS

- [ ] **Step 2: Verify end-to-end prediction still works**

Run: `cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer && python3 -c "
from core.analyzer import CombinedRaceAnalyzer, convert_race
from core.fetcher import RaceCard, HorseEntry

# Build a minimal race card
horses = [HorseEntry(number=i, name=f'Horse{i}', draw=i, jockey=f'J{i}', trainer=f'T{i}', weight='126', rating=50+i*3, win_odds=3+i*2, last_6='3/4/5/2/1/3') for i in range(1, 6)]
rc = RaceCard(race_number=1, venue='Sha Tin', date='2026/04/17', distance='1200', class_type='Class 3', going='Good', track='Turf', course='A', horses=horses)

analyzer = CombinedRaceAnalyzer()
race = convert_race(rc)
analyses = analyzer.analyze_race(race)
print(f'Analyzed {len(analyses)} horses')
for a in analyses[:3]:
    print(f'  #{a.horse.number} {a.horse.name}: prob={a.combined_probability:.3f}, bet={a.bet_recommendation}')
print('End-to-end OK')
"`
Expected: prints analysis results and "End-to-end OK"

- [ ] **Step 3: Verify imports across all entry points**

Run:
```bash
cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer
python3 -c "from core.analyzer import CombinedRaceAnalyzer" && echo "core.analyzer OK"
python3 -c "from ml.model import HKJCRacingMLModel" && echo "ml.model OK"
python3 -c "from ml.features import FeatureEngineering" && echo "ml.features OK"
python3 -c "from ml.training import build_training_data_from_db" && echo "ml.training OK"
python3 -c "from ml.experiments import get_experiments" && echo "ml.experiments OK"
python3 -c "from core.scorer import compute_weighted_score" && echo "core.scorer OK"
python3 -c "from core.recommender import assign_bet_recommendations" && echo "core.recommender OK"
python3 -c "from data.validation import validate_race_card" && echo "data.validation OK"
python3 -c "from ml_predictor import ML_AVAILABLE" && echo "ml_predictor shim OK"
python3 -c "from hkjc_racing_analyzer import CombinedRaceAnalyzer" && echo "analyzer shim OK"
```
Expected: all print "OK"

- [ ] **Step 4: Final commit with all changes**

If any files were missed, stage and commit them:

```bash
git status
git add -A  # only if all changes are intentional
git commit -m "Architecture refactor complete — all tests passing"
```
