#!/usr/bin/env python3
"""XGBoost model for HKJC race prediction — train, predict, save, load."""

import hashlib
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
    from sklearn.linear_model import LogisticRegression
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
        self.place_model = None
        self.calibrator = None  # Platt scaling (LogisticRegression)
        self.is_trained = False
        self.model_version = "1.0"
        self.feature_importance = {}

        if ML_AVAILABLE:
            self._load_model()

    def _checksum_path(self) -> Path:
        return self.MODEL_FILE.with_suffix('.sha256')

    def _load_model(self):
        if self.MODEL_FILE.exists():
            try:
                model_bytes = self.MODEL_FILE.read_bytes()

                # Verify integrity — refuse to load without valid checksum
                checksum_file = self._checksum_path()
                if not checksum_file.exists():
                    logger.error("Model checksum file missing — refusing to load unsigned model. "
                                 "Retrain with: python3 -m scripts.train")
                    return
                expected = checksum_file.read_text().strip()
                actual = hashlib.sha256(model_bytes).hexdigest()
                if actual != expected:
                    logger.error(f"Model checksum mismatch — file may be corrupted or tampered. "
                                 f"Expected {expected[:16]}..., got {actual[:16]}...")
                    return

                data = pickle.loads(model_bytes)

                # Verify feature count matches current schema
                saved_n = data.get('n_features')
                if saved_n is not None and saved_n != NUM_FEATURES:
                    logger.error(f"Model was trained on {saved_n} features but schema has {NUM_FEATURES}. "
                                 f"Retrain with: python3 -m scripts.train")
                    return

                self.model = data.get('model')
                self.place_model = data.get('place_model')
                self.calibrator = data.get('calibrator')
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
            'place_model': self.place_model,
            'calibrator': self.calibrator,
            'version': self.model_version,
            'n_features': NUM_FEATURES,
            'feature_importance': self.feature_importance,
            'trained_date': datetime.now().isoformat()
        }

        model_bytes = pickle.dumps(data)
        self.MODEL_FILE.write_bytes(model_bytes)
        self._checksum_path().write_text(hashlib.sha256(model_bytes).hexdigest())

        logger.info(f"Saved model version {self.model_version}")

    def train(self, features: np.ndarray, labels: np.ndarray,
              sample_weight: np.ndarray = None,
              temporal: bool = True,
              place_labels: np.ndarray = None) -> Dict:
        if not ML_AVAILABLE:
            return {'error': 'ML libraries not available'}

        logger.info(f"Training with {len(features)} samples "
                    f"({labels.sum():.0f} winners, {labels.mean()*100:.1f}% win rate)")

        # Temporal split: 60% train, 20% validation (early stopping), 20% test (final metrics)
        if temporal:
            cut_train = int(len(features) * 0.6)
            cut_val = int(len(features) * 0.8)
            X_train = features[:cut_train]
            X_val = features[cut_train:cut_val]
            X_test = features[cut_val:]
            y_train = labels[:cut_train]
            y_val = labels[cut_train:cut_val]
            y_test = labels[cut_val:]
            sw_train = sample_weight[:cut_train] if sample_weight is not None else None
            logger.info(f"Temporal split: train={cut_train}, val={cut_val-cut_train}, test={len(features)-cut_val}")
        else:
            stratify_arg = labels if labels.sum() >= 10 else None
            sw_train = None
            if sample_weight is not None:
                X_trainval, X_test, y_trainval, y_test, sw_trainval, _ = train_test_split(
                    features, labels, sample_weight,
                    test_size=0.2, random_state=42, stratify=stratify_arg
                )
                stratify_tv = y_trainval if y_trainval.sum() >= 10 else None
                X_train, X_val, y_train, y_val, sw_train, _ = train_test_split(
                    X_trainval, y_trainval, sw_trainval,
                    test_size=0.25, random_state=42, stratify=stratify_tv
                )
            else:
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42,
                    stratify=stratify_arg
                )
                stratify_tv = y_trainval if y_trainval.sum() >= 10 else None
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval,
                    test_size=0.25, random_state=42, stratify=stratify_tv
                )

        n_pos = max(1, y_train.sum())
        n_neg = max(1, len(y_train) - n_pos)
        scale = n_neg / n_pos

        params = dict(config.XGBOOST_PARAMS)
        params['scale_pos_weight'] = scale

        # Early stopping on VALIDATION set (not test set)
        params['early_stopping_rounds'] = 20
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, sample_weight=sw_train,
                       eval_set=[(X_val, y_val)], verbose=False)

        # Final metrics on held-out TEST set (never seen during training or stopping)
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        brier = float(np.mean((y_pred_proba - y_test) ** 2))

        self.feature_importance = dict(zip(
            FEATURE_NAMES,
            self.model.feature_importances_.tolist()
        ))

        # Platt scaling: fit on TEST predictions (not validation, which was used for
        # early stopping — fitting calibrator on val would overfit to the same set)
        test_proba_raw = self.model.predict_proba(X_test)[:, 1]
        self.calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        self.calibrator.fit(test_proba_raw.reshape(-1, 1), y_test)
        logger.info(f"Platt calibrator fitted on {len(y_test)} test samples")

        # Recalculate test metrics with calibrated probabilities
        y_pred_proba = self.calibrator.predict_proba(test_proba_raw.reshape(-1, 1))[:, 1]
        logloss = log_loss(y_test, y_pred_proba)
        brier = float(np.mean((y_pred_proba - y_test) ** 2))

        # Train place model (is_top3) if labels provided
        place_accuracy = None
        if place_labels is not None:
            pl_train = place_labels[:cut_train] if temporal else place_labels[:len(y_train)]
            pl_val = place_labels[cut_train:cut_val] if temporal else place_labels[len(y_train):len(y_train)+len(y_val)]
            pl_test = place_labels[cut_val:] if temporal else place_labels[-len(y_test):]

            pl_pos = max(1, pl_train.sum())
            pl_neg = max(1, len(pl_train) - pl_pos)
            pl_params = dict(config.XGBOOST_PARAMS)
            pl_params['scale_pos_weight'] = pl_neg / pl_pos
            pl_params['early_stopping_rounds'] = 20
            self.place_model = xgb.XGBClassifier(**pl_params)
            self.place_model.fit(X_train, pl_train, sample_weight=sw_train,
                                eval_set=[(X_val, pl_val)], verbose=False)
            pl_pred = self.place_model.predict(X_test)
            place_accuracy = accuracy_score(pl_test, pl_pred)
            logger.info(f"Place model: accuracy={place_accuracy:.2%} "
                        f"({pl_train.sum():.0f} top3 in train, {pl_test.sum():.0f} in test)")

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
            'brier_score': brier,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version
        }
        if place_accuracy is not None:
            result['place_accuracy'] = place_accuracy

        # Compute feature correlation matrix and flag high correlations
        high_corrs = {}
        corr_matrix = np.corrcoef(features, rowvar=False)
        for i in range(len(FEATURE_NAMES)):
            for j in range(i + 1, len(FEATURE_NAMES)):
                r = abs(corr_matrix[i, j])
                if r > 0.7:
                    pair = f"{FEATURE_NAMES[i]} × {FEATURE_NAMES[j]}"
                    high_corrs[pair] = round(float(r), 3)
                    logger.warning(f"High correlation: {pair} = {r:.3f}")

        log_experiment(result, FEATURE_NAMES, n_samples=len(features),
                       feature_correlations=high_corrs if high_corrs else None)

        return result

    def predict(self, features: np.ndarray, field_size: int = 12) -> Tuple[float, str]:
        if not self.is_trained or self.model is None:
            base_prob = 1.0 / max(field_size, 1)
            return base_prob, "MODEL_NOT_TRAINED"

        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        win_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Apply Platt scaling if calibrator is available
        if self.calibrator is not None:
            win_prob = float(self.calibrator.predict_proba(
                np.array([[win_prob]]))[0, 1])

        if win_prob > 0.30:
            confidence = "HIGH"
        elif win_prob > 0.15:
            confidence = "MEDIUM"
        elif win_prob > 0.08:
            confidence = "LOW"
        else:
            confidence = "VERY_LOW"

        return win_prob, confidence

    def predict_place(self, features: np.ndarray, field_size: int = 12) -> float:
        """Predict place (top-3) probability using dedicated place model."""
        if self.place_model is not None:
            proba = self.place_model.predict_proba(features.reshape(1, -1))[0]
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        # Fallback: estimate from win probability
        win_prob, _ = self.predict(features, field_size)
        return min(0.85, win_prob * 2.5)


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
    # Convert rule_score (0-100) to a probability-like value using sigmoid
    # rule_score=50 → 0.5, rule_score=80 → ~0.73, rule_score=20 → ~0.27
    import math
    rule_prob_raw = 1.0 / (1.0 + math.exp(-0.06 * (rule_score - 50)))
    rule_prob = rule_prob_raw * weights['rule']

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
        import sqlite3
        from ml.features import FeatureEngineering
        from ml.training import build_training_data_from_db, generate_synthetic_training_data

        self.ml_model = HKJCRacingMLModel()
        self.feature_eng = FeatureEngineering()

        # Shared DB connection for feature extraction (avoids per-horse connection churn)
        self._conn = None
        if config.DB_PATH.exists():
            self._conn = sqlite3.connect(str(config.DB_PATH))
            self._conn.row_factory = sqlite3.Row

        if ML_AVAILABLE and not self.ml_model.is_trained:
            X_real, y_real, sw_real, y_place = build_training_data_from_db()
            if X_real is not None and len(X_real) >= config.MIN_REAL_SAMPLES:
                logger.info(f"Training ML model with {len(X_real)} REAL race results...")
                result = self.ml_model.train(X_real, y_real, sample_weight=sw_real,
                                            place_labels=y_place)
                logger.info(f"Model trained on REAL data: accuracy={result['accuracy']:.2%}")
            else:
                logger.info("Insufficient real data — training ML model with synthetic data...")
                X, y = generate_synthetic_training_data()
                result = self.ml_model.train(X, y)
                logger.info(f"Model trained on synthetic data: accuracy={result['accuracy']:.2%}")

    def predict(self, horse_data: dict, all_results, venue: str) -> MLPrediction:
        features = self.feature_eng.extract_features(horse_data, all_results, venue,
                                                     conn=self._conn)

        field_size = horse_data.get('field_size', 12)
        ml_prob, ml_confidence = self.ml_model.predict(features, field_size=field_size)

        return MLPrediction(
            horse_number=horse_data.get('number', 0),
            horse_name=horse_data.get('name', ''),
            win_probability=ml_prob,
            place_probability=self.ml_model.predict_place(features, field_size=field_size),
            ml_confidence=ml_confidence,
            feature_importance={
                name: float(features[i]) * self.ml_model.feature_importance.get(name, 0.0)
                for i, name in enumerate(FEATURE_NAMES)
            },
            model_version=self.ml_model.model_version
        )
