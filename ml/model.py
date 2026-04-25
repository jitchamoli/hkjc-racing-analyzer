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
from scipy.special import logit

import config
from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
    from sklearn.isotonic import IsotonicRegression
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
    MIN_SAVE_SAMPLES = 1000  # _save_model() refuses below this — prevents synthetic pickle overwrite

    def __init__(self):
        self.model = None
        self.place_model = None
        self.calibrator = None  # Isotonic calibrator (Wave 2c: replaced Platt/LogisticRegression)
        self.is_trained = False
        self.model_version = "1.0"
        self.feature_importance = {}
        self._metadata = None

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
                self._last_train_n_samples = data.get('n_samples', None)
                self._metadata = {
                    'n_samples': data.get('n_samples'),
                    'trained_date': data.get('trained_date'),
                    'n_features': data.get('n_features'),
                    'version': data.get('version'),
                }
                self.is_trained = self.model is not None
                logger.info(f"Loaded model version {self.model_version}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def _save_model(self):
        # F-04 belt-and-suspenders: refuse to overwrite production pickle
        # if this training run used fewer than MIN_SAVE_SAMPLES. Prevents
        # synthetic/test data from silently clobbering the real-data pickle.
        n_samples = getattr(self, '_last_train_n_samples', None)
        if n_samples is not None and n_samples < self.MIN_SAVE_SAMPLES:
            logger.warning(
                f"_save_model() refusing to write: n_samples={n_samples} "
                f"< MIN_SAVE_SAMPLES={self.MIN_SAVE_SAMPLES}. "
                f"This looks like synthetic/test data. Pickle NOT overwritten."
            )
            return

        self.MODEL_DIR.mkdir(exist_ok=True)

        data = {
            'model': self.model,
            'place_model': self.place_model,
            'calibrator': self.calibrator,
            'version': self.model_version,
            'n_features': NUM_FEATURES,
            'n_samples': n_samples,  # NEW: persisted for F-05 integrity check
            'feature_importance': self.feature_importance,
            'trained_date': datetime.now().isoformat()
        }

        model_bytes = pickle.dumps(data)
        self.MODEL_FILE.write_bytes(model_bytes)
        self._checksum_path().write_text(hashlib.sha256(model_bytes).hexdigest())

        logger.info(f"Saved model version {self.model_version} (n_samples={n_samples})")

    def train(self, features: np.ndarray, labels: np.ndarray,
              sample_weight: np.ndarray = None,
              temporal: bool = True,
              place_labels: np.ndarray = None,
              save: bool = True) -> Dict:
        if not ML_AVAILABLE:
            return {'error': 'ML libraries not available'}

        # Record sample count on self so _save_model() can enforce MIN_SAVE_SAMPLES
        self._last_train_n_samples = len(features)

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

        # Isotonic calibrator: fit on VALIDATION predictions with the same recency
        # weights used for XGBoost training — otherwise the calibrator learns
        # a different prior than the model it's calibrating.
        # Wave 2c: switched from Platt (LogisticRegression on logits) to
        # IsotonicRegression to handle the non-linear mid-range miscalibration
        # seen in the [18-25%] bin (33 races, ~3% actual vs ~20% predicted).
        # Isotonic is non-parametric and can reshape the calibration curve freely.
        # Input: logit(val_proba) — isotonic operates on the raw logit domain
        # to spread inputs over a useful range (same as Platt pre-Wave-2c).
        val_proba = self.model.predict_proba(X_val)[:, 1]
        val_logit = logit(np.clip(val_proba, 1e-6, 1 - 1e-6))
        sw_val = sample_weight[cut_train:cut_val] if (sample_weight is not None and temporal) else None
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_logit.ravel(), y_val, sample_weight=sw_val)
        logger.info(f"Isotonic calibrator fitted on {len(y_val)} validation samples (logit-input)"
                    + (" (recency-weighted)" if sw_val is not None else ""))

        # Re-evaluate test metrics with calibrated probabilities (test is truly held-out)
        test_raw = self.model.predict_proba(X_test)[:, 1]
        test_logit = logit(np.clip(test_raw, 1e-6, 1 - 1e-6))
        y_pred_proba = self.calibrator.predict(test_logit.ravel())
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
        if save:
            self._save_model()
        else:
            logger.info(f"train(save=False) — skipping pickle write (n_samples={len(features)})")

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

        # Only log to the production experiment ledger when this is a real
        # training run that actually persisted (save=True AND samples are
        # above MIN_SAVE_SAMPLES). Tests with save=True but tiny samples
        # also tripped the ledger — the F-04 save-guard rejects the pickle
        # but log_experiment fired anyway. Mirror the same check here.
        if save and len(features) >= self.MIN_SAVE_SAMPLES:
            from ml.experiments import log_experiment
            log_experiment(result, FEATURE_NAMES, n_samples=len(features),
                           feature_correlations=high_corrs if high_corrs else None)

        return result

    def predict(self, features: np.ndarray, field_size: int = 12) -> Tuple[float, str]:
        if not self.is_trained or self.model is None:
            # F-17: never silently return 1/field_size — surface the failure loudly
            # so load errors aren't masked by plausible-looking probabilities.
            raise RuntimeError(
                "Model not trained — refuse to predict. "
                "Retrain via `python3 -m scripts.train`."
            )

        # F-05: integrity check — fail loudly if pickle is compromised.
        # This runs on every predict() call; cheap enough (dict lookups).
        if self._metadata is not None:
            from ml.integrity import check_model_integrity
            check_model_integrity(self._metadata)

        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        win_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Apply calibration if calibrator is available.
        # Calibrator was fit on logit(val_proba), so transform input the same way.
        # Supports both IsotonicRegression (Wave 2c+, uses .predict(1D)) and
        # legacy LogisticRegression pickles (uses .predict_proba(2D)).
        if self.calibrator is not None:
            win_logit = float(logit(np.clip(win_prob, 1e-6, 1 - 1e-6)))
            try:
                # IsotonicRegression: predict(1D array) → values directly
                win_prob = float(self.calibrator.predict(np.array([win_logit]))[0])
            except (ValueError, TypeError):
                # Legacy LogisticRegression: predict_proba(2D array) → proba
                win_prob = float(self.calibrator.predict_proba(
                    np.array([[win_logit]]))[0, 1])

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
            # F-05: integrity check mirrors predict() — place is also a prediction entry point.
            if self._metadata is not None:
                from ml.integrity import check_model_integrity
                check_model_integrity(self._metadata)
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
    ml_available: bool,
    field_size: int = 12,
) -> Tuple[float, str, Dict]:
    weights = {'rule': config.RULE_WEIGHT, 'ml': config.ML_WEIGHT}

    if not ml_available:
        weights = {'rule': 1.0, 'ml': 0.0}

    ml_prob_adjusted = ml_prob * weights['ml']
    # Anchor sigmoid to the field-size base rate so an "average" horse
    # (rule_score=50) outputs ~1/field_size win probability instead of 50%.
    # Without this anchor the rule engine was grossly uncalibrated.
    import math
    base_prob = 1.0 / max(field_size, 2)
    logit_base = math.log(base_prob / (1.0 - base_prob))
    rule_prob_raw = 1.0 / (1.0 + math.exp(-(0.06 * (rule_score - 50) + logit_base)))
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
            # Auto-train fallback: use the same 5y window the production trainer
            # uses by default (TRAINING_YEARS_BACK env var, default 5). v6.2 was
            # the regression where training on the full 49y dropped ROI ~30%; we
            # don't want this fallback path to silently reproduce that.
            import os
            yrs_str = os.environ.get('TRAINING_YEARS_BACK', '5')
            yrs = None if yrs_str.strip() == '0' else int(yrs_str)
            X_real, y_real, sw_real, y_place = build_training_data_from_db(years_back=yrs)
            if X_real is not None and len(X_real) >= config.MIN_REAL_SAMPLES:
                logger.info(f"Training ML model with {len(X_real)} REAL race results "
                            f"({'no cutoff' if yrs is None else f'{yrs}y window'})...")
                result = self.ml_model.train(X_real, y_real, sample_weight=sw_real,
                                            place_labels=y_place)
                logger.info(f"Model trained on REAL data: accuracy={result['accuracy']:.2%}")
            else:
                logger.info("Insufficient real data — training ML model with synthetic data...")
                X, y = generate_synthetic_training_data()
                # F-04: never save a synthetic-trained pickle to disk.
                result = self.ml_model.train(X, y, save=False)
                logger.info(f"Model trained on synthetic data (in-memory only, not persisted): "
                            f"accuracy={result['accuracy']:.2%}")

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
