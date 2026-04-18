"""Core ML predictor — re-exports from ml/ package.

All ML logic has moved to the ml/ package:
  ml/feature_schema.py — shared feature definitions
  ml/features.py       — feature extraction
  ml/model.py          — XGBoost model
  ml/training.py       — training data builders
  ml/experiments.py    — experiment tracking
"""

from ml.feature_schema import (  # noqa: F401
    FEATURE_SCHEMA, FEATURE_NAMES, FEATURE_DEFAULTS,
    NUM_FEATURES, validate_features,
)
from ml.features import FeatureEngineering  # noqa: F401
from ml.model import (  # noqa: F401
    HKJCRacingMLModel, RaceResult, HorseFeatures, MLPrediction,
    CombinedRacingAnalyzer, create_combined_prediction, ML_AVAILABLE,
)
from ml.training import build_training_data_from_db, generate_synthetic_training_data  # noqa: F401
