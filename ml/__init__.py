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
