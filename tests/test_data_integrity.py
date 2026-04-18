"""Tests for data integrity -- feature vectors, model pickle, training labels."""

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
        """Each feature should have std > 0.01 -- constant features are wasted."""
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
