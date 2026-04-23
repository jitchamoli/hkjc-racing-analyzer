"""F-05 / F-17 tests: integrity check refuses to predict against a bad pickle;
untrained model raises instead of returning silent fallback."""
import numpy as np
import pytest

from ml.integrity import check_model_integrity, IntegrityError
from ml.model import HKJCRacingMLModel
from ml.feature_schema import NUM_FEATURES


def test_integrity_accepts_real_pickle():
    """A pickle with n_samples >= 1000 and recent trained_date must pass."""
    metadata = {
        'n_samples': 4403,
        'trained_date': '2026-04-23T10:00:00',
        'n_features': 23,
        'version': '20260423-100000',
    }
    check_model_integrity(metadata)  # no raise


def test_integrity_rejects_synthetic_n_samples():
    """n_samples < 1000 must raise IntegrityError."""
    metadata = {
        'n_samples': 500,
        'trained_date': '2026-04-23T01:29:55.559167',
        'n_features': 23,
        'version': '20260423-012955',
    }
    with pytest.raises(IntegrityError, match="n_samples"):
        check_model_integrity(metadata)


def test_integrity_rejects_missing_n_samples():
    """Missing n_samples (legacy pickle) must raise IntegrityError."""
    metadata = {
        'trained_date': '2026-04-23T01:29:55',
        'n_features': 23,
        'version': '20260423-012955',
    }
    with pytest.raises(IntegrityError, match="n_samples"):
        check_model_integrity(metadata)


def test_integrity_rejects_old_trained_date():
    """trained_date older than 90 days raises IntegrityError."""
    metadata = {
        'n_samples': 4403,
        'trained_date': '2025-01-01T00:00:00',
        'n_features': 23,
        'version': '20250101-000000',
    }
    with pytest.raises(IntegrityError, match="trained_date"):
        check_model_integrity(metadata)


def test_integrity_accepts_exactly_min_samples():
    """n_samples == MIN_SAMPLES (1000) must PASS — threshold is inclusive."""
    metadata = {
        'n_samples': 1000,  # exactly at threshold
        'trained_date': '2026-04-23T10:00:00',
        'n_features': 23,
        'version': '20260423-100000',
    }
    check_model_integrity(metadata)  # no raise


def test_integrity_rejects_just_below_min_samples():
    """n_samples == MIN_SAMPLES - 1 must FAIL."""
    metadata = {
        'n_samples': 999,
        'trained_date': '2026-04-23T10:00:00',
        'n_features': 23,
        'version': '20260423-100000',
    }
    with pytest.raises(IntegrityError, match="n_samples"):
        check_model_integrity(metadata)


# ---------------------------------------------------------------------------
# F-17: untrained model must raise, not silently return 1/field_size
# ---------------------------------------------------------------------------

def test_predict_raises_on_untrained_model():
    """F-17: HKJCRacingMLModel.predict() must raise RuntimeError when untrained.

    Prevents silent load failures from producing plausible-looking probabilities.
    """
    model = HKJCRacingMLModel.__new__(HKJCRacingMLModel)
    model.model = None
    model.place_model = None
    model.calibrator = None
    model.is_trained = False
    model._metadata = None
    model.feature_importance = {}
    model.model_version = None

    dummy_features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict(dummy_features, field_size=12)


def test_predict_place_raises_on_untrained_model():
    """F-17: predict_place() also raises when model is untrained (via fallback to predict())."""
    model = HKJCRacingMLModel.__new__(HKJCRacingMLModel)
    model.model = None
    model.place_model = None
    model.calibrator = None
    model.is_trained = False
    model._metadata = None
    model.feature_importance = {}
    model.model_version = None

    dummy_features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict_place(dummy_features, field_size=12)
