"""F-05 tests: integrity check refuses to predict against a bad pickle."""
import pytest

from ml.integrity import check_model_integrity, IntegrityError


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
