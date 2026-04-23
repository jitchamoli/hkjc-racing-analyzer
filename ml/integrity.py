"""F-05: pre-bet integrity check for the production model pickle.

Refuses to let predictions run against a synthetic-trained, legacy, or
stale model. Called at prediction entry points. All checks are hard
failures — no warnings — because betting on a compromised model is
strictly worse than skipping the race.
"""
from datetime import datetime, timedelta
from typing import Dict, Any


class IntegrityError(RuntimeError):
    """Raised when a model pickle fails integrity checks and predictions
    must not run against it."""


MIN_SAMPLES = 1000
"""Minimum training sample count. Synthetic fixtures use 500 — this
threshold catches them."""

MAX_AGE_DAYS = 90
"""Maximum age of trained_date before integrity check fails. Forces
periodic retraining and blocks abandoned/stale models."""


def check_model_integrity(metadata: Dict[str, Any]) -> None:
    """Validate a model pickle's metadata. Raises IntegrityError on failure.

    Args:
        metadata: The dict unpickled from models/hkjc_racing_model.pkl.
                  Expected keys: n_samples, trained_date, n_features, version.

    Checks:
        - n_samples is present and >= MIN_SAMPLES (rejects synthetic pickles)
        - trained_date is present, parseable, and within MAX_AGE_DAYS
    """
    n_samples = metadata.get('n_samples')
    if n_samples is None:
        raise IntegrityError(
            "Integrity check failed: n_samples missing from model metadata. "
            "Legacy pickle without sample-count metadata. "
            "Retrain via `python3 -m scripts.train` before running predictions."
        )
    if n_samples < MIN_SAMPLES:
        raise IntegrityError(
            f"Integrity check failed: n_samples={n_samples} "
            f"< MIN_SAMPLES={MIN_SAMPLES}. "
            f"This pickle looks synthetic-trained (the audit found a "
            f"500-sample pytest artifact in production). "
            f"Retrain via `python3 -m scripts.train` on real DB data."
        )

    trained_str = metadata.get('trained_date')
    if trained_str is None:
        raise IntegrityError(
            "Integrity check failed: trained_date missing from model metadata. "
            "Retrain via `python3 -m scripts.train`."
        )
    try:
        trained = datetime.fromisoformat(trained_str)
    except ValueError:
        raise IntegrityError(
            f"Integrity check failed: trained_date={trained_str!r} "
            f"is not parseable as ISO datetime. "
            f"Retrain via `python3 -m scripts.train`."
        )

    age = datetime.now() - trained
    if age > timedelta(days=MAX_AGE_DAYS):
        raise IntegrityError(
            f"Integrity check failed: trained_date={trained_str} "
            f"is {age.days} days old (> MAX_AGE_DAYS={MAX_AGE_DAYS}). "
            f"Model is stale. Retrain via `python3 -m scripts.train`."
        )
