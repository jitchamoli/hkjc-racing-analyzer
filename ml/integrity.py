"""F-05: pre-bet integrity check for the production model pickle.

Refuses to let predictions run against a synthetic-trained, legacy, or
stale model. Called at prediction entry points. All checks are hard
failures — no warnings — because betting on a compromised model is
strictly worse than skipping the race.
"""
from datetime import datetime, timedelta
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional


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

    # Normalize to naive datetime for comparison. train.py writes
    # datetime.now().isoformat() which is naive; defend against future
    # tz-aware writers by stripping tzinfo on read.
    if trained.tzinfo is not None:
        trained = trained.replace(tzinfo=None)
    age = datetime.now() - trained
    if age > timedelta(days=MAX_AGE_DAYS):
        raise IntegrityError(
            f"Integrity check failed: trained_date={trained_str} "
            f"is {age.days} days old (> MAX_AGE_DAYS={MAX_AGE_DAYS}). "
            f"Model is stale. Retrain via `python3 -m scripts.train`."
        )


def _read_verified_pickle(path: Path, label: str) -> Dict[str, Any]:
    """Load a pickle only after its .sha256 sidecar matches."""
    if not path.exists():
        raise IntegrityError(f"{label} missing: {path}")

    sha_path = path.with_suffix('.sha256')
    if not sha_path.exists():
        raise IntegrityError(f"{label} checksum sidecar missing: {sha_path}")

    raw = path.read_bytes()
    expected = sha_path.read_text().strip().split()[0].lower()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise IntegrityError(
            f"{label} checksum mismatch: expected {expected[:16]}..., "
            f"got {actual[:16]}..."
        )

    try:
        payload = pickle.loads(raw)
    except Exception as e:
        raise IntegrityError(f"{label} pickle could not be loaded: {e}") from e

    if not isinstance(payload, dict):
        raise IntegrityError(f"{label} pickle has invalid payload type: {type(payload).__name__}")
    return payload


def _race_key_sort_tuple(key: Any) -> tuple:
    # Race keys are 'YYYY/MM/DD-N' (training) or 'YYYY/MM/DD-RN' (split_rows_by_race).
    # Lexicographic compare breaks intra-day: '2024/05/29-10' < '2024/05/29-9'.
    s = str(key)
    idx = s.rfind('-')
    if idx < 0:
        return (s, 0)
    num = s[idx + 1:].lstrip('Rr')
    try:
        return (s[:idx], int(num))
    except ValueError:
        return (s[:idx], 0)


def check_grouped_fold_metadata(fold_metadata: Any) -> None:
    """Require a race-grouped temporal split in the saved production artifact."""
    if not isinstance(fold_metadata, dict):
        raise IntegrityError(
            "Integrity check failed: fold_metadata missing from model artifact. "
            "Retrain with grouped folds before dashboard generation."
        )

    n_groups = fold_metadata.get('n_groups')
    folds = fold_metadata.get('folds')
    if not isinstance(n_groups, int) or n_groups <= 0:
        raise IntegrityError("Integrity check failed: fold_metadata.n_groups is invalid.")
    if not isinstance(folds, dict):
        raise IntegrityError("Integrity check failed: fold_metadata.folds is missing.")

    required = ('train', 'val', 'calib', 'test')
    total_groups = 0
    previous_last = None
    for name in required:
        meta = folds.get(name)
        if not isinstance(meta, dict):
            raise IntegrityError(f"Integrity check failed: fold_metadata missing {name!r} fold.")
        n_rows = meta.get('n_rows')
        fold_groups = meta.get('n_groups')
        first_group = meta.get('first_group')
        last_group = meta.get('last_group')
        if not isinstance(n_rows, int) or n_rows <= 0:
            raise IntegrityError(f"Integrity check failed: {name} fold has invalid n_rows.")
        if not isinstance(fold_groups, int) or fold_groups <= 0:
            raise IntegrityError(f"Integrity check failed: {name} fold has invalid n_groups.")
        if not first_group or not last_group:
            raise IntegrityError(f"Integrity check failed: {name} fold missing group bounds.")
        first_key = _race_key_sort_tuple(first_group)
        last_key = _race_key_sort_tuple(last_group)
        if first_key > last_key:
            raise IntegrityError(f"Integrity check failed: {name} fold group bounds are reversed.")
        if previous_last is not None and first_key <= previous_last:
            raise IntegrityError(
                f"Integrity check failed: grouped folds overlap or are not temporal "
                f"at {name} fold boundary."
            )
        total_groups += fold_groups
        previous_last = last_key

    if total_groups != n_groups:
        raise IntegrityError(
            f"Integrity check failed: fold group count mismatch "
            f"({total_groups} fold groups != n_groups {n_groups})."
        )


def check_production_model_artifact(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Validate the production HK model artifact used by the dashboard.

    This is stricter than check_model_integrity(): it verifies the signed file,
    feature schema, required model components, and grouped-fold metadata.
    """
    import config
    from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES

    path = Path(model_path) if model_path is not None else config.MODEL_PATH
    payload = _read_verified_pickle(path, "production model")

    for key in ('model', 'place_model', 'calibrator', 'place_calibrator'):
        if payload.get(key) is None:
            raise IntegrityError(f"Integrity check failed: production model missing {key}.")

    saved_n = payload.get('n_features')
    if saved_n != NUM_FEATURES:
        raise IntegrityError(
            f"Integrity check failed: model n_features={saved_n}, schema has {NUM_FEATURES}."
        )

    saved_names = payload.get('feature_names')
    if list(saved_names or []) != list(FEATURE_NAMES):
        raise IntegrityError(
            "Integrity check failed: production model feature_names do not match "
            "the current schema."
        )

    metadata = {
        'n_samples': payload.get('n_samples'),
        'trained_date': payload.get('trained_date'),
        'n_features': payload.get('n_features'),
        'version': payload.get('version'),
    }
    check_model_integrity(metadata)
    check_grouped_fold_metadata(payload.get('fold_metadata'))
    return payload


def check_blend_calibrator_artifact(blend_path: Optional[Path] = None) -> Dict[str, Any]:
    """Validate the post-blend calibrator artifact used by core.scorer."""
    import config

    default_path = config.MODEL_DIR / 'blend_calibrator.pkl'
    path = Path(blend_path) if blend_path is not None else default_path
    payload = _read_verified_pickle(path, "blend calibrator")

    calibrator = payload.get('calibrator')
    if calibrator is None or not hasattr(calibrator, 'predict'):
        raise IntegrityError("Integrity check failed: blend calibrator missing predict().")

    metadata = payload.get('metadata')
    if not isinstance(metadata, dict):
        raise IntegrityError("Integrity check failed: blend calibrator metadata missing.")

    check_model_integrity({
        'n_samples': metadata.get('n_samples'),
        'trained_date': metadata.get('trained_date'),
        'n_features': None,
        'version': metadata.get('version'),
    })

    n_winners = metadata.get('n_winners')
    base_rate = metadata.get('base_rate')
    if not isinstance(n_winners, int) or n_winners <= 0:
        raise IntegrityError("Integrity check failed: blend calibrator n_winners is invalid.")
    if not isinstance(base_rate, (int, float)) or not 0 < float(base_rate) < 1:
        raise IntegrityError("Integrity check failed: blend calibrator base_rate is invalid.")

    return payload


def check_dashboard_ml_integrity(
    model_path: Optional[Path] = None,
    blend_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fail-closed preflight for race-day dashboard generation."""
    model_payload = check_production_model_artifact(model_path)
    blend_payload = check_blend_calibrator_artifact(blend_path)
    return {
        'model_version': model_payload.get('version'),
        'model_n_samples': model_payload.get('n_samples'),
        'model_fold_groups': (model_payload.get('fold_metadata') or {}).get('n_groups'),
        'blend_version': (blend_payload.get('metadata') or {}).get('version'),
        'blend_n_samples': (blend_payload.get('metadata') or {}).get('n_samples'),
    }
