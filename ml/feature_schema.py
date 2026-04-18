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
    {"name": "weight_carried_norm", "default": 0.65, "range": (0.0, 1.0)},
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
