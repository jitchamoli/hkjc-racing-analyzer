"""Shared feature schema — single source of truth for train and predict.

Both extract_features() and build_training_data_from_db() MUST build
feature vectors in FEATURE_SCHEMA order. This prevents train/predict mismatch.

20 features — all use data available BEFORE the race (no leakage).
No direct market odds features — ML must provide independent signal.

Wave 2c drops:
- field_rel_speed (was feature 17): r=0.718 with recent_form, r=0.710 with
  last_start_margin. adjusted_speed already carries the speed-family signal.
- days_since_last_win (was feature 19/20): r=0.741 with recent_form. recent_form
  already captures recency indirectly via finish positions in last 6 races.
"""

import numpy as np

FEATURE_SCHEMA = [
    {"name": "prior_win_rate", "default": 0.5, "range": (0.0, 1.0)},       # 1
    {"name": "recent_form", "default": 0.5, "range": (0.0, 1.0)},          # 2
    {"name": "draw_advantage", "default": 0.6, "range": (0.0, 1.0)},       # 3
    {"name": "jockey_win_rate", "default": 0.08, "range": (0.0, 0.6)},     # 4
    {"name": "trainer_win_rate", "default": 0.06, "range": (0.0, 0.5)},    # 5
    {"name": "field_size", "default": 0.857, "range": (0.0, 1.5)},         # 6
    {"name": "field_rel_draw", "default": 0.5, "range": (0.0, 1.0)},       # 7
    {"name": "prior_pace", "default": 0.5, "range": (0.0, 1.0)},           # 8
    {"name": "adjusted_speed", "default": 0.5, "range": (0.0, 1.0)},       # 9  (Beyer-style: par-adjusted, weight-corrected)
    {"name": "cd_form_prior", "default": 0.5, "range": (0.0, 1.0)},        # 10
    {"name": "class_level", "default": 0.4, "range": (0.0, 1.0)},          # 11 (inverted: Class1=0.8, Class5=0.0)
    {"name": "freshness", "default": 0.5, "range": (0.0, 1.0)},            # 12
    {"name": "going_performance", "default": 0.5, "range": (0.0, 1.0)},    # 13 (merged going_pref+going_form_rate)
    {"name": "weight_vs_field", "default": 0.5, "range": (0.0, 1.0)},      # 14 (carried weight relative to field median)
    {"name": "jockey_horse_coop", "default": 0.5, "range": (0.0, 1.0)},    # 15
    {"name": "distance_pb_ratio", "default": 0.5, "range": (0.0, 1.0)},    # 16
    # field_rel_speed (was 17) dropped Wave 2c: r=0.718 with recent_form,
    # r=0.710 with last_start_margin. adjusted_speed covers the speed signal.
    {"name": "last_start_margin", "default": 0.5, "range": (0.0, 1.0)},    # 17
    {"name": "class_change", "default": 0.5, "range": (0.0, 1.0)},         # 18 (class drop=high, rise=low)
    # days_since_last_win (was 19) dropped Wave 2c: r=0.741 with recent_form.
    {"name": "horse_weight_change", "default": 0.5, "range": (0.0, 1.0)},  # 19 (body weight change from prior race)
    {"name": "speed_consistency", "default": 0.5, "range": (0.0, 1.0)},    # 20 (low std of adj speed = reliable)
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
