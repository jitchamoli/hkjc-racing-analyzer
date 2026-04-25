"""Shared feature schema — single source of truth for train and predict.

Both extract_features() and build_training_data_from_db() MUST build
feature vectors in FEATURE_SCHEMA order. This prevents train/predict mismatch.

16 features — all use data available BEFORE the race (no leakage).
No direct market odds features — ML must provide independent signal.

Wave 2c drops:
- field_rel_speed: r=0.718 with recent_form, r=0.710 with last_start_margin.
- days_since_last_win: r=0.741 with recent_form.

v6.4 drops (2026-04-25, after sweep on 50K rows showed 0.0 importance):
- prior_pace: dominated by last_start_margin (16.9% importance)
- adjusted_speed: ditto — speed signal redundant with form signal
- distance_pb_ratio: ditto — finish-time signal redundant with form signal
- speed_consistency: ditto — std-of-speed becomes noise when speed itself is dead

The 4 dropped features were 100% redundant with last_start_margin in the v6.3
build. Brier on the 16-feature build matches the 20-feature build (statistically
indistinguishable; details in models/experiment_log.json).
"""

import numpy as np

FEATURE_SCHEMA = [
    {"name": "prior_win_rate", "default": 0.5, "range": (0.0, 1.0)},       # 1
    {"name": "recent_form", "default": 0.5, "range": (0.0, 1.0)},          # 2
    {"name": "draw_advantage", "default": 0.6, "range": (0.0, 1.0)},       # 3
    {"name": "jockey_win_rate", "default": 0.08, "range": (0.0, 0.6)},     # 4
    {"name": "trainer_win_rate", "default": 0.06, "range": (0.0, 0.5)},    # 5
    {"name": "field_size", "default": 0.857, "range": (0.0, 1.0)},         # 6 (clipped to [0,1])
    {"name": "field_rel_draw", "default": 0.5, "range": (0.0, 1.0)},       # 7 (clipped to [0,1])
    {"name": "cd_form_prior", "default": 0.5, "range": (0.0, 1.0)},        # 8
    {"name": "class_level", "default": 0.4, "range": (0.0, 1.0)},          # 9 (Group=1.0, Class1=0.8, Class5=0.0)
    {"name": "freshness", "default": 0.5, "range": (0.0, 1.0)},            # 10
    {"name": "going_performance", "default": 0.5, "range": (0.0, 1.0)},    # 11
    {"name": "weight_vs_field", "default": 0.5, "range": (0.0, 1.0)},      # 12
    {"name": "jockey_horse_coop", "default": 0.5, "range": (0.0, 1.0)},    # 13
    {"name": "last_start_margin", "default": 0.5, "range": (0.0, 1.0)},    # 14
    {"name": "class_change", "default": 0.5, "range": (0.0, 1.0)},         # 15
    {"name": "horse_weight_change", "default": 0.5, "range": (0.0, 1.0)},  # 16
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
