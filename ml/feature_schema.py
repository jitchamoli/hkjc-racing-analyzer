"""Shared feature schema — single source of truth for train and predict.

Both extract_features() and build_training_data_from_db() MUST build
feature vectors in FEATURE_SCHEMA order. This prevents train/predict mismatch.

21 features — all use data available BEFORE the race (no leakage).
No direct market odds features — ML must provide independent signal.

Wave 2c drops:
- field_rel_speed: r=0.718 with recent_form, r=0.710 with last_start_margin.
- days_since_last_win: r=0.741 with recent_form.

v6.4 drops (2026-04-25, after sweep on 50K rows showed 0.0 importance):
- prior_pace: dominated by last_start_margin (16.9% importance)
- adjusted_speed: ditto — speed signal redundant with form signal
- distance_pb_ratio: ditto — finish-time signal redundant with form signal
- speed_consistency: ditto — std-of-speed becomes noise when speed itself is dead

v6.5 adds (2026-04-26): three "data-already-here" features sourced from the
Renavon backfill that were previously only used by the rule engine:
- horse_age: 99.8% DB coverage; younger horses (3-5y) systematically improve
- rating_change: 62.6% DB coverage; rating delta from previous race signals trend
- jockey_form_30d: short-window jockey strike rate captures hot streaks the
  all-time aggregate hides

v6.6 adds (2026-05-09): two high-ROI features identified in the round-2 audit:
- jt_pair_win_rate: jockey×trainer pairing strike rate. 305 pairs ≥50 joint
  runs; top pairs hit 22-28% wins vs 8.3% baseline. Stable-jockey signal.
- speed_fig_norm: par-time-normalised speed figure of the horse's last run,
  weight-adjusted (Benter +0.1s/lb above 120). Already in rule engine via
  analytics/speed_figures.py at 5% weight; promoted to ML schema for the
  independent 40% blend channel.

v6.8 adds (2026-05-14): three running-position proxy features (Path A) derived
from race_results.running_position (99.4% DB coverage, space-separated
position-at-each-call). All rank-normalized for distance independence.
- early_speed: 1st-call position / field_size (front-runners ~0.1, closers ~1.0)
- late_kick: change between 2nd-last call and finish, normalized (closing ≥0.5,
  fading <0.5). Captures finishing kick that last_start_margin misses.
- running_style: categorical front/stalker/closer based on avg call position
  in last race {0.0, 0.5, 1.0}. Pace-map fit signal for THIS race.
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
    {"name": "horse_age", "default": 0.5, "range": (0.0, 1.0)},            # 17 (3yo→1.0, 9yo→0.0)
    {"name": "rating_change", "default": 0.5, "range": (0.0, 1.0)},        # 18 (+10pts→1.0, -10→0.0)
    {"name": "jockey_form_30d", "default": 0.08, "range": (0.0, 0.6)},     # 19 (last 30-day strike rate)
    {"name": "jt_pair_win_rate", "default": 0.08, "range": (0.0, 0.5)},    # 20 (Bayesian-smoothed jockey×trainer)
    {"name": "speed_fig_norm", "default": 0.5, "range": (0.0, 1.0)},       # 21 (par-time normalised, last run)
    {"name": "early_speed", "default": 0.5, "range": (0.0, 1.0)},          # 22 (1st-call rank, front=0.0 closer=1.0; default avg)
    {"name": "late_kick", "default": 0.5, "range": (0.0, 1.0)},            # 23 (closing rate; >0.5=closing, <0.5=fading)
    {"name": "running_style", "default": 0.5, "range": (0.0, 1.0)},        # 24 (front=0.0, stalker=0.5, closer=1.0)
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
