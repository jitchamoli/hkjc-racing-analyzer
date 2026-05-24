"""Shared feature schema — single source of truth for train and predict.

Both extract_features() and build_training_data_from_db() MUST build
feature vectors in FEATURE_SCHEMA order. This prevents train/predict mismatch.

23 features (v6.11) — all use data available BEFORE the race (no leakage).
No direct market odds features — ML must provide independent signal.

v6.11 drops (2026-05-24, Tier 1 prune from 3-expert PRUNE review):
- draw_advantage: r=−0.0095 with is_winner on 50k rows (literally zero signal).
  field_rel_draw (r=−0.085) already carries the draw signal.
- speed_fig_norm: HK is paceless (small fields, jockey-controlled tempo);
  final times compress. r=0.53 with last_start_margin. Memory
  speed_fig_starved_not_dead documented prior NEUTRAL-for-betting verdict.
- running_style: r=+0.914 with early_speed (near-duplicate; 3-bin
  coarsened version of the same continuous signal).

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

v6.9 adds (2026-05-24): mid-race sectional pace from sectional_times
(99.4% Path B backfill). From the PR #52 former-jockey review:
"trip-trouble and unspent finishers live in the 800m→400m segment". Cheap
diagnostic on n=47,931 horse-race observations showed:
- r(mid_minus_late, is_top3) = +0.084 (real, weak, in the predicted direction)
- r(mid_minus_late, late_kick) = +0.23 (orthogonal — late_kick is positions-
  based, mid_race_pace is time-based; they measure different things)
- mid_race_pace: race-mean-normalized (800-400 split) minus (last-400 split),
  averaged across the horse's last 3 prior runs with 3+ sectionals. Mapped
  to [0,1] via 0.5 + clamp(raw, -0.2, 0.2) / 0.4. Positive (>0.5) = slow in
  mid-race then fast late = "unspent finisher". Defaults to 0.5 when <2
  prior sectional runs available.

v6.10 adds (2026-05-24): sire_top3_rate_on_going from PR #52 handicap-expert
review ("biggest single missing feature; HK is a closed gene pool so per-sire
sample is large enough where per-horse-going is too sparse"). Cheap diagnostic
on n=55,387 rows: strongest pass in 6+ weeks.
- r(sire_top3_rate_on_going, is_top3) = +0.0772 (80% of the signal strength
  of horse_prior_win_rate)
- r vs horse_prior_win_rate = +0.29  (orthogonal)
- r vs per-horse going_performance = +0.45 (orthogonal — sire prior carries
  info that horse's own going history can't, because horse has ~25 starts
  total while sire has 500-3,000 progeny starts)
- Source: renavon_horses (99.7% sire coverage of currently-active HK horses)
- Beta(2, 8) Bayesian smoothing — shrinks rare-sires (<10 progeny starts on
  going) toward 20% base top-3 rate. Going bucketed FIRM/GY/YIELDING/SOFT/WET-FAST
  to avoid tiny per-going-string buckets.
- Re-rank test on burned 6-meeting holdout (60 races): at λ=0.05, QIN +4 hits,
  TOP-5 +1, WIN/TOP-4 unchanged — strictly no regressions.
"""

import numpy as np

FEATURE_SCHEMA = [
    {"name": "prior_win_rate", "default": 0.5, "range": (0.0, 1.0)},       # 1
    {"name": "recent_form", "default": 0.5, "range": (0.0, 1.0)},          # 2
    {"name": "jockey_win_rate", "default": 0.08, "range": (0.0, 0.6)},     # 3
    {"name": "trainer_win_rate", "default": 0.06, "range": (0.0, 0.5)},    # 4
    {"name": "field_size", "default": 0.857, "range": (0.0, 1.0)},         # 5 (clipped to [0,1])
    {"name": "field_rel_draw", "default": 0.5, "range": (0.0, 1.0)},       # 6 (clipped to [0,1])
    {"name": "cd_form_prior", "default": 0.5, "range": (0.0, 1.0)},        # 7
    {"name": "class_level", "default": 0.4, "range": (0.0, 1.0)},          # 8 (Group=1.0, Class1=0.8, Class5=0.0)
    {"name": "freshness", "default": 0.5, "range": (0.0, 1.0)},            # 9
    {"name": "going_performance", "default": 0.5, "range": (0.0, 1.0)},    # 10
    {"name": "weight_vs_field", "default": 0.5, "range": (0.0, 1.0)},      # 11
    {"name": "jockey_horse_coop", "default": 0.5, "range": (0.0, 1.0)},    # 12
    {"name": "last_start_margin", "default": 0.5, "range": (0.0, 1.0)},    # 13
    {"name": "class_change", "default": 0.5, "range": (0.0, 1.0)},         # 14
    {"name": "horse_weight_change", "default": 0.5, "range": (0.0, 1.0)},  # 15
    {"name": "horse_age", "default": 0.5, "range": (0.0, 1.0)},            # 16 (3yo→1.0, 9yo→0.0)
    {"name": "rating_change", "default": 0.5, "range": (0.0, 1.0)},        # 17 (+10pts→1.0, -10→0.0)
    {"name": "jockey_form_30d", "default": 0.08, "range": (0.0, 0.6)},     # 18 (last 30-day strike rate)
    {"name": "jt_pair_win_rate", "default": 0.08, "range": (0.0, 0.5)},    # 19 (Bayesian-smoothed jockey×trainer)
    {"name": "early_speed", "default": 0.5, "range": (0.0, 1.0)},          # 20 (1st-call rank, front=0.0 closer=1.0; default avg)
    {"name": "late_kick", "default": 0.5, "range": (0.0, 1.0)},            # 21 (closing rate; >0.5=closing, <0.5=fading)
    {"name": "mid_race_pace", "default": 0.5, "range": (0.0, 1.0)},        # 22 (mid_minus_late: race-mean-normalized 800-400 split minus last-400 split, mean of last 3 prior sectional runs)
    {"name": "sire_top3_rate_on_going", "default": 0.20, "range": (0.0, 1.0)},  # 23 (Beta(2,8)-smoothed top-3 rate of horse's sire on TODAY's going bucket, all prior progeny starts; default = base top-3 rate)
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
