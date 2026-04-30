#!/usr/bin/env python3
"""Score combination, calibration, and probability normalization."""

import math
import logging
from typing import List, Dict, Tuple

import config

logger = logging.getLogger(__name__)


class BettingWeights:
    """Factor weight constants for the 24-factor rule engine.

    Note: odds and odds_movement REMOVED — market signal enters via the 30%
    market blend in calibrate_probabilities(). Keeping them here would
    triple-count market information (rule odds + rule odds_move + blend).
    """
    RATING = 0.08
    FORM = 0.10
    JOCKEY = 0.05
    TRAINER = 0.03
    DRAW = 0.05
    TIPS = 0.11
    PACE = 0.04
    SECTIONAL = 0.03
    CD_FORM = 0.08
    CLASS_MOVE = 0.03
    GEAR = 0.02
    WEIGHT = 0.04
    FRESHNESS = 0.05
    RATING_TREND = 0.03
    STAKES = 0.02
    GOING = 0.05
    ROLLING_FORM = 0.06
    MARGIN = 0.05
    JOCKEY_CHANGE = 0.04
    JOCKEY_FORM = 0.04
    TRAINER_FORM = 0.03
    FIELD_SIZE = 0.05
    CAREER_PLACE = 0.03
    AGE = 0.02

    # Mapping factor keys to weights
    WEIGHT_MAP = {
        "rating": RATING, "form": FORM,
        "jockey": JOCKEY, "trainer": TRAINER, "draw": DRAW,
        "tips": TIPS, "pace": PACE, "sectional": SECTIONAL,
        "cd_form": CD_FORM,
        "class_movement": CLASS_MOVE, "gear": GEAR,
        "weight_carried": WEIGHT, "freshness": FRESHNESS,
        "rating_trend": RATING_TREND, "season_stakes": STAKES,
        "going_pref": GOING, "rolling_form": ROLLING_FORM,
        "margin": MARGIN, "jockey_change": JOCKEY_CHANGE,
        "jockey_form": JOCKEY_FORM, "trainer_form": TRAINER_FORM,
        "field_size": FIELD_SIZE, "career_place": CAREER_PLACE,
        "age": AGE,
    }


# Historical set — previously forced inclusion even at score=0.
# As of C1 (2026-04-23), no longer load-bearing; retained for documentation.
# Missing factors (score <= 0) are now uniformly excluded via renormalization.
ALWAYS_INCLUDED = {"rating", "form", "jockey", "trainer", "draw",
                   "class_movement", "gear", "weight_carried"}


def compute_weighted_score(factors: Dict[str, float]) -> float:
    """Compute rule-based score from factor dict with dynamic weight normalization.

    Factors with score <= 0 are excluded as "no data" and weights are
    renormalized across the remaining factors to sum to 1.0.

    Historical note (C1, 2026-04-23 audit): `ALWAYS_INCLUDED` previously
    forced inclusion of rating/gear/etc. even at score=0, dragging rule
    scores down ~8 points in 86% of backtest races where DB was missing
    these fields. The set is retained for documentation but no longer
    has load-bearing inclusion logic — exclusion is now uniform across
    all factors.
    """
    weight_map = {}
    for key, score in factors.items():
        if score > 0:
            if key in BettingWeights.WEIGHT_MAP:
                weight_map[key] = BettingWeights.WEIGHT_MAP[key]

    total_weight = sum(weight_map.values())
    if total_weight > 0:
        return sum(factors[k] * (w / total_weight) for k, w in weight_map.items())
    return 50.0


def _dynamic_blend_weights(field_size: int, distance: int) -> Tuple[float, float]:
    """Compute dynamic market/model blend weights based on race characteristics.

    HK bettors are sharp — larger fields and shorter races benefit from
    heavier market weighting. Smaller fields and longer distances have
    thinner pools where the model has more edge.

    F-09 grid sweep (2026-04-23, 15 points × 73-race holdout):
    best Brier at base_model_w=0.45, field_threshold=10, distance_threshold=1200.
    All 15 points fell within ±0.002 Brier (noise floor), so the tie-break rule
    (within ±0.005 Brier, prefer simpler schedule = base_model_w closest to 0.5)
    selected base_model_w=0.45 over the prior heuristic 0.60.
    See docs/audits/2026-04-23/raw/wave2-sweep-summary.json.

    Returns (model_weight, market_weight) summing to 1.0.
    """
    # F-09 empirically chosen base 45/55 model/market, sprint threshold 1200m,
    # field threshold 10. Replaced hard cutoffs (40pp swing on field_size
    # differing by 1 horse, e.g. 10 vs 9 at 1200m) with logistic ramps so
    # the weight changes smoothly across boundaries — no noise amplification
    # at exactly threshold values.
    base_model_w = 0.45

    # Sigmoid ramps: σ(x/τ) is 0 far below threshold, 1 far above, ~0.5 at threshold.
    # τ = scale (smoothness): larger = smoother transition.
    sprint_factor = 1.0 / (1.0 + math.exp(-(1200 - distance) / 100.0))   # high for sprints
    big_field_factor = 1.0 / (1.0 + math.exp(-(field_size - 10) / 1.5))  # high for ≥10 horses
    long_distance_factor = 1.0 / (1.0 + math.exp(-(distance - 1800) / 100.0))  # high for ≥1800m
    small_field_factor = 1.0 / (1.0 + math.exp(-(10 - field_size) / 1.5))   # high for <10 horses

    # Large field + sprint → market sharper → shift toward market (max +0.15)
    market_shift = 0.15 * sprint_factor * big_field_factor
    # Small field OR long distance → thinner pool → shift toward model (max +0.15)
    # Use OR via algebraic combination: 1 - (1-small)(1-long)
    pool_thinness = 1.0 - (1.0 - small_field_factor) * (1.0 - long_distance_factor)
    model_shift = 0.15 * pool_thinness

    # Apply both shifts (they can partially offset on edge cases)
    model_w = base_model_w + model_shift - market_shift
    model_w = max(0.30, min(0.70, model_w))
    market_w = 1.0 - model_w

    return model_w, market_w


_BLEND_CALIBRATOR = None  # cached on first load
_BLEND_CALIBRATOR_LOADED = False


def _load_blend_calibrator():
    """Lazy-load models/blend_calibrator.pkl. Returns None if absent.

    The blend calibrator is an IsotonicRegression fitted on
    (production calibrated_prob → actual_won) over the held-out 20% of
    races. It corrects the residual under-prediction bias the post-C1
    pipeline exhibits on top picks. Built by `python3 -m scripts.calibrate_blend`.
    """
    global _BLEND_CALIBRATOR, _BLEND_CALIBRATOR_LOADED
    if _BLEND_CALIBRATOR_LOADED:
        return _BLEND_CALIBRATOR
    _BLEND_CALIBRATOR_LOADED = True
    import hashlib, pickle
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    pkl = repo_root / 'models' / 'blend_calibrator.pkl'
    sha = pkl.with_suffix('.sha256')
    if not pkl.exists():
        return None
    try:
        raw = pkl.read_bytes()
        if sha.exists():
            expected = sha.read_text().strip()
            if hashlib.sha256(raw).hexdigest() != expected:
                # Don't silently load a tampered/stale file
                return None
        payload = pickle.loads(raw)
        _BLEND_CALIBRATOR = payload.get('calibrator')
    except Exception:
        _BLEND_CALIBRATOR = None
    return _BLEND_CALIBRATOR


def calibrate_probabilities(analyses, field_size: int = 12,
                            distance: int = 1200) -> List[float]:
    """Convert combined_probability scores into calibrated race-level probabilities.

    Pipeline:
      1. Mild softmax over combined_probability → cal_probs
      2. Dynamic blend with market-implied probs → blended
      3. (optional) Apply post-blend isotonic calibrator if present
      4. Normalize so sum==1

    The post-blend calibrator (step 3) corrects a residual under-prediction
    bias on the post-C1 anchored-sigmoid distribution — top picks were
    predicted ~32% but actual win rate was ~46% on a 5,966-race holdout.
    Calibrator is fit by `scripts/calibrate_blend.py`; absent file → step 3
    is skipped (backwards-compatible).

    Per-horse isotonic preserves rank ordering (monotonic), so renormalizing
    after applying maintains both the win-prob calibration and sum-to-one.
    """
    if not analyses:
        return []

    # combined_probability is already a probability in [0,1]
    # (rule_prob × 0.6 + ml_prob × 0.4 from create_combined_prediction).
    # Just normalize so they sum to 1.0 — softmax with temp here would
    # collapse model separation: e.g. raw 0.50 vs 0.05 horses become
    # 0.13 vs 0.066 after softmax(temp=1.5), destroying signal before
    # the market blend.
    raw_scores = [a.combined_probability for a in analyses]
    raw_total = sum(raw_scores) or 1.0
    cal_probs = [s / raw_total for s in raw_scores]

    # Blend with market-implied probabilities
    market_probs = []
    for a in analyses:
        if a.horse.win_odds and a.horse.win_odds > 0:
            market_probs.append(1.0 / a.horse.win_odds)
        else:
            market_probs.append(0.0)
    market_total = sum(market_probs) or 1.0
    market_probs = [p / market_total for p in market_probs]

    # Dynamic blend weights based on race characteristics
    model_w, market_w = _dynamic_blend_weights(field_size, distance)

    blended = []
    for i in range(len(cal_probs)):
        if market_probs[i] > 0:
            b = model_w * cal_probs[i] + market_w * market_probs[i]
        else:
            # F-20: no market signal (horse has no live odds).
            # Math: model_w * cal + market_w * cal = (model_w + market_w) * cal = cal.
            # Using cal_probs[i] directly is equivalent to treating cal_prob as its own
            # market proxy. Post-normalization (line below) keeps this consistent with
            # the blended horses — no systematic over/under-weighting.
            b = cal_probs[i]
        blended.append(b)

    # Pre-isotonic renorm: input to isotonic should sum to 1 within the race.
    # Post-isotonic renorm (Fix A): the calibrator was fit per-horse so its
    # output is monotonic but does NOT sum to 1 across a race — empirically
    # the post-iso sum landed at 1.16-1.28 on in-distribution inputs, which
    # then inflated:
    #   - expected_return = win_odds × prob - 1 (each horse looked +25% richer)
    #   - value_score = prob × 100 (top picks displayed 50%+ when truly ~40%)
    #   - recommender top_2/3/4_share thresholds (gates fired ~16-28% more)
    # Restore sum=1 after isotonic so downstream EV/threshold consumers see
    # a proper probability distribution. Per-horse calibration shape (rank
    # ordering, ratios over short ranges) is preserved by the renorm.
    blend_total = sum(blended) or 1.0
    blended = [p / blend_total for p in blended]

    iso = _load_blend_calibrator()
    if iso is not None:
        try:
            import numpy as _np
            blended = list(iso.predict(_np.array(blended)))
        except Exception:
            pass  # fall through to raw blended values on any sklearn error

    iso_total = sum(blended) or 1.0
    return [p / iso_total for p in blended]


def apply_calibrated_probs(analyses, calibrated_probs: List[float]) -> None:
    """Write calibrated probabilities, value_score, and expected returns onto analyses.

    F-19: On the first call (model_probability is still None), snapshot the
    PRE-blend pure model probability into model_probability so accuracy.py
    can measure the model itself, not the model+market blend. At call time
    a.combined_probability still holds the pre-blend value (set by analyzer);
    we capture it BEFORE overwriting with the post-blend calibrated value.
    Without this, accuracy reports systematically credit the market for the
    model's perceived performance.
    """
    for i, a in enumerate(analyses):
        if a.model_probability is None:
            a.model_probability = a.combined_probability  # pre-blend snapshot
        a.combined_probability = calibrated_probs[i]
        a.value_score = calibrated_probs[i] * 100
        if a.horse.win_odds and a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
        else:
            a.expected_return = -1.0
