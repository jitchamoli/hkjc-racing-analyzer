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
    # F-09 empirically chosen: base 45/55 model/market, sprint threshold 1200m
    model_w = 0.45
    market_w = 0.55

    # Large field (10+) + sprint (≤1200m): market is sharpest → shift +15% toward market
    if field_size >= 10 and distance <= 1200:
        market_w = min(0.70, market_w + 0.15)
        model_w = 1.0 - market_w
    # Small field (<10) or long distance (≥1800m): thinner pool → shift +15% toward model
    elif field_size < 10 or distance >= 1800:
        model_w = min(0.70, model_w + 0.15)
        market_w = 1.0 - model_w

    return model_w, market_w


def calibrate_probabilities(analyses, field_size: int = 12,
                            distance: int = 1200) -> List[float]:
    """Convert combined_probability scores into calibrated race-level probabilities.

    Uses mild softmax normalization (to ensure probabilities sum to 1),
    then blends with market-implied probabilities using dynamic weights
    based on field size and distance.

    Note: Individual ML probabilities are already isotonic-calibrated at the
    model level (Wave 2c switch from Platt), so the softmax here is purely
    for normalization, not recalibration.
    """
    if not analyses:
        return []

    # Mild softmax — just enough to normalize, not distort
    temp = config.SOFTMAX_TEMPERATURE
    raw_scores = [a.combined_probability for a in analyses]
    max_score = max(raw_scores) if raw_scores else 0.5
    exp_scores = [math.exp((s - max_score) * temp) for s in raw_scores]
    exp_total = sum(exp_scores) or 1.0
    cal_probs = [e / exp_total for e in exp_scores]

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

    blend_total = sum(blended) or 1.0
    return [p / blend_total for p in blended]


def apply_calibrated_probs(analyses, calibrated_probs: List[float]) -> None:
    """Write calibrated probabilities, value_score, and expected returns onto analyses.

    F-19: On the first call (model_probability is still None), snapshot the
    calibrated prob into model_probability so accuracy.py can measure the model
    without contamination from subsequent BANKER/market-fav rank swaps.
    """
    for i, a in enumerate(analyses):
        if a.model_probability is None:
            a.model_probability = calibrated_probs[i]
        a.combined_probability = calibrated_probs[i]
        a.value_score = calibrated_probs[i] * 100
        if a.horse.win_odds and a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
        else:
            a.expected_return = -1.0
