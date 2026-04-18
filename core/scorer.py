#!/usr/bin/env python3
"""Score combination, calibration, and probability normalization."""

import math
import logging
from typing import List, Dict, Tuple

import config

logger = logging.getLogger(__name__)


class BettingWeights:
    """Factor weight constants for the 26-factor rule engine."""
    RATING = 0.07
    FORM = 0.09
    ODDS = 0.08
    JOCKEY = 0.04
    TRAINER = 0.02
    DRAW = 0.04
    TIPS = 0.10
    PACE = 0.03
    SECTIONAL = 0.03
    CD_FORM = 0.04
    ODDS_MOVE = 0.04
    CLASS_MOVE = 0.03
    GEAR = 0.02
    WEIGHT = 0.04
    FRESHNESS = 0.04
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
        "rating": RATING, "form": FORM, "odds": ODDS,
        "jockey": JOCKEY, "trainer": TRAINER, "draw": DRAW,
        "tips": TIPS, "pace": PACE, "sectional": SECTIONAL,
        "cd_form": CD_FORM, "odds_movement": ODDS_MOVE,
        "class_movement": CLASS_MOVE, "gear": GEAR,
        "weight_carried": WEIGHT, "freshness": FRESHNESS,
        "rating_trend": RATING_TREND, "season_stakes": STAKES,
        "going_pref": GOING, "rolling_form": ROLLING_FORM,
        "margin": MARGIN, "jockey_change": JOCKEY_CHANGE,
        "jockey_form": JOCKEY_FORM, "trainer_form": TRAINER_FORM,
        "field_size": FIELD_SIZE, "career_place": CAREER_PLACE,
        "age": AGE,
    }


# Factors always included (score of 0 means no data; these use 50 as neutral)
ALWAYS_INCLUDED = {"rating", "form", "odds", "jockey", "trainer", "draw",
                   "class_movement", "gear", "weight_carried"}


def compute_weighted_score(factors: Dict[str, float]) -> float:
    """Compute rule-based score from factor dict with dynamic weight normalization.

    Factors with score == 0 are excluded (except ALWAYS_INCLUDED factors).
    Remaining weights are renormalized to sum to 1.0.
    """
    weight_map = {}
    for key, score in factors.items():
        if key in ALWAYS_INCLUDED or score > 0:
            if key in BettingWeights.WEIGHT_MAP:
                weight_map[key] = BettingWeights.WEIGHT_MAP[key]

    total_weight = sum(weight_map.values())
    if total_weight > 0:
        return sum(factors[k] * (w / total_weight) for k, w in weight_map.items())
    return 50.0


def calibrate_probabilities(analyses) -> List[float]:
    """Convert raw combined_probability scores into calibrated probabilities.

    Uses softmax with temperature scaling, then blends with market-implied
    probabilities (30% market, 70% model).
    """
    if not analyses:
        return []

    temp = config.SOFTMAX_TEMPERATURE
    raw_scores = [a.combined_probability for a in analyses]
    max_score = max(raw_scores) if raw_scores else 0.5
    exp_scores = [math.exp((s - max_score) * temp) for s in raw_scores]
    exp_total = sum(exp_scores) or 1.0
    cal_probs = [e / exp_total for e in exp_scores]

    # Blend with market-implied probabilities
    market_probs = []
    for a in analyses:
        if a.horse.win_odds > 0:
            market_probs.append(1.0 / a.horse.win_odds)
        else:
            market_probs.append(0.0)
    market_total = sum(market_probs) or 1.0
    market_probs = [p / market_total for p in market_probs]

    blended = []
    for i in range(len(cal_probs)):
        if market_probs[i] > 0:
            b = config.MODEL_BLEND_WEIGHT * cal_probs[i] + config.MARKET_BLEND_WEIGHT * market_probs[i]
        else:
            b = cal_probs[i]
        blended.append(b)

    blend_total = sum(blended) or 1.0
    return [p / blend_total for p in blended]


def apply_calibrated_probs(analyses, calibrated_probs: List[float]) -> None:
    """Write calibrated probabilities and expected returns onto analyses."""
    for i, a in enumerate(analyses):
        a.combined_probability = calibrated_probs[i]
        if a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
        else:
            a.expected_return = -1.0
