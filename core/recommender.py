#!/usr/bin/env python3
"""Bet recommendation engine — assigns BANKER/WIN/QIN/PLACE/SKIP per horse."""

import logging
from typing import List

import config
from core.scorer import calibrate_probabilities, apply_calibrated_probs

logger = logging.getLogger(__name__)


def assign_bet_recommendations(analyses, field_size: int = 12,
                               distance: int = 1200) -> None:
    """Assign realistic bet recommendations based on field shape.

    Calibrates probabilities via softmax + market blend (dynamic weights
    based on field size and distance), handles market-favourite promotion
    and BANKER overrides, then assigns bet types per horse.
    """
    if not analyses:
        return

    n = len(analyses)
    calibrated_probs = calibrate_probabilities(analyses, field_size=field_size,
                                               distance=distance)
    apply_calibrated_probs(analyses, calibrated_probs)

    top = analyses[0] if n >= 1 else None
    second = analyses[1] if n >= 2 else None

    top_prob = calibrated_probs[0] if n >= 1 else 0
    second_prob = calibrated_probs[1] if n >= 2 else 0
    third_prob = calibrated_probs[2] if n >= 3 else 0

    gap_1_2 = top_prob - second_prob
    gap_2_3 = second_prob - third_prob
    top_3_share = top_prob + second_prob + third_prob

    market_prob_top = (1 / top.horse.win_odds) if top and top.horse.win_odds and top.horse.win_odds > 0 else 0

    top_is_banker = any("BANKER" in r for r in analyses[0].reasoning) if analyses else False

    # Market favourite promotion (skip if BANKER)
    market_fav = min(analyses, key=lambda a: a.horse.win_odds if a.horse.win_odds and a.horse.win_odds > 0 else 999)
    market_fav_prob = (1 / market_fav.horse.win_odds) if market_fav.horse.win_odds and market_fav.horse.win_odds > 0 else 0
    if (market_fav_prob >= 0.30 and market_fav.rank > 1
            and market_fav.horse.win_odds > 0
            and not top_is_banker):
        old_idx = next(i for i, a in enumerate(analyses) if a is market_fav)
        analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]
        calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]
        _resync_ranks_and_probs(analyses, calibrated_probs)
        top = analyses[0]
        top_prob = calibrated_probs[0]
        second_prob = calibrated_probs[1] if n >= 2 else 0
        third_prob = calibrated_probs[2] if n >= 3 else 0
        gap_1_2 = top_prob - second_prob
        gap_2_3 = second_prob - third_prob
        top_3_share = top_prob + second_prob + third_prob
        market_prob_top = market_fav_prob

    # BANKER promotion
    banker_horse = None
    for a in analyses:
        if any("BANKER" in r for r in a.reasoning):
            if banker_horse is None or a.rule_score > banker_horse.rule_score:
                banker_horse = a

    if banker_horse and banker_horse.rank > 1:
        old_idx = next(i for i, a in enumerate(analyses) if a is banker_horse)
        analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]
        calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]
        _resync_ranks_and_probs(analyses, calibrated_probs)
        top = analyses[0]
        top_prob = calibrated_probs[0]
        second_prob = calibrated_probs[1] if n >= 2 else 0
        third_prob = calibrated_probs[2] if n >= 3 else 0
        gap_1_2 = top_prob - second_prob
        gap_2_3 = second_prob - third_prob
        top_3_share = top_prob + second_prob + third_prob

    # Determine race-level bet type
    race_bet_type = "QIN"  # default

    if banker_horse and banker_horse.rank == 1:
        race_bet_type = "WIN"
    elif market_prob_top >= 0.35 and gap_1_2 > 0.08:
        race_bet_type = "WIN"
    elif gap_1_2 > 0.06 and top and top.expected_return > 0.15:
        race_bet_type = "WIN"
    elif gap_1_2 < 0.02 and top_3_share > 0.35:
        if gap_2_3 < 0.02 and n >= 3:
            race_bet_type = "TRIO"
        else:
            race_bet_type = "QNL PL"

    # Race-level skip check
    MIN_EV = config.MIN_EXPECTED_RETURN
    top_ev = analyses[0].expected_return if analyses else -1
    top_cal_prob = calibrated_probs[0] if calibrated_probs else 0
    top_market_prob = (1 / analyses[0].horse.win_odds) if analyses and analyses[0].horse.win_odds and analyses[0].horse.win_odds > 0 else 1
    has_edge = top_cal_prob > top_market_prob
    skip_race = top_ev < MIN_EV and not has_edge and not (banker_horse and banker_horse.rank == 1)

    # Assign per-horse recommendations
    for i, a in enumerate(analyses):
        if skip_race:
            a.bet_recommendation = "SKIP"
            continue

        rank = a.rank
        bet_has_ev = a.expected_return >= MIN_EV
        has_place_value = (a.horse.place_odds > 1.2) if a.horse.place_odds > 0 else (a.horse.win_odds >= 3.0)

        if race_bet_type == "WIN":
            if rank == 1:
                is_banker = any("BANKER" in r for r in a.reasoning)
                a.bet_recommendation = "BANKER WIN" if is_banker else "WIN"
            elif rank == 2 and bet_has_ev and has_place_value:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "QIN":
            if rank <= 2:
                a.bet_recommendation = "QIN"
            elif rank == 3 and bet_has_ev and has_place_value:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "QNL PL":
            if rank <= 2:
                a.bet_recommendation = "QNL PL"
            elif rank == 3 and bet_has_ev and has_place_value:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "TRIO":
            if rank <= 3:
                a.bet_recommendation = "TRIO"
            elif rank == 4 and bet_has_ev and has_place_value:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "PLACE":
            if rank <= 3 and bet_has_ev:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        else:
            a.bet_recommendation = "SKIP"


def _resync_ranks_and_probs(analyses, calibrated_probs: List[float]) -> None:
    """After a rank swap, update all ranks, probabilities, and expected returns."""
    for i, a in enumerate(analyses):
        a.rank = i + 1
        a.combined_probability = calibrated_probs[i]
        if a.horse.win_odds and a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
