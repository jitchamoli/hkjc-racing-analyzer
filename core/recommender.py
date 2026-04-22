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

    # === QUINELLA-FIRST STRATEGY (revised Apr 22) ===
    # Backtest findings (71 held-out races, Harville-model simulated P&L):
    #   - WIN pool: NO edge (agrees with market 98.6%, -37% ROI)
    #   - Straight QIN (top-2, 1 combo): +11-16% ROI, 16.9% hit — the validated edge
    #   - Box-3 QIN (3 combos): +0.4% / -5% ROI on most races
    # Default is straight QIN on top-2. Rank-3 is added as a 3rd leg (effectively
    # box-3) ONLY when it has strong historical credentials — this concentrates
    # the box-3 stake on races where rank-3 is genuinely live, not just because
    # the model ranks it.
    # Caveat: 71-race sample has wide CIs — weakly positive evidence, not conclusive.

    top_is_banker_now = any("BANKER" in r for r in analyses[0].reasoning)

    # Require live odds — without them EV and market comparisons are meaningless
    has_live_odds = any(a.horse.win_odds and a.horse.win_odds > 0 for a in analyses)

    # Conviction gates
    top_2_share = top_prob + second_prob
    MIN_TOP2_CONVICTION = 0.25  # 1.5× uniform 12-horse
    MIN_TOP3_CONVICTION = 0.30  # for box-3 upgrade

    # Rank-3 inclusion gate: only if horse has genuinely strong history
    rank3_strong = n >= 3 and _rank3_has_strong_history(analyses[2])

    if not has_live_odds or n < 2:
        race_bet_type = "SKIP"
    elif top_is_banker_now and market_prob_top >= 0.35 and gap_1_2 > 0.08:
        race_bet_type = "WIN"
    elif gap_1_2 < 0.02 and gap_2_3 < 0.02 and top_3_share > 0.35 and n >= 3:
        race_bet_type = "TRIO"
    elif rank3_strong and top_3_share >= MIN_TOP3_CONVICTION:
        # Box-3 justified — rank-3 has strong history AND model has top-3 edge
        race_bet_type = "BOX Q"
    elif top_2_share >= MIN_TOP2_CONVICTION:
        # DEFAULT: straight quinella (validated +11-16% ROI)
        race_bet_type = "QIN"
    else:
        race_bet_type = "SKIP"

    # Assign per-horse recommendations
    MIN_EV = config.MIN_EXPECTED_RETURN
    for a in analyses:
        rank = a.rank
        has_conviction = any("WINNING TRIP" in r or "BANKER" in r for r in a.reasoning)
        has_place_value = (a.horse.place_odds > 1.2) if a.horse.place_odds > 0 else (a.horse.win_odds >= 3.0)

        if race_bet_type == "SKIP":
            a.bet_recommendation = "SKIP"
        elif race_bet_type == "WIN":
            if rank == 1:
                is_banker = any("BANKER" in r for r in a.reasoning)
                a.bet_recommendation = "BANKER WIN" if is_banker else "WIN"
            elif rank == 2 and has_place_value and a.expected_return >= MIN_EV:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "TRIO":
            if rank <= 3:
                a.bet_recommendation = "TRIO"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "BOX Q":
            # Box-3 quinella — rank-3 has strong history, all top-3 as QIN legs
            if rank <= 3:
                a.bet_recommendation = "QIN"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "QIN":
            # Straight quinella — top 2 only
            if rank <= 2:
                a.bet_recommendation = "QIN"
            elif rank == 3 and has_place_value and has_conviction:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        else:
            a.bet_recommendation = "SKIP"


def _rank3_has_strong_history(analysis) -> bool:
    """Check if rank-3 horse has genuinely strong historical credentials.

    Used to gate inclusion as a 3rd QIN leg (box-3 upgrade). We require
    specific evidence of past performance, not just the model ranking — the
    backtest showed box-3 ONLY becomes profitable when rank-3 is a real
    contender, not a speculative pick.

    Returns True if any of:
      - BANKER or WINNING TRIP flagged in reasoning
      - Recent winner (form analyzer)
      - Career win rate ≥ 25% OR career top-3 rate ≥ 40%
      - Rule score ≥ 70 (composite historical signal)
    """
    import re
    for r in analysis.reasoning:
        ru = r.upper()
        if "BANKER" in ru or "WINNING TRIP" in ru:
            return True
        if ru.startswith("RECENT WINNER") or ru.startswith("STRONG FORM"):
            return True
        if r.startswith("Career:"):
            m = re.search(r"(\d+)%W", r)
            if m and int(m.group(1)) >= 25:
                return True
            m = re.search(r"(\d+)% top3", r)
            if m and int(m.group(1)) >= 40:
                return True
    return analysis.rule_score >= 70.0


def _resync_ranks_and_probs(analyses, calibrated_probs: List[float]) -> None:
    """After a rank swap, update all ranks, probabilities, expected returns, and value_score."""
    for i, a in enumerate(analyses):
        a.rank = i + 1
        a.combined_probability = calibrated_probs[i]
        a.value_score = calibrated_probs[i] * 100
        if a.horse.win_odds and a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
