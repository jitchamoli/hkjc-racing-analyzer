"""Task #47 regression: BANKER override cal-prob guard.

Discovered HV 2026-05-13 R9: SILVERY BREEZE locked at rank 1 by BANKER 6/8
override even though its cal_prob (0.097) was materially below market favourite
LO RIDER's cal_prob (0.168). Rule-engine BANKER signals shouldn't trump
post-blend probability ordering when the gap is large.
"""
import pytest
from unittest.mock import patch
from core.analyzer import Horse, HorseAnalysis
from core.recommender import assign_bet_recommendations


def _make_analysis(num, name, win_odds, rule_score, reasoning, rank,
                   combined_prob):
    horse = Horse(
        number=num, name=name, draw=1, jockey='J', trainer='T',
        weight='130', rating=80, win_odds=win_odds, place_odds=2.5,
        last_6='5-3-2', tips_odds=win_odds,
    )
    return HorseAnalysis(
        horse=horse, rank=rank, rule_score=rule_score,
        rule_confidence='HIGH', ml_prediction=None,
        combined_probability=combined_prob, combined_confidence='HIGH',
        value_score=combined_prob * 100, expected_return=0.0,
        bet_recommendation='SKIP', reasoning=reasoning,
        prediction_breakdown={},
    )


def test_banker_override_declined_when_cal_prob_far_below_market_fav():
    """SILVERY BREEZE prototype: BANKER 6/8 with low cal_prob shouldn't
    displace a market favourite with 1.7x higher cal_prob.
    """
    # Patch calibrate_probabilities to return our pre-set probs (no model needed)
    fixed_probs = [0.45, 0.097]  # rank-1 mkt fav 0.45, BANKER candidate 0.097
    with patch('core.recommender.calibrate_probabilities', return_value=fixed_probs):
        analyses = [
            # Index 0: market favourite (low odds, no BANKER), cal_prob 0.45
            _make_analysis(1, 'LO RIDER', 2.2, 60.0,
                           reasoning=['Strong recent form'], rank=1,
                           combined_prob=0.45),
            # Index 1: BANKER candidate, low cal_prob 0.097, longshot odds
            _make_analysis(2, 'SILVERY BREEZE', 10.0, 73.8,
                           reasoning=['BANKER (6/8 signals: rating surge, class rise)'],
                           rank=2, combined_prob=0.097),
        ]
        assign_bet_recommendations(analyses, field_size=12)

    # Rank 1 should stay with the market favourite — guard refused the swap
    assert analyses[0].horse.name == 'LO RIDER', (
        f"BANKER guard failed: rank 1 swapped to {analyses[0].horse.name} "
        f"despite cal_prob being far below market fav's"
    )


def test_banker_override_fires_when_cal_prob_close_to_market_fav():
    """Guard should NOT block legitimate BANKER promotions where the BANKER
    candidate's cal_prob is comparable to the market fav's.
    """
    # BANKER candidate has 0.32 cal_prob, market fav has 0.40. Ratio 0.8 > 0.7 → swap allowed.
    fixed_probs = [0.40, 0.32]
    with patch('core.recommender.calibrate_probabilities', return_value=fixed_probs):
        analyses = [
            _make_analysis(1, 'MKT_FAV', 2.5, 60.0,
                           reasoning=['Solid form'], rank=1,
                           combined_prob=0.40),
            _make_analysis(2, 'BANKER_HORSE', 3.5, 72.0,
                           reasoning=['BANKER (6/8 signals fire)'],
                           rank=2, combined_prob=0.32),
        ]
        assign_bet_recommendations(analyses, field_size=12)

    # BANKER swap should fire: rank 1 is now BANKER_HORSE
    assert analyses[0].horse.name == 'BANKER_HORSE', (
        f"BANKER swap blocked unfairly: rank 1 stayed at {analyses[0].horse.name} "
        f"despite cal_prob being close to market fav's"
    )


def test_multi_banker_tiebreak_uses_calibrated_not_rule_score():
    """When two horses both fire BANKER, the higher calibrated_prob (post-blend)
    should win, not the higher rule_score (pre-blend). Was using rule_score.
    """
    # Two horses fire BANKER. Horse A has higher rule_score but lower cal_prob.
    # Horse B has lower rule_score but higher cal_prob. Tiebreak should pick B.
    fixed_probs = [0.40, 0.15, 0.30]  # market_fav 0.40, A 0.15, B 0.30
    with patch('core.recommender.calibrate_probabilities', return_value=fixed_probs):
        analyses = [
            _make_analysis(1, 'MKT_FAV', 2.5, 55.0,
                           reasoning=['Solid'], rank=1, combined_prob=0.40),
            _make_analysis(2, 'BANKER_A_HIGH_RULE', 6.0, 80.0,
                           reasoning=['BANKER (6/8 signals)'],
                           rank=2, combined_prob=0.15),
            _make_analysis(3, 'BANKER_B_HIGH_CAL', 4.5, 70.0,
                           reasoning=['BANKER (6/8 signals)'],
                           rank=3, combined_prob=0.30),
        ]
        assign_bet_recommendations(analyses, field_size=12)

    # Expectation depends on the cal-prob guard:
    # - BANKER_A: cal 0.15, ratio to mkt_fav 0.375 → guard rejects
    # - BANKER_B: cal 0.30, ratio to mkt_fav 0.75 → guard allows
    # So BANKER_B is the only valid candidate; tiebreak picks it.
    # Even with both allowed, calibrated tiebreak picks B (0.30 > 0.15).
    assert analyses[0].horse.name == 'BANKER_B_HIGH_CAL', (
        f"Multi-BANKER tiebreak picked wrong horse: rank 1 is "
        f"{analyses[0].horse.name}; expected BANKER_B_HIGH_CAL "
        f"(higher cal_prob despite lower rule_score)"
    )
