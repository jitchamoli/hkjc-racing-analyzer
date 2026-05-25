"""GPT-5.5 R13 #3 (2026-05-25) — REWRITTEN.

Original tests pinned the BANKER promotion + cal-prob guard mechanism
(Task #47, SILVERY BREEZE 2026-05-13). Under the R12 contract
(`rank = calibrated_prob ordering`), market-fav AND BANKER rank
promotions were REMOVED — they made rank mean "rule override winner"
instead of "highest final probability" and were the source of
cross-surface drift between ticket / UI / WA / replay.

New invariant: `assign_bet_recommendations` output must be globally
sorted by calibrated `combined_probability`. BANKER stays in
reasoning text as diagnostic only; BANKER WIN bet type fires only
when the BANKER horse is ALREADY rank-1 by calibrated_prob.

The dropped promotion logic can still be evaluated as a shadow /
counterfactual variant (see memory feedback_contract_tests_over_unit_tests),
but it is no longer production ranking.
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


def test_banker_does_not_promote_lower_cal_prob_horse():
    """SILVERY BREEZE prototype: BANKER 6/8 with cal_prob 0.097 must NOT
    displace a market favourite with cal_prob 0.45. Under the new
    contract there's no promotion at all — rank-1 is whoever has the
    highest calibrated_prob, regardless of BANKER reasoning."""
    fixed_probs = [0.45, 0.097]
    with patch('core.recommender.calibrate_probabilities', return_value=fixed_probs):
        analyses = [
            _make_analysis(1, 'LO RIDER', 2.2, 60.0,
                           reasoning=['Strong recent form'], rank=1,
                           combined_prob=0.45),
            _make_analysis(2, 'SILVERY BREEZE', 10.0, 73.8,
                           reasoning=['BANKER (6/8 signals: rating surge, class rise)'],
                           rank=2, combined_prob=0.097),
        ]
        assign_bet_recommendations(analyses, field_size=12)
    assert analyses[0].horse.name == 'LO RIDER', (
        f"rank-1 must be highest calibrated_prob; got {analyses[0].horse.name}"
    )
    # SILVERY BREEZE keeps BANKER in reasoning (diagnostic), but does NOT
    # get BANKER WIN bet type (only fires for rank-1 by calibrated_prob).
    silvery = next(a for a in analyses if a.horse.name == 'SILVERY BREEZE')
    assert silvery.bet_recommendation != 'BANKER WIN', (
        "BANKER WIN must NOT fire on a non-rank-1 horse"
    )


def test_banker_does_not_promote_even_when_cal_prob_close():
    """Removed-promotion contract: BANKER NEVER swaps ranks, even when its
    cal_prob is near the market fav's. The 'guard threshold 0.7' from the
    old promotion mechanism is gone — rank ordering is purely
    calibrated_prob. BANKER stays informational."""
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
    assert analyses[0].horse.name == 'MKT_FAV', (
        f"rank-1 must stay with highest cal_prob; got {analyses[0].horse.name}"
    )


def test_global_calibrated_prob_ordering_holds_with_banker_in_field():
    """Even with multiple BANKER signals in the field, the final ordering
    must be strictly by calibrated_prob descending. No rank swap may
    happen for rule-engine reasons."""
    fixed_probs = [0.40, 0.15, 0.30]
    with patch('core.recommender.calibrate_probabilities', return_value=fixed_probs):
        analyses = [
            _make_analysis(1, 'MKT_FAV', 2.5, 55.0,
                           reasoning=['Solid'], rank=1, combined_prob=0.40),
            _make_analysis(2, 'BANKER_A_LOW_CAL', 6.0, 80.0,
                           reasoning=['BANKER (6/8 signals)'],
                           rank=2, combined_prob=0.15),
            _make_analysis(3, 'BANKER_B_MED_CAL', 4.5, 70.0,
                           reasoning=['BANKER (6/8 signals)'],
                           rank=3, combined_prob=0.30),
        ]
        assign_bet_recommendations(analyses, field_size=12)
    probs_in_order = [a.combined_probability for a in analyses]
    assert probs_in_order == sorted(probs_in_order, reverse=True), (
        f"final order must be calibrated_prob descending; got {probs_in_order}"
    )
    # rank-1 = MKT_FAV (0.40), rank-2 = BANKER_B (0.30), rank-3 = BANKER_A (0.15)
    assert analyses[0].horse.name == 'MKT_FAV'
    assert analyses[1].horse.name == 'BANKER_B_MED_CAL'
    assert analyses[2].horse.name == 'BANKER_A_LOW_CAL'


def test_banker_win_fires_only_when_banker_horse_is_rank_1_by_cal_prob():
    """BANKER WIN bet type is gated on rank-1 AND BANKER reasoning.
    Under the new contract, BANKER WIN fires only when the BANKER horse
    is naturally rank-1 by calibrated_prob — no longer artificially
    promoted there by rule-engine override."""
    # BANKER horse has the highest cal_prob → bet_recommendation = BANKER WIN
    fixed_probs = [0.50, 0.20]
    with patch('core.recommender.calibrate_probabilities', return_value=fixed_probs):
        analyses = [
            _make_analysis(1, 'BANKER_FAV', 2.0, 70.0,
                           reasoning=['BANKER (7/8 signals)'],
                           rank=1, combined_prob=0.50),
            _make_analysis(2, 'OTHER', 5.0, 50.0,
                           reasoning=['Average'], rank=2,
                           combined_prob=0.20),
        ]
        assign_bet_recommendations(analyses, field_size=8)
    assert analyses[0].horse.name == 'BANKER_FAV'
    # WIN bet types: BANKER_FAV gets BANKER WIN since it's both rank-1
    # AND has BANKER reasoning. Whether the race-level type comes out
    # as WIN or QIN depends on conviction gates — the assertion here is
    # that IF a WIN bet fires on rank-1, it's the BANKER variant.
    if analyses[0].bet_recommendation in ('WIN', 'BANKER WIN'):
        assert analyses[0].bet_recommendation == 'BANKER WIN', (
            f"rank-1 with BANKER reasoning must get BANKER WIN, "
            f"got {analyses[0].bet_recommendation}"
        )
