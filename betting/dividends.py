#!/usr/bin/env python3
"""Quinella dividend estimation + expected-value calculation.

Uses the Harville independence model — standard industry approximation for
pari-mutuel pools. Assumes horses finish independently, which diverges from
reality when pace dynamics strongly correlate finish positions. Accurate
enough for strategy ranking and dividend ballpark estimates.
"""

import itertools
from typing import List, Optional, Tuple


HK_QUINELLA_TAKEOUT = 0.175  # HKJC quinella pool takeout rate


def harville_pair_prob(p_a: float, p_b: float) -> float:
    """P(unordered pair {A,B} finishes 1st-2nd) under Harville independence.

    P(A 1st) * P(B 2nd | A removed) + P(B 1st) * P(A 2nd | B removed)
    = P(A)*P(B)/(1-P(A)) + P(B)*P(A)/(1-P(B))
    """
    if p_a <= 0 or p_b <= 0 or p_a >= 1 or p_b >= 1:
        return 0.0
    return p_a * p_b / (1 - p_a) + p_b * p_a / (1 - p_b)


def estimate_quinella_dividend(market_probs: List[float], i: int, j: int,
                               takeout: float = HK_QUINELLA_TAKEOUT) -> Optional[float]:
    """Estimate HK quinella dividend per $1 stake for unordered pair (i, j).

    market_probs should be market-implied win probs (1/odds), normalized to sum=1.
    Returns None if probs invalid.
    """
    if i >= len(market_probs) or j >= len(market_probs):
        return None
    p_q = harville_pair_prob(market_probs[i], market_probs[j])
    if p_q <= 0:
        return None
    return (1.0 - takeout) / p_q


def quinella_bet_ev(model_probs: List[float], market_probs: List[float],
                    pair_indices: List[Tuple[int, int]],
                    stake_per_combo: float = 1.0,
                    takeout: float = HK_QUINELLA_TAKEOUT) -> dict:
    """Expected value of a multi-combo quinella bet.

    Args:
        model_probs: our win probability per horse (used for P(hit))
        market_probs: market-implied win probability per horse (used for dividend)
        pair_indices: list of (i, j) unordered horse pairs we bet on
        stake_per_combo: dollar stake per combination (default $1)
        takeout: pool takeout (HK default 17.5%)

    Returns:
        dict with keys: total_stake, expected_return, expected_profit, ev_pct,
                        hit_prob, avg_dividend, per_pair
    """
    total_stake = len(pair_indices) * stake_per_combo
    total_hit_prob = 0.0  # P(any of our pairs hits)
    expected_return = 0.0
    per_pair = []

    for (i, j) in pair_indices:
        p_hit = harville_pair_prob(model_probs[i], model_probs[j])
        div = estimate_quinella_dividend(market_probs, i, j, takeout)
        if div is not None:
            total_hit_prob += p_hit
            expected_return += p_hit * div * stake_per_combo
            per_pair.append({'pair': (i, j), 'p_hit': p_hit, 'dividend': div})

    expected_profit = expected_return - total_stake
    ev_pct = (expected_profit / total_stake * 100) if total_stake > 0 else 0
    avg_div = (sum(p['dividend'] for p in per_pair) / len(per_pair)
               if per_pair else 0)

    return {
        'total_stake': total_stake,
        'expected_return': expected_return,
        'expected_profit': expected_profit,
        'ev_pct': ev_pct,
        'hit_prob': total_hit_prob,
        'avg_dividend': avg_div,
        'per_pair': per_pair,
    }


def normalize_market_probs(win_odds: List[float]) -> List[float]:
    """Convert odds to normalized market-implied probabilities (remove overround)."""
    implied = [(1.0 / o) if o and o > 0 else 0.0 for o in win_odds]
    total = sum(implied)
    if total <= 0:
        n = len(win_odds)
        return [1.0 / n] * n if n > 0 else []
    return [p / total for p in implied]
