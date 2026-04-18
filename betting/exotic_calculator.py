#!/usr/bin/env python3
"""Exotic bet value calculator — exacta, trifecta, quinella, first four.

Uses calibrated win probabilities to price exotic combinations, then
compares against market-implied probabilities to find mispriced bets.
This is where the Benter edge lives — crowds price exotics poorly.
"""

import logging
from itertools import permutations
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _conditional_prob(probs: List[float], indices: List[int]) -> float:
    """Compute conditional probability: P(i1 wins, i2 second, i3 third, ...).

    Uses sequential elimination:
      P(A 1st) × P(B 1st | A removed) × P(C 1st | A,B removed) × ...
    """
    result = 1.0
    removed_sum = 0.0
    for idx in indices:
        denom = 1.0 - removed_sum
        if denom <= 0:
            return 0.0
        result *= probs[idx] / denom
        removed_sum += probs[idx]
    return result


def price_exactas(probs: List[float], numbers: List[int],
                  top_n: int = 10) -> List[Dict]:
    """Price all exacta combinations and return top N by probability.

    Exacta: pick 1st and 2nd in correct order.
    """
    n = len(probs)
    combos = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            prob = _conditional_prob(probs, [i, j])
            if prob > 0:
                combos.append({
                    'type': 'EXACTA',
                    'horses': [numbers[i], numbers[j]],
                    'probability': prob,
                    'min_odds': 1.0 / prob if prob > 0 else 999,
                })
    combos.sort(key=lambda x: x['probability'], reverse=True)
    return combos[:top_n]


def price_quinellas(probs: List[float], numbers: List[int],
                    top_n: int = 5) -> List[Dict]:
    """Price quinella combinations (1st and 2nd in any order)."""
    n = len(probs)
    combos = []
    for i in range(n):
        for j in range(i + 1, n):
            prob = (_conditional_prob(probs, [i, j]) +
                    _conditional_prob(probs, [j, i]))
            if prob > 0:
                combos.append({
                    'type': 'QUINELLA',
                    'horses': sorted([numbers[i], numbers[j]]),
                    'probability': prob,
                    'min_odds': 1.0 / prob if prob > 0 else 999,
                })
    combos.sort(key=lambda x: x['probability'], reverse=True)
    return combos[:top_n]


def price_trifectas(probs: List[float], numbers: List[int],
                    top_n: int = 10) -> List[Dict]:
    """Price trifecta combinations (1st, 2nd, 3rd in order)."""
    n = len(probs)
    combos = []
    # Only compute for top 8 most likely horses to keep it fast
    top_indices = sorted(range(n), key=lambda i: probs[i], reverse=True)[:8]
    for perm in permutations(top_indices, 3):
        prob = _conditional_prob(probs, list(perm))
        if prob > 0:
            combos.append({
                'type': 'TRIFECTA',
                'horses': [numbers[p] for p in perm],
                'probability': prob,
                'min_odds': 1.0 / prob if prob > 0 else 999,
            })
    combos.sort(key=lambda x: x['probability'], reverse=True)
    return combos[:top_n]


def price_first4s(probs: List[float], numbers: List[int],
                  top_n: int = 5) -> List[Dict]:
    """Price First 4 combinations (1st through 4th in order)."""
    n = len(probs)
    combos = []
    # Only compute for top 6 most likely horses
    top_indices = sorted(range(n), key=lambda i: probs[i], reverse=True)[:6]
    for perm in permutations(top_indices, 4):
        prob = _conditional_prob(probs, list(perm))
        if prob > 0:
            combos.append({
                'type': 'FIRST4',
                'horses': [numbers[p] for p in perm],
                'probability': prob,
                'min_odds': 1.0 / prob if prob > 0 else 999,
            })
    combos.sort(key=lambda x: x['probability'], reverse=True)
    return combos[:top_n]


def find_value_exotics(model_probs: List[float], market_probs: List[float],
                       numbers: List[int], min_edge: float = 0.10) -> Dict[str, List[Dict]]:
    """Find exotic bets where model probability exceeds market-implied.

    Args:
        model_probs: calibrated probabilities from our model (sum to 1)
        market_probs: probabilities implied by win odds (sum to ~1)
        numbers: horse numbers
        min_edge: minimum edge required (0.10 = 10%)

    Returns:
        Dict with keys 'exacta', 'trifecta', 'quinella', 'first4',
        each containing list of value combos with edge %.
    """
    result = {}

    # Price with both model and market
    for bet_type, pricer, top_n in [
        ('exacta', price_exactas, 20),
        ('quinella', price_quinellas, 10),
        ('trifecta', price_trifectas, 20),
        ('first4', price_first4s, 10),
    ]:
        model_combos = pricer(model_probs, numbers, top_n=top_n)
        market_combos = pricer(market_probs, numbers, top_n=top_n * 2)

        # Build market lookup
        market_lookup = {}
        for mc in market_combos:
            key = tuple(mc['horses'])
            market_lookup[key] = mc['probability']

        value_bets = []
        for combo in model_combos:
            key = tuple(combo['horses'])
            market_prob = market_lookup.get(key, 0)
            if market_prob > 0:
                edge = combo['probability'] / market_prob - 1
                if edge >= min_edge:
                    combo['market_prob'] = market_prob
                    combo['market_odds'] = 1.0 / market_prob
                    combo['edge'] = edge
                    value_bets.append(combo)

        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        result[bet_type] = value_bets[:5]

    return result


def format_exotic_summary(exotics: Dict[str, List[Dict]],
                          horse_names: Dict[int, str] = None) -> str:
    """Format exotic value bets for display."""
    lines = []
    for bet_type in ['exacta', 'quinella', 'trifecta', 'first4']:
        bets = exotics.get(bet_type, [])
        if not bets:
            continue
        lines.append(f"\n  {bet_type.upper()} VALUE PICKS:")
        for b in bets:
            horses_str = ' → '.join(f'#{h}' for h in b['horses'])
            if horse_names:
                horses_str = ' → '.join(
                    f'#{h} {horse_names.get(h, "")}'.strip() for h in b['horses']
                )
            lines.append(
                f"    {horses_str}  |  "
                f"Model: {b['probability']*100:.2f}%  |  "
                f"Min odds: ${b['min_odds']:.0f}  |  "
                f"Edge: +{b['edge']*100:.0f}%"
            )
    return '\n'.join(lines) if lines else '  No exotic value found'
