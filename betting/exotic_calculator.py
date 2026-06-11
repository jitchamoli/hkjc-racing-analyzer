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
                    top_n: int = 5,
                    place_probs: List[float] = None,
                    draws: List[int] = None,
                    styles: List[str] = None) -> List[Dict]:
    """Price quinella combinations with place-model boost and pace pairing.

    Enhanced quinella pricing that:
    1. Uses win probabilities for base pricing (conditional elimination)
    2. Boosts consistent placers using dedicated place model probabilities
    3. Boosts leader+closer pace combos (classic quinella pairing)
    4. Boosts adjacent good-draw pairs (draw clustering in tight HK tracks)
    """
    n = len(probs)
    combos = []
    for i in range(n):
        for j in range(i + 1, n):
            # Base quinella probability from win model
            base_prob = (_conditional_prob(probs, [i, j]) +
                         _conditional_prob(probs, [j, i]))
            if base_prob <= 0:
                continue

            boost = 1.0

            # Place model boost: consistent placers the win model underweights
            # A horse with high place_prob but low win_prob is quinella gold
            if place_probs:
                for idx in (i, j):
                    if probs[idx] > 0:
                        place_win_ratio = place_probs[idx] / probs[idx]
                        # Ratio > 3.5 means horse places way more than it wins
                        if place_win_ratio > 3.5:
                            boost *= 1.15  # 15% boost per consistent placer

            # Pace pairing boost: leader + closer = classic quinella
            if styles and len(styles) > max(i, j):
                s_i = (styles[i] or '').lower()
                s_j = (styles[j] or '').lower()
                is_pace_pair = (
                    ('leader' in s_i and 'closer' in s_j) or
                    ('closer' in s_i and 'leader' in s_j) or
                    ('leader' in s_i and 'midfield' in s_j) or
                    ('midfield' in s_i and 'leader' in s_j)
                )
                if is_pace_pair:
                    boost *= 1.10  # 10% boost for pace diversity

            # Draw clustering boost: adjacent draws in good positions
            if draws and len(draws) > max(i, j):
                d_i, d_j = draws[i], draws[j]
                if d_i > 0 and d_j > 0:
                    draw_gap = abs(d_i - d_j)
                    if draw_gap <= 2 and min(d_i, d_j) <= 6:
                        boost *= 1.08  # 8% boost for adjacent good draws

            boost = min(boost, 1.35)  # cap to prevent inflated edge detection
            prob = base_prob * boost
            combos.append({
                'type': 'QUINELLA',
                'horses': sorted([numbers[i], numbers[j]]),
                'probability': prob,
                'min_odds': 1.0 / prob if prob > 0 else 999,
                'boost': boost,
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
                       numbers: List[int], min_edge: float = 0.10,
                       place_probs: List[float] = None,
                       draws: List[int] = None,
                       styles: List[str] = None) -> Dict[str, List[Dict]]:
    """Find exotic bets where model probability exceeds market-implied.

    Args:
        model_probs: calibrated probabilities from our model (sum to 1)
        market_probs: probabilities implied by win odds (sum to ~1)
        numbers: horse numbers
        min_edge: minimum edge required (0.10 = 10%)
        place_probs: place model probabilities (for quinella placer boost)
        draws: draw positions (for quinella draw clustering)
        styles: running styles (for quinella pace pairing)

    Returns:
        Dict with keys 'exacta', 'trifecta', 'quinella', 'first4',
        each containing list of value combos with edge %.
    """
    result = {}

    # Price with both model and market
    # Quinella uses enhanced pricing with place model + pace + draws
    for bet_type, top_n in [
        ('exacta', 20),
        ('quinella', 10),
        ('trifecta', 20),
        ('first4', 10),
    ]:
        # Market lookup must cover EVERY model combo we want to flag as
        # a value bet, not just the market's own top-N. Otherwise a
        # model-favored long-shot pair gets market_prob=0 and is silently
        # skipped — under-reporting genuine edges. For quinella the full
        # combo space is C(N,2) ≤ 105 for typical fields; for exacta
        # N*(N-1) ≤ 210. Safe to enumerate exhaustively.
        n_horses = len(numbers)
        if bet_type == 'quinella':
            model_combos = price_quinellas(model_probs, numbers, top_n=top_n,
                                           place_probs=place_probs,
                                           draws=draws, styles=styles)
            # Market leg = RAW Harville price (the actual pool-implied price).
            # Do NOT apply the model's place/pace/draw boosts here: the boost's
            # place_win_ratio uses the base-prob denominator, so boosting the
            # market (market_probs in the denominator) distorts the edge
            # asymmetrically — and the market has no place model. Edge =
            # model_boosted / market_raw is the true overlay (audit pass-3).
            full_quinella = (n_horses * (n_horses - 1)) // 2
            market_combos = price_quinellas(market_probs, numbers, top_n=full_quinella)
        elif bet_type == 'exacta':
            model_combos = price_exactas(model_probs, numbers, top_n=top_n)
            full_exacta = n_horses * (n_horses - 1)
            market_combos = price_exactas(market_probs, numbers, top_n=full_exacta)
        elif bet_type == 'trifecta':
            # Trifecta combo space (N*(N-1)*(N-2)) gets large; keep a 5x cap
            # so we cover any realistic edge zone without exhaustive pricing.
            model_combos = price_trifectas(model_probs, numbers, top_n=top_n)
            market_combos = price_trifectas(market_probs, numbers, top_n=top_n * 5)
        else:
            model_combos = price_first4s(model_probs, numbers, top_n=top_n)
            market_combos = price_first4s(market_probs, numbers, top_n=top_n * 5)

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
            elif bet_type in ('trifecta', 'first4'):
                # Combo not in expanded market top-N — only possible for
                # trifecta/first4 where exhaustive pricing is too expensive.
                # Flag rather than drop so callers can see we lacked market data.
                combo['market_prob'] = None
                combo['no_market_data'] = True
                # Don't include in value_bets — without market_prob we can't
                # measure edge, but the flag preserves observability via logs.

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
