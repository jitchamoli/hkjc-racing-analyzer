"""Pure-prob and edge math with 6-gate filtering.

pure_prob = softmax(0.6·sigmoid(rule_score) + 0.4·ml_prob)
— our independent opinion BEFORE the market blend.

Feeds edge_pp, ev_win, overlay; gated by EV + prob floor + edge gap + rank
+ odds freshness + model integrity. See
docs/superpowers/specs/2026-04-24-dashboard-redesign-design.md §4.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class EdgeGates:
    """Gate thresholds. Defaults per spec §4."""
    min_ev: float = 0.05
    min_prob: float = 0.10
    min_edge_pp: float = 5.0
    max_model_rank: int = 2
    max_odds_age_min: int = 30


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rule_score_to_prob(rule_score: float) -> float:
    return _sigmoid(0.06 * (rule_score - 50))


def compute_pure_probs(horses: List[Dict]) -> List[float]:
    raw = [0.6 * _rule_score_to_prob(h['rule_score']) + 0.4 * h['ml_prob']
           for h in horses]
    max_raw = max(raw) if raw else 0
    exp = [math.exp(r - max_raw) for r in raw]
    total = sum(exp) or 1.0
    return [e / total for e in exp]


def compute_market_probs(horses: List[Dict]) -> List[float]:
    inv = [(1.0 / h['win_odds']) if h.get('win_odds', 0) > 0 else 0.0
           for h in horses]
    total = sum(inv) or 1.0
    return [v / total for v in inv]


def compute_edge_fields(pure: float, market: float, win_odds: float) -> Dict[str, float]:
    return {
        'edge_pp': (pure - market) * 100.0,
        'ev_win': pure * win_odds - 1.0 if win_odds > 0 else -1.0,
        'overlay': pure * win_odds if win_odds > 0 else 0.0,
    }


def apply_edge_gates(
    ev_win: float, pure_prob: float, edge_pp: float,
    model_rank: int, odds_age_min: float, integrity_ok: bool,
    gates: EdgeGates = None,
) -> Tuple[bool, List[str]]:
    gates = gates or EdgeGates()
    failed = []
    if ev_win < gates.min_ev: failed.append('G1')
    if pure_prob < gates.min_prob: failed.append('G2')
    if edge_pp < gates.min_edge_pp: failed.append('G3')
    if model_rank > gates.max_model_rank: failed.append('G4')
    if odds_age_min > gates.max_odds_age_min: failed.append('G5')
    if not integrity_ok: failed.append('G6')
    return (len(failed) == 0, failed)
