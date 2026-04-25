"""Pure-prob and edge math with 6-gate filtering.

pure_prob = softmax(0.6·sigmoid_anchored(rule_score, 1/N) + 0.4·ml_prob)
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
    """Gate thresholds. Defaults per spec §4.

    G3 (min_edge_pp = 5.0) is intentionally rare — it fires 0 times on the
    current 124-race holdout (mean edge_pp = -14pp; the model rarely
    disagrees with the market by ≥+5pp on a top-2 ranked horse). This is
    by design: ★ EDGE is a rare-cherry-on-top, not a regular tier.
    Don't lower the threshold without ≥200 disagreement points to avoid
    chasing noise (per CLAUDE.md "EDGE Gate Empirics 2026-04-25").
    """
    min_ev: float = 0.05
    min_prob: float = 0.10
    min_edge_pp: float = 5.0
    max_model_rank: int = 2
    max_odds_age_min: int = 30


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rule_score_to_prob(rule_score: float, field_size: int) -> float:
    # Base-rate-anchored sigmoid: a score=50 horse outputs ~1/field_size,
    # not 0.5. Mirrors ml/model.py:create_combined_prediction so the
    # edge math sees the same rule-prob distribution as the analyzer.
    base = 1.0 / max(field_size, 2)
    logit_base = math.log(base / (1.0 - base))
    return _sigmoid(0.06 * (rule_score - 50) + logit_base)


def compute_pure_probs(horses: List[Dict], field_size: int = None) -> List[float]:
    n = field_size if field_size is not None else len(horses)
    raw = [0.6 * _rule_score_to_prob(h['rule_score'], n) + 0.4 * h['ml_prob']
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
    model_rank: int, odds_age_min, integrity_ok: bool,
    gates: EdgeGates = None,
) -> Tuple[bool, List[str]]:
    gates = gates or EdgeGates()
    failed = []
    if ev_win < gates.min_ev: failed.append('G1')
    if pure_prob < gates.min_prob: failed.append('G2')
    if edge_pp < gates.min_edge_pp: failed.append('G3')
    if model_rank > gates.max_model_rank: failed.append('G4')
    # G5: stale odds invalidate EV. odds_age_min=None means "no snapshot
    # exists at all" — strictly worse than stale, so fail the gate.
    if odds_age_min is None or odds_age_min > gates.max_odds_age_min:
        failed.append('G5')
    if not integrity_ok: failed.append('G6')
    return (len(failed) == 0, failed)
