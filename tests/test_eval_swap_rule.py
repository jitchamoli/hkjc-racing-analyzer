"""Tests for scripts/eval_swap_rule.py — the swap-rule backtest CLI.

Built after 2026-05-17 rank-5 swap incident as a pre-ship gate. Tests pin
the rule-application logic + the rank_out/box_size validation that catches
silent miscount bugs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_swap_rule import _apply_rank_swap_rule, _box_hit


def _horse(rank, number, win_odds=10.0, gear='B/TT'):
    return {'rank': rank, 'number': number, 'win_odds': win_odds, 'gear': gear}


class TestApplyRankSwapRule:
    def test_no_swap_when_no_candidate_meets_threshold(self):
        # rank-5 @ $20; threshold = $20 * 0.5 = $10; nothing under $10
        horses = [_horse(i, i, win_odds=20 + i) for i in range(1, 12)]
        result = _apply_rank_swap_rule(horses, rank_out=5, odds_ratio=0.5,
                                        require_gear=True)
        assert result is None

    def test_swap_fires_when_lower_rank_qualifies(self):
        horses = [
            _horse(1, 1, win_odds=3.0),
            _horse(2, 2, win_odds=5.0),
            _horse(3, 3, win_odds=8.0),
            _horse(4, 4, win_odds=12.0),
            _horse(5, 5, win_odds=40.0),   # rank 5 longshot
            _horse(6, 6, win_odds=10.0),   # rank 6 → 10 <= 40*0.5=20 ✅
            _horse(7, 7, win_odds=15.0),
        ]
        result = _apply_rank_swap_rule(horses, rank_out=5, odds_ratio=0.5,
                                        require_gear=True)
        assert result is not None
        assert result['fired'] is True
        assert result['swap_out_num'] == 5
        assert result['swap_in_num'] == 6  # the qualifying lower rank

    def test_require_gear_filters_no_gear_horses(self):
        horses = [_horse(i, i, win_odds=30.0) for i in range(1, 6)]
        horses[5:5] = [{'rank': 5, 'number': 5, 'win_odds': 40.0, 'gear': 'B'}]
        # Add a rank-6 horse with NO gear that would otherwise qualify
        horses.append({'rank': 6, 'number': 6, 'win_odds': 5.0, 'gear': ''})
        result = _apply_rank_swap_rule(horses, rank_out=5, odds_ratio=0.5,
                                        require_gear=True)
        # No qualifying gear horse → no swap
        assert result is None

    def test_picks_lowest_odds_candidate_when_multiple_qualify(self):
        horses = [
            _horse(1, 1, win_odds=3.0),
            _horse(2, 2, win_odds=5.0),
            _horse(3, 3, win_odds=8.0),
            _horse(4, 4, win_odds=12.0),
            _horse(5, 5, win_odds=40.0),
            _horse(6, 6, win_odds=15.0),  # qualifies (15 <= 20)
            _horse(7, 7, win_odds=8.0),   # qualifies (8 <= 20) — shorter odds = pick this
            _horse(8, 8, win_odds=12.0),
        ]
        result = _apply_rank_swap_rule(horses, rank_out=5, odds_ratio=0.5,
                                        require_gear=True)
        assert result['swap_in_num'] == 7  # shortest qualifying odds

    def test_returns_none_when_field_too_small(self):
        # Field of 4 < rank_out=5
        horses = [_horse(i, i) for i in range(1, 5)]
        result = _apply_rank_swap_rule(horses, rank_out=5, odds_ratio=0.5,
                                        require_gear=True)
        assert result is None

    def test_returns_none_when_rank_out_horse_has_no_odds(self):
        horses = [_horse(i, i) for i in range(1, 5)]
        horses.append({'rank': 5, 'number': 5, 'win_odds': 0, 'gear': 'B'})
        horses += [_horse(i, i) for i in range(6, 8)]
        result = _apply_rank_swap_rule(horses, rank_out=5, odds_ratio=0.5,
                                        require_gear=True)
        assert result is None


class TestBoxHit:
    def test_full_top3_in_box_is_hit(self):
        assert _box_hit([1, 2, 3, 4, 5], [1, 2, 3]) is True

    def test_top3_partial_is_miss(self):
        assert _box_hit([1, 2, 3, 4, 5], [1, 2, 6]) is False

    def test_empty_actual_returns_false(self):
        assert _box_hit([1, 2, 3, 4, 5], []) is False

    def test_top3_in_any_order_hits(self):
        assert _box_hit([5, 4, 3, 2, 1], [1, 2, 3]) is True
