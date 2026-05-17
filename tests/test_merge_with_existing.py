"""Locks the freeze + merge behavior in dashboard/generator.py.

Audit 2026-05-17 caught 4 holes in the original freeze logic:
- top_2_share / top_3_share weren't frozen (race-level shape desynced)
- suggested_stake was recomputed after freeze (no-op)
- edge_horse_* weren't frozen (banner could name wrong horse)
- odds_snapshots got poisoned by post-race writes

These tests pin the fix so a future refactor cannot silently undo it.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from dashboard.generator import (
    _merge_with_existing,
    _FROZEN_PREDICTION_KEYS_RACE,
    _FROZEN_PREDICTION_KEYS_HORSE,
    _RESULT_KEYS_RACE,
    _RESULT_KEYS_HORSE,
)


def _write_old(json_path: str, races: list) -> None:
    with open(json_path, 'w') as f:
        json.dump(races, f)


@pytest.fixture
def tmp_json():
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)
    wr = path.replace('.json', '_with_results.json')
    if os.path.exists(wr):
        os.unlink(wr)


def test_freeze_preserves_conviction_tier_on_finished_race(tmp_json):
    old = [{
        'race_number': 1,
        'results_available': True,
        'conviction_tier': 'MED',   # original pre-race tier
        'has_edge': False,
        'horses': [{'number': 8, 'rank': 1, 'calibrated_prob': 0.21, 'win_odds': 5.0}],
    }]
    _write_old(tmp_json, old)
    new = [{
        'race_number': 1,
        'conviction_tier': 'EDGE',   # bogus post-race recompute
        'has_edge': True,
        'horses': [{'number': 8, 'rank': 9, 'calibrated_prob': 0.05, 'win_odds': 20.0}],
    }]
    merged = _merge_with_existing(tmp_json, new)
    assert merged[0]['conviction_tier'] == 'MED'
    assert merged[0]['has_edge'] is False


def test_freeze_preserves_shape_fields(tmp_json):
    """Audit fix HIGH-1 — top_2_share/top_3_share must move with frozen probs."""
    old = [{
        'race_number': 1,
        'results_available': True,
        'top_2_share': 0.32,
        'top_3_share': 0.43,
        'horses': [{'number': 1, 'rank': 1, 'calibrated_prob': 0.20, 'win_odds': 5.0}],
    }]
    _write_old(tmp_json, old)
    new = [{
        'race_number': 1,
        'top_2_share': 0.55,  # post-race recompute (bogus)
        'top_3_share': 0.70,
        'horses': [{'number': 1, 'rank': 1, 'calibrated_prob': 0.15, 'win_odds': 7.0}],
    }]
    merged = _merge_with_existing(tmp_json, new)
    assert merged[0]['top_2_share'] == 0.32
    assert merged[0]['top_3_share'] == 0.43


def test_freeze_preserves_suggested_stake(tmp_json):
    old = [{
        'race_number': 1,
        'results_available': True,
        'suggested_stake': {'amount': 100.0, 'pct': 1.0, 'method': 'tier-fraction (MED)'},
        'bankroll': 10000,
        'horses': [{'number': 1, 'rank': 1}],
    }]
    _write_old(tmp_json, old)
    new = [{
        'race_number': 1,
        'suggested_stake': {'amount': 300.0, 'pct': 3.0, 'method': 'tier-fraction (EDGE)'},
        'bankroll': 10000,
        'horses': [{'number': 1, 'rank': 1}],
    }]
    merged = _merge_with_existing(tmp_json, new)
    assert merged[0]['suggested_stake']['amount'] == 100.0
    assert merged[0]['suggested_stake']['method'] == 'tier-fraction (MED)'


def test_freeze_preserves_edge_horse_identifiers(tmp_json):
    """Audit fix MED — banner must name horse whose frozen edge_fires=True."""
    old = [{
        'race_number': 1,
        'results_available': True,
        'edge_horse_rank': 2,
        'edge_horse_number': 7,
        'edge_horse_name': 'SUPER HONG KONG',
        'has_edge': True,
        'horses': [
            {'number': 8, 'rank': 1, 'edge_fires': False, 'edge_pp': -2.0},
            {'number': 7, 'rank': 2, 'edge_fires': True, 'edge_pp': 6.5},
        ],
    }]
    _write_old(tmp_json, old)
    # Post-race recompute names a different horse
    new = [{
        'race_number': 1,
        'edge_horse_rank': 2,
        'edge_horse_number': 1,
        'edge_horse_name': 'HAILTOTHEVICTORS',
        'has_edge': True,
        'horses': [
            {'number': 8, 'rank': 1, 'edge_fires': False, 'edge_pp': -5.0},
            {'number': 1, 'rank': 2, 'edge_fires': True, 'edge_pp': 8.0},
        ],
    }]
    merged = _merge_with_existing(tmp_json, new)
    assert merged[0]['edge_horse_number'] == 7
    assert merged[0]['edge_horse_name'] == 'SUPER HONG KONG'


def test_freeze_preserves_horse_level_predictions(tmp_json):
    old = [{
        'race_number': 1,
        'results_available': True,
        'horses': [
            {'number': 8, 'rank': 1, 'calibrated_prob': 0.21, 'win_odds': 5.0,
             'edge_pp': -2.0, 'edge_fires': False, 'bet_recommendation': 'QIN'},
            {'number': 1, 'rank': 2, 'calibrated_prob': 0.10, 'win_odds': 13.0,
             'edge_pp': -1.0, 'edge_fires': False, 'bet_recommendation': 'QIN'},
        ],
    }]
    _write_old(tmp_json, old)
    # Post-race recompute reshuffles + bogus EDGE
    new = [{
        'race_number': 1,
        'horses': [
            {'number': 1, 'rank': 1, 'calibrated_prob': 0.08, 'win_odds': 20.0,
             'edge_pp': 7.0, 'edge_fires': True, 'bet_recommendation': 'WIN'},
            {'number': 8, 'rank': 2, 'calibrated_prob': 0.20, 'win_odds': 5.5,
             'edge_pp': -5.0, 'edge_fires': False, 'bet_recommendation': 'QIN'},
        ],
    }]
    merged = _merge_with_existing(tmp_json, new)
    horses = merged[0]['horses']
    by_num = {h['number']: h for h in horses}
    assert by_num[8]['rank'] == 1
    assert by_num[8]['calibrated_prob'] == 0.21
    assert by_num[8]['edge_fires'] is False
    assert by_num[1]['rank'] == 2
    assert by_num[1]['edge_fires'] is False  # the bogus EDGE should NOT have leaked


def test_freeze_resorts_horses_by_frozen_rank(tmp_json):
    """Post-race reshuffle must not survive — sort by frozen rank, not new order."""
    old = [{
        'race_number': 1,
        'results_available': True,
        'horses': [
            {'number': 8, 'rank': 1},
            {'number': 1, 'rank': 2},
            {'number': 5, 'rank': 3},
        ],
    }]
    _write_old(tmp_json, old)
    new = [{
        'race_number': 1,
        'horses': [
            {'number': 5, 'rank': 1},  # reshuffled order
            {'number': 8, 'rank': 2},
            {'number': 1, 'rank': 3},
        ],
    }]
    merged = _merge_with_existing(tmp_json, new)
    numbers_in_order = [h['number'] for h in merged[0]['horses']]
    assert numbers_in_order == [8, 1, 5]


def test_unfinished_race_takes_new_predictions(tmp_json):
    """Live races MUST get the fresh predictions, not the old ones."""
    old = [{
        'race_number': 1,
        'results_available': False,   # not finished yet
        'conviction_tier': 'MED',
        'horses': [{'number': 8, 'rank': 1, 'calibrated_prob': 0.21, 'win_odds': 5.0}],
    }]
    _write_old(tmp_json, old)
    new = [{
        'race_number': 1,
        'conviction_tier': 'HIGH',
        'horses': [{'number': 8, 'rank': 1, 'calibrated_prob': 0.28, 'win_odds': 4.0}],
    }]
    merged = _merge_with_existing(tmp_json, new)
    assert merged[0]['conviction_tier'] == 'HIGH'
    assert merged[0]['horses'][0]['win_odds'] == 4.0


def test_result_fields_preserved_on_finished_race(tmp_json):
    """The original responsibility of _merge_with_existing — actual_results etc must survive."""
    old = [{
        'race_number': 1,
        'results_available': True,
        'actual_winner': 6,
        'actual_top3': [6, 10, 7],
        'accuracy': {'rank_1_win': False},
        'horses': [
            {'number': 8, 'rank': 1, 'actual_position': 6},
            {'number': 6, 'rank': 5, 'actual_position': 1},
        ],
    }]
    _write_old(tmp_json, old)
    new = [{
        'race_number': 1,
        'horses': [
            {'number': 8, 'rank': 1},
            {'number': 6, 'rank': 5},
        ],
    }]
    merged = _merge_with_existing(tmp_json, new)
    assert merged[0]['actual_winner'] == 6
    assert merged[0]['actual_top3'] == [6, 10, 7]
    by_num = {h['number']: h for h in merged[0]['horses']}
    assert by_num[8]['actual_position'] == 6
    assert by_num[6]['actual_position'] == 1


def test_frozen_keys_constants_are_complete():
    """Pin the explicit list so a future commit can't silently shrink it."""
    expected_race = {
        'conviction_tier', 'has_edge', 'suggested_stake', 'bankroll',
        'odds_age_minutes', 'top_2_share', 'top_3_share',
        'edge_horse_rank', 'edge_horse_number', 'edge_horse_name',
    }
    expected_horse = {
        'rank', 'win_odds', 'calibrated_prob', 'edge_pp', 'ev_win',
        'overlay', 'edge_fires', 'pure_prob', 'market_prob', 'model_rank',
        'bet_recommendation', 'confidence', 'expected_return', 'value_score',
    }
    assert expected_race.issubset(_FROZEN_PREDICTION_KEYS_RACE)
    assert expected_horse.issubset(_FROZEN_PREDICTION_KEYS_HORSE)


def test_no_overlap_between_result_and_frozen_keys():
    """Result fields (actual_*) and pre-race prediction fields should not collide."""
    assert _RESULT_KEYS_RACE.isdisjoint(_FROZEN_PREDICTION_KEYS_RACE)
    assert _RESULT_KEYS_HORSE.isdisjoint(_FROZEN_PREDICTION_KEYS_HORSE)
