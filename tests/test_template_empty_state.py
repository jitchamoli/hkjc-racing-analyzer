"""Template renders honest-day message when no race clears MED+ tier."""
import json
import re
from dashboard.template import generate_html


def _race(num, tier='SKIP', has_edge=False):
    return {
        'race_number': num, 'venue': 'Sha Tin', 'time': '13:00',
        'distance': 1200, 'class_type': 'Class 4', 'going': 'Good',
        'course': 'A', 'conviction_tier': tier, 'has_edge': has_edge,
        'top_2_share': 0.15, 'top_3_share': 0.22,
        'horses': [{'number': 1, 'name': 'Test', 'jockey': 'TJ', 'trainer': 'TT',
                    'draw': 1, 'win_odds': 5.0, 'rank': 1, 'pure_prob': 0.2,
                    'market_prob': 0.2, 'edge_pp': 0, 'ev_win': 0, 'overlay': 1.0,
                    'edge_fires': False, 'model_rank': 1, 'reasoning': [],
                    'calibrated_prob': 0.2, 'rule_score': 50, 'ml_prob': 0.2,
                    'combined_score': 20, 'expected_return': 0,
                    'bet_recommendation': 'SKIP', 'confidence': '',
                    'place_prob': 0.3, 'last_6': '', 'age': 4,
                    'horse_weight': 1100, 'days_since_last_run': 14,
                    'rating_change': 0, 'going_form': None,
                    'jockey_horse_stats': None, 'distance_pb': None,
                    'distance_record': None, 'odds_movement': {},
                    'steam_move': {'is_steam': False}, 'cold_horse': {'is_cold': False},
                    'weight': 130, 'gear': '', 'rating': 60, 'place_odds': 2.5}],
        'speed_map': {}, 'alerts': [], 'tier_stats': {},
    }


def test_empty_state_renders_honest_day_message():
    races = [_race(i) for i in range(1, 12)]
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')
    assert 'No bets meet the threshold today' in html


def test_hero_renders_edge_race():
    races = [_race(1), _race(5, tier='EDGE', has_edge=True)]
    races[1]['horses'][0]['edge_fires'] = True
    races[1]['horses'][0]['edge_pp'] = 10.0
    races[1]['horses'][0]['ev_win'] = 0.83
    races[1]['top_2_share'] = 0.45
    races[1]['top_3_share'] = 0.55
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')
    assert 'EDGE' in html


def test_strip_shows_all_11_races():
    races = [_race(i) for i in range(1, 12)]
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')
    for i in range(1, 12):
        assert re.search(rf'\bR{i}\b', html), f'R{i} missing from strip'


def test_killed_fields_not_referenced():
    """Legacy fields (is_best_ev, grade-aplus, etc.) should not drive UI."""
    races = [_race(1)]
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')
    assert 'is_best_ev' not in html.lower()
    assert 'grade-aplus' not in html.lower()
