"""place_odds must freeze at the off, like win_odds.

_FROZEN_PREDICTION_KEYS_HORSE was written when place_odds was always 0.0
(the odds pages had been dead since May), so nobody noticed it was missing
from the list. With place odds live from 2026-09-16, a finished race stored
a FROZEN win price beside a still-updating place price:

    R7 #1 LOVE TOGETHER   win $10.0 (frozen 22:07)   place $1.8 (live 22:21)

which is not a coherent market and corrupts any post-race read that compares
the two (place-side products, EDGE PLACE tracking, the dashboard card).
"""
from dashboard.generator import _FROZEN_PREDICTION_KEYS_HORSE, _merge_with_existing


def test_place_odds_is_in_the_post_off_freeze_list():
    assert 'place_odds' in _FROZEN_PREDICTION_KEYS_HORSE


def test_a_finished_races_place_odds_survive_a_later_refresh(tmp_path, monkeypatch):
    import dashboard.generator as g
    monkeypatch.setattr(g, '_post_time_passed', lambda *a, **k: True)   # race has off'd

    path = tmp_path / 'race_data.json'
    old = [{'race_number': 1, 'time': '19:10', 'horses': [
        {'number': 7, 'rank': 1, 'win_odds': 10.0, 'place_odds': 3.4}]}]
    import json
    path.write_text(json.dumps({'races': old}))

    fresh = [{'race_number': 1, 'time': '19:10', 'horses': [
        {'number': 7, 'rank': 1, 'win_odds': 7.3, 'place_odds': 1.8}]}]
    merged = _merge_with_existing(str(path), fresh, race_date='2026/09/16')

    h = merged[0]['horses'][0]
    assert h['win_odds'] == 10.0, 'precondition: win odds already froze'
    assert h['place_odds'] == 3.4, 'place odds drifted after the off'
