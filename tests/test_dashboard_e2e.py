"""End-to-end: build_race_data → generate_html, no network."""
import json
import re
import shutil
import subprocess
import pytest
from unittest.mock import MagicMock

from dashboard.generator import build_race_data
from dashboard.template import generate_html


def _mock_horse(number):
    h = MagicMock()
    h.number = number
    h.name = f'Horse{number}'
    h.jockey = f'J{number}'
    h.trainer = f'T{number}'
    h.draw = number
    h.weight = 130
    h.gear = ''
    h.rating = 60
    h.win_odds = 5.0 + number
    h.place_odds = 2.0 + number * 0.3
    h.last_6 = ''
    h.age = 4
    h.horse_weight = 1100
    h.days_since_last_run = 14
    h.rating_change = 0
    h.running_style = 'Stalker'
    return h


def _mock_race(num):
    race = MagicMock()
    race.race_number = num
    race.venue = 'Sha Tin'
    race.time = '13:00'
    race.distance = '1200'
    race.class_type = 'Class 4'
    race.going = 'Good'
    race.course = 'A'
    race.date = '2026/04/27'
    race.horses = [_mock_horse(i + 1) for i in range(8)]
    return race


def _mock_analysis(rank, horse):
    a = MagicMock()
    a.horse = horse
    a.rule_score = 60.0
    a.combined_probability = 0.125
    a.value_score = 12.5
    a.expected_return = 0.0
    a.bet_recommendation = 'QIN' if rank < 4 else 'SKIP'
    a.combined_confidence = 'MEDIUM'
    a.reasoning = ['Test reasoning']
    a.ml_prediction = MagicMock()
    a.ml_prediction.win_probability = 0.12
    a.ml_prediction.place_probability = 0.30
    return a


def _make_mocks():
    """Return (db_mock, odds_db_mock) suitable for build_race_data."""
    db = MagicMock()
    db.get_horse_going_form.return_value = None
    db.get_jockey_horse_stats.return_value = None
    db.get_distance_best_time.return_value = None
    db.get_distance_record.return_value = None
    db.get_horse_format_history.return_value = []
    # db.conn.execute used for real dividend lookup — return no rows
    db.conn.execute.return_value.fetchall.return_value = []

    odds_db = MagicMock()
    odds_db.get_movement_summary.return_value = {}
    # _compute_odds_age_minutes calls odds_db._get_conn()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = (None,)
    odds_db._get_conn.return_value = mock_conn

    return db, odds_db


def test_e2e_generator_to_template():
    """build_race_data produces correct fields; generate_html renders them."""
    db, odds_db = _make_mocks()
    race = _mock_race(1)
    analyses = [_mock_analysis(i, h) for i, h in enumerate(race.horses)]

    race_data = build_race_data(race, analyses, '2026/04/27', db, odds_db, analyzer=None)

    # New per-horse fields present
    h0 = race_data['horses'][0]
    for field in ('pure_prob', 'market_prob', 'edge_pp', 'ev_win', 'overlay', 'edge_fires', 'model_rank'):
        assert field in h0, f'missing per-horse field: {field}'

    # Killed fields absent
    assert 'combined_score' not in h0
    assert 'strategy_confidence' not in race_data
    assert 'is_best_ev' not in race_data
    assert 'is_worst_ev' not in race_data

    # Odds age exposed at race level
    assert 'odds_age_minutes' in race_data

    # Render HTML
    race_json = json.dumps([race_data])
    html = generate_html(race_json, '2026-04-27 13:00:00', '2026/04/27')

    assert html.startswith('<!DOCTYPE')
    assert 'id="hero"' in html
    assert 'id="strip"' in html
    assert 'alerts-rail' in html

    # New fields land in embedded JSON
    assert '"pure_prob":' in html

    # Killed fields not present
    assert 'is_best_ev' not in html
    assert 'combined_score' not in html


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_e2e_renders_valid_js():
    """Rendered <script> block passes node --check (catches apostrophe/newline bugs)."""
    db, odds_db = _make_mocks()
    race = _mock_race(1)
    analyses = [_mock_analysis(i, h) for i, h in enumerate(race.horses)]

    race_data = build_race_data(race, analyses, '2026/04/27', db, odds_db, analyzer=None)
    html = generate_html(json.dumps([race_data]), '2026-04-27 13:00:00', '2026/04/27')

    scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
    assert scripts, 'No <script> block found in rendered HTML'
    body = max(scripts, key=len)

    result = subprocess.run(
        ['node', '--check', '-'],
        input=body, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, f'node --check failed:\n{result.stderr[:500]}'
