import math
import sqlite3

import numpy as np
import pytest

from scripts.calibration_report import eligible_race, summarize, atomic_json
from scripts.top4_diagnostic import meeting_folds, capture, settle_box, pair_ticket


def race():
    return {'race_number': 1, 'time': '13:00', 'conviction_tier': 'HIGH',
            'bet_ticket': {'frozen_at': '2026-10-11T12:30:00'},
            'horses': [{'number': i, 'rank': i, 'calibrated_prob': .2}
                       for i in range(1, 6)],
            'actual_results': [{'number': i, 'position': i} for i in range(1, 6)]}


def test_calibration_complete_and_dead_heat():
    r = race()
    rows, reason = eligible_race('2026/10/11', r)
    assert reason is None
    assert sum(x['outcome'] for x in rows) == 1
    r['actual_results'][1]['position'] = 1
    rows, _ = eligible_race('2026/10/11', r)
    assert [x['outcome'] for x in rows[:2]] == [.5, .5]


@pytest.mark.parametrize('change', ['late', 'missing', 'invalid', 'quarantine'])
def test_calibration_excludes_unsafe_races(change):
    r = race()
    if change == 'late':
        r['bet_ticket']['frozen_at'] = '2026-10-11T13:00:00'
    elif change == 'missing':
        r['actual_results'].pop()
    elif change == 'invalid':
        r['horses'][0]['calibrated_prob'] = math.nan
    else:
        r['bet_ticket']['post_off_unverified'] = True
    assert eligible_race('2026/10/11', r)[0] == []


def test_calibration_band_and_brier():
    rows, _ = eligible_race('2026/10/11', race())
    report = summarize(rows)
    group = report['groups'][0]
    assert group['band'] == '[0.20,0.30)'
    assert group['mean_probability'] == pytest.approx(.2)
    assert group['observed'] == pytest.approx(.2)
    assert group['brier'] == pytest.approx(.16)
    assert group['insufficient_evidence']
    assert report['race_log_loss'] == pytest.approx(-math.log(.2))
    assert summarize(rows[:1])['race_log_loss'] is None


def test_aware_hkt_timestamp_and_duplicate_runner():
    r = race()
    r['bet_ticket']['frozen_at'] = '2026-10-11T04:30:00+00:00'
    assert eligible_race('2026/10/11', r)[1] is None
    r['horses'][1]['number'] = 1
    assert eligible_race('2026/10/11', r)[0] == []


def test_meeting_folds_never_split_date():
    dates = np.array(['2026/01/%02d' % d for d in range(1, 11) for _ in range(6)])
    for train, val, test in meeting_folds(dates, 2):
        assert max(dates[train]) < min(dates[val])
        assert max(dates[val]) < min(dates[test])
        assert not set(dates[train]) & set(dates[test])


def test_capture_and_official_box_dividend():
    assert capture([1, 2, 3, 4, 5], {1: 1, 2: 2, 3: 3, 4: 4, 5: 5})['top4_in5']
    assert settle_box([1, 2, 3, 4, 5], [('1,2,3', 200)], 3) == (100, 200)
    assert settle_box([1, 2, 3, 4, 5], [('1,2,3,4', 500)], 4) == (50, 500)


def test_frozen_pair_preserves_units_and_paper():
    bet = {'label': 'BOX TRIO-5', 'pool': 'TRIO', 'unit_stake': 20,
           'paper': True, 'combos': 10,
           'legs': [{'number': n, 'rank': n} for n in range(1, 6)]}
    pair = pair_ticket(bet, [6, 2, 3, 4, 5, 1], [('1,2,3', 200)])
    assert pair['paper']
    assert pair['production'] == (200, 400)
    assert pair['ranker'] == (200, 0)
    assert bet['legs'][0]['number'] == 1


def test_unsupported_pair_is_not_invented():
    assert pair_ticket({'pool': 'WIN'}, [1, 2], []) is None


def test_atomic_report_is_idempotent(tmp_path):
    path = tmp_path / 'audit.json'
    atomic_json(path, {'n': 1})
    first = path.read_bytes()
    atomic_json(path, {'n': 1})
    assert path.read_bytes() == first
    assert not path.with_name(path.name + '.tmp').exists()


def test_research_run_does_not_write_models_or_tickets(tmp_path, monkeypatch):
    import scripts.top4_diagnostic as research
    db = tmp_path / 'history.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE race_dividends (race_date, race_number, pool, combination, dividend)')
    conn.commit()
    conn.close()
    model = tmp_path / 'production.pkl'
    model.write_bytes(b'unchanged production')
    before = db.read_bytes()
    meta = [{'race_date': f'2026/01/{day:02d}', 'race_number': 1,
             'horse_number': n, 'finish_position': n}
            for day in range(1, 11) for n in range(1, 6)]
    X = np.array([[r['horse_number']] for r in meta], dtype=float)
    groups = np.array([r['race_date'] + '-1' for r in meta], dtype=object)
    y = np.array([r['finish_position'] == 1 for r in meta], dtype=int)
    monkeypatch.setattr(research.config, 'DB_PATH', db)
    monkeypatch.setattr(research.config, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(research, 'build_training_data_from_db',
                        lambda **kwargs: (X, y, np.ones(len(y)), y, groups, meta))
    result = research.run(2, 1)
    assert result['races']
    assert db.read_bytes() == before
    assert model.read_bytes() == b'unchanged production'
