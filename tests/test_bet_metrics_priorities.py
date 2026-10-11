import csv
import json
from copy import deepcopy

import pytest

from scripts import stake_ab, validate_hk


def test_replay_metrics_distinguish_hits_from_profitable_tickets():
    acc, metrics = {}, {}
    validate_hk._book_bet(acc, 'BOX TRIO-5', 10, 10, True, 5, metrics)
    validate_hk._book_bet(acc, 'BOX TRIO-5', 10, 10, True, 20, metrics)
    validate_hk._book_bet(acc, 'BOX TRIO-5', 10, 10, False, 0, metrics)
    validate_hk._book_bet(acc, 'BOX TRIO-5', 10, 10, True, None, metrics)
    assert acc['BOX TRIO-5'] == [400, 250, 3, 1]
    assert metrics['BOX TRIO-5'] == {'tickets': 4, 'profitable': 1}


def test_replay_report_has_denominators_and_intervals(capsys):
    rows = [{'race_date': '2026/10/07', 'races': 4,
             'ticket_json': json.dumps({'TRIO': [400, 250, 3, 1]}),
             'metrics_json': json.dumps({'TRIO': {'tickets': 4, 'profitable': 1}})}]
    validate_hk._report_replay(rows, 'test')
    text = capsys.readouterr().out
    assert '3/4' in text
    assert '75.0%' in text
    assert '25.0%' in text
    assert '95% CI' in text
    assert 'LOWER BOUND' in text


def test_legacy_replay_does_not_invent_denominator(capsys):
    rows = [{'race_date': '2026/05/24', 'ticket_json': '{"TRIO": [100, 0, 0]}'}]
    validate_hk._report_replay(rows, 'legacy')
    assert 'unknown' in capsys.readouterr().out


@pytest.mark.parametrize('variant,excluded', [
    ('no_trio6', {'BOX TRIO-6'}),
    ('no_f4_saver', {'F4 SAVER'}),
    ('no_qin_protection', {'BOX QIN-4 PROTECTION'}),
    ('lean_ticket', {'BOX TRIO-6', 'F4 SAVER', 'BOX QIN-4 PROTECTION'}),
])
def test_reduction_variants_zero_only_targeted_bets(variant, excluded):
    fires = {'BOX TRIO-6': 20, 'F4 SAVER': 1, 'BOX QIN-4 PROTECTION': 6,
             'BOX TRIO-5': 10, 'F4 BOX-5': 5}
    race = {'conviction_tier': 'HIGH', stake_ab.MEETING_DATE_KEY: '2026/10/07'}
    original = deepcopy(fires)
    stakes = stake_ab.variant_stakes(variant, race, fires, 10000)
    assert stakes == {label: 0 if label in excluded else combos * 10
                      for label, combos in fires.items()}
    assert fires == original


def test_new_variants_do_not_backfill_burned_history():
    race = {stake_ab.MEETING_DATE_KEY: '2026/10/04'}
    assert stake_ab.variant_stakes('lean_ticket', race, {'BOX TRIO-5': 10}, 10000) == {
        'BOX TRIO-5': 0}


def test_comparison_uses_only_paired_meetings():
    rows = [
        {'race_date': '2026/10/04', 'variant': 'flat_min', 'stake_hkd': 100, 'pnl_hkd': -100},
        {'race_date': '2026/10/07', 'variant': 'flat_min', 'stake_hkd': 100, 'pnl_hkd': 50},
        {'race_date': '2026/10/07', 'variant': 'lean_ticket', 'stake_hkd': 100, 'pnl_hkd': 0},
    ]
    assert stake_ab._per_meeting_winner(rows, 'lean_ticket') == [0]


def test_box_capture_uniform_field_is_combinatorial_and_result_blind():
    race = {'horses': [{'number': n, 'calibrated_prob': 1/8,
                       'actual_position': n} for n in range(1, 9)]}
    bet = {'pool': 'TRIO', 'legs': [{'number': n} for n in range(1, 6)]}
    from math import comb
    assert stake_ab.box_capture_probability(bet, race) == pytest.approx(comb(5, 3)/comb(8, 3))
    for h in race['horses']:
        h['actual_position'] = 9-h['number']
    assert stake_ab.box_capture_probability(bet, race) == pytest.approx(comb(5, 3)/comb(8, 3))
    bet['pool'] = 'FIRST 4'
    assert stake_ab.box_capture_probability(bet, race) == pytest.approx(comb(5, 4)/comb(8, 4))


def test_capture_rejects_invalid_probabilities_and_duplicate_legs():
    race = {'horses': [{'number': 1, 'calibrated_prob': .6},
                      {'number': 2, 'calibrated_prob': .6}]}
    bet = {'pool': 'QUINELLA', 'legs': [{'number': 1}, {'number': 2}]}
    assert stake_ab.box_capture_probability(bet, race) is None
    race['horses'][1]['calibrated_prob'] = .4
    bet['legs'][1]['number'] = 1
    assert stake_ab.box_capture_probability(bet, race) is None


def test_stake_log_migrates_old_rows(tmp_path, monkeypatch):
    path = tmp_path / 'stake.csv'
    monkeypatch.setattr(stake_ab, 'LOG_PATH', path)
    old = {'race_date': '2026/10/04', 'race_number': 1, 'variant': 'flat_min',
           'bet_label': 'TRIO', 'combos': 10, 'stake_hkd': 100, 'hit': 0,
           'dividend_per_1': 0, 'pnl_hkd': -100}
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(old))
        w.writeheader()
        w.writerow(old)
    stake_ab._append_log([{**old, 'race_date': '2026/10/07',
                          'capture_probability': .2}], '2026/10/07')
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['capture_probability'] == ''
    assert rows[1]['capture_probability'] == '0.2'


def test_capture_report_groups_by_pool_and_ignores_zero_stakes(capsys):
    rows = [
        {'variant': 'flat_min', 'bet_label': 'F4 BOX-5', 'stake_hkd': 50,
         'capture_probability': .15, 'hit': 1},
        {'variant': 'flat_min', 'bet_label': 'F4 BOX-5', 'stake_hkd': 50,
         'capture_probability': .15, 'hit': 0},
        {'variant': 'flat_min', 'bet_label': 'F4 BOX-5', 'stake_hkd': 0,
         'capture_probability': .15, 'hit': 1},
    ]
    stake_ab.report_capture_calibration(rows)
    text = capsys.readouterr().out
    assert 'F4 BOX-5' in text and '15.0%' in text and '50.0%' in text
    assert 'uncalibrated' in text


def test_new_variants_report_paired_pnl_without_ship_verdict(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'ab.csv'
    monkeypatch.setattr(stake_ab, 'LOG_PATH', path)
    base = {'race_date': '2026/10/07', 'race_number': 1, 'variant': 'flat_min',
            'bet_label': 'F4 SAVER', 'combos': 1, 'stake_hkd': 10, 'hit': 0,
            'dividend_per_1': 0, 'pnl_hkd': -10, 'capture_probability': ''}
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=stake_ab.LOG_HEADERS)
        w.writeheader()
        w.writerows([base, {**base, 'variant': 'lean_ticket', 'stake_hkd': 0, 'pnl_hkd': 0}])
    stake_ab.report()
    text = capsys.readouterr().out
    assert 'lean_ticket' in text and 'paired' in text and '+10' in text
    assert not any('lean_ticket' in line and 'posterior=' in line for line in text.splitlines())


@pytest.mark.parametrize('unverified', [False, True])
def test_post_race_evaluation_logs_shadows_and_capture_from_frozen_legs(tmp_path, monkeypatch, unverified):
    import sqlite3
    monkeypatch.setattr(stake_ab, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(stake_ab, 'LOG_PATH', tmp_path / 'stake.csv')
    monkeypatch.setattr(stake_ab.config, 'DB_PATH', tmp_path / 'db.sqlite')
    with sqlite3.connect(stake_ab.config.DB_PATH) as c:
        c.execute('CREATE TABLE race_dividends (race_date, race_number, pool, combination, dividend)')
        c.execute("INSERT INTO race_dividends VALUES ('2026/10/07', 1, 'TRIO', '1,2,3', 200)")
    bet = {'pool': 'TRIO', 'label': 'BOX TRIO-6', 'combos': 20,
           'legs': [{'number': n, 'name': f'H{n}'} for n in range(1, 7)]}
    race = {'race_number': 1, 'results_available': True, 'conviction_tier': 'HIGH',
            'horses': [{'number': n, 'name': f'H{n}', 'rank': n,
                        'calibrated_prob': 1/8, 'actual_position': n}
                       for n in range(1, 9)],
            'bet_ticket': {'race_date': '2026/10/07', 'bets': [bet],
                           'post_off_unverified': unverified}}
    path = tmp_path / 'results' / 'race_data_20261007_with_results.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'races': [race]}))
    before = path.read_bytes()
    assert stake_ab.evaluate_date('2026/10/07', 10000) == 0
    with stake_ab.LOG_PATH.open() as f:
        rows = list(csv.DictReader(f))
    if unverified:
        assert rows == []
        assert path.read_bytes() == before
        return
    base = next(r for r in rows if r['variant'] == 'flat_min')
    reduced = next(r for r in rows if r['variant'] == 'no_trio6')
    assert base['hit'] == '1' and float(base['pnl_hkd']) == 0
    assert float(base['capture_probability']) == pytest.approx(20/56)
    assert float(reduced['stake_hkd']) == 0 and float(reduced['pnl_hkd']) == 0
    assert path.read_bytes() == before


def test_replay_metrics_persist_and_keep_legacy_rows(tmp_path, monkeypatch):
    path = tmp_path / 'replay.csv'
    path.write_text('race_date,races,ticket_json\n2026/05/24,1,"{}"\n')
    monkeypatch.setattr(validate_hk, 'REPLAY_LOG_FILE', path)
    replay = {'_races': 1, 'labels': {'TRIO': [100, 200, 1, 0]},
              'metrics': {'TRIO': {'tickets': 1, 'profitable': 1}}}
    validate_hk._write_replay_row('2026/10/07', replay)
    validate_hk._write_replay_row('2026/10/07', replay)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]['metrics_json'] == ''
    assert json.loads(rows[1]['metrics_json']) == replay['metrics']
