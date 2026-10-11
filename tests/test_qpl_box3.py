"""QPL boxes score every held pair, not just pairs involving the first leg."""
import json
import sqlite3

import pytest

from betting.bankroll_guard import _ticket_total_stake, daily_status
from scripts.stake_ab import _score_bet_from_ticket
from scripts.validate_hk import _score_meeting_replay


@pytest.mark.parametrize('positions,expected', [([4, 1, 2], 60), ([1, 2, 3], 120), ([1, 4, 5], 0)])
def test_qpl_box_replay_and_stake_analysis_agree(positions, expected):
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE race_bet_ticket (race_date TEXT, race_number INTEGER, ticket_json TEXT);
        CREATE TABLE prediction_log (race_date TEXT, race_number INTEGER,
            horse_number INTEGER, horse_name TEXT, actual_position INTEGER, results_filled INTEGER);
        CREATE TABLE race_dividends (race_date TEXT, race_number INTEGER,
            pool TEXT, combination TEXT, dividend REAL);
    ''')
    horses = [{'number': i, 'name': f'H{i}', 'rank': i, 'actual_position': p}
              for i, p in enumerate(positions, 1)]
    # Include the rest of the finishers so replay does not skip the race.
    horses += [{'number': 9 + p, 'name': f'Other{p}', 'actual_position': p}
               for p in (1, 2, 3) if p not in positions]
    bet = {'label': 'QPL BOX-3', 'pool': 'QIN PLACE', 'combos': 3,
           'unit_stake': 10, 'legs': [{'number': i, 'name': f'H{i}'} for i in (1, 2, 3)]}
    conn.execute('INSERT INTO race_bet_ticket VALUES (?,?,?)',
                 ('2026/10/11', 1, json.dumps({'bets': [bet]})))
    for h in horses:
        conn.execute('INSERT INTO prediction_log VALUES (?,?,?,?,?,1)',
                     ('2026/10/11', 1, h['number'], h['name'], h['actual_position']))
    for pair, dividend in [('1,2', 20), ('1,3', 40), ('2,3', 60)]:
        conn.execute('INSERT INTO race_dividends VALUES (?,?,?,?,?)',
                     ('2026/10/11', 1, 'QUINELLA PLACE', pair, dividend))
    replay = _score_meeting_replay(conn, '2026/10/11')
    assert replay['labels']['QPL BOX-3'][:2] == [30, expected]
    hit, dividend = _score_bet_from_ticket(bet, {'race_number': 1, 'horses': horses}, conn, '2026/10/11')
    assert hit == (expected > 0)
    assert dividend * 10 == expected
    assert _ticket_total_stake(json.dumps({'bets': [bet]}), counted_only=True) == (30, 1)
    status = daily_status('2026/10/11', bankroll=100, conn=conn)
    assert status.staked == 30
    assert status.realized_pnl == expected - 30
    if expected == 0:
        assert 'LOSS CAP' in status.halt_reason
    conn.close()
