"""Locks the contract that validate_hk --replay can price WIN and PLACE
pool bets from race_dividends, not from prediction_log.actual_odds
(populated 0 for every row, GPT-5.5 Finding #4 2026-05-25).
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import pytest

import config


def _db():
    if not Path(config.DB_PATH).exists():
        pytest.skip("No horse_history.db found")
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def test_win_div_returns_per_dollar_dividend_for_known_winner():
    from scripts.validate_hk import _win_div
    conn = _db()
    # Pick any (race_date, race_number, combination) from race_dividends WIN.
    row = conn.execute(
        "SELECT race_date, race_number, combination, dividend "
        "FROM race_dividends WHERE pool='WIN' LIMIT 1"
    ).fetchone()
    conn.row_factory = None
    assert row is not None, 'race_dividends should have at least one WIN row'

    div = _win_div(conn, row['race_date'], row['race_number'],
                   int(row['combination']))
    assert div is not None, (
        'WIN dividend lookup must succeed when race_dividends has the row'
    )
    # HKJC dividends are per HK$10; per-$1 should be < declared dividend.
    assert 0 < div < row['dividend']


def test_win_div_returns_none_for_missing_race():
    from scripts.validate_hk import _win_div
    conn = _db()
    assert _win_div(conn, '1900/01/01', 99, 99) is None


def test_place_div_returns_per_dollar_dividend_for_known_placer():
    from scripts.validate_hk import _place_div
    conn = _db()
    row = conn.execute(
        "SELECT race_date, race_number, combination, dividend "
        "FROM race_dividends WHERE pool='PLACE' LIMIT 1"
    ).fetchone()
    assert row is not None

    div = _place_div(conn, row['race_date'], row['race_number'],
                     int(row['combination']))
    assert div is not None
    assert 0 < div < row['dividend']


def test_place_div_returns_none_for_unplaced_horse():
    from scripts.validate_hk import _place_div
    conn = _db()
    row = conn.execute(
        "SELECT race_date, race_number FROM race_dividends "
        "WHERE pool='PLACE' GROUP BY race_date, race_number LIMIT 1"
    ).fetchone()
    placed = {r['combination'] for r in conn.execute(
        "SELECT combination FROM race_dividends "
        "WHERE pool='PLACE' AND race_date=? AND race_number=?",
        (row['race_date'], row['race_number']),
    ).fetchall()}
    unplaced = next(n for n in range(1, 15) if str(n) not in placed)
    assert _place_div(conn, row['race_date'], row['race_number'], unplaced) is None


def test_qpl_banker_replay_scores_winning_ticket():
    """GPT-5.5 R7 Finding #1: QIN PLACE BANKER bets were silently unscored
    by _score_meeting_replay (missing pool branch). Replay must now price
    a QPL ticket whose pivot+banker pair both placed top-3."""
    from scripts.validate_hk import _score_meeting_replay
    import json
    conn = _db()
    # Find a race with a QUINELLA PLACE dividend row AND filled prediction_log.
    row = conn.execute("""
        SELECT d.race_date, d.race_number, d.combination, d.dividend
        FROM race_dividends d
        JOIN prediction_log p ON p.race_date=d.race_date AND p.race_number=d.race_number
        WHERE d.pool='QUINELLA PLACE' AND p.results_filled=1
        LIMIT 1
    """).fetchone()
    if row is None:
        pytest.skip('no race with both QPL dividend and filled prediction_log')
    date, rno, combo, div_declared = (row['race_date'], row['race_number'],
                                       row['combination'], row['dividend'])
    pivot_num, banker_num = sorted(int(x) for x in combo.split(','))
    # Pull horse names for these numbers.
    names = {r['horse_number']: r['horse_name'] for r in conn.execute(
        "SELECT horse_number, horse_name FROM prediction_log "
        "WHERE race_date=? AND race_number=? AND results_filled=1",
        (date, rno),
    ).fetchall()}
    pivot_name = names.get(pivot_num)
    banker_name = names.get(banker_num)
    if not (pivot_name and banker_name):
        pytest.skip('placed horses missing from prediction_log for this race')

    # Inject a synthetic QPL bet for this race into race_bet_ticket.
    fake_ticket = {
        'bets': [{
            'label': 'QIN PLACE BANKER',
            'pool': 'QIN PLACE',
            'combos': 1,
            'legs': [
                {'rank': 1, 'number': pivot_num, 'name': pivot_name},
                {'rank': 2, 'number': banker_num, 'name': banker_name},
            ],
            'unit_stake': 10.0,
        }],
    }
    conn.execute("DELETE FROM race_bet_ticket WHERE race_date=? AND race_number=?",
                 (date, rno))
    conn.execute(
        "INSERT INTO race_bet_ticket (race_date, race_number, ticket_json, logged_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (date, rno, json.dumps(fake_ticket)),
    )
    try:
        result = _score_meeting_replay(conn, date)
        assert result is not None
        scored = result['labels'].get('QIN PLACE BANKER')
        assert scored is not None, 'QPL pool branch did not match the bet'
        stake, ret, hits = scored
        assert hits == 1, f'expected 1 hit, got {hits}'
        assert ret > 0, f'expected positive return on QPL hit, got {ret}'
        # Stake = combos × unit_stake = 1 × 10 = 10
        assert stake == 10.0
    finally:
        conn.execute("DELETE FROM race_bet_ticket WHERE race_date=? AND race_number=?",
                     (date, rno))
        conn.commit()
