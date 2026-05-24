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
    # Pick a race with PLACE dividends, then query a horse number that
    # isn't in the PLACE rows (i.e., didn't place).
    row = conn.execute(
        "SELECT race_date, race_number FROM race_dividends "
        "WHERE pool='PLACE' GROUP BY race_date, race_number LIMIT 1"
    ).fetchone()
    placed = {r['combination'] for r in conn.execute(
        "SELECT combination FROM race_dividends "
        "WHERE pool='PLACE' AND race_date=? AND race_number=?",
        (row['race_date'], row['race_number']),
    ).fetchall()}
    # Find a 1-14 number that didn't place.
    unplaced = next(n for n in range(1, 15) if str(n) not in placed)
    assert _place_div(conn, row['race_date'], row['race_number'], unplaced) is None
