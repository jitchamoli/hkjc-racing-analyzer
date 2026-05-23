"""scripts/validate_hk — _exotic_div must filter by combination.

PR #52 strict review MEDIUM #3: without a combination filter, fetchone()
returns whichever row SQLite picks by rowid. Current data has one row per
(race, pool) so the bug is dormant, but the May-22 dividend-basis bug
(10× ROI inflation) was the same family of mistake — silent fetch with
no filter. Lock down the schema contract.
"""
from __future__ import annotations

import sqlite3

import pytest

from scripts.validate_hk import _exotic_div


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("""
        CREATE TABLE race_dividends (
            race_date TEXT, race_number INTEGER, pool TEXT,
            combination TEXT, dividend REAL,
            PRIMARY KEY (race_date, race_number, pool, combination)
        )
    """)
    yield c
    c.close()


def test_exotic_div_returns_dividend_for_matching_combination(conn):
    conn.execute("INSERT INTO race_dividends VALUES "
                 "('2026/05/20', 1, 'TRIO', '1,3,7', 390.0)")
    # HKJC dividends are per HK$10 → script divides by 10 for $1 basis.
    assert _exotic_div(conn, '2026/05/20', 1, 'TRIO', '1,3,7') == 39.0


def test_exotic_div_returns_none_when_no_row(conn):
    assert _exotic_div(conn, '2026/05/20', 1, 'TRIO', '1,3,7') is None


def test_exotic_div_only_returns_the_requested_combination(conn):
    """If a race ever has supplementary/consolation rows (it doesn't today,
    but the May-22 dividend bug is the cautionary tale), the helper must
    pick the row matching the actual winning combination, not whichever
    SQLite hands back first."""
    conn.execute("INSERT INTO race_dividends VALUES "
                 "('2026/05/20', 1, 'TRIO', '1,3,7', 390.0)")
    conn.execute("INSERT INTO race_dividends VALUES "
                 "('2026/05/20', 1, 'TRIO', '2,5,9', 9999.0)")
    assert _exotic_div(conn, '2026/05/20', 1, 'TRIO', '1,3,7') == 39.0
    assert _exotic_div(conn, '2026/05/20', 1, 'TRIO', '2,5,9') == 999.9


def test_exotic_div_for_first_4_pool(conn):
    conn.execute("INSERT INTO race_dividends VALUES "
                 "('2026/05/20', 1, 'FIRST 4', '1,3,7,8', 3190.0)")
    assert _exotic_div(conn, '2026/05/20', 1, 'FIRST 4', '1,3,7,8') == 319.0


def test_exotic_div_rejects_when_combination_mismatches(conn):
    conn.execute("INSERT INTO race_dividends VALUES "
                 "('2026/05/20', 1, 'TRIO', '1,3,7', 390.0)")
    # Caller passed wrong combination (e.g. swapped horse numbers).
    # Better None than silent wrong dividend.
    assert _exotic_div(conn, '2026/05/20', 1, 'TRIO', '1,3,8') is None
