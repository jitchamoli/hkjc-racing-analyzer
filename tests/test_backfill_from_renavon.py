"""Tests for scripts/backfill_from_renavon.py.

Covers the highest-risk paths:
  1. rating_change copy — race_results row with rating_change=0 gets updated
     to the Renavon-derived value.
  2. No-match case — race_results rows with no matching Renavon row stay untouched.
  3. derive_last_6_and_rating_change — per-horse history is computed chronologically,
     prior-only (no leakage from the current race).
  4. dates_to_apply filter — dates not in our DB are skipped.

The real Renavon CSV is never read; we build the same DataFrame shape in-memory
and call the script's pure functions directly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.database import HorseDatabase
from scripts import backfill_from_renavon as bfr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test_hkjc.db"
    instance = HorseDatabase(db_path=str(path))
    yield instance
    instance.close()


def _seed_result(db, race_date, race_no, horse_name, horse_no=1, rating_change=0, rating=0):
    db.conn.execute(
        """INSERT INTO race_results
              (race_date, race_number, horse_name, horse_number, rating_change, rating)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (race_date, race_no, horse_name, horse_no, rating_change, rating),
    )
    db.conn.commit()


def _renavon_df(rows):
    """Build a Renavon-shaped DataFrame from a list of dicts.

    Each row dict should have: race_date (YYYY/MM/DD), race_number, horse_name,
    horse_number, finishing_position, entry_rating, gear, horse_age.
    """
    cols = [
        'race_date', 'race_number', 'horse_name', 'horse_number',
        'finishing_position', 'entry_rating', 'gear', 'horse_age',
    ]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# 1. rating_change copy
# ---------------------------------------------------------------------------

def test_rating_change_copied_from_renavon(db):
    """Race-results row with rating_change=0 must be updated from Renavon delta."""
    horse = "TEST HORSE"
    _seed_result(db, "2026/03/15", 5, horse, horse_no=4, rating_change=0)
    _seed_result(db, "2026/04/19", 5, horse, horse_no=4, rating_change=0)

    df = _renavon_df([
        {'race_date': '2026/03/15', 'race_number': 5, 'horse_name': horse,
         'horse_number': 4, 'finishing_position': 3, 'entry_rating': 60,
         'gear': 'B', 'horse_age': '4'},
        {'race_date': '2026/04/19', 'race_number': 5, 'horse_name': horse,
         'horse_number': 4, 'finishing_position': 1, 'entry_rating': 65,
         'gear': 'B', 'horse_age': '4'},
    ])

    df = bfr.derive_last_6_and_rating_change(df)
    bfr.backfill_dates(db, df, dates_to_apply={'2026/03/15', '2026/04/19'})

    rows = db.conn.execute(
        "SELECT race_date, rating_change FROM race_results WHERE horse_name = ? ORDER BY race_date",
        (horse,),
    ).fetchall()
    by_date = {r[0]: r[1] for r in rows}
    assert by_date['2026/03/15'] == 0   # first race, no prior rating
    assert by_date['2026/04/19'] == 5   # 65 - 60


# ---------------------------------------------------------------------------
# 2. No-match case
# ---------------------------------------------------------------------------

def test_no_match_row_left_untouched(db):
    """Race-results row with no corresponding Renavon row must keep its values."""
    _seed_result(db, "2026/04/19", 5, "MATCHED", horse_no=4, rating_change=0, rating=60)
    _seed_result(db, "2026/04/19", 5, "UNMATCHED", horse_no=7, rating_change=0, rating=70)

    df = _renavon_df([
        {'race_date': '2026/04/19', 'race_number': 5, 'horse_name': 'MATCHED',
         'horse_number': 4, 'finishing_position': 2, 'entry_rating': 65,
         'gear': 'B', 'horse_age': '4'},
        # Provide a prior race for MATCHED so rating_change can be non-zero
        {'race_date': '2026/03/15', 'race_number': 5, 'horse_name': 'MATCHED',
         'horse_number': 4, 'finishing_position': 3, 'entry_rating': 60,
         'gear': 'B', 'horse_age': '4'},
    ])
    df = bfr.derive_last_6_and_rating_change(df)
    bfr.backfill_dates(db, df, dates_to_apply={'2026/04/19'})

    # MATCHED gets a rating where missing (rating WAS 60 originally so stays 60 per backfill guard)
    matched = db.conn.execute(
        "SELECT rating_change FROM race_results WHERE horse_name = 'MATCHED' AND race_date = '2026/04/19'"
    ).fetchone()[0]
    assert matched == 5  # 65 - 60

    # UNMATCHED row has no Renavon entry — must stay at its seeded values
    unmatched = db.conn.execute(
        "SELECT rating_change, rating FROM race_results WHERE horse_name = 'UNMATCHED'"
    ).fetchone()
    assert unmatched[0] == 0
    assert unmatched[1] == 70


# ---------------------------------------------------------------------------
# 3. derive_last_6_and_rating_change correctness (no leakage)
# ---------------------------------------------------------------------------

def test_derive_uses_only_prior_races_no_leakage():
    """last_6 and rating_change MUST be computed from races strictly before the
    current one — never include the current race itself."""
    horse = "LEAKY"
    df = _renavon_df([
        {'race_date': '2026/01/01', 'race_number': 1, 'horse_name': horse,
         'horse_number': 1, 'finishing_position': 4, 'entry_rating': 50,
         'gear': '', 'horse_age': '3'},
        {'race_date': '2026/02/01', 'race_number': 2, 'horse_name': horse,
         'horse_number': 1, 'finishing_position': 2, 'entry_rating': 55,
         'gear': '', 'horse_age': '3'},
        {'race_date': '2026/03/01', 'race_number': 3, 'horse_name': horse,
         'horse_number': 1, 'finishing_position': 1, 'entry_rating': 60,
         'gear': '', 'horse_age': '3'},
    ])
    out = bfr.derive_last_6_and_rating_change(df)
    out = out.sort_values('race_date').reset_index(drop=True)

    # Row 0 (first race ever): no prior → empty last_6, zero rating_change
    assert out.iloc[0]['last_6'] == ''
    assert out.iloc[0]['rating_change'] == 0

    # Row 1: prior is race 0 (finished 4th, rating 50)
    assert out.iloc[1]['last_6'] == '4'
    assert out.iloc[1]['rating_change'] == 5  # 55 - 50

    # Row 2: priors are races 0 & 1; most-recent-first → "2 4"
    assert out.iloc[2]['last_6'] == '2 4'
    assert out.iloc[2]['rating_change'] == 5  # 60 - 55


def test_derive_resets_history_between_horses():
    """When the iterator moves from one horse to the next, history must reset
    — last_6 for horse B's first race must not include horse A's history."""
    df = _renavon_df([
        {'race_date': '2026/01/01', 'race_number': 1, 'horse_name': 'HORSE A',
         'horse_number': 1, 'finishing_position': 1, 'entry_rating': 80,
         'gear': '', 'horse_age': '4'},
        {'race_date': '2026/02/01', 'race_number': 2, 'horse_name': 'HORSE A',
         'horse_number': 1, 'finishing_position': 3, 'entry_rating': 78,
         'gear': '', 'horse_age': '4'},
        {'race_date': '2026/01/15', 'race_number': 3, 'horse_name': 'HORSE B',
         'horse_number': 2, 'finishing_position': 5, 'entry_rating': 50,
         'gear': '', 'horse_age': '3'},
    ])
    out = bfr.derive_last_6_and_rating_change(df)
    horse_b_row = out[out['horse_name'] == 'HORSE B'].iloc[0]
    assert horse_b_row['last_6'] == ''
    assert horse_b_row['rating_change'] == 0


# ---------------------------------------------------------------------------
# 4. dates_to_apply filter
# ---------------------------------------------------------------------------

def test_dates_to_apply_filters_out_unwanted_dates(db):
    """A Renavon date not in dates_to_apply must NOT touch the DB."""
    # Only seed one date in our DB
    _seed_result(db, "2026/04/19", 5, "TARGET", horse_no=4, rating_change=0)

    df = _renavon_df([
        # In scope
        {'race_date': '2026/03/15', 'race_number': 5, 'horse_name': 'TARGET',
         'horse_number': 4, 'finishing_position': 3, 'entry_rating': 60,
         'gear': 'B', 'horse_age': '4'},
        {'race_date': '2026/04/19', 'race_number': 5, 'horse_name': 'TARGET',
         'horse_number': 4, 'finishing_position': 1, 'entry_rating': 65,
         'gear': 'B', 'horse_age': '4'},
        # Out of scope (this date isn't in our DB at all)
        {'race_date': '2025/12/01', 'race_number': 1, 'horse_name': 'OTHER',
         'horse_number': 1, 'finishing_position': 1, 'entry_rating': 50,
         'gear': '', 'horse_age': '3'},
    ])
    df = bfr.derive_last_6_and_rating_change(df)

    # Only allow the one date we actually have
    n = bfr.backfill_dates(db, df, dates_to_apply={'2026/04/19'})

    # Updated only the row in the allowed date
    assert n == 1
    rating_change = db.conn.execute(
        "SELECT rating_change FROM race_results WHERE horse_name = 'TARGET'"
    ).fetchone()[0]
    assert rating_change == 5  # 65 - 60

    # OTHER was never inserted (filtered out)
    other = db.conn.execute(
        "SELECT COUNT(*) FROM race_results WHERE horse_name = 'OTHER'"
    ).fetchone()[0]
    assert other == 0
