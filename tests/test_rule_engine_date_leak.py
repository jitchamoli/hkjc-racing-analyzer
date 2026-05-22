"""Regression guard for the rule-engine date-leak class (2026-05-22).

The rule engine queries the horse's history during backtest, but race_results
CONTAINS the race being scored — so without a `before_date` filter these
lookups returned the current race's own result (finish_position/is_winner/
margin), a direct outcome leak. This bug was found and fixed TWICE (the second
time it had 6 siblings). These tests assert every leak-prone DB method respects
`before_date` so it can never silently return again.
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.database import HorseDatabase


@pytest.fixture
def db(tmp_path):
    """A horse 'TESTER' with three runs: 2026/01/01, 02/01, 03/01 (the last is
    the 'current' race being scored). before_date='2026/03/01' must exclude it."""
    d = HorseDatabase(db_path=str(tmp_path / "leak.db"))
    runs = [
        ("2026/01/01", 1, 5, 0, 0, 3.0),   # date, race_no, finish, is_winner, is_top3, margin
        ("2026/02/01", 1, 1, 1, 1, 0.0),
        ("2026/03/01", 1, 8, 0, 0, 6.0),   # the CURRENT race — must be excluded
    ]
    for date, rno, fin, win, top3, margin in runs:
        d.conn.execute(
            "INSERT INTO race_results (race_date, race_number, horse_name, jockey, "
            "trainer, finish_position, is_winner, is_top3, margin, win_odds, going) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (date, rno, "TESTER", "J WIN", "T WIN", fin, win, top3, margin, 4.0, "Good"),
        )
    d.conn.commit()
    return d

CUR = "2026/03/01"  # the race being scored


def test_get_horse_history_excludes_current_and_later(db):
    leaked = db.get_horse_history("TESTER", limit=10)              # no filter (live)
    clean = db.get_horse_history("TESTER", limit=10, before_date=CUR)
    assert len(leaked) == 3 and len(clean) == 2
    assert all(r["race_date"] < CUR for r in clean)


def test_get_rolling_stats_excludes_current_race_result(db):
    # Leaking: 3 runs incl. the 8th-place current race. Clean: 2 prior runs.
    leaked = db.get_rolling_stats("TESTER", n_races=5)
    clean = db.get_rolling_stats("TESTER", n_races=5, before_date=CUR)
    assert leaked["runs"] == 3 and clean["runs"] == 2
    # the current race (8th) must not be in the clean average
    assert clean["avg_pos"] == pytest.approx((5 + 1) / 2)


def test_get_beaten_margins_excludes_current(db):
    clean = db.get_beaten_margins("TESTER", n_races=5, before_date=CUR)
    assert clean["runs"] == 2  # current race's margin=6.0 excluded


def test_get_horse_record_excludes_current(db):
    clean = db.get_horse_record("TESTER", before_date=CUR)
    assert clean["total_starts"] == 2  # not 3


def test_get_last_jockey_is_prior_not_current(db):
    # all rides are 'J WIN' here; the point is the count/window excludes current
    clean_hist = db.get_horse_history("TESTER", limit=1, before_date=CUR)
    assert clean_hist[0]["race_date"] == "2026/02/01"  # not 03/01


def test_recent_form_methods_accept_before_date(db):
    # smoke: the jockey/trainer recent-form + going-record methods must accept
    # before_date and filter (regression against the 6-sibling miss)
    jf = db.get_jockey_recent_form("J WIN", n_rides=10, before_date=CUR)
    tf = db.get_trainer_recent_form("T WIN", n_runners=10, before_date=CUR)
    gr = db.get_horse_going_record("TESTER", "Good", before_date=CUR)
    assert jf["rides"] == 2 and tf["runners"] == 2 and gr["runs"] == 2
