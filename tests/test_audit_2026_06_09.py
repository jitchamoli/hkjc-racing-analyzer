"""Regression tests for the 2026-06-09 production-quality audit fixes.

Covers the behavioral changes:
  #2  validate_hk._score_meeting: per-strategy box guard (no longer drops the
      whole race when <6 horses are ranked) + number-keyed hit detection.
  #3  backtest.get_real_exotic_dividend: matches the actual finishing combo so
      dead-heat races (>1 winning row) resolve to the right dividend.
"""
import sqlite3

import pytest

import scripts.validate_hk as vh
from scripts.backtest import get_real_exotic_dividend
from betting.dividends import hkjc_declared_to_unit
from ml.features import FeatureEngineering


def _conn_with_predlog(rows):
    """rows: list of (race_number, horse_number, horse_name, predicted_rank,
    actual_position)."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute(
        "CREATE TABLE prediction_log (race_date TEXT, race_number INTEGER, "
        "horse_number INTEGER, horse_name TEXT, predicted_rank INTEGER, "
        "actual_position INTEGER, results_filled INTEGER)"
    )
    for rno, hnum, hname, prank, apos in rows:
        c.execute(
            "INSERT INTO prediction_log VALUES ('2026/06/03',?,?,?,?,?,1)",
            (rno, hnum, hname, prank, apos),
        )
    c.execute(
        "CREATE TABLE race_dividends (race_date TEXT, race_number INTEGER, "
        "pool TEXT, combination TEXT, dividend REAL)"
    )
    c.commit()
    return c


def test_score_meeting_keeps_five_horse_race():
    # 5 ranked horses (numbers 11..15), finishers pos1=#15 pos2=#14 (a miss for
    # every strategy, so no dividend needed — we only assert stake is counted).
    rows = [
        (1, 11, "ALPHA", 1, None),
        (1, 12, "BRAVO", 2, None),
        (1, 13, "CHARLIE", 3, None),
        (1, 14, "DELTA", 4, 2),
        (1, 15, "ECHO", 5, 1),
    ]
    acc = vh._score_meeting(_conn_with_predlog(rows), "2026/06/03")
    # Old code did `if len(ranked) < 6: continue` and dropped the whole race.
    assert acc["_races"] == 1
    # Strategies whose box fits in 5 horses are scored...
    assert acc["straight_QIN"][0] == 1     # box 2
    assert acc["BOX_Q4"][0] == 6           # box 4
    assert acc["TRIO_5"][0] == 10          # box 5
    assert acc["F4_BOX_5"][0] == 5         # box 5
    # ...TRIO-6 (box 6) cannot be filled by 5 horses -> skipped, not mis-scored.
    assert acc["TRIO_6"][0] == 0


def test_score_meeting_hits_by_number():
    # straight_QIN hits when our top-2 numbers == finisher top-2 numbers.
    rows = [
        (1, 11, "ALPHA", 1, 1),
        (1, 12, "BRAVO", 2, 2),
        (1, 13, "CHARLIE", 3, 3),
        (1, 14, "DELTA", 4, None),
        (1, 15, "ECHO", 5, None),
        (1, 16, "FOX", 6, None),
    ]
    conn = _conn_with_predlog(rows)
    # QUINELLA dividend for the winning pair "11,12" (declared per HK$10).
    conn.execute(
        "INSERT INTO race_dividends VALUES ('2026/06/03',1,'QUINELLA','11,12',80.0)"
    )
    conn.commit()
    acc = vh._score_meeting(conn, "2026/06/03")
    assert acc["straight_QIN"][2] == 1                       # 1 hit
    assert acc["straight_QIN"][1] == hkjc_declared_to_unit(80.0)


def test_extract_features_refuses_empty_race_date_with_conn():
    # audit 2026-06-09 #5: with a DB conn but no race_date, the before_date filter
    # would coerce to None and leak the current race's own results. Fail closed.
    conn = sqlite3.connect(":memory:")
    horse_data = {"name": "ALPHA", "distance": "1200", "going": "GOOD",
                  "field_size": 12}  # no race_date
    with pytest.raises(ValueError, match="race_date is empty"):
        FeatureEngineering.extract_features(horse_data, [], "Sha Tin", conn=conn)


def test_exotic_dividend_matches_combination():
    c = sqlite3.connect(":memory:")
    c.execute(
        "CREATE TABLE race_dividends (race_date TEXT, race_number INTEGER, "
        "pool TEXT, combination TEXT, dividend REAL)"
    )
    # Dead-heat: two winning TRIO rows for the same race/pool.
    c.execute("INSERT INTO race_dividends VALUES ('2026/06/03',1,'TRIO','1,2,3',150.0)")
    c.execute("INSERT INTO race_dividends VALUES ('2026/06/03',1,'TRIO','1,2,4',900.0)")
    c.commit()
    # Combination-matched lookup returns the right row, not an arbitrary one.
    assert get_real_exotic_dividend(c, "2026/06/03", 1, "TRIO", "1,2,3") == \
        hkjc_declared_to_unit(150.0)
    assert get_real_exotic_dividend(c, "2026/06/03", 1, "TRIO", "1,2,4") == \
        hkjc_declared_to_unit(900.0)
    # No-combination fallback still returns *a* row (legacy behavior preserved).
    assert get_real_exotic_dividend(c, "2026/06/03", 1, "TRIO") is not None
    # Unknown combination -> None.
    assert get_real_exotic_dividend(c, "2026/06/03", 1, "TRIO", "7,8,9") is None
