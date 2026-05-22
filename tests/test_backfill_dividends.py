"""Tests for scripts/backfill_dividends.py.

Covers the three highest-risk paths:
  1. Idempotency — running twice yields the same row count (UPSERT, no dupes).
  2. Skip-if-present — dates already covered are not re-scraped (no --force).
  3. End-of-card handling — scraper returns only the races before card ended;
     backfill processes exactly what it got and inserts dividends for those.

All HTTP/scraper calls are mocked via unittest.mock so tests never hit HKJC.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from data.database import HorseDatabase
from scripts import backfill_dividends as bd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_race_result(db: HorseDatabase, race_date: str, race_number: int, horse_name: str = "TEST HORSE"):
    """Insert a minimal race_results row so find_dates_to_backfill sees the date."""
    db.conn.execute(
        """INSERT INTO race_results (race_date, race_number, horse_name)
           VALUES (?, ?, ?)""",
        (race_date, race_number, horse_name),
    )
    db.conn.commit()


def _make_race(race_no: int, dividends: list[dict]) -> dict:
    """Build the dict shape scrape_meeting_results returns."""
    return {'race_no': race_no, 'dividends': dividends}


def _sample_dividends() -> list[dict]:
    return [
        {'pool': 'WIN', 'combination': '5', 'dividend': 12.5},
        {'pool': 'PLACE', 'combination': '5', 'dividend': 3.4},
        {'pool': 'PLACE', 'combination': '2', 'dividend': 5.1},
        {'pool': 'QUINELLA', 'combination': '2-5', 'dividend': 45.0},
    ]


@pytest.fixture
def db(tmp_path):
    """Fresh HorseDatabase pointing at a temp file."""
    path = tmp_path / "test_hkjc.db"
    instance = HorseDatabase(db_path=str(path))
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# 1. Idempotency
# ---------------------------------------------------------------------------

def test_backfill_is_idempotent(db):
    """Running backfill_date twice for the same date must not duplicate rows."""
    date = "2026/04/19"
    _seed_race_result(db, date, 1)

    mock_races = [_make_race(1, _sample_dividends())]

    with patch.object(bd, 'scrape_meeting_results', new=AsyncMock(return_value=mock_races)):
        n1 = asyncio.run(bd.backfill_date(date, db))
        n2 = asyncio.run(bd.backfill_date(date, db))

    assert n1 == len(mock_races[0]['dividends']) == 4
    assert n2 == 4  # tries to write same 4 rows again

    # But the table itself must still contain exactly 4 rows (UPSERT, no dupes)
    count = db.conn.execute(
        "SELECT COUNT(*) FROM race_dividends WHERE race_date = ?", (date,)
    ).fetchone()[0]
    assert count == 4, "UPSERT must prevent duplicate rows on re-run"


# ---------------------------------------------------------------------------
# 2. Skip-if-present logic
# ---------------------------------------------------------------------------

def test_find_dates_skips_already_covered(db):
    """Dates that already have any race_dividends rows must NOT be returned."""
    covered_date = "2026/04/19"
    missing_date = "2026/04/20"
    _seed_race_result(db, covered_date, 1)
    _seed_race_result(db, missing_date, 1)
    # Cover the first date with one dividend row
    db.conn.execute(
        """INSERT INTO race_dividends (race_date, race_number, pool, combination, dividend)
           VALUES (?, 1, 'WIN', '5', 12.5)""",
        (covered_date,),
    )
    db.conn.commit()

    dates = asyncio.run(bd.find_dates_to_backfill(db.conn, last_n=None, force=False))

    assert covered_date not in dates
    assert missing_date in dates


def test_find_dates_force_returns_all_dates(db):
    """--force must return every date, even those already covered."""
    covered_date = "2026/04/19"
    _seed_race_result(db, covered_date, 1)
    db.conn.execute(
        """INSERT INTO race_dividends (race_date, race_number, pool, combination, dividend)
           VALUES (?, 1, 'WIN', '5', 12.5)""",
        (covered_date,),
    )
    db.conn.commit()

    dates = asyncio.run(bd.find_dates_to_backfill(db.conn, last_n=None, force=True))
    assert covered_date in dates


# ---------------------------------------------------------------------------
# 3. End-of-card behaviour
# ---------------------------------------------------------------------------

def test_backfill_processes_only_races_returned_by_scraper(db):
    """The scraper handles end-of-card detection (stops after 2 empty races
    and returns only the races it parsed). backfill_date must:
      - process exactly the races the scraper returned (no crash on truncated list)
      - insert dividends only for those races
    """
    date = "2026/04/19"
    _seed_race_result(db, date, 1)

    # Simulate: scraper returned only race 1 because races 2 & 3 came back empty
    # (end of card detected inside the scraper itself).
    mock_races = [_make_race(1, _sample_dividends())]

    with patch.object(bd, 'scrape_meeting_results', new=AsyncMock(return_value=mock_races)) as mock_scraper:
        n = asyncio.run(bd.backfill_date(date, db))

    assert mock_scraper.called
    assert n == 4  # all 4 dividends from race 1

    # Confirm no rows leaked for races 2 or 3
    rows = db.conn.execute(
        "SELECT race_number FROM race_dividends WHERE race_date = ?", (date,)
    ).fetchall()
    assert {r[0] for r in rows} == {1}


def test_backfill_returns_zero_when_scraper_returns_nothing(db):
    """Empty scraper response (e.g. no meeting that day) must return 0 cleanly."""
    date = "2026/04/19"
    _seed_race_result(db, date, 1)

    with patch.object(bd, 'scrape_meeting_results', new=AsyncMock(return_value=[])):
        n = asyncio.run(bd.backfill_date(date, db))

    assert n == 0
    count = db.conn.execute("SELECT COUNT(*) FROM race_dividends").fetchone()[0]
    assert count == 0


def test_backfill_skips_invalid_dividend_entries(db):
    """Rows with empty pool/combination or non-positive dividend must be dropped."""
    date = "2026/04/19"
    _seed_race_result(db, date, 1)

    mixed = [
        {'pool': 'WIN', 'combination': '5', 'dividend': 12.5},     # keep
        {'pool': '', 'combination': '5', 'dividend': 12.5},        # drop: no pool
        {'pool': 'WIN', 'combination': '', 'dividend': 12.5},      # drop: no combo
        {'pool': 'WIN', 'combination': '6', 'dividend': 0},        # drop: zero div
        {'pool': 'WIN', 'combination': '7', 'dividend': -1.0},     # drop: negative
    ]
    mock_races = [_make_race(1, mixed)]
    with patch.object(bd, 'scrape_meeting_results', new=AsyncMock(return_value=mock_races)):
        n = asyncio.run(bd.backfill_date(date, db))

    assert n == 1
    count = db.conn.execute("SELECT COUNT(*) FROM race_dividends").fetchone()[0]
    assert count == 1


def test_parse_dividends_captures_multi_combo_continuation_rows():
    """Multi-combo pools (PLACE, QUINELLA PLACE) emit continuation rows with
    only 2 cells — HKJC omits the pool cell. They inherit the current pool.

    Regression (2026-05-22): the prior `len(row) < 3: continue` dropped every
    2-cell row, so only the FIRST combo of PLACE/QUINELLA PLACE was stored
    (e.g. 2026/05/20 R2 kept QPL 2-10 but lost 8-10 and 2-8 @ $405). Layout
    below is the real 0520 R2 dividend table.
    """
    from data.scraper import _parse_dividends
    table = [
        ['Dividend'],
        ['Pool', 'Winning Combination', 'Dividend (HK$)'],
        ['WIN', '10', '19.50'],
        ['PLACE', '10', '12.00'],
        ['2', '20.50'],
        ['8', '102.50'],
        ['QUINELLA', '2,10', '73.50'],
        ['QUINELLA PLACE', '2,10', '27.50'],
        ['8,10', '200.00'],
        ['2,8', '405.00'],
        ['FORECAST', '10,2', '107.00'],
        ['TIERCE', '10,2,8', '2,720.00'],
        ['TRIO', '2,8,10', '816.00'],
    ]
    divs = _parse_dividends([table])
    by_pool = {}
    for d in divs:
        by_pool.setdefault(d['pool'], {})[d['combination']] = d['dividend']

    # All 3 PLACE combos
    assert by_pool['PLACE'] == {'10': 12.0, '2': 20.5, '8': 102.5}
    # All 3 QUINELLA PLACE pairs — including the friend's 2-8 @ $405
    assert by_pool['QUINELLA PLACE'] == {'2,10': 27.5, '8,10': 200.0, '2,8': 405.0}
    # Single-combo pools still correct
    assert by_pool['QUINELLA'] == {'2,10': 73.5}
    assert by_pool['TRIO'] == {'2,8,10': 816.0}
