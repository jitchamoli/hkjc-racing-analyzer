"""Path B Phase 1: sectional_times column + ingestion persistence.

Verifies the migration adds the column and that HorseDatabase.ingest_race_day
round-trips sectional_times (including the upsert preserve/update behaviour).
Uses a temp DB — no network, deterministic.
"""
import pytest

from data.database import HorseDatabase


@pytest.fixture
def db(tmp_path):
    d = HorseDatabase(db_path=str(tmp_path / "test.db"))
    yield d
    d.close()


def _race(sectional_times="27.84 23.69 24.22 24.10"):
    """One-race ingest payload with a single horse carrying sectional_times."""
    return [{
        "date": "2026/05/13",
        "race_no": 1,
        "venue": "ST",
        "distance": 1650,
        "going": "GOOD",
        "class_type": "4",
        "course": "A",
        "entries": [{
            "number": 1,
            "name": "TEST HORSE",
            "jockey": "J Test",
            "trainer": "T Test",
            "draw": 3,
            "weight": "126",
            "horse_weight": 1100,
            "win_odds": 4.5,
            "actual_position": 1,
            "margin": 0,
            "running_position": "5 4 3 1",
            "finish_time": "1:39.85",
            "sectional_times": sectional_times,
        }],
        "dividends": [],
    }]


class TestSectionalColumn:
    def test_column_exists_after_migration(self, db):
        cols = [r[1] for r in db.conn.execute(
            "PRAGMA table_info(race_results)").fetchall()]
        assert "sectional_times" in cols


class TestSectionalPersistence:
    def test_ingest_writes_sectional_times(self, db):
        db.ingest_race_day(_race("27.84 23.69 24.22 24.10"))
        row = db.conn.execute(
            "SELECT sectional_times FROM race_results "
            "WHERE race_date='2026/05/13' AND race_number=1 AND horse_name='TEST HORSE'"
        ).fetchone()
        assert row is not None
        assert row[0] == "27.84 23.69 24.22 24.10"

    def test_missing_sectional_times_defaults_empty(self, db):
        race = _race()
        del race[0]["entries"][0]["sectional_times"]
        db.ingest_race_day(race)
        row = db.conn.execute(
            "SELECT sectional_times FROM race_results WHERE horse_name='TEST HORSE'"
        ).fetchone()
        assert row[0] == ""

    def test_reingest_preserves_existing_when_new_is_empty(self, db):
        # First ingest with sectionals, then re-ingest without — upsert must
        # keep the existing value (same CASE-WHEN pattern as running_position).
        db.ingest_race_day(_race("27.84 23.69 24.22 24.10"))
        race = _race("")
        db.ingest_race_day(race)
        row = db.conn.execute(
            "SELECT sectional_times FROM race_results WHERE horse_name='TEST HORSE'"
        ).fetchone()
        assert row[0] == "27.84 23.69 24.22 24.10"

    def test_reingest_updates_when_new_value_present(self, db):
        # First ingest empty, then re-ingest with sectionals — upsert fills it.
        db.ingest_race_day(_race(""))
        db.ingest_race_day(_race("28.00 24.00 24.50 24.20"))
        row = db.conn.execute(
            "SELECT sectional_times FROM race_results WHERE horse_name='TEST HORSE'"
        ).fetchone()
        assert row[0] == "28.00 24.00 24.50 24.20"
