"""Unit tests for data/uk_database.py.

Uses a cached JSON fixture (2 real races from theracingapi.com captured
2026-05-16) so tests don't hit the live API. Validates schema mapping,
field coercion, idempotency, and summary aggregation.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from data.uk_database import UKHorseDatabase, _to_int, _to_float, _norm_text

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "uk_results_sample.json"


@pytest.fixture
def races():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "uk_test.db"
    instance = UKHorseDatabase(db_path=str(path))
    yield instance
    instance.close()


def test_fixture_has_runners(races):
    assert len(races) >= 2, "fixture must have at least 2 races"
    assert all(r.get('runners') for r in races), "every fixture race must have runners"


def test_ingest_writes_expected_counts(db, races):
    expected_runners = sum(len(r.get('runners', [])) for r in races)
    out = db.ingest_results(races)
    assert out['races'] == len(races)
    assert out['runners'] == expected_runners


def test_ingest_is_idempotent(db, races):
    db.ingest_results(races)
    first = db.get_summary()
    db.ingest_results(races)
    second = db.get_summary()
    assert first == second, "re-ingesting same data must not duplicate rows"


def test_schema_field_mapping(db, races):
    """Verify a known runner roundtrips through the schema with correct types."""
    db.ingest_results(races)
    cur = db.conn.cursor()
    # Sample first runner of first race
    sample_runner = races[0]['runners'][0]
    row = cur.execute(
        "SELECT * FROM uk_race_results WHERE race_id = ? AND horse_id = ?",
        (races[0]['race_id'], sample_runner['horse_id']),
    ).fetchone()
    assert row is not None, "ingested runner row should be retrievable"
    row_d = dict(row)
    # Position should be int when numeric, NULL when 'PU'/'F' etc.
    if str(sample_runner.get('position', '')).isdigit():
        assert row_d['finish_position'] == int(sample_runner['position'])
    # sp_decimal should be numeric (float)
    if sample_runner.get('sp_dec'):
        assert isinstance(row_d['sp_decimal'], float)
    # horse_id preserved verbatim
    assert row_d['horse_id'] == sample_runner['horse_id']


def test_summary_counts(db, races):
    db.ingest_results(races)
    s = db.get_summary()
    expected_runners = sum(len(r.get('runners', [])) for r in races)
    assert s['races'] == len(races)
    assert s['runners'] == expected_runners
    # All fixture runners are unique horses
    unique_horse_ids = {r['horse_id'] for race in races for r in race['runners']}
    assert s['horses'] == len(unique_horse_ids)


def test_coercion_helpers():
    # _to_int
    assert _to_int("12") == 12
    assert _to_int("12.5") == 12
    assert _to_int(None) is None
    assert _to_int("") is None
    assert _to_int("–") is None  # API's "not available" marker
    assert _to_int("not-a-number") is None

    # _to_float
    assert _to_float("9.00") == 9.00
    assert _to_float("–") is None
    assert _to_float(None) is None

    # _norm_text
    assert _norm_text("  hello  ") == "hello"
    assert _norm_text("") is None
    assert _norm_text(None) is None
    # '–' is preserved (it's meaningful as subscription-gate marker)
    assert _norm_text("–") == "–"


def test_favourite_flag_set_from_sp(db, races):
    """sp containing 'F' (favourite marker) should set is_favourite=1."""
    # Inject a synthetic 'F'-marked runner into a copy of the fixture
    import copy
    races_copy = copy.deepcopy(races)
    races_copy[0]['runners'][0]['sp'] = '7/4F'
    db.ingest_results(races_copy)
    cur = db.conn.cursor()
    horse_id = races_copy[0]['runners'][0]['horse_id']
    row = cur.execute(
        "SELECT is_favourite FROM uk_race_results WHERE horse_id = ?",
        (horse_id,),
    ).fetchone()
    assert row[0] == 1


def test_db_path_uses_uk_specific_file(db):
    """Sanity: the DB file path is the UK-specific one, not HK."""
    # The conn was created in the fixture against tmp_path/uk_test.db
    # Just verify the schema is the UK one (uk_races table exists)
    cur = db.conn.cursor()
    tables = {row[0] for row in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert 'uk_races' in tables
    assert 'uk_race_results' in tables
    # Confirm HK tables are NOT present
    assert 'race_results' not in tables  # HK schema must stay separate
