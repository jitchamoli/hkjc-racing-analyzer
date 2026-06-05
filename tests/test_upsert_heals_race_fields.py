"""A corrected re-scrape must heal race-level fields that were first ingested
empty/zero.

Bug (money-path review 2026-06-05): venue, distance and going were in the
INSERT but missing from the upsert's ON CONFLICT DO UPDATE SET. So a row first
ingested via the JSON fallback (distance=0, going='') could never be corrected
by a later full scrape — the bad value was permanent. venue was fixed earlier
the same day; this pins distance + going too.
"""
from data.database import HorseDatabase


def _race(distance, going, venue="Happy Valley"):
    return [{
        'date': '2026/06/03', 'race_no': 9, 'venue': venue,
        'distance': distance, 'class_type': 'Class 3', 'going': going, 'course': 'C',
        'entries': [
            {'name': 'HORSE A', 'number': 1, 'actual_position': 1, 'win_odds': 3.0},
            {'name': 'HORSE B', 'number': 2, 'actual_position': 2, 'win_odds': 5.0},
        ],
    }]


def _read(db, field):
    c = db.conn.cursor()
    c.execute(f"SELECT DISTINCT {field} FROM race_results WHERE race_date='2026/06/03'")
    return [r[0] for r in c.fetchall()]


def test_distance_and_going_heal_on_reingest():
    db = HorseDatabase(':memory:')
    try:
        # First ingest: fallback shape — distance missing (0), going empty.
        db.ingest_race_day(_race(0, ''))
        assert _read(db, 'distance') == [0]
        assert _read(db, 'going') == ['']
        # Corrected re-scrape with real values.
        db.ingest_race_day(_race(1650, 'Good To Firm'))
        assert _read(db, 'distance') == [1650], "distance did not heal"
        assert _read(db, 'going') == ['Good To Firm'], "going did not heal"
    finally:
        db.close()


def test_good_values_not_clobbered_by_empty_reingest():
    db = HorseDatabase(':memory:')
    try:
        db.ingest_race_day(_race(1650, 'Good To Firm'))
        # A later partial re-ingest with missing race-level fields must NOT wipe.
        db.ingest_race_day(_race(0, ''))
        assert _read(db, 'distance') == [1650], "good distance clobbered by empty"
        assert _read(db, 'going') == ['Good To Firm'], "good going clobbered by empty"
    finally:
        db.close()
