"""ingest_race_day must warn when a race has a dropped finisher.

This is the systemic backstop for the dead-heat parser drop (2026-06-05): a
complete result set has contiguous placings, so max(finishing position) ==
number of finishers. If a finisher is silently lost upstream (e.g. a new
unhandled placing marker), max > count — and ingest_race_day logs a loud
WARNING so we catch it at ingest instead of months later in an audit.

A correctly-KEPT dead heat must NOT trip the guard (two horses share a
placing and the next number is skipped: 1,2,2,4 → max==count==4).
"""
import logging
from data.database import HorseDatabase


def _race(positions):
    """positions: list of finishing positions; one entry each."""
    return [{
        'date': '2026/06/03', 'race_no': 9, 'venue': 'Happy Valley',
        'distance': 1650, 'class_type': 'Class 3', 'going': 'Good', 'course': 'C',
        'entries': [
            {'name': f'HORSE{i}', 'number': i, 'actual_position': p,
             'win_odds': 5.0}
            for i, p in enumerate(positions, start=1)
        ],
    }]


def test_warns_on_dropped_finisher(caplog):
    # 10 rows but top placing is 12 → 2 dropped (the real R9 dead-heat case).
    races = _race([1, 2, 3, 4, 5, 6, 7, 8, 11, 12])
    db = HorseDatabase(':memory:')
    try:
        with caplog.at_level(logging.WARNING):
            db.ingest_race_day(races)
    finally:
        db.close()
    assert any('DROPPED FINISHER' in r.message for r in caplog.records), \
        "guard did not warn on a dropped finisher"


def test_no_warning_on_kept_dead_heat(caplog):
    # Dead heat for 2nd kept correctly: 1,2,2,4 → max==count==4, no drop.
    races = _race([1, 2, 2, 4])
    db = HorseDatabase(':memory:')
    try:
        with caplog.at_level(logging.WARNING):
            db.ingest_race_day(races)
    finally:
        db.close()
    assert not any('DROPPED FINISHER' in r.message for r in caplog.records), \
        "guard false-positived on a correctly-kept dead heat"


def test_no_warning_on_complete_field(caplog):
    races = _race([1, 2, 3, 4, 5, 6])
    db = HorseDatabase(':memory:')
    try:
        with caplog.at_level(logging.WARNING):
            db.ingest_race_day(races)
    finally:
        db.close()
    assert not any('DROPPED FINISHER' in r.message for r in caplog.records)
