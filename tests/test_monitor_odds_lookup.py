"""Regression: monitor's _look_up_latest_odds must use odds_snapshots, not odds_history."""
import os
import tempfile

from data.odds_tracker import OddsDatabase
from monitor.race_day import _look_up_latest_odds


def test_look_up_latest_odds_finds_recorded_snapshot():
    """After recording a snapshot, _look_up_latest_odds returns the stored win_odds."""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    try:
        odds_db = OddsDatabase(db_path=tmp.name)
        odds_db.record_snapshot('2026/04/27', 5, [
            {'number': 4, 'name': 'Test Horse', 'win_odds': 9.5, 'place_odds': 3.2},
        ])
        result = _look_up_latest_odds(odds_db, '2026/04/27', 5, 4)
        assert result == 9.5
    finally:
        os.unlink(tmp.name)


def test_look_up_latest_odds_returns_zero_when_no_match():
    """Defensive return-0 path when no snapshot exists for the horse."""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    try:
        odds_db = OddsDatabase(db_path=tmp.name)
        result = _look_up_latest_odds(odds_db, '2026/04/27', 99, 99)
        assert result == 0
    finally:
        os.unlink(tmp.name)
