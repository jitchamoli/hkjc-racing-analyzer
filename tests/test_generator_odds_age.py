"""Regression: G5 odds-freshness gate reads real snapshot timestamps."""
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

from data.odds_tracker import OddsDatabase
from dashboard.generator import _compute_odds_age_minutes


def _make_odds_db(ts_str=None):
    """Return an OddsDatabase in a temp file, optionally with one snapshot."""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    db = OddsDatabase(db_path=tmp.name)
    if ts_str is not None:
        # Directly insert a row with a fixed timestamp (bypass datetime.now())
        conn = db._get_conn()
        try:
            conn.execute(
                "INSERT INTO odds_snapshots "
                "(race_date, race_number, horse_number, horse_name, win_odds, place_odds, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('2026/04/27', 5, 1, 'TestHorse', 8.0, 2.5, ts_str),
            )
            conn.commit()
        finally:
            conn.close()
    return db, tmp.name


def test_no_snapshot_returns_none():
    """When no snapshot exists, age is None — G5 must FAIL the gate.

    Previously this returned 0 (treated as 'fresh') which inverted the
    gate's intent: pre-market races with zero odds data could fire EDGE
    on the strength of rule_score alone, with no live-odds anchor.
    """
    db, path = _make_odds_db()
    try:
        age = _compute_odds_age_minutes(db, '2026/04/27', 5)
        assert age is None, f"Expected None for no-snapshot, got {age}"
    finally:
        os.unlink(path)


def test_snapshot_45min_ago_returns_45():
    """Snapshot recorded 45 minutes ago → age ≈ 45."""
    ts_str = (datetime.now() - timedelta(minutes=45)).strftime('%Y-%m-%d %H:%M:%S')
    db, path = _make_odds_db(ts_str=ts_str)
    try:
        age = _compute_odds_age_minutes(db, '2026/04/27', 5)
        # Allow ±2 min for test timing jitter
        assert 43 <= age <= 47, f"Expected ~45, got {age}"
    finally:
        os.unlink(path)
