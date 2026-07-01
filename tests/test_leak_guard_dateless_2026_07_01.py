"""Regression tests for the date-less leak guard (2026-07-01 expert review).

Pins the invariant: extract_features / the _compute_* helpers must NEVER issue a
race_results query without an `AND race_date < ?` filter. With no race_date the
feature vector must be the leak-free DEFAULT vector — never the horse's own
(current-race-inclusive) DB history.

Design from the Modeling & Backtesting Auditor: the unfiltered `else` branches
were deleted structurally and each helper early-returns its default when
before_date is falsy. These tests prove (1) a horse WITH a winning DB row scores
identically to an unknown horse when no date is supplied (leak-impossible),
(2) supplying a date DOES change the vector (we disabled leakage, not the DB),
and (3) the helper refuses an unfiltered read even when handed a live conn.
"""
import sqlite3

import numpy as np
import pytest

import config
from ml.features import FeatureEngineering, clear_feature_caches

# (column, sqlite type) — numeric columns typed like the real race_results so
# the finish_position <= 5 comparison behaves as in production.
_COLS = [
    ("horse_name", "TEXT"), ("race_date", "TEXT"), ("race_number", "INTEGER"),
    ("finish_position", "INTEGER"), ("is_winner", "INTEGER"), ("is_top3", "INTEGER"),
    ("jockey", "TEXT"), ("trainer", "TEXT"), ("sectional_times", "TEXT"),
    ("running_position", "TEXT"), ("finish_time", "REAL"), ("distance", "INTEGER"),
    ("going", "TEXT"), ("class_type", "TEXT"), ("weight", "INTEGER"),
]
_COL_NAMES = [c for c, _ in _COLS]


def _make_db(path):
    con = sqlite3.connect(str(path))
    con.execute(f"CREATE TABLE race_results ({', '.join(f'{c} {t}' for c, t in _COLS)})")
    # LEAKY: a clear winner in the past — if any query drops the date filter,
    # this row leaks into the features and shifts them off default.
    con.execute(
        f"INSERT INTO race_results ({', '.join(_COL_NAMES)}) "
        f"VALUES ({', '.join('?' * len(_COL_NAMES))})",
        ("LEAKY", "2026/06/01", 1, 1, 1, 1, "PURTON", "SIZE",
         "23.5 23.0 22.8", "1 1 1", 68.5, 1200, "GOOD", "3", 126),
    )
    con.commit()
    con.close()


@pytest.fixture
def leaky_db(tmp_path, monkeypatch):
    db = tmp_path / "leaky.db"
    _make_db(db)
    monkeypatch.setattr(config, "DB_PATH", db)
    clear_feature_caches()
    yield db
    clear_feature_caches()


def test_dateless_extract_is_leak_free(leaky_db):
    """No race_date => LEAKY's own win must NOT enter the vector.

    A horse with a winning DB row and an unknown horse must produce the SAME
    (default) vector when no date is supplied.
    """
    leaky = {"name": "LEAKY", "distance": 1200, "field_size": 12, "draw": 5}
    unknown = {"name": "NOBODY_XYZ_9999", "distance": 1200, "field_size": 12, "draw": 5}

    v_leaky = FeatureEngineering.extract_features(leaky, [], "Sha Tin")
    clear_feature_caches()
    v_unknown = FeatureEngineering.extract_features(unknown, [], "Sha Tin")

    assert np.allclose(np.asarray(v_leaky, dtype=float),
                       np.asarray(v_unknown, dtype=float)), (
        "date-less extract_features leaked LEAKY's own DB history into the vector"
    )


def test_dated_extract_uses_prior_history(leaky_db):
    """Positive control: WITH a race_date after the prior row, the vector must
    differ from the leak-free default — proving we disabled leakage, not the DB.
    """
    dateless = {"name": "LEAKY", "distance": 1200, "field_size": 12, "draw": 5}
    dated = {"name": "LEAKY", "distance": 1200, "field_size": 12, "draw": 5,
             "race_date": "2026/07/01"}

    v_dateless = FeatureEngineering.extract_features(dateless, [], "Sha Tin")
    clear_feature_caches()
    v_dated = FeatureEngineering.extract_features(dated, [], "Sha Tin")

    assert not np.allclose(np.asarray(v_dateless, dtype=float),
                           np.asarray(v_dated, dtype=float)), (
        "dated extract_features ignored the horse's prior history — DB not read"
    )


def test_helper_refuses_unfiltered_read_with_live_conn(leaky_db):
    """Unit lock: _compute_jockey_win_rate with a live conn but before_date=None
    must return the league default (0.08), NOT the DB aggregate — no unfiltered
    race_results query may run even when a connection is handed in.
    """
    con = sqlite3.connect(str(leaky_db))
    con.row_factory = sqlite3.Row
    try:
        rate = FeatureEngineering._compute_jockey_win_rate("PURTON", [], con, None)
    finally:
        con.close()
    assert rate == 0.08, f"expected league default 0.08, got {rate} (unfiltered read leaked)"
