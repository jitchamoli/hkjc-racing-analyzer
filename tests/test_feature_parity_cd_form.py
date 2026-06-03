"""Train/predict parity for cd_form_prior (Feature index 6).

Bug (10-expert review 2026-06-03): extract_features() filtered prior runs at
the same distance with `p.get('distance') == distance`, but at inference
`distance` arrives as a STRING ('1200') while the DB stores it as INTEGER and
the training site (build_training_data_from_db) compares int==int. In Python
`1200 == '1200'` is False, so prior_at_dist was ALWAYS empty at inference and
cd_form_prior ALWAYS fell through to its 0.5 default — even though ~82% of
training rows had a real value. The model learned a feature it never saw live.

Fix: coerce distance to int in extract_features (mirrors calculate_draw_advantage
and the training site), so inference matches training.

These tests pin the parity.
"""
import sqlite3

import pytest

from ml.feature_schema import FEATURE_NAMES
from ml.features import FeatureEngineering

CD_IDX = FEATURE_NAMES.index("cd_form_prior")

_SCHEMA = """
CREATE TABLE race_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_date TEXT NOT NULL, race_number INTEGER NOT NULL, venue TEXT,
    distance INTEGER, class_type TEXT, going TEXT, course TEXT,
    horse_name TEXT NOT NULL, horse_number INTEGER, jockey TEXT, trainer TEXT,
    draw INTEGER DEFAULT 0, weight TEXT, rating REAL DEFAULT 0,
    win_odds REAL DEFAULT 0, finish_position INTEGER DEFAULT 0,
    is_winner INTEGER DEFAULT 0, is_top3 INTEGER DEFAULT 0,
    days_since_last_run INTEGER DEFAULT 0, rating_change INTEGER DEFAULT 0,
    horse_weight INTEGER DEFAULT 0, gear TEXT, margin REAL DEFAULT 0,
    last_6 TEXT DEFAULT '', age TEXT DEFAULT '', running_position TEXT DEFAULT '',
    finish_time TEXT DEFAULT '', sectional_times TEXT DEFAULT '',
    weather TEXT DEFAULT '',
    UNIQUE(race_date, race_number, horse_name)
)
"""


def _seed_conn():
    """In-memory DB: one horse with 5 prior runs at 1200m (int), 2 wins,
    all >180d before the current race so cd_score is a clean 0.4 (not 0.5)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    rows = [
        # (race_date, race_number, distance, is_winner, is_top3)
        ("2023/01/05", 1, 1200, 1, 1),
        ("2023/01/20", 2, 1200, 0, 1),
        ("2023/02/10", 3, 1200, 1, 1),
        ("2023/03/01", 4, 1200, 0, 0),
        ("2023/03/20", 5, 1200, 0, 0),
    ]
    for rd, rn, dist, win, top3 in rows:
        conn.execute(
            "INSERT INTO race_results (race_date, race_number, distance, "
            "horse_name, is_winner, is_top3) VALUES (?,?,?,?,?,?)",
            (rd, rn, dist, "TEST HORSE", win, top3),
        )
    conn.commit()
    return conn


def _cd_form(distance_value):
    conn = _seed_conn()
    horse_data = {
        "name": "TEST HORSE",
        "distance": distance_value,
        "race_date": "2023/12/01",
        "race_number": 9,
        "field_size": 12,
    }
    feats = FeatureEngineering.extract_features(horse_data, [], "", conn=conn)
    conn.close()
    return feats[CD_IDX]


def test_cd_form_string_distance_matches_int_distance():
    """The core parity bug: passing distance as the string '1200' must produce
    the SAME cd_form_prior as the int 1200 (the DB/training type)."""
    cd_str = _cd_form("1200")
    cd_int = _cd_form(1200)
    assert cd_str == pytest.approx(cd_int), (
        f"cd_form parity broken: str-distance={cd_str:.4f} int-distance={cd_int:.4f}. "
        f"Inference must coerce distance to int to match training."
    )


def test_cd_form_string_distance_is_not_the_default():
    """With 5 prior runs at 1200m (2 wins), cd_form must compute the real value
    (~0.4), NOT the 0.5 'no same-distance history' default. Pre-fix the string
    path silently returned 0.5 because the filter never matched."""
    cd_str = _cd_form("1200")
    assert cd_str != pytest.approx(0.5), (
        "cd_form_prior fell back to the 0.5 default for a horse WITH prior "
        "same-distance runs — the distance-type mismatch bug has regressed."
    )
    assert cd_str == pytest.approx(0.4, abs=0.01), (
        f"Expected cd_score 2/5=0.4 (no recent-win boost; wins >180d old), got {cd_str:.4f}"
    )
