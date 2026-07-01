"""Regression tests for the 2026-07-01 expert re-review bug fixes.

1. weight_vs_field (ML Feature 11) was DEAD at inference: field_weights arrives
   as a list of card strings, so the median math raised TypeError → caught →
   the feature collapsed to a constant 0.5 on every live horse (train/predict
   parity break). It must now compute a real, direction-correct value.

2. get_beaten_margins under-credited a fresh win: winners store margin=0 and were
   filtered out, so last_margin took the most recent LOSING race. A horse that
   won its last start (but lost earlier in the window) must yield last_margin=0
   and last_winner=True so beaten_margins scores it "Won last start".
"""
import os
import tempfile

import pytest

from ml.feature_schema import FEATURE_NAMES
from ml.features import FeatureEngineering

_WVF = FEATURE_NAMES.index("weight_vs_field")


def test_weight_vs_field_not_constant_with_string_field_weights():
    """The bug: string field_weights → TypeError → constant 0.5. Fixed: real value."""
    # 6-horse field, our horse (115) is the LIGHTEST → below median → score > 0.5.
    horse = {"name": "Lightweight", "distance": 1200, "field_size": 6,
             "weight": "115",
             "field_weights": ["133", "130", "128", "125", "120", "115"]}
    feats = FeatureEngineering.extract_features(horse, [], "Sha Tin")
    assert feats[_WVF] != 0.5, "weight_vs_field collapsed to the dead-feature default"
    assert feats[_WVF] > 0.5, "lighter-than-median horse should score above 0.5"


def test_weight_vs_field_directional_heavy():
    """Heaviest horse in the field → above median → score < 0.5."""
    horse = {"name": "Topweight", "distance": 1200, "field_size": 6,
             "weight": "133",
             "field_weights": ["133", "130", "128", "125", "120", "115"]}
    feats = FeatureEngineering.extract_features(horse, [], "Sha Tin")
    assert feats[_WVF] < 0.5, "heavier-than-median horse should score below 0.5"


@pytest.fixture
def margins_db():
    from data.database import HorseDatabase
    path = tempfile.mktemp(suffix=".db")
    db = HorseDatabase(path)
    yield db
    db.conn.close()
    for suf in ("", "-wal", "-shm"):
        try:
            os.remove(path + suf)
        except OSError:
            pass


def _add(db, horse, date, rno, pos, margin, cls="3"):
    db.conn.execute(
        "INSERT INTO race_results (horse_name, race_date, race_number, "
        "finish_position, margin, class_type) VALUES (?,?,?,?,?,?)",
        (horse, date, rno, pos, margin, cls))
    db.conn.commit()


def test_beaten_margins_won_last_start(margins_db):
    """Won last start (margin 0) but lost earlier → last_margin 0, last_winner True."""
    _add(margins_db, "WINNER", "2026/05/01", 1, 4, 3.0)   # older: lost by 3L
    _add(margins_db, "WINNER", "2026/06/01", 1, 1, 0.0)   # last start: WON
    r = margins_db.get_beaten_margins("WINNER", before_date="2026/07/01")
    assert r["last_winner"] is True
    assert r["last_margin"] == 0, (
        f"fresh win must give last_margin 0 (got {r['last_margin']} — stale loss leaked)")


def test_beaten_margins_lost_last_start(margins_db):
    """Lost last start → last_margin is that race's actual margin, not an older one."""
    _add(margins_db, "LOSER", "2026/05/01", 1, 1, 0.0)    # older: won
    _add(margins_db, "LOSER", "2026/06/01", 1, 5, 4.5)    # last start: lost by 4.5L
    r = margins_db.get_beaten_margins("LOSER", before_date="2026/07/01")
    assert r["last_winner"] is False
    assert r["last_margin"] > 0
