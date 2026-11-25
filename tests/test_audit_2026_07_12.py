"""Regression tests for the 2026-07-12 expert-panel audit.

Three defects that changed what we scored and what we bet:
  1. get_horse_going_record dropped 64% of race_results — the raw SQL LIKE
     filters didn't know HKJC's short going codes (G/GF/GD/GY/Y/WS/S).
  2. BOX QIN-4 PROTECTION was silently suppressed on every short-favourite
     race, because primary_nums swept in the QIN PLACE BANKER's legs.
  3. The jockey/trainer tier fallback was dead in LIVE (not just backtest),
     collapsing every rider to default_score on the first meeting of a season.
"""
import sqlite3

import pytest

from core.analyzer import BettingStrategy, _TIER_LIST_ASOF
from data.database import HorseDatabase


# --------------------------------------------------------------------------
# 1. going record must understand HKJC short codes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected", [
    ("G", "FIRM"), ("GF", "FIRM"), ("GD", "FIRM"), ("FT", "FIRM"),
    ("GOOD", "FIRM"), ("Good To Firm", "FIRM"),
    ("GY", "SOFT"), ("Y", "SOFT"), ("YS", "SOFT"),
    ("S", "SOFT"), ("H", "SOFT"), ("WS", "SOFT"),
    ("GOOD TO YIELDING", "SOFT"), ("SOFT", "SOFT"), ("HEAVY", "SOFT"),
])
def test_going_surface_class_understands_short_codes(code, expected):
    """'GY'/'WS'/'S' contain no YIELDING/SOFT substring — the old raw keyword
    test called them all FIRM."""
    assert HorseDatabase._going_surface_class(code) == expected


def _seed(db, rows):
    for date, going, pos in rows:
        db.conn.execute(
            "INSERT INTO race_results (race_date, race_number, horse_number, "
            "horse_name, going, finish_position, is_winner, is_top3) "
            "VALUES (?, 1, 1, 'TESTER', ?, ?, ?, ?)",
            (date, going, pos, 1 if pos == 1 else 0, 1 if pos <= 3 else 0),
        )
    db.conn.commit()


def test_going_record_counts_short_code_priors(tmp_path):
    """A horse whose firm-side history is stored as 'G'/'GF' must not read as
    a first-timer on GOOD TO FIRM."""
    db = HorseDatabase(db_path=str(tmp_path / "t.db"))
    _seed(db, [
        ("2025/01/01", "G", 1),    # firm, won
        ("2025/02/01", "GF", 2),   # firm, placed
        ("2025/03/01", "GD", 5),   # firm
        ("2025/04/01", "GY", 1),   # SOFT side — must not count
        ("2025/05/01", "WS", 1),   # SOFT side — must not count
    ])
    rec = db.get_horse_going_record("TESTER", "Good To Firm")
    assert rec["runs"] == 3, "short-code firm priors were dropped"
    assert rec["wins"] == 1
    assert rec["places"] == 2


def test_going_record_soft_side_counts_short_codes(tmp_path):
    db = HorseDatabase(db_path=str(tmp_path / "t.db"))
    _seed(db, [
        ("2025/01/01", "GY", 1),
        ("2025/02/01", "WS", 4),
        ("2025/03/01", "G", 1),   # FIRM — must not count
    ])
    rec = db.get_horse_going_record("TESTER", "GOOD TO YIELDING")
    assert rec["runs"] == 2
    assert rec["wins"] == 1


def test_going_record_still_honours_before_date(tmp_path):
    """The leak guard must survive the rewrite."""
    db = HorseDatabase(db_path=str(tmp_path / "t.db"))
    _seed(db, [("2025/01/01", "G", 1), ("2025/06/01", "G", 1)])
    rec = db.get_horse_going_record("TESTER", "GOOD", before_date="2025/03/01")
    assert rec["runs"] == 1, "before_date filter dropped or leaked"


def test_going_record_ignores_unknown_going(tmp_path):
    """Blank going is not firm-side experience."""
    db = HorseDatabase(db_path=str(tmp_path / "t.db"))
    _seed(db, [("2025/01/01", "", 1), ("2025/02/01", "G", 1)])
    assert db.get_horse_going_record("TESTER", "GOOD")["runs"] == 1


# --------------------------------------------------------------------------
# 2. BOX QIN-4 PROTECTION must survive a QIN PLACE BANKER
# --------------------------------------------------------------------------

def test_qin_protection_not_suppressed_by_qin_place_banker():
    """QIN PLACE BANKER's legs are exactly the top 4. Counting them as
    'primary' made has_non_primary False and killed the protection bet on
    every short-favourite race — and QIN PLACE is a different pool, so nothing
    covered it."""
    from dashboard.generator import build_bet_ticket

    horses = [
        {"number": 1, "name": "A", "rank": 1, "win_odds": 1.5, "calibrated_prob": 0.45},
        {"number": 2, "name": "B", "rank": 2, "win_odds": 6.0, "calibrated_prob": 0.18},
        {"number": 3, "name": "C", "rank": 3, "win_odds": 9.0, "calibrated_prob": 0.12},
        {"number": 4, "name": "D", "rank": 4, "win_odds": 12.0, "calibrated_prob": 0.09},
        {"number": 5, "name": "E", "rank": 5, "win_odds": 15.0, "calibrated_prob": 0.08},
        {"number": 6, "name": "F", "rank": 6, "win_odds": 20.0, "calibrated_prob": 0.05},
    ]
    race = {
        "race_number": 1, "conviction_tier": "HIGH", "going": "Good To Firm",
        "horses": horses, "distance": 1400,
        "suggested_stake": {"amount": 300.0},
    }
    ticket = build_bet_ticket(
        race, race_date="2026/07/12", model_version="test",
        generated_at="2026-07-12 10:00:00",
    )
    labels = [b["label"] for b in ticket.get("bets", [])]

    assert "QPL BOX-3" in labels
    assert "QIN PLACE BANKER" not in labels
    assert "BOX QIN-4 PROTECTION" in labels, (
        "protection bet suppressed by the QIN PLACE BANKER's legs"
    )


# --------------------------------------------------------------------------
# 3. jockey/trainer tier fallback must be live in production
# --------------------------------------------------------------------------

def test_tier_fallback_active_for_races_after_asof():
    """analyze_horse always passes before_date=race.date, so gating the tier
    fallback on 'before_date is set' killed it in LIVE too."""
    elite, _ = BettingStrategy.analyze_jockey(
        "Z Purton", db=None, before_date="2026/07/01")
    unknown, _ = BettingStrategy.analyze_jockey(
        "A Nobody", db=None, before_date="2026/07/01")
    assert elite > unknown, (
        "tier fallback dead in live — every rider collapsed to default_score"
    )


def test_tier_fallback_still_suppressed_before_asof():
    """The leak guard must still hold for historical races: the tier sets are a
    later opinion and must not be applied backwards."""
    elite, _ = BettingStrategy.analyze_jockey(
        "Z Purton", db=None, before_date="2024/01/01")
    unknown, _ = BettingStrategy.analyze_jockey(
        "A Nobody", db=None, before_date="2024/01/01")
    assert elite == unknown, "tier opinion leaked backwards into a 2024 race"


def test_tier_asof_boundary_is_inclusive():
    on_asof, _ = BettingStrategy.analyze_jockey(
        "Z Purton", db=None, before_date=_TIER_LIST_ASOF)
    neutral, _ = BettingStrategy.analyze_jockey(
        "A Nobody", db=None, before_date=_TIER_LIST_ASOF)
    assert on_asof == neutral, "a race ON the as-of date must not use the tiers"
