"""Targeted tests for the 6 Tier-1 handicap quick-wins shipped 2026-05-23
from the 4-expert review. Each test pins ONE behavior change so a regression
re-surfacing the old bug fails this file rather than slipping through.
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from analytics.class_movement import ClassMovementAnalyzer
from analytics.draw_bias import DrawBiasAnalyzer
from analytics.pace import PaceAnalyzer
from core.fetcher import RaceCard
from data.database import HorseDatabase


# ---------------------------------------------------------------------------
# Fix #1: GOOD TO YIELDING surface-class leak
# ---------------------------------------------------------------------------

def test_going_surface_class_buckets_yielding_as_soft():
    assert HorseDatabase._going_surface_class('GOOD TO YIELDING') == 'SOFT'
    assert HorseDatabase._going_surface_class('YIELDING') == 'SOFT'
    assert HorseDatabase._going_surface_class('SOFT') == 'SOFT'
    assert HorseDatabase._going_surface_class('HEAVY') == 'SOFT'
    assert HorseDatabase._going_surface_class('WET-SLOW') == 'SOFT'


def test_going_surface_class_buckets_good_firm_fast_as_firm():
    assert HorseDatabase._going_surface_class('GOOD') == 'FIRM'
    assert HorseDatabase._going_surface_class('GOOD TO FIRM') == 'FIRM'
    assert HorseDatabase._going_surface_class('FAST') == 'FIRM'
    assert HorseDatabase._going_surface_class('FIRM') == 'FIRM'


def test_going_surface_class_handles_empty_and_none():
    assert HorseDatabase._going_surface_class('') == 'FIRM'
    assert HorseDatabase._going_surface_class(None) == 'FIRM'


def _seed_db_with_going(tmpdir: str, going_runs: list[tuple[str, int]]):
    """Seed an in-memory-style DB with one horse running on the given goings.
    going_runs is list of (going_string, finish_position)."""
    db_path = Path(tmpdir) / 'h.db'
    db = HorseDatabase(str(db_path))
    cur = db.conn.cursor()
    for i, (going, pos) in enumerate(going_runs):
        cur.execute(
            """INSERT INTO race_results
               (race_date, race_number, horse_name, going, finish_position,
                is_winner, is_top3, win_odds)
               VALUES (?, ?, 'TEST_HORSE', ?, ?, ?, ?, ?)""",
            (f'2025/01/{i+1:02d}', 1, going, pos,
             1 if pos == 1 else 0, 1 if pos <= 3 else 0, 5.0),
        )
    db.conn.commit()
    return db


def test_get_horse_going_record_does_not_leak_soft_into_firm():
    """A horse with only YIELDING priors should have 0 firm-side runs."""
    with tempfile.TemporaryDirectory() as d:
        db = _seed_db_with_going(d, [
            ('YIELDING', 3),
            ('GOOD TO YIELDING', 5),
            ('SOFT', 1),
        ])
        # Today: GOOD (firm). Old code matched 'GOOD TO YIELDING' via LIKE 'GOOD%'.
        rec = db.get_horse_going_record('TEST_HORSE', 'GOOD')
        assert rec['runs'] == 0, (
            f"Soft-side priors leaked into firm query: {rec}"
        )


def test_get_horse_going_record_matches_soft_to_soft():
    """A horse with YIELDING priors should match when today is YIELDING."""
    with tempfile.TemporaryDirectory() as d:
        db = _seed_db_with_going(d, [
            ('YIELDING', 2),
            ('GOOD TO YIELDING', 4),
        ])
        rec = db.get_horse_going_record('TEST_HORSE', 'YIELDING')
        assert rec['runs'] == 2


def test_get_horse_going_record_matches_firm_to_firm_excluding_soft():
    """A horse with mixed priors: today=GOOD only counts firm-side runs."""
    with tempfile.TemporaryDirectory() as d:
        db = _seed_db_with_going(d, [
            ('GOOD', 1),
            ('GOOD TO FIRM', 3),
            ('GOOD TO YIELDING', 2),  # soft-side — must be excluded
        ])
        rec = db.get_horse_going_record('TEST_HORSE', 'GOOD')
        assert rec['runs'] == 2  # only the two firm-side rows


# ---------------------------------------------------------------------------
# Fix #2: ST chute bucket (1000m + 1200m straight chute → high-draw bias)
# ---------------------------------------------------------------------------

def test_st_1000m_uses_chute_bucket_with_high_draw_ideal():
    """High draw on ST 1000m chute should score better than low draw on the
    same race (opposite of round-track short bias)."""
    course = 'Turf - "A" Course'
    score_high, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=9, distance='1000', course=course, venue='Sha Tin',
    )
    score_low, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=2, distance='1000', course=course, venue='Sha Tin',
    )
    assert score_high > score_low, (
        f"ST 1000m chute should favour high draws; got high={score_high} low={score_low}"
    )


def test_st_1200m_also_uses_chute_bucket():
    """ST 1200m runs on the chute too (per pace expert)."""
    course = 'Turf - "A" Course'
    score_high, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=8, distance='1200', course=course, venue='Sha Tin',
    )
    score_low, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=2, distance='1200', course=course, venue='Sha Tin',
    )
    assert score_high > score_low


def test_hv_1200m_still_uses_round_short_bucket_low_draw_bias():
    """HV 1200m is NOT a chute — low draw should still win on a tight turning track."""
    course = 'Turf - "A" Course'
    score_low, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=1, distance='1200', course=course, venue='Happy Valley',
    )
    score_high, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=10, distance='1200', course=course, venue='Happy Valley',
    )
    assert score_low > score_high


# ---------------------------------------------------------------------------
# Fix #3: field_size clips ideal draw
# ---------------------------------------------------------------------------

def test_field_size_clips_ideal_when_smaller_than_default():
    """Ideal=8 on ST mid+B but field=6 → effective ideal=6, so draw=6 hits ideal."""
    course = 'Turf - "B" Course'
    # Without field_size, ideal would be 6 (ST mid B → ideal 6); pick another
    # configuration where ideal exceeds a small field.
    score_no_clip, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=8, distance='1800', course='Turf - "C+3" Course', venue='Sha Tin',
        field_size=0,
    )
    score_clipped, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=6, distance='1800', course='Turf - "C+3" Course', venue='Sha Tin',
        field_size=6,
    )
    # With field=6 clipping ideal from 9→6, draw=6 should hit ideal (score≈95).
    assert score_clipped >= 90, f"Expected clipped draw=6 to hit ideal; got {score_clipped}"


def test_field_size_zero_means_no_clipping_backward_compat():
    """Old callers that don't pass field_size should get identical behavior."""
    course = 'Turf - "A" Course'
    score_default, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=5, distance='1600', course=course, venue='Sha Tin',
    )
    score_zero, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=5, distance='1600', course=course, venue='Sha Tin', field_size=0,
    )
    assert score_default == score_zero


# ---------------------------------------------------------------------------
# Fix #4: weather field on RaceCard + DB schema
# ---------------------------------------------------------------------------

def test_racecard_has_weather_field():
    rc = RaceCard(race_number=1)
    assert hasattr(rc, 'weather')
    assert rc.weather == ''


def test_racecard_accepts_weather_value():
    rc = RaceCard(race_number=1, weather='Cloudy')
    assert rc.weather == 'Cloudy'


def test_race_results_schema_has_weather_column():
    """ALTER TABLE migration adds weather column on init."""
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / 'h.db'
        db = HorseDatabase(str(db_path))
        cols = {row[1] for row in db.conn.execute('PRAGMA table_info(race_results)')}
        assert 'weather' in cols


# ---------------------------------------------------------------------------
# Fix #5: Class-drop multi-race lookback
# ---------------------------------------------------------------------------

def test_class_drop_after_3_poor_finishes_at_higher_class_gets_premium():
    """C2-C2-C2 with avg finish 7 → dropping to C3 today should score 92+
    (the new multi-race drop branch), not just the regular 72 (single drop)."""
    history = [
        {'class_type': 'Class 2', 'finish_position': 7},
        {'class_type': 'Class 2', 'finish_position': 8},
        {'class_type': 'Class 2', 'finish_position': 6},
    ]
    score, reason = ClassMovementAnalyzer.analyze_class_movement(history, 'Class 3')
    assert score >= 90, f"Expected multi-race drop bonus; got {score} ({reason})"
    assert 'DROP AFTER POOR FORM' in reason


def test_class_drop_after_good_form_does_not_fire_multi_race_branch():
    """Same 3-race drop but avg finish 3 (was running well) → just the
    regular drop score (~72), not the after-poor-form premium."""
    history = [
        {'class_type': 'Class 2', 'finish_position': 2},
        {'class_type': 'Class 2', 'finish_position': 3},
        {'class_type': 'Class 2', 'finish_position': 4},
    ]
    score, reason = ClassMovementAnalyzer.analyze_class_movement(history, 'Class 3')
    assert score < 90, f"Should NOT fire DROP AFTER POOR FORM with good form; got {score}"
    assert 'DROP AFTER POOR FORM' not in reason


def test_class_drop_single_race_history_falls_back_to_old_logic():
    """Only 1 prior race — multi-race branch can't fire."""
    history = [{'class_type': 'Class 2', 'finish_position': 8}]
    score, reason = ClassMovementAnalyzer.analyze_class_movement(history, 'Class 3')
    assert 'DROP AFTER POOR FORM' not in reason
    assert score == 72.0  # single-race drop, existing logic


def test_class_drop_with_one_prior_at_same_class_does_not_fire():
    """[Class 2, Class 3, Class 4] → Class 4 must NOT fire DROP AFTER POOR FORM.
    One prior was at the SAME class as today (Class 4), so this isn't a clean
    "all priors at higher class" sequence. Previous median-rank logic over-fired
    (median(2,3,4)=3 < 4), but the all() fix correctly rejects."""
    history = [
        {'class_type': 'Class 2', 'finish_position': 8},
        {'class_type': 'Class 3', 'finish_position': 7},
        {'class_type': 'Class 4', 'finish_position': 9},
    ]
    score, reason = ClassMovementAnalyzer.analyze_class_movement(history, 'Class 4')
    assert 'DROP AFTER POOR FORM' not in reason, (
        f"[2,3,4]→4 over-fired; one prior is at same class as today, "
        f"not a clean drop. Got: {reason}"
    )


def test_class_drop_all_higher_class_does_fire():
    """[Class 1, Class 3, Class 3] → Class 4: all priors at strictly higher
    class than today → IS a legitimate multi-race drop and should fire."""
    history = [
        {'class_type': 'Class 1', 'finish_position': 7},
        {'class_type': 'Class 3', 'finish_position': 8},
        {'class_type': 'Class 3', 'finish_position': 6},
    ]
    score, reason = ClassMovementAnalyzer.analyze_class_movement(history, 'Class 4')
    assert 'DROP AFTER POOR FORM' in reason, (
        f"[1,3,3]→4: all priors at strictly higher class, avg_fp=7, "
        f"should fire DROP AFTER POOR FORM. Got: {reason}"
    )


# ---------------------------------------------------------------------------
# Fix #6: classify_style_from_db returns '' on missing data
# ---------------------------------------------------------------------------

def test_classify_style_from_db_returns_empty_on_no_horse_name():
    assert PaceAnalyzer.classify_style_from_db('') == ''
    assert PaceAnalyzer.classify_style_from_db(None) == ''


def test_classify_style_from_db_returns_empty_when_no_db_data():
    """No prior data → empty string so caller falls through to last_6 heuristic."""
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / 'h.db'
        db = HorseDatabase(str(db_path))
        style = PaceAnalyzer.classify_style_from_db(
            'UNKNOWN_HORSE', conn=db.conn, before_date='2025-01-01',
        )
        assert style == ''
