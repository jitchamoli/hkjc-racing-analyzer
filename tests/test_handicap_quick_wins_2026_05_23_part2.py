"""Targeted tests for the 3 follow-up handicap fixes shipped 2026-05-23
(daily track variant, class-adjusted margins, rail-degradation taper)."""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from analytics.draw_bias import DrawBiasAnalyzer
from analytics.speed_figures import (
    adjusted_speed_figure,
    compute_daily_variant,
    parse_finish_time,
)
from data.database import HorseDatabase


# ---------------------------------------------------------------------------
# Fix #7: Daily track variant
# ---------------------------------------------------------------------------

def _seed_db_with_races(tmpdir: str, rows: list[dict]):
    """Seed an in-memory-style DB. rows = list of dicts with race_date,
    race_number, distance, going, class_type, finish_time, finish_position."""
    db_path = Path(tmpdir) / 'h.db'
    db = HorseDatabase(str(db_path))
    cur = db.conn.cursor()
    for i, r in enumerate(rows):
        cur.execute(
            """INSERT INTO race_results
               (race_date, race_number, horse_name, distance, going, class_type,
                finish_time, finish_position, is_winner, is_top3, win_odds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r['race_date'], r['race_number'], r.get('horse_name', f'H{i}'),
             r.get('distance', 1200), r.get('going', 'GOOD'),
             r.get('class_type', 'Class 3'), r.get('finish_time', ''),
             r.get('finish_position', 1),
             1 if r.get('finish_position', 1) == 1 else 0,
             1 if r.get('finish_position', 1) <= 3 else 0, 5.0),
        )
    db.conn.commit()
    return db


def test_compute_daily_variant_returns_zero_for_first_race_of_card():
    """No earlier races on the card → variant must be 0 (no signal yet)."""
    with tempfile.TemporaryDirectory() as d:
        db = _seed_db_with_races(d, [])
        v = compute_daily_variant(db.conn, '2025/05/24', 1, {})
        assert v == 0.0


def test_compute_daily_variant_returns_zero_with_empty_par_times():
    """If par_times has no lookup match, every race is skipped → 0."""
    with tempfile.TemporaryDirectory() as d:
        db = _seed_db_with_races(d, [
            {'race_date': '2025/05/24', 'race_number': 1, 'distance': 1200,
             'going': 'GOOD', 'class_type': 'Class 3', 'finish_time': '1:09.50'},
        ])
        v = compute_daily_variant(db.conn, '2025/05/24', 2, {})
        assert v == 0.0


def test_compute_daily_variant_uses_only_earlier_races_on_same_date():
    """Race 5 on date D must use races 1..4 on D — not later races, not
    other dates. This is the leak-safety property."""
    par = {(1200, 'GOOD', 'Class 3'): 70.0}  # par 70 secs
    with tempfile.TemporaryDirectory() as d:
        db = _seed_db_with_races(d, [
            # Card 2025/05/24 — races 1-4 ran +0.5s under par (fast track)
            {'race_date': '2025/05/24', 'race_number': 1, 'distance': 1200,
             'going': 'GOOD', 'class_type': 'Class 3', 'finish_time': '1:09.50'},  # 69.5
            {'race_date': '2025/05/24', 'race_number': 2, 'distance': 1200,
             'going': 'GOOD', 'class_type': 'Class 3', 'finish_time': '1:09.50'},
            {'race_date': '2025/05/24', 'race_number': 3, 'distance': 1200,
             'going': 'GOOD', 'class_type': 'Class 3', 'finish_time': '1:09.50'},
            {'race_date': '2025/05/24', 'race_number': 4, 'distance': 1200,
             'going': 'GOOD', 'class_type': 'Class 3', 'finish_time': '1:09.50'},
            # Race 5 — would-be future on same card; must be IGNORED
            {'race_date': '2025/05/24', 'race_number': 5, 'distance': 1200,
             'going': 'GOOD', 'class_type': 'Class 3', 'finish_time': '1:11.50'},  # slow
            # Different date — must be IGNORED
            {'race_date': '2025/05/25', 'race_number': 1, 'distance': 1200,
             'going': 'GOOD', 'class_type': 'Class 3', 'finish_time': '1:12.00'},
        ])
        v = compute_daily_variant(db.conn, '2025/05/24', 5, par)
        # Median of (69.5 - 70.0) × 4 = -0.5. Race 5 itself excluded.
        assert abs(v - (-0.5)) < 0.01


def test_adjusted_speed_figure_applies_daily_variant():
    """If daily variant says track ran 0.5s fast (variant=-0.5), then a horse
    finishing at par for the static corpus is actually 0.5s SLOWER than the
    day-adjusted par → should get a NEGATIVE figure."""
    par_times = {(1200, 'GOOD', 'Class 3'): 70.0}
    # No variant: time of exactly 70.0 → figure ≈ 0
    fig_no_variant = adjusted_speed_figure(
        '1:10.00', 1200, 'GOOD', 'Class 3', 120, par_times,
    )
    # With variant -0.5: effective par = 69.5; horse ran 70.0 → 0.5s slower
    fig_with_variant = adjusted_speed_figure(
        '1:10.00', 1200, 'GOOD', 'Class 3', 120, par_times, daily_variant=-0.5,
    )
    assert fig_no_variant > fig_with_variant
    assert fig_no_variant > -0.5 and fig_no_variant < 0.5  # near par
    assert fig_with_variant < 0  # slower than day-adjusted par


def test_adjusted_speed_figure_default_variant_is_backward_compat():
    """Old callers passing 6 args (no daily_variant) get same result."""
    par_times = {(1200, 'GOOD', 'Class 3'): 70.0}
    a = adjusted_speed_figure('1:09.50', 1200, 'GOOD', 'Class 3', 120, par_times)
    b = adjusted_speed_figure('1:09.50', 1200, 'GOOD', 'Class 3', 120, par_times, 0.0)
    assert a == b


# ---------------------------------------------------------------------------
# Fix #8: Class-adjusted beaten margins
# ---------------------------------------------------------------------------

def test_class_adjust_margin_class_3_is_baseline():
    """Class 3 = rank 3 = divisor 1.0 → margin unchanged."""
    assert HorseDatabase._class_adjust_margin(2.0, 'Class 3') == 2.0


def test_class_adjust_margin_class_1_shrinks_margin():
    """Class 1 = rank 1 = divisor 1.15 → 1L becomes ~0.87L (more impressive)."""
    adj = HorseDatabase._class_adjust_margin(1.0, 'Class 1')
    assert adj < 1.0
    assert abs(adj - (1.0 / 1.15)) < 0.01


def test_class_adjust_margin_class_5_expands_margin():
    """Class 5 = rank 5 = divisor 0.85 → 1L becomes ~1.18L (less impressive)."""
    adj = HorseDatabase._class_adjust_margin(1.0, 'Class 5')
    assert adj > 1.0
    assert abs(adj - (1.0 / 0.85)) < 0.01


def test_class_adjust_margin_zero_and_missing_class_pass_through():
    assert HorseDatabase._class_adjust_margin(0, 'Class 3') == 0.0
    assert HorseDatabase._class_adjust_margin(None, 'Class 3') == 0.0
    # Unparseable class → multiplier defaults to 1.0
    assert HorseDatabase._class_adjust_margin(2.0, 'NONSENSE') == 2.0


def test_get_beaten_margins_applies_class_adjustment_on_last_margin():
    """A 1L beating in Class 1 should produce a smaller last_margin
    than a 1L beating in Class 5."""
    with tempfile.TemporaryDirectory() as d:
        db_c1 = _seed_db_with_races(d, [{
            'race_date': '2025/01/01', 'race_number': 1, 'horse_name': 'C1H',
            'class_type': 'Class 1', 'finish_position': 3,
        }])
        # Patch in margin via direct UPDATE since seed helper doesn't include margin
        db_c1.conn.execute(
            "UPDATE race_results SET margin = 1.0 WHERE horse_name = 'C1H'"
        )
        db_c1.conn.commit()
        rec_c1 = db_c1.get_beaten_margins('C1H')

    with tempfile.TemporaryDirectory() as d:
        db_c5 = _seed_db_with_races(d, [{
            'race_date': '2025/01/01', 'race_number': 1, 'horse_name': 'C5H',
            'class_type': 'Class 5', 'finish_position': 3,
        }])
        db_c5.conn.execute(
            "UPDATE race_results SET margin = 1.0 WHERE horse_name = 'C5H'"
        )
        db_c5.conn.commit()
        rec_c5 = db_c5.get_beaten_margins('C5H')

    assert rec_c1['last_margin'] < rec_c5['last_margin'], (
        f"Class 1: {rec_c1['last_margin']}, Class 5: {rec_c5['last_margin']}"
    )


# ---------------------------------------------------------------------------
# Fix #9: Rail-degradation taper across the meeting
# ---------------------------------------------------------------------------

def test_rail_degradation_nudges_inside_ideal_at_race_6_on_true_rail():
    """Race 6 on rail A with inside-favoring distance/venue → score for
    draw=1 should be LOWER than the same setup at race 1."""
    course = 'Turf - "A" Course'
    s_early, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=1, distance='1200', course=course, venue='Happy Valley',
        race_no=1,
    )
    s_late, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=1, distance='1200', course=course, venue='Happy Valley',
        race_no=8,
    )
    assert s_late < s_early, (
        f"Late-card inside should taper; got early={s_early}, late={s_late}"
    )


def test_rail_degradation_does_not_fire_on_outside_rail():
    """Outer-rail configurations (B, C, C+3) already favor middle/outside
    draws — no taper needed."""
    course = 'Turf - "C+3" Course'
    s_early, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=8, distance='1200', course=course, venue='Happy Valley',
        race_no=1,
    )
    s_late, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=8, distance='1200', course=course, venue='Happy Valley',
        race_no=8,
    )
    assert s_early == s_late


def test_rail_degradation_default_race_no_zero_is_backward_compat():
    """Old callers that don't pass race_no get the no-taper behavior."""
    course = 'Turf - "A" Course'
    s_default, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=1, distance='1200', course=course, venue='Happy Valley',
    )
    s_race_1, _ = DrawBiasAnalyzer.analyze_draw_with_rail(
        draw=1, distance='1200', course=course, venue='Happy Valley',
        race_no=1,
    )
    assert s_default == s_race_1
