"""Tier-stats hit-rate + Wilson 95% CI calculation."""
import math
import pytest
import sqlite3
import tempfile
import os
from betting.tier_stats import compute_tier_stats, wilson_ci_95


def test_wilson_ci_known_proportion():
    low, high = wilson_ci_95(7, 10)
    assert math.isclose(low, 0.397, abs_tol=0.01)
    assert math.isclose(high, 0.893, abs_tol=0.01)


def test_wilson_ci_small_sample():
    low, high = wilson_ci_95(1, 3)
    assert low < 0.10
    assert high > 0.70  # actual Wilson upper for 1/3 ≈ 0.792 (< 0.80 threshold in spec)


def test_wilson_ci_zero_hits():
    low, high = wilson_ci_95(0, 10)
    assert low == 0.0
    assert 0.20 < high < 0.35


def test_wilson_ci_zero_total():
    low, high = wilson_ci_95(0, 0)
    assert low == 0.0 and high == 1.0


def test_compute_tier_stats_end_to_end():
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE prediction_log (
                race_date TEXT, race_number INTEGER, horse_name TEXT,
                horse_number INTEGER, predicted_rank INTEGER,
                conviction_tier TEXT, bet_recommendation TEXT,
                PRIMARY KEY (race_date, race_number, horse_name)
            );
            CREATE TABLE race_results (
                race_date TEXT, race_number INTEGER, horse_name TEXT,
                horse_number INTEGER, finish_position INTEGER, is_winner INTEGER,
                PRIMARY KEY (race_date, race_number, horse_name)
            );
        """)
        for i in range(3):
            conn.execute(
                "INSERT INTO prediction_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f'2026/04/{20+i:02d}', 1, f'H{i}', 1, 1, 'HIGH', 'WIN'),
            )
            conn.execute(
                "INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?)",
                (f'2026/04/{20+i:02d}', 1, f'H{i}', 1, 1 if i < 2 else 3, 1 if i < 2 else 0),
            )
        conn.commit()

        stats = compute_tier_stats(conn, window=10)
        assert 'HIGH' in stats
        assert stats['HIGH']['hit'] == 2
        assert stats['HIGH']['total'] == 3
        assert math.isclose(stats['HIGH']['rate'], 2 / 3, abs_tol=0.01)
        assert 0 < stats['HIGH']['ci_low'] < stats['HIGH']['rate']
        assert stats['HIGH']['rate'] < stats['HIGH']['ci_high'] < 1.0
    finally:
        conn.close()
        os.unlink(db_path)


def test_compute_tier_stats_sliding_window():
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE prediction_log (
                race_date TEXT, race_number INTEGER, horse_name TEXT,
                horse_number INTEGER, predicted_rank INTEGER,
                conviction_tier TEXT, bet_recommendation TEXT,
                PRIMARY KEY (race_date, race_number, horse_name)
            );
            CREATE TABLE race_results (
                race_date TEXT, race_number INTEGER, horse_name TEXT,
                horse_number INTEGER, finish_position INTEGER, is_winner INTEGER,
                PRIMARY KEY (race_date, race_number, horse_name)
            );
        """)
        for i in range(5):
            is_win = 1 if i < 2 else 0
            conn.execute(
                "INSERT INTO prediction_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f'2026/04/{20+i:02d}', 1, f'H{i}', 1, 1, 'HIGH', 'WIN'),
            )
            conn.execute(
                "INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?)",
                (f'2026/04/{20+i:02d}', 1, f'H{i}', 1, 1 if is_win else 5, is_win),
            )
        conn.commit()

        stats = compute_tier_stats(conn, window=2)
        assert stats['HIGH']['hit'] == 0
        assert stats['HIGH']['total'] == 2
    finally:
        conn.close()
        os.unlink(db_path)
