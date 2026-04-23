"""Shared pytest fixtures for HKJC Racing Analyzer tests."""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

import config


@pytest.fixture
def sample_horse_data():
    """Minimal horse data dict with all fields extract_features expects."""
    return {
        "number": 1,
        "name": "Test Horse",
        "rating": 65.0,
        "draw": 3,
        "jockey": "Z Purton",
        "trainer": "J Size",
        "last_6": "1/3/2/5/4/1",
        "distance": 1200,
        "running_style": "Stalker",
        "last_400m_avg": 23.5,
        "cd_wins": 2,
        "cd_runs": 5,
        "opening_odds": 4.0,
        "win_odds": 3.5,
        "tips_odds": 4.2,
        "course": "A",
        "venue": "Sha Tin",
        "gear": "B",
        "weight": "126",
        "class_type": "Class 3",
        "field_size": 12,
        "going": "Good",
        "days_since_last_run": 21,
        "race_date": "2026/04/17",
        "race_number": 1,
    }


@pytest.fixture
def sample_race_card():
    """Minimal RaceCard-like object for testing."""
    from core.fetcher import RaceCard, HorseEntry

    horses = []
    for i in range(1, 13):
        horses.append(HorseEntry(
            number=i,
            name=f"Horse {i}",
            draw=i,
            jockey=f"Jockey {i}",
            trainer=f"Trainer {i}",
            weight="126",
            rating=50.0 + i * 3,
            win_odds=3.0 + i * 2,
            last_6="3/4/5/2/1/3",
        ))

    return RaceCard(
        race_number=1,
        venue="Sha Tin",
        date="2026/04/17",
        time="13:00",
        distance="1200",
        class_type="Class 3",
        going="Good",
        track="Turf",
        course="A",
        horses=horses,
    )


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite DB with fake race results for testing."""
    db_file = tmp_path / "test_horse_history.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE race_results (
            race_date TEXT, race_number INTEGER, horse_name TEXT,
            horse_number INTEGER, jockey TEXT, trainer TEXT,
            finish_position INTEGER, is_winner INTEGER, is_top3 INTEGER,
            win_odds REAL, draw INTEGER, weight TEXT, horse_weight INTEGER,
            distance INTEGER, going TEXT, class_type TEXT,
            running_position TEXT, finish_time TEXT, margin TEXT,
            days_since_last_run INTEGER, rating REAL, rating_change INTEGER,
            age TEXT, last_6 TEXT, gear TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE jockey_stats (
            name TEXT PRIMARY KEY, rides INTEGER, wins INTEGER,
            places INTEGER, win_rate REAL, place_rate REAL
        )
    """)
    conn.execute("""
        CREATE TABLE trainer_stats (
            name TEXT PRIMARY KEY, runners INTEGER, wins INTEGER,
            places INTEGER, win_rate REAL, place_rate REAL
        )
    """)
    conn.execute("""
        CREATE TABLE horses (
            name TEXT PRIMARY KEY, total_starts INTEGER,
            total_wins INTEGER, total_places INTEGER
        )
    """)

    # Insert 20 fake race results across 2 races
    for race_no in [1, 2]:
        for horse_no in range(1, 11):
            pos = horse_no  # horse 1 wins, horse 2 is second, etc.
            conn.execute("""
                INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "2026/04/10", race_no, f"Horse {horse_no}",
                horse_no, f"Jockey {horse_no}", f"Trainer {horse_no}",
                pos, 1 if pos == 1 else 0, 1 if pos <= 3 else 0,
                3.0 + horse_no, horse_no, "126", 1100,
                1200, "Good", "Class 3",
                f"{horse_no} {horse_no+1}", f"1:{10+horse_no}.{horse_no}0", "",
                21, 60 + horse_no, 0,
                "4", "3/4/5", "",
            ))

    conn.execute("INSERT INTO jockey_stats VALUES (?, ?, ?, ?, ?, ?)",
                 ("Z Purton", 100, 20, 35, 0.20, 0.35))
    conn.execute("INSERT INTO trainer_stats VALUES (?, ?, ?, ?, ?, ?)",
                 ("J Size", 80, 12, 25, 0.15, 0.31))
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture
def trained_model():
    """Small model trained on synthetic data for testing."""
    from ml.model import HKJCRacingMLModel, ML_AVAILABLE
    from ml.training import generate_synthetic_training_data

    if not ML_AVAILABLE:
        pytest.skip("XGBoost not available")

    model = HKJCRacingMLModel()
    X, y = generate_synthetic_training_data(500)
    model.train(X, y, save=False)
    return model
