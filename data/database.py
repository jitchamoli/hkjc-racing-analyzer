#!/usr/bin/env python3
"""
Persistent Horse Database for HKJC Racing Analyzer.
SQLite-based local database that accumulates horse, jockey, and trainer
stats across race days. Never replaced by fresh race data — only appended to.
"""

import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / 'tracking_data'
DB_DIR.mkdir(exist_ok=True)
DB_FILE = DB_DIR / 'horse_history.db'


class HorseDatabase:
    """Persistent local database for horse racing history."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_FILE)
        self.conn = sqlite3.connect(self.db_path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()

        # Horse master record — one row per horse, updated each race
        c.execute("""
            CREATE TABLE IF NOT EXISTS horses (
                name TEXT PRIMARY KEY,
                brand_no TEXT,
                age TEXT,
                sex TEXT,
                import_cat TEXT,
                trainer TEXT,
                rating REAL DEFAULT 0,
                total_starts INTEGER DEFAULT 0,
                total_wins INTEGER DEFAULT 0,
                total_places INTEGER DEFAULT 0,
                total_earnings REAL DEFAULT 0,
                last_seen_date TEXT,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Race results — one row per horse per race
        c.execute("""
            CREATE TABLE IF NOT EXISTS race_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_date TEXT NOT NULL,
                race_number INTEGER NOT NULL,
                venue TEXT,
                distance INTEGER,
                class_type TEXT,
                going TEXT,
                course TEXT,
                horse_name TEXT NOT NULL,
                horse_number INTEGER,
                jockey TEXT,
                trainer TEXT,
                draw INTEGER DEFAULT 0,
                weight TEXT,
                rating REAL DEFAULT 0,
                win_odds REAL DEFAULT 0,
                finish_position INTEGER DEFAULT 0,
                is_winner INTEGER DEFAULT 0,
                is_top3 INTEGER DEFAULT 0,
                days_since_last_run INTEGER DEFAULT 0,
                rating_change INTEGER DEFAULT 0,
                horse_weight INTEGER DEFAULT 0,
                gear TEXT,
                margin REAL DEFAULT 0,
                UNIQUE(race_date, race_number, horse_name)
            )
        """)

        # Add columns if upgrading from older schema
        for col_def in [
            "margin REAL DEFAULT 0",
            "last_6 TEXT DEFAULT ''",
            "age TEXT DEFAULT ''",
            "running_position TEXT DEFAULT ''",
            "finish_time TEXT DEFAULT ''",
        ]:
            try:
                c.execute(f"ALTER TABLE race_results ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists

        # Jockey season stats
        c.execute("""
            CREATE TABLE IF NOT EXISTS jockey_stats (
                name TEXT PRIMARY KEY,
                season TEXT,
                rides INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                places INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                place_rate REAL DEFAULT 0,
                last_updated TEXT
            )
        """)

        # Trainer season stats
        c.execute("""
            CREATE TABLE IF NOT EXISTS trainer_stats (
                name TEXT PRIMARY KEY,
                season TEXT,
                runners INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                places INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                place_rate REAL DEFAULT 0,
                last_updated TEXT
            )
        """)

        # Prediction tracking — one row per horse per race, filled with results later
        c.execute("""
            CREATE TABLE IF NOT EXISTS prediction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_date TEXT NOT NULL,
                race_number INTEGER NOT NULL,
                venue TEXT,
                distance INTEGER,
                class_type TEXT,
                going TEXT,
                field_size INTEGER,
                horse_name TEXT NOT NULL,
                horse_number INTEGER,
                predicted_rank INTEGER,
                rule_score REAL,
                ml_prob REAL,
                calibrated_prob REAL,
                expected_return REAL,
                bet_recommendation TEXT,
                confidence TEXT,
                win_odds REAL,
                model_version TEXT,
                actual_position INTEGER DEFAULT 0,
                actual_won INTEGER DEFAULT 0,
                actual_top3 INTEGER DEFAULT 0,
                actual_odds REAL DEFAULT 0,
                profit_loss REAL DEFAULT 0,
                results_filled INTEGER DEFAULT 0,
                predicted_at TEXT,
                results_filled_at TEXT,
                UNIQUE(race_date, race_number, horse_name)
            )
        """)

        # Race-level dividends (scraped from HKJC results pages)
        c.execute("""
            CREATE TABLE IF NOT EXISTS race_dividends (
                race_date TEXT NOT NULL,
                race_number INTEGER NOT NULL,
                pool TEXT NOT NULL,
                combination TEXT NOT NULL,
                dividend REAL NOT NULL,
                PRIMARY KEY (race_date, race_number, pool, combination)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_dividends_date ON race_dividends(race_date, race_number)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_dividends_pool ON race_dividends(pool)")

        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_plog_date_filled ON prediction_log(race_date, results_filled)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_plog_date ON prediction_log(race_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_plog_bet ON prediction_log(bet_recommendation)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_horse ON race_results(horse_name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_horse_date ON race_results(horse_name, race_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_date ON race_results(race_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_venue_dist ON race_results(venue, distance)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_jockey ON race_results(jockey)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_trainer ON race_results(trainer)")

        self.conn.commit()

    # ------------------------------------------------------------------
    # Ingest race results
    # ------------------------------------------------------------------

    def ingest_race_day(self, race_data: List[Dict]):
        """Ingest a full race day of results into the database.

        race_data: list of race dicts with 'entries' containing horse results.
        This is the same format as training_data/historical_races.json.
        """
        try:
            self._ingest_race_day_inner(race_data)
        except Exception:
            self.conn.rollback()
            logger.error("ingest_race_day failed — rolled back", exc_info=True)
            raise

    def _ingest_race_day_inner(self, race_data: List[Dict]):
        c = self.conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M')

        for race in race_data:
            race_date = race.get('date', '')
            race_no = race.get('race_no', 0)
            venue = race.get('venue', '')
            distance = int(race.get('distance', 0) or 0)
            class_type = race.get('class_type', '')
            going = race.get('going', '')
            course = race.get('course', '')

            for entry in race.get('entries', []):
                horse_name = entry.get('name', '')
                if not horse_name:
                    continue

                pos = entry.get('actual_position', 0)
                is_winner = 1 if pos == 1 else 0
                is_top3 = 1 if 1 <= pos <= 3 else 0

                # Upsert race result
                margin = entry.get('margin', 0) or 0
                hw = entry.get('horse_weight', 0) or 0
                last_6 = entry.get('last_6', '') or ''
                age = entry.get('age', '') or ''
                running_pos = entry.get('running_position', '') or ''
                finish_time = entry.get('finish_time', '') or ''
                days_since = entry.get('days_since_last_run', 0) or 0
                rat_change = entry.get('rating_change', 0) or 0
                c.execute("""
                    INSERT INTO race_results (
                        race_date, race_number, venue, distance, class_type,
                        going, course, horse_name, horse_number, jockey, trainer,
                        draw, weight, rating, win_odds, finish_position,
                        is_winner, is_top3, gear, margin, horse_weight,
                        last_6, age, running_position, finish_time,
                        days_since_last_run, rating_change
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(race_date, race_number, horse_name)
                    DO UPDATE SET
                        finish_position=CASE WHEN excluded.finish_position > 0 THEN excluded.finish_position ELSE race_results.finish_position END,
                        is_winner=CASE WHEN excluded.finish_position > 0 THEN excluded.is_winner ELSE race_results.is_winner END,
                        is_top3=CASE WHEN excluded.finish_position > 0 THEN excluded.is_top3 ELSE race_results.is_top3 END,
                        win_odds=CASE WHEN excluded.win_odds > 0 THEN excluded.win_odds ELSE race_results.win_odds END,
                        margin=CASE WHEN excluded.margin > 0 THEN excluded.margin ELSE race_results.margin END,
                        horse_weight=CASE WHEN excluded.horse_weight > 0 THEN excluded.horse_weight ELSE race_results.horse_weight END,
                        last_6=CASE WHEN excluded.last_6 != '' THEN excluded.last_6 ELSE race_results.last_6 END,
                        age=CASE WHEN excluded.age != '' THEN excluded.age ELSE race_results.age END,
                        running_position=CASE WHEN excluded.running_position != '' THEN excluded.running_position ELSE race_results.running_position END,
                        finish_time=CASE WHEN excluded.finish_time != '' THEN excluded.finish_time ELSE race_results.finish_time END,
                        days_since_last_run=CASE WHEN excluded.days_since_last_run > 0 THEN excluded.days_since_last_run ELSE race_results.days_since_last_run END,
                        rating_change=CASE WHEN excluded.rating_change != 0 THEN excluded.rating_change ELSE race_results.rating_change END
                """, (
                    race_date, race_no, venue, distance, class_type,
                    going, course, horse_name, entry.get('number', 0),
                    entry.get('jockey', ''), entry.get('trainer', ''),
                    entry.get('draw', 0), entry.get('weight', ''),
                    entry.get('rating', 0), entry.get('win_odds', 0),
                    pos, is_winner, is_top3, entry.get('gear', ''), margin, hw,
                    last_6, age, running_pos, finish_time,
                    days_since, rat_change,
                ))

                # Upsert horse master record — derive stats from race_results
                # to avoid double-counting on re-ingestion
                # Upsert horse master record — stats are recomputed by rebuild_stats()
                # so no need for correlated subqueries here
                c.execute("""
                    INSERT INTO horses (name, trainer, rating, total_starts,
                        total_wins, total_places, last_seen_date)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        trainer=excluded.trainer,
                        rating=CASE WHEN excluded.rating > 0 THEN excluded.rating ELSE horses.rating END,
                        last_seen_date=MAX(horses.last_seen_date, excluded.last_seen_date)
                """, (
                    horse_name, entry.get('trainer', ''), entry.get('rating', 0),
                    is_winner, is_top3, race_date,
                ))

        # Ingest dividends if present on the race dict
        for race in race_data:
            race_date = race.get('date', '')
            race_no = race.get('race_no', 0)
            for d in race.get('dividends', []) or []:
                pool = d.get('pool', '').strip()
                combo = d.get('combination', '').strip()
                div = d.get('dividend', 0)
                if not pool or not combo or div <= 0:
                    continue
                c.execute("""
                    INSERT INTO race_dividends (race_date, race_number, pool, combination, dividend)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(race_date, race_number, pool, combination)
                    DO UPDATE SET dividend = excluded.dividend
                """, (race_date, race_no, pool, combo, float(div)))

        # Backfill days_since_last_run from race_results where missing
        # REPLACE converts YYYY/MM/DD to YYYY-MM-DD for julianday()
        c.execute("""
            UPDATE race_results SET days_since_last_run = (
                SELECT CAST(julianday(REPLACE(race_results.race_date,'/','-'))
                          - julianday(REPLACE(prev.race_date,'/','-')) AS INTEGER)
                FROM race_results AS prev
                WHERE prev.horse_name = race_results.horse_name
                  AND prev.race_date < race_results.race_date
                ORDER BY prev.race_date DESC LIMIT 1
            )
            WHERE (days_since_last_run IS NULL OR days_since_last_run = 0)
              AND EXISTS (
                SELECT 1 FROM race_results AS prev
                WHERE prev.horse_name = race_results.horse_name
                  AND prev.race_date < race_results.race_date
              )
        """)

        # Single commit for atomicity — upserts + backfill are all-or-nothing
        self.conn.commit()

        logger.info(f"Ingested {len(race_data)} races into horse database")

        # Rebuild stats from source of truth to avoid double-counting on re-ingestion
        self.rebuild_stats()

    # ------------------------------------------------------------------
    # Query methods — used by analyzer at prediction time
    # ------------------------------------------------------------------

    def get_quinella_dividend(self, race_date: str, race_number: int,
                              horse_a: int, horse_b: int) -> Optional[float]:
        """Look up actual HKJC quinella dividend for an unordered pair.

        Returns dividend per $1 stake, or None if not in DB (date not backfilled
        or pair didn't win). Caller can fall back to Harville estimate.
        """
        lo, hi = sorted([int(horse_a), int(horse_b)])
        combo = f"{lo},{hi}"
        row = self.conn.execute(
            "SELECT dividend FROM race_dividends "
            "WHERE race_date = ? AND race_number = ? AND pool = 'QUINELLA' AND combination = ?",
            (race_date, race_number, combo)
        ).fetchone()
        return float(row['dividend']) if row else None

    def get_race_dividends(self, race_date: str, race_number: int) -> List[Dict]:
        """All dividend pools for a race (WIN, PLACE, QIN, QPL, TRIO, etc.)."""
        c = self.conn.cursor()
        c.execute(
            "SELECT pool, combination, dividend FROM race_dividends "
            "WHERE race_date = ? AND race_number = ? ORDER BY pool, combination",
            (race_date, race_number)
        )
        return [dict(r) for r in c.fetchall()]

    def get_horse_record(self, horse_name: str) -> Optional[Dict]:
        """Get horse's complete racing record (computed from race_results)."""
        c = self.conn.cursor()
        c.execute("""
            SELECT
                horse_name as name,
                COUNT(*) as total_starts,
                SUM(is_winner) as total_wins,
                SUM(is_top3) as total_places,
                MAX(rating) as rating,
                MAX(trainer) as trainer,
                MAX(race_date) as last_seen_date
            FROM race_results
            WHERE horse_name = ?
        """, (horse_name,))
        row = c.fetchone()
        if not row or not row['total_starts']:
            return None
        return dict(row)

    def get_horse_history(self, horse_name: str, limit: int = 10) -> List[Dict]:
        """Get horse's past race results, most recent first."""
        c = self.conn.cursor()
        c.execute("""
            SELECT * FROM race_results
            WHERE horse_name = ?
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (horse_name, limit))
        return [dict(r) for r in c.fetchall()]

    @staticmethod
    def _normalize_venue(venue: str) -> str:
        """Normalize venue string to canonical form for exact matching."""
        v = (venue or '').lower()
        if 'sha' in v or v == 'st':
            return 'Sha Tin'
        if 'happy' in v or v == 'hv':
            return 'Happy Valley'
        return venue or ''

    def get_horse_at_venue_distance(self, horse_name: str, venue: str,
                                     distance: int, tolerance: int = 100) -> Dict:
        """Get horse's record at specific venue and distance (C&D form)."""
        c = self.conn.cursor()
        venue_norm = self._normalize_venue(venue)
        c.execute("""
            SELECT
                COUNT(*) as runs,
                SUM(is_winner) as wins,
                SUM(is_top3) as places,
                AVG(finish_position) as avg_pos
            FROM race_results
            WHERE horse_name = ?
              AND venue = ?
              AND ABS(distance - ?) <= ?
        """, (horse_name, venue_norm, distance, tolerance))
        row = c.fetchone()
        return dict(row) if row else {'runs': 0, 'wins': 0, 'places': 0, 'avg_pos': 0}

    def get_horse_cd_history(self, horse_name: str, venue: str,
                              distance: int, tolerance: int = 100) -> list:
        """Get horse's race-by-race C&D history with dates for winning trip detection."""
        c = self.conn.cursor()
        venue_norm = self._normalize_venue(venue)
        c.execute("""
            SELECT race_date, venue, distance, finish_position
            FROM race_results
            WHERE horse_name = ?
              AND venue = ?
              AND ABS(distance - ?) <= ?
            ORDER BY race_date
        """, (horse_name, venue_norm, distance, tolerance))
        return [{'date': r[0], 'venue': r[1], 'distance': r[2], 'position': r[3]}
                for r in c.fetchall()]

    def get_jockey_stats(self, jockey_name: str) -> Optional[Dict]:
        """Get jockey's season statistics."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM jockey_stats WHERE name = ?", (jockey_name,))
        row = c.fetchone()
        return dict(row) if row else None

    def get_jockey_recent_form(self, jockey_name: str, n_rides: int = 10) -> Dict:
        """Get jockey's recent form over last N rides.

        Hot streak detection — a jockey winning 4/10 recent rides is on fire,
        regardless of their season average. Bill Benter found this among the
        most predictive features.
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT is_winner, is_top3, finish_position
            FROM race_results
            WHERE jockey = ? AND finish_position > 0
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (jockey_name, n_rides))
        rows = c.fetchall()
        if not rows:
            return {'rides': 0, 'wins': 0, 'places': 0, 'win_rate': 0, 'place_rate': 0, 'hot': False}
        n = len(rows)
        wins = sum(r['is_winner'] for r in rows)
        places = sum(r['is_top3'] for r in rows)
        win_rate = wins / n if n else 0
        place_rate = places / n if n else 0
        return {
            'rides': n,
            'wins': wins,
            'places': places,
            'win_rate': round(win_rate, 3),
            'place_rate': round(place_rate, 3),
            'hot': win_rate >= 0.25,  # 25%+ recent win rate = hot streak
        }

    def get_trainer_recent_form(self, trainer_name: str, n_runners: int = 10) -> Dict:
        """Get trainer's recent form over last N runners."""
        c = self.conn.cursor()
        c.execute("""
            SELECT is_winner, is_top3, finish_position
            FROM race_results
            WHERE trainer = ? AND finish_position > 0
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (trainer_name, n_runners))
        rows = c.fetchall()
        if not rows:
            return {'runners': 0, 'wins': 0, 'places': 0, 'win_rate': 0, 'place_rate': 0, 'hot': False}
        n = len(rows)
        wins = sum(r['is_winner'] for r in rows)
        places = sum(r['is_top3'] for r in rows)
        win_rate = wins / n if n else 0
        place_rate = places / n if n else 0
        return {
            'runners': n,
            'wins': wins,
            'places': places,
            'win_rate': round(win_rate, 3),
            'place_rate': round(place_rate, 3),
            'hot': win_rate >= 0.20,  # 20%+ recent win rate = hot streak
        }

    def get_trainer_stats(self, trainer_name: str) -> Optional[Dict]:
        """Get trainer's season statistics."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM trainer_stats WHERE name = ?", (trainer_name,))
        row = c.fetchone()
        return dict(row) if row else None

    def get_jockey_at_venue(self, jockey_name: str, venue: str) -> Dict:
        """Get jockey's record at specific venue."""
        c = self.conn.cursor()
        venue_norm = self._normalize_venue(venue)
        c.execute("""
            SELECT
                COUNT(*) as rides,
                SUM(is_winner) as wins,
                SUM(is_top3) as places
            FROM race_results
            WHERE jockey = ? AND venue = ?
        """, (jockey_name, venue_norm))
        row = c.fetchone()
        return dict(row) if row else {'rides': 0, 'wins': 0, 'places': 0}

    def get_trainer_at_distance(self, trainer_name: str, distance: int,
                                 tolerance: int = 100) -> Dict:
        """Get trainer's record at specific distance."""
        c = self.conn.cursor()
        c.execute("""
            SELECT
                COUNT(*) as runners,
                SUM(is_winner) as wins,
                SUM(is_top3) as places
            FROM race_results
            WHERE trainer = ? AND ABS(distance - ?) <= ?
        """, (trainer_name, distance, tolerance))
        row = c.fetchone()
        return dict(row) if row else {'runners': 0, 'wins': 0, 'places': 0}

    def get_horse_going_record(self, horse_name: str, going: str) -> Dict:
        """Get horse's record on a specific going (track condition).

        Single query: exact match OR primary keyword match (e.g. 'Good' matches 'Good To Firm').
        """
        c = self.conn.cursor()
        going_upper = going.upper().strip() if going else ''
        if not going_upper:
            return {'runs': 0, 'wins': 0, 'places': 0, 'avg_pos': 0}

        primary = going_upper.split(' TO ')[0].split()[0] if going_upper else going_upper
        c.execute("""
            SELECT
                COUNT(*) as runs,
                SUM(is_winner) as wins,
                SUM(is_top3) as places,
                AVG(CASE WHEN finish_position > 0 THEN finish_position END) as avg_pos
            FROM race_results
            WHERE horse_name = ? AND (UPPER(going) = ? OR UPPER(going) LIKE ?)
        """, (horse_name, going_upper, f'{primary}%'))
        row = c.fetchone()
        return dict(row) if row else {'runs': 0, 'wins': 0, 'places': 0, 'avg_pos': 0}

    def get_horse_going_form(self, horse_name: str, going: str, before_date: str = '') -> dict:
        """Get horse's record on a specific going condition.
        Returns: {'runs': int, 'wins': int, 'places': int, 'avg_pos': float}
        """
        params = [horse_name, going]
        if before_date:
            params.append(before_date)

        query = """
            SELECT COUNT(*) as runs,
                   COALESCE(SUM(is_winner), 0) as wins,
                   COALESCE(SUM(is_top3), 0) as places,
                   COALESCE(AVG(CASE WHEN finish_position > 0 THEN finish_position END), 0) as avg_pos
            FROM race_results
            WHERE horse_name = ? AND going = ?
        """
        if before_date:
            query += ' AND race_date < ?'
        row = self.conn.execute(query, params).fetchone()

        return {
            'runs': row['runs'] or 0,
            'wins': row['wins'] or 0,
            'places': row['places'] or 0,
            'avg_pos': round(row['avg_pos'] or 0, 1),
        }

    def get_jockey_horse_stats(self, jockey: str, horse_name: str, before_date: str = '') -> dict:
        """Get jockey's record on a specific horse.
        Returns: {'rides': int, 'wins': int, 'places': int, 'avg_pos': float}
        """
        params = [jockey, horse_name]
        if before_date:
            params.append(before_date)

        query = """
            SELECT COUNT(*) as rides,
                   COALESCE(SUM(is_winner), 0) as wins,
                   COALESCE(SUM(is_top3), 0) as places,
                   COALESCE(AVG(CASE WHEN finish_position > 0 THEN finish_position END), 0) as avg_pos
            FROM race_results
            WHERE jockey = ? AND horse_name = ?
        """
        if before_date:
            query += ' AND race_date < ?'
        row = self.conn.execute(query, params).fetchone()

        return {
            'rides': row['rides'] or 0,
            'wins': row['wins'] or 0,
            'places': row['places'] or 0,
            'avg_pos': round(row['avg_pos'] or 0, 1),
        }

    def get_distance_best_time(self, horse_name: str, distance: int, before_date: str = '') -> str:
        """Get horse's personal best finish time at a given distance.
        Returns: time string like '1:09.2' or '' if no data.
        """
        params = [horse_name, distance]
        if before_date:
            params.append(before_date)

        query = """
            SELECT finish_time FROM race_results
            WHERE horse_name = ? AND distance = ?
            AND finish_time IS NOT NULL AND finish_time != ''
            AND finish_time LIKE '%:%'
        """
        if before_date:
            query += ' AND race_date < ?'
        query += """
            ORDER BY
              CAST(SUBSTR(finish_time, 1, INSTR(finish_time,':')-1) AS REAL)*60
              + CAST(SUBSTR(finish_time, INSTR(finish_time,':')+1) AS REAL)
            ASC LIMIT 1
        """
        row = self.conn.execute(query, params).fetchone()

        return row['finish_time'] if row else ''

    def get_distance_record(self, venue: str, distance: int) -> str:
        """Get track record (fastest time) at venue + distance.
        Returns: time string like '1:08.1' or '' if no data.
        """
        venue_norm = self._normalize_venue(venue)
        row = self.conn.execute("""
            SELECT finish_time FROM race_results
            WHERE venue = ? AND distance = ?
            AND finish_time IS NOT NULL AND finish_time != ''
            AND finish_time LIKE '%:%'
            ORDER BY
              CAST(SUBSTR(finish_time, 1, INSTR(finish_time,':')-1) AS REAL)*60
              + CAST(SUBSTR(finish_time, INSTR(finish_time,':')+1) AS REAL)
            ASC LIMIT 1
        """, (venue_norm, distance)).fetchone()

        return row['finish_time'] if row else ''

    def get_rolling_stats(self, horse_name: str, n_races: int = 5) -> Dict:
        """Get rolling performance stats over last N races.

        Returns avg position, win rate, place rate, and trend direction.
        This is one of the highest-impact features in ML racing models.
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT finish_position, is_winner, is_top3, margin, win_odds
            FROM race_results
            WHERE horse_name = ? AND finish_position > 0
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (horse_name, n_races))
        rows = c.fetchall()
        if not rows:
            return {'runs': 0, 'avg_pos': 0, 'win_rate': 0, 'place_rate': 0,
                    'avg_margin': 0, 'trend': 'unknown', 'recent_3_avg': 0}

        positions = [r['finish_position'] for r in rows]
        wins = sum(r['is_winner'] for r in rows)
        places = sum(r['is_top3'] for r in rows)
        margins = [r['margin'] for r in rows if r['margin'] and r['margin'] > 0]
        n = len(rows)

        avg_pos = sum(positions) / n
        recent_3 = positions[:3] if len(positions) >= 3 else positions
        recent_3_avg = sum(recent_3) / len(recent_3)

        # Trend: compare recent 3 vs older runs
        if len(positions) >= 4:
            older = positions[3:]
            older_avg = sum(older) / len(older)
            # Lower position = better, so improving means recent < older
            if recent_3_avg < older_avg - 1:
                trend = 'improving'
            elif recent_3_avg > older_avg + 1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'limited'

        return {
            'runs': n,
            'avg_pos': round(avg_pos, 1),
            'win_rate': round(wins / n, 3) if n else 0,
            'place_rate': round(places / n, 3) if n else 0,
            'avg_margin': round(sum(margins) / len(margins), 2) if margins else 0,
            'trend': trend,
            'recent_3_avg': round(recent_3_avg, 1),
        }

    def get_last_jockey(self, horse_name: str) -> Optional[str]:
        """Get the jockey who rode this horse in its most recent race.

        Used to detect jockey changes — a new jockey booking often signals
        the trainer is making a strategic move.
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT jockey FROM race_results
            WHERE horse_name = ? AND jockey != ''
            ORDER BY race_date DESC, race_number DESC
            LIMIT 1
        """, (horse_name,))
        row = c.fetchone()
        return row['jockey'] if row else None

    def get_beaten_margins(self, horse_name: str, n_races: int = 5) -> Dict:
        """Get beaten margin stats for recent races.

        Margins < 2.75 lengths behind winner = strong next-out signal.
        This is a key feature in professional handicapping.
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT finish_position, margin
            FROM race_results
            WHERE horse_name = ? AND finish_position > 0
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (horse_name, n_races))
        rows = c.fetchall()
        if not rows:
            return {'runs': 0, 'close_finishes': 0, 'avg_margin': 0, 'last_margin': 0}

        margins = [r['margin'] for r in rows if r['margin'] and r['margin'] > 0]
        # Close finishes: within 2.75 lengths of winner (non-winners)
        close = [m for m in margins if m <= 2.75]
        last_margin = margins[0] if margins else 0
        # Distinguish winner (margin=0, position=1) from missing margin data
        last_winner = rows[0]['finish_position'] == 1 if rows else False

        return {
            'runs': len(rows),
            'close_finishes': len(close),
            'avg_margin': round(sum(margins) / len(margins), 2) if margins else 0,
            'last_margin': round(last_margin, 2),
            'last_winner': last_winner,
        }

    def get_horse_days_since_last(self, horse_name: str,
                                   current_date: str) -> int:
        """Calculate days since horse's last race."""
        c = self.conn.cursor()
        c.execute("""
            SELECT race_date FROM race_results
            WHERE horse_name = ? AND race_date < ?
            ORDER BY race_date DESC LIMIT 1
        """, (horse_name, current_date))
        row = c.fetchone()
        if not row:
            return 0
        try:
            last = datetime.strptime(row['race_date'], '%Y/%m/%d')
            curr = datetime.strptime(current_date, '%Y/%m/%d')
            return (curr - last).days
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """Get database summary stats."""
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM horses")
        n_horses = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM race_results")
        n_results = c.fetchone()[0]
        c.execute("SELECT COUNT(DISTINCT race_date) FROM race_results")
        n_days = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM jockey_stats")
        n_jockeys = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM trainer_stats")
        n_trainers = c.fetchone()[0]
        c.execute("SELECT MIN(race_date), MAX(race_date) FROM race_results")
        dates = c.fetchone()
        return {
            'horses': n_horses,
            'race_results': n_results,
            'race_days': n_days,
            'jockeys': n_jockeys,
            'trainers': n_trainers,
            'date_range': f"{dates[0]} to {dates[1]}" if dates[0] else "No data",
        }

    def get_top_jockeys(self, min_rides: int = 5) -> List[Dict]:
        """Get top jockeys by win rate."""
        c = self.conn.cursor()
        c.execute("""
            SELECT name, rides, wins, places, win_rate, place_rate
            FROM jockey_stats
            WHERE rides >= ?
            ORDER BY win_rate DESC
        """, (min_rides,))
        return [dict(r) for r in c.fetchall()]

    def get_top_trainers(self, min_runners: int = 5) -> List[Dict]:
        """Get top trainers by win rate."""
        c = self.conn.cursor()
        c.execute("""
            SELECT name, runners, wins, places, win_rate, place_rate
            FROM trainer_stats
            WHERE runners >= ?
            ORDER BY win_rate DESC
        """, (min_runners,))
        return [dict(r) for r in c.fetchall()]

    @staticmethod
    def _current_season() -> str:
        """Derive HK racing season string (Jul-Jun). E.g. '2025-26' for Jul 2025 - Jun 2026."""
        now = datetime.now()
        if now.month >= 7:
            return f"{now.year}-{str(now.year + 1)[-2:]}"
        else:
            return f"{now.year - 1}-{str(now.year)[-2:]}"

    def rebuild_stats(self):
        """Rebuild jockey_stats and trainer_stats from race_results.

        Fixes drift caused by re-ingestion double-counting.  Safe to call
        at any time — it replaces the stats tables entirely.
        """
        c = self.conn.cursor()
        season = self._current_season()

        # Rebuild jockey_stats
        c.execute("DELETE FROM jockey_stats")
        c.execute("""
            INSERT INTO jockey_stats (name, season, rides, wins, places, win_rate, place_rate, last_updated)
            SELECT
                jockey,
                ?,
                COUNT(*) as rides,
                SUM(is_winner) as wins,
                SUM(is_top3) as places,
                CAST(SUM(is_winner) AS REAL) / COUNT(*) as win_rate,
                CAST(SUM(is_top3) AS REAL) / COUNT(*) as place_rate,
                MAX(race_date) as last_updated
            FROM race_results
            WHERE jockey != ''
            GROUP BY jockey
        """, (season,))

        # Rebuild trainer_stats
        c.execute("DELETE FROM trainer_stats")
        c.execute("""
            INSERT INTO trainer_stats (name, season, runners, wins, places, win_rate, place_rate, last_updated)
            SELECT
                trainer,
                ?,
                COUNT(*) as runners,
                SUM(is_winner) as wins,
                SUM(is_top3) as places,
                CAST(SUM(is_winner) AS REAL) / COUNT(*) as win_rate,
                CAST(SUM(is_top3) AS REAL) / COUNT(*) as place_rate,
                MAX(race_date) as last_updated
            FROM race_results
            WHERE trainer != ''
            GROUP BY trainer
        """, (season,))

        self.conn.commit()
        logger.info("Rebuilt jockey_stats and trainer_stats from race_results")

    # ------------------------------------------------------------------
    # Prediction tracking
    # ------------------------------------------------------------------

    def log_predictions(self, race_date: str, races: list, model_version: str = ''):
        """Store predictions for all horses in all races for a given date.

        races: list of race dicts from generate_predictions.py JSON output.
        Upserts — safe to call multiple times for the same date.
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c = self.conn.cursor()

        for race in races:
            race_no = race.get('race_number', 0)
            venue = race.get('venue', '')
            distance = race.get('distance', 0)
            class_type = race.get('class_type', '')
            going = race.get('going', '')
            horses = race.get('horses', [])
            field_size = len(horses)

            for h in horses:
                c.execute("""
                    INSERT INTO prediction_log (
                        race_date, race_number, venue, distance, class_type, going,
                        field_size, horse_name, horse_number, predicted_rank,
                        rule_score, ml_prob, calibrated_prob, expected_return,
                        bet_recommendation, confidence, win_odds, model_version,
                        predicted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(race_date, race_number, horse_name)
                    DO UPDATE SET
                        predicted_rank=excluded.predicted_rank,
                        rule_score=excluded.rule_score,
                        ml_prob=excluded.ml_prob,
                        calibrated_prob=excluded.calibrated_prob,
                        expected_return=excluded.expected_return,
                        bet_recommendation=excluded.bet_recommendation,
                        confidence=excluded.confidence,
                        win_odds=excluded.win_odds,
                        model_version=excluded.model_version,
                        predicted_at=excluded.predicted_at
                """, (
                    race_date, race_no, venue, distance, class_type, going,
                    field_size, h.get('name', ''), h.get('number', 0),
                    h.get('rank', 0), h.get('rule_score', 0),
                    h.get('ml_prob', 0), h.get('calibrated_prob', 0),
                    h.get('expected_return', 0), h.get('bet_recommendation', 'SKIP'),
                    h.get('confidence', ''), h.get('win_odds', 0),
                    model_version, now,
                ))

        self.conn.commit()
        logger.info(f"Logged predictions for {len(races)} races on {race_date}")

    def fill_prediction_results(self, race_date: str, race_number: int,
                                results: list):
        """Fill actual results into prediction_log for a single race.

        results: list of dicts with 'number', 'position', 'name'.
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c = self.conn.cursor()

        for r in results:
            pos = r.get('position', 0) or 0
            name = r.get('name', '')
            if not name or not pos:
                continue

            won = 1 if pos == 1 else 0
            top3 = 1 if 1 <= pos <= 3 else 0

            # Compute profit/loss per $10 unit
            row = c.execute("""
                SELECT bet_recommendation, win_odds FROM prediction_log
                WHERE race_date = ? AND race_number = ? AND horse_name = ?
            """, (race_date, race_number, name)).fetchone()

            profit = 0.0
            if row:
                bet = row[0] or 'SKIP'
                odds = row[1] or 0
                if bet in ('WIN', 'BANKER WIN'):
                    profit = (odds * 10 - 10) if won else -10.0
                elif bet == 'PLACE':
                    profit = (max(odds / 3, 1.2) * 10 - 10) if top3 else -10.0
                elif bet in ('QIN', 'QNL PL', 'TRIO'):
                    # Exotic bets: mark as -10 loss per horse; race-level query handles wins
                    profit = -10.0 if not top3 else 0.0
                # SKIP = no bet = no profit/loss

            c.execute("""
                UPDATE prediction_log SET
                    actual_position = ?,
                    actual_won = ?,
                    actual_top3 = ?,
                    profit_loss = ?,
                    results_filled = 1,
                    results_filled_at = ?
                WHERE race_date = ? AND race_number = ? AND horse_name = ?
            """, (pos, won, top3, round(profit, 1), now,
                  race_date, race_number, name))

        self.conn.commit()

    def get_accuracy_summary(self, days: int = 30) -> Dict:
        """Get prediction accuracy summary over the last N days."""
        c = self.conn.cursor()
        cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).strftime('%Y/%m/%d')

        # Only races where results have been filled
        rows = c.execute("""
            SELECT race_date, race_number, horse_name, predicted_rank,
                   bet_recommendation, calibrated_prob, win_odds,
                   actual_position, actual_won, actual_top3, profit_loss, confidence
            FROM prediction_log
            WHERE results_filled = 1 AND race_date >= ?
            ORDER BY race_date, race_number, predicted_rank
        """, (cutoff,)).fetchall()

        if not rows:
            return {'message': 'No results data available'}

        # Group by race
        races = {}
        for r in rows:
            key = f"{r[0]}-{r[1]}"
            if key not in races:
                races[key] = []
            races[key].append(r)

        total_races = len(races)
        rank1_wins = 0
        top3_overlaps = 0
        winner_ranks = []
        bets_placed = 0
        bets_won = 0
        total_profit = 0.0
        banker_count = 0
        banker_wins = 0
        by_confidence = {}

        for race_key, horses in races.items():
            horses.sort(key=lambda x: x[3])  # sort by predicted_rank

            # Rank 1 win check
            if horses and horses[0][8]:  # actual_won
                rank1_wins += 1

            # Top-3 overlap
            our_top3 = {h[2] for h in horses[:3]}  # horse_name
            actual_top3_set = {h[2] for h in horses if h[9]}  # actual_top3
            top3_overlaps += len(our_top3 & actual_top3_set)

            # Where did winner rank?
            for h in horses:
                if h[8]:  # actual_won
                    winner_ranks.append(h[3])  # predicted_rank
                    break

            # Bet P&L
            for h in horses:
                bet = h[4]
                if bet and bet != 'SKIP':
                    bets_placed += 1
                    total_profit += h[10]  # profit_loss
                    if h[8] and bet in ('WIN', 'BANKER WIN'):
                        bets_won += 1
                    elif h[9] and bet == 'PLACE':
                        bets_won += 1

                    if 'BANKER' in bet:
                        banker_count += 1
                        if h[8]:
                            banker_wins += 1

                # By confidence
                conf = h[11] or 'UNKNOWN'
                if conf not in by_confidence:
                    by_confidence[conf] = {'total': 0, 'wins': 0}
                if h[3] == 1:  # rank 1 only
                    by_confidence[conf]['total'] += 1
                    if h[8]:
                        by_confidence[conf]['wins'] += 1

        return {
            'period_days': days,
            'total_races': total_races,
            'rank1_win_rate': round(rank1_wins / total_races * 100, 1) if total_races else 0,
            'rank1_wins': rank1_wins,
            'top3_overlap_rate': round(top3_overlaps / (total_races * 3) * 100, 1) if total_races else 0,
            'avg_winner_rank': round(sum(winner_ranks) / len(winner_ranks), 1) if winner_ranks else 0,
            'bets_placed': bets_placed,
            'bets_won': bets_won,
            'bet_hit_rate': round(bets_won / bets_placed * 100, 1) if bets_placed else 0,
            'total_profit': round(total_profit, 1),
            'roi_percent': round(total_profit / (bets_placed * 10) * 100, 1) if bets_placed else 0,
            'banker_count': banker_count,
            'banker_wins': banker_wins,
            'banker_hit_rate': round(banker_wins / banker_count * 100, 1) if banker_count else 0,
            'by_confidence': by_confidence,
        }

    def get_accuracy_by_class(self, days: int = 30) -> list:
        """Get accuracy breakdown by class type."""
        c = self.conn.cursor()
        cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).strftime('%Y/%m/%d')

        rows = c.execute("""
            SELECT class_type,
                   COUNT(DISTINCT race_date || '-' || race_number) as races,
                   SUM(CASE WHEN predicted_rank = 1 AND actual_won = 1 THEN 1 ELSE 0 END) as rank1_wins,
                   SUM(CASE WHEN predicted_rank <= 3 AND actual_top3 = 1 THEN 1 ELSE 0 END) as top3_hits,
                   SUM(CASE WHEN bet_recommendation != 'SKIP' THEN profit_loss ELSE 0 END) as profit
            FROM prediction_log
            WHERE results_filled = 1 AND race_date >= ?
            GROUP BY class_type
            ORDER BY races DESC
        """, (cutoff,)).fetchall()

        result = []
        for r in rows:
            races = r[1] or 1
            result.append({
                'class': r[0] or 'Unknown',
                'races': races,
                'rank1_win_rate': round(r[2] / races * 100, 1),
                'top3_hit_rate': round(r[3] / (races * 3) * 100, 1),
                'profit': round(r[4] or 0, 1),
            })
        return result

    def get_bet_type_roi(self, days: int = 30) -> list:
        """Get ROI breakdown by bet recommendation type."""
        c = self.conn.cursor()
        cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).strftime('%Y/%m/%d')

        rows = c.execute("""
            SELECT bet_recommendation,
                   COUNT(*) as bets,
                   SUM(CASE WHEN actual_won = 1 OR actual_top3 = 1 THEN 1 ELSE 0 END) as hits,
                   SUM(profit_loss) as profit
            FROM prediction_log
            WHERE results_filled = 1 AND bet_recommendation != 'SKIP'
                  AND race_date >= ?
            GROUP BY bet_recommendation
            ORDER BY profit DESC
        """, (cutoff,)).fetchall()

        result = []
        for r in rows:
            bets = r[1] or 1
            result.append({
                'bet_type': r[0],
                'count': bets,
                'hits': r[2],
                'hit_rate': round(r[2] / bets * 100, 1),
                'profit': round(r[3] or 0, 1),
                'roi': round((r[3] or 0) / (bets * 10) * 100, 1),
            })
        return result

    def close(self):
        self.conn.close()


# ------------------------------------------------------------------
# CLI: ingest from training data file
# ------------------------------------------------------------------

def ingest_from_training_data():
    """Load training_data/historical_races.json into the horse database."""
    training_file = Path(__file__).parent.parent / 'training_data' / 'historical_races.json'
    if not training_file.exists():
        print("No training data found")
        return

    with open(training_file) as f:
        races = json.load(f)

    db = HorseDatabase()
    db.ingest_race_day(races)

    summary = db.get_summary()
    print(f"Database updated:")
    print(f"  Horses: {summary['horses']}")
    print(f"  Race results: {summary['race_results']}")
    print(f"  Race days: {summary['race_days']}")
    print(f"  Jockeys: {summary['jockeys']}")
    print(f"  Trainers: {summary['trainers']}")
    print(f"  Date range: {summary['date_range']}")

    print("\nTop Jockeys:")
    for j in db.get_top_jockeys(min_rides=3):
        print(f"  {j['name']:<20} {j['rides']}R {j['wins']}W ({j['win_rate']:.0%})")

    print("\nTop Trainers:")
    for t in db.get_top_trainers(min_runners=3):
        print(f"  {t['name']:<20} {t['runners']}R {t['wins']}W ({t['win_rate']:.0%})")

    db.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ingest_from_training_data()
