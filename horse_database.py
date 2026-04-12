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

DB_DIR = Path(__file__).parent / 'tracking_data'
DB_DIR.mkdir(exist_ok=True)
DB_FILE = DB_DIR / 'horse_history.db'


class HorseDatabase:
    """Persistent local database for horse racing history."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_FILE)
        self.conn = sqlite3.connect(self.db_path)
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

        # Add margin column if upgrading from older schema
        try:
            c.execute("ALTER TABLE race_results ADD COLUMN margin REAL DEFAULT 0")
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

        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_horse ON race_results(horse_name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_date ON race_results(race_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_venue_dist ON race_results(venue, distance)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_jockey ON race_results(jockey)")

        self.conn.commit()

    # ------------------------------------------------------------------
    # Ingest race results
    # ------------------------------------------------------------------

    def ingest_race_day(self, race_data: List[Dict]):
        """Ingest a full race day of results into the database.

        race_data: list of race dicts with 'entries' containing horse results.
        This is the same format as training_data/historical_races.json.
        """
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
                c.execute("""
                    INSERT INTO race_results (
                        race_date, race_number, venue, distance, class_type,
                        going, course, horse_name, horse_number, jockey, trainer,
                        draw, weight, rating, win_odds, finish_position,
                        is_winner, is_top3, gear, margin
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(race_date, race_number, horse_name)
                    DO UPDATE SET
                        finish_position=excluded.finish_position,
                        is_winner=excluded.is_winner,
                        is_top3=excluded.is_top3,
                        win_odds=CASE WHEN excluded.win_odds > 0 THEN excluded.win_odds ELSE race_results.win_odds END,
                        margin=CASE WHEN excluded.margin > 0 THEN excluded.margin ELSE race_results.margin END
                """, (
                    race_date, race_no, venue, distance, class_type,
                    going, course, horse_name, entry.get('number', 0),
                    entry.get('jockey', ''), entry.get('trainer', ''),
                    entry.get('draw', 0), entry.get('weight', ''),
                    entry.get('rating', 0), entry.get('win_odds', 0),
                    pos, is_winner, is_top3, entry.get('gear', ''), margin,
                ))

                # Upsert horse master record
                c.execute("""
                    INSERT INTO horses (name, trainer, rating, total_starts,
                        total_wins, total_places, last_seen_date)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        trainer=excluded.trainer,
                        rating=CASE WHEN excluded.rating > 0 THEN excluded.rating ELSE horses.rating END,
                        total_starts=horses.total_starts + 1,
                        total_wins=horses.total_wins + excluded.total_wins,
                        total_places=horses.total_places + excluded.total_places,
                        last_seen_date=excluded.last_seen_date
                """, (
                    horse_name, entry.get('trainer', ''), entry.get('rating', 0),
                    is_winner, is_top3, race_date,
                ))

                # Update jockey stats
                jockey = entry.get('jockey', '')
                if jockey:
                    c.execute("""
                        INSERT INTO jockey_stats (name, season, rides, wins, places, last_updated)
                        VALUES (?, ?, 1, ?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            rides=jockey_stats.rides + 1,
                            wins=jockey_stats.wins + ?,
                            places=jockey_stats.places + ?,
                            win_rate=CAST(jockey_stats.wins + ? AS REAL) / (jockey_stats.rides + 1),
                            place_rate=CAST(jockey_stats.places + ? AS REAL) / (jockey_stats.rides + 1),
                            last_updated=?
                    """, (
                        jockey, '2025-26', is_winner, is_top3, now,
                        is_winner, is_top3, is_winner, is_top3, now,
                    ))

                # Update trainer stats
                trainer = entry.get('trainer', '')
                if trainer:
                    c.execute("""
                        INSERT INTO trainer_stats (name, season, runners, wins, places, last_updated)
                        VALUES (?, ?, 1, ?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            runners=trainer_stats.runners + 1,
                            wins=trainer_stats.wins + ?,
                            places=trainer_stats.places + ?,
                            win_rate=CAST(trainer_stats.wins + ? AS REAL) / (trainer_stats.runners + 1),
                            place_rate=CAST(trainer_stats.places + ? AS REAL) / (trainer_stats.runners + 1),
                            last_updated=?
                    """, (
                        trainer, '2025-26', is_winner, is_top3, now,
                        is_winner, is_top3, is_winner, is_top3, now,
                    ))

        self.conn.commit()
        logger.info(f"Ingested {len(race_data)} races into horse database")

    # ------------------------------------------------------------------
    # Query methods — used by analyzer at prediction time
    # ------------------------------------------------------------------

    def get_horse_record(self, horse_name: str) -> Optional[Dict]:
        """Get horse's complete racing record."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM horses WHERE name = ?", (horse_name,))
        row = c.fetchone()
        if not row:
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

    def get_horse_at_venue_distance(self, horse_name: str, venue: str,
                                     distance: int, tolerance: int = 100) -> Dict:
        """Get horse's record at specific venue and distance (C&D form)."""
        c = self.conn.cursor()
        # Normalize venue
        venue_pattern = '%' + ('Sha Tin' if 'sha' in venue.lower() or 'ST' in venue else 'Happy Valley') + '%'
        c.execute("""
            SELECT
                COUNT(*) as runs,
                SUM(is_winner) as wins,
                SUM(is_top3) as places,
                AVG(finish_position) as avg_pos
            FROM race_results
            WHERE horse_name = ?
              AND venue LIKE ?
              AND ABS(distance - ?) <= ?
        """, (horse_name, venue_pattern, distance, tolerance))
        row = c.fetchone()
        return dict(row) if row else {'runs': 0, 'wins': 0, 'places': 0, 'avg_pos': 0}

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
        venue_pattern = '%' + ('Sha Tin' if 'sha' in venue.lower() or 'ST' in venue else 'Happy Valley') + '%'
        c.execute("""
            SELECT
                COUNT(*) as rides,
                SUM(is_winner) as wins,
                SUM(is_top3) as places
            FROM race_results
            WHERE jockey = ? AND venue LIKE ?
        """, (jockey_name, venue_pattern))
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

        Going values: GOOD, GOOD TO FIRM, GOOD TO YIELDING, YIELDING, SOFT, etc.
        Uses LIKE matching so 'GOOD' matches 'GOOD' and 'GOOD TO FIRM'.
        """
        c = self.conn.cursor()
        # Normalize: extract primary going keyword
        going_upper = going.upper().strip() if going else ''
        if not going_upper:
            return {'runs': 0, 'wins': 0, 'places': 0, 'avg_pos': 0}

        # Exact match first, then fallback to partial
        c.execute("""
            SELECT
                COUNT(*) as runs,
                SUM(is_winner) as wins,
                SUM(is_top3) as places,
                AVG(CASE WHEN finish_position > 0 THEN finish_position END) as avg_pos
            FROM race_results
            WHERE horse_name = ? AND UPPER(going) = ?
        """, (horse_name, going_upper))
        row = c.fetchone()
        result = dict(row) if row else {'runs': 0, 'wins': 0, 'places': 0, 'avg_pos': 0}

        # If no exact match, try partial (e.g. 'GOOD' matches 'GOOD TO FIRM')
        if result['runs'] == 0:
            primary = going_upper.split(' TO ')[0].split()[0] if going_upper else ''
            if primary:
                c.execute("""
                    SELECT
                        COUNT(*) as runs,
                        SUM(is_winner) as wins,
                        SUM(is_top3) as places,
                        AVG(CASE WHEN finish_position > 0 THEN finish_position END) as avg_pos
                    FROM race_results
                    WHERE horse_name = ? AND UPPER(going) LIKE ?
                """, (horse_name, f'%{primary}%'))
                row = c.fetchone()
                result = dict(row) if row else {'runs': 0, 'wins': 0, 'places': 0, 'avg_pos': 0}

        return result

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

        return {
            'runs': len(rows),
            'close_finishes': len(close),
            'avg_margin': round(sum(margins) / len(margins), 2) if margins else 0,
            'last_margin': round(last_margin, 2),
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

    def close(self):
        self.conn.close()


# ------------------------------------------------------------------
# CLI: ingest from training data file
# ------------------------------------------------------------------

def ingest_from_training_data():
    """Load training_data/historical_races.json into the horse database."""
    training_file = Path(__file__).parent / 'training_data' / 'historical_races.json'
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
