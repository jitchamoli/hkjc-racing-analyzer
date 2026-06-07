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

from betting.dividends import hkjc_declared_to_unit

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / 'tracking_data'
DB_DIR.mkdir(exist_ok=True)
DB_FILE = DB_DIR / 'horse_history.db'


def apply_perf_pragmas(conn: sqlite3.Connection) -> None:
    """Connection-local PRAGMAs for read-heavy workloads.

    These have to be set per-connection — they don't persist except for
    journal_mode (already on as WAL). Measured speedup: backtest +1.4×
    on top of the composite indexes (14.4s → 10.4s).

    Safe everywhere — `temp_store=MEMORY` only affects temp tables/sorts
    (we don't materialize large ones); `mmap_size` is advisory and SQLite
    falls back to read() if the mapping fails.
    """
    try:
        conn.execute("PRAGMA cache_size = -20000")    # 20MB page cache (was 2MB)
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped IO
        conn.execute("PRAGMA temp_store = MEMORY")    # temp tables/sorts in RAM
    except sqlite3.DatabaseError:
        # Older SQLite or restricted environment — fall through silently;
        # the connection still works at default settings.
        pass


class HorseDatabase:
    """Persistent local database for horse racing history."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_FILE)
        self.conn = sqlite3.connect(self.db_path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        apply_perf_pragmas(self.conn)
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
            # Path B: space-separated per-section times from the HKJC
            # sectional page, e.g. "27.84 23.69 24.22 24.10". Last token
            # is the last-400m time. Mirrors the running_position pattern.
            "sectional_times TEXT DEFAULT ''",
            # Race-day weather text from the HKJC meeting page (e.g.
            # "Fine", "Cloudy", "Rain"). Captured to start collecting a year
            # of data — not yet read by any feature.
            "weather TEXT DEFAULT ''",
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
                conviction_tier TEXT DEFAULT NULL,
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

        # Add columns if upgrading prediction_log from older schema
        try:
            c.execute("ALTER TABLE prediction_log ADD COLUMN conviction_tier TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # column already exists

        # Race-level frozen bet ticket — the actual staked bets (post-gate).
        # Added 2026-05-24 (Finding #2) so validate_hk can score what was
        # actually recommended/staked, not synthetic strategies built from
        # predicted_rank that ignore tier/gate filters. ticket_json is the
        # bet_ticket dict from dashboard/generator.build_bet_ticket.
        c.execute("""
            CREATE TABLE IF NOT EXISTS race_bet_ticket (
                race_date TEXT NOT NULL,
                race_number INTEGER NOT NULL,
                ticket_json TEXT NOT NULL,
                logged_at TEXT,
                PRIMARY KEY (race_date, race_number)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_btkt_date ON race_bet_ticket(race_date)")

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

        # Barrier-trial results (scraped from Trackwork/BTResult.aspx). One row
        # per (trial day, heat, horse). Source for candidate trial-form features
        # — query with trial_date < race_date for leak-free priors. finish_position
        # is the 1-based finishing order within the heat; margin is LBW.
        c.execute("""
            CREATE TABLE IF NOT EXISTS barrier_trials (
                trial_date TEXT NOT NULL,
                venue TEXT,
                heat INTEGER NOT NULL,
                horse_name TEXT NOT NULL,
                finish_position INTEGER,
                margin TEXT,
                finish_time TEXT,
                running_position TEXT,
                jockey TEXT,
                trainer TEXT,
                draw TEXT,
                gear TEXT,
                scraped_at TEXT,
                PRIMARY KEY (trial_date, heat, horse_name)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_trials_horse_date ON barrier_trials(horse_name, trial_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trials_date ON barrier_trials(trial_date)")

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
        # Composite indexes for the hot feature-extraction paths used by
        # backtest + live scoring. Without these, queries like
        #   WHERE jockey = ? AND race_date < ? ORDER BY race_date DESC LIMIT N
        # only use the single-column index then sort in memory. With them,
        # full-holdout backtest drops from ~42s to ~14s (3× speedup),
        # output bit-identical.
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_jockey_date ON race_results(jockey, race_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_trainer_date ON race_results(trainer, race_date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_jt_pair_date ON race_results(jockey, trainer, race_date)")

        self.conn.commit()

    # ------------------------------------------------------------------
    # Ingest barrier trials
    # ------------------------------------------------------------------

    def ingest_barrier_trials(self, trials: List[Dict], venue: str = "") -> int:
        """Upsert parsed barrier-trial rows into barrier_trials. Idempotent
        (INSERT OR REPLACE on (trial_date, heat, horse_name)). Returns rows written.

        `trials` are dicts from HKJCParser.parse_barrier_trials (must carry
        'date' = trial_date in YYYY/MM/DD)."""
        if not trials:
            return 0
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        n = 0
        for t in trials:
            if not t.get("date") or not t.get("horse_name"):
                continue
            self.conn.execute(
                """INSERT OR REPLACE INTO barrier_trials (
                    trial_date, venue, heat, horse_name, finish_position, margin,
                    finish_time, running_position, jockey, trainer, draw, gear, scraped_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (t.get("date"), venue, t.get("heat"), t.get("horse_name"),
                 t.get("position"), t.get("margin"), t.get("time"),
                 t.get("running_position"), t.get("jockey"), t.get("trainer"),
                 t.get("draw"), t.get("gear"), now),
            )
            n += 1
        self.conn.commit()
        return n

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

        # Track the date range we are touching so the days_since_last_run
        # backfill below can be scoped instead of scanning all 368K rows.
        ingested_dates = sorted({r.get('date', '') for r in race_data if r.get('date')})

        for race in race_data:
            race_date = race.get('date', '')
            race_no = race.get('race_no', 0)
            venue = race.get('venue', '')
            distance = int(race.get('distance', 0) or 0)
            class_type = race.get('class_type', '')
            going = race.get('going', '')
            course = race.get('course', '')

            # Dropped-finisher guard: a complete result set has contiguous
            # placings, so max(finishing position) must equal the number of
            # finishers. max > count means a finisher is missing — the exact
            # signature of the dead-heat parser drop fixed 2026-06-05 (HKJC's
            # "9 DH" cell was rejected, silently losing both tied horses).
            # Warn loudly here so any future parse drop is caught at ingest,
            # whatever the source (scraper OR _build_races_from_json). A
            # correctly-kept dead heat does NOT trip this (1,2,2,4 → max==count).
            finish_positions = [e.get('actual_position', 0) for e in race.get('entries', [])
                                if (e.get('actual_position') or 0) >= 1]
            if finish_positions and max(finish_positions) > len(finish_positions):
                missing = max(finish_positions) - len(finish_positions)
                logger.warning(
                    f"DROPPED FINISHER(S) in {race_date} R{race_no}: "
                    f"{len(finish_positions)} rows but top placing is {max(finish_positions)} "
                    f"({missing} missing — likely an unparsed dead heat). "
                    f"Re-scrape this race; check the placing parser for new markers."
                )

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
                sectional_times = entry.get('sectional_times', '') or ''
                days_since = entry.get('days_since_last_run', 0) or 0
                rat_change = entry.get('rating_change', 0) or 0
                c.execute("""
                    INSERT INTO race_results (
                        race_date, race_number, venue, distance, class_type,
                        going, course, horse_name, horse_number, jockey, trainer,
                        draw, weight, rating, win_odds, finish_position,
                        is_winner, is_top3, gear, margin, horse_weight,
                        last_6, age, running_position, finish_time,
                        days_since_last_run, rating_change, sectional_times
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(race_date, race_number, horse_name)
                    DO UPDATE SET
                        finish_position=CASE WHEN excluded.finish_position > 0 THEN excluded.finish_position ELSE race_results.finish_position END,
                        is_winner=CASE WHEN excluded.finish_position > 0 THEN excluded.is_winner ELSE race_results.is_winner END,
                        is_top3=CASE WHEN excluded.finish_position > 0 THEN excluded.is_top3 ELSE race_results.is_top3 END,
                        win_odds=CASE WHEN excluded.win_odds > 0 THEN excluded.win_odds ELSE race_results.win_odds END,
                        margin=CASE WHEN excluded.finish_position > 0 THEN excluded.margin ELSE race_results.margin END,
                        horse_weight=CASE WHEN excluded.horse_weight > 0 THEN excluded.horse_weight ELSE race_results.horse_weight END,
                        last_6=CASE WHEN excluded.last_6 != '' THEN excluded.last_6 ELSE race_results.last_6 END,
                        age=CASE WHEN excluded.age != '' THEN excluded.age ELSE race_results.age END,
                        rating=CASE WHEN excluded.rating > 0 THEN excluded.rating ELSE race_results.rating END,
                        gear=CASE WHEN excluded.gear != '' THEN excluded.gear ELSE race_results.gear END,
                        class_type=CASE WHEN excluded.class_type != '' THEN excluded.class_type ELSE race_results.class_type END,
                        course=CASE WHEN excluded.course != '' THEN excluded.course ELSE race_results.course END,
                        venue=CASE WHEN excluded.venue != '' THEN excluded.venue ELSE race_results.venue END,
                        distance=CASE WHEN excluded.distance > 0 THEN excluded.distance ELSE race_results.distance END,
                        going=CASE WHEN excluded.going != '' THEN excluded.going ELSE race_results.going END,
                        jockey=CASE WHEN excluded.jockey != '' THEN excluded.jockey ELSE race_results.jockey END,
                        trainer=CASE WHEN excluded.trainer != '' THEN excluded.trainer ELSE race_results.trainer END,
                        draw=CASE WHEN excluded.draw > 0 THEN excluded.draw ELSE race_results.draw END,
                        weight=CASE WHEN excluded.weight != '' AND excluded.weight != '0' THEN excluded.weight ELSE race_results.weight END,
                        horse_number=CASE WHEN excluded.horse_number > 0 THEN excluded.horse_number ELSE race_results.horse_number END,
                        running_position=CASE WHEN excluded.running_position != '' THEN excluded.running_position ELSE race_results.running_position END,
                        finish_time=CASE WHEN excluded.finish_time != '' THEN excluded.finish_time ELSE race_results.finish_time END,
                        days_since_last_run=CASE WHEN excluded.days_since_last_run > 0 THEN excluded.days_since_last_run ELSE race_results.days_since_last_run END,
                        rating_change=CASE WHEN excluded.rating_change != 0 THEN excluded.rating_change ELSE race_results.rating_change END,
                        sectional_times=CASE WHEN excluded.sectional_times != '' THEN excluded.sectional_times ELSE race_results.sectional_times END
                """, (
                    race_date, race_no, venue, distance, class_type,
                    going, course, horse_name, entry.get('number', 0),
                    entry.get('jockey', ''), entry.get('trainer', ''),
                    entry.get('draw', 0), entry.get('weight', ''),
                    entry.get('rating', 0), entry.get('win_odds', 0),
                    pos, is_winner, is_top3, entry.get('gear', ''), margin, hw,
                    last_6, age, running_pos, finish_time,
                    days_since, rat_change, sectional_times,
                ))

                # Upsert horse master record. total_starts/wins/places are
                # recomputed from race_results below (see _refresh_horse_aggregates)
                # so the values inserted here are placeholders that will be
                # immediately overwritten — they only matter for the very
                # first-ever insert of a brand-new horse before the refresh runs.
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

        # Backfill days_since_last_run from race_results where missing.
        # REPLACE converts YYYY/MM/DD to YYYY-MM-DD for julianday().
        # Scope to the dates we just ingested — without this WHERE clause
        # the correlated subquery scans all 368K rows on every ingestion
        # (renavon-era DB) and holds a write lock for minutes.
        if ingested_dates:
            min_date = ingested_dates[0]
            c.execute("""
                UPDATE race_results SET days_since_last_run = (
                    SELECT CAST(julianday(REPLACE(race_results.race_date,'/','-'))
                              - julianday(REPLACE(prev.race_date,'/','-')) AS INTEGER)
                    FROM race_results AS prev
                    WHERE prev.horse_name = race_results.horse_name
                      AND prev.race_date < race_results.race_date
                    ORDER BY prev.race_date DESC LIMIT 1
                )
                WHERE race_date >= ?
                  AND (days_since_last_run IS NULL OR days_since_last_run = 0)
                  AND EXISTS (
                    SELECT 1 FROM race_results AS prev
                    WHERE prev.horse_name = race_results.horse_name
                      AND prev.race_date < race_results.race_date
                  )
            """, (min_date,))

        # Refresh horses.total_* aggregates from race_results so the
        # master-table counters don't permanently lag (the columns existed
        # but were only set on first insert before this fix). Scoped to the
        # horses we just touched.
        touched_horses = sorted({
            entry.get('name', '')
            for r in race_data
            for entry in r.get('entries', [])
            if entry.get('name')
        })
        if touched_horses:
            placeholders = ','.join('?' for _ in touched_horses)
            c.execute(f"""
                UPDATE horses SET
                    total_starts = COALESCE((SELECT COUNT(*) FROM race_results r
                                             WHERE r.horse_name = horses.name), 0),
                    total_wins   = COALESCE((SELECT SUM(is_winner) FROM race_results r
                                             WHERE r.horse_name = horses.name), 0),
                    total_places = COALESCE((SELECT SUM(is_top3) FROM race_results r
                                             WHERE r.horse_name = horses.name), 0)
                WHERE name IN ({placeholders})
            """, touched_horses)

        # Single commit for atomicity — upserts + backfill are all-or-nothing
        self.conn.commit()

        logger.info(f"Ingested {len(race_data)} races into horse database")

        # Rebuild stats from source of truth to avoid double-counting on
        # re-ingestion. Swallow failures here — race_results / horses are
        # already committed and the operator should NOT see the ingest as
        # failed if only the derived stats tables fell over (e.g. missing
        # renavon ranking tables). Stats can be rebuilt later via
        # `python3 -m scripts.scrape_history --rankings`.
        try:
            self.rebuild_stats()
        except Exception:
            logger.error("rebuild_stats failed after ingest — race_results "
                         "already committed; jockey/trainer stats may be stale. "
                         "Re-run `scripts.scrape_history --rankings` to refresh.",
                         exc_info=True)

    # ------------------------------------------------------------------
    # Query methods — used by analyzer at prediction time
    # ------------------------------------------------------------------

    def get_quinella_dividend(self, race_date: str, race_number: int,
                              horse_a: int, horse_b: int) -> Optional[float]:
        """Look up actual HKJC quinella dividend for an unordered pair.

        Returns the raw HKJC dividend, declared PER HK$10 stake (the figure
        punters see), or None if not in DB (date not backfilled or pair didn't
        win). NOTE: this is per-$10, NOT per-$1 — used for DISPLAY. The backtest
        P&L path uses its own get_real_quinella_dividend (per-$1) for stake math.
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

    def get_horse_record(self, horse_name: str, before_date: str = '') -> Optional[Dict]:
        """Get horse's complete racing record (computed from race_results).

        before_date: backtest-leak guard — without it total_starts/wins/places
        include the current race and later ones. Live passes none. See
        get_horse_history.
        """
        c = self.conn.cursor()
        df = ' AND race_date < ?' if before_date else ''
        c.execute(f"""
            SELECT
                horse_name as name,
                COUNT(*) as total_starts,
                SUM(is_winner) as total_wins,
                SUM(is_top3) as total_places,
                MAX(rating) as rating,
                MAX(trainer) as trainer,
                MAX(race_date) as last_seen_date
            FROM race_results
            WHERE horse_name = ?{df}
        """, (horse_name, before_date) if before_date else (horse_name,))
        row = c.fetchone()
        if not row or not row['total_starts']:
            return None
        return dict(row)

    def get_horse_history(self, horse_name: str, limit: int = 10,
                          before_date: str = '') -> List[Dict]:
        """Get horse's past race results, most recent first.

        before_date: if set, only return races STRICTLY before this date
        (YYYY/MM/DD). Required by the rule engine in backtest, where the DB
        holds the whole timeline — without it this returns the race being
        scored (and later ones), leaking the current outcome (e.g. finish_time
        into speed_fig). Live is unaffected (the race isn't in the DB yet) so
        the default keeps the old no-filter behaviour.
        """
        c = self.conn.cursor()
        if before_date:
            c.execute("""
                SELECT * FROM race_results
                WHERE horse_name = ? AND race_date < ?
                ORDER BY race_date DESC, race_number DESC
                LIMIT ?
            """, (horse_name, before_date, limit))
        else:
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

    def get_horse_format_history(self, horse_name: str, venue: str,
                                  distance: int, going: str = '',
                                  before_date: str = '',
                                  limit: int = 6) -> list:
        """v6.6 dashboard: per-horse history at the SAME race format.

        "Same format" = same venue + same distance (exact). Optionally
        filter by going if `going` is provided (loose: returns all when
        empty). Returns the most-recent `limit` runs with finish position,
        odds, margin, jockey, race_date — used by the per-horse detail
        panel so users can see "this horse's last 5 runs at Sha Tin 1200m
        on Good".
        """
        c = self.conn.cursor()
        venue_norm = self._normalize_venue(venue)
        if going and before_date:
            c.execute("""
                SELECT race_date, venue, distance, going, class_type,
                       finish_position, win_odds, margin, jockey, draw
                FROM race_results
                WHERE horse_name = ?
                  AND venue = ? AND distance = ? AND going = ?
                  AND race_date < ?
                ORDER BY race_date DESC LIMIT ?
            """, (horse_name, venue_norm, int(distance), going, before_date, limit))
        elif before_date:
            c.execute("""
                SELECT race_date, venue, distance, going, class_type,
                       finish_position, win_odds, margin, jockey, draw
                FROM race_results
                WHERE horse_name = ?
                  AND venue = ? AND distance = ?
                  AND race_date < ?
                ORDER BY race_date DESC LIMIT ?
            """, (horse_name, venue_norm, int(distance), before_date, limit))
        else:
            c.execute("""
                SELECT race_date, venue, distance, going, class_type,
                       finish_position, win_odds, margin, jockey, draw
                FROM race_results
                WHERE horse_name = ?
                  AND venue = ? AND distance = ?
                ORDER BY race_date DESC LIMIT ?
            """, (horse_name, venue_norm, int(distance), limit))
        return [dict(r) for r in c.fetchall()]

    def get_horse_cd_history(self, horse_name: str, venue: str,
                              distance: int, tolerance: int = 100,
                              before_date: str = '') -> list:
        """Get horse's race-by-race C&D history with dates for winning trip detection.

        before_date: backtest-leak guard — without it the horse's CURRENT race
        (same venue+distance) is counted into its own C&D win/place record.
        Live passes none (race not yet in DB). See get_horse_history.
        """
        c = self.conn.cursor()
        venue_norm = self._normalize_venue(venue)
        df = ' AND race_date < ?' if before_date else ''
        params = ([horse_name, venue_norm, distance, tolerance, before_date]
                  if before_date else [horse_name, venue_norm, distance, tolerance])
        c.execute(f"""
            SELECT race_date, venue, distance, finish_position
            FROM race_results
            WHERE horse_name = ?
              AND venue = ?
              AND ABS(distance - ?) <= ?{df}
            ORDER BY race_date
        """, params)
        return [{'date': r[0], 'venue': r[1], 'distance': r[2], 'position': r[3]}
                for r in c.fetchall()]

    @staticmethod
    def _season_start_for(date_str: str) -> str:
        """First day of the HK season containing date_str (Jul 1 - Jun 30)."""
        y, m = int(date_str[:4]), int(date_str[5:7])
        return f"{y if m >= 7 else y - 1}/07/01"

    def _person_stats_asof(self, col: str, name: str, before_date: str) -> Optional[Dict]:
        """Compute a jockey/trainer's CURRENT-SEASON-to-date win/place rate from
        race_results with race_date < before_date — the leak-free equivalent of
        the pre-aggregated season tables (which include future races). Used by
        the rule engine in backtest. Returns rides AND runners keys so it serves
        both jockey (count_field='rides') and trainer ('runners')."""
        season_start = self._season_start_for(before_date)
        row = self.conn.execute(
            f"SELECT COUNT(*) AS n, COALESCE(SUM(is_winner),0) AS wins, "
            f"COALESCE(SUM(is_top3),0) AS places FROM race_results "
            f"WHERE {col} = ? AND race_date >= ? AND race_date < ? AND finish_position > 0",
            (name, season_start, before_date)).fetchone()
        n = row['n'] or 0
        if not n:
            return None
        wins, places = row['wins'] or 0, row['places'] or 0
        return {'name': name, 'rides': n, 'runners': n, 'wins': wins, 'places': places,
                'win_rate': wins / n, 'place_rate': places / n}

    def get_jockey_stats(self, jockey_name: str, before_date: str = '') -> Optional[Dict]:
        """Get jockey's season statistics. before_date: backtest-leak guard —
        the season table is rebuilt over the whole season incl. future races, so
        in backtest pass before_date to compute season-to-date from race_results."""
        if before_date:
            return self._person_stats_asof('jockey', jockey_name, before_date)
        c = self.conn.cursor()
        c.execute("SELECT * FROM jockey_stats WHERE name = ?", (jockey_name,))
        row = c.fetchone()
        return dict(row) if row else None

    def get_jockey_recent_form(self, jockey_name: str, n_rides: int = 10,
                               before_date: str = '') -> Dict:
        """Get jockey's recent form over last N rides.

        Hot streak detection — a jockey winning 4/10 recent rides is on fire,
        regardless of their season average. Bill Benter found this among the
        most predictive features.

        before_date: backtest-leak guard (see get_horse_history).
        """
        c = self.conn.cursor()
        df = ' AND race_date < ?' if before_date else ''
        c.execute(f"""
            SELECT is_winner, is_top3, finish_position
            FROM race_results
            WHERE jockey = ? AND finish_position > 0{df}
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (jockey_name, before_date, n_rides) if before_date else (jockey_name, n_rides))
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

    def get_trainer_recent_form(self, trainer_name: str, n_runners: int = 10,
                                before_date: str = '') -> Dict:
        """Get trainer's recent form over last N runners.

        before_date: backtest-leak guard (see get_horse_history).
        """
        c = self.conn.cursor()
        df = ' AND race_date < ?' if before_date else ''
        c.execute(f"""
            SELECT is_winner, is_top3, finish_position
            FROM race_results
            WHERE trainer = ? AND finish_position > 0{df}
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (trainer_name, before_date, n_runners) if before_date else (trainer_name, n_runners))
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

    def get_trainer_stats(self, trainer_name: str, before_date: str = '') -> Optional[Dict]:
        """Get trainer's season statistics. before_date: backtest-leak guard
        (see get_jockey_stats) — computes season-to-date from race_results."""
        if before_date:
            return self._person_stats_asof('trainer', trainer_name, before_date)
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

    @staticmethod
    def _going_surface_class(going: str) -> str:
        """Bucket going into FIRM-side (GOOD/FIRM/FAST) vs SOFT-side
        (YIELDING/SOFT/HEAVY/WET-SLOW). Prior LIKE 'GOOD%' pattern leaked
        'GOOD TO YIELDING' into firm queries — credited soft-side runs as
        firm-side experience."""
        g = (going or '').upper()
        if any(kw in g for kw in ('YIELDING', 'SOFT', 'HEAVY', 'WET-SLOW', 'WET SLOW')):
            return 'SOFT'
        return 'FIRM'

    def get_horse_going_record(self, horse_name: str, going: str,
                               before_date: str = '') -> Dict:
        """Get horse's record on a specific going (track condition).

        Filters prior runs by FIRM vs SOFT surface class so a 'GOOD TO YIELDING'
        prior doesn't get credited as firm-side experience.
        before_date: if set, only count races STRICTLY before this date — same
        backtest-leak fix as get_horse_history (live passes no filter).
        """
        c = self.conn.cursor()
        if not (going and going.strip()):
            return {'runs': 0, 'wins': 0, 'places': 0, 'avg_pos': 0}

        if self._going_surface_class(going) == 'SOFT':
            class_filter = (
                "(UPPER(going) LIKE '%YIELDING%' "
                "OR UPPER(going) LIKE '%SOFT%' "
                "OR UPPER(going) LIKE '%HEAVY%' "
                "OR UPPER(going) LIKE '%WET%')"
            )
        else:
            class_filter = (
                "UPPER(going) NOT LIKE '%YIELDING%' "
                "AND UPPER(going) NOT LIKE '%SOFT%' "
                "AND UPPER(going) NOT LIKE '%HEAVY%' "
                "AND UPPER(going) NOT LIKE '%WET%' "
                "AND (UPPER(going) LIKE '%GOOD%' "
                "OR UPPER(going) LIKE '%FIRM%' "
                "OR UPPER(going) LIKE '%FAST%')"
            )

        if before_date:
            c.execute(f"""
                SELECT
                    COUNT(*) as runs,
                    SUM(is_winner) as wins,
                    SUM(is_top3) as places,
                    AVG(CASE WHEN finish_position > 0 THEN finish_position END) as avg_pos
                FROM race_results
                WHERE horse_name = ? AND {class_filter}
                  AND race_date < ?
            """, (horse_name, before_date))
        else:
            c.execute(f"""
                SELECT
                    COUNT(*) as runs,
                    SUM(is_winner) as wins,
                    SUM(is_top3) as places,
                    AVG(CASE WHEN finish_position > 0 THEN finish_position END) as avg_pos
                FROM race_results
                WHERE horse_name = ? AND {class_filter}
            """, (horse_name,))
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

    def get_rolling_stats(self, horse_name: str, n_races: int = 5,
                          before_date: str = '') -> Dict:
        """Get rolling performance stats over last N races.

        Returns avg position, win rate, place rate, and trend direction.
        This is one of the highest-impact features in ML racing models.

        before_date: backtest-leak guard — without it this reads the CURRENT
        race's own finish_position/is_winner/margin (the race is in the DB).
        Live passes none (the race isn't in the DB yet). See get_horse_history.
        """
        c = self.conn.cursor()
        df = ' AND race_date < ?' if before_date else ''
        c.execute(f"""
            SELECT finish_position, is_winner, is_top3, margin, win_odds
            FROM race_results
            WHERE horse_name = ? AND finish_position > 0{df}
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (horse_name, before_date, n_races) if before_date else (horse_name, n_races))
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

    def get_last_jockey(self, horse_name: str, before_date: str = '') -> Optional[str]:
        """Get the jockey who rode this horse in its most recent race.

        Used to detect jockey changes — a new jockey booking often signals
        the trainer is making a strategic move.

        before_date: backtest-leak guard — without it the "most recent race"
        is the current race itself (so the jockey-change factor never fires in
        backtest). See get_horse_history.
        """
        c = self.conn.cursor()
        df = ' AND race_date < ?' if before_date else ''
        c.execute(f"""
            SELECT jockey FROM race_results
            WHERE horse_name = ? AND jockey != ''{df}
            ORDER BY race_date DESC, race_number DESC
            LIMIT 1
        """, (horse_name, before_date) if before_date else (horse_name,))
        row = c.fetchone()
        return row['jockey'] if row else None

    # Class-rank multipliers for beaten-margin adjustment. Lower rank = higher
    # class (G1=0, Class 5=5). Dividing raw margin by these maps "1L beaten in
    # Class 1" to a smaller effective margin (more impressive) than "1L beaten
    # in Class 5", which is what professional handicappers want.
    _CLASS_MARGIN_DIVISOR = {0: 1.30, 1: 1.15, 2: 1.05, 3: 1.00,
                             4: 0.95, 5: 0.85}

    @staticmethod
    def _class_adjust_margin(margin, class_type: str) -> float:
        if not margin or margin <= 0:
            return 0.0
        from ml.features import extract_class_rank
        rank = extract_class_rank(class_type or '')
        if rank is None:
            return float(margin)
        return float(margin) / HorseDatabase._CLASS_MARGIN_DIVISOR.get(rank, 1.0)

    def get_beaten_margins(self, horse_name: str, n_races: int = 5,
                           before_date: str = '') -> Dict:
        """Get beaten margin stats for recent races.

        Margins are class-adjusted before bucketing (1L in Class 1 is a
        different beating than 1L in Class 5).

        before_date: backtest-leak guard (see get_horse_history) — without it
        this reads the current race's own margin/finish.
        """
        c = self.conn.cursor()
        df = ' AND race_date < ?' if before_date else ''
        c.execute(f"""
            SELECT finish_position, margin, class_type
            FROM race_results
            WHERE horse_name = ? AND finish_position > 0{df}
            ORDER BY race_date DESC, race_number DESC
            LIMIT ?
        """, (horse_name, before_date, n_races) if before_date else (horse_name, n_races))
        rows = c.fetchall()
        if not rows:
            return {'runs': 0, 'close_finishes': 0, 'avg_margin': 0, 'last_margin': 0}

        adj_margins = [
            self._class_adjust_margin(r['margin'], r['class_type'])
            for r in rows if r['margin'] and r['margin'] > 0
        ]
        close = [m for m in adj_margins if m <= 2.75]
        last_margin = adj_margins[0] if adj_margins else 0
        last_winner = rows[0]['finish_position'] == 1 if rows else False

        return {
            'runs': len(rows),
            'close_finishes': len(close),
            'avg_margin': round(sum(adj_margins) / len(adj_margins), 2) if adj_margins else 0,
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
            # DB dates are YYYY/MM/DD, but normalize '-' too (some legacy /
            # Renavon rows) so a format mismatch doesn't silently return 0.
            last = datetime.strptime(row['race_date'].replace('-', '/'), '%Y/%m/%d')
            curr = datetime.strptime(current_date.replace('-', '/'), '%Y/%m/%d')
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

    @staticmethod
    def _current_season_start() -> str:
        """First day of the current HK racing season as 'YYYY/MM/DD'.
        Seasons run Jul 1 – Jun 30, so July 2025 starts the 2025-26 season.
        """
        now = datetime.now()
        year = now.year if now.month >= 7 else now.year - 1
        return f"{year}/07/01"

    def rebuild_stats(self):
        """Rebuild jockey_stats and trainer_stats from race_results.

        Computes CURRENT-SEASON only (race_date >= season start) so visiting
        or retired jockeys/trainers from prior seasons don't show inflated
        lifetime totals. Renavon snapshots are frozen at 2026/04/22 and no
        longer ingested, so race_results is the freshest local source. The
        HKJC seasonal scrape (`scripts/scrape_history.py --rankings`)
        overwrites these rows afterwards with authoritative figures.

        Safe to call at any time — replaces both tables entirely.
        """
        c = self.conn.cursor()
        season = self._current_season()
        season_start = self._current_season_start()

        c.execute("DELETE FROM jockey_stats")
        c.execute("""
            INSERT INTO jockey_stats (name, season, rides, wins, places, win_rate, place_rate, last_updated)
            SELECT jockey, ?, COUNT(*), SUM(is_winner), SUM(is_top3),
                   CAST(SUM(is_winner) AS REAL) / COUNT(*),
                   CAST(SUM(is_top3) AS REAL) / COUNT(*),
                   MAX(race_date)
            FROM race_results
            WHERE jockey != '' AND race_date >= ?
            GROUP BY jockey
        """, (season, season_start))

        c.execute("DELETE FROM trainer_stats")
        c.execute("""
            INSERT INTO trainer_stats (name, season, runners, wins, places, win_rate, place_rate, last_updated)
            SELECT trainer, ?, COUNT(*), SUM(is_winner), SUM(is_top3),
                   CAST(SUM(is_winner) AS REAL) / COUNT(*),
                   CAST(SUM(is_top3) AS REAL) / COUNT(*),
                   MAX(race_date)
            FROM race_results
            WHERE trainer != '' AND race_date >= ?
            GROUP BY trainer
        """, (season, season_start))

        self.conn.commit()
        logger.info(
            f"Rebuilt jockey_stats and trainer_stats from current-season race_results (since {season_start})"
        )

    # ------------------------------------------------------------------
    # Card-field backfill (rating, last_6, gear, age, rating_change)
    # ------------------------------------------------------------------

    def backfill_card_fields(self, race_date: str, race_data: list) -> int:
        """Fill rating / last_6 / gear / age / rating_change into race_results
        from a saved race_data prediction JSON.

        Card-only fields aren't on HKJC results pages, so the scrape pipeline
        leaves them sparse. The prediction JSON (results/race_data_*.json)
        captures them at race-card scrape time. This walks the saved races
        and updates matching race_results rows by (race_date, race_number,
        horse_name), only writing when the new value is non-empty AND the
        existing row's value is missing/zero (preserves manually corrected data).

        Returns the count of rows actually updated.
        """
        c = self.conn.cursor()
        n_updated = 0
        for race in race_data:
            race_no = race.get('race_number') or race.get('race_no') or 0
            for h in race.get('horses', []):
                horse_name = h.get('name', '')
                if not horse_name:
                    continue
                rating = h.get('rating') or 0
                last_6 = h.get('last_6') or ''
                gear = h.get('gear') or ''
                age = str(h.get('age') or '') or ''
                rating_change = h.get('rating_change') or 0

                # Only update fields where new value is meaningful AND
                # existing row's value is missing — preserves manual edits.
                c.execute("""
                    UPDATE race_results SET
                        rating = CASE WHEN ? > 0 AND (rating IS NULL OR rating = 0) THEN ? ELSE rating END,
                        last_6 = CASE WHEN ? != '' AND (last_6 IS NULL OR last_6 = '') THEN ? ELSE last_6 END,
                        gear = CASE WHEN ? != '' AND (gear IS NULL OR gear = '') THEN ? ELSE gear END,
                        age = CASE WHEN ? != '' AND (age IS NULL OR age = '') THEN ? ELSE age END,
                        rating_change = CASE WHEN ? != 0 AND (rating_change IS NULL OR rating_change = 0) THEN ? ELSE rating_change END
                    WHERE race_date = ? AND race_number = ? AND horse_name = ?
                """, (
                    rating, rating,
                    last_6, last_6,
                    gear, gear,
                    age, age,
                    rating_change, rating_change,
                    race_date, race_no, horse_name,
                ))
                if c.rowcount > 0:
                    n_updated += c.rowcount
        self.conn.commit()
        return n_updated

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

            ticket = race.get('bet_ticket')
            if ticket:
                # Mirror prediction_log's pattern: pre-race tickets can update
                # as odds drift, but once ANY horse in this race has a result,
                # the ticket is frozen. Prevents post-results regen from
                # overwriting the actual staked ticket — the "frozen" promise
                # that validate_hk --replay depends on (GPT-5.5 Finding #3
                # 2026-05-24).
                results_filled = c.execute(
                    "SELECT 1 FROM prediction_log WHERE race_date=? AND race_number=? "
                    "AND COALESCE(results_filled, 0) = 1 LIMIT 1",
                    (race_date, race_no),
                ).fetchone()
                if not results_filled:
                    c.execute("""
                        INSERT INTO race_bet_ticket (
                            race_date, race_number, ticket_json, logged_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(race_date, race_number) DO UPDATE SET
                            ticket_json=excluded.ticket_json,
                            logged_at=excluded.logged_at
                    """, (race_date, race_no, json.dumps(ticket), now))

            for h in horses:
                c.execute("""
                    INSERT INTO prediction_log (
                        race_date, race_number, venue, distance, class_type, going,
                        field_size, horse_name, horse_number, predicted_rank,
                        rule_score, ml_prob, calibrated_prob, expected_return,
                        bet_recommendation, confidence, win_odds, model_version,
                        conviction_tier, predicted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        conviction_tier=excluded.conviction_tier,
                        predicted_at=excluded.predicted_at
                    WHERE COALESCE(prediction_log.results_filled, 0) = 0
                """, (
                    race_date, race_no, venue, distance, class_type, going,
                    field_size, h.get('name', ''), h.get('number', 0),
                    h.get('rank', 0), h.get('rule_score', 0),
                    h.get('ml_prob', 0), h.get('calibrated_prob', 0),
                    h.get('expected_return', 0), h.get('bet_recommendation', 'SKIP'),
                    h.get('confidence', ''), h.get('win_odds', 0),
                    model_version,
                    race.get('conviction_tier'),  # per-race, same for every horse
                    now,
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

    # Per-combo cost table for exotic stakes (1 unit per combo)
    _BET_COMBOS = {
        # bet_type → leg_count → combo count
        'QIN':   {2: 1, 3: 3, 4: 6},   # straight QIN / BOX Q3 / BOX Q4
        'TRIO':  {3: 1, 4: 4, 5: 10, 6: 20},
    }

    def _real_exotic_pnl(self, race_date: str, race_number: int,
                         bet_type: str, leg_count: int, hit: bool,
                         actual_top2_nums, actual_top3_nums):
        """Return (pnl, stake) for an exotic bet using real HKJC dividends.

        Falls back to (None, stake) if the dividend can't be looked up — caller
        should then fall back to the stored profit_loss column.
        """
        combos = self._BET_COMBOS.get(bet_type, {}).get(leg_count)
        if combos is None:
            return None, 0.0
        stake = float(combos)
        if not hit:
            return -stake, stake
        pool = 'QUINELLA' if bet_type == 'QIN' else 'TRIO'
        winning_nums = actual_top2_nums if bet_type == 'QIN' else actual_top3_nums
        nums = sorted(int(n) for n in winning_nums if n)
        if (bet_type == 'QIN' and len(nums) != 2) or (bet_type == 'TRIO' and len(nums) != 3):
            return None, stake
        combo = ','.join(str(n) for n in nums)
        row = self.conn.execute(
            "SELECT dividend FROM race_dividends "
            "WHERE race_date=? AND race_number=? AND pool=? AND combination=?",
            (race_date, race_number, pool, combo)
        ).fetchone()
        if not row:
            return None, stake
        dividend = hkjc_declared_to_unit(row[0])
        if dividend is None:
            return None, stake
        return dividend - stake, stake

    def get_accuracy_summary(self, days: int = 30) -> Dict:
        """Get prediction accuracy summary over the last N days.

        Bets are evaluated at race-level (one bet per race, taken from the
        rank-1 row since `bet_recommendation` is replicated across the top-3
        rows of each race). Hit logic per bet type:
          WIN, BANKER WIN, BANKER : rank-1 horse won
          PLACE                   : rank-1 horse placed (top-3)
          QIN                     : ≥2 of model's top-3 picks finished in top 2
          TRIO                    : all 3 of model's top-3 picks finished in top 3
        SKIP is excluded.

        P&L sources (2026-05-17):
          - WIN / BANKER WIN / PLACE: prediction_log.profit_loss (per-row,
            computed from win_odds when fill_prediction_results runs)
          - QIN / TRIO: REAL HKJC dividend looked up from race_dividends,
            with stake = #combos × 1u (QIN: 1/3/6 combos for 2/3/4 legs,
            TRIO: 1/4/10/20 combos for 3/4/5/6 legs). Falls back to stored
            profit_loss if the dividend lookup misses.
        """
        c = self.conn.cursor()
        cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).strftime('%Y/%m/%d')

        # Only races where results have been filled. horse_number is needed
        # for joining actual top-2/top-3 numbers to race_dividends.
        rows = c.execute("""
            SELECT race_date, race_number, horse_name, predicted_rank,
                   bet_recommendation, calibrated_prob, win_odds,
                   actual_position, actual_won, actual_top3, profit_loss, confidence,
                   horse_number, conviction_tier
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
        total_profit = 0.0   # 1-unit-stake terms
        total_stake = 0.0    # 1-unit-stake terms (1u for WIN/PLACE/straight,
                             # 3u BOX Q3, 6u BOX Q4)
        banker_count = 0
        banker_wins = 0
        # Two independent breakdowns:
        # - by_confidence: prediction_log.confidence (ML combined_confidence
        #   VERY_HIGH/HIGH/MEDIUM/LOW). Model's self-confidence.
        # - by_conviction: prediction_log.conviction_tier (EDGE/HIGH/MED/LOW
        #   from betting/conviction.py). Strategy-level edge over market.
        # These are different dimensions; pre-2026-05-17 the report mislabeled
        # by_confidence as "conviction" — audit fix made the labels honest.
        by_confidence = {}
        by_conviction = {}

        for race_key, horses in races.items():
            horses.sort(key=lambda x: x[3])  # sort by predicted_rank
            rank1 = horses[0]

            # Rank 1 win check
            if rank1[8]:  # actual_won
                rank1_wins += 1

            # Top-3 overlap
            our_top3 = {h[2] for h in horses[:3]}
            actual_top3_set = {h[2] for h in horses if h[9]}
            top3_overlaps += len(our_top3 & actual_top3_set)

            # Where did winner rank?
            for h in horses:
                if h[8]:
                    winner_ranks.append(h[3])
                    break

            # By-confidence (ML self-confidence) AND by-conviction (strategy edge)
            conf = rank1[11] or 'UNKNOWN'
            conv = rank1[13] or 'UNKNOWN'
            for bucket, key in ((by_confidence, conf), (by_conviction, conv)):
                bucket.setdefault(key, {'total': 0, 'wins': 0,
                                        'bets': 0, 'hits': 0,
                                        'profit': 0.0, 'stake': 0.0})
                bucket[key]['total'] += 1
                if rank1[8]:
                    bucket[key]['wins'] += 1

            # Race-level bet evaluation
            bet = rank1[4]
            if not bet or bet == 'SKIP':
                continue
            bets_placed += 1

            # 2026-05-17 audit fix: for exotic bets, hit check must look at the
            # actual legs in that bet (all horses marked with this bet_recommendation),
            # not top-3. BOX Q4 stakes 6u across 4 legs — checking only top-3 missed
            # any race where rank-1 + rank-4 finished top-2, falsely scoring it -6u.
            if bet in ('WIN', 'BANKER WIN', 'BANKER'):
                hit = bool(rank1[8])
            elif bet == 'PLACE':
                hit = bool(rank1[9])
            elif bet in ('QIN', 'TRIO'):
                legs = [h for h in horses if h[4] == bet]
                legs_in_top2 = sum(1 for h in legs if h[7] in (1, 2))
                legs_in_top3 = sum(1 for h in legs if h[9])
                hit = (legs_in_top2 >= 2) if bet == 'QIN' else (legs_in_top3 >= 3)
            else:
                hit = False  # unknown bet type

            if hit:
                bets_won += 1
            if 'BANKER' in bet:
                banker_count += 1
                if hit:
                    banker_wins += 1

            # P&L: normalize everything to 1-unit-stake terms.
            #   - WIN / PLACE / BANKER WIN: stored profit_loss is in $10-stake
            #     dollars (see fill_prediction_results) → divide by 10
            #   - QIN / TRIO: look up real HKJC dividend; cost = #combos × 1u
            if bet in ('QIN', 'TRIO'):
                leg_count = sum(1 for h in horses if h[4] == bet)
                actual_top2 = [h[12] for h in horses if h[7] in (1, 2)]
                actual_top3 = [h[12] for h in horses if h[9]]
                pnl, stake = self._real_exotic_pnl(
                    horses[0][0], horses[0][1], bet, leg_count, hit,
                    actual_top2, actual_top3,
                )
                if pnl is None:
                    pnl = (rank1[10] or 0) / 10.0
                    stake = 1.0
            else:
                pnl = (rank1[10] or 0) / 10.0
                stake = 1.0
            total_profit += pnl
            total_stake += stake
            # Per-bucket bet aggregates (using same 1u-stake terms)
            for bucket, key in ((by_confidence, conf), (by_conviction, conv)):
                tier = bucket[key]
                tier['bets'] += 1
                if hit:
                    tier['hits'] += 1
                tier['profit'] += pnl
                tier['stake'] += stake

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
            'total_stake': round(total_stake, 1),
            'roi_percent': round(total_profit / total_stake * 100, 1) if total_stake else 0,
            'banker_count': banker_count,
            'banker_wins': banker_wins,
            'banker_hit_rate': round(banker_wins / banker_count * 100, 1) if banker_count else 0,
            'by_confidence': by_confidence,
            'by_conviction': by_conviction,
        }

    def get_accuracy_by_class(self, days: int = 30) -> list:
        """Get accuracy breakdown by class type with real exotic dividend P&L.

        Same race-level P&L logic as get_accuracy_summary — 1-unit-stake terms,
        real dividends for QIN/TRIO via _real_exotic_pnl, /10 normalization for
        stored WIN/PLACE profit_loss.
        """
        c = self.conn.cursor()
        cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).strftime('%Y/%m/%d')

        rows = c.execute("""
            SELECT race_date, race_number, class_type, horse_name, predicted_rank,
                   bet_recommendation, actual_position, actual_won, actual_top3,
                   profit_loss, horse_number
            FROM prediction_log
            WHERE results_filled = 1 AND race_date >= ?
            ORDER BY race_date, race_number, predicted_rank
        """, (cutoff,)).fetchall()

        # Group by race
        races = {}
        for r in rows:
            races.setdefault(f"{r[0]}-{r[1]}", []).append(r)

        # Aggregate by class
        by_class = {}  # class → {'races','rank1_wins','top3_hits','profit'}
        for horses in races.values():
            horses.sort(key=lambda x: x[4])  # by predicted_rank
            rank1 = horses[0]
            klass = rank1[2] or 'Unknown'
            d = by_class.setdefault(klass, {'races': 0, 'rank1_wins': 0,
                                            'top3_hits': 0, 'profit': 0.0})
            d['races'] += 1
            if rank1[7]:  # actual_won
                d['rank1_wins'] += 1
            d['top3_hits'] += sum(1 for h in horses[:3] if h[8])  # actual_top3

            bet = rank1[5]
            if not bet or bet == 'SKIP':
                continue

            # Race-level hit + 1u-stake P&L (mirrors get_accuracy_summary).
            # Exotic hit check uses actual bet legs (audit fix 2026-05-17).
            if bet in ('WIN', 'BANKER WIN', 'BANKER'):
                hit = bool(rank1[7])
            elif bet == 'PLACE':
                hit = bool(rank1[8])
            elif bet in ('QIN', 'TRIO'):
                legs = [h for h in horses if h[5] == bet]
                legs_in_top2 = sum(1 for h in legs if h[6] in (1, 2))
                legs_in_top3 = sum(1 for h in legs if h[8])
                hit = (legs_in_top2 >= 2) if bet == 'QIN' else (legs_in_top3 >= 3)
            else:
                hit = False

            if bet in ('QIN', 'TRIO'):
                leg_count = sum(1 for h in horses if h[5] == bet)
                actual_top2 = [h[10] for h in horses if h[6] in (1, 2)]
                actual_top3 = [h[10] for h in horses if h[8]]
                pnl, _ = self._real_exotic_pnl(
                    rank1[0], rank1[1], bet, leg_count, hit,
                    actual_top2, actual_top3,
                )
                if pnl is None:
                    pnl = (rank1[9] or 0) / 10.0
            else:
                pnl = (rank1[9] or 0) / 10.0
            d['profit'] += pnl

        # Render sorted by race count (most frequent class first)
        result = []
        for klass, d in sorted(by_class.items(), key=lambda kv: -kv[1]['races']):
            n = d['races'] or 1
            result.append({
                'class': klass,
                'races': n,
                'rank1_win_rate': round(d['rank1_wins'] / n * 100, 1),
                'top3_hit_rate': round(d['top3_hits'] / (n * 3) * 100, 1),
                'profit': round(d['profit'], 1),
            })
        return result

    def get_bet_type_roi(self, days: int = 30) -> list:
        """Get ROI breakdown by bet recommendation type — race-level.

        Same hit rules as get_accuracy_summary. P&L is summed from the rank-1
        row of each race so each race contributes exactly once per bet type.
        """
        c = self.conn.cursor()
        cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).strftime('%Y/%m/%d')

        rows = c.execute("""
            SELECT race_date, race_number, horse_name, predicted_rank,
                   bet_recommendation, actual_position, actual_won,
                   actual_top3, profit_loss, horse_number
            FROM prediction_log
            WHERE results_filled = 1 AND race_date >= ?
            ORDER BY race_date, race_number, predicted_rank
        """, (cutoff,)).fetchall()

        races = {}
        for r in rows:
            races.setdefault(f"{r[0]}-{r[1]}", []).append(r)

        by_type = {}  # bet_type -> {'count','hits','profit','stake'}
        for horses in races.values():
            horses.sort(key=lambda x: x[3])
            rank1 = horses[0]
            bet = rank1[4]
            if not bet or bet == 'SKIP':
                continue

            # Exotic hit check uses actual bet legs (audit fix 2026-05-17).
            if bet in ('WIN', 'BANKER WIN', 'BANKER'):
                hit = bool(rank1[6])
            elif bet == 'PLACE':
                hit = bool(rank1[7])
            elif bet in ('QIN', 'TRIO'):
                legs = [h for h in horses if h[4] == bet]
                legs_in_top2 = sum(1 for h in legs if h[5] in (1, 2))
                legs_in_top3 = sum(1 for h in legs if h[7])
                hit = (legs_in_top2 >= 2) if bet == 'QIN' else (legs_in_top3 >= 3)
            else:
                hit = False

            # P&L in 1-unit-stake terms (see get_accuracy_summary for full
            # rationale on unit normalization).
            if bet in ('QIN', 'TRIO'):
                leg_count = sum(1 for h in horses if h[4] == bet)
                actual_top2 = [h[9] for h in horses if h[5] in (1, 2)]
                actual_top3 = [h[9] for h in horses if h[7]]
                pnl, stake = self._real_exotic_pnl(
                    horses[0][0], horses[0][1], bet, leg_count, hit,
                    actual_top2, actual_top3,
                )
                if pnl is None:
                    pnl = (rank1[8] or 0) / 10.0
                    stake = 1.0
            else:
                pnl = (rank1[8] or 0) / 10.0
                stake = 1.0

            d = by_type.setdefault(bet, {'count': 0, 'hits': 0, 'profit': 0.0, 'stake': 0.0})
            d['count'] += 1
            if hit:
                d['hits'] += 1
            d['profit'] += pnl
            d['stake'] += stake

        result = []
        for bet, d in sorted(by_type.items(), key=lambda kv: -kv[1]['profit']):
            n = d['count']
            stake = d['stake'] or n  # fallback to 1u/bet if no stake tracked
            result.append({
                'bet_type': bet,
                'count': n,
                'hits': d['hits'],
                'hit_rate': round(d['hits'] / n * 100, 1) if n else 0.0,
                'profit': round(d['profit'], 1),
                'roi': round(d['profit'] / stake * 100, 1) if stake else 0.0,
            })
        return result

    def close(self):
        # Per SQLite recommendation: run `PRAGMA optimize` just before close
        # so the query planner's per-index statistics stay fresh. Cheap
        # (~50ms) — only re-ANALYZEs indexes that were materially updated
        # since last optimize.
        try:
            self.conn.execute("PRAGMA optimize")
        except sqlite3.DatabaseError:
            pass
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
