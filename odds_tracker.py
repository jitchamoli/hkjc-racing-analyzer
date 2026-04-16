#!/usr/bin/env python3
"""
HKJC Odds Movement Tracker
SQLite-backed odds history with movement analysis.
Detects steam moves, drifters, and late market support.
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracking_data")
DB_PATH = os.path.join(DB_DIR, "odds_history.db")


# ---------------------------------------------------------------------------
# SQLite Odds Database
# ---------------------------------------------------------------------------

class OddsDatabase:
    """Stores timestamped odds snapshots in SQLite."""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS odds_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_date TEXT NOT NULL,
                    race_number INTEGER NOT NULL,
                    horse_number INTEGER NOT NULL,
                    horse_name TEXT,
                    win_odds REAL,
                    place_odds REAL,
                    timestamp TEXT NOT NULL,
                    source TEXT DEFAULT 'auto'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_race_horse
                ON odds_snapshots(race_date, race_number, horse_number)
            """)
            conn.commit()
        finally:
            conn.close()

    def record_snapshot(
        self,
        race_date: str,
        race_number: int,
        horses: List[Dict],
        source: str = "auto",
    ):
        """Insert a snapshot for all horses in a race.

        horses: list of dicts with keys: number, name, win_odds, place_odds
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        try:
            conn.executemany(
                """INSERT INTO odds_snapshots
                   (race_date, race_number, horse_number, horse_name,
                    win_odds, place_odds, timestamp, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        race_date,
                        race_number,
                        h.get("number", 0),
                        h.get("name", ""),
                        h.get("win_odds", 0),
                        h.get("place_odds", 0),
                        ts,
                        source,
                    )
                    for h in horses
                    if h.get("number", 0) > 0
                ],
            )
            conn.commit()
            logger.debug(
                f"Recorded odds snapshot: R{race_number} ({len(horses)} horses)"
            )
        finally:
            conn.close()

    def get_odds_history(
        self, race_date: str, race_number: int, horse_number: int
    ) -> List[Tuple[str, float]]:
        """Return timestamped win odds for a horse: [(timestamp, odds), ...]."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT timestamp, win_odds FROM odds_snapshots
                   WHERE race_date=? AND race_number=? AND horse_number=?
                   ORDER BY timestamp ASC""",
                (race_date, race_number, horse_number),
            ).fetchall()
            return rows
        finally:
            conn.close()

    def get_opening_odds(
        self, race_date: str, race_number: int
    ) -> Dict[int, float]:
        """Return first recorded odds for each horse: {number: odds}."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT horse_number, win_odds FROM odds_snapshots
                   WHERE race_date=? AND race_number=?
                   AND id IN (
                       SELECT MIN(id) FROM odds_snapshots
                       WHERE race_date=? AND race_number=?
                       GROUP BY horse_number
                   )""",
                (race_date, race_number, race_date, race_number),
            ).fetchall()
            return {r[0]: r[1] for r in rows if r[1] > 0}
        finally:
            conn.close()

    def get_latest_odds(
        self, race_date: str, race_number: int
    ) -> Dict[int, float]:
        """Return most recent recorded odds for each horse."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT horse_number, win_odds FROM odds_snapshots
                   WHERE race_date=? AND race_number=?
                   AND id IN (
                       SELECT MAX(id) FROM odds_snapshots
                       WHERE race_date=? AND race_number=?
                       GROUP BY horse_number
                   )""",
                (race_date, race_number, race_date, race_number),
            ).fetchall()
            return {r[0]: r[1] for r in rows if r[1] > 0}
        finally:
            conn.close()

    def snapshot_count(self, race_date: str, race_number: int) -> int:
        """Number of distinct snapshots recorded for a race."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT COUNT(DISTINCT timestamp) FROM odds_snapshots
                   WHERE race_date=? AND race_number=?""",
                (race_date, race_number),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Odds Movement Analyzer
# ---------------------------------------------------------------------------

class OddsMovementAnalyzer:
    """Detects steam moves, drifters, and late money from odds history."""

    @staticmethod
    def analyze_movement(
        opening_odds: float,
        current_odds: float,
        history: Optional[List[Tuple[str, float]]] = None,
    ) -> Tuple[float, str]:
        """Score odds movement.

        Returns (score 0-100, reason string).
        Higher score = more positive movement (money coming in).
        """
        if opening_odds <= 0 or current_odds <= 0:
            return 0.0, "No odds movement data"

        # Percentage change (positive = shortening = good)
        pct_change = (opening_odds - current_odds) / opening_odds * 100

        # Late movement detection
        late_move = ""
        if history and len(history) >= 4:
            # Compare last quarter of snapshots to earlier
            split = len(history) * 3 // 4
            early_avg = sum(h[1] for h in history[:split]) / split
            late_avg = sum(h[1] for h in history[split:]) / (len(history) - split)
            late_pct = (early_avg - late_avg) / early_avg * 100 if early_avg > 0 else 0

            if late_pct > 15:
                late_move = " [LATE SUPPORT]"
            elif late_pct < -15:
                late_move = " [LATE DRIFT]"

        # Score based on movement
        if pct_change >= 30:
            score = 92.0
            label = f"STEAM: Shortened {pct_change:.0f}% (${opening_odds:.1f}→${current_odds:.1f})"
        elif pct_change >= 20:
            score = 80.0
            label = f"Strong support: -{pct_change:.0f}% (${opening_odds:.1f}→${current_odds:.1f})"
        elif pct_change >= 10:
            score = 68.0
            label = f"Money coming: -{pct_change:.0f}% (${opening_odds:.1f}→${current_odds:.1f})"
        elif pct_change >= -10:
            score = 50.0
            label = f"Stable market (${current_odds:.1f})"
        elif pct_change >= -25:
            score = 35.0
            label = f"Mild drift: +{abs(pct_change):.0f}% (${opening_odds:.1f}→${current_odds:.1f})"
        else:
            score = 20.0
            label = f"DRIFT: +{abs(pct_change):.0f}% (${opening_odds:.1f}→${current_odds:.1f})"

        # Late move bonus/penalty
        if "[LATE SUPPORT]" in late_move:
            score = min(98.0, score + 10)
        elif "[LATE DRIFT]" in late_move:
            score = max(10.0, score - 8)

        return score, label + late_move

    @staticmethod
    def analyze_from_tips(
        tips_odds: float, current_odds: float
    ) -> Tuple[float, str]:
        """Simpler analysis using tips odds as proxy for opening odds.

        Useful when we don't have full SQLite history.
        """
        if tips_odds <= 0 or current_odds <= 0:
            return 0.0, "No odds movement data"

        return OddsMovementAnalyzer.analyze_movement(tips_odds, current_odds)


# ---------------------------------------------------------------------------
# Polling helper
# ---------------------------------------------------------------------------

async def poll_odds_snapshot(fetcher, race_date: str, total_races: int = 11):
    """Fetch current odds for all races and store in database.

    Call this periodically (every 10-15 min) on race day.
    """
    db = OddsDatabase()
    for race_no in range(1, total_races + 1):
        try:
            odds = await fetcher.get_live_odds(race_date, race_no)
            if odds:
                horses = [
                    {"number": num, "name": "", "win_odds": ov, "place_odds": 0}
                    for num, ov in odds.items()
                ]
                db.record_snapshot(race_date, race_no, horses, source="poll")
                logger.info(f"Polled R{race_no}: {len(odds)} horses")
        except Exception as e:
            logger.warning(f"Failed to poll R{race_no}: {e}")
