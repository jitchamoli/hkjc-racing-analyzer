#!/usr/bin/env python3
"""Backfill historical HKJC dividends into race_dividends table.

Walks distinct race dates in race_results and scrapes the dividend tables
from each results page. Idempotent — uses UPSERT so re-running is safe.

Usage:
    python3 -m scripts.backfill_dividends                 # All missing dates
    python3 -m scripts.backfill_dividends --date 2026/04/19
    python3 -m scripts.backfill_dividends --last 20       # Last 20 meetings
    python3 -m scripts.backfill_dividends --force         # Re-scrape even if present
"""

import argparse
import asyncio
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.scraper import scrape_meeting_results
from data.database import HorseDatabase


async def find_dates_to_backfill(conn, last_n=None, force=False):
    """Return list of race dates that need dividend backfill."""
    all_dates = [row[0] for row in conn.execute(
        "SELECT DISTINCT race_date FROM race_results ORDER BY race_date DESC"
    ).fetchall()]

    if last_n:
        all_dates = all_dates[:last_n]

    if force:
        return all_dates

    # Only return dates missing dividends
    missing = []
    for d in all_dates:
        row = conn.execute(
            "SELECT COUNT(*) FROM race_dividends WHERE race_date = ?", (d,)
        ).fetchone()
        if row[0] == 0:
            missing.append(d)
    return missing


async def backfill_date(date: str, db: HorseDatabase) -> int:
    """Scrape a single date's results and ingest dividends.

    Returns number of dividend rows inserted.
    """
    races = await scrape_meeting_results(date, max_races=12)
    if not races:
        return 0

    # We only want the dividends — ingest_race_day also re-ingests race_results
    # which is a no-op via UPSERT but wastes time. Just ingest dividends directly.
    c = db.conn.cursor()
    total_divs = 0
    for race in races:
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
            """, (date, race_no, pool, combo, float(div)))
            total_divs += 1
    db.conn.commit()
    return total_divs


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='Single date to backfill (YYYY/MM/DD)')
    parser.add_argument('--last', type=int, help='Only the last N meetings')
    parser.add_argument('--force', action='store_true',
                        help='Re-scrape dates even if dividends already present')
    args = parser.parse_args()

    db = HorseDatabase()

    if args.date:
        dates = [args.date]
    else:
        dates = await find_dates_to_backfill(db.conn, last_n=args.last, force=args.force)

    if not dates:
        print("No dates need backfill (all have dividends). Use --force to re-scrape.")
        db.close()
        return 0

    print(f"Backfilling dividends for {len(dates)} meeting dates...")
    total_races = 0
    total_divs = 0
    for i, date in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] {date} ...", end=' ', flush=True)
        try:
            n = await backfill_date(date, db)
            total_divs += n
            if n > 0:
                total_races += 1
            print(f"{n} dividend rows")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone. {total_divs} dividend rows inserted across {total_races} meetings.")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
