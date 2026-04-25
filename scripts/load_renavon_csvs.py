#!/usr/bin/env python3
"""Load full Renavon HKJC export into our SQLite DB as separate tables.

The one-time export contains 11 gzipped CSVs covering race results, horses,
sectional times, dividends, etc. This script ingests the most useful ones
into dedicated `renavon_*` tables so the data lives alongside our scraped
race_results without disturbing the existing schema.

Idempotent: drops + recreates the renavon_* tables on each run.
Tables created:
  renavon_race_results      — 368K rows × 84 cols  (entry_rating, gear, sectionals, breeding, ...)
  renavon_horses            —  17K rows × 32 cols  (current/start/season ratings, breeding, career stats)
  renavon_sectional_times   — 126K rows × 44 cols  (extended pace data per section)
  renavon_dividends         —  varies              (pool payouts — we also have race_dividends)
  renavon_racing_changes    —     443 rows × 18    (scratches, jockey/gear changes)
  renavon_track_conditions  —      26 rows × 10    (penetrometer/hammer readings)
  renavon_jockey_rankings   —    1015 × 20         (per-snapshot rankings)
  renavon_trainer_rankings  —    1091 × 20
  renavon_jackpots          —   49223 × 12         (multi-race pool jackpots)
  renavon_pool_investments  —  306657 × 16         (pool size snapshots)
  renavon_race_catalog      — 1.6M rows × 19       (race catalog 1977-present)

Usage: python3 -m scripts.load_renavon_csvs
"""
import gzip
import os
import sys
import csv

import pandas as pd
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import HorseDatabase

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENAVON_DIR = os.path.join(PROJECT_ROOT, 'renavon_Hongkong_Jockey_Club_races_and_horses')


def identify_csv(path: str) -> str:
    """Inspect first row of a gz CSV and return the dataset name."""
    with gzip.open(path, 'rt') as g:
        reader = csv.reader(g)
        header = next(reader, [])
    cs = set(c.lower() for c in header)

    # 84 cols with finishing_position + entry_rating → main race results
    if 'entry_rating' in cs and 'finishing_position' in cs and len(header) > 80:
        return 'race_results'
    # 32 cols horse profiles
    if 'current_rating' in cs and 'start_of_season_rating' in cs and 'sire' in cs:
        return 'horses'
    # 44 cols sectional only (no entry_rating)
    if 'first_section_position' in cs and 'wire_to_wire' in cs:
        return 'sectional_times'
    # Racing changes: change_type
    if 'change_type' in cs or ('barrier_changes' in cs):
        return 'racing_changes'
    # Pool investments: has pool_id AND investment_amount (1.6M snapshot rows)
    if 'pool_id' in cs and 'investment_amount' in cs:
        return 'pool_investments'
    # Race catalog: has racename + track_going + race-level metadata, not pure track readings
    if 'racename' in cs and 'track_going' in cs:
        return 'race_catalog'
    # Track conditions (pure): just penetrometer/hammer readings per meeting day
    if 'penetrometer_reading' in cs and len(header) <= 12:
        return 'track_conditions'
    # Jockey rankings
    if 'jockey_code' in cs and 'ranking_type' in cs:
        return 'jockey_rankings'
    # Trainer rankings
    if 'trainer_code' in cs and 'ranking_type' in cs:
        return 'trainer_rankings'
    # Jackpots
    if 'jackpot_id' in cs:
        return 'jackpots'
    # Dividends fallback: pool_type with is_multi_race_pool, no race_id-only columns
    if 'pool_type' in cs and 'is_multi_race_pool' in cs:
        return 'dividends'
    return 'unknown'


def load_csv_to_db(conn: sqlite3.Connection, csv_path: str, table_name: str):
    """Load a gzipped CSV into a table. Drops existing table first.

    Uses pandas chunked read for memory efficiency on large files.
    """
    logger.info(f"  Loading {os.path.basename(csv_path)} → {table_name} ...")
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    # First chunk: write with table creation
    first = True
    total = 0
    chunk_size = 50_000
    for chunk in pd.read_csv(csv_path, na_values=['\\N', ''], compression='gzip',
                              chunksize=chunk_size, low_memory=False):
        # Normalize date columns: 'YYYY-MM-DD HH:MM:SS' → 'YYYY/MM/DD'
        for col in chunk.columns:
            if not (col.endswith('_date') or col == 'date'):
                continue
            # Force string dtype before .str accessor — pandas may infer float
            # for chunks where the column is all-NaN
            chunk[col] = (
                chunk[col].astype('string')
                .str.split(' ').str[0]
                .str.replace('-', '/', regex=False)
                .where(lambda s: s.notna() & (s != 'nan') & (s != '<NA>'), None)
            )
        chunk.to_sql(table_name, conn, if_exists='replace' if first else 'append',
                     index=False, chunksize=10_000)
        first = False
        total += len(chunk)
    logger.info(f"    Inserted {total:,} rows")
    return total


def main():
    if not os.path.isdir(RENAVON_DIR):
        logger.error(f"Renavon folder not found: {RENAVON_DIR}")
        sys.exit(1)

    db = HorseDatabase()
    conn = db.conn  # reuse existing SQLite connection

    # Identify each CSV by its columns, not filename
    logger.info("Identifying CSVs by column fingerprint...")
    files = {}
    for f in sorted(os.listdir(RENAVON_DIR)):
        if not f.endswith('.gz'):
            continue
        path = os.path.join(RENAVON_DIR, f)
        kind = identify_csv(path)
        if kind == 'unknown':
            logger.warning(f"  {f}: UNKNOWN — skipping")
            continue
        files[kind] = path
        logger.info(f"  {f}: {kind}")

    # Load each into a renavon_* table
    logger.info(f"\nIngesting {len(files)} datasets into SQLite ...")
    total_rows = 0
    for kind, path in files.items():
        table = f"renavon_{kind}"
        n = load_csv_to_db(conn, path, table)
        total_rows += n

    # Add useful indexes
    logger.info("\nCreating indexes for join performance ...")
    INDEXES = {
        'renavon_race_results': [
            "CREATE INDEX IF NOT EXISTS idx_rrr_date_race ON renavon_race_results(race_date, race_number)",
            "CREATE INDEX IF NOT EXISTS idx_rrr_horse ON renavon_race_results(horse_name)",
            "CREATE INDEX IF NOT EXISTS idx_rrr_horse_id ON renavon_race_results(horse_id)",
        ],
        'renavon_horses': [
            "CREATE INDEX IF NOT EXISTS idx_rh_name ON renavon_horses(horse_name)",
            "CREATE INDEX IF NOT EXISTS idx_rh_id ON renavon_horses(horse_id)",
        ],
        'renavon_sectional_times': [
            # data (4) has race_id but not race_number; index by (date, race_id) instead
            "CREATE INDEX IF NOT EXISTS idx_rst_date ON renavon_sectional_times(race_date)",
            "CREATE INDEX IF NOT EXISTS idx_rst_horse ON renavon_sectional_times(horse_name)",
        ],
    }
    for table, idx_list in INDEXES.items():
        # Only create if the table was actually loaded
        if conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone():
            for sql in idx_list:
                conn.execute(sql)
    conn.commit()

    logger.info(f"\n✓ Loaded {total_rows:,} total rows across {len(files)} renavon_* tables.")

    # Show DB summary
    logger.info("\nFinal DB tables (renavon_* prefix):")
    for r in conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name LIKE 'renavon_%'
        ORDER BY name
    """).fetchall():
        n = conn.execute(f"SELECT COUNT(*) FROM {r[0]}").fetchone()[0]
        logger.info(f"  {r[0]}: {n:,} rows")

    db.close()


if __name__ == '__main__':
    main()
