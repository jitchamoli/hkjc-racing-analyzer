#!/usr/bin/env python3
"""Rebuild tracking_data/horse_history.db from the core dump and local sources.

The rebuild uses two sources:
  1. renavon_Hongkong_Jockey_Club_races_and_horses/  → renavon_* tables
     (locally supplied export; not included in this repository)
  2. tracking_data/backup/db_core_dump.sql.gz       → core tables
     (race_results 5y window + horses + jockey/trainer_stats + race_dividends
     + prediction_log; refreshed via `python3 -m scripts.dump_db`)

Use after any local DB loss. Idempotent — safe to re-run; existing DB is
moved aside before rebuild so nothing is silently overwritten.

Usage:
  python3 -m scripts.restore_db                  # full rebuild; requires local Renavon exports
  python3 -m scripts.restore_db --skip-renavon   # only restore core dump
                                                  (faster if renavon_* tables
                                                  are already loaded)
"""
import argparse
import gzip
import logging
import os
import shutil
import subprocess
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUMP_PATH = PROJECT_ROOT / 'tracking_data' / 'backup' / 'db_core_dump.sql.gz'
RENAVON_DIR = PROJECT_ROOT / 'renavon_Hongkong_Jockey_Club_races_and_horses'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-renavon', action='store_true',
                    help='Skip Renavon CSV reload (use if renavon_* tables are already populated)')
    args = ap.parse_args()

    db_path = config.DB_PATH

    # Move aside any existing DB so we don't silently merge into a partial state.
    if db_path.exists() and db_path.stat().st_size > 0:
        backup = db_path.with_suffix(f'.db.pre-restore-{datetime.now():%Y%m%d-%H%M%S}')
        shutil.move(str(db_path), str(backup))
        logger.info(f"Moved existing DB aside → {backup.name}")

    # Init schema via HorseDatabase (CREATE TABLE IF NOT EXISTS for our tables)
    from data.database import HorseDatabase
    HorseDatabase().close()
    logger.info(f"Initialized fresh schema at {db_path}")

    # Restore core tables from SQL dump
    if not DUMP_PATH.exists():
        logger.error(f"Core dump missing: {DUMP_PATH}")
        sys.exit(1)
    logger.info(f"Restoring core tables from {DUMP_PATH.name} ...")
    conn = sqlite3.connect(str(db_path))
    try:
        # Drop the empty CREATE-TABLE-from-init versions; the dump's CREATE
        # statements will recreate them with the correct schema + indexes.
        # Sourced from dump_db.CORE_TABLES — a local copy of this list is how
        # barrier_trials came to be dumped-but-not-dropped-and-restored (audit
        # 2026-07-12). The two lists must never diverge again.
        from scripts.dump_db import CORE_TABLES
        for tbl in CORE_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        with gzip.open(DUMP_PATH, 'rt') as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()
    # Quick row-count summary
    conn = sqlite3.connect(str(db_path))
    for tbl in ('race_results', 'horses', 'jockey_stats',
                'trainer_stats', 'race_dividends', 'prediction_log'):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        logger.info(f"  {tbl}: {n:,} rows")
    conn.close()

    # Reload Renavon
    if not args.skip_renavon:
        if not RENAVON_DIR.exists():
            logger.warning(f"Renavon dir missing: {RENAVON_DIR} — skipping renavon_* reload")
        else:
            logger.info("Reloading Renavon CSVs (this takes 1-2 min) ...")
            subprocess.run(
                [sys.executable, '-m', 'scripts.load_renavon_csvs'],
                check=True, cwd=str(PROJECT_ROOT),
            )

    # Final size report
    sz = db_path.stat().st_size / (1024 ** 2)
    logger.info(f"\n✓ Restore complete. DB size: {sz:.1f} MB")
    logger.info(f"  Refresh the dump after future DB updates: python3 -m scripts.dump_db")


if __name__ == '__main__':
    main()
