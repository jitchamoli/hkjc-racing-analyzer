#!/usr/bin/env python3
"""Refresh tracking_data/backup/db_core_dump.sql.gz from the live DB.

Dumps only the tables that contain derived/ingested data not recoverable
from the committed Renavon CSVs (which cover renavon_* tables). Run after
any meaningful DB change (post-race ingest, retrain, dividend backfill)
and commit the updated dump file so a future DB loss is recoverable.

Usage:
  python3 -m scripts.dump_db
"""
import gzip
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUMP_PATH = PROJECT_ROOT / 'tracking_data' / 'backup' / 'db_core_dump.sql.gz'

CORE_TABLES = (
    'race_results', 'horses', 'jockey_stats',
    'trainer_stats', 'race_dividends', 'prediction_log',
)


def main():
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)

    db_path = config.DB_PATH
    if not db_path.exists() or db_path.stat().st_size == 0:
        logger.error(f"DB missing or empty: {db_path}")
        sys.exit(1)

    # Use sqlite3 CLI for the dump — its .dump output is the canonical
    # restore format and round-trips cleanly through executescript.
    cmd = ['sqlite3', str(db_path)]
    script = '\n'.join(f'.dump {t}' for t in CORE_TABLES)
    proc = subprocess.run(cmd, input=script, capture_output=True, text=True, check=True)

    raw = proc.stdout.encode('utf-8')
    # mtime=0 makes the gzip output byte-identical when content is identical,
    # so `git diff --quiet` correctly reports no-op when DB hasn't changed.
    with gzip.GzipFile(filename=str(DUMP_PATH), mode='wb',
                       compresslevel=9, mtime=0) as f:
        f.write(raw)

    sz = DUMP_PATH.stat().st_size / (1024 ** 2)
    logger.info(f"✓ Wrote {DUMP_PATH.name} — {sz:.2f} MB")
    # Per-table row counts so the commit message can include them
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    for t in CORE_TABLES:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        logger.info(f"  {t}: {n:,} rows")
    conn.close()
    logger.info("\nNext step: git add tracking_data/backup/db_core_dump.sql.gz && git commit")


if __name__ == '__main__':
    main()
