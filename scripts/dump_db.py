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
    # GPT-5.5 R13 #4 2026-05-25 — frozen ticket audit trail. validate_hk
    # --replay scores from race_bet_ticket; restoring without it loses
    # the historical staked record and silently zeroes future replay
    # numbers on the restored DB.
    'race_bet_ticket',
    # audit 2026-07-12 — 8,460 rows the dump did not cover. horse_history.db is
    # gitignored and Renavon (frozen at 2026/04/22) does not carry trials, so
    # restore_db was silently recreating this table EMPTY: unrecoverable for any
    # date that has aged out of the BTResult.aspx dropdown. The feature is killed
    # but CLAUDE.md keeps the infra accruing via `scrape_trials --since-last`.
    'barrier_trials',
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

    raw_text = proc.stdout

    # Coverage guard. `.dump <table>` on a table that doesn't exist (renamed,
    # dropped, typo'd) writes an empty BEGIN/COMMIT and exits 0 — so check=True
    # above can NEVER catch it. barrier_trials was absent from CORE_TABLES for
    # months and restore_db silently recreated it empty (audit 2026-07-12).
    # auto_dump.sh commits + pushes whenever the bytes change, so a silently
    # truncated dump would overwrite the last good backup. Verify every table is
    # really in there, and that a non-empty table produced INSERTs.
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    counts = {}
    missing = []
    for t in CORE_TABLES:
        counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if f'CREATE TABLE {t}' not in raw_text and f'CREATE TABLE "{t}"' not in raw_text:
            missing.append(f"{t} (no CREATE TABLE in dump)")
        elif counts[t] > 0 and f'INSERT INTO {t} ' not in raw_text \
                and f'INSERT INTO "{t}"' not in raw_text:
            missing.append(f"{t} ({counts[t]:,} rows in DB but no INSERTs in dump)")
    conn.close()
    if missing:
        logger.error("Refusing to write a truncated dump — tables not covered:")
        for m in missing:
            logger.error(f"  ✗ {m}")
        sys.exit(1)

    raw = raw_text.encode('utf-8')
    # mtime=0 makes the gzip output byte-identical when content is identical,
    # so `git diff --quiet` correctly reports no-op when DB hasn't changed.
    with gzip.GzipFile(filename=str(DUMP_PATH), mode='wb',
                       compresslevel=9, mtime=0) as f:
        f.write(raw)

    sz = DUMP_PATH.stat().st_size / (1024 ** 2)
    logger.info(f"✓ Wrote {DUMP_PATH.name} — {sz:.2f} MB")
    for t in CORE_TABLES:
        logger.info(f"  {t}: {counts[t]:,} rows")
    logger.info("\nNext step: git add tracking_data/backup/db_core_dump.sql.gz && git commit")


if __name__ == '__main__':
    main()
