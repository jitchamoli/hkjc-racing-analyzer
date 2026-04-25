#!/usr/bin/env python3
"""Retroactive card-field backfill.

Walks every results/race_data_*.json file and fills rating / last_6 / gear /
age / rating_change into existing race_results rows by (race_date,
race_number, horse_name). Idempotent — skips fields that already have data.

Usage: python3 -m scripts.backfill_card_fields
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import HorseDatabase

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


DATE_RE = re.compile(r'race_data_(\d{8})(?:_with_results)?\.json$')


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
    )

    # Collect one file per date — prefer _with_results.json over the base file
    file_by_date = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'race_data_*.json'))):
        m = DATE_RE.search(os.path.basename(path))
        if not m:
            continue
        date_str = m.group(1)
        is_with_results = '_with_results' in os.path.basename(path)
        existing = file_by_date.get(date_str)
        if not existing or is_with_results:
            file_by_date[date_str] = path

    if not file_by_date:
        logger.info("No race_data_*.json files found.")
        return

    logger.info(f"Found {len(file_by_date)} dated prediction files.")

    db = HorseDatabase()
    total_updates = 0
    file_count = 0

    for date_str, path in sorted(file_by_date.items()):
        race_date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"  Skip {path}: {e}")
            continue

        races = data if isinstance(data, list) else data.get('races', [])
        if not races:
            continue

        n = db.backfill_card_fields(race_date, races)
        if n > 0:
            logger.info(f"  {race_date}: +{n} row updates  (from {os.path.basename(path)})")
            total_updates += n
            file_count += 1

    db.close()
    logger.info(f"\nDone. {total_updates} row updates across {file_count} dates.")


if __name__ == '__main__':
    main()
