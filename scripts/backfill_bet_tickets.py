"""One-shot backfill: read results/race_data_*.json files and seed the
`race_bet_ticket` table with any bet_ticket dicts present.

Added 2026-05-24 alongside Finding #2 fix. Going forward, `log_predictions`
writes the ticket directly, so this is only needed for files that were
generated before the persist code was deployed.

Usage:
    python3 -m scripts.backfill_bet_tickets         # scan results/ dir
    python3 -m scripts.backfill_bet_tickets --date 2026/05/24
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sqlite3
import sys
from datetime import datetime

import config


def _race_data_path(date: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        f"race_data_{date.replace('/', '')}.json",
    )


def _candidates(date: str | None) -> list[str]:
    if date:
        with_results = _race_data_path(date).replace('.json', '_with_results.json')
        base = _race_data_path(date)
        return [p for p in (with_results, base) if os.path.exists(p)]
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    # Prefer _with_results variants where present.
    files = sorted(glob.glob(os.path.join(results_dir, "race_data_2*.json")))
    return [f for f in files if "shadow" not in f]


def _date_from_path(path: str) -> str:
    name = os.path.basename(path)
    digits = name.replace("race_data_", "").split(".")[0].split("_")[0]
    return f"{digits[:4]}/{digits[4:6]}/{digits[6:8]}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="YYYY/MM/DD — backfill a single meeting only")
    args = p.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    total_tickets = 0
    for path in _candidates(args.date):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {path}: {e}")
            continue
        races = data.get('races', data) if isinstance(data, dict) else data
        date = _date_from_path(path)
        n = 0
        for race in races:
            if not isinstance(race, dict):
                continue
            ticket = race.get('bet_ticket')
            if not ticket:
                continue
            rno = race.get('race_number')
            if rno is None:
                continue
            # Mirror the same post-results freeze guard as log_predictions
            # (GPT-5.5 Finding #3 2026-05-24). A backfill must not silently
            # overwrite a frozen ticket that was already validated.
            results_filled = conn.execute(
                "SELECT 1 FROM prediction_log WHERE race_date=? AND race_number=? "
                "AND COALESCE(results_filled, 0) = 1 LIMIT 1",
                (date, rno),
            ).fetchone()
            if results_filled:
                continue
            conn.execute(
                "INSERT INTO race_bet_ticket (race_date, race_number, ticket_json, logged_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(race_date, race_number) DO UPDATE SET "
                "ticket_json=excluded.ticket_json, logged_at=excluded.logged_at",
                (date, rno, json.dumps(ticket), now),
            )
            n += 1
        if n:
            print(f"  {date}: {n} tickets from {os.path.basename(path)}")
            total_tickets += n
    conn.commit()
    conn.close()
    print(f"\ntotal: {total_tickets} bet_ticket rows upserted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
