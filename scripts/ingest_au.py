"""Ingest AU race results from theracingapi.com into au_history.db.

Mirrors scripts/ingest_uk.py — same API client (data.uk_racing_api.UKRacingAPI
is jurisdiction-agnostic despite its name; only the `region` filter differs),
different DB destination. Forces `region='aus'` on every call so we never
accidentally mix AU into the UK store.

Idempotent (INSERT OR REPLACE on race_id, horse_id).

Usage:
    python3 -m scripts.ingest_au --date 2026-05-17
    python3 -m scripts.ingest_au --start 2026-05-10 --end 2026-05-17
    python3 -m scripts.ingest_au --since-last   # from (db_max_date + 1) → today
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

from data.au_database import AUHorseDatabase
from data.uk_racing_api import MissingCredentialsError, UKRacingAPI


_REGION = "aus"  # locked — this script is AU-only by design


def _today() -> str:
    return date.today().isoformat()


def _add_days(d: str, n: int) -> str:
    return (datetime.fromisoformat(d).date() + timedelta(days=n)).isoformat()


def _resolve_range(args: argparse.Namespace, db: AUHorseDatabase) -> tuple[str, str]:
    if args.date:
        return args.date, args.date
    if args.start and args.end:
        return args.start, args.end
    if args.since_last:
        summary = db.get_summary()
        last = summary.get("last_date")
        if not last:
            print("DB is empty — use --date or --start/--end for first ingestion.",
                  file=sys.stderr)
            sys.exit(2)
        start = _add_days(last, 1)
        end = _today()
        if start > end:
            print(f"DB already current (last={last}, today={end}). Nothing to do.")
            sys.exit(0)
        return start, end
    print("Pass --date, --start/--end, or --since-last.", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="Single date YYYY-MM-DD")
    p.add_argument("--start", help="Range start YYYY-MM-DD (use with --end)")
    p.add_argument("--end", help="Range end YYYY-MM-DD (use with --start)")
    p.add_argument("--since-last", action="store_true",
                   help="Auto-fill from last date in DB through today")
    args = p.parse_args()

    try:
        api = UKRacingAPI()
    except MissingCredentialsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    db = AUHorseDatabase()
    start, end = _resolve_range(args, db)

    print(f"Fetching AU results {start} → {end}")
    races = api.get_results_for_range(start, end, region=_REGION)
    print(f"  API returned {len(races):,} races")

    if not races:
        print("Nothing to ingest.")
        return 0

    written = db.ingest_results(races)
    summary = db.get_summary()
    print(f"  Wrote: {written['races']} races, {written['runners']:,} runners")
    print(f"  DB now: {summary['races']:,} races, {summary['runners']:,} runners "
          f"({summary['first_date']} → {summary['last_date']})")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
