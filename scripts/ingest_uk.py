"""Ingest UK race results from theracingapi.com into uk_history.db.

Idempotent — re-ingesting the same date is safe (INSERT OR REPLACE on
race_id, horse_id). Use this each week to keep the DB current for
forward-validation via scripts.validate_uk.

Usage:
    python3 -m scripts.ingest_uk --date 2026-05-17
    python3 -m scripts.ingest_uk --start 2026-05-10 --end 2026-05-17
    python3 -m scripts.ingest_uk --since-last   # from (db_max_date + 1) → today
    python3 -m scripts.ingest_uk --since-last --region gb   # UK only
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

from data.uk_database import UKHorseDatabase
from data.uk_racing_api import MissingCredentialsError, UKRacingAPI

# Locked region — this script is UK/IRE-only by design. Matches the locked
# constants in ingest_au.py / ingest_jp.py. Without it, region=None pulled
# EVERY region on the API plan (FR/AU/JP/IRE) into uk_history.db, contaminating
# UK model training, backtest, and forward-validation. Strict jurisdiction
# isolation is a documented critical rule.
_REGION = "gb"


def _today() -> str:
    return date.today().isoformat()


def _add_days(d: str, n: int) -> str:
    return (datetime.fromisoformat(d).date() + timedelta(days=n)).isoformat()


def _resolve_range(args: argparse.Namespace, db: UKHorseDatabase) -> tuple[str, str]:
    if args.date:
        return args.date, args.date
    if args.start and args.end:
        return args.start, args.end
    if args.since_last:
        summary = db.get_summary()
        last = summary.get("last_date")
        if not last:
            print("DB is empty — use --date or --start/--end for first ingestion.", file=sys.stderr)
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="Single date YYYY-MM-DD")
    p.add_argument("--start", help="Range start YYYY-MM-DD (use with --end)")
    p.add_argument("--end", help="Range end YYYY-MM-DD (use with --start)")
    p.add_argument("--since-last", action="store_true",
                   help="Auto-fill from last date in DB through today")
    p.add_argument("--region", default=_REGION,
                   help=f"API region filter (default '{_REGION}' = UK/IRE only). "
                        "Override only if you know the DB-isolation implications.")
    args = p.parse_args()

    try:
        api = UKRacingAPI()
    except MissingCredentialsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    db = UKHorseDatabase()
    start, end = _resolve_range(args, db)

    print(f"Fetching results {start} → {end}" + (f" (region={args.region})" if args.region else ""))
    races = api.get_results_for_range(start, end, region=args.region)
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
