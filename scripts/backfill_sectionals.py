"""Backfill race_results.sectional_times from HKJC sectional pages (Path B, Phase 2).

Iterates over (race_date, race_number) pairs already present in race_results,
fetches each race's HKJC sectional page, and UPDATEs sectional_times by
horse_number.

Idempotent + resumable: a race where any row already has sectional_times is
skipped (unless --force), so an interrupted run just picks up where it left
off. Fetches run concurrently (semaphore-limited) and the plain-requests
sectional pages need no Playwright.

Usage:
  python3 -m scripts.backfill_sectionals                     # everything missing
  python3 -m scripts.backfill_sectionals --from 2024/05/01   # date window
  python3 -m scripts.backfill_sectionals --from 2024/05/01 --to 2025/01/01
  python3 -m scripts.backfill_sectionals --force             # re-fetch even if present
  python3 -m scripts.backfill_sectionals --concurrency 8     # tune parallelism
"""
import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

import config
from data.sectional_scraper import fetch_sectional_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHUNK = 80  # races fetched + committed per batch (bounds memory, keeps it resumable)


def _races_to_process(conn, from_date, to_date, force):
    q = "SELECT DISTINCT race_date, race_number FROM race_results WHERE 1=1"
    params = []
    if from_date:
        q += " AND race_date >= ?"
        params.append(from_date)
    if to_date:
        q += " AND race_date <= ?"
        params.append(to_date)
    if not force:
        # Skip any race that already has at least one sectional_times row.
        q += (" AND race_date || '#' || race_number NOT IN "
              "(SELECT DISTINCT race_date || '#' || race_number FROM race_results "
              "WHERE sectional_times != '')")
    q += " ORDER BY race_date, race_number"
    return [(r[0], r[1]) for r in conn.execute(q, params).fetchall()]


def _sectional_string(horse: dict) -> str:
    """Per-section times joined, e.g. '27.84 23.69 24.22 24.10'."""
    times = [s["time"] for s in horse.get("sections", [])]
    return " ".join(f"{t:.2f}" for t in times) if times else ""


async def backfill(from_date=None, to_date=None, force=False, concurrency=8):
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    races = _races_to_process(conn, from_date, to_date, force)
    total = len(races)
    if total == 0:
        logger.info("Nothing to backfill — all races in range already have sectionals.")
        conn.close()
        return
    logger.info(
        f"Backfilling {total:,} races"
        f"{f' from {from_date}' if from_date else ''}"
        f"{f' to {to_date}' if to_date else ''}"
        f"  (concurrency={concurrency})"
    )

    sem = asyncio.Semaphore(concurrency)
    session = requests.Session()
    t0 = time.time()

    stats = {"races_ok": 0, "races_empty": 0, "horses_updated": 0, "horses_unmatched": 0}

    async def fetch(race_date, race_no):
        async with sem:
            data = await asyncio.to_thread(
                fetch_sectional_data, race_date, race_no, session
            )
        return race_date, race_no, data

    for start in range(0, total, CHUNK):
        chunk = races[start:start + CHUNK]
        fetched = await asyncio.gather(*(fetch(d, n) for d, n in chunk))

        for race_date, race_no, data in fetched:
            if not data or not data.get("horses"):
                stats["races_empty"] += 1
                continue
            stats["races_ok"] += 1
            for h in data["horses"]:
                st = _sectional_string(h)
                if not st:
                    continue
                cur = conn.execute(
                    "UPDATE race_results SET sectional_times = ? "
                    "WHERE race_date = ? AND race_number = ? AND horse_number = ?",
                    (st, race_date, race_no, h["horse_no"]),
                )
                if cur.rowcount > 0:
                    stats["horses_updated"] += cur.rowcount
                else:
                    stats["horses_unmatched"] += 1
        conn.commit()

        done = min(start + CHUNK, total)
        rate = done / (time.time() - t0)
        eta = (total - done) / rate if rate > 0 else 0
        logger.info(
            f"  {done:,}/{total:,} races  |  "
            f"ok={stats['races_ok']} empty={stats['races_empty']}  |  "
            f"horses updated={stats['horses_updated']:,}  |  "
            f"{rate:.1f} races/s  ETA {eta/60:.1f}m"
        )

    conn.close()
    elapsed = time.time() - t0
    logger.info(
        f"\nDone in {elapsed/60:.1f}m — "
        f"{stats['races_ok']:,} races with sectionals, "
        f"{stats['races_empty']:,} empty (no page), "
        f"{stats['horses_updated']:,} horse rows updated, "
        f"{stats['horses_unmatched']:,} unmatched (horse_no not in race_results)."
    )
    logger.info("Reminder: run `python3 -m scripts.dump_db` to refresh the SQL backup.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="from_date", default=None,
                    help="Start date YYYY/MM/DD (inclusive)")
    ap.add_argument("--to", dest="to_date", default=None,
                    help="End date YYYY/MM/DD (inclusive)")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch even races that already have sectional_times")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Max concurrent sectional-page fetches (default 8)")
    args = ap.parse_args()
    asyncio.run(backfill(args.from_date, args.to_date, args.force, args.concurrency))
    return 0


if __name__ == "__main__":
    sys.exit(main())
