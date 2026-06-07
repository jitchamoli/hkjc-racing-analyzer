#!/usr/bin/env python3
"""Scrape HKJC barrier-trial results into the barrier_trials table.

Source: Trackwork/BTResult.aspx (server-rendered HTML, full historical archive
via the site's date dropdown). Idempotent — INSERT OR REPLACE on
(trial_date, heat, horse_name), so re-running any date is safe.

Usage:
    python3 -m scripts.scrape_trials --date 2026/05/29
    python3 -m scripts.scrape_trials --backfill      # every date in the dropdown
    python3 -m scripts.scrape_trials --since-last     # dropdown dates newer than DB max
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup

from core.fetcher import HKJCApiClient, HKJCParser
from data.database import HorseDatabase

_BT_URL = "https://racing.hkjc.com/racing/information/English/Trackwork/BTResult.aspx"
_UA = {"User-Agent": "Mozilla/5.0"}


def site_trial_dates() -> list:
    """All trial dates in the BTResult dropdown -> YYYY/MM/DD, newest first."""
    html = requests.get(_BT_URL, timeout=25, headers=_UA).text
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for o in soup.find_all("option"):
        m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", (o.get("value") or "").strip())
        if m:
            d, mo, y = m.groups()
            iso = f"{y}/{mo}/{d}"
            if iso not in seen:
                seen.add(iso)
                out.append(iso)
    return out


def _detect_venue(html: str) -> str:
    low = (html or "").lower()
    if "happy valley" in low:
        return "HV"
    if "conghua" in low:
        return "Conghua"
    if "sha tin" in low:
        return "ST"
    return ""


def scrape_one(api: HKJCApiClient, db: HorseDatabase, date: str) -> int:
    html = api.fetch_barrier_trial_html(date)
    if not html:
        return 0
    rows = HKJCParser.parse_barrier_trials(html, trial_date=date)
    return db.ingest_barrier_trials(rows, venue=_detect_venue(html))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="Single date YYYY/MM/DD")
    p.add_argument("--backfill", action="store_true", help="Every date in the site dropdown")
    p.add_argument("--since-last", action="store_true", help="Dropdown dates newer than DB max")
    p.add_argument("--sleep", type=float, default=0.4, help="Politeness delay between dates (s)")
    args = p.parse_args()

    db = HorseDatabase()
    api = HKJCApiClient()

    if args.date:
        dates = [args.date]
    elif args.backfill or args.since_last:
        dates = site_trial_dates()
        if args.since_last:
            last = db.conn.execute("SELECT MAX(trial_date) FROM barrier_trials").fetchone()[0]
            if last:
                dates = [d for d in dates if d > last]   # YYYY/MM/DD sorts chronologically
    else:
        print("Pass --date, --backfill, or --since-last", file=sys.stderr)
        return 2

    print(f"Scraping {len(dates)} trial date(s)...")
    total_rows, dates_with_data = 0, 0
    for i, d in enumerate(dates, 1):
        try:
            n = scrape_one(api, db, d)
        except Exception as e:
            print(f"  [{i}/{len(dates)}] {d}: ERROR {type(e).__name__}: {e}")
            continue
        total_rows += n
        dates_with_data += 1 if n else 0
        if i % 25 == 0 or n == 0 or len(dates) <= 5:
            print(f"  [{i}/{len(dates)}] {d}: {n} runners")
        if i < len(dates):
            time.sleep(args.sleep)

    cnt = db.conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT trial_date) FROM barrier_trials").fetchone()
    print(f"Done. Wrote {total_rows} runners across {dates_with_data} dates this run.")
    print(f"DB now: {cnt[0]} trial runners across {cnt[1]} trial dates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
