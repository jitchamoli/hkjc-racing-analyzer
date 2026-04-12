#!/usr/bin/env python3
"""
Historical Results Scraper for HKJC Racing Analyzer.
Fetches past race results from HKJC and ingests them into horse_database.
Uses Playwright for JS-rendered pages.

Usage:
    python3 scrape_history.py                  # Last 10 meeting dates
    python3 scrape_history.py --days 30        # Last 30 days of meetings
    python3 scrape_history.py --date 2026/04/09  # Specific date
"""

import asyncio
import argparse
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


RESULTS_BASE = "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"


async def scrape_meeting_results(race_date: str, max_races: int = 12) -> List[Dict]:
    """Scrape all race results for a single meeting date using Playwright.

    Uses JS-based table extraction since the page renders via Angular.
    Returns list of race dicts compatible with HorseDatabase.ingest_race_day().
    """
    from playwright.async_api import async_playwright

    races = []
    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
        )

        for race_no in range(1, max_races + 1):
            url = f"{RESULTS_BASE}?RaceDate={race_date}&RaceNo={race_no}"
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=45000)
                await page.wait_for_timeout(4000)

                # Extract tables via JavaScript (handles Angular-rendered content)
                table_data = await page.evaluate("""
                    () => {
                        const tables = document.querySelectorAll('table');
                        const result = [];
                        tables.forEach(table => {
                            const rows = table.querySelectorAll('tr');
                            if (rows.length >= 3) {
                                const data = [];
                                rows.forEach(row => {
                                    const cells = Array.from(row.querySelectorAll('td, th'));
                                    data.push(cells.map(c => c.textContent.trim()));
                                });
                                result.push(data);
                            }
                        });
                        return result;
                    }
                """)
            except Exception as e:
                logger.warning(f"  Race {race_no}: failed to load ({e})")
                await page.close()
                break
            await page.close()

            race_data = _parse_tables(table_data, race_date, race_no)
            if not race_data or not race_data.get('entries'):
                logger.info(f"  Race {race_no}: no results (end of card)")
                break

            races.append(race_data)
            winner = next((e['name'] for e in race_data['entries'] if e.get('actual_position') == 1), '?')
            logger.info(
                f"  Race {race_no}: {len(race_data['entries'])} runners, "
                f"winner = {winner}"
            )

        await browser.close()
    except Exception as e:
        logger.error(f"Browser error: {e}")
    finally:
        await pw.stop()

    return races


def _parse_tables(table_data: List[List[List[str]]], race_date: str, race_no: int) -> Optional[Dict]:
    """Parse JS-extracted table data into a race dict.

    table_data is a list of tables, each table is a list of rows,
    each row is a list of cell strings.

    Expected table structure from HKJC:
      Table 0: Race header (class, distance, going, course)
      Table 1: Results (Pla, Horse No, Horse, Jockey, Trainer, Act.Wt, Horse Wt, Dr, LBW, RunPos, Time, Odds)
    """
    if not table_data:
        return None

    venue = "Sha Tin"
    distance = 1200
    going = "Good"
    class_type = ""
    course = ""

    # --- Parse race header (Table 0) ---
    for table in table_data:
        for row in table[:5]:
            full_text = " ".join(row)
            if "Happy Valley" in full_text:
                venue = "Happy Valley"
            if "Sha Tin" in full_text:
                venue = "Sha Tin"
            dm = re.search(r"(\d{4})\s*M", full_text)
            if dm:
                distance = int(dm.group(1))
            for g in ["Good To Firm", "Good To Yielding", "Good", "Yielding To Soft",
                       "Yielding", "Soft", "Heavy", "Wet Fast", "Wet Slow"]:
                if g.lower() in full_text.lower():
                    going = g
                    break
            cm = re.search(r"(Class \d|Griffin|Group \d|Listed)", full_text, re.I)
            if cm:
                class_type = cm.group(0)
            cr = re.search(r'"([A-C](?:\+\d)?)"?\s*Course', full_text)
            if cr:
                course = cr.group(0)

    # --- Parse results table ---
    entries = []
    for table in table_data:
        if len(table) < 3:
            continue

        # Check header row
        header = " ".join(table[0]).lower()
        if "pla" not in header or "horse" not in header:
            continue

        # Columns: Pla(0) Horse No(1) Horse(2) Jockey(3) Trainer(4) Act.Wt(5)
        #          Declar.HorseWt(6) Dr(7) LBW(8) RunningPosition(9) FinishTime(10) WinOdds(11)
        for row in table[1:]:
            if len(row) < 8:
                continue

            # Position
            pos_text = row[0].strip()
            if not re.match(r"^\d{1,2}$", pos_text):
                continue
            pos = int(pos_text)
            if pos < 1 or pos > 24:
                continue

            # Horse number
            num_text = re.sub(r"[^\d]", "", row[1])
            if not num_text:
                continue
            horse_num = int(num_text)

            # Horse name — strip brand number e.g. "ZOUPER FELLOW\xa0(K284)"
            raw_name = row[2].strip() if len(row) > 2 else ""
            name = re.sub(r"\s*\(.*?\)\s*", "", raw_name).strip().upper()
            name = name.replace("\xa0", " ").strip()
            if not name or len(name) < 2:
                continue

            jockey = row[3].strip() if len(row) > 3 else ""
            trainer = row[4].strip() if len(row) > 4 else ""

            # Actual weight (lbs)
            weight = row[5].strip() if len(row) > 5 else ""

            # Draw
            draw = 0
            if len(row) > 7:
                dr_text = re.sub(r"[^\d]", "", row[7])
                if dr_text:
                    draw = int(dr_text)

            # LBW (lengths behind winner) — for margin tracking
            margin = 0.0
            if len(row) > 8:
                lbw = row[8].strip()
                if lbw == "-":
                    margin = 0.0  # winner
                else:
                    # Parse "1-1/4", "3/4", "2", "SH", "N", "HD" etc
                    lbw_clean = lbw.replace("\xa0", " ").strip()
                    margin = _parse_lbw(lbw_clean)

            # Win odds (last column)
            win_odds = 0.0
            if len(row) > 11:
                try:
                    odds_text = row[11].strip()
                    if odds_text and odds_text != "---":
                        win_odds = float(odds_text)
                except (ValueError, IndexError):
                    pass
            # Fallback: try last non-empty column
            if win_odds == 0:
                for cell in reversed(row):
                    cell = cell.strip()
                    try:
                        val = float(cell)
                        if 1.0 < val < 999:
                            win_odds = val
                            break
                    except ValueError:
                        continue

            entries.append({
                "number": horse_num,
                "name": name,
                "jockey": jockey,
                "trainer": trainer,
                "draw": draw,
                "weight": weight,
                "rating": 0.0,  # Not available on results page
                "win_odds": win_odds,
                "actual_position": pos,
                "gear": "",
                "margin": margin,
            })

        if entries:
            break

    if not entries:
        return None

    return {
        "date": race_date,
        "race_no": race_no,
        "venue": venue,
        "distance": distance,
        "going": going,
        "class_type": class_type,
        "course": course,
        "entries": entries,
    }


def _parse_lbw(lbw: str) -> float:
    """Parse HKJC lengths-behind-winner string to float.

    Examples: '-' → 0, '1' → 1.0, '3/4' → 0.75, '1-1/4' → 1.25,
              'SH' → 0.1, 'HD' → 0.2, 'N' → 0.05, 'NK' → 0.3
    """
    if not lbw or lbw == "-":
        return 0.0

    lbw = lbw.strip().upper()

    # Standard abbreviations
    abbrevs = {"SH": 0.1, "SHD": 0.1, "HD": 0.2, "N": 0.05, "NK": 0.3, "NS": 0.05, "NOSE": 0.05}
    if lbw in abbrevs:
        return abbrevs[lbw]

    # "2-1/4" → 2.25
    m = re.match(r"(\d+)\s*-\s*(\d+)/(\d+)", lbw)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))

    # "3/4" → 0.75
    m = re.match(r"(\d+)/(\d+)", lbw)
    if m:
        return int(m.group(1)) / int(m.group(2))

    # Plain number "2"
    try:
        return float(lbw)
    except ValueError:
        return 0.0


async def get_recent_meeting_dates(n_days: int = 60, max_meetings: int = 10) -> List[str]:
    """Get HKJC meeting dates by checking recent dates with Playwright.

    HK racing typically runs Wed + Sat/Sun, sometimes with midweek extras.
    We check all likely dates in the range.
    """
    from playwright.async_api import async_playwright

    dates = []
    today = datetime.now()
    candidates = []

    for i in range(1, n_days):  # Skip today (i=0), start from yesterday
        d = today - timedelta(days=i)
        # HK races most commonly on Wed(2), Sat(5), Sun(6), occasionally others
        if d.weekday() in (2, 5, 6, 0):  # Wed, Sat, Sun, Mon (public holiday racing)
            candidates.append(d.strftime('%Y/%m/%d'))

    if not candidates:
        return []

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )

        for date_str in candidates:
            if len(dates) >= max_meetings:
                break

            page = await context.new_page()
            url = f"{RESULTS_BASE}?RaceDate={date_str}&RaceNo=1"
            try:
                await page.goto(url, wait_until="networkidle", timeout=20000)
                await page.wait_for_timeout(3000)
                text = await page.evaluate("document.body.innerText")
                # Check if it has actual race results (horse names are uppercase)
                has_data = "Jockey" in text and "Trainer" in text and len(text) > 3000
                if has_data:
                    dates.append(date_str)
                    logger.info(f"Found meeting: {date_str}")
            except Exception:
                pass
            await page.close()

        await browser.close()
    except Exception as e:
        logger.error(f"Browser error scanning dates: {e}")
    finally:
        await pw.stop()

    return dates


async def main():
    parser = argparse.ArgumentParser(description="Scrape HKJC historical results")
    parser.add_argument("--date", help="Specific date (YYYY/MM/DD)")
    parser.add_argument("--days", type=int, default=60, help="Look back N days for meetings")
    args = parser.parse_args()

    from horse_database import HorseDatabase
    db = HorseDatabase()

    if args.date:
        dates = [args.date]
    else:
        logger.info(f"Finding meeting dates in last {args.days} days...")
        dates = await get_recent_meeting_dates(args.days)

    if not dates:
        logger.warning("No meeting dates found")
        return

    total_races = 0
    total_runners = 0

    for date_str in dates:
        # Check if we already have this date in DB
        existing = db.conn.execute(
            "SELECT COUNT(*) FROM race_results WHERE race_date = ?", (date_str,)
        ).fetchone()[0]
        if existing > 0:
            logger.info(f"Skipping {date_str} — already have {existing} results in DB")
            continue

        logger.info(f"Scraping results for {date_str}...")
        races = await scrape_meeting_results(date_str)

        if races:
            db.ingest_race_day(races)
            n_runners = sum(len(r['entries']) for r in races)
            total_races += len(races)
            total_runners += n_runners
            logger.info(f"  Ingested {len(races)} races, {n_runners} runners")
        else:
            logger.info(f"  No results found for {date_str}")

        time.sleep(2)  # Be polite between meetings

    # Print summary
    total_db = db.conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]
    total_dates = db.conn.execute("SELECT COUNT(DISTINCT race_date) FROM race_results").fetchone()[0]
    logger.info(f"\nDone! Scraped {total_races} races, {total_runners} runners")
    logger.info(f"DB now has {total_db} results across {total_dates} meeting dates")


if __name__ == "__main__":
    asyncio.run(main())
