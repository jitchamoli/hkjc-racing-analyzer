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

            # Actual weight carried (lbs) — column 5
            weight = row[5].strip() if len(row) > 5 else ""

            # Declared horse body weight (lbs) — column 6
            horse_weight = 0
            if len(row) > 6:
                hw_text = re.sub(r"[^\d]", "", row[6])
                if hw_text:
                    val = int(hw_text)
                    if 900 <= val <= 1400:  # HK horse body weights range
                        horse_weight = val

            # Draw — column 7
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

            # Running position — column 9 (e.g. "2 1 1" or "10 8 6 3")
            running_position = ""
            if len(row) > 9:
                rp = row[9].strip()
                # Normalize: keep digits and spaces
                rp_clean = re.sub(r"[^\d\s]", " ", rp).strip()
                if rp_clean and any(c.isdigit() for c in rp_clean):
                    running_position = rp_clean

            # Finish time — column 10 (e.g. "0:57.32" or "1:09.84")
            finish_time = ""
            if len(row) > 10:
                ft = row[10].strip()
                if re.match(r"\d+:\d{2}\.\d{1,2}", ft):
                    finish_time = ft

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
                "weight": weight,           # Act.Wt (carried weight, lbs)
                "horse_weight": horse_weight,  # Declar.HorseWt (body weight, lbs)
                "rating": 0.0,  # Not available on results page
                "win_odds": win_odds,
                "actual_position": pos,
                "gear": "",
                "margin": margin,
                "running_position": running_position,
                "finish_time": finish_time,
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


async def scrape_jockey_rankings() -> List[Dict]:
    """Scrape full-season jockey rankings from HKJC (JS-rendered page).

    Returns list of dicts: {name, rides, wins, seconds, thirds, fourths,
                            win_rate, place_rate, stakes}.
    """
    from playwright.async_api import async_playwright

    url = "https://racing.hkjc.com/racing/information/English/Jockey/JockeyRanking.aspx"
    results = []

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            viewport={"width": 1440, "height": 900},
        )
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(5000)

        table_data = await page.evaluate("""
            () => {
                const tables = document.querySelectorAll('table');
                for (const table of tables) {
                    const rows = table.querySelectorAll('tr');
                    if (rows.length < 5) continue;
                    const text = table.textContent.toLowerCase();
                    if (!text.includes('jockey') || !text.includes('win')) continue;
                    const data = [];
                    for (const row of rows) {
                        const cells = Array.from(row.querySelectorAll('td, th'));
                        data.push(cells.map(c => c.textContent.trim()));
                    }
                    return data;
                }
                return [];
            }
        """)

        if not table_data or len(table_data) < 3:
            logger.warning("No jockey ranking table found")
            await browser.close()
            await pw.stop()
            return results

        # Header rows: Row 0 = "Jockeys' Ranking", Row 1 = column names
        # Data starts at row 2
        # Columns: Jockey(0) | Win(1) | 2nd(2) | 3rd(3) | 4th(4) | 5th(5) | Total Rides(6) | Stakes(7)
        for row in table_data[2:]:
            if len(row) < 6:
                continue
            name = row[0].strip()
            if not name or name.lower() in ('total', '', 'jockey'):
                continue
            try:
                wins = int(re.sub(r'[^\d]', '', row[1]) or 0)
                seconds = int(re.sub(r'[^\d]', '', row[2]) or 0)
                thirds = int(re.sub(r'[^\d]', '', row[3]) or 0)
                rides = int(re.sub(r'[^\d]', '', row[6]) or 0) if len(row) > 6 else 0
                if rides == 0:
                    rides = wins + seconds + thirds
                if rides > 0:
                    results.append({
                        'name': name,
                        'rides': rides,
                        'wins': wins,
                        'places': wins + seconds + thirds,
                        'win_rate': round(wins / rides, 4) if rides else 0,
                        'place_rate': round((wins + seconds + thirds) / rides, 4) if rides else 0,
                    })
            except (ValueError, IndexError):
                continue

        await browser.close()
    except Exception as e:
        logger.error(f"Jockey ranking scrape failed: {e}")
    finally:
        await pw.stop()

    logger.info(f"Scraped {len(results)} jockey rankings")
    return results


async def scrape_trainer_rankings() -> List[Dict]:
    """Scrape full-season trainer rankings from HKJC (JS-rendered page)."""
    from playwright.async_api import async_playwright

    url = "https://racing.hkjc.com/racing/information/English/Trainers/TrainerRanking.aspx"
    results = []

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            viewport={"width": 1440, "height": 900},
        )
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(5000)

        table_data = await page.evaluate("""
            () => {
                const tables = document.querySelectorAll('table');
                for (const table of tables) {
                    const rows = table.querySelectorAll('tr');
                    if (rows.length < 5) continue;
                    const text = table.textContent.toLowerCase();
                    if (!text.includes('trainer') || !text.includes('win')) continue;
                    const data = [];
                    for (const row of rows) {
                        const cells = Array.from(row.querySelectorAll('td, th'));
                        data.push(cells.map(c => c.textContent.trim()));
                    }
                    return data;
                }
                return [];
            }
        """)

        if not table_data or len(table_data) < 3:
            logger.warning("No trainer ranking table found")
            await browser.close()
            await pw.stop()
            return results

        # Same layout as jockey: Row 0 title, Row 1 column names, Row 2+ data
        # Columns: Trainer(0) | Win(1) | 2nd(2) | 3rd(3) | 4th(4) | 5th(5) | Total Runners(6) | Stakes(7)
        for row in table_data[2:]:
            if len(row) < 6:
                continue
            name = row[0].strip()
            if not name or name.lower() in ('total', '', 'trainer'):
                continue
            try:
                wins = int(re.sub(r'[^\d]', '', row[1]) or 0)
                seconds = int(re.sub(r'[^\d]', '', row[2]) or 0)
                thirds = int(re.sub(r'[^\d]', '', row[3]) or 0)
                runners = int(re.sub(r'[^\d]', '', row[6]) or 0) if len(row) > 6 else 0
                if runners == 0:
                    runners = wins + seconds + thirds
                if runners > 0:
                    results.append({
                        'name': name,
                        'runners': runners,
                        'wins': wins,
                        'places': wins + seconds + thirds,
                        'win_rate': round(wins / runners, 4) if runners else 0,
                        'place_rate': round((wins + seconds + thirds) / runners, 4) if runners else 0,
                    })
            except (ValueError, IndexError):
                continue

        await browser.close()
    except Exception as e:
        logger.error(f"Trainer ranking scrape failed: {e}")
    finally:
        await pw.stop()

    logger.info(f"Scraped {len(results)} trainer rankings")
    return results


async def _scrape_and_store_rankings(db):
    """Scrape HKJC rankings and merge into DB stats tables.

    HKJC full-season stats are more accurate than our DB-only stats
    (we only have ~10 meets vs ~80 in a full season). We use HKJC
    numbers when they have more rides/runners than our local data.
    """
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d')

    jockeys = await scrape_jockey_rankings()
    if jockeys:
        c = db.conn.cursor()
        for j in jockeys:
            # Only update if HKJC has more rides than our local data
            c.execute("SELECT rides FROM jockey_stats WHERE name = ?", (j['name'],))
            row = c.fetchone()
            local_rides = row[0] if row else 0
            if j['rides'] >= local_rides:
                c.execute("""
                    INSERT INTO jockey_stats (name, season, rides, wins, places, win_rate, place_rate, last_updated)
                    VALUES (?, '2025-26', ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        rides=excluded.rides, wins=excluded.wins, places=excluded.places,
                        win_rate=excluded.win_rate, place_rate=excluded.place_rate,
                        last_updated=excluded.last_updated
                """, (j['name'], j['rides'], j['wins'], j['places'],
                      j['win_rate'], j['place_rate'], now))
        db.conn.commit()
        logger.info(f"Updated {len(jockeys)} jockey rankings in DB")
    else:
        logger.warning("No jockey rankings scraped")

    trainers = await scrape_trainer_rankings()
    if trainers:
        c = db.conn.cursor()
        for t in trainers:
            c.execute("SELECT runners FROM trainer_stats WHERE name = ?", (t['name'],))
            row = c.fetchone()
            local_runners = row[0] if row else 0
            if t['runners'] >= local_runners:
                c.execute("""
                    INSERT INTO trainer_stats (name, season, runners, wins, places, win_rate, place_rate, last_updated)
                    VALUES (?, '2025-26', ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        runners=excluded.runners, wins=excluded.wins, places=excluded.places,
                        win_rate=excluded.win_rate, place_rate=excluded.place_rate,
                        last_updated=excluded.last_updated
                """, (t['name'], t['runners'], t['wins'], t['places'],
                      t['win_rate'], t['place_rate'], now))
        db.conn.commit()
        logger.info(f"Updated {len(trainers)} trainer rankings in DB")
    else:
        logger.warning("No trainer rankings scraped")


async def main():
    parser = argparse.ArgumentParser(description="Scrape HKJC historical results")
    parser.add_argument("--date", help="Specific date (YYYY/MM/DD)")
    parser.add_argument("--days", type=int, default=60, help="Look back N days for meetings")
    parser.add_argument("--rankings", action="store_true", help="Scrape jockey/trainer season rankings")
    args = parser.parse_args()

    from horse_database import HorseDatabase
    db = HorseDatabase()

    if args.rankings:
        await _scrape_and_store_rankings(db)
        db.close()
        return

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
