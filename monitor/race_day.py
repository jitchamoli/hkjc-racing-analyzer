#!/usr/bin/env python3
"""Race day odds monitor.

Wakes up ~6 minutes before each race, fetches odds, detects market moves,
and regenerates the dashboard with auto-refresh enabled.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fetcher import HKJCDataFetcher
from core.analyzer import CombinedRaceAnalyzer
from data.database import HorseDatabase
from data.odds_tracker import OddsDatabase
from dashboard.template import generate_html
from dashboard.generator import build_race_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MONITOR] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

import config
POLL_INTERVAL = config.POLL_INTERVAL
PRE_RACE_WINDOW = config.PRE_RACE_WINDOW


def _look_up_latest_odds(odds_db, race_date, race_num, horse_num):
    """Peek the most recent win_odds for a horse from odds history.

    Returns 0 if no row is found or the query fails — callers must
    guard against a 0 return (no comparison if prev or curr is 0).
    """
    try:
        conn = odds_db._get_conn()
        try:
            row = conn.execute(
                "SELECT win_odds FROM odds_snapshots "
                "WHERE race_date = ? AND race_number = ? AND horse_number = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (race_date, race_num, horse_num),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
    except Exception:
        return 0


def parse_race_time(time_str: str, race_date: str) -> datetime:
    """Parse race time string like '13:00' into datetime."""
    try:
        date_part = race_date.replace('/', '-')
        return datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return None


async def fetch_race_times(fetcher: HKJCDataFetcher, race_date: str) -> list:
    """Get race times for all races on a given date.

    Returns list of (race_number, race_time_datetime) sorted by time.
    """
    race_times = []
    for race_no in range(1, 12):
        race = await fetcher.get_race_card(race_date, race_no)
        if not race or not race.horses:
            break
        race_time = parse_race_time(race.time, race_date)
        if race_time:
            race_times.append((race_no, race_time))
            logger.info(f"Race {race_no}: {race.time} ({len(race.horses)} runners)")
        else:
            logger.warning(f"Race {race_no}: no time available")
    return race_times


async def snapshot_odds(fetcher: HKJCDataFetcher, race_date: str, race_no: int,
                        odds_db: OddsDatabase = None):
    """Fetch and store current odds for a race."""
    owns_db = odds_db is None
    odds_db = odds_db or OddsDatabase()
    try:
        race = await fetcher.get_race_card(race_date, race_no)
        if race and race.horses:
            horses = [
                {
                    'number': h.number,
                    'name': h.name,
                    'win_odds': h.win_odds,
                    'place_odds': h.place_odds,
                }
                for h in race.horses
                if h.number > 0 and h.win_odds > 0
            ]
            if horses:
                odds_db.record_snapshot(race_date, race_no, horses, source='monitor')
                logger.info(f"R{race_no}: Snapshotted odds for {len(horses)} horses")
                return len(horses)
    except Exception as e:
        logger.warning(f"R{race_no}: Failed to snapshot odds: {e}")
    return 0


async def regenerate_dashboard(race_date: str, fetcher: HKJCDataFetcher = None,
                               analyzer: CombinedRaceAnalyzer = None,
                               db: HorseDatabase = None, odds_db: OddsDatabase = None):
    """Regenerate dashboard with auto-refresh enabled."""
    owns_db = db is None
    try:
        fetcher = fetcher or HKJCDataFetcher()
        analyzer = analyzer or CombinedRaceAnalyzer()
        db = db or HorseDatabase()
        odds_db = odds_db or OddsDatabase()

        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)

        all_races = []

        for race_no in range(1, 12):
            race = await fetcher.get_race_card(race_date, race_no)
            if not race or not race.horses:
                break
            analyses = analyzer.analyze_race(race)
            race_data = build_race_data(race, analyses, race_date, db, odds_db, analyzer=analyzer)
            all_races.append(race_data)

        if owns_db:
            db.conn.close()

        date_str = race_date.replace('/', '')
        json_data = json.dumps(all_races)
        json_data = json_data.replace('</script>', r'<\/script>').replace('</Script>', r'<\/Script>')
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        venue = "Happy Valley" if any(r.get("venue", "").startswith("Happy") for r in all_races) else "Sha Tin"

        html = generate_html(json_data, now, race_date, venue, auto_refresh=True)

        html_file = results_dir / f"racing_dashboard_{date_str}.html"
        latest_file = results_dir / "racing_dashboard.html"
        # Atomic write so a browser polling the file never sees a half-written
        # HTML during regeneration. os.replace is atomic on POSIX/Windows.
        # try/finally cleans up tmp on write failure so stale .tmp.{pid} files
        # don't accumulate in the results directory.
        for path in (html_file, latest_file):
            tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
            try:
                tmp.write_text(html)
                os.replace(tmp, path)
            except Exception:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise

        logger.info(f"Dashboard regenerated with {len(all_races)} races")

    except Exception as e:
        logger.error(f"Dashboard regeneration failed: {e}")


async def run_monitor(race_date: str):
    """Main monitor loop."""
    logger.info(f"Starting race day monitor for {race_date}")
    logger.info(f"Pre-race window: {PRE_RACE_WINDOW}s, Poll interval: {POLL_INTERVAL}s")

    fetcher = HKJCDataFetcher()
    analyzer = CombinedRaceAnalyzer()
    db = HorseDatabase()
    odds_db = OddsDatabase()

    try:
        # Step 1: Get race times
        race_times = await fetch_race_times(fetcher, race_date)
        if not race_times:
            logger.error("No race times found. Waiting 30 minutes then retrying...")
            await asyncio.sleep(1800)
            race_times = await fetch_race_times(fetcher, race_date)
            if not race_times:
                logger.error("Still no race times. Exiting.")
                return

        logger.info(f"Found {len(race_times)} races")

        # Step 2: Process each race
        for race_no, race_time in race_times:
            wake_time = race_time - timedelta(seconds=PRE_RACE_WINDOW)
            now = datetime.now()

            if now < wake_time:
                sleep_secs = (wake_time - now).total_seconds()
                logger.info(f"Sleeping {sleep_secs:.0f}s until R{race_no} window ({wake_time.strftime('%H:%M')})")
                await asyncio.sleep(sleep_secs)
            elif now > race_time + timedelta(minutes=5):
                logger.info(f"R{race_no} already started 5+ min ago, skipping")
                continue

            # Snapshot and regenerate
            logger.info(f"R{race_no}: Pre-race window active")
            await snapshot_odds(fetcher, race_date, race_no, odds_db=odds_db)

            # Sharp-move auto-rerun trigger (spec §7): if any top-2 pick moved
            # ≥ 25% (steam) or ≥ 40% (cold), force a full re-run.
            date_str = race_date.replace('/', '')
            race_data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'results', f'race_data_{date_str}.json',
            )
            sharp_move = False
            if os.path.exists(race_data_path):
                try:
                    with open(race_data_path) as f:
                        prev = json.load(f)
                    prev_races = prev if isinstance(prev, list) else prev.get('races', [])
                    for race_prev in prev_races:
                        for h in race_prev.get('horses', [])[:2]:  # top-2 only
                            prev_odds = h.get('win_odds', 0)
                            curr_odds = _look_up_latest_odds(
                                odds_db, race_date,
                                race_prev['race_number'], h['number'],
                            )
                            if prev_odds and curr_odds:
                                pct = (curr_odds - prev_odds) / prev_odds
                                if pct <= -0.25 or pct >= 0.40:
                                    sharp_move = True
                                    logger.info(
                                        f"Sharp move on R{race_prev['race_number']} "
                                        f"#{h['number']}: {prev_odds} → {curr_odds} "
                                        f"({pct:+.1%}) — forcing re-run"
                                    )
                                    break
                        if sharp_move:
                            break
                except Exception as e:
                    logger.warning(f"Sharp-move detection failed: {e}")

            if sharp_move:
                # Full re-run in subprocess — re-fetches cards AND odds.
                # Use sys.executable + absolute path so this works regardless
                # of CWD or which python is on PATH.
                from pathlib import Path
                gen_script = Path(__file__).resolve().parent.parent / 'generate_predictions.py'
                subprocess.run(
                    [sys.executable, str(gen_script), race_date],
                    check=False, timeout=300,
                )
            else:
                # Keep the existing call exactly as it was before this patch
                await regenerate_dashboard(race_date, fetcher=fetcher, analyzer=analyzer,
                                           db=db, odds_db=odds_db)

        logger.info("All races processed. Monitor complete.")
    finally:
        db.conn.close()


def main():
    import re
    race_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y/%m/%d')
    if not re.fullmatch(r'\d{4}/\d{2}/\d{2}', race_date):
        print(f"Invalid date format: {race_date}. Expected YYYY/MM/DD")
        sys.exit(1)
    asyncio.run(run_monitor(race_date))


if __name__ == '__main__':
    main()
