#!/usr/bin/env python3
"""Race day odds monitor.

Wakes up ~6 minutes before each race, fetches odds, detects market moves,
and regenerates the dashboard with auto-refresh enabled.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fetcher import HKJCDataFetcher
from core.analyzer import CombinedRaceAnalyzer
from data.database import HorseDatabase
from data.odds_tracker import OddsDatabase, OddsMovementAnalyzer
from analytics.pace import PaceAnalyzer
from dashboard.template import generate_html

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MONITOR] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

import config
POLL_INTERVAL = config.POLL_INTERVAL
PRE_RACE_WINDOW = config.PRE_RACE_WINDOW


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
    owns_resources = fetcher is None
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

            race_data = {
                "race_number": race.race_number,
                "venue": race.venue,
                "time": race.time,
                "distance": race.distance,
                "class_type": race.class_type,
                "going": race.going,
                "course": race.course,
                "horses": [],
            }

            race_going = race.going or ''
            race_distance = int(race.distance) if race.distance and str(race.distance).isdigit() else 1200
            odds_summary = odds_db.get_movement_summary(race_date, race.race_number)

            speed_map_input = [
                {'number': h.number, 'name': h.name, 'draw': h.draw,
                 'running_style': getattr(h, 'running_style', ''),
                 'last_6': h.last_6}
                for h in race.horses
            ]
            speed_map = PaceAnalyzer.generate_speed_map(speed_map_input, race_distance)
            race_data["speed_map"] = speed_map

            for i, a in enumerate(analyses):
                h = a.horse
                ml_prob = a.ml_prediction.win_probability if a.ml_prediction else 0
                calibrated_prob = a.combined_probability if a.combined_probability else 0

                going_form = db.get_horse_going_form(h.name, race_going, race_date)
                jh_stats = db.get_jockey_horse_stats(h.jockey, h.name, race_date)
                dist_pb = db.get_distance_best_time(h.name, race_distance, race_date)
                dist_record = db.get_distance_record(race.venue or '', race_distance)

                horse_odds_move = odds_summary.get(h.number, {})
                is_steam, steam_severity, steam_label = OddsMovementAnalyzer.detect_steam_move(
                    horse_odds_move.get('opening', 0), h.win_odds or 0
                )
                is_cold, cold_label = OddsMovementAnalyzer.detect_cold_horse(
                    horse_odds_move.get('opening', 0), h.win_odds or 0, a.rule_score
                )

                race_data["horses"].append({
                    "rank": i + 1,
                    "number": h.number,
                    "name": h.name,
                    "jockey": h.jockey,
                    "trainer": h.trainer,
                    "draw": h.draw,
                    "weight": h.weight,
                    "gear": h.gear,
                    "rating": h.rating,
                    "win_odds": h.win_odds,
                    "place_odds": h.place_odds,
                    "last_6": h.last_6,
                    "age": getattr(h, 'age', ''),
                    "horse_weight": h.horse_weight,
                    "days_since_last_run": h.days_since_last_run,
                    "rating_change": h.rating_change,
                    "rule_score": a.rule_score,
                    "ml_prob": ml_prob,
                    "combined_score": a.value_score,
                    "calibrated_prob": calibrated_prob,
                    "expected_return": a.expected_return,
                    "bet_recommendation": a.bet_recommendation,
                    "confidence": a.combined_confidence,
                    "reasoning": a.reasoning,
                    "going_form": going_form,
                    "jockey_horse_stats": jh_stats,
                    "distance_pb": dist_pb,
                    "distance_record": dist_record,
                    "odds_movement": horse_odds_move,
                    "steam_move": {"is_steam": is_steam, "severity": steam_severity, "label": steam_label},
                    "cold_horse": {"is_cold": is_cold, "label": cold_label},
                })

            all_races.append(race_data)

        if owns_resources:
            db.conn.close()

        date_str = race_date.replace('/', '')
        json_data = json.dumps(all_races)
        json_data = json_data.replace('</script>', r'<\/script>').replace('</Script>', r'<\/Script>')
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        venue = "Happy Valley" if any(r.get("venue", "").startswith("Happy") for r in all_races) else "Sha Tin"

        html = generate_html(json_data, now, race_date, venue, auto_refresh=True)

        html_file = results_dir / f"racing_dashboard_{date_str}.html"
        latest_file = results_dir / "racing_dashboard.html"
        html_file.write_text(html)
        latest_file.write_text(html)

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
            await regenerate_dashboard(race_date, fetcher=fetcher, analyzer=analyzer,
                                       db=db, odds_db=odds_db)

        logger.info("All races processed. Monitor complete.")
    finally:
        db.conn.close()


def main():
    race_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y/%m/%d')
    asyncio.run(run_monitor(race_date))


if __name__ == '__main__':
    main()
