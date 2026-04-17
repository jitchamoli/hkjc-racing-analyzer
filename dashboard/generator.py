#!/usr/bin/env python3
"""
Generate race predictions and dashboard - non-interactive
"""

import asyncio
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hkjc_data_fetcher import HKJCDataFetcher
from hkjc_racing_analyzer import CombinedRaceAnalyzer
from dashboard.template import generate_html
from data.database import HorseDatabase
from data.odds_tracker import OddsDatabase, OddsMovementAnalyzer
from analytics.pace import PaceAnalyzer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    race_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y/%m/%d')

    logger.info(f"Generating predictions for {race_date}")

    fetcher = HKJCDataFetcher()
    analyzer = CombinedRaceAnalyzer()

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_races = []

    db = HorseDatabase()
    odds_db = OddsDatabase()

    for race_no in range(1, 12):
        logger.info(f"Fetching Race {race_no}...")

        race = await fetcher.get_race_card(race_date, race_no)

        if not race or not race.horses:
            logger.info(f"No data for Race {race_no}, stopping")
            break

        analyses = analyzer.analyze_race(race)

        # Enhanced analytics data
        race_going = race.going or ''
        race_distance = int(race.distance) if race.distance and str(race.distance).isdigit() else 1200

        # Speed map
        speed_map_input = [
            {'number': h.number, 'name': h.name, 'draw': h.draw,
             'running_style': h.running_style if hasattr(h, 'running_style') else '',
             'last_6': h.last_6}
            for h in race.horses
        ]
        speed_map = PaceAnalyzer.generate_speed_map(speed_map_input, race_distance)

        # Odds movement
        odds_summary = odds_db.get_movement_summary(race_date, race.race_number)

        race_data = {
            "race_number": race.race_number,
            "venue": race.venue,
            "time": race.time,
            "distance": race.distance,
            "class_type": race.class_type,
            "going": race.going,
            "course": race.course,
            "horses": []
        }

        for i, a in enumerate(analyses):
            h = a.horse
            ml_prob = a.ml_prediction.win_probability if a.ml_prediction else 0
            calibrated_prob = a.combined_probability if a.combined_probability else 0

            # Enhanced analytics
            going_form = db.get_horse_going_form(h.name, race_going, race_date)
            jh_stats = db.get_jockey_horse_stats(h.jockey, h.name, race_date)
            dist_pb = db.get_distance_best_time(h.name, race_distance, race_date)
            dist_record = db.get_distance_record(race.venue or '', race_distance)

            # Odds movement for this horse
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
                "age": h.age if hasattr(h, 'age') else "",
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

        race_data["speed_map"] = speed_map
        all_races.append(race_data)
        logger.info(f"  Race {race_no}: {len(race.horses)} horses, bet: {analyses[0].bet_recommendation if analyses else 'N/A'}")

    db.conn.close()

    date_str = race_date.replace('/', '')
    json_file = os.path.join(results_dir, f"race_data_{date_str}.json")

    with open(json_file, 'w') as f:
        json.dump(all_races, f, indent=2)

    logger.info(f"Saved predictions to {json_file}")
    logger.info(f"Total races: {len(all_races)}")

    json_data = json.dumps(all_races)
    # Prevent </script> in JSON from breaking the <script> block (XSS defense)
    json_data = json_data.replace('</script>', r'<\/script>').replace('</Script>', r'<\/Script>')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Detect venue from race data
    venue = "Happy Valley" if any(r.get("venue", "").startswith("Happy") for r in all_races) else "Sha Tin"

    html = generate_html(json_data, now, race_date, venue)

    html_file = os.path.join(results_dir, f"racing_dashboard_{date_str}.html")
    with open(html_file, 'w') as f:
        f.write(html)

    # Always write to a fixed URL so you can bookmark one path
    latest_file = os.path.join(results_dir, "racing_dashboard.html")
    with open(latest_file, 'w') as f:
        f.write(html)

    logger.info(f"Saved dashboard to {html_file}")
    logger.info(f"Saved latest dashboard to {latest_file}")
    logger.info("Done!")


if __name__ == '__main__':
    asyncio.run(main())
