#!/usr/bin/env python3
"""
Generate race predictions and dashboard - non-interactive
"""

import asyncio
import itertools
import re
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fetcher import HKJCDataFetcher
from core.analyzer import CombinedRaceAnalyzer
from dashboard.template import generate_html
from data.database import HorseDatabase
from data.odds_tracker import OddsDatabase, OddsMovementAnalyzer
from analytics.pace import PaceAnalyzer
from betting.exotic_calculator import find_value_exotics

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Result fields set by results_fetcher that should survive prediction re-runs
_RESULT_KEYS_RACE = {'results_available', 'actual_results', 'actual_winner', 'actual_top3', 'actual_win_odds', 'accuracy'}
_RESULT_KEYS_HORSE = {'actual_position'}


def _merge_with_existing(json_file: str, new_races: list) -> list:
    """Preserve result data from a prior prediction file when re-generating.

    If the existing file has actual_position / accuracy fields (set by
    results_fetcher), carry them forward into the new predictions so a
    re-run for fresher odds doesn't wipe scored results.
    """
    # Prefer _with_results variant (has scored results), fall back to base file
    results_file = json_file.replace('.json', '_with_results.json')
    source = None
    for candidate in [results_file, json_file]:
        if os.path.exists(candidate):
            source = candidate
            break

    if not source:
        return new_races

    try:
        with open(source) as f:
            old_data = json.load(f)
        # _with_results wraps races in {"races": [...]}
        old_races = old_data.get('races', old_data) if isinstance(old_data, dict) else old_data
    except (json.JSONDecodeError, OSError):
        return new_races

    # Build lookup: race_number -> old race data
    old_by_num = {r['race_number']: r for r in old_races if isinstance(r, dict)}

    # Sanity check: if all races with results share the same winner, data is likely bogus
    winners_with_results = [
        r.get('actual_winner') for r in old_by_num.values()
        if r.get('results_available') and r.get('actual_winner')
    ]
    if len(winners_with_results) >= 3 and len(set(winners_with_results)) == 1:
        logger.warning(f"  Skipping result merge — all {len(winners_with_results)} races "
                       f"have same winner #{winners_with_results[0]} (likely bogus data)")
        return new_races

    for race in new_races:
        old_race = old_by_num.get(race['race_number'])
        if not old_race or not old_race.get('results_available'):
            continue

        # Copy race-level result fields
        for key in _RESULT_KEYS_RACE:
            if key in old_race:
                race[key] = old_race[key]

        # Copy horse-level result fields (match by horse number)
        old_horses = {h['number']: h for h in old_race.get('horses', []) if 'number' in h}
        for horse in race.get('horses', []):
            old_h = old_horses.get(horse['number'])
            if old_h:
                for key in _RESULT_KEYS_HORSE:
                    if key in old_h:
                        horse[key] = old_h[key]

        logger.info(f"  Race {race['race_number']}: preserved results from prior run")

    return new_races


def build_race_data(race, analyses, race_date, db, odds_db):
    """Build the JSON-serializable race data dict from analyses.

    Shared between generator and monitor to avoid duplication.
    Returns race_data dict with horses, speed_map, and exotic_value.
    """
    race_going = race.going or ''
    race_distance = int(race.distance) if race.distance and str(race.distance).isdigit() else 1200

    # Speed map
    speed_map_input = [
        {'number': h.number, 'name': h.name, 'draw': h.draw,
         'running_style': getattr(h, 'running_style', ''),
         'last_6': h.last_6}
        for h in race.horses
    ]
    speed_map = PaceAnalyzer.generate_speed_map(speed_map_input, race_distance)

    # Odds movement
    odds_summary = odds_db.get_movement_summary(race_date, race.race_number)

    # Loop-invariant: same venue+distance for all horses in this race
    dist_record = db.get_distance_record(race.venue or '', race_distance)

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

    for i, a in enumerate(analyses):
        h = a.horse
        ml_prob = a.ml_prediction.win_probability if a.ml_prediction else 0
        place_prob = a.ml_prediction.place_probability if a.ml_prediction else 0
        calibrated_prob = a.combined_probability if a.combined_probability else 0

        going_form = db.get_horse_going_form(h.name, race_going, race_date)
        jh_stats = db.get_jockey_horse_stats(h.jockey, h.name, race_date)
        dist_pb = db.get_distance_best_time(h.name, race_distance, race_date)

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
            "age": h.age,
            "horse_weight": h.horse_weight,
            "days_since_last_run": h.days_since_last_run,
            "rating_change": h.rating_change,
            "rule_score": a.rule_score,
            "ml_prob": ml_prob,
            "place_prob": place_prob,
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

    # === Post-bet analytics: Harville dividend + strategy confidence + EV ===
    from betting.dividends import (
        estimate_quinella_dividend, quinella_bet_ev, normalize_market_probs
    )
    horses_json = race_data['horses']
    qin_indices = [i for i, h in enumerate(horses_json) if h['bet_recommendation'] == 'QIN']
    win_odds_list = [h.get('win_odds') or 0 for h in horses_json]
    model_probs = [h.get('calibrated_prob') or 0 for h in horses_json]
    market_probs = normalize_market_probs(win_odds_list)

    # Confidence tier from model conviction on top-2/top-3
    sorted_probs = sorted(model_probs, reverse=True)
    top_2_share = sum(sorted_probs[:2]) if len(sorted_probs) >= 2 else 0
    top_3_share = sum(sorted_probs[:3]) if len(sorted_probs) >= 3 else 0
    if top_2_share >= 0.33 or top_3_share >= 0.42:
        confidence = 'HIGH'
    elif top_2_share >= 0.27 or top_3_share >= 0.35:
        confidence = 'MEDIUM'
    elif top_2_share >= 0.22:
        confidence = 'LOW'
    else:
        confidence = 'NONE'
    race_data['strategy_confidence'] = confidence
    race_data['top_2_share'] = round(top_2_share, 4)
    race_data['top_3_share'] = round(top_3_share, 4)

    # Quinella dividend estimate + EV (only when QIN bet is active)
    if len(qin_indices) >= 2:
        pair_combos = list(itertools.combinations(qin_indices, 2))
        ev_result = quinella_bet_ev(model_probs, market_probs, pair_combos)
        race_data['qin_bet_ev'] = {
            'n_combos': len(pair_combos),
            'total_stake': ev_result['total_stake'],
            'expected_profit': round(ev_result['expected_profit'], 2),
            'ev_pct': round(ev_result['ev_pct'], 1),
            'hit_prob': round(ev_result['hit_prob'], 4),
            'avg_dividend': round(ev_result['avg_dividend'], 2),
        }
        # Top pair dividend (rank 1+2) for at-a-glance display
        if len(qin_indices) >= 2:
            top_div = estimate_quinella_dividend(market_probs, qin_indices[0], qin_indices[1])
            race_data['qin_top_pair_dividend'] = round(top_div, 2) if top_div else None

    # Exotic bet value detection
    if len(analyses) >= 4:
        model_probs = [h['calibrated_prob'] for h in race_data['horses']]
        market_probs = []
        for h in race_data['horses']:
            wo = h.get('win_odds', 0)
            market_probs.append(1.0 / wo if wo and wo > 0 else 0)
        mp_total = sum(market_probs) or 1.0
        market_probs = [p / mp_total for p in market_probs]
        horse_nums = [h['number'] for h in race_data['horses']]
        place_probs = [h.get('place_prob', 0) for h in race_data['horses']]
        horse_draws = [h.get('draw', 0) for h in race_data['horses']]
        horse_styles = [
            next((r.split(':')[0] for r in (h.get('reasoning') or [])
                  if 'Leader' in r or 'Stalker' in r or 'Midfield' in r or 'Closer' in r), '')
            for h in race_data['horses']
        ]
        exotics = find_value_exotics(
            model_probs, market_probs, horse_nums, min_edge=0.10,
            place_probs=place_probs, draws=horse_draws, styles=horse_styles,
        )
        race_data["exotic_value"] = exotics

    return race_data


async def main():
    race_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y/%m/%d')
    if not re.fullmatch(r'\d{4}/\d{2}/\d{2}', race_date):
        print(f"Invalid date format: {race_date}. Expected YYYY/MM/DD")
        sys.exit(1)

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
        race_data = build_race_data(race, analyses, race_date, db, odds_db)
        all_races.append(race_data)
        logger.info(f"  Race {race_no}: {len(race.horses)} horses, bet: {analyses[0].bet_recommendation if analyses else 'N/A'}")

    date_str = race_date.replace('/', '')
    json_file = os.path.join(results_dir, f"race_data_{date_str}.json")

    # Merge: preserve result fields from prior run (e.g. actual_position, accuracy)
    all_races = _merge_with_existing(json_file, all_races)

    # Cross-race EV ranking — flag relative best/worst among active bets.
    # Most races show negative EV because model ≈ market for QIN pairs (takeout
    # drag). The ranking is still informative: highest EV = smallest drag =
    # best value of the day, even when absolute EV is negative.
    races_with_ev = [r for r in all_races if r.get('qin_bet_ev')]
    if len(races_with_ev) >= 2:
        best = max(races_with_ev, key=lambda r: r['qin_bet_ev']['ev_pct'])
        worst = min(races_with_ev, key=lambda r: r['qin_bet_ev']['ev_pct'])
        # Only flag if there's a meaningful spread between best and worst
        ev_spread = best['qin_bet_ev']['ev_pct'] - worst['qin_bet_ev']['ev_pct']
        if ev_spread >= 5.0:  # at least 5 percentage points apart
            for r in all_races:
                r['is_best_ev'] = (r['race_number'] == best['race_number'])
                r['is_worst_ev'] = (r['race_number'] == worst['race_number'])

    with open(json_file, 'w') as f:
        json.dump(all_races, f, indent=2)

    # Log predictions to DB for longitudinal accuracy tracking
    model_ver = analyzer.ml_analyzer.ml_model.model_version if hasattr(analyzer, 'ml_analyzer') else ''
    db.log_predictions(race_date, all_races, model_version=model_ver)

    db.close()

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
