#!/usr/bin/env python3
"""
HKJC Results Fetcher - Fetch actual race results from HKJC website
After races are finished, fetches results, scores prediction accuracy,
and regenerates the interactive dashboard with results overlaid.
"""

import asyncio
import re
import sys
import os
import json
import math
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fetcher import HKJCDataFetcher
from data.database import HorseDatabase

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsFetcher:
    def __init__(self):
        self.fetcher = HKJCDataFetcher()
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    async def fetch_race_results(self, race_date: str, race_no: int) -> List[Dict]:
        """Fetch actual results for a specific race."""
        logger.info(f"Fetching results for Race {race_no} on {race_date}")
        results = await self.fetcher.get_results(race_date, race_no)
        return results

    async def fetch_day_results(self, race_date: str, num_races: int = 11) -> Dict[int, List[Dict]]:
        """Fetch results for all FINISHED races on a given day.

        Races are run in chronological order, so results become available in
        order. Once we hit a race with no results, all subsequent races are
        guaranteed to be unfinished too — break to avoid wasted HTTP calls
        on the current/upcoming race plus any future ones.

        We allow a single empty between hits to tolerate transient scrape
        failures on a finished race (rare but possible).
        """
        logger.info(f"Fetching all results for {race_date}")

        day_results = {}
        consecutive_empty = 0
        seen_any_results = False

        for race_no in range(1, num_races + 1):
            results = await self.fetch_race_results(race_date, race_no)
            if results:
                day_results[race_no] = results
                logger.info(f"  Race {race_no}: {len(results)} finished")
                consecutive_empty = 0
                seen_any_results = True
            else:
                logger.info(f"  Race {race_no}: No results yet")
                consecutive_empty += 1
                # Stop once we've seen 2 unfinished races in a row AFTER
                # at least one finished race — those are the upcoming races
                # and any further races are guaranteed unfinished.
                if seen_any_results and consecutive_empty >= 2:
                    logger.info(
                        f"  Stopping at R{race_no}: races run in order, "
                        f"R{race_no+1}..R{num_races} are not yet finished"
                    )
                    break

        return day_results

    def load_prediction_data(self, race_date: str) -> Optional[List]:
        """Load the prediction JSON data."""
        date_str = race_date.replace('/', '')
        filepath = os.path.join(self.results_dir, f"race_data_{date_str}.json")

        if not os.path.exists(filepath):
            logger.warning(f"No prediction data found: {filepath}")
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    def merge_results_with_predictions(self, predictions: List, actual_results: Dict) -> List:
        """Merge actual results into prediction data and score accuracy."""
        for race in predictions:
            race_no = race.get('race_number')
            if race_no in actual_results:
                race['actual_results'] = actual_results[race_no]
                race['results_available'] = True

                actual_positions = {
                    r['number']: r['position']
                    for r in actual_results[race_no]
                }
                actual_winner = next(
                    (r['number'] for r in actual_results[race_no] if r.get('position') == 1), None
                )
                actual_top3 = set(
                    r['number'] for r in actual_results[race_no] if r.get('position', 99) <= 3
                )
                actual_win_odds = next(
                    (r.get('win_odds', 0) for r in actual_results[race_no] if r.get('position') == 1), 0
                )

                race['actual_winner'] = actual_winner
                race['actual_top3'] = list(actual_top3)
                race['actual_win_odds'] = actual_win_odds

                for horse in race.get('horses', []):
                    horse_no = horse.get('number')
                    horse['actual_position'] = actual_positions.get(horse_no)

                # Score accuracy
                race['accuracy'] = self._score_race_accuracy(race)
            else:
                race['results_available'] = False

        return predictions

    @staticmethod
    def _score_race_accuracy(race: Dict) -> Dict:
        """Score how well our predictions matched actual results."""
        horses = race.get('horses', [])
        actual_winner = race.get('actual_winner')
        actual_top3 = set(race.get('actual_top3', []))

        if not horses or not actual_winner:
            return {}

        # Our top picks
        our_rank1 = horses[0]['number'] if horses else None
        our_rank2 = horses[1]['number'] if len(horses) > 1 else None
        our_rank3 = horses[2]['number'] if len(horses) > 2 else None
        our_top3 = {our_rank1, our_rank2, our_rank3} - {None}

        # Win prediction correct?
        win_correct = our_rank1 == actual_winner

        # Top 3 overlap
        top3_overlap = len(our_top3 & actual_top3)

        # What rank did the actual winner have in our predictions?
        winner_our_rank = None
        for h in horses:
            if h['number'] == actual_winner:
                winner_our_rank = h['rank']
                break

        # Did our bet recommendation hit?
        bet_hit = False
        profit = 0.0
        for h in horses:
            if h['number'] == actual_winner:
                if h['bet_recommendation'] in ('WIN', 'BANKER WIN'):
                    bet_hit = True
                    profit = h.get('win_odds', 0) - 1  # net profit per $1
            elif h.get('actual_position') and h['actual_position'] <= 3:
                if h['bet_recommendation'] in ('PLACE', 'QIN', 'QNL PL', 'TRIO'):
                    bet_hit = True

        return {
            'win_correct': win_correct,
            'top3_overlap': top3_overlap,
            'winner_our_rank': winner_our_rank,
            'bet_hit': bet_hit,
            'profit': round(profit, 1),
        }

    def compute_day_summary(self, predictions: List) -> Dict:
        """Compute overall day accuracy summary with comparison vs public favourite."""
        races_with_results = [r for r in predictions if r.get('results_available')]
        if not races_with_results:
            return {}

        total = len(races_with_results)
        wins_correct = sum(1 for r in races_with_results if r.get('accuracy', {}).get('win_correct'))
        top3_total = sum(r.get('accuracy', {}).get('top3_overlap', 0) for r in races_with_results)
        bets_hit = sum(1 for r in races_with_results if r.get('accuracy', {}).get('bet_hit'))

        winner_ranks = [
            r['accuracy']['winner_our_rank']
            for r in races_with_results
            if r.get('accuracy', {}).get('winner_our_rank')
        ]
        avg_winner_rank = sum(winner_ranks) / len(winner_ranks) if winner_ranks else 0

        # --- Public favourite win rate ---
        # Public favourite = shortest odds horse in each race
        fav_wins = 0
        for r in races_with_results:
            horses = r.get('horses', [])
            actual_winner = r.get('actual_winner')
            if not horses or not actual_winner:
                continue
            # Find shortest odds horse (public favourite)
            fav = min(horses, key=lambda h: h.get('win_odds', 999) if h.get('win_odds', 0) > 0 else 999)
            if fav.get('number') == actual_winner:
                fav_wins += 1

        # --- Our BEST bet hit rate ---
        # QIN hit = winner is in our rank 1 or rank 2
        # WIN/BANKER hit = our rank 1 won
        best_bet_hits = 0
        best_bet_roi = 0.0
        for r in races_with_results:
            horses = r.get('horses', [])
            actual_winner = r.get('actual_winner')
            actual_top3 = set(r.get('actual_top3', []))
            if not horses or not actual_winner:
                continue
            rank1 = horses[0] if horses else None
            rank2 = horses[1] if len(horses) > 1 else None
            rank1_num = rank1.get('number') if rank1 else None
            rank2_num = rank2.get('number') if rank2 else None
            bet_type = rank1.get('bet_recommendation', '') if rank1 else ''

            if bet_type in ('BANKER WIN', 'WIN'):
                # WIN bet hits if rank 1 won
                if rank1_num == actual_winner:
                    best_bet_hits += 1
                    best_bet_roi += (rank1.get('win_odds', 0) - 1)
                else:
                    best_bet_roi -= 1  # lost the bet
            elif bet_type == 'QIN':
                # QIN hits if winner is rank 1 or rank 2
                if actual_winner in (rank1_num, rank2_num):
                    # Check if both rank 1 and rank 2 are in top 2
                    actual_top2 = set(
                        h['number'] for h in
                        sorted([hh for hh in horses if (hh.get('actual_position') or 99) <= 2],
                               key=lambda x: x.get('actual_position') or 99)
                    )
                    if rank1_num in actual_top2 and rank2_num in actual_top2:
                        best_bet_hits += 1
                        # QIN payout estimation (rough)
                        best_bet_roi += 3.0  # avg QIN payout
                    else:
                        best_bet_roi -= 1
                else:
                    best_bet_roi -= 1
            elif bet_type in ('QNL PL', 'TRIO'):
                # Simpler: did any of our picks hit top 3?
                our_picks = {h.get('number') for h in horses if h.get('bet_recommendation') == bet_type}
                if our_picks & actual_top3:
                    best_bet_hits += 1
                else:
                    best_bet_roi -= 1

        # --- Random baseline ---
        # Average field size
        avg_field = sum(len(r.get('horses', [])) for r in races_with_results) / total if total else 12
        random_win_rate = round(1.0 / avg_field * 100, 1)

        return {
            'total_races': total,
            'wins_correct': wins_correct,
            'win_accuracy': round(wins_correct / total * 100, 1) if total else 0,
            'top3_overlap_total': top3_total,
            'top3_accuracy': round(top3_total / (total * 3) * 100, 1) if total else 0,
            'bets_hit': bets_hit,
            'bet_hit_rate': round(bets_hit / total * 100, 1) if total else 0,
            'avg_winner_rank': round(avg_winner_rank, 1),
            # Comparison stats
            'fav_wins': fav_wins,
            'fav_win_rate': round(fav_wins / total * 100, 1) if total else 0,
            'best_bet_hits': best_bet_hits,
            'best_bet_hit_rate': round(best_bet_hits / total * 100, 1) if total else 0,
            'best_bet_roi': round(best_bet_roi, 1),
            'random_win_rate': random_win_rate,
        }

    def save_merged_data(self, data: List, race_date: str, summary: Dict) -> str:
        """Save merged prediction + actual results data."""
        date_str = race_date.replace('/', '')
        output = {
            'race_date': race_date,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'day_summary': summary,
            'races': data,
        }
        filepath = os.path.join(self.results_dir, f"race_data_{date_str}_with_results.json")

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved merged data to {filepath}")
        return filepath

    async def update_dashboard_with_results(self, race_date: str):
        """Fetch results, merge with predictions, score accuracy, regenerate dashboard."""
        predictions = self.load_prediction_data(race_date)
        if not predictions:
            logger.error("No prediction data to update")
            return None

        actual_results = await self.fetch_day_results(race_date)

        if not actual_results:
            logger.warning("No actual results found - races may not be finished yet")
            return None

        merged = self.merge_results_with_predictions(predictions, actual_results)
        summary = self.compute_day_summary(merged)

        # Fill prediction_log with actual results for longitudinal tracking
        self._fill_prediction_log(race_date, actual_results)

        # Refresh tier_stats cache — new results shift the windowed hit-rate
        self._refresh_tier_stats()

        output_file = self.save_merged_data(merged, race_date, summary)

        logger.info("Results fetched successfully!")
        logger.info(f"  Date: {race_date}")
        logger.info(f"  Races with results: {len(actual_results)}")
        if summary:
            logger.info(f"  Win accuracy: {summary['wins_correct']}/{summary['total_races']} ({summary['win_accuracy']}%)")
            logger.info(f"  Top 3 accuracy: {summary['top3_accuracy']}%")
            logger.info(f"  Avg winner rank in our predictions: {summary['avg_winner_rank']}")

        for race_no, results in sorted(actual_results.items()):
            winners = [r for r in results if r.get('position') == 1]
            if winners:
                logger.info(f"  Race {race_no} winner: #{winners[0]['number']} {winners[0]['name']}")

        # Regenerate dashboard with results
        self._regenerate_dashboard(merged, summary, race_date)

        return output_file

    @staticmethod
    def _fill_prediction_log(race_date: str, actual_results: Dict):
        """Fill prediction_log table with actual results for accuracy tracking."""
        try:
            db = HorseDatabase()
            try:
                for race_no, results in actual_results.items():
                    db.fill_prediction_results(race_date, race_no, results)
            finally:
                db.close()
            logger.info(f"Filled prediction_log with results for {len(actual_results)} races")
        except Exception as e:
            logger.warning(f"Could not fill prediction_log: {e}")

    @staticmethod
    def _refresh_tier_stats():
        """Refresh tier_stats.json cache after new results ingested."""
        try:
            from betting.tier_stats import refresh_cache
            db = HorseDatabase()
            try:
                refresh_cache(db.conn, window=73)
            finally:
                db.close()
            logger.info("Refreshed tier_stats.json cache")
        except Exception as e:
            logger.warning(f"tier_stats refresh failed: {e}")

    def _regenerate_dashboard(self, data: List, summary: Dict, race_date: str):
        """Re-render the dashboard from the merged-results data so actual_position
        fields are baked into the embedded RACES const. STRIKE/MISS badges
        depend on actual_position being present at JS-eval time.

        Writes BOTH the date-stamped archive AND the always-latest path so a
        browser viewing either stays in sync.

        v6.5 update: replaces the legacy runtime-injection mechanism that
        called helper functions (renderDashboard, renderRaceDetail) which
        were removed in the card-stack rewrite.
        """
        from datetime import datetime
        from dashboard.template import generate_html

        date_str = race_date.replace('/', '')
        archive_path = os.path.join(self.results_dir, f"racing_dashboard_{date_str}.html")
        latest_path = os.path.join(self.results_dir, "racing_dashboard.html")

        # Build venue from race data so the rendered page header is accurate
        venue = 'Happy Valley' if any(
            (r.get('venue') or '').startswith('Happy') for r in data
        ) else 'Sha Tin'

        json_data = json.dumps(data)
        json_data = json_data.replace('</script>', r'<\/script>').replace('</Script>', r'<\/Script>')
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        html = generate_html(json_data, now, race_date, venue)

        # Atomic write — same pattern the generator uses.
        # try/finally ensures tmp is unlinked on write failure (disk full,
        # SIGKILL between truncate and write) so stale .tmp.{pid} files don't
        # accumulate.
        for path in (archive_path, latest_path):
            tmp = f"{path}.tmp.{os.getpid()}"
            try:
                with open(tmp, 'w') as f:
                    f.write(html)
                os.replace(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

        logger.info(f"Dashboard re-rendered with results: {archive_path}")
        logger.info(f"Dashboard re-rendered with results: {latest_path}")
        return


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fetch HKJC race results')
    parser.add_argument('date', nargs='?', help='Race date (YYYY/MM/DD), defaults to today')
    parser.add_argument('--auto-learn', action='store_true',
                        help='Auto-ingest results into horse DB and retrain ML model')
    args = parser.parse_args()

    import re
    race_date = args.date or datetime.now().strftime('%Y/%m/%d')
    if not re.fullmatch(r'\d{4}/\d{2}/\d{2}', race_date):
        print(f"Invalid date format: {race_date}. Expected YYYY/MM/DD")
        return

    fetcher = ResultsFetcher()
    await fetcher.update_dashboard_with_results(race_date)
    await fetcher.fetcher.close()

    # Auto-learn: feed results to horse database and retrain ML model
    if args.auto_learn:
        await auto_learn(race_date)


def _build_races_from_json(json_races: list, race_date: str) -> list:
    """Convert race dicts from race_data_*_with_results.json into the
    scraper's race-result shape so HorseDatabase.ingest_race_day() can
    consume them. Used as a fallback when the scraper misses races (e.g.
    end-of-card timing) but the JSON has actual_results.
    """
    out = []
    for r in json_races:
        results = r.get('actual_results') or []
        if not results:
            continue
        horses_pred = {h.get('number'): h for h in r.get('horses', []) if h.get('number')}
        entries = []
        for res in results:
            num = res.get('number')
            name = (res.get('name') or '').split('(')[0].strip()
            if not name or not num:
                # database.py:232 silently skips entries with empty horse_name,
                # which would lose the horse from training data with no signal.
                # Drop here with a warning instead.
                logger.warning(
                    f"_build_races_from_json: dropping result with missing name/number "
                    f"in race {r.get('race_number')}: {res}"
                )
                continue
            pred = horses_pred.get(num, {})
            entries.append({
                'number': num,
                'name': name,
                'jockey': pred.get('jockey', ''),
                'trainer': pred.get('trainer', ''),
                'draw': pred.get('draw', 0),
                'weight': pred.get('weight', 0),
                'horse_weight': pred.get('horse_weight', 0),
                'rating': pred.get('rating', 0),
                'rating_change': pred.get('rating_change', 0),
                'age': pred.get('age', ''),
                'gear': pred.get('gear', ''),
                'last_6': pred.get('last_6', ''),
                'actual_position': res.get('position'),
                'win_odds': res.get('win_odds') or pred.get('win_odds', 0),
                'finish_time': '',
                'running_position': '',
                'margin': 0,
            })
        out.append({
            'date': race_date,
            'race_no': r.get('race_number'),
            'venue': r.get('venue', ''),
            'distance': r.get('distance', 0),
            'class_type': r.get('class_type', ''),
            'going': r.get('going', ''),
            'course': r.get('course', ''),
            'entries': entries,
            'dividends': [],
        })
    return out


async def _attach_sectional_times(today_races: list, race_date: str):
    """Fetch HKJC sectional data per race and attach `sectional_times` — a
    space-separated per-section time string, e.g. "27.84 23.69 24.22 24.10" —
    to each result entry, matched by horse number (Path B, Phase 1).

    Best-effort: a race whose sectional page is missing simply keeps an empty
    sectional_times. Fetches run concurrently (blocking requests in threads).
    """
    import requests
    from data.sectional_scraper import fetch_sectional_data

    session = requests.Session()

    def _fetch_one(race_no):
        return race_no, fetch_sectional_data(race_date, race_no, session=session)

    race_nos = [r.get('race_no') for r in today_races if r.get('race_no')]
    fetched = await asyncio.gather(
        *(asyncio.to_thread(_fetch_one, rn) for rn in race_nos)
    )
    by_race = {rn: data for rn, data in fetched if data}

    n_attached = 0
    for race in today_races:
        sdata = by_race.get(race.get('race_no'))
        if not sdata:
            continue
        sect_by_horse = {}
        for h in sdata['horses']:
            times = [s['time'] for s in h['sections']]
            if times:
                sect_by_horse[h['horse_no']] = ' '.join(f"{t:.2f}" for t in times)
        for entry in race.get('entries', []):
            st = sect_by_horse.get(entry.get('number'))
            if st:
                entry['sectional_times'] = st
                n_attached += 1
                # Backfill finish_time from sectional sum when HKJC has not yet
                # published official times (often the case for the last race of
                # the card). The sectionals partition the total — their sum IS
                # the finish time by definition. Next official fetch overwrites
                # this with the authoritative value if it differs.
                if not (entry.get('finish_time') or '').strip():
                    try:
                        total = sum(float(x) for x in st.split())
                        mins, secs = divmod(total, 60)
                        entry['finish_time'] = f"{int(mins)}:{secs:05.2f}"
                    except (ValueError, TypeError):
                        pass
    logger.info(f"Attached sectional times to {n_attached} result entries "
                f"({len(by_race)}/{len(race_nos)} races had sectional pages)")


async def auto_learn(race_date: str = None):
    """Automatically ingest results into horse DB and retrain ML model."""
    from data.database import HorseDatabase
    from scrape_history import scrape_meeting_results
    from pathlib import Path
    import subprocess

    race_date = race_date or datetime.now().strftime('%Y/%m/%d')

    # 1. Scrape actual results from HKJC and ingest into horse database
    today_races = await scrape_meeting_results(race_date)

    if today_races:
        # Path B (Phase 1): enrich each race with sectional times before ingest.
        await _attach_sectional_times(today_races, race_date)
        db = HorseDatabase()
        try:
            db.ingest_race_day(today_races)

            date_str = race_date.replace('/', '')
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            pred_races = None
            for variant in (f'race_data_{date_str}_with_results.json',
                            f'race_data_{date_str}.json'):
                pred_path = os.path.join(results_dir, variant)
                if os.path.exists(pred_path):
                    try:
                        with open(pred_path) as f:
                            pred_data = json.load(f)
                        pred_races = pred_data if isinstance(pred_data, list) else pred_data.get('races', [])
                    except Exception as e:
                        logger.warning(f"Could not read {variant}: {e}")
                    break

            if pred_races:
                scraped_race_nos = {r.get('race_no') or r.get('race_number') for r in today_races}
                missing = [r for r in pred_races
                           if r.get('race_number') not in scraped_race_nos
                           and r.get('actual_results')]
                if missing:
                    missing_nos = [r['race_number'] for r in missing]
                    logger.warning(
                        f"Scraper missed race(s) {missing_nos} that have results in JSON "
                        f"— ingesting partial (top finishers only) from JSON. "
                        f"Re-run --auto-learn after HKJC publishes full result pages to capture remaining horses."
                    )
                    fallback = _build_races_from_json(missing, race_date)
                    if fallback:
                        db.ingest_race_day(fallback)

                n_back = db.backfill_card_fields(race_date, pred_races)
                if n_back > 0:
                    logger.info(f"Backfilled card fields on {n_back} race_results rows from JSON")

            summary = db.get_summary()
            logger.info(f"Horse DB updated: {summary['horses']} horses, "
                        f"{summary['race_results']} results, {summary['race_days']} days")
        finally:
            db.close()
    else:
        logger.warning(f"No results scraped for {race_date} — DB not updated")

    # 2. Retrain ML model
    # Timeout 600s — full retrain on 50K+ rows takes ~5min, was failing at 300s
    logger.info("Auto-retraining ML model...")
    retrain_ok = False
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / 'train_ml_model.py')],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            retrain_ok = True
            # Extract accuracy from output
            for line in result.stdout.split('\n'):
                if 'Win accuracy' in line:
                    logger.info(f"ML retrain: {line.strip()}")
        else:
            logger.warning(f"ML retrain failed: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        logger.warning("ML retrain timed out (600s) — run 'python3 train_ml_model.py' manually")

    # 3. Re-fit the post-blend calibrator on the just-retrained model.
    # Without this step the calibrator stays frozen against the previous
    # model's probability distribution — its isotonic mapping no longer
    # corresponds to current pipeline outputs, leaving residual under-
    # confidence visible in the next backtest.
    # Skip when retrain failed/timed out — the model on disk is still
    # the OLD model, so re-fitting would produce the same calibrator-vs-
    # model mismatch this whole step exists to prevent.
    if not retrain_ok:
        logger.warning("Skipping blend calibration — retrain did not succeed; "
                       "calibrator on disk left untouched (matches existing model)")
    else:
        logger.info("Re-fitting blend calibrator...")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'scripts.calibrate_blend'],
                capture_output=True, text=True, timeout=300,
                cwd=str(Path(__file__).parent.parent),
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[-3:]:
                    logger.info(f"Blend calibration: {line}")
            else:
                logger.warning(f"Blend calibration failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("Blend calibration timed out — run 'python3 -m scripts.calibrate_blend' manually")

    # 4. Recalibrate gear scores from new race_results.
    # Skip when no new rows since the last calibration — the gear-scores
    # JSON stores n_total_rows in its _meta block, so we can compare
    # cheaply without re-reading the whole file.
    gear_path = Path(__file__).parent.parent / 'models' / 'gear_scores.json'
    skip_gear = False
    if gear_path.exists():
        try:
            with open(gear_path) as f:
                last_meta = json.load(f).get('_meta', {})
            last_n = int(last_meta.get('n_total_rows') or 0)
            import sqlite3
            import config
            _c = sqlite3.connect(str(config.DB_PATH))
            current_n = _c.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]
            _c.close()
            if last_n and current_n == last_n:
                skip_gear = True
                logger.info(f"Skipping gear calibration — race_results unchanged at {current_n:,} rows")
        except Exception as e:
            logger.warning(f"Could not check gear-scores freshness ({e}); re-running calibration")

    if not skip_gear:
        logger.info("Recalibrating gear scores...")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'scripts.calibrate_gear'],
                capture_output=True, text=True, timeout=60,
                cwd=str(Path(__file__).parent.parent),
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[-3:]:
                    logger.info(f"Gear calibration: {line}")
            else:
                logger.warning(f"Gear calibration failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("Gear calibration timed out — run 'python3 -m scripts.calibrate_gear' manually")


if __name__ == '__main__':
    asyncio.run(main())