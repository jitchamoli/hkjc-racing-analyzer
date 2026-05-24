#!/usr/bin/env python3
"""
Train ML model from real HKJC historical race results.
Fetches past race cards + results, builds feature vectors, and trains XGBoost.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fetcher import HKJCApiClient, HKJCParser, HKJCDataFetcher
from ml.model import HKJCRacingMLModel, RaceResult, ML_AVAILABLE
from ml.features import FeatureEngineering
from ml.training import build_training_data_from_db, generate_synthetic_training_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path(__file__).parent / 'training_data'
TRAINING_DATA_DIR.mkdir(exist_ok=True)
TRAINING_DATA_FILE = TRAINING_DATA_DIR / 'historical_races.json'


def find_race_dates(n_meetings: int = 15) -> List[str]:
    """Find recent meeting dates with results."""
    client = HKJCApiClient()
    valid = []
    d = datetime.now()

    for i in range(120):  # scan up to 120 days back
        check = d - timedelta(days=i)
        date_str = check.strftime('%Y/%m/%d')
        html = client.fetch_results_html(date_str, 1)
        if html and ("table_bd" in html or "tableBorderBlue" in html):
            # Count races
            race_nos = set(re.findall(r'RaceNo=(\d+)', html))
            n_races = len(race_nos) if race_nos else 11
            valid.append((date_str, n_races))
            logger.info(f"Found meeting: {date_str} ({n_races} races)")
            if len(valid) >= n_meetings:
                break
        time.sleep(0.3)

    return valid


def fetch_meeting_data(client: HKJCApiClient, race_date: str,
                       n_races: int) -> List[Dict]:
    """Fetch race cards + results for a full meeting."""
    meeting_data = []

    for race_no in range(1, n_races + 1):
        # Fetch race card
        card_html = client.fetch_race_card_html(race_date, race_no)
        if not card_html:
            continue
        time.sleep(0.5)

        race_card = HKJCParser.parse_race_card(card_html, race_no, race_date)
        if not race_card or not race_card.horses:
            continue
        entries = race_card.horses

        # Parse race metadata from card
        race_info = {
            'venue': race_card.venue,
            'distance': race_card.distance,
            'class_type': race_card.class_type,
            'going': race_card.going,
            'course': getattr(race_card, 'track', ''),
        }

        # Fetch results
        results_html = client.fetch_results_html(race_date, race_no)
        if not results_html:
            continue
        time.sleep(0.3)

        results = HKJCParser.parse_results(results_html)
        if not results:
            continue

        # Build position lookup
        pos_lookup = {}
        for r in results:
            pos_lookup[r['number']] = r['position']

        # Only use races where we have at least 4 results
        if len(pos_lookup) < 4:
            continue

        race_record = {
            'date': race_date,
            'race_no': race_no,
            'venue': race_info.get('venue', 'Sha Tin'),
            'distance': race_info.get('distance', '1200'),
            'class_type': race_info.get('class_type', ''),
            'going': race_info.get('going', ''),
            'course': race_info.get('course', ''),
            'entries': [],
        }

        for entry in entries:
            horse_no = entry.number
            actual_pos = pos_lookup.get(horse_no, 0)
            if actual_pos == 0:
                continue  # horse scratched or no result

            race_record['entries'].append({
                'number': horse_no,
                'name': entry.name,
                'jockey': entry.jockey,
                'trainer': entry.trainer,
                'draw': entry.draw,
                'weight': entry.weight,
                'rating': entry.rating,
                'last_6': entry.last_6,
                'gear': entry.gear,
                'win_odds': entry.win_odds,
                'actual_position': actual_pos,
                'is_winner': actual_pos == 1,
                'is_top3': actual_pos <= 3,
            })

        if len(race_record['entries']) >= 4:
            meeting_data.append(race_record)
            logger.info(f"  R{race_no}: {len(race_record['entries'])} entries, "
                        f"winner=#{[e['number'] for e in race_record['entries'] if e['is_winner']]}")

    return meeting_data


def build_features_from_record(entry: Dict, race: Dict) -> np.ndarray:
    """Build ML feature vector from a historical race entry."""
    fe = FeatureEngineering()

    horse_data = {
        'number': entry['number'],
        'name': entry['name'],
        'rating': entry['rating'],
        'draw': entry['draw'],
        'jockey': entry['jockey'],
        'trainer': entry['trainer'],
        'last_6': entry['last_6'],
        'distance': race['distance'],
        'race_date': race.get('date', ''),
        'running_style': '',
        'last_400m_avg': 0.0,
        'cd_wins': 0,
        'cd_runs': 0,
        'opening_odds': 0,
        'win_odds': entry['win_odds'],
        'tips_odds': 0,
        'course': race.get('course', ''),
        'venue': race.get('venue', 'Sha Tin'),
        'gear': entry.get('gear', ''),
        'weight': entry.get('weight', ''),
        'class_type': race.get('class_type', ''),
    }

    features = fe.extract_features(horse_data, [], race.get('venue', ''))
    return features


def build_training_data(all_races: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert historical races into X (features) and y (labels)."""
    X_list = []
    y_list = []

    for race in all_races:
        for entry in race['entries']:
            try:
                features = build_features_from_record(entry, race)
                X_list.append(features)
                y_list.append(1 if entry['is_winner'] else 0)
            except Exception as e:
                logger.warning(f"Feature extraction failed for #{entry['number']}: {e}")
                continue

    return np.array(X_list), np.array(y_list)


def main():
    logger.info("=== HKJC ML Model Training Pipeline ===")

    # Step 1: Check for cached historical data
    all_races = []
    if TRAINING_DATA_FILE.exists():
        with open(TRAINING_DATA_FILE) as f:
            all_races = json.load(f)
        logger.info(f"Loaded {len(all_races)} cached races from {TRAINING_DATA_FILE}")

    # Step 2: Fetch new data (skip if SKIP_FETCH=1 — when DB already has all
    # we need, the Playwright meeting fetch is wasteful and prone to hangs).
    new_races = 0
    if os.environ.get('SKIP_FETCH', '0') == '1':
        logger.info("SKIP_FETCH=1 — bypassing meeting-data fetch (using DB only)")
    else:
        cached_dates = set(r['date'] for r in all_races)

        logger.info("Finding recent meeting dates...")
        meetings = find_race_dates(n_meetings=15)
        logger.info(f"Found {len(meetings)} meetings")

        client = HKJCApiClient()
        for date_str, n_races in meetings:
            if date_str in cached_dates:
                logger.info(f"  {date_str}: already cached, skipping")
                continue

            logger.info(f"Fetching {date_str} ({n_races} races)...")
            race_data = fetch_meeting_data(client, date_str, n_races)
            all_races.extend(race_data)
            new_races += len(race_data)
            time.sleep(1)

        if new_races > 0:
            with open(TRAINING_DATA_FILE, 'w') as f:
                json.dump(all_races, f)
            logger.info(f"Saved {len(all_races)} total races to {TRAINING_DATA_FILE}")

    # Step 3: Build features — prefer DB data (richer, more complete)
    # 2026-04-25: TRAINING_YEARS_BACK env var caps training to recent races.
    # Empirical: training on 49y (360K rows) gave ROI +927% vs +1359% on
    # 7K-row recent-only window. Default 5y if unset; override with
    # TRAINING_YEARS_BACK=0 (no cutoff) or any int.
    yrs_str = os.environ.get('TRAINING_YEARS_BACK', '5')
    yrs = None if yrs_str.strip() == '0' else int(yrs_str)
    X_db, y_db, sw_db, y_place_db, groups_db = build_training_data_from_db(
        years_back=yrs, return_groups=True,
    )

    place_labels = None
    if X_db is not None and len(X_db) >= 200:
        logger.info(f"Using DB training data: {len(X_db)} samples, "
                    f"{y_db.sum():.0f} winners ({y_db.mean()*100:.1f}%)")
        X_combined = X_db
        y_combined = y_db
        sample_weights = sw_db
        place_labels = y_place_db
        groups = groups_db
    else:
        # Fallback to scraped training_data
        total_entries = sum(len(r['entries']) for r in all_races)
        logger.info(f"Building features from {len(all_races)} scraped races, {total_entries} entries...")
        X_real, y_real = build_training_data(all_races)
        logger.info(f"Scraped data: {X_real.shape[0]} samples, {y_real.sum():.0f} winners "
                    f"({y_real.mean()*100:.1f}% win rate)")

        if X_real.shape[0] < 500:
            logger.info("Augmenting with synthetic data (real data < 500 samples)...")
            X_syn, y_syn = generate_synthetic_training_data(2000)
            X_combined = np.vstack([X_real, X_syn])
            y_combined = np.concatenate([y_real, y_syn])
            sample_weights = np.ones(len(y_combined))
            sample_weights[:len(y_real)] = 3.0
            groups = None
            logger.info(f"Combined: {X_combined.shape[0]} samples "
                        f"(real={X_real.shape[0]}, synthetic={X_syn.shape[0]})")
        else:
            X_combined = X_real
            y_combined = y_real
            sample_weights = None
            groups = None
            logger.info("Enough real data — training without synthetic augmentation")

    # Step 5: Train model
    if not ML_AVAILABLE:
        logger.error("XGBoost not available — cannot train")
        return

    model = HKJCRacingMLModel()

    # Backup old model before training (restore if training fails)
    backup_path = model.MODEL_FILE.with_suffix('.pkl.bak')
    sha_backup = model._checksum_path().with_suffix('.sha256.bak')
    if model.MODEL_FILE.exists():
        model.MODEL_FILE.rename(backup_path)
        if model._checksum_path().exists():
            model._checksum_path().rename(sha_backup)
        logger.info("Backed up old model")

    try:
        result = model.train(X_combined, y_combined, sample_weight=sample_weights,
                             place_labels=place_labels, groups=groups)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if backup_path.exists():
            backup_path.rename(model.MODEL_FILE)
            if sha_backup.exists():
                sha_backup.rename(model._checksum_path())
            logger.info("Restored model from backup")
        raise

    # Training succeeded — remove backup
    if backup_path.exists():
        backup_path.unlink()
    if sha_backup.exists():
        sha_backup.unlink()
    logger.info(f"Model trained: accuracy={result['accuracy']:.2%}, "
                f"precision={result.get('precision', 0):.2%}, "
                f"recall={result.get('recall', 0):.2%}")

    # Step 6: Feature importance
    logger.info("\nFeature Importance:")
    for name, imp in sorted(model.feature_importance.items(),
                            key=lambda x: x[1], reverse=True):
        bar = '#' * int(imp * 100)
        logger.info(f"  {name:<22} {imp:.3f} {bar}")

    # Step 7: Validate on today's results if available
    today = datetime.now().strftime('%Y/%m/%d')
    today_races = [r for r in all_races if r['date'] == today]
    if today_races:
        logger.info(f"\n=== Validation on today's races ({today}) ===")
        correct = 0
        top3_hits = 0
        total = 0
        for race in today_races:
            entries = race['entries']
            if not entries:
                continue

            # Score each horse
            scores = []
            for entry in entries:
                features = build_features_from_record(entry, race)
                prob, _, _ = model.predict(features, field_size=len(entries))
                scores.append((entry['number'], entry['name'], prob,
                              entry['actual_position']))

            scores.sort(key=lambda x: x[2], reverse=True)
            winner = [s for s in scores if s[3] == 1]
            our_pick = scores[0] if scores else None

            if our_pick and winner:
                total += 1
                w = winner[0]
                w_rank = next((i+1 for i, s in enumerate(scores) if s[0] == w[0]), 99)
                if our_pick[3] == 1:
                    correct += 1
                our_top3_nums = [s[0] for s in scores[:3]]
                actual_top3 = [s[0] for s in scores if s[3] <= 3]
                overlap = len(set(our_top3_nums) & set(actual_top3))
                top3_hits += overlap
                logger.info(f"  R{race['race_no']}: Pick=#{our_pick[0]} ({our_pick[2]:.1%}), "
                           f"Winner=#{w[0]} (rank {w_rank}), Top3 overlap={overlap}/3")

        if total > 0:
            logger.info(f"\n  Win accuracy: {correct}/{total} ({correct/total:.0%})")
            logger.info(f"  Top 3 hits: {top3_hits}/{total*3} ({top3_hits/(total*3):.0%})")

    logger.info("\n=== Training complete ===")


if __name__ == '__main__':
    main()
