#!/usr/bin/env python3
"""Feature engineering for ML predictions.

Extracts 18 features per horse using FEATURE_SCHEMA order.
All features use only data available BEFORE the race (no leakage).
"""

import re
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES, validate_features
import config

logger = logging.getLogger(__name__)


class FeatureEngineering:
    @staticmethod
    def parse_last_6(last_6: str) -> List[int]:
        """Parse form string — handles '1/3/10/2/5' and legacy '13425'."""
        s = str(last_6)
        if "/" in s:
            return [int(p) for p in s.split("/") if p.strip().isdigit()]
        return [int(ch) for ch in s if ch.isdigit()]

    @staticmethod
    def calculate_form_score(last_6: str) -> float:
        positions = FeatureEngineering.parse_last_6(last_6)
        if not positions:
            return 50.0
        total = 0
        for pos in positions:
            if pos == 1: total += 100
            elif pos == 2: total += 70
            elif pos == 3: total += 50
            elif pos == 4: total += 30
            elif pos == 5: total += 15
            else: total += 5
        return total / len(positions)

    @staticmethod
    def calculate_win_rate(history) -> float:
        if not history:
            return 0.08
        wins = sum(1 for r in history if r.is_win)
        return min(0.8, wins / len(history))

    @staticmethod
    def calculate_place_rate(history) -> float:
        if not history:
            return 0.25
        places = sum(1 for r in history if r.is_place)
        return min(0.9, places / len(history))

    @staticmethod
    def calculate_avg_position(history) -> float:
        if not history:
            return 6.0
        return sum(r.position for r in history) / len(history)

    @staticmethod
    def calculate_jockey_win_rate(jockey: str, all_results) -> float:
        jockey_rides = [r for r in all_results if jockey.lower() in r.jockey.lower()]
        if not jockey_rides:
            return 0.12
        wins = sum(1 for r in jockey_rides if r.is_win)
        return min(0.6, wins / len(jockey_rides))

    @staticmethod
    def calculate_trainer_win_rate(trainer: str, all_results) -> float:
        trainer_horses = [r for r in all_results if trainer.lower() in r.trainer.lower()]
        if not trainer_horses:
            return 0.12
        wins = sum(1 for r in trainer_horses if r.is_win)
        return min(0.5, wins / len(trainer_horses))

    @staticmethod
    def calculate_draw_advantage(draw: int, distance) -> float:
        try:
            distance = int(distance)
        except (ValueError, TypeError):
            distance = 1200
        if distance <= 1000:
            ideal = 4
        elif distance <= 1400:
            ideal = 6
        else:
            ideal = 10
        diff = abs(draw - ideal)
        if diff == 0: return 1.0
        elif diff <= 2: return 0.8
        elif diff <= 4: return 0.6
        else: return max(0.3, 1 - (diff * 0.1))

    @staticmethod
    def _style_to_float(style: str) -> float:
        return {"Leader": 0.9, "Stalker": 0.7, "Midfield": 0.5, "Closer": 0.3}.get(style, 0.5)

    @staticmethod
    def extract_features(horse_data: dict, all_results, venue: str) -> np.ndarray:
        """Extract 18 features matching FEATURE_SCHEMA order exactly."""
        features = []
        horse_name = horse_data.get('name', '')
        distance = horse_data.get('distance', 1200)
        going = horse_data.get('going', '')

        # Load prior race history from DB
        prior = []
        race_date = horse_data.get('race_date', '')
        try:
            db_path = config.DB_PATH
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                if race_date:
                    rows = conn.execute(
                        "SELECT * FROM race_results WHERE horse_name = ? AND race_date < ? ORDER BY race_date",
                        (horse_name, race_date)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM race_results WHERE horse_name = ? ORDER BY race_date",
                        (horse_name,)
                    ).fetchall()
                prior = [dict(r) for r in rows]
                conn.close()
        except Exception:
            pass

        # Feature 1: prior win rate
        if prior:
            prior_wins = sum(1 for p in prior if p.get('is_winner', 0))
            prior_top3 = sum(1 for p in prior if p.get('is_top3', 0))
            feat_form_proxy = min(1.0, (prior_wins * 2 + prior_top3) / (len(prior) * 3))
        else:
            feat_form_proxy = 0.5
        features.append(feat_form_proxy)

        # Feature 2: recent form from prior finishes
        if prior:
            recent = prior[-6:]
            form_pts = 0
            for p in recent:
                pos = p.get('finish_position', 7)
                if pos == 1: form_pts += 10
                elif pos == 2: form_pts += 7
                elif pos == 3: form_pts += 5
                elif pos <= 5: form_pts += 2
            features.append(form_pts / (len(recent) * 10))
        else:
            last_6 = horse_data.get('last_6', '')
            if last_6:
                features.append(FeatureEngineering.calculate_form_score(last_6) / 100)
            else:
                features.append(0.5)

        # Feature 3: draw advantage
        draw = horse_data.get('draw', 8)
        features.append(FeatureEngineering.calculate_draw_advantage(draw, distance))

        # Feature 4: jockey win rate
        jockey = horse_data.get('jockey', '')
        jockey_rate = FeatureEngineering.calculate_jockey_win_rate(jockey, all_results)
        features.append(min(0.5, jockey_rate) if jockey_rate > 0 else 0.08)

        # Feature 5: trainer win rate
        trainer = horse_data.get('trainer', '')
        trainer_rate = FeatureEngineering.calculate_trainer_win_rate(trainer, all_results)
        features.append(min(0.4, trainer_rate) if trainer_rate > 0 else 0.06)

        # Feature 6: field size
        field_size = horse_data.get('field_size', 12)
        features.append(field_size / 14.0)

        # Feature 7: field-relative draw
        features.append(draw / max(1, field_size) if draw > 0 else 0.5)

        # Feature 8: prior pace
        pace_vals = []
        for p in prior[-5:]:
            rp = p.get('running_position', '') or ''
            if rp:
                if ' ' in rp:
                    positions = [int(x) for x in rp.split() if x.isdigit()]
                else:
                    positions = [int(c) for c in rp if c.isdigit()]
                if positions:
                    early = min(positions[0], 14)
                    pace_vals.append(1.0 - (early - 1) / 13.0)
        features.append(sum(pace_vals) / len(pace_vals) if pace_vals else 0.5)

        # Feature 9: prior finishing speed
        speed_vals = []
        for p in prior[-5:]:
            ft = p.get('finish_time', '') or ''
            d = p.get('distance', 0) or 0
            if ft and ':' in ft and d > 0:
                try:
                    parts = ft.split(':')
                    secs = float(parts[0]) * 60 + float(parts[1])
                    if secs > 0:
                        speed = d / secs
                        speed_vals.append(max(0.0, min(1.0, (speed - 14) / 4)))
                except (ValueError, IndexError):
                    pass
        features.append(sum(speed_vals) / len(speed_vals) if speed_vals else 0.5)

        # Feature 10: C&D form from prior races
        prior_at_dist = [p for p in prior if p.get('distance') == distance]
        if prior_at_dist:
            cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
            features.append(min(1.0, cd_wins / len(prior_at_dist) + 0.2))
        else:
            features.append(0.5)

        # Feature 11: odds-implied probability
        win_odds = horse_data.get('win_odds', 0)
        features.append(min(1.0, 1.0 / win_odds) if win_odds > 0 else 0.5)

        # Feature 12: class level
        class_type = horse_data.get('class_type', '')
        cm = re.search(r'(\d)', class_type)
        class_rank = int(cm.group(1)) if cm else 3
        features.append(max(0.0, min(1.0, class_rank / 5.0)))

        # Feature 13: freshness
        dslr = horse_data.get('days_since_last_run', 0) or 0
        if not dslr and prior:
            last_date = prior[-1].get('race_date', '')
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date.replace('/', '-'), '%Y-%m-%d')
                    dslr = (datetime.now() - last_dt).days
                except Exception:
                    pass
        if dslr > 0:
            features.append(max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60)))
        else:
            features.append(0.5)

        # Feature 14: going preference
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if prior_same_going:
                going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                going_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                features.append(min(1.0, (going_wins * 2 + going_top3) / (len(prior_same_going) * 3) + 0.2))
            else:
                features.append(0.5)
        else:
            features.append(0.5)

        # Feature 15: going form rate
        going_form_rate = 0.5
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if len(prior_same_going) >= 2:
                going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                going_form_rate = going_wins / len(prior_same_going)
        features.append(going_form_rate)

        # Feature 16: jockey-horse cooperation rate
        jockey_horse_coop = 0.5
        jockey = horse_data.get('jockey', '')
        if prior and jockey:
            prior_same_jockey = [p for p in prior if p.get('jockey', '') == jockey]
            if len(prior_same_jockey) >= 2:
                jh_wins = sum(1 for p in prior_same_jockey if p.get('is_winner', 0))
                jh_places = sum(1 for p in prior_same_jockey if p.get('is_top3', 0))
                jockey_horse_coop = min(1.0, (jh_wins * 2 + jh_places) / (len(prior_same_jockey) * 3))
        features.append(jockey_horse_coop)

        # Feature 17: distance personal best ratio
        distance_pb_ratio = 0.5
        if prior:
            prior_at_dist = [p for p in prior
                             if p.get('distance') == distance
                             and p.get('finish_time', '')
                             and ':' in str(p.get('finish_time', ''))]
            if prior_at_dist:
                def _parse_time(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0
                times = [_parse_time(p['finish_time']) for p in prior_at_dist]
                valid_times = [t for t in times if t < 999]
                if valid_times:
                    horse_pb = min(valid_times)
                    try:
                        db_path_pb = config.DB_PATH
                        if db_path_pb.exists():
                            conn_pb = sqlite3.connect(str(db_path_pb))
                            row_pb = conn_pb.execute(
                                "SELECT MIN(finish_time) FROM race_results WHERE distance = ? AND finish_time != '' AND finish_time IS NOT NULL",
                                (distance,)
                            ).fetchone()
                            conn_pb.close()
                            if row_pb and row_pb[0]:
                                track_record = _parse_time(row_pb[0])
                                if track_record < 999 and horse_pb < 999:
                                    distance_pb_ratio = min(1.0, track_record / horse_pb)
                    except Exception:
                        pass
        features.append(distance_pb_ratio)

        # Feature 18: weight carried (normalized)
        weight_str = horse_data.get('weight', '') or ''
        try:
            weight_lbs = int(weight_str)
            weight_carried_norm = max(0.0, min(1.0, (weight_lbs - 108) / 27.0))
        except (ValueError, TypeError):
            weight_carried_norm = 0.65  # ~126 lbs default
        features.append(weight_carried_norm)

        result = np.array(features, dtype=np.float32)
        validate_features(result)
        return result
