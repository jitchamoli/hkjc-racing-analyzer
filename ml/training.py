#!/usr/bin/env python3
"""Training data builders for the ML model.

Three sources in priority order:
1. SQLite DB (build_training_data_from_db) — preferred when >= MIN_DB_SAMPLES
2. Scraped race data (build_training_data) — fallback
3. Synthetic data (generate_synthetic_training_data) — last resort
"""

import re
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np

import config
from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES
from ml.features import FeatureEngineering

logger = logging.getLogger(__name__)


def build_training_data_from_db() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Build training data from real race results in horse_database.

    Returns (features, labels, sample_weights) or (None, None, None) if insufficient data.
    """
    conn = None
    try:
        db_path = config.DB_PATH
        if not db_path.exists():
            return None, None, None

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT r.*, j.win_rate as jockey_wr, j.place_rate as jockey_pr,
                   t.win_rate as trainer_wr, t.place_rate as trainer_pr
            FROM race_results r
            LEFT JOIN jockey_stats j ON r.jockey = j.name
            LEFT JOIN trainer_stats t ON r.trainer = t.name
            WHERE r.finish_position > 0
            ORDER BY r.race_date, r.race_number
        """).fetchall()

        if len(rows) < config.MIN_REAL_SAMPLES:
            logger.info(f"Only {len(rows)} results in DB — not enough for real training")
            return None, None, None

        logger.info(f"Building training data from {len(rows)} real race results")

        # Precompute field sizes per race
        field_sizes = {}
        for row in rows:
            r = dict(row)
            rk = f"{r.get('race_date', '')}-{r.get('race_number', 0)}"
            field_sizes[rk] = field_sizes.get(rk, 0) + 1

        # Build per-horse chronological race history for PRIOR-only lookups
        horse_history = {}
        for row in rows:
            r = dict(row)
            name = r.get('horse_name', '')
            if name:
                if name not in horse_history:
                    horse_history[name] = []
                horse_history[name].append(r)
        for name in horse_history:
            horse_history[name].sort(key=lambda x: x.get('race_date', ''))

        X, y, weights = [], [], []
        for row in rows:
            r = dict(row)
            draw = r.get('draw', 0) or 0
            distance = r.get('distance', 1200) or 1200
            jockey_wr = r.get('jockey_wr', 0) or 0
            trainer_wr = r.get('trainer_wr', 0) or 0
            win_odds = r.get('win_odds', 0) or 0
            horse_name = r.get('horse_name', '')
            race_date = r.get('race_date', '')
            race_no = r.get('race_number', 0)
            race_key = f"{race_date}-{race_no}"
            field_size = field_sizes.get(race_key, 12)
            going = r.get('going', '') or ''

            prior = [h for h in horse_history.get(horse_name, [])
                     if h.get('race_date', '') < race_date]

            # Feature 1: prior win rate
            if prior:
                prior_wins = sum(1 for p in prior if p.get('is_winner', 0))
                prior_top3 = sum(1 for p in prior if p.get('is_top3', 0))
                feat_form_proxy = min(1.0, (prior_wins * 2 + prior_top3) / (len(prior) * 3))
            else:
                feat_form_proxy = 0.5

            # Feature 2: recent form
            if prior:
                recent = prior[-6:]
                form_pts = 0
                for p in recent:
                    pos = p.get('finish_position', 7)
                    if pos == 1: form_pts += 10
                    elif pos == 2: form_pts += 7
                    elif pos == 3: form_pts += 5
                    elif pos <= 5: form_pts += 2
                feat_form = form_pts / (len(recent) * 10)
            else:
                feat_form = 0.5

            # Feature 3: draw advantage
            feat_draw = FeatureEngineering.calculate_draw_advantage(draw, distance)

            # Feature 4: jockey win rate
            feat_jockey = min(0.5, jockey_wr) if jockey_wr > 0 else 0.08

            # Feature 5: trainer win rate
            feat_trainer = min(0.4, trainer_wr) if trainer_wr > 0 else 0.06

            # Feature 6: field size
            feat_field_size = field_size / 14.0

            # Feature 7: field-relative draw
            feat_field_draw = draw / max(1, field_size) if draw > 0 else 0.5

            # Feature 8: prior pace
            pace_vals = []
            for p in prior[-5:]:
                rp = p.get('running_position', '') or ''
                fs = field_sizes.get(f"{p.get('race_date','')}-{p.get('race_number',0)}", 12)
                if rp:
                    if ' ' in rp:
                        positions = [int(x) for x in rp.split() if x.isdigit()]
                    else:
                        positions = [int(c) for c in rp if c.isdigit()]
                    if positions and fs > 1:
                        early = min(positions[0], fs)
                        pace_vals.append(1.0 - (early - 1) / max(1, fs - 1))
            feat_pace = sum(pace_vals) / len(pace_vals) if pace_vals else 0.5

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
            feat_finish = sum(speed_vals) / len(speed_vals) if speed_vals else 0.5

            # Feature 10: C&D form
            prior_at_dist = [p for p in prior if p.get('distance') == distance]
            if prior_at_dist:
                cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
                feat_cd = min(1.0, cd_wins / len(prior_at_dist) + 0.2)
            else:
                feat_cd = 0.5

            # Feature 11: odds-implied probability
            feat_odds = min(1.0, 1.0 / win_odds) if win_odds > 0 else 0.5

            # Feature 12: class level
            class_type = r.get('class_type', '') or ''
            cm = re.search(r'(\d)', class_type)
            class_rank = int(cm.group(1)) if cm else 3
            feat_class = max(0.0, min(1.0, class_rank / 5.0))

            # Feature 13: freshness
            dslr = r.get('days_since_last_run', 0) or 0
            if dslr > 0:
                feat_freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))
            else:
                feat_freshness = 0.5

            # Feature 14: going preference
            if prior and going:
                prior_same_going = [p for p in prior if p.get('going', '') == going]
                if prior_same_going:
                    going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                    going_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                    feat_going = min(1.0, (going_wins * 2 + going_top3) / (len(prior_same_going) * 3) + 0.2)
                else:
                    feat_going = 0.5
            else:
                feat_going = 0.5

            # Feature 15: going form rate
            if prior and going:
                prior_same_going_f = [p for p in prior if p.get('going', '') == going]
                if len(prior_same_going_f) >= 2:
                    gf_wins = sum(1 for p in prior_same_going_f if p.get('is_winner', 0))
                    feat_going_form = gf_wins / len(prior_same_going_f)
                else:
                    feat_going_form = 0.5
            else:
                feat_going_form = 0.5

            # Feature 16: jockey-horse cooperation
            jockey_name = r.get('jockey', '') or ''
            if prior and jockey_name:
                prior_same_jh = [p for p in prior if p.get('jockey', '') == jockey_name]
                if len(prior_same_jh) >= 2:
                    jh_w = sum(1 for p in prior_same_jh if p.get('is_winner', 0))
                    jh_p = sum(1 for p in prior_same_jh if p.get('is_top3', 0))
                    feat_jh_coop = min(1.0, (jh_w * 2 + jh_p) / (len(prior_same_jh) * 3))
                else:
                    feat_jh_coop = 0.5
            else:
                feat_jh_coop = 0.5

            # Feature 17: distance PB ratio
            feat_dist_pb = 0.5
            prior_dist_times = [p for p in prior
                                if p.get('distance') == distance
                                and p.get('finish_time', '')
                                and ':' in str(p.get('finish_time', ''))]
            if prior_dist_times:
                def _pt(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0
                horse_times = [_pt(p['finish_time']) for p in prior_dist_times]
                valid_ht = [t for t in horse_times if t < 999]
                if valid_ht:
                    horse_pb_t = min(valid_ht)
                    try:
                        tr_row = conn.execute(
                            "SELECT MIN(finish_time) FROM race_results WHERE distance = ? AND finish_time != '' AND finish_time IS NOT NULL AND race_date < ?",
                            (distance, race_date)
                        ).fetchone()
                        if tr_row and tr_row[0]:
                            tr_secs = _pt(tr_row[0])
                            if tr_secs < 999 and horse_pb_t < 999:
                                feat_dist_pb = min(1.0, tr_secs / horse_pb_t)
                    except Exception:
                        pass

            # Feature 18: odds movement (not available in historical DB)
            feat_odds_move = 0.5

            features = [
                feat_form_proxy, feat_form, feat_draw,
                feat_jockey, feat_trainer, feat_field_size,
                feat_field_draw, feat_pace, feat_finish, feat_cd, feat_odds,
                feat_class, feat_freshness, feat_going,
                feat_going_form, feat_jh_coop, feat_dist_pb, feat_odds_move,
            ]
            assert len(features) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(features)}"
            X.append(features)
            y.append(1 if r.get('is_winner', 0) else 0)
            weights.append(1.0)

        conn.close()

    except Exception as e:
        logger.warning(f"Error building training data from DB: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return None, None, None

    return np.array(X, dtype=np.float32), np.array(y), np.array(weights)


def generate_synthetic_training_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data calibrated to real HK racing distributions."""
    np.random.seed(42)

    X = []
    y = []

    for _ in range(n_samples):
        rating = np.clip(np.random.normal(60, 18), 15, 100)
        form_score = np.random.beta(2.5, 3) * 100
        draw_adv = np.random.uniform(0.3, 1.0)

        jockey_rate = np.random.choice(
            [np.random.uniform(0.15, 0.25),
             np.random.uniform(0.08, 0.15),
             np.random.uniform(0.03, 0.08)],
            p=[0.15, 0.35, 0.50]
        )
        trainer_rate = np.random.choice(
            [np.random.uniform(0.12, 0.20),
             np.random.uniform(0.06, 0.12),
             np.random.uniform(0.02, 0.06)],
            p=[0.15, 0.40, 0.45]
        )
        field_size = np.random.choice([6, 8, 10, 12, 14], p=[0.05, 0.15, 0.25, 0.35, 0.20])
        feat_field_size = field_size / 14.0
        feat_field_draw = draw_adv * np.random.uniform(0.3, 1.0)
        prior_pace = np.random.uniform(0.3, 0.9)
        prior_speed = np.random.uniform(0.2, 0.9)
        cd_form = np.random.uniform(0.2, 1.0)
        odds_implied = np.random.uniform(0.03, 0.50)
        class_level = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.40, 0.25, 0.15]) / 5.0
        dslr = np.random.choice([7, 14, 21, 28, 45, 90], p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])
        freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))
        going_pref = np.random.uniform(0.3, 0.8)
        going_form_rate = np.random.uniform(0.0, 0.5)
        jockey_horse_coop = np.random.uniform(0.0, 0.5)
        distance_pb_ratio = np.random.uniform(0.7, 1.0)
        odds_movement_feat = np.random.uniform(0.3, 0.7)

        features = [
            rating / 100, form_score / 100, draw_adv,
            jockey_rate, trainer_rate, feat_field_size,
            feat_field_draw, prior_pace, prior_speed, cd_form, odds_implied,
            class_level, freshness, going_pref,
            going_form_rate, jockey_horse_coop, distance_pb_ratio, odds_movement_feat,
        ]
        assert len(features) == NUM_FEATURES
        X.append(features)

        win_logit = (
            1.8 * (rating / 100 - 0.6)
            + 1.2 * (form_score / 100 - 0.4)
            + 0.5 * (draw_adv - 0.6)
            + 3.0 * (jockey_rate - 0.10)
            + 2.0 * (trainer_rate - 0.08)
            + 0.3 * (feat_field_size - 0.8)
            + 0.4 * (feat_field_draw - 0.5)
            + 0.5 * (prior_pace - 0.5)
            + 0.3 * (prior_speed - 0.5)
            + 0.5 * (cd_form - 0.5)
            + 1.5 * (odds_implied - 0.1)
            + 0.3 * (class_level - 0.5)
            + 0.2 * (freshness - 0.5)
            + 0.3 * (going_pref - 0.5)
            + 0.4 * (going_form_rate - 0.25)
            + 0.3 * (jockey_horse_coop - 0.25)
            + 0.2 * (distance_pb_ratio - 0.85)
            + 0.1 * (odds_movement_feat - 0.5)
            - 2.3
        )
        win_prob = 1.0 / (1.0 + np.exp(-win_logit))
        is_win = np.random.random() < win_prob
        y.append(1 if is_win else 0)

    return np.array(X, dtype=np.float32), np.array(y)
