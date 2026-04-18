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
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np

import config
from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES
from ml.features import FeatureEngineering

logger = logging.getLogger(__name__)


def build_training_data_from_db() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Build training data from real race results in horse_database.

    Returns (features, win_labels, sample_weights, place_labels) or (None, None, None, None).
    """
    conn = None
    try:
        db_path = config.DB_PATH
        if not db_path.exists():
            return None, None, None, None

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
            return None, None, None, None

        logger.info(f"Building training data from {len(rows)} real race results")

        # Precompute field sizes and median weights per race
        field_sizes = {}
        race_weights = {}  # race_key → list of carried weights
        for row in rows:
            r = dict(row)
            rk = f"{r.get('race_date', '')}-{r.get('race_number', 0)}"
            field_sizes[rk] = field_sizes.get(rk, 0) + 1
            wt = r.get('weight', 0) or 0
            try:
                wt = int(wt)
            except (ValueError, TypeError):
                wt = 0
            if wt > 0:
                race_weights.setdefault(rk, []).append(wt)
        field_median_weights = {}
        for rk, wts in race_weights.items():
            sw = sorted(wts)
            field_median_weights[rk] = sw[len(sw) // 2]

        # Precompute date-aware track records per distance
        # For each distance, store chronologically sorted (date, best_time_so_far)
        # so we can look up the track record as of any given race_date
        _dist_times = {}  # distance → list of (date, seconds)
        tr_rows = conn.execute(
            "SELECT race_date, distance, finish_time FROM race_results "
            "WHERE finish_time != '' AND finish_time IS NOT NULL AND finish_time LIKE '%:%' "
            "ORDER BY race_date"
        ).fetchall()
        for tr_r in tr_rows:
            rd, d, ft = tr_r[0], tr_r[1], tr_r[2]
            if not d or not rd:
                continue
            try:
                parts = str(ft).split(':')
                secs = float(parts[0]) * 60 + float(parts[1])
                if 0 < secs < 999:
                    _dist_times.setdefault(d, []).append((rd, secs))
            except (ValueError, IndexError):
                pass
        # Build cumulative best per distance (running minimum by date)
        track_records_by_date = {}  # distance → list of (date, cumulative_best_time)
        for d, entries in _dist_times.items():
            entries.sort(key=lambda x: x[0])
            best = float('inf')
            cumulative = []
            for rd, secs in entries:
                if secs < best:
                    best = secs
                cumulative.append((rd, best))
            track_records_by_date[d] = cumulative

        def _get_track_record(dist, before_date):
            """Get best finish time at this distance before the given date."""
            entries = track_records_by_date.get(dist, [])
            best = None
            for rd, cum_best in entries:
                if rd >= before_date:
                    break
                best = cum_best
            return best

        # Precompute all finish times per race (for field-relative speed)
        def _parse_secs(ft_str):
            try:
                parts = str(ft_str).split(':')
                s = float(parts[0]) * 60 + float(parts[1])
                return s if 0 < s < 999 else None
            except (ValueError, IndexError):
                return None

        race_finish_times = {}  # race_key → list of seconds
        for row in rows:
            r = dict(row)
            rk = f"{r.get('race_date', '')}-{r.get('race_number', 0)}"
            ft = r.get('finish_time', '') or ''
            if ft and ':' in ft:
                s = _parse_secs(ft)
                if s is not None:
                    race_finish_times.setdefault(rk, []).append(s)

        # Compute recency weights: newest race = 1.0, oldest decays to 0.3
        all_dates = sorted(set(dict(row).get('race_date', '') for row in rows))
        if len(all_dates) > 1:
            date_rank = {d: i for i, d in enumerate(all_dates)}
            max_rank = len(all_dates) - 1
        else:
            date_rank = {all_dates[0]: 0} if all_dates else {}
            max_rank = 1

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

        X, y, y_place, weights = [], [], [], []
        for row in rows:
            r = dict(row)
            draw = r.get('draw', 0) or 0
            distance = r.get('distance', 1200) or 1200
            jockey_wr = r.get('jockey_wr', 0) or 0
            trainer_wr = r.get('trainer_wr', 0) or 0
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

            # Feature 11: class level (INVERTED: Class 1=0.8, Class 5=0.0)
            class_type = r.get('class_type', '') or ''
            cm = re.search(r'(\d)', class_type)
            class_rank = int(cm.group(1)) if cm else 3
            feat_class = max(0.0, min(1.0, 1.0 - class_rank / 5.0))

            # Feature 12: freshness
            dslr = r.get('days_since_last_run', 0) or 0
            if dslr > 0:
                feat_freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))
            else:
                feat_freshness = 0.5

            # Feature 13: going performance (merged — win rate + place rate on this going)
            going_perf = 0.5
            if prior and going:
                prior_same_going = [p for p in prior if p.get('going', '') == going]
                if prior_same_going:
                    g_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                    g_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                    n = len(prior_same_going)
                    going_perf = min(1.0, (g_wins / n) * 0.6 * 5 + (g_top3 / n) * 0.4 * 2)

            # Feature 14: weight vs field (carried weight relative to field median)
            weight_vs_field = 0.5
            weight_val = r.get('weight', 0) or 0
            try:
                weight_val = int(weight_val)
            except (ValueError, TypeError):
                weight_val = 0
            if weight_val > 0:
                median_wt = field_median_weights.get(race_key)
                if median_wt:
                    diff = weight_val - median_wt
                    weight_vs_field = max(0.0, min(1.0, 0.5 - diff / 20.0))
                else:
                    weight_vs_field = max(0.0, min(1.0, 1.0 - (weight_val - 108) / 27.0))

            # Feature 15: jockey-horse cooperation
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

            # Feature 16: distance PB ratio (uses precomputed track records)
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
                    best_tr = _get_track_record(distance, race_date)
                    if best_tr is not None and horse_pb_t < 999:
                        feat_dist_pb = min(1.0, best_tr / horse_pb_t)

            # Feature 17: field-relative speed (avg percentile within prior race fields)
            feat_frs = 0.5
            if prior:
                percentiles = []
                for p in prior[-5:]:
                    p_ft = p.get('finish_time', '') or ''
                    if not (p_ft and ':' in p_ft):
                        continue
                    horse_s = _parse_secs(p_ft)
                    if horse_s is None:
                        continue
                    p_rk = f"{p.get('race_date','')}-{p.get('race_number',0)}"
                    all_secs = race_finish_times.get(p_rk, [])
                    if len(all_secs) >= 3:
                        rank = sum(1 for s in all_secs if s > horse_s)
                        percentiles.append(rank / (len(all_secs) - 1) if len(all_secs) > 1 else 0.5)
                if percentiles:
                    feat_frs = sum(percentiles) / len(percentiles)

            # Feature 18: last start margin (NEW)
            feat_last_margin = 0.5
            if prior:
                last_race = prior[-1]
                m = last_race.get('margin', 0) or 0
                if last_race.get('is_winner', 0):
                    feat_last_margin = 1.0
                elif m > 0:
                    feat_last_margin = max(0.1, min(0.9, 1.0 - m / 12.0))

            # Feature 19: class change (drop = positive signal)
            feat_class_change = 0.5
            if prior:
                last_class_str = prior[-1].get('class_type', '') or ''
                last_cm = re.search(r'(\d)', last_class_str)
                last_class_rank = int(last_cm.group(1)) if last_cm else 3
                diff = class_rank - last_class_rank  # positive = dropped in class
                feat_class_change = max(0.0, min(1.0, 0.5 + diff * 0.125))

            # Feature 20: days since last win (recency of winning)
            feat_days_last_win = 0.5
            if prior:
                for p in reversed(prior):
                    if p.get('is_winner', 0):
                        win_date = p.get('race_date', '')
                        if win_date and race_date:
                            try:
                                w_dt = datetime.strptime(win_date.replace('/', '-'), '%Y-%m-%d')
                                r_dt = datetime.strptime(race_date.replace('/', '-'), '%Y-%m-%d')
                                days = (r_dt - w_dt).days
                                feat_days_last_win = max(0.1, min(0.95, 1.0 - days / 350.0))
                            except Exception:
                                pass
                        break
                else:
                    feat_days_last_win = 0.15

            # Feature 21: jockey × trainer combo (interaction)
            feat_jt_combo = 0.5
            if feat_jockey > 0 and feat_trainer > 0:
                feat_jt_combo = min(1.0, (feat_jockey * feat_trainer) ** 0.5 * 5.0)

            # Feature 22: horse body weight change (fitness signal)
            feat_hw_change = 0.5
            hw_current = r.get('horse_weight', 0) or 0
            if prior and hw_current > 0:
                prior_hws = [p.get('horse_weight', 0) or 0 for p in prior[-3:] if (p.get('horse_weight', 0) or 0) > 0]
                if prior_hws:
                    avg_prior_hw = sum(prior_hws) / len(prior_hws)
                    if avg_prior_hw > 0:
                        pct_change = (hw_current - avg_prior_hw) / avg_prior_hw
                        feat_hw_change = max(0.0, min(1.0, 0.5 - pct_change * 10))

            features = [
                feat_form_proxy, feat_form, feat_draw,
                feat_jockey, feat_trainer, feat_field_size,
                feat_field_draw, feat_pace, feat_finish, feat_cd,
                feat_class, feat_freshness, going_perf,
                weight_vs_field, feat_jh_coop, feat_dist_pb, feat_frs,
                feat_last_margin, feat_class_change, feat_days_last_win,
                feat_jt_combo, feat_hw_change,
            ]
            assert len(features) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(features)}"
            X.append(features)
            y.append(1 if r.get('is_winner', 0) else 0)
            y_place.append(1 if r.get('is_top3', 0) else 0)
            # Recency weighting: newest races = 1.0, oldest decay to 0.3
            rank = date_rank.get(race_date, 0)
            weights.append(0.3 + 0.7 * (rank / max(1, max_rank)))

        conn.close()

    except Exception as e:
        logger.warning(f"Error building training data from DB: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return None, None, None, None

    return np.array(X, dtype=np.float32), np.array(y), np.array(weights), np.array(y_place)


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
        going_perf = np.random.uniform(0.2, 0.8)  # merged going performance
        weight_vs_field = np.random.normal(0.5, 0.15)  # relative to field median
        weight_vs_field = np.clip(weight_vs_field, 0.0, 1.0)
        jockey_horse_coop = np.random.uniform(0.0, 0.5)
        distance_pb_ratio = np.random.uniform(0.7, 1.0)
        field_rel_speed = np.random.beta(3, 3)  # centered, most horses mid-pack

        # Feature 1 must be prior_win_rate (not rating/100) to match DB training
        prior_win_rate = np.random.beta(2, 8)  # right-skewed, most values 0.05-0.30
        last_margin = np.random.beta(3, 3)  # 0-1, centered around 0.5
        class_change = np.random.choice([-2, -1, 0, 1, 2], p=[0.05, 0.15, 0.60, 0.15, 0.05])
        class_change_feat = max(0.0, min(1.0, 0.5 + class_change * 0.125))
        days_last_win = np.random.beta(2, 4)  # right-skewed, most haven't won recently
        days_last_win = np.clip(days_last_win * 0.85 + 0.1, 0.1, 0.95)
        jt_combo = min(1.0, (jockey_rate * trainer_rate) ** 0.5 * 5.0)
        hw_change = np.random.normal(0.5, 0.1)  # body weight change
        hw_change = np.clip(hw_change, 0.0, 1.0)

        features = [
            prior_win_rate, form_score / 100, draw_adv,
            jockey_rate, trainer_rate, feat_field_size,
            feat_field_draw, prior_pace, prior_speed, cd_form,
            1.0 - class_level,  # inverted: Class 1=high, Class 5=low
            freshness, going_perf,
            weight_vs_field, jockey_horse_coop, distance_pb_ratio, field_rel_speed,
            last_margin, class_change_feat, days_last_win,
            jt_combo, hw_change,
        ]
        assert len(features) == NUM_FEATURES
        X.append(features)

        win_logit = (
            2.5 * (prior_win_rate - 0.15)
            + 1.2 * (form_score / 100 - 0.4)
            + 0.5 * (draw_adv - 0.6)
            + 3.0 * (jockey_rate - 0.10)
            + 2.0 * (trainer_rate - 0.08)
            + 0.3 * (feat_field_size - 0.8)
            + 0.4 * (feat_field_draw - 0.5)
            + 0.5 * (prior_pace - 0.5)
            + 0.3 * (prior_speed - 0.5)
            + 0.5 * (cd_form - 0.5)
            # odds_implied removed — no longer a feature
            + 0.3 * ((1.0 - class_level) - 0.5)
            + 0.2 * (freshness - 0.5)
            + 0.5 * (going_perf - 0.5)
            + 0.6 * (weight_vs_field - 0.5)  # lighter than field = edge
            + 0.3 * (jockey_horse_coop - 0.25)
            + 0.2 * (distance_pb_ratio - 0.85)
            + 0.8 * (field_rel_speed - 0.5)  # faster than field = strong
            + 1.5 * (last_margin - 0.5)  # strong signal
            + 1.2 * (class_change_feat - 0.5)  # class drop = strong edge
            + 0.8 * (days_last_win - 0.4)  # recent winners
            + 1.0 * (jt_combo - 0.3)  # strong jockey+trainer combos
            + 0.3 * (hw_change - 0.5)  # body weight fitness
            - 2.8
        )
        win_prob = 1.0 / (1.0 + np.exp(-win_logit))
        is_win = np.random.random() < win_prob
        y.append(1 if is_win else 0)

    return np.array(X, dtype=np.float32), np.array(y)
