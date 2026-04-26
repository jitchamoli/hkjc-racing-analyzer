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
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np

import config
from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES
from ml.features import FeatureEngineering, extract_class_rank

logger = logging.getLogger(__name__)


def _date_30d_before(race_date: str) -> str:
    """Return YYYY/MM/DD 30 days before race_date. Used by Feature 19."""
    try:
        dt = datetime.strptime(race_date.replace('/', '-'), '%Y-%m-%d')
        return (dt - timedelta(days=30)).strftime('%Y/%m/%d')
    except (ValueError, TypeError):
        return '0000/00/00'


def build_training_data_from_db(years_back: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Build training data from real race results in horse_database.

    Args:
        years_back: if set, only include races within this many years of the
            most-recent race. Older races stay in race_results for inference-
            time history lookups but are excluded from training. Empirically
            (2026-04-25) training on 49 years gave ROI +927% vs +1359% for
            the recent-only window — older eras dilute current-market signal.

    Returns (features, win_labels, sample_weights, place_labels) or (None, None, None, None).
    """
    conn = None
    try:
        db_path = config.DB_PATH
        if not db_path.exists():
            return None, None, None, None

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Optional recency cutoff
        cutoff_clause = ""
        cutoff_args: tuple = ()
        if years_back:
            from datetime import datetime, timedelta
            max_date = conn.execute("SELECT MAX(race_date) FROM race_results").fetchone()[0]
            if max_date:
                # Dates are YYYY/MM/DD; parse, subtract years, format back
                mx = datetime.strptime(max_date.replace('/', '-'), '%Y-%m-%d')
                cutoff_dt = mx - timedelta(days=365 * years_back)
                cutoff_str = cutoff_dt.strftime('%Y/%m/%d')
                cutoff_clause = " AND r.race_date >= ?"
                cutoff_args = (cutoff_str,)
                logger.info(f"Training data recency cutoff: race_date >= {cutoff_str} ({years_back}y back from {max_date})")

        rows = conn.execute(f"""
            SELECT r.*
            FROM race_results r
            WHERE r.finish_position > 0{cutoff_clause}
            ORDER BY r.race_date, r.race_number
        """, cutoff_args).fetchall()

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

        # field_rel_speed was precomputed here (race_finish_times lookup).
        # Dropped Wave 2c: r=0.718 with recent_form, r=0.710 with last_start_margin.
        def _parse_secs(ft_str):
            try:
                parts = str(ft_str).split(':')
                s = float(parts[0]) * 60 + float(parts[1])
                return s if 0 < s < 999 else None
            except (ValueError, IndexError):
                return None

        # v6.4: per-row exponential recency weights computed post-loop with np.exp.
        # par_times precompute removed v6.4 — only adjusted_speed used it.

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

        # Build per-jockey/trainer chronological history for PRIOR-only win rate calc
        jockey_history = {}
        trainer_history = {}
        for row in rows:
            r = dict(row)
            j = r.get('jockey', '') or ''
            t = r.get('trainer', '') or ''
            rd = r.get('race_date', '')
            is_win = 1 if r.get('is_winner', 0) else 0
            if j:
                jockey_history.setdefault(j, []).append((rd, is_win))
            if t:
                trainer_history.setdefault(t, []).append((rd, is_win))

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
            # Lightly-raced horses (≤3 starts) get floor of 0.35 — most HK winners
            # have 0 prior wins in their current class
            if prior:
                prior_wins = sum(1 for p in prior if p.get('is_winner', 0))
                prior_top3 = sum(1 for p in prior if p.get('is_top3', 0))
                feat_form_proxy = min(1.0, (prior_wins * 2 + prior_top3) / (len(prior) * 3))
                if len(prior) <= 3 and feat_form_proxy < 0.35:
                    feat_form_proxy = 0.35
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

            # Feature 4: jockey win rate (prior-only, Bayesian-smoothed)
            jockey_name = r.get('jockey', '') or ''
            jockey_prior = [(d, w) for d, w in jockey_history.get(jockey_name, []) if d < race_date]
            j_rides = len(jockey_prior)
            j_wins = sum(w for _, w in jockey_prior)
            # Beta prior at HK league avg (0.08) with strength 15 — matches features.py
            feat_jockey = min(0.5, (j_wins + 0.08 * 15) / (j_rides + 15))

            # Feature 5: trainer win rate (prior-only, Bayesian-smoothed)
            trainer_name = r.get('trainer', '') or ''
            trainer_prior = [(d, w) for d, w in trainer_history.get(trainer_name, []) if d < race_date]
            t_rides = len(trainer_prior)
            t_wins = sum(w for _, w in trainer_prior)
            feat_trainer = min(0.4, (t_wins + 0.08 * 15) / (t_rides + 15))

            # Feature 6: field size — clipped to [0,1]; legacy data has stray
            # rows with field_size > 14 (oversize fields pre-2000s).
            feat_field_size = min(1.0, field_size / 14.0)

            # Feature 7: field-relative draw — clipped to [0,1]; old DB rows
            # had draw values up to 28 from oversize legacy fields, which
            # pushed feat_field_draw to 15.0 in training (matches F-16 fix
            # already applied at inference in features.py).
            feat_field_draw = min(1.0, draw / max(1, field_size)) if draw > 0 else 0.5

            # prior_pace, adjusted_speed dropped v6.4 — see feature_schema.py.

            # Feature 8: C&D form (winning trip replication boost)
            prior_at_dist = [p for p in prior if p.get('distance') == distance]
            if prior_at_dist:
                cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
                feat_cd = cd_wins / len(prior_at_dist) + 0.2
                # Boost if won at this distance recently (within 180 days)
                if cd_wins > 0 and race_date:
                    for p in reversed(prior_at_dist):
                        if p.get('is_winner', 0):
                            win_date = p.get('race_date', '')
                            if win_date:
                                try:
                                    w_dt = datetime.strptime(win_date.replace('/', '-'), '%Y-%m-%d')
                                    r_dt = datetime.strptime(race_date.replace('/', '-'), '%Y-%m-%d')
                                    if (r_dt - w_dt).days <= 180:
                                        feat_cd += 0.15
                                except Exception:
                                    pass
                            break
                feat_cd = min(1.0, feat_cd)
            else:
                feat_cd = 0.5

            # Feature 9: class level (INVERTED: Group 1/2=1.0, Class 1=0.8, Class 5=0.0)
            # Uses extract_class_rank to handle G1/G2/G3/GRIFFIN/4YO correctly.
            class_type = r.get('class_type', '') or ''
            class_rank_or_none = extract_class_rank(class_type)
            class_rank = class_rank_or_none if class_rank_or_none is not None else 3
            feat_class = max(0.0, min(1.0, 1.0 - class_rank / 5.0))

            # Feature 10: freshness
            # 87.7% of race_results rows have days_since_last_run = 0/null (it's
            # only populated for races where predictions were generated). Falling
            # back to 0.5 there made freshness a near-constant feature (0.0
            # importance in v6.3). Compute from prior race date when the DB
            # column is missing — same fallback as features.py at inference.
            dslr = r.get('days_since_last_run', 0) or 0
            if not dslr and prior:
                last_date = prior[-1].get('race_date', '') or ''
                if last_date and race_date:
                    try:
                        last_dt = datetime.strptime(last_date.replace('/', '-'), '%Y-%m-%d')
                        curr_dt = datetime.strptime(race_date.replace('/', '-'), '%Y-%m-%d')
                        dslr = (curr_dt - last_dt).days
                    except Exception:
                        dslr = 0
            if dslr > 0:
                feat_freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))
            else:
                feat_freshness = 0.5

            # Feature 11: going performance (merged — win rate + place rate on this going)
            going_perf = 0.5
            if prior and going:
                prior_same_going = [p for p in prior if p.get('going', '') == going]
                if prior_same_going:
                    g_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                    g_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                    n = len(prior_same_going)
                    going_perf = min(1.0, (g_wins / n) * 0.6 * 5 + (g_top3 / n) * 0.4 * 2)

            # Feature 12: weight vs field (carried weight relative to field median)
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

            # Feature 13: jockey-horse cooperation
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

            # distance_pb_ratio dropped v6.4 — see feature_schema.py.

            # Feature 14: last start margin (top importance ~16% in v6.4)
            feat_last_margin = 0.5
            if prior:
                last_race = prior[-1]
                m = last_race.get('margin', 0) or 0
                if last_race.get('is_winner', 0):
                    feat_last_margin = 1.0
                elif m > 0:
                    feat_last_margin = max(0.1, min(0.9, 1.0 - m / 12.0))

            # Feature 15: class change — ternary {dropping=1.0, same=0.5, rising=0.0}
            # Uses extract_class_rank — Group races (G1/G2/G3) properly placed
            # above the digit ladder, GRIFFIN/MAIDEN/4YO handled. M15: when
            # EITHER side is unknown (current race or prior had unparseable
            # class_type), fall back to 0.5 — don't compare against the
            # default-3 placeholder, which would silently encode wrong
            # direction for Group races (G1 → unparseable → falls back to
            # rank=3 → "dropping" vs Class 1).
            feat_class_change = 0.5
            if prior and class_rank_or_none is not None:
                last_class_str = prior[-1].get('class_type', '') or ''
                last_class_rank = extract_class_rank(last_class_str)
                if last_class_rank is None:
                    feat_class_change = 0.5
                elif last_class_rank > class_rank_or_none:
                    feat_class_change = 0.0  # rising (harder race) — negative signal
                elif last_class_rank < class_rank_or_none:
                    feat_class_change = 1.0  # dropping (easier race) — positive signal
                else:
                    feat_class_change = 0.5  # same class

            # days_since_last_win (was Feature 19) dropped Wave 2c — see feature_schema.py.

            # Feature 16: horse body weight change (fitness signal)
            feat_hw_change = 0.5
            hw_current = r.get('horse_weight', 0) or 0
            if prior and hw_current > 0:
                prior_hws = [p.get('horse_weight', 0) or 0 for p in prior[-3:] if (p.get('horse_weight', 0) or 0) > 0]
                if prior_hws:
                    avg_prior_hw = sum(prior_hws) / len(prior_hws)
                    if avg_prior_hw > 0:
                        pct_change = (hw_current - avg_prior_hw) / avg_prior_hw
                        feat_hw_change = max(0.0, min(1.0, 0.5 - pct_change * 10))

            # speed_consistency dropped v6.4.

            # Feature 17 (v6.5): horse age. 3yo→1.0, 9yo+→0.0. Default 0.5.
            age_raw = r.get('age', 0) or 0
            try:
                age_int = int(str(age_raw).strip()[:2]) if age_raw else 0
            except (ValueError, TypeError):
                age_int = 0
            if age_int and 2 <= age_int <= 15:
                feat_age = max(0.0, min(1.0, (9 - age_int) / 6.0))
            else:
                feat_age = 0.5

            # Feature 18 (v6.5): rating change from previous race. +10pts→1.0,
            # 0→0.5, -10→0.0. Default 0.5 when missing.
            rc_raw = r.get('rating_change', 0) or 0
            try:
                rc_int = int(rc_raw)
            except (ValueError, TypeError):
                rc_int = 0
            feat_rating_change = max(0.0, min(1.0, 0.5 + rc_int / 20.0))

            # Feature 19 (v6.5): jockey 30-day form (Bayesian-smoothed).
            # Filter jockey_history (already chronological per name) to the
            # 30 days BEFORE race_date. Default 0.08 (HK league avg) when
            # the jockey has no rides in that window.
            j_30d = [(d, w) for d, w in jockey_history.get(jockey_name, [])
                     if d < race_date and d >= _date_30d_before(race_date)]
            j30_rides = len(j_30d)
            j30_wins = sum(w for _, w in j_30d)
            feat_jockey_30d = min(0.6, (j30_wins + 0.08 * 15) / (j30_rides + 15))

            features = [
                feat_form_proxy, feat_form, feat_draw,
                feat_jockey, feat_trainer, feat_field_size,
                feat_field_draw, feat_cd,
                feat_class, feat_freshness, going_perf,
                weight_vs_field, feat_jh_coop,
                feat_last_margin, feat_class_change,
                feat_hw_change,
                feat_age, feat_rating_change, feat_jockey_30d,
            ]
            assert len(features) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(features)}"
            X.append(features)
            y.append(1 if r.get('is_winner', 0) else 0)
            y_place.append(1 if r.get('is_top3', 0) else 0)
            weights.append(0.0)  # placeholder; overwritten below

        # Recency weights — exponential decay so recent races dominate.
        # Diagnostic 2026-04-25: linear 0.3→1.0 over 5y window made the most
        # recent quarter only ~14% of effective training weight. With 94% horse
        # turnover and 50% jockey turnover since 2021, older rows describe a
        # different field than the one we're predicting on. Switched to
        # exponential half-life of n/3 so the newest third weighs ~2× the
        # oldest third (was 1.5× under linear 0.3→1.0).
        n = len(weights)
        if n > 1:
            # Exponential: w(i) = 0.25 + 0.75 * exp((i - (n-1)) / (n/3))
            # Newest row (i=n-1) weight = 1.0; oldest (i=0) ≈ 0.29 (≈ exp(-3))
            idx = np.arange(n, dtype=np.float32)
            decay = np.exp((idx - (n - 1)) / (n / 3.0))
            row_weights = (0.25 + 0.75 * decay).astype(np.float32)
        else:
            row_weights = np.ones(n, dtype=np.float32)
        weights = list(row_weights)

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
        cd_form = np.random.uniform(0.2, 1.0)
        class_level = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.40, 0.25, 0.15]) / 5.0
        dslr = np.random.choice([7, 14, 21, 28, 45, 90], p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])
        freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))
        going_perf = np.random.uniform(0.2, 0.8)
        weight_vs_field = np.random.normal(0.5, 0.15)
        weight_vs_field = np.clip(weight_vs_field, 0.0, 1.0)
        jockey_horse_coop = np.random.uniform(0.0, 0.5)

        prior_win_rate = np.random.beta(2, 8)
        last_margin = np.random.beta(3, 3)
        class_change_feat = np.random.choice([0.0, 0.5, 1.0], p=[0.15, 0.60, 0.25])
        hw_change = np.random.normal(0.5, 0.1)
        hw_change = np.clip(hw_change, 0.0, 1.0)

        features = [
            prior_win_rate, form_score / 100, draw_adv,
            jockey_rate, trainer_rate, feat_field_size,
            feat_field_draw, cd_form,
            1.0 - class_level,
            freshness, going_perf,
            weight_vs_field, jockey_horse_coop,
            last_margin, class_change_feat,
            hw_change,
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
            + 0.5 * (cd_form - 0.5)
            + 0.3 * ((1.0 - class_level) - 0.5)
            + 0.2 * (freshness - 0.5)
            + 0.5 * (going_perf - 0.5)
            + 0.6 * (weight_vs_field - 0.5)
            + 0.3 * (jockey_horse_coop - 0.25)
            + 1.5 * (last_margin - 0.5)
            + 1.2 * (class_change_feat - 0.5)
            + 0.3 * (hw_change - 0.5)
            - 2.8
        )
        win_prob = 1.0 / (1.0 + np.exp(-win_logit))
        is_win = np.random.random() < win_prob
        y.append(1 if is_win else 0)

    return np.array(X, dtype=np.float32), np.array(y)
