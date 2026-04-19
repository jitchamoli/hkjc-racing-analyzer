#!/usr/bin/env python3
"""Feature engineering for ML predictions.

Extracts 23 features per horse using FEATURE_SCHEMA order.
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
    def calculate_jockey_win_rate(jockey: str, all_results, conn=None,
                                  before_date: str = None) -> float:
        """Calculate jockey win rate from prior race_results (matches training).

        Uses race_results with date filter to avoid leaking future data.
        Falls back to jockey_stats table, then all_results.
        """
        if not jockey:
            return 0.12
        try:
            c = conn
            owns = c is None
            if owns and config.DB_PATH.exists():
                c = sqlite3.connect(str(config.DB_PATH))
                c.row_factory = sqlite3.Row
            if c is not None:
                # Prior-only from race_results (matches training pipeline)
                if before_date:
                    row = c.execute(
                        "SELECT COUNT(*) as rides, SUM(is_winner) as wins "
                        "FROM race_results WHERE jockey = ? AND race_date < ?",
                        (jockey, before_date)
                    ).fetchone()
                else:
                    row = c.execute(
                        "SELECT COUNT(*) as rides, SUM(is_winner) as wins "
                        "FROM race_results WHERE jockey = ?",
                        (jockey,)
                    ).fetchone()
                if owns:
                    c.close()
                if row and row['rides'] and row['rides'] > 0:
                    return min(0.6, row['wins'] / row['rides'])
                # Fallback to stats table if no race_results
                if not owns:
                    stat = conn.execute(
                        "SELECT win_rate FROM jockey_stats WHERE name = ?", (jockey,)
                    ).fetchone()
                    if stat and stat['win_rate'] and stat['win_rate'] > 0:
                        return min(0.6, stat['win_rate'])
        except Exception as e:
            logger.warning(f"DB lookup failed for jockey {jockey}: {e}")
        jockey_rides = [r for r in all_results if jockey.lower() in r.jockey.lower()]
        if not jockey_rides:
            return 0.12
        wins = sum(1 for r in jockey_rides if r.is_win)
        return min(0.6, wins / len(jockey_rides))

    @staticmethod
    def calculate_trainer_win_rate(trainer: str, all_results, conn=None,
                                   before_date: str = None) -> float:
        """Calculate trainer win rate from prior race_results (matches training).

        Uses race_results with date filter to avoid leaking future data.
        Falls back to trainer_stats table, then all_results.
        """
        if not trainer:
            return 0.12
        try:
            c = conn
            owns = c is None
            if owns and config.DB_PATH.exists():
                c = sqlite3.connect(str(config.DB_PATH))
                c.row_factory = sqlite3.Row
            if c is not None:
                if before_date:
                    row = c.execute(
                        "SELECT COUNT(*) as rides, SUM(is_winner) as wins "
                        "FROM race_results WHERE trainer = ? AND race_date < ?",
                        (trainer, before_date)
                    ).fetchone()
                else:
                    row = c.execute(
                        "SELECT COUNT(*) as rides, SUM(is_winner) as wins "
                        "FROM race_results WHERE trainer = ?",
                        (trainer,)
                    ).fetchone()
                if owns:
                    c.close()
                if row and row['rides'] and row['rides'] > 0:
                    return min(0.5, row['wins'] / row['rides'])
                if not owns:
                    stat = conn.execute(
                        "SELECT win_rate FROM trainer_stats WHERE name = ?", (trainer,)
                    ).fetchone()
                    if stat and stat['win_rate'] and stat['win_rate'] > 0:
                        return min(0.5, stat['win_rate'])
        except Exception as e:
            logger.warning(f"DB lookup failed for trainer {trainer}: {e}")
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
    def extract_features(horse_data: dict, all_results, venue: str,
                         conn: sqlite3.Connection = None) -> np.ndarray:
        """Extract 23 features matching FEATURE_SCHEMA order exactly.

        Args:
            conn: Optional shared SQLite connection. If None, opens/closes its own.
        """
        features = []
        horse_name = horse_data.get('name', '')
        distance = horse_data.get('distance', 1200)
        going = horse_data.get('going', '')
        field_size = horse_data.get('field_size', 12)

        # Load prior race history from DB
        prior = []
        race_date = horse_data.get('race_date', '')
        owns_conn = conn is None
        try:
            if conn is None:
                db_path = config.DB_PATH
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
            if conn is not None:
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
        except Exception as e:
            logger.warning(f"Failed to load prior history for {horse_name}: {e}")

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

        # Feature 4: jockey win rate (prior-only to match training)
        jockey = horse_data.get('jockey', '')
        jockey_rate = FeatureEngineering.calculate_jockey_win_rate(
            jockey, all_results, conn=conn, before_date=race_date or None)
        features.append(min(0.5, jockey_rate) if jockey_rate > 0 else 0.08)

        # Feature 5: trainer win rate (prior-only to match training)
        trainer = horse_data.get('trainer', '')
        trainer_rate = FeatureEngineering.calculate_trainer_win_rate(
            trainer, all_results, conn=conn, before_date=race_date or None)
        features.append(min(0.4, trainer_rate) if trainer_rate > 0 else 0.06)

        # Feature 6: field size
        features.append(field_size / 14.0)

        # Feature 7: field-relative draw
        features.append(draw / max(1, field_size) if draw > 0 else 0.5)

        # Feature 8: prior pace (field-size-aware normalization)
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
                    pace_vals.append(1.0 - (early - 1) / max(1, field_size - 1))
        features.append(sum(pace_vals) / len(pace_vals) if pace_vals else 0.5)

        # Feature 9: adjusted speed (Beyer-style par-adjusted, weight-corrected)
        from analytics.speed_figures import (
            compute_par_times, adjusted_speed_figure,
            normalize_speed_figure, normalize_consistency,
        )
        adj_speed_vals = []
        # Compute par times from DB (cached on first call via horse_data)
        par_times = horse_data.get('_par_times')
        if par_times is None and conn is not None:
            par_times = compute_par_times(conn, before_date=race_date or None)
            horse_data['_par_times'] = par_times  # cache for subsequent horses
        if par_times and prior:
            for p in prior[-5:]:
                ft = p.get('finish_time', '') or ''
                d = p.get('distance', 0) or 0
                g = p.get('going', '') or ''
                ct = p.get('class_type', '') or ''
                w = p.get('weight', 0) or 0
                fig = adjusted_speed_figure(ft, d, g, ct, w, par_times)
                if fig is not None:
                    adj_speed_vals.append(fig)
        if adj_speed_vals:
            features.append(normalize_speed_figure(sum(adj_speed_vals) / len(adj_speed_vals)))
        else:
            features.append(0.5)

        # Feature 10: C&D form from prior races
        prior_at_dist = [p for p in prior if p.get('distance') == distance]
        if prior_at_dist:
            cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
            features.append(min(1.0, cd_wins / len(prior_at_dist) + 0.2))
        else:
            features.append(0.5)

        # Feature 11: class level (INVERTED: Class 1=0.8, Class 5=0.0)
        class_type = horse_data.get('class_type', '')
        cm = re.search(r'(\d)', class_type)
        class_rank = int(cm.group(1)) if cm else 3
        features.append(max(0.0, min(1.0, 1.0 - class_rank / 5.0)))

        # Feature 12: freshness
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

        # Feature 13: going performance (merged — combines win rate + place rate on this going)
        going_perf = 0.5
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if prior_same_going:
                g_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                g_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                n = len(prior_same_going)
                # Weighted blend: 60% win-rate signal, 40% place-rate signal
                going_perf = min(1.0, (g_wins / n) * 0.6 * 5 + (g_top3 / n) * 0.4 * 2)
        features.append(going_perf)

        # Feature 14: weight vs field (carried weight relative to field median)
        # Lighter weight = advantage. Normalized so below-median → >0.5, above → <0.5
        weight_vs_field = 0.5
        field_weights = horse_data.get('field_weights', [])
        weight_str = horse_data.get('weight', '') or ''
        try:
            horse_wt = int(weight_str)
            if field_weights:
                median_wt = sorted(field_weights)[len(field_weights) // 2]
                # Diff: negative = lighter than median = advantage
                diff = horse_wt - median_wt
                weight_vs_field = max(0.0, min(1.0, 0.5 - diff / 20.0))
            else:
                # No field data — use absolute position in 108-135 range
                weight_vs_field = max(0.0, min(1.0, 1.0 - (horse_wt - 108) / 27.0))
        except (ValueError, TypeError):
            pass
        features.append(weight_vs_field)

        # Feature 15: jockey-horse cooperation rate
        jockey_horse_coop = 0.5
        jockey = horse_data.get('jockey', '')
        if prior and jockey:
            prior_same_jockey = [p for p in prior if p.get('jockey', '') == jockey]
            if len(prior_same_jockey) >= 2:
                jh_wins = sum(1 for p in prior_same_jockey if p.get('is_winner', 0))
                jh_places = sum(1 for p in prior_same_jockey if p.get('is_top3', 0))
                jockey_horse_coop = min(1.0, (jh_wins * 2 + jh_places) / (len(prior_same_jockey) * 3))
        features.append(jockey_horse_coop)

        # Feature 16: distance personal best ratio
        distance_pb_ratio = 0.5
        if prior:
            prior_dist_with_time = [p for p in prior
                                    if p.get('distance') == distance
                                    and p.get('finish_time', '')
                                    and ':' in str(p.get('finish_time', ''))]
            if prior_dist_with_time:
                def _parse_time(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0
                times = [_parse_time(p['finish_time']) for p in prior_dist_with_time]
                valid_times = [t for t in times if t < 999]
                if valid_times:
                    horse_pb = min(valid_times)
                    try:
                        if conn is not None:
                            # Always filter by date to prevent future leakage
                            cutoff = race_date or datetime.now().strftime('%Y/%m/%d')
                            pb_query = ("SELECT finish_time FROM race_results "
                                        "WHERE distance = ? AND finish_time != '' "
                                        "AND finish_time IS NOT NULL AND finish_time LIKE '%:%' "
                                        "AND race_date < ?")
                            pb_params = [distance, cutoff]
                            row_pb = conn.execute(pb_query, pb_params).fetchall()
                            best_time = None
                            for rr in row_pb:
                                t = _parse_time(rr[0])
                                if t < 999 and (best_time is None or t < best_time):
                                    best_time = t
                            if best_time is not None and best_time < 999 and horse_pb < 999:
                                distance_pb_ratio = min(1.0, best_time / horse_pb)
                    except Exception as e:
                        logger.warning(f"Failed to compute distance PB for {horse_name}: {e}")
        features.append(distance_pb_ratio)

        # Feature 17: field-relative speed (avg percentile within prior race fields)
        # Solves the problem of absolute speed conflating sprinters with stayers
        # Uses a single batched query for all prior races instead of per-race queries
        field_rel_speed = 0.5
        if prior and conn is not None:
            # Collect race keys for the last 5 prior races that have finish times
            race_lookups = []
            for p in prior[-5:]:
                p_ft = p.get('finish_time', '') or ''
                if p_ft and ':' in p_ft:
                    race_lookups.append((p.get('race_date', ''), p.get('race_number', 0), p_ft))

            if race_lookups:
                try:
                    # Batch query: get all finish times for all relevant races at once
                    placeholders = ' OR '.join(['(race_date = ? AND race_number = ?)'] * len(race_lookups))
                    params = []
                    for rd, rn, _ in race_lookups:
                        params.extend([rd, rn])
                    batch_rows = conn.execute(
                        f"SELECT race_date, race_number, finish_time FROM race_results "
                        f"WHERE ({placeholders}) AND finish_time LIKE '%:%'",
                        params
                    ).fetchall()

                    # Group finish times by race
                    race_times_map = {}
                    for row in batch_rows:
                        rk = (row[0], row[1])
                        try:
                            parts = str(row[2]).split(':')
                            s = float(parts[0]) * 60 + float(parts[1])
                            if 0 < s < 999:
                                race_times_map.setdefault(rk, []).append(s)
                        except (ValueError, IndexError):
                            pass

                    # Compute percentile for each prior race
                    percentiles = []
                    for rd, rn, p_ft in race_lookups:
                        try:
                            parts = p_ft.split(':')
                            horse_secs = float(parts[0]) * 60 + float(parts[1])
                            if horse_secs <= 0 or horse_secs >= 999:
                                continue
                        except (ValueError, IndexError):
                            continue
                        all_secs = race_times_map.get((rd, rn), [])
                        if len(all_secs) >= 3:
                            rank = sum(1 for s in all_secs if s > horse_secs)
                            percentiles.append(rank / (len(all_secs) - 1) if len(all_secs) > 1 else 0.5)

                    if percentiles:
                        field_rel_speed = sum(percentiles) / len(percentiles)
                except Exception as e:
                    logger.warning(f"Failed to compute field_rel_speed for {horse_name}: {e}")
        features.append(field_rel_speed)

        # Feature 18: last start margin (NEW — how far behind winner last race)
        last_margin = 0.5
        if prior:
            last_race = prior[-1]
            m = last_race.get('margin', 0) or 0
            if last_race.get('is_winner', 0):
                last_margin = 1.0  # won last race
            elif m > 0:
                # Normalize: 0 lengths = 0.9, 5 lengths = 0.4, 10+ lengths = 0.1
                last_margin = max(0.1, min(0.9, 1.0 - m / 12.0))
        features.append(last_margin)

        # Feature 19: class change (drop = positive signal, rise = negative)
        class_change_feat = 0.5
        if prior:
            last_class_str = prior[-1].get('class_type', '') or ''
            last_cm = re.search(r'(\d)', last_class_str)
            last_class = int(last_cm.group(1)) if last_cm else 3
            curr_cm = re.search(r'(\d)', class_type)
            curr_class = int(curr_cm.group(1)) if curr_cm else 3
            # Drop (e.g. Class 2→Class 3) = higher number = positive signal
            # Normalize: -4 (big rise) → 0.0, 0 (same) → 0.5, +4 (big drop) → 1.0
            diff = curr_class - last_class  # positive = dropped in class
            class_change_feat = max(0.0, min(1.0, 0.5 + diff * 0.125))
        features.append(class_change_feat)

        # Feature 20: days since last win (recency of winning)
        days_last_win = 0.5
        if prior:
            for p in reversed(prior):
                if p.get('is_winner', 0):
                    win_date = p.get('race_date', '')
                    if win_date and race_date:
                        try:
                            w_dt = datetime.strptime(win_date.replace('/', '-'), '%Y-%m-%d')
                            r_dt = datetime.strptime(race_date.replace('/', '-'), '%Y-%m-%d')
                            days = (r_dt - w_dt).days
                            days_last_win = max(0.1, min(0.95, 1.0 - days / 350.0))
                        except Exception:
                            pass
                    break
            else:
                days_last_win = 0.15
        features.append(days_last_win)

        # Feature 21: jockey × trainer combo win rate (interaction)
        jt_combo = 0.5
        jockey_rate_val = features[3]   # jockey_win_rate (feature 4, index 3)
        trainer_rate_val = features[4]  # trainer_win_rate (feature 5, index 4)
        if jockey_rate_val > 0 and trainer_rate_val > 0:
            # Geometric mean normalized — strong pairs score high
            jt_combo = min(1.0, (jockey_rate_val * trainer_rate_val) ** 0.5 * 5.0)
        features.append(jt_combo)

        # Feature 22: horse body weight change from prior race (fitness signal)
        hw_change = 0.5
        horse_weight_val = horse_data.get('horse_weight', 0) or 0
        try:
            horse_weight_val = int(horse_weight_val)
        except (ValueError, TypeError):
            horse_weight_val = 0
        if prior and horse_weight_val > 0:
            prior_hws = [p.get('horse_weight', 0) or 0 for p in prior[-3:] if (p.get('horse_weight', 0) or 0) > 0]
            if prior_hws:
                avg_prior_hw = sum(prior_hws) / len(prior_hws)
                if avg_prior_hw > 0:
                    # % change: positive = gained weight, negative = lost weight
                    pct_change = (horse_weight_val - avg_prior_hw) / avg_prior_hw
                    # Normalize: -3% → 0.8 (slimmed down = fit), +3% → 0.2 (gained = concern)
                    hw_change = max(0.0, min(1.0, 0.5 - pct_change * 10))
        features.append(hw_change)

        # Feature 23: speed consistency (std of adjusted speed figures — low = reliable)
        speed_con = 0.5
        if len(adj_speed_vals) >= 3:
            import numpy as _np
            speed_con = normalize_consistency(float(_np.std(adj_speed_vals)))
        features.append(speed_con)

        # Feature 24 (late_money_signal) REMOVED — constant 0.5 in training (DB lacks
        # opening odds), variable at prediction time. The model never learned from it,
        # so non-default values at inference were out-of-distribution noise.
        # Odds signal enters via the market blend in calibrate_probabilities() instead.

        if owns_conn and conn is not None:
            conn.close()

        result = np.array(features, dtype=np.float32)
        validate_features(result)
        return result
