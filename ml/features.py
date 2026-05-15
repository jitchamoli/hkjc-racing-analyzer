#!/usr/bin/env python3
"""Feature engineering for ML predictions.

Extracts 16 features per horse using FEATURE_SCHEMA order.
All features use only data available BEFORE the race (no leakage).
v6.4: dropped prior_pace, adjusted_speed, distance_pb_ratio, speed_consistency
(redundant with last_start_margin). Wave 2c: dropped field_rel_speed and
days_since_last_win for structural correlations.
"""

import re
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES, validate_features
import config

logger = logging.getLogger(__name__)


def parse_age_score(age_raw) -> float:
    """Convert raw age (any of 5, '5', '5.0') into the v6.5 age feature.

    Single source of truth — used by both inference (extract_features)
    and training (build_training_data_from_db). Encoding: 3yo→1.0,
    6yo→0.5, 9yo+→0.0; missing/unparseable → 0.5 fallback.

    Regression: the prior `[:2]` slice silently failed on '5.0' format
    (`int('5.')` raises) and 94% of training rows fell back to 0.5
    while inference parsed '5' correctly — train/predict mismatch
    masquerading as a dead feature.
    """
    age_raw = age_raw or 0
    try:
        age_int = int(float(str(age_raw).strip())) if age_raw else 0
    except (ValueError, TypeError):
        age_int = 0
    if age_int and 2 <= age_int <= 15:
        return max(0.0, min(1.0, (9 - age_int) / 6.0))
    return 0.5


# Per-process caches for jockey/trainer Bayesian-smoothed win rates.
# A typical race day has ~30 unique jockeys riding across 11 races; without
# caching, extract_features fires one query per (horse, race) → 264 round-trips.
# Keyed by (name, before_date); cleared between race-day runs by
# clear_feature_caches() so backtests don't see stale rates.
_jockey_rate_cache: dict = {}
_trainer_rate_cache: dict = {}
_jockey_window_cache: dict = {}  # keyed by (jockey, before_date, window_days)
_jt_pair_cache: dict = {}        # keyed by (jockey, trainer, before_date)
_speed_fig_cache: dict = {}      # keyed by (horse_name, before_date)
_par_times_cache: dict = {}      # keyed by before_date — par_times is O(N) to compute
_running_pos_cache: dict = {}    # keyed by (horse_name, before_date) — v6.8 Features 22-24


def clear_feature_caches() -> None:
    """Reset jockey/trainer rate caches. Call at the start of each race-day run."""
    _jockey_rate_cache.clear()
    _trainer_rate_cache.clear()
    _jockey_window_cache.clear()
    _jt_pair_cache.clear()
    _speed_fig_cache.clear()
    _par_times_cache.clear()
    _running_pos_cache.clear()


def extract_class_rank(class_type: str) -> Optional[int]:
    """Extract HKJC class rank from a class string.

    Returns int 0-5 where 0 = highest tier (Group 1/2), 5 = lowest (Class 5).
    None if the string is empty or unparseable. The previous regex r'(\\d)'
    incorrectly turned 'G1' into class 1 — Group races sit ABOVE class 1, not
    inside the digit ladder.

    HKJC tier ladder (high → low):
        Group 1/2  → 0
        Group 3    → 1   (≈ top of class 1 in difficulty)
        Class 1    → 1
        Class 2    → 2
        Class 3    → 3
        Class 4    → 4
        Class 5    → 5
        GRIFFIN, MAIDEN     → 4   (restricted, lower-tier)
        4YO / AGE-restricted → 3   (mixed-class restricted)
    """
    if not class_type:
        return None
    s = str(class_type).strip().upper()
    if not s:
        return None
    if s in ('G1', 'GROUP 1', 'G2', 'GROUP 2'):
        return 0
    if s in ('G3', 'GROUP 3'):
        return 1
    m = re.match(r'^(?:CLASS\s+)?(\d)$', s)
    if m:
        return int(m.group(1))
    if 'GRIFFIN' in s or 'MAIDEN' in s:
        return 4
    if '4YO' in s or 'AGE' in s or s.startswith('Y'):
        return 3
    return None


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
        Falls back to jockey_stats table, then all_results. Result is cached
        per (jockey, before_date) to kill the N+1 query pattern when
        extract_features is called per-horse.
        """
        if not jockey:
            return 0.12
        cache_key = (jockey, before_date)
        if cache_key in _jockey_rate_cache:
            return _jockey_rate_cache[cache_key]
        rate = FeatureEngineering._compute_jockey_win_rate(jockey, all_results, conn, before_date)
        _jockey_rate_cache[cache_key] = rate
        return rate

    @staticmethod
    def _compute_jockey_win_rate(jockey, all_results, conn, before_date) -> float:
        c = None
        owns = conn is None
        try:
            if owns and config.DB_PATH.exists():
                c = sqlite3.connect(str(config.DB_PATH))
                c.row_factory = sqlite3.Row
            else:
                c = conn
            if c is not None:
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
                rides = (row['rides'] if row else 0) or 0
                wins = (row['wins'] if row else 0) or 0
                if rides > 0:
                    # Bayesian smoothing: Beta prior at HK league avg (0.08) with strength 15
                    return min(0.6, (wins + 0.08 * 15) / (rides + 15))
                # Fall back to stats table only when (a) caller passed a conn
                # AND (b) we're not date-filtering. The jockey_stats table
                # reflects current full-season state — using it during
                # historical inference (before_date set) leaks future data.
                if not owns and before_date is None:
                    stat = c.execute(
                        "SELECT win_rate FROM jockey_stats WHERE name = ?", (jockey,)
                    ).fetchone()
                    if stat and stat['win_rate'] and stat['win_rate'] > 0:
                        return min(0.6, stat['win_rate'])
        except Exception as e:
            logger.warning(f"DB lookup failed for jockey {jockey}: {e}")
        finally:
            if owns and c is not None:
                c.close()
        jockey_rides = [r for r in all_results if jockey.lower() in r.jockey.lower()]
        if not jockey_rides:
            return 0.08
        wins = sum(1 for r in jockey_rides if r.is_win)
        return min(0.6, (wins + 0.08 * 15) / (len(jockey_rides) + 15))

    @staticmethod
    def calculate_trainer_win_rate(trainer: str, all_results, conn=None,
                                   before_date: str = None) -> float:
        """Calculate trainer win rate from prior race_results (matches training).

        Uses race_results with date filter to avoid leaking future data.
        Falls back to trainer_stats table, then all_results. Cached per
        (trainer, before_date) — see calculate_jockey_win_rate.
        """
        if not trainer:
            return 0.12
        cache_key = (trainer, before_date)
        if cache_key in _trainer_rate_cache:
            return _trainer_rate_cache[cache_key]
        rate = FeatureEngineering._compute_trainer_win_rate(trainer, all_results, conn, before_date)
        _trainer_rate_cache[cache_key] = rate
        return rate

    @staticmethod
    def _compute_trainer_win_rate(trainer, all_results, conn, before_date) -> float:
        c = None
        owns = conn is None
        try:
            if owns and config.DB_PATH.exists():
                c = sqlite3.connect(str(config.DB_PATH))
                c.row_factory = sqlite3.Row
            else:
                c = conn
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
                rides = (row['rides'] if row else 0) or 0
                wins = (row['wins'] if row else 0) or 0
                if rides > 0:
                    # Bayesian smoothing: Beta prior at HK league avg (0.08) with strength 15
                    return min(0.5, (wins + 0.08 * 15) / (rides + 15))
                # Same date-filter guard as jockey: stats table is current-
                # season only and would leak future data when scoring a
                # historical card (before_date set).
                if not owns and before_date is None:
                    stat = c.execute(
                        "SELECT win_rate FROM trainer_stats WHERE name = ?", (trainer,)
                    ).fetchone()
                    if stat and stat['win_rate'] and stat['win_rate'] > 0:
                        return min(0.5, stat['win_rate'])
        except Exception as e:
            logger.warning(f"DB lookup failed for trainer {trainer}: {e}")
        finally:
            if owns and c is not None:
                c.close()
        trainer_horses = [r for r in all_results if trainer.lower() in r.trainer.lower()]
        if not trainer_horses:
            return 0.08
        wins = sum(1 for r in trainer_horses if r.is_win)
        return min(0.5, (wins + 0.08 * 15) / (len(trainer_horses) + 15))

    @staticmethod
    def calculate_jockey_form_window(jockey: str, conn=None,
                                     before_date: str = None,
                                     window_days: int = 30) -> float:
        """v6.5 Feature 19: Bayesian-smoothed jockey strike rate over a recent window.

        Captures hot/cold streaks the all-time aggregate (Feature 4) hides.
        Same Bayesian Beta(0.08, 15) prior as the all-time rate so unknown
        or new jockeys default to HK league average. Cached per
        (jockey, before_date, window_days) to avoid N+1 SQL.
        """
        if not jockey:
            return 0.08
        cache_key = (jockey, before_date, window_days)
        if cache_key in _jockey_window_cache:
            return _jockey_window_cache[cache_key]
        rate = FeatureEngineering._compute_jockey_form_window(
            jockey, conn, before_date, window_days,
        )
        _jockey_window_cache[cache_key] = rate
        return rate

    @staticmethod
    def _compute_jockey_form_window(jockey, conn, before_date, window_days):
        c = None
        owns = conn is None
        try:
            if owns and config.DB_PATH.exists():
                c = sqlite3.connect(str(config.DB_PATH))
                c.row_factory = sqlite3.Row
            else:
                c = conn
            if c is None:
                return 0.08
            # Use julianday for date arithmetic; race_date is YYYY/MM/DD so
            # REPLACE to YYYY-MM-DD for sqlite to parse correctly.
            if before_date:
                row = c.execute(
                    "SELECT COUNT(*) as rides, SUM(is_winner) as wins "
                    "FROM race_results WHERE jockey = ? "
                    "  AND race_date < ? "
                    "  AND julianday(REPLACE(race_date,'/','-')) >= "
                    "      julianday(REPLACE(?,'/','-')) - ?",
                    (jockey, before_date, before_date, window_days),
                ).fetchone()
            else:
                # No reference date — fall back to all-time (degenerates
                # gracefully to Feature 4's signal).
                row = c.execute(
                    "SELECT COUNT(*) as rides, SUM(is_winner) as wins "
                    "FROM race_results WHERE jockey = ?",
                    (jockey,),
                ).fetchone()
            rides = (row['rides'] if row else 0) or 0
            wins = (row['wins'] if row else 0) or 0
            if rides > 0:
                return min(0.6, (wins + 0.08 * 15) / (rides + 15))
        except Exception as e:
            logger.warning(f"DB lookup failed for jockey-form-window {jockey}: {e}")
        finally:
            if owns and c is not None:
                c.close()
        return 0.08

    @staticmethod
    def calculate_jt_pair_win_rate(jockey: str, trainer: str, conn=None,
                                   before_date: str = None) -> float:
        """v6.6 Feature 20: Bayesian-smoothed jockey×trainer pairing strike rate.

        Top HK pairings (Cruz×Purton, Size×Moreira) signal stable confidence —
        when a stable's preferred jockey is on a given runner, the model
        learns this is a "stable confidence" mount. Per round-2 audit, 305
        pairs have ≥50 joint runs and top pairs hit 22-28% wins vs 8.3%
        baseline — major signal.

        Bayesian Beta(0.08, 15) prior so unknown/new pairings default to HK
        league average. Cached per (jockey, trainer, before_date).
        """
        if not jockey or not trainer:
            return 0.08
        cache_key = (jockey, trainer, before_date)
        if cache_key in _jt_pair_cache:
            return _jt_pair_cache[cache_key]
        rate = FeatureEngineering._compute_jt_pair_win_rate(
            jockey, trainer, conn, before_date,
        )
        _jt_pair_cache[cache_key] = rate
        return rate

    @staticmethod
    def _compute_jt_pair_win_rate(jockey, trainer, conn, before_date):
        c = None
        owns = conn is None
        try:
            if owns and config.DB_PATH.exists():
                c = sqlite3.connect(str(config.DB_PATH))
                c.row_factory = sqlite3.Row
            else:
                c = conn
            if c is None:
                return 0.08
            if before_date:
                row = c.execute(
                    "SELECT COUNT(*) as rides, SUM(is_winner) as wins "
                    "FROM race_results WHERE jockey = ? AND trainer = ? "
                    "AND race_date < ?",
                    (jockey, trainer, before_date),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT COUNT(*) as rides, SUM(is_winner) as wins "
                    "FROM race_results WHERE jockey = ? AND trainer = ?",
                    (jockey, trainer),
                ).fetchone()
            rides = (row['rides'] if row else 0) or 0
            wins = (row['wins'] if row else 0) or 0
            if rides > 0:
                return min(0.5, (wins + 0.08 * 15) / (rides + 15))
        except Exception as e:
            logger.warning(f"DB lookup failed for jt-pair {jockey}/{trainer}: {e}")
        finally:
            if owns and c is not None:
                c.close()
        return 0.08

    @staticmethod
    def parse_running_position(rp: str):
        """Parse race_results.running_position into a list of ints.

        Modern data is space-separated ('7 6 2'). Legacy concatenated format
        ('762') is also supported but rare in our DB (>99% modern). Returns
        an empty list for unparseable input.
        """
        if not rp:
            return []
        s = str(rp).strip()
        if not s:
            return []
        if ' ' in s:
            try:
                return [int(p) for p in s.split() if p.strip().isdigit()]
            except (ValueError, TypeError):
                return []
        # Legacy concat format — single-digit per position. Ambiguous when
        # any position > 9 but rare in our DB so accept the lossy parse.
        try:
            return [int(c) for c in s if c.isdigit()]
        except (ValueError, TypeError):
            return []

    @staticmethod
    def compute_running_position_features(rp: str, field_size: int):
        """Returns (early_speed, late_kick, running_style) all in [0, 1].

        early_speed: 1st-call position / max_pos. Front-runners ≈ 0.0,
            closers ≈ 1.0 (numerically larger position). Rank-normalized so
            it's distance-independent.
        late_kick: closing rate from second-last call to finish.
            >0.5 means improved (closed ground), <0.5 means faded.
            Encoded as 0.5 + delta/(2*max_pos) and clipped.
        running_style: avg position over all calls except finish, mapped to
            front (≤1/3) → 0.0, stalker → 0.5, closer (≥2/3) → 1.0.
        Returns (0.5, 0.5, 0.5) defaults when unparseable.
        """
        positions = FeatureEngineering.parse_running_position(rp)
        if not positions or field_size <= 1:
            return 0.5, 0.5, 0.5
        max_pos = float(max(field_size, max(positions)))
        # early_speed: 1st-call position normalized
        early = positions[0] / max_pos
        early = max(0.0, min(1.0, early))
        # late_kick: change between 2nd-last position and finish (last value)
        # Positive delta = improved (closed ground); negative = faded.
        if len(positions) >= 2:
            delta = positions[-2] - positions[-1]  # positive = closed
            late = 0.5 + delta / (2.0 * max_pos)
            late = max(0.0, min(1.0, late))
        else:
            late = 0.5
        # running_style: avg of all calls except the finish (last)
        non_finish = positions[:-1] if len(positions) > 1 else positions
        avg_pos = sum(non_finish) / len(non_finish)
        third = max_pos / 3.0
        if avg_pos <= third:
            style = 0.0  # front-runner
        elif avg_pos >= 2 * third:
            style = 1.0  # closer
        else:
            style = 0.5  # stalker
        return early, late, style

    @staticmethod
    def calculate_running_position_features(horse_name: str, conn=None,
                                            before_date: str = None):
        """v6.8 Features 22-24: running-position proxies for sectional speed.

        Looks up the horse's most recent prior run, parses running_position,
        returns (early_speed, late_kick, running_style). All ∈ [0, 1].
        Returns (0.5, 0.5, 0.5) when no prior run or parse fails.

        Cached per (horse_name, before_date) — same pattern as speed_fig_norm
        to avoid N+1 queries.
        """
        if not horse_name:
            return 0.5, 0.5, 0.5
        cache_key = (horse_name, before_date)
        if cache_key in _running_pos_cache:
            return _running_pos_cache[cache_key]
        result = FeatureEngineering._compute_running_position_features(
            horse_name, conn, before_date,
        )
        _running_pos_cache[cache_key] = result
        return result

    @staticmethod
    def _compute_running_position_features(horse_name, conn, before_date):
        c = None
        owns = conn is None
        try:
            if owns and config.DB_PATH.exists():
                c = sqlite3.connect(str(config.DB_PATH))
                c.row_factory = sqlite3.Row
            else:
                c = conn
            if c is None:
                return 0.5, 0.5, 0.5
            if before_date:
                row = c.execute(
                    "SELECT rr.running_position, "
                    "  (SELECT COUNT(*) FROM race_results rr2 "
                    "   WHERE rr2.race_date = rr.race_date "
                    "     AND rr2.race_number = rr.race_number "
                    "     AND rr2.finish_position > 0) as field_size "
                    "FROM race_results rr WHERE rr.horse_name = ? "
                    "  AND rr.race_date < ? AND rr.running_position != '' "
                    "ORDER BY rr.race_date DESC, rr.race_number DESC LIMIT 1",
                    (horse_name, before_date),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT rr.running_position, "
                    "  (SELECT COUNT(*) FROM race_results rr2 "
                    "   WHERE rr2.race_date = rr.race_date "
                    "     AND rr2.race_number = rr.race_number "
                    "     AND rr2.finish_position > 0) as field_size "
                    "FROM race_results rr WHERE rr.horse_name = ? "
                    "  AND rr.running_position != '' "
                    "ORDER BY rr.race_date DESC, rr.race_number DESC LIMIT 1",
                    (horse_name,),
                ).fetchone()
            if not row:
                return 0.5, 0.5, 0.5
            return FeatureEngineering.compute_running_position_features(
                row['running_position'], row['field_size'] or 12,
            )
        except Exception as e:
            logger.warning(f"running_position lookup failed for {horse_name}: {e}")
            return 0.5, 0.5, 0.5
        finally:
            if owns and c is not None:
                c.close()

    @staticmethod
    def calculate_speed_fig_norm(horse_name: str, conn=None,
                                 before_date: str = None) -> float:
        """v6.6 Feature 21: par-time-normalised speed figure of last race.

        Wraps analytics.speed_figures: looks up the horse's most recent run
        before `before_date`, computes adjusted_speed_figure (weight + par
        time normalised), then maps to [0,1] via normalize_speed_figure.

        Cached per (horse_name, before_date). Returns 0.5 (neutral) when
        no eligible prior run exists.
        """
        if not horse_name:
            return 0.5
        cache_key = (horse_name, before_date)
        if cache_key in _speed_fig_cache:
            return _speed_fig_cache[cache_key]
        rate = FeatureEngineering._compute_speed_fig_norm(
            horse_name, conn, before_date,
        )
        _speed_fig_cache[cache_key] = rate
        return rate

    @staticmethod
    def _compute_speed_fig_norm(horse_name, conn, before_date):
        from analytics.speed_figures import (
            compute_par_times, adjusted_speed_figure, normalize_speed_figure,
        )
        c = None
        owns = conn is None
        try:
            if owns and config.DB_PATH.exists():
                c = sqlite3.connect(str(config.DB_PATH))
                c.row_factory = sqlite3.Row
            else:
                c = conn
            if c is None:
                return 0.5
            # Look up the horse's most recent prior race
            if before_date:
                row = c.execute(
                    "SELECT finish_time, distance, going, class_type, weight "
                    "FROM race_results WHERE horse_name = ? AND race_date < ? "
                    "ORDER BY race_date DESC, race_number DESC LIMIT 1",
                    (horse_name, before_date),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT finish_time, distance, going, class_type, weight "
                    "FROM race_results WHERE horse_name = ? "
                    "ORDER BY race_date DESC, race_number DESC LIMIT 1",
                    (horse_name,),
                ).fetchone()
            if not row or not row['finish_time']:
                return 0.5
            # Cache par_times by before_date — computing it is O(N) over all
            # historical rows. Without this cache we'd recompute per horse,
            # making training O(N²) over the corpus.
            par_times = _par_times_cache.get(before_date)
            if par_times is None:
                par_times = compute_par_times(c, before_date=before_date)
                _par_times_cache[before_date] = par_times
            sf = adjusted_speed_figure(
                row['finish_time'], row['distance'] or 0,
                row['going'] or '', row['class_type'] or '',
                row['weight'] or 120, par_times,
            )
            if sf is None:
                return 0.5
            return float(normalize_speed_figure(sf))
        except Exception as e:
            logger.warning(f"Speed fig lookup failed for {horse_name}: {e}")
        finally:
            if owns and c is not None:
                c.close()
        return 0.5

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
        """Extract 16 features matching FEATURE_SCHEMA order exactly (v6.4).

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
        # Lightly-raced horses (≤3 starts) get a floor of 0.35 — penalizing maidens
        # too heavily misses the ~73% of HK winners that had 0 prior wins in their class
        if prior:
            prior_wins = sum(1 for p in prior if p.get('is_winner', 0))
            prior_top3 = sum(1 for p in prior if p.get('is_top3', 0))
            feat_form_proxy = min(1.0, (prior_wins * 2 + prior_top3) / (len(prior) * 3))
            if len(prior) <= 3 and feat_form_proxy < 0.35:
                feat_form_proxy = 0.35
        else:
            feat_form_proxy = 0.5
        features.append(feat_form_proxy)

        # Feature 2: recent form from prior finishes.
        # Training (build_training_data_from_db) defaults to 0.5 when prior
        # is empty; we mirror that here. Previously this fallback computed
        # from the card's `last_6` string when prior was empty — but that
        # uses a different point scale (calculate_form_score: 100/70/50/30/15/5)
        # vs our DB-based scale (10/7/5/2), creating a different inference-
        # time distribution for horses without DB history (rare: mid-season
        # imports, restored DB rows). Removed for train/predict consistency.
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
            features.append(0.5)

        # Feature 3: draw advantage
        draw = horse_data.get('draw', 8)
        features.append(FeatureEngineering.calculate_draw_advantage(draw, distance))

        # Feature 4: jockey win rate (prior-only to match training)
        # Cap 0.6 + fallback 0.08 — matches FeatureEngineering.calculate_jockey_win_rate
        # cap exactly. Old caps (0.5 / 0.06) created train/predict drift after the
        # Task #51 fix routed both paths through the helper.
        jockey = horse_data.get('jockey', '')
        jockey_rate = FeatureEngineering.calculate_jockey_win_rate(
            jockey, all_results, conn=conn, before_date=race_date or None)
        features.append(jockey_rate if jockey_rate > 0 else 0.08)

        # Feature 5: trainer win rate (prior-only to match training)
        # Cap 0.5 + fallback 0.08 — matches FeatureEngineering.calculate_trainer_win_rate.
        trainer = horse_data.get('trainer', '')
        trainer_rate = FeatureEngineering.calculate_trainer_win_rate(
            trainer, all_results, conn=conn, before_date=race_date or None)
        features.append(trainer_rate if trainer_rate > 0 else 0.08)

        # Feature 6: field size — clipped to [0,1] (parity with training fix v6.4).
        # Legacy oversize fields and bad-data rows could push field_size > 14.
        features.append(min(1.0, field_size / 14.0))

        # Feature 7: field-relative draw — clipped to [0,1] (F-16: 14-gate races
        # can push draw/field_size > 1 when field_size defaults to 12 at inference)
        features.append(min(1.0, draw / max(1, field_size)) if draw > 0 else 0.5)

        # prior_pace, adjusted_speed dropped v6.4 — both had 0.0 importance
        # in v6.3 because last_start_margin (16.9%) carried the form signal.

        # Feature 8: C&D form from prior races (winning trip = strong signal).
        # Removed the +0.2 floor that previously made a 0-for-5 horse (cd_score=0.2)
        # rank as "more C&D ability" than an unraced horse (default 0.5). Now:
        # unraced = 0.5 (high uncertainty); 0-for-N = 0.0 (clear failure); higher
        # win rate scales smoothly. Recent winning trip (≤180d) adds +0.15.
        prior_at_dist = [p for p in prior if p.get('distance') == distance]
        if prior_at_dist:
            cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
            cd_score = cd_wins / len(prior_at_dist)
            if cd_wins > 0 and race_date:
                for p in reversed(prior_at_dist):
                    if p.get('is_winner', 0):
                        try:
                            w_dt = datetime.strptime(p['race_date'].replace('/', '-'), '%Y-%m-%d')
                            r_dt = datetime.strptime(race_date.replace('/', '-'), '%Y-%m-%d')
                            if (r_dt - w_dt).days <= 180:
                                cd_score += 0.15  # winning trip replication boost
                        except Exception:
                            pass
                        break
            features.append(min(1.0, cd_score))
        else:
            features.append(0.5)

        # Feature 9: class level (INVERTED: Group 1/2=1.0, Class 1=0.8, Class 5=0.0)
        class_type = horse_data.get('class_type', '')
        class_rank = extract_class_rank(class_type)
        if class_rank is None:
            class_rank = 3
        features.append(max(0.0, min(1.0, 1.0 - class_rank / 5.0)))

        # Feature 10: freshness (closer to ideal 21d spacing → higher score)
        # When days_since_last_run is missing, fall back to (race_date - prior race_date)
        # — NOT (datetime.now() - prior race_date), which would give wrong values when
        # back-testing past races or predicting future-dated cards. If race_date itself
        # is missing/malformed, fall through to dslr=0 (matches training fallback in
        # ml/training.py:293-302) instead of using wall-clock — that's a parity gap
        # the multi-expert review flagged.
        dslr = horse_data.get('days_since_last_run', 0) or 0
        if not dslr and prior and race_date:
            last_date = prior[-1].get('race_date', '')
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date.replace('/', '-'), '%Y-%m-%d')
                    curr_dt = datetime.strptime(race_date.replace('/', '-'), '%Y-%m-%d')
                    dslr = (curr_dt - last_dt).days
                except Exception:
                    pass
        if dslr > 0:
            features.append(max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60)))
        else:
            features.append(0.5)

        # Feature 11: going performance (merged — combines win rate + place rate on this going)
        # Bayesian-smoothed with HK base rates (~8% win, ~25% top3). Without
        # smoothing, 1 win on the going saturates to 1.0 — same as a 4/8
        # battle-tested horse. Beta(0.08, 5) prior on win rate, Beta(0.25, 5)
        # on place rate (strength=5: posterior is half-prior at n=5 prior runs).
        going_perf = 0.5
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if prior_same_going:
                g_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                g_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                n = len(prior_same_going)
                # Bayesian smoothing — small samples shrink toward HK base rates
                win_rate_smooth = (g_wins + 0.08 * 5) / (n + 5)
                top3_rate_smooth = (g_top3 + 0.25 * 5) / (n + 5)
                going_perf = min(1.0, win_rate_smooth * 0.6 * 5 + top3_rate_smooth * 0.4 * 2)
        features.append(going_perf)

        # Feature 12: weight vs field (carried weight relative to field median)
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

        # Feature 13: jockey-horse cooperation rate
        jockey_horse_coop = 0.5
        jockey = horse_data.get('jockey', '')
        if prior and jockey:
            prior_same_jockey = [p for p in prior if p.get('jockey', '') == jockey]
            if len(prior_same_jockey) >= 2:
                jh_wins = sum(1 for p in prior_same_jockey if p.get('is_winner', 0))
                jh_places = sum(1 for p in prior_same_jockey if p.get('is_top3', 0))
                jockey_horse_coop = min(1.0, (jh_wins * 2 + jh_places) / (len(prior_same_jockey) * 3))
        features.append(jockey_horse_coop)

        # distance_pb_ratio dropped v6.4 — 0.0 importance in v6.3.

        # Feature 14: last start margin (how far behind winner last race)
        # — top importance feature in v6.4 (~16%); subsumes the dropped
        # speed-figure features.
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

        # Feature 15: class change — ternary {dropping=1.0, same=0.5, rising=0.0}
        # Uses proper class extractor — handles G1/G2/G3 above the digit ladder.
        class_change_feat = 0.5
        if prior:
            last_class_str = prior[-1].get('class_type', '') or ''
            last_class = extract_class_rank(last_class_str)
            curr_class = extract_class_rank(class_type)
            if last_class is None or curr_class is None:
                class_change_feat = 0.5
            elif last_class > curr_class:
                class_change_feat = 0.0  # rising (harder race) — negative signal
            elif last_class < curr_class:
                class_change_feat = 1.0  # dropping (easier race) — positive signal
            else:
                class_change_feat = 0.5  # same class
        features.append(class_change_feat)

        # days_since_last_win (was Feature 19) dropped Wave 2c — see feature_schema.py.

        # Feature 16: horse body weight change from prior race (fitness signal)
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

        # speed_consistency dropped v6.4 — degenerate after adjusted_speed dropped.

        # Feature 17: horse age (v6.5). 3yo→1.0, 6yo→0.5, 9yo+→0.0.
        # Younger HK horses systematically improve; older horses decline.
        features.append(parse_age_score(horse_data.get('age', 0)))

        # Feature 18: rating change from previous race (v6.5). +10pts→1.0,
        # 0→0.5, -10→0.0. Captures handicapper's view of recent form change.
        rc_raw = horse_data.get('rating_change', 0) or 0
        try:
            rc = int(rc_raw)
        except (ValueError, TypeError):
            rc = 0
        rating_change_score = max(0.0, min(1.0, 0.5 + rc / 20.0))
        features.append(rating_change_score)

        # Feature 19: jockey 30-day form (v6.5). Captures hot streaks the
        # all-time strike rate (Feature 4) hides. Bayesian-smoothed with
        # the same Beta prior so unknown/new jockeys default to league avg.
        jockey_30d = FeatureEngineering.calculate_jockey_form_window(
            jockey, conn=conn, before_date=race_date or None, window_days=30,
        )
        features.append(min(0.6, jockey_30d) if jockey_30d > 0 else 0.08)

        # Feature 20: jockey×trainer pairing strike rate (v6.6). Stable-
        # confidence signal — top HK pairings (Cruz×Purton, Size×Moreira)
        # win at 22-28% vs 8.3% baseline.
        jt_pair = FeatureEngineering.calculate_jt_pair_win_rate(
            jockey, trainer, conn=conn, before_date=race_date or None,
        )
        features.append(min(0.5, jt_pair) if jt_pair > 0 else 0.08)

        # Feature 21: speed figure of last race (v6.6). Par-time + weight
        # normalised — independent ML signal from the 5%-weight rule
        # engine speed_fig factor. Uses horse_name set earlier in the
        # function (line 498) to avoid a latent shadowing trap if a caller
        # passes a dict missing the 'name' key.
        sf_norm = FeatureEngineering.calculate_speed_fig_norm(
            horse_name, conn=conn, before_date=race_date or None,
        )
        features.append(sf_norm)

        # Features 22-24 (v6.8 Path A): running-position proxies for sectional
        # speed. Derived from race_results.running_position (99.4% DB coverage,
        # space-separated positions at each call). Independent orthogonal channel
        # — early/late position dynamics aren't captured by margin or speed_fig.
        early_speed, late_kick, running_style = (
            FeatureEngineering.calculate_running_position_features(
                horse_name, conn=conn, before_date=race_date or None,
            )
        )
        features.append(early_speed)
        features.append(late_kick)
        features.append(running_style)

        if owns_conn and conn is not None:
            conn.close()

        result = np.array(features, dtype=np.float32)
        validate_features(result)
        return result
