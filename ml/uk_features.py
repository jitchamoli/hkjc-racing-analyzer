"""UK feature extraction for ML training.

Reads from tracking_data/uk_history.db and outputs a pandas DataFrame of
one row per (race_id, horse_id) with engineered features + is_winner target.

Features (15 total):
  Direct from API:    sp_dec, or_rating, weight_lbs, draw, age
  Per-horse history:  runs_prior, wins_prior, places_prior, avg_pos_last5,
                      days_since_last_run
  Jockey/trainer:     jockey_50r_wr, trainer_50r_wr  (last 50 rides/runners)
  Race-context:       sp_rank, field_size, or_vs_field_avg

All history-based features are computed using only ROW-WISE PRIOR DATA
(no look-ahead). Verified by the use of shift(1) and cumulative sum minus
current row patterns.

Per the project's HK feature memory: features without market info don't
beat market in efficient UK pool. sp_dec / sp_rank are deliberately
included as primary inputs — the model competes with market by adding
structural signal, not by replacing it.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

import config


FEATURE_COLS = [
    'sp_dec', 'or_rating', 'weight_lbs', 'draw', 'age',
    'runs_prior', 'wins_prior', 'places_prior',
    'avg_pos_last5', 'days_since_last_run',
    'jockey_50r_wr', 'trainer_50r_wr',
    'sp_rank', 'field_size', 'or_vs_field_avg',
]
TARGET = 'is_winner'


def load_runner_table(db_path: Optional[str] = None) -> pd.DataFrame:
    """Load all runners + race context into a single DataFrame.

    Non-finishers (no finish_position) are dropped — we can't compute
    is_winner for them. NULL OR is left as NaN; race-level fillna fixes it.
    """
    conn = sqlite3.connect(str(db_path or config.UK_DB_PATH))
    df = pd.read_sql_query("""
        SELECT
            rr.race_id, rr.horse_id, rr.horse,
            r.date, r.course, r.dist_str, r.race_class,
            rr.saddle_number, rr.draw, rr.sp_decimal AS sp_dec,
            rr.finish_position, rr.age, rr.weight_lbs,
            rr.official_rating AS or_rating,
            rr.jockey_id, rr.trainer_id
        FROM uk_race_results rr
        JOIN uk_races r ON r.race_id = rr.race_id
        WHERE rr.finish_position IS NOT NULL
        ORDER BY r.date, rr.race_id
    """, conn, parse_dates=['date'])
    conn.close()
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized feature engineering. Returns DataFrame with FEATURE_COLS + TARGET.

    Each row corresponds to one runner. Features are computed using only
    prior data (look-ahead-safe).
    """
    df = df.copy()
    df['is_winner'] = (df['finish_position'] == 1).astype(int)
    df['is_placer'] = (df['finish_position'] <= 3).astype(int)

    # --- Horse history features (sort by horse, date for groupby ops) ---
    df = df.sort_values(['horse_id', 'date'], kind='stable').reset_index(drop=True)
    grp = df.groupby('horse_id', sort=False)

    df['runs_prior'] = grp.cumcount()
    # cumsum includes current; subtract current to get prior-only
    df['wins_prior'] = grp['is_winner'].cumsum() - df['is_winner']
    df['places_prior'] = grp['is_placer'].cumsum() - df['is_placer']
    # rolling mean of last 5 finishes EXCLUDING current — shift(1) does that
    df['avg_pos_last5'] = (
        grp['finish_position']
        .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    # days since last run; first run for a horse gets NaN → fill -1 sentinel
    df['days_since_last_run'] = grp['date'].diff().dt.days.fillna(-1)

    # --- Jockey/trainer rolling form: last 50 rides win rate ---
    df['jockey_50r_wr'] = _rolling_wr(df, 'jockey_id', n_rides=50)
    df['trainer_50r_wr'] = _rolling_wr(df, 'trainer_id', n_rides=50)

    # --- Race-context features (group by race) ---
    df['field_size'] = df.groupby('race_id')['horse_id'].transform('count')
    df['sp_rank'] = df.groupby('race_id')['sp_dec'].rank(method='min')
    df['or_field_avg'] = df.groupby('race_id')['or_rating'].transform('mean')
    df['or_vs_field_avg'] = df['or_rating'] - df['or_field_avg']

    # Restore chronological ordering for downstream temporal splits
    df = df.sort_values(['date', 'race_id'], kind='stable').reset_index(drop=True)

    # Fill remaining NaNs with sensible defaults
    df['avg_pos_last5'] = df['avg_pos_last5'].fillna(99)  # no prior runs → bad sentinel
    df['or_rating'] = df['or_rating'].fillna(df['or_field_avg']).fillna(70)
    df['or_vs_field_avg'] = df['or_vs_field_avg'].fillna(0)
    df['jockey_50r_wr'] = df['jockey_50r_wr'].fillna(0)
    df['trainer_50r_wr'] = df['trainer_50r_wr'].fillna(0)
    # Draw is NaN for NH races (jumps) — they don't use barriers. Use 0 as
    # "no draw applicable" — XGBoost handles missingness either way but
    # consistent value is cleaner.
    df['draw'] = df['draw'].fillna(0)

    return df


def _rolling_wr(df: pd.DataFrame, by_col: str, n_rides: int = 50) -> pd.Series:
    """For each row, win-rate of `by_col` (jockey or trainer) over the last
    `n_rides` PRIOR rides (excluding current). Returned in original df order.

    Implemented via cumsum + shift within group — pandas .rolling() across
    groups was including rows from adjacent groups in the window, producing
    win counts > ride counts. cumsum-based math is group-safe and O(N).
    """
    s = df.sort_values([by_col, 'date'], kind='stable').copy()
    s['_orig_idx'] = s.index
    s = s.reset_index(drop=True)
    grp = s.groupby(by_col, sort=False)

    # Cumulative within group
    cum_wins = grp['is_winner'].cumsum()
    cum_count = grp.cumcount() + 1  # 1-indexed total rides incl. current

    # Prior-only (exclude current row)
    prior_wins = cum_wins - s['is_winner']
    prior_count = cum_count - 1

    # Last N rides: subtract cum stats from (n+1) positions back.
    # Must shift WITHIN group — plain .shift() crosses group boundaries
    # and pulls in the previous group's last cumsum value, producing
    # nonsensical negative win counts at group starts.
    wins_n_ago = (
        grp['is_winner']
        .transform(lambda x: x.cumsum().shift(n_rides + 1))
        .fillna(0)
    )
    wins_last_n = prior_wins - wins_n_ago
    count_last_n = prior_count.clip(upper=n_rides)

    wr = (wins_last_n / count_last_n.replace(0, np.nan)).fillna(0)
    return pd.Series(wr.values, index=s['_orig_idx']).sort_index()


def build_training_table(db_path: Optional[str] = None) -> pd.DataFrame:
    """One-shot helper: load + compute features. Returns the runner-level
    DataFrame with FEATURE_COLS + TARGET + race_id + date for splitting.
    """
    raw = load_runner_table(db_path)
    return compute_features(raw)
