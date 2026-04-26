#!/usr/bin/env python3
"""Backfill card-only fields (rating, last_6, gear, age, rating_change) into
race_results from the Renavon HKJC Race Results CSV export.

Renavon's `data (8).csv.gz` is the HKJC Race Results dataset — 368K rows × 84
cols including `entry_rating` (rating-at-time-of-race), `gear`, `horse_age`,
sectional times, etc. This script:

1. Loads only the columns we need from the gzipped CSV
2. Converts race_date YYYY-MM-DD → YYYY/MM/DD (our DB format)
3. Derives `last_6` per (race_date, horse) from the 6 most recent finishing
   positions strictly before that race
4. Derives `rating_change` per (race_date, horse) as the delta between this
   race's entry_rating and the previous race's entry_rating
5. Reuses HorseDatabase.backfill_card_fields() for the actual UPDATE pass

Usage: python3 -m scripts.backfill_from_renavon
"""
import gzip
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import HorseDatabase

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


RENAVON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'renavon_Hongkong_Jockey_Club_races_and_horses',
)
RACE_RESULTS_CSV = os.path.join(RENAVON_DIR, 'data (8).csv.gz')

# Columns we actually need (out of 84) — keeps memory footprint small
WANTED_COLS = [
    'race_date', 'race_number', 'horse_name', 'horse_number',
    'finishing_position', 'entry_rating', 'gear', 'horse_age',
]


def load_renavon_results() -> pd.DataFrame:
    """Read Renavon HKJC Race Results into a tidy DataFrame.

    Source dates look like '2026-04-26 00:00:00'; we strip to YYYY-MM-DD
    then convert to our DB format YYYY/MM/DD.
    """
    if not os.path.exists(RACE_RESULTS_CSV):
        raise FileNotFoundError(
            f"Renavon CSV not found: {RACE_RESULTS_CSV}\n"
            f"Place the one-time-export gzipped CSVs in {RENAVON_DIR}"
        )

    logger.info(f"Loading {RACE_RESULTS_CSV} ...")
    # Renavon uses '\N' for nulls
    df = pd.read_csv(
        RACE_RESULTS_CSV,
        usecols=WANTED_COLS,
        na_values=['\\N', ''],
        compression='gzip',
        dtype={'horse_age': 'object', 'gear': 'object'},
    )
    logger.info(f"  Loaded {len(df):,} rows")

    # Normalize date — DROP time portion, convert / to be DB-friendly
    df['race_date'] = (
        df['race_date']
        .astype(str)
        .str.split(' ').str[0]   # '2026-04-26 00:00:00' → '2026-04-26'
        .str.replace('-', '/', regex=False)  # → '2026/04/26'
    )
    return df


def derive_last_6_and_rating_change(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, derive:
      - last_6: space-separated string of last 6 finishing positions strictly before this race (most recent first → leftmost)
      - rating_change: entry_rating - prior race's entry_rating

    Manual per-horse iteration to avoid groupby column-dropping quirks.
    """
    df = df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)

    last_6 = [''] * len(df)
    rating_change = [0] * len(df)

    # Walk through, tracking per-horse history as we go
    history_pos = []        # finishing positions for current horse, in chronological order
    history_rating = []     # entry_ratings for current horse
    current_horse = None

    horse_col = df['horse_name'].values
    pos_col = df['finishing_position'].values
    rat_col = df['entry_rating'].values

    for i in range(len(df)):
        h = horse_col[i]
        if h != current_horse:
            history_pos = []
            history_rating = []
            current_horse = h

        # last_6: most recent first
        prior = history_pos[-6:][::-1]
        last_6[i] = ' '.join(str(int(p)) for p in prior if pd.notna(p))

        # rating_change vs most recent prior non-null rating
        r_now = rat_col[i]
        if pd.notna(r_now) and history_rating:
            prior_r = next((p for p in reversed(history_rating) if pd.notna(p)), None)
            if prior_r is not None:
                rating_change[i] = int(r_now - prior_r)

        # Update history with this race's data (for the NEXT row)
        history_pos.append(pos_col[i])
        history_rating.append(rat_col[i])

    df['last_6'] = last_6
    df['rating_change'] = rating_change
    return df


def backfill_dates(db: HorseDatabase, df: pd.DataFrame, dates_to_apply=None) -> int:
    """Group df by race_date and call db.backfill_card_fields for each.

    If dates_to_apply is provided (set of YYYY/MM/DD strings), only those
    dates are processed. Otherwise every date in df is processed.
    """
    total_updates = 0
    grouped = df.groupby('race_date')
    n_dates = grouped.ngroups
    logger.info(f"Backfilling {n_dates} unique race dates...")

    processed = 0
    for race_date, group in grouped:
        if dates_to_apply is not None and race_date not in dates_to_apply:
            continue

        # Build race_data structure: list of races, each with horses[]
        races_dict = {}
        for _, row in group.iterrows():
            rno = int(row['race_number']) if pd.notna(row['race_number']) else 0
            if rno == 0:
                continue
            race = races_dict.setdefault(rno, {'race_number': rno, 'horses': []})
            horse = {
                'name': row['horse_name'],
                'rating': int(row['entry_rating']) if pd.notna(row['entry_rating']) else 0,
                'gear': str(row['gear']) if pd.notna(row['gear']) else '',
                'age': str(row['horse_age']) if pd.notna(row['horse_age']) else '',
                'last_6': row.get('last_6', ''),
                'rating_change': int(row.get('rating_change', 0) or 0),
            }
            race['horses'].append(horse)

        races = list(races_dict.values())
        n = db.backfill_card_fields(race_date, races)
        if n > 0:
            logger.info(f"  {race_date}: {len(races)} races · {n} rows updated")
            total_updates += n
        processed += 1

    logger.info(f"Processed {processed} dates · {total_updates} total row updates")
    return total_updates


def main():
    db = HorseDatabase()

    # Compute target dates: every date currently in our race_results
    # Renavon coverage is broader than ours, so filter to overlap.
    our_dates = {
        r[0] for r in db.conn.execute(
            "SELECT DISTINCT race_date FROM race_results"
        ).fetchall()
    }
    logger.info(f"Our race_results has {len(our_dates)} distinct dates")

    df = load_renavon_results()
    df = derive_last_6_and_rating_change(df)

    backfill_dates(db, df, dates_to_apply=our_dates)

    # Report final coverage
    total = db.conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]
    print()
    logger.info("Final card-field coverage:")
    for col in ['rating', 'last_6', 'gear', 'age', 'rating_change']:
        nn = db.conn.execute(
            f"SELECT COUNT(*) FROM race_results "
            f"WHERE {col} IS NOT NULL AND {col} != '' AND {col} != 0"
        ).fetchone()[0]
        bar = '#' * int(nn / total * 30)
        logger.info(f"  {col:<15}: {nn:>5}/{total} ({100*nn/total:5.1f}%) {bar}")

    db.close()


if __name__ == '__main__':
    main()
