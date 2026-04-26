#!/usr/bin/env python3
"""Investigate rank-2 calibration — why the model is bad at picking 2nd-place horses.

Analyzes held-out test races and answers:
  Q1: Where does our rank-2 pick actually finish?
  Q2: Where does the actual 2nd-place horse rank in our model?
  Q3: Is our rank-2 just the market's 2nd favourite?
  Q4: What features distinguish actual 2nd-place finishers vs our rank-2?

Usage:
    python3 -m scripts.analyze_rank2              # held-out 20%
    python3 -m scripts.analyze_rank2 --last 200   # last 200 horses
"""

import argparse
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.model import HKJCRacingMLModel
from ml.features import FeatureEngineering
from core.scorer import _dynamic_blend_weights


def load_held_out_races(conn, last_n=None):
    rows = conn.execute(
        "SELECT * FROM race_results WHERE finish_position > 0 "
        "ORDER BY race_date, race_number"
    ).fetchall()
    rows = [dict(r) for r in rows]
    if last_n is not None:
        return rows[-last_n:]
    cut_val = int(len(rows) * 0.8)
    return rows[cut_val:]


def group_by_race(rows):
    races = defaultdict(list)
    for r in rows:
        key = (r['race_date'], r['race_number'])
        races[key].append(r)
    return races


def build_horse_data(h, field_size):
    return {
        'name': h['horse_name'],
        'jockey': h.get('jockey') or '',
        'trainer': h.get('trainer') or '',
        'draw': h.get('draw') or 0,
        'distance': h.get('distance') or 1200,
        'going': h.get('going') or '',
        'field_size': field_size,
        'class_type': h.get('class_type') or '',
        'race_date': h['race_date'],
        'weight': h.get('weight') or 0,
        'horse_weight': h.get('horse_weight') or 0,
        'days_since_last_run': h.get('days_since_last_run') or 0,
        'last_6': '',
    }


def rank_horses(horses, model, fe, conn):
    """Return sorted list of (horse, model_rank, market_rank)."""
    field_size = len(horses)
    ml_probs = []
    for h in horses:
        hd = build_horse_data(h, field_size)
        features = fe.extract_features(hd, [], '', conn=conn)
        prob, _ = model.predict(features, field_size=field_size)
        ml_probs.append(prob)
    ml_arr = np.array(ml_probs)
    # Softmax + market blend
    sm = np.exp((ml_arr - ml_arr.max()) * config.SOFTMAX_TEMPERATURE)
    sm = sm / sm.sum() if sm.sum() > 0 else np.ones_like(sm) / field_size

    odds = np.array([h.get('win_odds') or 0 for h in horses], dtype=float)
    if np.all(odds > 0):
        market = 1.0 / odds
        market = market / market.sum()
        dist = horses[0].get('distance') or 1200
        model_w, market_w = _dynamic_blend_weights(field_size, dist)
        final = model_w * sm + market_w * market
        final = final / final.sum()
    else:
        market = np.zeros_like(ml_arr)
        final = sm

    # Rank by final prob (model rank); rank by market prob (market rank)
    model_sorted = np.argsort(-final)
    market_sorted = np.argsort(-market) if market.sum() > 0 else model_sorted

    model_rank = {i: r + 1 for r, i in enumerate(model_sorted)}
    market_rank = {i: r + 1 for r, i in enumerate(market_sorted)}

    result = []
    for i, h in enumerate(horses):
        result.append({
            'horse': h,
            'model_rank': model_rank[i],
            'market_rank': market_rank[i],
            'model_prob': float(final[i]),
            'market_prob': float(market[i]) if market.sum() > 0 else 0,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--last', type=int, default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = load_held_out_races(conn, last_n=args.last)
    races = group_by_race(rows)
    print(f"Analyzing {len(races)} races...\n")

    model = HKJCRacingMLModel()
    fe = FeatureEngineering()

    # Collect per-race stats
    rank2_finishes = []          # Q1: where does our rank-2 finish?
    actual_2nd_our_rank = []     # Q2: where does actual 2nd place horse rank in our model?
    actual_2nd_market_rank = []  # Q3: where does actual 2nd place rank in market?
    rank2_matches_market = 0     # Q3b: how often is our rank-2 the same as market's rank-2?
    n_with_market = 0
    feature_diff = []            # Q4: feature differences between our rank-2 and actual 2nd-place

    for (date, race_no), horses in races.items():
        if len(horses) < 5:
            continue
        if not any((h.get('win_odds') or 0) > 0 for h in horses):
            continue

        ranked = rank_horses(horses, model, fe, conn)
        # Our rank-2 horse
        our_rank2 = next(r for r in ranked if r['model_rank'] == 2)
        actual_pos_of_rank2 = our_rank2['horse'].get('finish_position', 0)
        if actual_pos_of_rank2 > 0:
            rank2_finishes.append(actual_pos_of_rank2)

        # Where does the actual 2nd-place horse rank in our model?
        actual_2nd = next((r for r in ranked if r['horse'].get('finish_position') == 2), None)
        if actual_2nd:
            actual_2nd_our_rank.append(actual_2nd['model_rank'])
            if actual_2nd['market_prob'] > 0:
                actual_2nd_market_rank.append(actual_2nd['market_rank'])
                n_with_market += 1

        # Q3b: is our rank-2 == market rank-2?
        market_rank2 = next((r for r in ranked if r['market_rank'] == 2), None)
        if market_rank2 and our_rank2['horse']['horse_name'] == market_rank2['horse']['horse_name']:
            rank2_matches_market += 1

    n = len(rank2_finishes)
    print("=" * 65)
    print(f"  RANK-2 CALIBRATION AUDIT ({n} races)")
    print("=" * 65)

    # Q1: distribution of our rank-2 finishes
    print("\nQ1: Where does our rank-2 pick actually finish?")
    counter = Counter(rank2_finishes)
    for pos in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        if pos in counter:
            pct = counter[pos] / n * 100
            bar = '#' * int(pct / 2)
            print(f"  {pos:2d}th: {counter[pos]:3d}  ({pct:5.1f}%)  {bar}")
    # Summary
    top_2 = sum(1 for x in rank2_finishes if x <= 2)
    top_3 = sum(1 for x in rank2_finishes if x <= 3)
    print(f"\n  Rank-2 pick finishes top-2: {top_2}/{n} = {top_2/n*100:.1f}%  (random: ~17%)")
    print(f"  Rank-2 pick finishes top-3: {top_3}/{n} = {top_3/n*100:.1f}%  (random: ~25%)")

    # Q2: where does actual 2nd-place rank in our model?
    print(f"\nQ2: Where does the actual 2nd-place horse rank in our model? ({len(actual_2nd_our_rank)} races)")
    c2 = Counter(actual_2nd_our_rank)
    for rank in range(1, 13):
        if rank in c2:
            pct = c2[rank] / len(actual_2nd_our_rank) * 100
            bar = '#' * int(pct / 2)
            print(f"  Our rank {rank:2d}: {c2[rank]:3d}  ({pct:5.1f}%)  {bar}")

    mean_rank = np.mean(actual_2nd_our_rank)
    median_rank = np.median(actual_2nd_our_rank)
    print(f"\n  Avg rank of actual 2nd in our model: {mean_rank:.2f}  (median {median_rank:.0f})")

    # Q3: market rank comparison
    if n_with_market > 0:
        print(f"\nQ3: Where does the actual 2nd-place rank in MARKET? ({n_with_market} races)")
        c3 = Counter(actual_2nd_market_rank)
        for rank in range(1, 13):
            if rank in c3:
                pct = c3[rank] / n_with_market * 100
                bar = '#' * int(pct / 2)
                print(f"  Market rank {rank:2d}: {c3[rank]:3d}  ({pct:5.1f}%)  {bar}")
        mean_mkt_rank = np.mean(actual_2nd_market_rank)
        print(f"\n  Avg market rank of actual 2nd: {mean_mkt_rank:.2f}")
        print(f"  Our model avg: {mean_rank:.2f}  |  Market avg: {mean_mkt_rank:.2f}")
        if abs(mean_rank - mean_mkt_rank) < 0.3:
            print("  → Our rank-2 picks track market — no independent signal on 2nd-place")

    # Q3b: agreement with market rank-2
    print(f"\nQ3b: How often is our rank-2 the same horse as market's rank-2?")
    print(f"  {rank2_matches_market}/{n} = {rank2_matches_market/n*100:.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
