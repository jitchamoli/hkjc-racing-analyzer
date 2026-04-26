#!/usr/bin/env python3
"""
Apr 22 forensic probe — Task 4 of the ML audit plan.

Runs the full probability-layer trace on ALL 9 Apr 22 races from the holdout.
Reuses helper functions from _probe_trace.py (imported, not duplicated).

Output: docs/audits/2026-04-23/raw/apr22-trace.json

Pipeline layers traced (same as Task 3):
  raw_ml_prob  — XGBoost predict_proba BEFORE Platt scaling
  platt_prob   — post-Platt Logistic calibration
  softmax_prob — post-softmax within race (temperature=SOFTMAX_TEMPERATURE)
  blended_prob — post-market-blend (final ranking probability)

OPAQUE LAYERS (same caveat as Task 3):
  rule_score, combined_prob — rule engine not called in backtest pipeline

Apr 22 is confirmed to be in the holdout (73-race holdout includes all 9 Apr 22 races).
Venue: Sha Tin (DB confirms — task description said HV, but DB is ground truth).

Read-only: no production code is modified.

Usage:
    python3 docs/audits/2026-04-23/raw/_probe_apr22.py
"""

import json
import os
import sqlite3
import sys
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from scripts.backtest import (
    load_held_out_races,
    group_by_race,
    build_horse_data,
)
from ml.model import HKJCRacingMLModel
from ml.features import FeatureEngineering
from ml.feature_schema import FEATURE_NAMES
from core.scorer import _dynamic_blend_weights

# Import helpers from Task 3 probe (read-only reuse)
_probe_trace_path = Path(__file__).parent / "_probe_trace.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("_probe_trace", _probe_trace_path)
_probe_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe_module)

get_raw_ml_prob = _probe_module.get_raw_ml_prob
get_platt_prob = _probe_module.get_platt_prob
compute_softmax = _probe_module.compute_softmax
compute_market_blend = _probe_module.compute_market_blend
build_feature_dict = _probe_module.build_feature_dict
classify_failure = _probe_module.classify_failure
find_suspect_features = _probe_module.find_suspect_features

OUTPUT_PATH = Path(__file__).parent / "apr22-trace.json"
FAILURES_PATH = Path(__file__).parent / "holdout-failures.json"

TARGET_DATE = "2026/04/22"


def build_per_horse_trace(h, features_vec, model, raw_ml, platt, softmax_p, blended_p,
                          market_norm, ranked_indices, idx):
    name = h['horse_name']
    rank = ranked_indices.index(idx) + 1
    return {
        'horse_name': name,
        'features': build_feature_dict(features_vec),
        'raw_ml_prob': round(raw_ml, 6),
        'platt_prob': round(platt, 6),
        'softmax_prob': round(softmax_p, 6),
        'blended_prob': round(blended_p, 6),
        'market_implied_prob': round(market_norm, 6) if market_norm > 0 else 0.0,
        'finish_position': h.get('finish_position'),
        'final_rank': rank,
        'win_odds': h.get('win_odds'),
        'rule_score': None,      # opaque — rule engine not in backtest
        'combined_prob': None,   # opaque — depends on rule_score
    }


def compute_failure_flags(ranked_horses, blended_list, failure_row):
    """Compute all 5 failure flags from per-horse data."""
    actual_top2 = {h['horse_name'] for h in ranked_horses
                   if h.get('finish_position') in (1, 2)}

    # ranked_horses is the original list (not sorted by prob)
    # We need sorted-by-blended-prob names for our top picks
    # The failure_row from holdout already has this data — reuse it for consistency
    rank1_miss = bool(failure_row.get('rank1_miss'))
    top2_miss = bool(failure_row.get('top2_miss'))
    winner_deep = bool(failure_row.get('winner_deep'))
    qin_box3_miss = bool(failure_row.get('qin_box3_miss'))
    qin_box4_miss = bool(failure_row.get('qin_box4_miss'))

    return {
        'rank1_miss': rank1_miss,
        'top2_miss': top2_miss,
        'winner_deep': winner_deep,
        'qin_box3_miss': qin_box3_miss,
        'qin_box4_miss': qin_box4_miss,
    }


def compute_platt_compression(raw_ml_probs, platt_probs):
    """Compute raw-to-platt compression metrics across a field."""
    raw_range = max(raw_ml_probs) - min(raw_ml_probs)
    platt_range = max(platt_probs) - min(platt_probs)
    if platt_range < 1e-9:
        compression_ratio = float('inf')
    else:
        compression_ratio = raw_range / platt_range
    return {
        'raw_ml_range': round(raw_range, 6),
        'platt_range': round(platt_range, 6),
        'compression_ratio': round(compression_ratio, 2) if math.isfinite(compression_ratio) else 'inf',
    }


def trace_apr22_race(date, race_no, horses, model, fe, conn, failure_row):
    """Produce full trace for one Apr 22 race."""
    field_size = len(horses)
    distance = (horses[0].get('distance') or 1200) if horses else 1200

    per_horse = []
    for h in horses:
        horse_data = build_horse_data(h, field_size)
        features_vec = fe.extract_features(horse_data, [], '', conn=conn)
        raw_ml = get_raw_ml_prob(model, features_vec)
        platt = get_platt_prob(model, raw_ml)
        per_horse.append({
            'horse': h,
            'features_vec': features_vec,
            'raw_ml_prob': raw_ml,
            'platt_prob': platt,
            'odds': h.get('win_odds') or 0,
        })

    # Softmax over Platt probs
    platt_list = [ph['platt_prob'] for ph in per_horse]
    softmax_list = compute_softmax(platt_list)

    # Market blend
    odds_list = [ph['odds'] for ph in per_horse]
    blended_list, market_norm_list, model_w, market_w = compute_market_blend(
        softmax_list, odds_list, field_size, distance
    )

    # Sort by blended prob for ranks
    ranked_indices = sorted(range(len(per_horse)), key=lambda i: blended_list[i], reverse=True)

    # Build per-horse traces
    horse_traces = {}
    for i, ph in enumerate(per_horse):
        name = ph['horse']['horse_name']
        trace = build_per_horse_trace(
            ph['horse'], ph['features_vec'], model,
            ph['raw_ml_prob'], ph['platt_prob'],
            softmax_list[i], blended_list[i],
            market_norm_list[i] if i < len(market_norm_list) else 0.0,
            ranked_indices, i
        )
        horse_traces[name] = trace

    # Identify rank1..rank4 (by blended prob)
    def get_rank_trace(rank):
        if rank - 1 < len(ranked_indices):
            idx = ranked_indices[rank - 1]
            name = per_horse[idx]['horse']['horse_name']
            return horse_traces[name]
        return None

    rank1_trace = get_rank_trace(1)
    rank2_trace = get_rank_trace(2)
    rank3_trace = get_rank_trace(3)
    rank4_trace = get_rank_trace(4)

    # Identify actual winner
    winner = next((h for h in horses if h.get('finish_position') == 1), None)
    winner_name = winner['horse_name'] if winner else None
    winner_trace = horse_traces.get(winner_name) if winner_name else None

    # Failure flags (reuse holdout data for consistency with Task 2)
    failure_flags = compute_failure_flags(horses, blended_list, failure_row)

    # Failure layer classification (require both rank1 and winner traces)
    if rank1_trace and winner_trace and winner_name != rank1_trace['horse_name']:
        failure_layer, suspect_features = classify_failure(rank1_trace, winner_trace)
    elif winner_name == (rank1_trace['horse_name'] if rank1_trace else None):
        failure_layer = 'no-miss'
        suspect_features = []
    else:
        failure_layer = 'no-miss'
        suspect_features = []

    # Platt compression across entire field
    raw_probs = [ph['raw_ml_prob'] for ph in per_horse]
    platt_probs = [ph['platt_prob'] for ph in per_horse]
    compression = compute_platt_compression(raw_probs, platt_probs)

    # top_N_share: sum of blended_prob across top N model picks
    top_probs_sorted = sorted(blended_list, reverse=True)
    top_2_share = round(sum(top_probs_sorted[:2]), 6)
    top_3_share = round(sum(top_probs_sorted[:3]), 6)
    top_4_share = round(sum(top_probs_sorted[:4]), 6)

    # Confirm holdout membership (all Apr 22 races confirmed in holdout)
    is_in_holdout = True

    return {
        'race_date': date,
        'race_no': race_no,
        'field_size': field_size,
        'distance': distance,
        'venue': horses[0].get('venue', 'unknown') if horses else 'unknown',
        'is_in_holdout': is_in_holdout,
        'top_2_share': top_2_share,
        'top_3_share': top_3_share,
        'top_4_share': top_4_share,
        'rank1': rank1_trace,
        'rank2': rank2_trace,
        'rank3': rank3_trace,
        'rank4': rank4_trace,
        'winner': winner_trace,
        'blend_weights': {'model_w': round(model_w, 4), 'market_w': round(market_w, 4)},
        'failure_flags': failure_flags,
        'failure_layer': failure_layer,
        'suspect_features': suspect_features,
        'platt_compression_observed': compression,
        # Convenience fields for reporting
        'rank1_horse': rank1_trace['horse_name'] if rank1_trace else None,
        'winner_horse': winner_name,
        'winner_our_rank': winner_trace['final_rank'] if winner_trace else None,
    }


def main():
    failures = json.load(open(FAILURES_PATH))
    apr22_failures = {r['race_no']: r for r in failures if r.get('race_date') == TARGET_DATE}
    print(f"Apr 22 races in holdout-failures.json: {len(apr22_failures)}")

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # Load the full holdout (same 20% cut as Task 3)
    all_rows = conn.execute(
        "SELECT * FROM race_results WHERE finish_position > 0 "
        "ORDER BY race_date, race_number"
    ).fetchall()
    all_rows = [dict(r) for r in all_rows]
    cut_val = int(len(all_rows) * 0.8)
    holdout_rows = all_rows[cut_val:]
    races = group_by_race(holdout_rows)

    # Filter to Apr 22 only
    apr22_races = {k: v for k, v in races.items() if k[0] == TARGET_DATE}
    print(f"Apr 22 races found in holdout slice: {len(apr22_races)}")
    for (d, rno), horses in sorted(apr22_races.items()):
        print(f"  R{rno}: {len(horses)} horses, venue={horses[0].get('venue','?')}")

    model = HKJCRacingMLModel()
    fe = FeatureEngineering()
    if not model.is_trained:
        print("ERROR: model not loaded")
        sys.exit(1)
    print(f"\nModel loaded: version={model.model_version}, n_features={len(FEATURE_NAMES)}")

    traces = []
    for (date, race_no) in sorted(apr22_races.keys()):
        horses = apr22_races[(date, race_no)]
        failure_row = apr22_failures.get(race_no, {})
        print(f"\nTracing {date} R{race_no} ({len(horses)} horses)...")
        try:
            trace = trace_apr22_race(date, race_no, horses, model, fe, conn, failure_row)
            traces.append(trace)

            r1 = trace['rank1']
            w = trace['winner']
            comp = trace['platt_compression_observed']
            flags = [k for k, v in trace['failure_flags'].items() if v]
            print(f"  Rank1: {r1['horse_name']} platt={r1['platt_prob']:.4f} "
                  f"blended={r1['blended_prob']:.4f} finish={r1['finish_position']}")
            if w and w['horse_name'] != r1['horse_name']:
                print(f"  Winner: {w['horse_name']} platt={w['platt_prob']:.4f} "
                      f"blended={w['blended_prob']:.4f} rank={w['final_rank']}")
            else:
                print(f"  Winner = Rank1 (correct call)")
            print(f"  Failure flags: {flags if flags else '(none)'}")
            print(f"  Failure layer: {trace['failure_layer']}")
            print(f"  Platt compression: raw_range={comp['raw_ml_range']:.4f}, "
                  f"platt_range={comp['platt_range']:.6f}, ratio={comp['compression_ratio']}x")
        except Exception as e:
            import traceback
            print(f"  ERROR tracing {date} R{race_no}: {e}")
            traceback.print_exc()

    conn.close()

    # Summary stats
    print(f"\n{'='*60}")
    print(f"Summary: {len(traces)} races traced for {TARGET_DATE}")
    ratios = [t['platt_compression_observed']['compression_ratio']
              for t in traces
              if isinstance(t['platt_compression_observed']['compression_ratio'], (int, float))]
    raw_ranges = [t['platt_compression_observed']['raw_ml_range'] for t in traces]
    platt_ranges = [t['platt_compression_observed']['platt_range'] for t in traces]
    if ratios:
        print(f"  Avg Platt compression ratio: {sum(ratios)/len(ratios):.1f}x")
        print(f"  Avg raw_ml_range:  {sum(raw_ranges)/len(raw_ranges):.4f}")
        print(f"  Avg platt_range:   {sum(platt_ranges)/len(platt_ranges):.6f}")

    flag_counts = defaultdict(int)
    for t in traces:
        for flag, val in t['failure_flags'].items():
            if val:
                flag_counts[flag] += 1
    print(f"  Failure flag counts:")
    for flag, count in sorted(flag_counts.items()):
        print(f"    {flag}: {count}/{len(traces)}")

    layer_counts = defaultdict(int)
    for t in traces:
        layer_counts[t['failure_layer']] += 1
    print(f"  Failure layer distribution:")
    for layer, count in sorted(layer_counts.items()):
        print(f"    {layer}: {count}/{len(traces)}")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(traces, f, indent=2, default=str)
    print(f"\nWrote {len(traces)} traces to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == '__main__':
    main()
