#!/usr/bin/env python3
"""
Forensic trace probe: for each selected severe-miss race, emit a full
feature + probability trace for rank-1 (model's pick) and the actual winner.

Read-only. Production code (core/, ml/, scripts/, data/) is never modified.
Output: docs/audits/2026-04-23/raw/trace-analysis.json

Pipeline layers traced:
  raw_ml_prob   — XGBoost predict_proba output, BEFORE Platt scaling
  platt_prob    — post-Platt Logistic calibration (what backtest calls ml_prob)
  softmax_prob  — post-softmax within race (temperature=1.5)
  blended_prob  — post-market-blend (final ranking probability)

OPAQUE LAYERS (noted honestly):
  rule_score       — rule engine not called in backtest; would require fetching
                     live race cards. Marked null/opaque.
  combined_prob    — the 60/40 rule+ML blend. Opaque for same reason; backtest
                     uses ML-only signal (ml_prob → softmax → market blend).
                     The backtest's "final_prob" is NOT the same as the
                     dashboard's combined_prob.

market_implied_prob — 1/odds, normalized within race. Available from backtest.

Usage:
    python3 docs/audits/2026-04-23/raw/_probe_trace.py
"""

import json
import os
import sqlite3
import sys
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ensure project root is on path
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

OUTPUT_PATH = Path(__file__).parent / "trace-analysis.json"
FAILURES_PATH = Path(__file__).parent / "holdout-failures.json"

THRESHOLD_USED = 0.35   # fallback from 0.40 (only 1 case at 0.40)
MAX_CASES = 10


def select_severe_misses(failures, threshold):
    sev = [
        r for r in failures
        if r.get('rank1_miss')
        and r.get('top_2_share', 0) >= threshold
        and not r.get('data_incomplete')
    ]
    sev.sort(key=lambda r: -r.get('top_2_share', 0))
    return sev[:MAX_CASES]


def get_raw_ml_prob(model, features_vec):
    """Return XGBoost output BEFORE Platt scaling."""
    proba = model.model.predict_proba(features_vec.reshape(1, -1))[0]
    return float(proba[1]) if len(proba) > 1 else float(proba[0])


def get_platt_prob(model, raw_prob):
    """Return Platt-calibrated probability from raw XGBoost prob."""
    if model.calibrator is not None:
        return float(model.calibrator.predict_proba(np.array([[raw_prob]]))[0, 1])
    return raw_prob


def compute_softmax(platt_probs, temperature=None):
    """Compute softmax over a list of platt probs at given temperature."""
    t = temperature if temperature is not None else config.SOFTMAX_TEMPERATURE
    arr = np.array(platt_probs, dtype=float)
    shifted = (arr - arr.max()) * t
    exp_vals = np.exp(shifted)
    total = exp_vals.sum()
    if total > 0:
        return (exp_vals / total).tolist()
    return (np.ones_like(arr) / len(arr)).tolist()


def compute_market_blend(softmax_probs, odds_list, field_size, distance):
    """Apply dynamic market blend and return blended probs."""
    model_w, market_w = _dynamic_blend_weights(field_size, distance)
    odds_arr = np.array([o if o and o > 0 else 0.0 for o in odds_list], dtype=float)
    sm_arr = np.array(softmax_probs, dtype=float)

    if np.all(odds_arr > 0):
        market_raw = 1.0 / odds_arr
        market_norm = market_raw / market_raw.sum()
        blended = model_w * sm_arr + market_w * market_norm
        blended = blended / blended.sum()
        return blended.tolist(), market_norm.tolist(), model_w, market_w
    else:
        return sm_arr.tolist(), [0.0] * len(odds_list), model_w, market_w


def build_feature_dict(features_vec):
    """Map feature vector to named dict."""
    return {FEATURE_NAMES[i]: float(features_vec[i]) for i in range(len(FEATURE_NAMES))}


def classify_failure(rank1_trace, winner_trace):
    """
    Classify failure layer by comparing intermediate probabilities.

    Rules (applied in order of specificity):
      calibration-driven: winner's raw_ml_prob > rank1's raw_ml_prob
        (ML ranked winner correctly but Platt/softmax flipped order)
      blend-driven: rank1 market_implied > rank1 platt_prob by large margin (>4x)
        AND winner would have won without market blend
      feature-driven: rank1 raw_ml_prob > winner's raw_ml_prob AND
        one or more features stand out as suspicious (fallback values or large delta)
      mixed: elements of multiple layers present
    """
    r1_raw = rank1_trace.get('raw_ml_prob', 0)
    w_raw = winner_trace.get('raw_ml_prob', 0)
    r1_platt = rank1_trace.get('platt_prob', 0)
    w_platt = winner_trace.get('platt_prob', 0)
    r1_mkt = rank1_trace.get('market_implied_prob', 0)
    r1_softmax = rank1_trace.get('softmax_prob', 0)
    w_softmax = winner_trace.get('softmax_prob', 0)
    r1_blended = rank1_trace.get('blended_prob', 0)
    w_blended = winner_trace.get('blended_prob', 0)

    calibration_driven = (w_raw is not None and r1_raw is not None and w_raw > r1_raw)
    # After Platt, is winner still ranked above rank1?
    platt_winner_higher = (w_platt is not None and r1_platt is not None and w_platt > r1_platt)

    # Blend-driven: market strongly boosted rank1 over winner
    # Check: rank1 market prob >> rank1 softmax prob, winner softmax > winner blended
    blend_driven = False
    if (r1_mkt is not None and r1_softmax is not None and r1_mkt > 0
            and w_softmax is not None and w_blended is not None):
        # Market share substantially higher than model share for rank1
        mkt_model_ratio = r1_mkt / max(r1_softmax, 0.001)
        winner_softmax_wins = w_softmax > r1_softmax
        if mkt_model_ratio > 1.5 and winner_softmax_wins:
            blend_driven = True

    # Identify suspicious features for rank1 vs winner
    suspect = find_suspect_features(rank1_trace.get('features', {}),
                                    winner_trace.get('features', {}))

    feature_driven = (not calibration_driven and not platt_winner_higher
                      and len(suspect) > 0)

    # Classify
    layers = []
    if calibration_driven or platt_winner_higher:
        layers.append('calibration-driven')
    if blend_driven and not calibration_driven:
        layers.append('blend-driven')
    if feature_driven or (suspect and not calibration_driven):
        layers.append('feature-driven')

    if len(layers) == 0:
        # Default: if winner had lower raw ML, model ranked rank1 correctly at raw level
        # but still lost — call it feature-driven (model had wrong features)
        layer = 'feature-driven'
    elif len(layers) == 1:
        layer = layers[0]
    else:
        layer = 'mixed'

    return layer, suspect


def find_suspect_features(r1_features, w_features):
    """
    Identify features where rank1 has suspicious values OR winner clearly dominates.
    Returns list of suspect feature names.
    """
    suspects = []
    fallback_thresholds = {
        'prior_win_rate': 0.5,      # exact fallback = no history
        'recent_form': 0.5,          # exact fallback
        'adjusted_speed': 0.5,       # exact fallback (no finish times)
        'cd_form_prior': 0.5,        # exact fallback
        'jockey_horse_coop': 0.5,    # exact fallback (<2 shared rides)
        'going_performance': 0.5,    # exact fallback
        'field_rel_speed': 0.5,      # exact fallback
        'last_start_margin': 0.5,    # exact fallback
        'distance_pb_ratio': 0.5,    # exact fallback
        'speed_consistency': 0.5,    # exact fallback
        'prior_pace': 0.5,           # exact fallback
        'days_since_last_win': 0.5,  # exact fallback (no wins in history)
    }
    notable_delta = 0.20  # feature delta threshold for "winner clearly stronger"

    for feat_name, fallback_val in fallback_thresholds.items():
        r1_val = r1_features.get(feat_name)
        w_val = w_features.get(feat_name)
        if r1_val is None:
            continue
        # Rank1 stuck at fallback
        if abs(r1_val - fallback_val) < 0.01:
            suspects.append(feat_name)
        # Winner clearly dominates on this feature (and rank1 is not at fallback)
        elif w_val is not None and (w_val - r1_val) >= notable_delta:
            suspects.append(feat_name)

    # Check non-fallback features for large winner advantage
    for feat_name in FEATURE_NAMES:
        if feat_name in fallback_thresholds:
            continue
        r1_val = r1_features.get(feat_name)
        w_val = w_features.get(feat_name)
        if r1_val is not None and w_val is not None and (w_val - r1_val) >= notable_delta:
            suspects.append(feat_name)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for s in suspects:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def trace_race(date, race_no, horses, model, fe, conn, failure_row):
    """Produce a full trace dict for one race."""
    field_size = len(horses)
    distance = (horses[0].get('distance') or 1200) if horses else 1200

    # Extract features and compute probability layers for each horse
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

    # Sort by blended prob to get ranks
    ranked_indices = sorted(range(len(per_horse)), key=lambda i: blended_list[i], reverse=True)

    # Map horse name → trace
    horse_traces = {}
    for i, ph in enumerate(per_horse):
        name = ph['horse']['horse_name']
        rank = ranked_indices.index(i) + 1
        horse_traces[name] = {
            'horse_name': name,
            'features': build_feature_dict(ph['features_vec']),
            'raw_ml_prob': round(ph['raw_ml_prob'], 6),
            'platt_prob': round(ph['platt_prob'], 6),
            'softmax_prob': round(softmax_list[i], 6),
            'blended_prob': round(blended_list[i], 6),
            'market_implied_prob': round(market_norm_list[i], 6) if market_norm_list[i] > 0 else 0.0,
            'finish_position': ph['horse'].get('finish_position'),
            'final_rank': rank,
            # Opaque layers — rule engine not called in backtest pipeline
            'rule_score': None,
            'combined_prob': None,
        }

    # Identify rank1 (highest blended prob) and winner
    rank1_name = per_horse[ranked_indices[0]]['horse']['horse_name']
    winner = next((h for h in horses if h.get('finish_position') == 1), None)
    winner_name = winner['horse_name'] if winner else None

    rank1_trace = horse_traces.get(rank1_name, {})
    winner_trace = horse_traces.get(winner_name, {}) if winner_name else {}

    # Classify failure layer
    failure_layer, suspect_features = classify_failure(rank1_trace, winner_trace)

    # Add failure_layer + suspects to both traces
    rank1_trace['failure_layer'] = failure_layer
    rank1_trace['suspect_features'] = suspect_features

    return {
        'race_date': date,
        'race_no': race_no,
        'field_size': field_size,
        'top_2_share': round(failure_row.get('top_2_share', 0), 6),
        'threshold_used': THRESHOLD_USED,
        'blend_weights': {'model_w': round(model_w, 4), 'market_w': round(market_w, 4)},
        'rank1': rank1_trace,
        'winner': winner_trace if winner_trace else None,
        'failure_layer': failure_layer,
        'suspect_features': suspect_features,
    }


def main():
    failures = json.load(open(FAILURES_PATH))
    selected = select_severe_misses(failures, THRESHOLD_USED)
    print(f"Selected {len(selected)} severe misses at threshold top_2_share >= {THRESHOLD_USED}")
    for r in selected:
        print(f"  {r['race_date']} R{r['race_no']}: {r['rank1_horse']} "
              f"finished {r['rank1_finish']}, top_2_share={r['top_2_share']:.3f}, "
              f"winner={r.get('winner_name')} rank={r.get('winner_our_rank')}")

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    all_rows = conn.execute(
        "SELECT * FROM race_results WHERE finish_position > 0 ORDER BY race_date, race_number"
    ).fetchall()
    all_rows = [dict(r) for r in all_rows]
    # Get the holdout cut
    cut_val = int(len(all_rows) * 0.8)
    holdout_rows = all_rows[cut_val:]
    races = group_by_race(holdout_rows)

    model = HKJCRacingMLModel()
    fe = FeatureEngineering()
    if not model.is_trained:
        print("ERROR: model not loaded")
        sys.exit(1)

    # Build index for failure_row lookup
    failure_index = {(r['race_date'], r['race_no']): r for r in failures}

    traces = []
    for failure_row in selected:
        date = failure_row['race_date']
        race_no = failure_row['race_no']
        key = (date, race_no)
        horses = races.get(key)
        if not horses:
            print(f"  WARNING: no horses found in holdout for {date} R{race_no}")
            continue
        print(f"\nTracing {date} R{race_no} ({len(horses)} horses)...")
        try:
            trace = trace_race(date, race_no, horses, model, fe, conn, failure_row)
            traces.append(trace)
            r1 = trace['rank1']
            w = trace['winner']
            print(f"  Rank1: {r1['horse_name']} platt={r1['platt_prob']:.4f} blended={r1['blended_prob']:.4f} finish={r1['finish_position']}")
            if w:
                print(f"  Winner: {w['horse_name']} platt={w['platt_prob']:.4f} blended={w['blended_prob']:.4f} rank={w['final_rank']}")
            print(f"  Failure layer: {trace['failure_layer']}, suspects: {trace['suspect_features']}")
        except Exception as e:
            import traceback
            print(f"  ERROR tracing {date} R{race_no}: {e}")
            traceback.print_exc()

    conn.close()

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(traces, f, indent=2, default=str)
    print(f"\nWrote {len(traces)} traces to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")


def group_by_race(rows):
    races = defaultdict(list)
    for r in rows:
        key = (r['race_date'], r['race_number'])
        races[key].append(r)
    return races


if __name__ == '__main__':
    main()
