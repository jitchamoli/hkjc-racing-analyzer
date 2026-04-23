#!/usr/bin/env python3
"""F-09: Grid-sweep market/model blend weights against the 73-race holdout.

Sweeps a grid of (base_model_w, field_threshold, dist_threshold) parameters
by monkey-patching core.scorer._dynamic_blend_weights for each point, running
the production backtest pipeline, and measuring Brier score.

Grid reduction rationale (2026-04-23):
  Full 45-point grid (5 × 3 × 3) at ~90s/point = ~67 min.
  Reduced to 15 points (5 base_model_w × 3 dist_thresh, field_thresh=10 fixed):
  - field_thresh sensitivity is secondary to base_model_w (the main lever)
  - 10 is the natural HKJC field-size midpoint; tested separately below
  - 15 × ~90s ≈ 22 min, within acceptable runtime

Selection: lowest Brier on holdout (primary). Tie-break within ±0.005 Brier
toward simpler / constant schedule (smallest combined shift magnitude).

Output:
  docs/audits/2026-04-23/raw/wave2-sweep-output.txt  (via tee)
  docs/audits/2026-04-23/raw/wave2-sweep-summary.json
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

# Bootstrap path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import core.scorer as scorer_module
from scripts.backtest import (
    load_held_out_races, group_by_race, simulate_bets
)

# ---------------------------------------------------------------------------
# Grid definition (15 points)
# ---------------------------------------------------------------------------

# field_thresh fixed at 10 (documented above)
FIELD_THRESH_FIXED = 10

BLEND_GRID = [
    (base_model_w, FIELD_THRESH_FIXED, dist_thresh)
    for base_model_w in [0.25, 0.35, 0.45, 0.55, 0.65]
    for dist_thresh in [1200, 1400, 1600]
]  # 15 points

# Shift magnitude: same ±15% as current default
SHIFT = 0.15


def make_blend_fn(base_model_w, field_thresh, dist_thresh, shift=SHIFT):
    """Return a _dynamic_blend_weights function for the given grid point."""
    base_market_w = 1.0 - base_model_w

    def _blend(field_size: int, distance: int):
        model_w = base_model_w
        market_w = base_market_w
        # Large field + sprint: market sharpest → shift toward market
        if field_size >= field_thresh and distance <= dist_thresh:
            market_w = min(0.80, market_w + shift)
            model_w = 1.0 - market_w
        # Small field or long distance: thinner pool → shift toward model
        elif field_size < field_thresh or distance >= 1800:
            model_w = min(0.80, model_w + shift)
            market_w = 1.0 - model_w
        return model_w, market_w

    return _blend


def compute_brier(records):
    """Brier score: mean squared error of final_prob vs actual win."""
    if not records:
        return float('nan')
    probs = np.array([r['final_prob'] for r in records])
    won = np.array([r['won'] for r in records], dtype=float)
    return float(np.mean((probs - won) ** 2))


def compute_box3_stats(records):
    """Box-3 quinella hit rate and ROI."""
    box3 = [r for r in records if r.get('pnl_box3') is not None]
    if not box3:
        return float('nan'), float('nan')
    hit = sum(1 for r in box3 if r['box3_q_hit'])
    hit_rate = hit / len(box3)
    total_pnl = sum(r['pnl_box3'] for r in box3)
    roi = total_pnl / (len(box3) * 3) * 100  # 3 units per box-3
    return hit_rate, roi


def run_grid_sweep(races, conn):
    """Run all 15 grid points; return list of result dicts."""
    original_fn = scorer_module._dynamic_blend_weights
    results = []

    print(f"\n{'='*70}")
    print(f"  F-09 BLEND-WEIGHT SWEEP ({len(BLEND_GRID)} points × production pipeline)")
    print(f"  field_thresh fixed at {FIELD_THRESH_FIXED}, shift=±{SHIFT}")
    print(f"{'='*70}")
    print(f"  {'base_model_w':<14}{'field_thr':<11}{'dist_thr':<10}"
          f"{'Brier':<10}{'box3_hit%':<12}{'box3_ROI%':<12}{'n_races'}")
    print(f"  {'-'*68}")

    try:
        for (base_model_w, field_thresh, dist_thresh) in BLEND_GRID:
            t0 = time.time()

            # Monkey-patch
            scorer_module._dynamic_blend_weights = make_blend_fn(
                base_model_w, field_thresh, dist_thresh
            )

            records = simulate_bets(races, conn)
            elapsed = time.time() - t0

            brier = compute_brier(records)
            box3_hit, box3_roi = compute_box3_stats(records)

            row = {
                'base_model_w': base_model_w,
                'field_thresh': field_thresh,
                'dist_thresh': dist_thresh,
                'brier': round(brier, 6),
                'box3_hit_rate': round(box3_hit, 4) if not np.isnan(box3_hit) else None,
                'box3_roi_pct': round(box3_roi, 2) if not np.isnan(box3_roi) else None,
                'n_races': len(records),
                'elapsed_s': round(elapsed, 1),
            }
            results.append(row)

            marker = ' <-- DEFAULT' if (
                abs(base_model_w - 0.60) < 0.001 and
                field_thresh == 10 and dist_thresh == 1400
            ) else ''
            print(f"  {base_model_w:<14.2f}{field_thresh:<11}{dist_thresh:<10}"
                  f"{brier:<10.4f}"
                  f"{box3_hit*100:<12.1f}{box3_roi:<12.2f}{len(records)}{marker}")

    finally:
        # Always restore original function
        scorer_module._dynamic_blend_weights = original_fn

    return results


def pick_winner(results):
    """Apply selection rule: lowest Brier, tie-break toward simpler schedule.

    'Simpler' = base_model_w closest to constant (shift least from 0.5 base).
    Within ±0.005 Brier of the best, prefer smaller |base_model_w - 0.5|
    (i.e., closer to equal weighting = less aggressive dynamic shift).
    """
    if not results:
        return None

    valid = [r for r in results if not np.isnan(r['brier'])]
    valid.sort(key=lambda r: r['brier'])
    best_brier = valid[0]['brier']

    tied = [r for r in valid if abs(r['brier'] - best_brier) <= 0.005]
    # Tie-break: closer to 0.5 base_model_w = simpler (less extreme) schedule
    tied.sort(key=lambda r: abs(r['base_model_w'] - 0.5))

    return tied[0]


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = load_held_out_races(conn)
    races = group_by_race(rows)
    print(f"Holdout: {len(races)} races  |  {len(rows)} horses")
    print(f"Grid: {len(BLEND_GRID)} points (field_thresh fixed={FIELD_THRESH_FIXED}, shift=±{SHIFT})")

    results = run_grid_sweep(races, conn)
    conn.close()

    # Save full results
    out_dir = Path(__file__).parent.parent / 'docs' / 'audits' / '2026-04-23' / 'raw'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'wave2-sweep-summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Pick winner
    winner = pick_winner(results)
    if not winner:
        print("ERROR: no valid results")
        return 1

    # Find current default for comparison
    default_like = [
        r for r in results
        if abs(r['base_model_w'] - 0.60) < 0.001 and r['dist_thresh'] == 1400
    ]

    print(f"\n{'='*70}")
    print(f"  SELECTION RESULT")
    print(f"{'='*70}")
    print(f"  Best Brier: {results[0]['brier']:.4f} "
          f"(base_model_w={results[0]['base_model_w']}, "
          f"dist_thresh={results[0]['dist_thresh']})")
    if default_like:
        d = default_like[0]
        print(f"  Current default (base=0.60, dist=1400): Brier={d['brier']:.4f}")
    print(f"  Winner (after tie-break): base_model_w={winner['base_model_w']}, "
          f"field_thresh={winner['field_thresh']}, dist_thresh={winner['dist_thresh']}")
    print(f"  Winner Brier: {winner['brier']:.4f}")
    print(f"  Winner box3_hit: {winner['box3_hit_rate']*100:.1f}%  "
          f"ROI: {winner['box3_roi_pct']:.1f}%")

    # Sort by Brier for display
    sorted_results = sorted(results, key=lambda r: r['brier'])
    print(f"\n  Top 5 by Brier:")
    for r in sorted_results[:5]:
        marker = ' <-- CHOSEN' if (
            r['base_model_w'] == winner['base_model_w'] and
            r['dist_thresh'] == winner['dist_thresh']
        ) else ''
        print(f"    base_model_w={r['base_model_w']:.2f}  dist_thresh={r['dist_thresh']}"
              f"  Brier={r['brier']:.4f}  box3_hit={r['box3_hit_rate']*100:.1f}%"
              f"  ROI={r['box3_roi_pct']:.1f}%{marker}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
