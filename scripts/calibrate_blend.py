#!/usr/bin/env python3
"""Fit a post-blend isotonic calibrator on production calibrated_prob.

Background: ml/training.py fits an isotonic calibrator on raw XGBoost output
(pre-rule-blend, pre-market-blend). The post-C1 backtest revealed a residual
under-prediction bias on the *post-blend* distribution — top picks predicted
~32% actually win ~46% of the time on the 5,966-race holdout.

This script fits a SECOND isotonic on (production calibrated_prob → actual_won)
across the held-out 20% of historical races. It saves to
`models/blend_calibrator.pkl` (with .sha256 sidecar) and is loaded + applied
by core/scorer.py:calibrate_probabilities after the market blend.

Usage:
    python3 -m scripts.calibrate_blend
    python3 -m scripts.calibrate_blend --last 1000   # smaller sample for iteration
"""

import argparse
import hashlib
import logging
import os
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def collect_calibration_pairs(last_n=None):
    """Iterate the 70-85% calib fold and collect (calibrated_prob, is_winner) per horse.

    Uses load_calibration_window (NOT load_held_out_races) so the blend
    calibrator is fit on data DISJOINT from both the model's training
    folds AND backtest's test fold. This was a leakage source — calibrator
    fit on the same slice backtest later evaluated on, producing optimistic
    metrics that didn't generalize to live race day.

    Reuses scripts.backtest helpers so the production pipeline is exercised
    identically to the live race-day path.
    """
    from scripts.backtest import (
        load_calibration_window, group_by_race, predict_race_production,
    )

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = load_calibration_window(conn)
    if last_n is not None and last_n < len(rows):
        rows = rows[-last_n:]
    races = group_by_race(rows)
    logger.info(f"Calib window (70-85%): {len(rows)} horses across {len(races)} races")

    probs, labels = [], []
    skipped = 0
    for i, ((rd, rn), horse_rows) in enumerate(races.items(), 1):
        if i % 500 == 0:
            logger.info(f"  processed {i}/{len(races)} races, collected {len(probs)} pairs")
        try:
            preds = predict_race_production(horse_rows, conn)
        except Exception as e:
            skipped += 1
            continue
        if not preds:
            skipped += 1
            continue
        for h_row, cal_prob, _market_prob, _analysis in preds:
            if cal_prob is None or cal_prob <= 0:
                continue
            probs.append(float(cal_prob))
            labels.append(int(h_row.get('is_winner') or 0))

    conn.close()
    logger.info(f"Collected {len(probs)} (cal_prob, is_winner) pairs "
                f"({sum(labels)} winners, base rate {sum(labels)/max(len(labels),1):.3f}). "
                f"Skipped {skipped} races (no live odds or pipeline error)")
    return np.array(probs), np.array(labels)


def fit_and_save(probs, labels):
    """Fit IsotonicRegression and write models/blend_calibrator.pkl + .sha256."""
    from sklearn.isotonic import IsotonicRegression

    if len(probs) < 1000:
        raise RuntimeError(
            f"Refusing to fit blend calibrator on {len(probs)} samples "
            f"(need ≥1000 for stable isotonic). Run on a larger holdout."
        )

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(probs, labels)

    metadata = {
        'n_samples': int(len(probs)),
        'n_winners': int(labels.sum()),
        'base_rate': float(labels.mean()),
        'trained_date': datetime.now().isoformat(timespec='seconds'),
        'version': datetime.now().strftime('%Y%m%d-%H%M%S'),
        # Sanity check the fit at common bins to log calibration shift.
        'bin_shift': {
            '0.10': float(iso.predict(np.array([0.10]))[0]),
            '0.20': float(iso.predict(np.array([0.20]))[0]),
            '0.30': float(iso.predict(np.array([0.30]))[0]),
            '0.40': float(iso.predict(np.array([0.40]))[0]),
            '0.50': float(iso.predict(np.array([0.50]))[0]),
        },
    }
    payload = {'calibrator': iso, 'metadata': metadata}

    out = config.MODELS_DIR / 'blend_calibrator.pkl' if hasattr(config, 'MODELS_DIR') \
          else Path(__file__).resolve().parent.parent / 'models' / 'blend_calibrator.pkl'
    sha = out.with_suffix('.sha256')

    raw = pickle.dumps(payload)
    out.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    sha.write_text(digest + '\n')

    logger.info(f"Saved blend calibrator → {out}")
    logger.info(f"  metadata: {metadata}")
    logger.info(f"  sha256: {digest[:16]}...")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--last', type=int, default=None,
                    help='Only use the last N races for fitting (default: full 20% holdout)')
    args = ap.parse_args()

    probs, labels = collect_calibration_pairs(last_n=args.last)
    fit_and_save(probs, labels)


if __name__ == '__main__':
    main()
