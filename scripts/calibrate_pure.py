#!/usr/bin/env python3
"""Fit an isotonic calibrator for pure_prob — the model's PRE-MARKET opinion.

Why this exists
---------------
`pure_prob` is the quantity ALL the edge math rides on:

    edge_pp = (pure_prob - market_prob) * 100
    ev_win  = pure_prob * win_odds - 1
    overlay = pure_prob * win_odds

and it feeds the six EDGE gates plus the exotic value-detection. But measured
over the last 15 meetings it was badly under-confident:

    rank-1  pure_prob 13.3%   market 23.3%   ACTUAL win rate 26.5%

i.e. off by roughly a factor of two, while the market (23.3%) and
calibrated_prob (26.3%) were both close to truth. Because edge_pp subtracts a
well-scaled market number from a mis-scaled model number, every top pick
carried a STRUCTURAL ~-10pp before any genuine disagreement — so the EDGE gate
(G3 needs >= +5pp) could essentially only fire on noise. Over 15 meetings only
3/147 races (2.0%) flagged EDGE at all.

Calibrating pure_prob makes edge_pp mean what it claims: the model's honest
probability minus the market's. It does NOT touch ranking — calibrated_prob
and the recommender are unaffected, because pure_prob is not in that path.

Discipline (mirrors scripts/calibrate_blend.py)
----------------------------------------------
* Fit on the 70-85% CALIB fold only — disjoint from the model's training folds
  AND from backtest's 85-100% test fold.
* Disable any cached pure calibrator during collection, so the isotonic is fit
  on the SAME (uncalibrated) domain it will later be applied to. Fitting a new
  isotonic on already-corrected output compounds the lift on every re-fit.
* Signed with a .sha256 sidecar; core/… refuses to load it unsigned.

Usage:
    python3 -m scripts.calibrate_pure
    python3 -m scripts.calibrate_pure --last 4000   # smaller sample to iterate
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parent.parent / 'models' / 'pure_calibrator.pkl'


def collect_pairs(last_n=None):
    """Collect (pure_prob, is_winner) over the calib fold."""
    from scripts.backtest import (
        load_calibration_window, group_by_race, predict_race_production,
    )
    from betting.edge_calculator import pure_probs_from_analyses
    import betting.edge_calculator as _ec

    # Fit on the domain we apply to — see module docstring.
    _ec._PURE_CALIBRATOR = None
    _ec._PURE_CALIBRATOR_LOADED = True   # short-circuit the loader to None

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    from data.database import apply_perf_pragmas
    apply_perf_pragmas(conn)

    rows = load_calibration_window(conn)
    if last_n is not None and last_n < len(rows):
        rows = rows[-last_n:]
    races = group_by_race(rows)
    logger.info(f"Calib window (70-85%): {len(rows)} horses across {len(races)} races")

    probs, labels = [], []
    skipped = 0
    for (rd, rn), horse_rows in races.items():
        preds = predict_race_production(horse_rows, conn)
        if not preds:
            skipped += 1
            continue
        analyses = [an for _row, _cal, _mkt, an in preds]
        odds = [(row.get('win_odds') or 0) for row, _c, _m, _a in preds]
        won = [1 if (row.get('finish_position') or 0) == 1 else 0
               for row, _c, _m, _a in preds]
        if len(analyses) < 2:
            skipped += 1
            continue
        pp = pure_probs_from_analyses(analyses, odds=odds)
        probs.extend(pp)
        labels.extend(won)

    conn.close()
    logger.info(f"Collected {len(probs)} (pure_prob, is_winner) pairs "
                f"({sum(labels)} winners, base rate {np.mean(labels):.3f}). "
                f"Skipped {skipped} races.")
    return np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)


def fit_and_save(probs, labels):
    from sklearn.isotonic import IsotonicRegression

    if len(probs) < 1000:
        raise RuntimeError(
            f"Refusing to fit pure calibrator on {len(probs)} samples "
            f"(need >=1000 for a stable isotonic)."
        )

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(probs, labels)

    metadata = {
        'n_samples': int(len(probs)),
        'n_winners': int(labels.sum()),
        'base_rate': float(labels.mean()),
        'trained_date': datetime.now().isoformat(timespec='seconds'),
        'version': datetime.now().strftime('%Y%m%d-%H%M%S'),
        'bin_shift': {f'{b:.2f}': float(iso.predict(np.array([b]))[0])
                      for b in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)},
    }
    payload = {'calibrator': iso, 'metadata': metadata}

    raw = pickle.dumps(payload)
    OUT_PATH.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    OUT_PATH.with_suffix('.sha256').write_text(digest + '\n')

    logger.info(f"Saved pure calibrator -> {OUT_PATH}")
    logger.info(f"  metadata: {metadata}")
    logger.info(f"  sha256: {digest[:16]}...")
    return OUT_PATH


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--last', type=int, default=None,
                    help='use only the last N calib-fold horses (faster iteration)')
    args = ap.parse_args()

    probs, labels = collect_pairs(args.last)
    fit_and_save(probs, labels)
    return 0


if __name__ == '__main__':
    sys.exit(main())
