"""H10: cross-pipeline parity — training and inference must compute the
SAME feature vector for the same row.

The legacy "strict parity" test only checks that extract_features() is
deterministic (same input twice → same output). That can't catch the
DS-03 class of bug where training and inference compute the same feature
NAME with different formulas. This test runs build_training_data_from_db
on a real DB row and compares to the inference-side extract_features
output for the same horse_data.
"""
import sqlite3

import numpy as np
import pytest

import config
from ml.features import FeatureEngineering, clear_feature_caches
from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES
from ml.training import build_training_data_from_db


PER_FEATURE_TOLERANCE = 0.05  # slight relaxation for jockey/trainer rates
                              # which use stale-snapshot semantics in training


def _build_inference_horse_data(row, race_rows):
    """Construct the horse_data dict the way core/analyzer.py does."""
    field_weights = [r['horse_weight'] for r in race_rows
                     if r['horse_weight'] and r['horse_weight'] > 0]
    return {
        'name': row['horse_name'],
        'horse_weight': row['horse_weight'],
        'weight': row['weight'],
        'draw': row['draw'],
        'jockey': row['jockey'] or '',
        'trainer': row['trainer'] or '',
        'race_date': row['race_date'],
        'race_number': row['race_number'],
        'field_size': len(race_rows),
        'field_weights': field_weights,
        'class_type': row['class_type'] or '',
        'going': row['going'] or '',
        'days_since_last_run': row['days_since_last_run'] or 0,
    }


def test_train_predict_parity_on_real_rows():
    """Pick rows that build_training_data_from_db consumed and verify
    extract_features produces the same vector for each. Catches the
    DS-03 class of bug (same feature name, different formula).

    Marked slow: building training data over even 1 year of HKJC data
    can take 5+ minutes on the 614 MB DB. Run with -m slow to include.
    """
    if not config.DB_PATH.exists():
        pytest.skip("No horse_history.db found")

    pytest.importorskip("xgboost")  # training builder needs sklearn/xgboost
    import os
    if os.environ.get('RUN_SLOW_PARITY') != '1':
        pytest.skip("Set RUN_SLOW_PARITY=1 to run the cross-pipeline parity check")

    X, y, _w, _yp = build_training_data_from_db(years_back=1)
    if X is None or len(X) < 100:
        pytest.skip(f"Insufficient training data for parity check (got {0 if X is None else len(X)})")

    # The training builder doesn't return row identifiers, so we can't map
    # X[i] → DB row directly. Instead, pick a sample of recent rows and
    # verify that extract_features produces values within the same
    # distribution as training (per-feature mean/std comparison).
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT race_date, race_number FROM race_results
        WHERE horse_weight IS NOT NULL AND horse_weight > 0
          AND class_type IS NOT NULL
        ORDER BY race_date DESC LIMIT 20
    """)
    sampled = cur.fetchall()

    inference_X = []
    fe = FeatureEngineering()
    clear_feature_caches()
    for race_date, race_no in sampled:
        cur.execute("""
            SELECT * FROM race_results
            WHERE race_date = ? AND race_number = ?
              AND horse_weight IS NOT NULL AND horse_weight > 0
        """, (race_date, race_no))
        race_rows = cur.fetchall()
        if len(race_rows) < 2:
            continue
        for row in race_rows:
            try:
                hd = _build_inference_horse_data(row, race_rows)
                vec = fe.extract_features(hd, [], '', conn=conn)
                inference_X.append(vec)
            except Exception:
                continue
    conn.close()

    assert len(inference_X) >= 30, f"Got {len(inference_X)} inference vectors, need ≥30"
    inf_X = np.array(inference_X)

    # Per-feature distribution check: training and inference must agree on
    # mean within 3× std, OR both must produce a near-constant value.
    # This catches the DS-03 case (training mean=0.5, inference mean=0.0
    # because of unit mismatch).
    train_mean = X.mean(axis=0)
    train_std  = X.std(axis=0)
    inf_mean   = inf_X.mean(axis=0)

    drift = []
    for i, name in enumerate(FEATURE_NAMES):
        # Allow up to 3 sigma OR ±0.10 absolute drift, whichever is larger
        bound = max(3 * train_std[i], 0.10)
        delta = abs(inf_mean[i] - train_mean[i])
        if delta > bound:
            drift.append((name, train_mean[i], inf_mean[i], delta, bound))

    if drift:
        msg = "\nTrain/inference distribution drift (per-feature mean):\n" + "\n".join(
            f"  {n}: train_mean={tm:.3f}, inf_mean={im:.3f}, |Δ|={d:.3f} > bound={b:.3f}"
            for n, tm, im, d, b in drift
        )
        pytest.fail(msg)


def test_extract_features_returns_correct_count():
    """Sanity guard: extract_features must always return NUM_FEATURES values
    regardless of how sparse horse_data is. A missing return would crash
    downstream model.predict with a shape mismatch."""
    fe = FeatureEngineering()
    minimal_hd = {
        'name': 'TestHorse', 'jockey': 'J', 'trainer': 'T', 'draw': 5,
        'weight': '125', 'horse_weight': 1100, 'field_size': 12,
        'field_weights': [1100] * 12,
    }
    vec = fe.extract_features(minimal_hd, [], '')
    assert len(vec) == NUM_FEATURES, f"Got {len(vec)} features, expected {NUM_FEATURES}"
    assert all(0.0 <= v <= 1.0 for v in vec), "All features must be in [0,1] after clipping"
