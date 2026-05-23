"""Direct trained-model A/B: v6.9 (25 features) vs v6.10 (26 features,
adds sire_top3_rate_on_going) on the last 6 burned meetings.

How it differs from test_sire_going_rerank.py:
  - That test took the production cal_prob (already v6.9-trained) and ADDED
    a linear blend of the candidate feature. Pure signal-additivity probe.
  - This script RE-RUNS BOTH MODELS from scratch on the same horse-data
    inputs. v6.10's predictions include the XGB interactions a linear blend
    can't find. Closer to "what will today actually look like".

Caveats:
  - Still burned data (both models trained on or near these races). Forward
    log is the only honest measure.
  - We only compare the ML head's softmax-ranking — no rule engine, no
    market blend, no recommender. So this is a "pure model lift" view, not
    a "what bets fire differently" view.
  - 60 races is small; apply small-sample discount (memory:
    feedback_small_sample_inflation_pattern).
"""
from __future__ import annotations

import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.special import logit  # noqa: E402

import config  # noqa: E402
from ml.features import FeatureEngineering  # noqa: E402


V6_10_PATH = config.MODEL_PATH
V6_9_PATH = config.MODEL_PATH.with_suffix('.pkl.pre-sire')


def _load_pickle(path):
    """Bypass the schema integrity check — we WANT both versions loaded."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def _model_win_prob(pkl: dict, features: np.ndarray) -> float:
    """Mirror HKJCRacingMLModel.predict but without integrity gating."""
    model = pkl['model']
    cal = pkl.get('calibrator')
    proba = model.predict_proba(features.reshape(1, -1))[0]
    raw = float(proba[1]) if len(proba) > 1 else float(proba[0])
    if cal is None:
        return raw
    lg = float(logit(np.clip(raw, 1e-6, 1 - 1e-6)))
    return float(cal.predict(np.array([lg]))[0])


def _model_place_prob(pkl: dict, features: np.ndarray) -> float:
    pm = pkl.get('place_model')
    pc = pkl.get('place_calibrator')
    if pm is None:
        return 0.25
    proba = pm.predict_proba(features.reshape(1, -1))[0]
    raw = float(proba[1]) if len(proba) > 1 else float(proba[0])
    if pc is None:
        return raw
    lg = float(logit(np.clip(raw, 1e-6, 1 - 1e-6)))
    return float(pc.predict(np.array([lg]))[0])


def _build_horse_data(row, conn) -> dict:
    """Construct the dict extract_features expects from a race_results row."""
    return {
        'name': row['horse_name'],
        'horse_name': row['horse_name'],
        'race_date': row['race_date'],
        'race_number': row['race_number'],
        'distance': row['distance'] or 1200,
        'going': row['going'] or '',
        'field_size': row['field_size'] or 12,
        'draw': row['draw'] or 8,
        'jockey': row['jockey'] or '',
        'trainer': row['trainer'] or '',
        'class_type': row['class_type'] or '',
        'days_since_last_run': row['days_since_last_run'] or 0,
        'weight': row['weight'] or '',
        'horse_weight': row['horse_weight'] or 0,
        'age': row['age'] or 0,
        'rating_change': row['rating_change'] or 0,
        'field_weights': [],
    }


def main() -> int:
    if not V6_9_PATH.exists():
        print(f"ERROR: {V6_9_PATH} missing — can't run A/B")
        return 1
    print(f"v6.10 model: {V6_10_PATH}")
    print(f"v6.9 model:  {V6_9_PATH}\n")
    v6_10 = _load_pickle(V6_10_PATH)
    v6_9 = _load_pickle(V6_9_PATH)
    print(f"v6.10 n_features = {v6_10.get('n_features')}, "
          f"version = {v6_10.get('version')}")
    print(f"v6.9  n_features = {v6_9.get('n_features')}, "
          f"version = {v6_9.get('version')}\n")
    assert v6_10.get('n_features') == 26, "v6.10 should be 26 features"
    assert v6_9.get('n_features') == 25, "v6.9 should be 25 features"

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # Last 6 meetings with results in prediction_log (same set as
    # test_sire_going_rerank for direct comparability).
    meetings = [r[0] for r in conn.execute("""
        SELECT DISTINCT race_date FROM prediction_log
        WHERE results_filled=1 ORDER BY race_date DESC LIMIT 6
    """).fetchall()]
    print(f"Testing on {len(meetings)} meetings: {meetings[-1]} .. {meetings[0]}\n")

    win, qin, t4, t5 = {'v6.9': 0, 'v6.10': 0}, \
                       {'v6.9': 0, 'v6.10': 0}, \
                       {'v6.9': 0, 'v6.10': 0}, \
                       {'v6.9': 0, 'v6.10': 0}
    n_races = 0

    for date in meetings:
        # All horses for this meeting from race_results — joined to
        # field_size on the fly so extract_features has everything.
        rows = conn.execute("""
            SELECT rr.*,
              (SELECT COUNT(*) FROM race_results rr2
               WHERE rr2.race_date=rr.race_date AND rr2.race_number=rr.race_number
                 AND rr2.finish_position > 0) AS field_size
            FROM race_results rr
            WHERE rr.race_date=? AND rr.finish_position > 0
            ORDER BY rr.race_number, rr.finish_position
        """, (date,)).fetchall()
        by_race = defaultdict(list)
        for r in rows:
            by_race[r['race_number']].append(r)

        for rno, race_rows in by_race.items():
            if len(race_rows) < 5:
                continue
            actual = {row['horse_name']: row['finish_position']
                      for row in race_rows}
            actual_top3 = {h for h, p in actual.items() if p in (1, 2, 3)}
            if len(actual_top3) != 3:
                continue
            winner = next((h for h, p in actual.items() if p == 1), None)
            top2_actual = {h for h, p in actual.items() if p in (1, 2)}

            # Compute features ONCE per horse (26 wide); slice [:25] for v6.9.
            per_horse = []
            for row in race_rows:
                hd = _build_horse_data(row, conn)
                try:
                    feats = FeatureEngineering.extract_features(
                        hd, all_results=[], conn=conn, venue=row['venue'] or 'ST',
                    )
                except Exception as e:
                    feats = None
                per_horse.append({'name': row['horse_name'], 'feats': feats})

            if any(p['feats'] is None for p in per_horse):
                continue  # skip races with extraction failures

            # Predictions per model. Use win × 1 (could weight by place but
            # keeping single-channel for cleanest A/B).
            for tag, pkl, n_feats in (('v6.9', v6_9, 25), ('v6.10', v6_10, 26)):
                scored = []
                for h in per_horse:
                    f = h['feats'][:n_feats]
                    w = _model_win_prob(pkl, f)
                    p = _model_place_prob(pkl, f)
                    # Use the geometric average of win and place as the
                    # ranking score — that's what the production combined
                    # score effectively uses for top-N ordering.
                    s = (w * p) ** 0.5
                    scored.append((s, h['name']))
                scored.sort(key=lambda x: -x[0])
                top_names = [n for _, n in scored]

                if top_names[0] == winner:
                    win[tag] += 1
                if set(top_names[:2]) >= top2_actual:
                    qin[tag] += 1
                if actual_top3 <= set(top_names[:4]):
                    t4[tag] += 1
                if actual_top3 <= set(top_names[:5]):
                    t5[tag] += 1
            n_races += 1

    n = n_races
    print("=" * 76)
    print(f"DIRECT TRAINED-MODEL A/B  —  {n} races, ML head only "
          f"(geo-mean of win × place softmax)")
    print("=" * 76)
    print(f"{'Model':<8} {'WIN':>14} {'QIN':>14} {'TOP-4':>14} {'TOP-5':>14}")
    print("-" * 76)
    for tag in ('v6.9', 'v6.10'):
        w_pc = 100 * win[tag] / n if n else 0
        q_pc = 100 * qin[tag] / n if n else 0
        t4_pc = 100 * t4[tag] / n if n else 0
        t5_pc = 100 * t5[tag] / n if n else 0
        print(f"{tag:<8} {win[tag]:>4}/{n:<3} {w_pc:>5.1f}% "
              f"{qin[tag]:>4}/{n:<3} {q_pc:>5.1f}% "
              f"{t4[tag]:>4}/{n:<3} {t4_pc:>5.1f}% "
              f"{t5[tag]:>4}/{n:<3} {t5_pc:>5.1f}%")
    print("-" * 76)
    d_w = win['v6.10'] - win['v6.9']
    d_q = qin['v6.10'] - qin['v6.9']
    d_4 = t4['v6.10'] - t4['v6.9']
    d_5 = t5['v6.10'] - t5['v6.9']
    print(f"Δ (v6.10 − v6.9):  WIN {d_w:+d}   QIN {d_q:+d}   "
          f"TOP-4 {d_4:+d}   TOP-5 {d_5:+d}\n")

    print("HONEST READ: this compares the ML head only. The full production")
    print("pipeline (rule engine 60%, ML 40%, market blend, post-blend cal)")
    print("dampens model differences. So a +N here typically becomes ≈ +N/2")
    print("in the actual recommender output. And 60 races is small — apply")
    print("the small-sample 3-5× discount before believing the magnitude.")
    print()
    print("Both models trained on (or near) these meetings — BURNED. The")
    print("ONLY honest measure is the forward log via scripts/validate_hk.py.")
    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
