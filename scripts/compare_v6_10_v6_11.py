"""Direct trained-model A/B: v6.10 (26 features) vs v6.11 (23 features,
pruned draw_advantage / speed_fig_norm / running_style) on the burned
last 6 meetings.

Conversion trick: current extract_features returns 23-feature v6.11 vec.
v6.10 wants 26 in a specific order, so we compute the 3 dropped features
on-the-fly (helpers still exist, deliberately kept as orphans) and inject
them at v6.10's indices 2 (draw_advantage), 20 (speed_fig_norm), 23
(running_style).

Same caveats as compare_v6_9_v6_10.py:
  - Both models trained on/near these meetings — BURNED.
  - Forward log is the only honest measure.
  - 60 races small; apply 3-5× discount per
    feedback_small_sample_inflation_pattern.
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


V6_11_PATH = config.MODEL_PATH
V6_10_PATH = config.MODEL_PATH.with_suffix('.pkl.pre-prune-tier1')


def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _model_win_prob(pkl, feats):
    model, cal = pkl['model'], pkl.get('calibrator')
    proba = model.predict_proba(feats.reshape(1, -1))[0]
    raw = float(proba[1]) if len(proba) > 1 else float(proba[0])
    if cal is None:
        return raw
    lg = float(logit(np.clip(raw, 1e-6, 1 - 1e-6)))
    return float(cal.predict(np.array([lg]))[0])


def _model_place_prob(pkl, feats):
    pm, pc = pkl.get('place_model'), pkl.get('place_calibrator')
    if pm is None:
        return 0.25
    proba = pm.predict_proba(feats.reshape(1, -1))[0]
    raw = float(proba[1]) if len(proba) > 1 else float(proba[0])
    if pc is None:
        return raw
    lg = float(logit(np.clip(raw, 1e-6, 1 - 1e-6)))
    return float(pc.predict(np.array([lg]))[0])


def _build_horse_data(row):
    return {
        'name': row['horse_name'], 'horse_name': row['horse_name'],
        'race_date': row['race_date'], 'race_number': row['race_number'],
        'distance': row['distance'] or 1200, 'going': row['going'] or '',
        'field_size': row['field_size'] or 12, 'draw': row['draw'] or 8,
        'jockey': row['jockey'] or '', 'trainer': row['trainer'] or '',
        'class_type': row['class_type'] or '',
        'days_since_last_run': row['days_since_last_run'] or 0,
        'weight': row['weight'] or '', 'horse_weight': row['horse_weight'] or 0,
        'age': row['age'] or 0, 'rating_change': row['rating_change'] or 0,
        'field_weights': [],
    }


def _v11_to_v10(v11_feats: np.ndarray, row, conn) -> np.ndarray:
    """Inject 3 pruned features at v6.10's positions.
    v6.10 indices: 2 = draw_advantage, 20 = speed_fig_norm, 23 = running_style.
    """
    draw_adv = FeatureEngineering.calculate_draw_advantage(
        row['draw'] or 8, row['distance'] or 1200,
    )
    speed_fig = FeatureEngineering.calculate_speed_fig_norm(
        row['horse_name'], conn=conn, before_date=row['race_date'],
    )
    _, _, run_style = FeatureEngineering.calculate_running_position_features(
        row['horse_name'], conn=conn, before_date=row['race_date'],
    )
    # v6.11 ordering: prior, recent, jockey_wr, trainer_wr, ...
    # v6.10 ordering: prior, recent, DRAW_ADV, jockey_wr, trainer_wr, ...
    # So insert at index 2.
    out = np.empty(26, dtype=np.float32)
    out[0:2]   = v11_feats[0:2]
    out[2]     = draw_adv
    out[3:20]  = v11_feats[2:19]      # jockey_wr ... jt_pair_win_rate
    out[20]    = speed_fig
    out[21:23] = v11_feats[19:21]     # early_speed, late_kick
    out[23]    = run_style
    out[24:26] = v11_feats[21:23]     # mid_race_pace, sire_top3
    return out


def main() -> int:
    if not V6_10_PATH.exists():
        print(f"ERROR: {V6_10_PATH} missing — can't run A/B")
        return 1
    v6_11 = _load_pickle(V6_11_PATH)
    v6_10 = _load_pickle(V6_10_PATH)
    print(f"v6.11 model: {V6_11_PATH}  (n_features={v6_11.get('n_features')})")
    print(f"v6.10 model: {V6_10_PATH}  (n_features={v6_10.get('n_features')})\n")
    assert v6_11.get('n_features') == 23, "v6.11 should be 23"
    assert v6_10.get('n_features') == 26, "v6.10 should be 26"

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    meetings = [r[0] for r in conn.execute("""
        SELECT DISTINCT race_date FROM prediction_log
        WHERE results_filled=1 ORDER BY race_date DESC LIMIT 6
    """).fetchall()]
    print(f"Testing on {len(meetings)} meetings: {meetings[-1]} .. {meetings[0]}\n")

    win, qin, t4, t5 = ({'v6.10': 0, 'v6.11': 0} for _ in range(4))
    n_races = 0
    per_meeting = {}        # date → {'n': X, 'v6.10': {...}, 'v6.11': {...}}

    for date in meetings:
        per_meeting[date] = {
            'n': 0,
            'v6.10': {'WIN': 0, 'QIN': 0, 'TOP-4': 0, 'TOP-5': 0},
            'v6.11': {'WIN': 0, 'QIN': 0, 'TOP-4': 0, 'TOP-5': 0},
        }
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
            actual = {row['horse_name']: row['finish_position'] for row in race_rows}
            actual_top3 = {h for h, p in actual.items() if p in (1, 2, 3)}
            if len(actual_top3) != 3:
                continue
            winner = next((h for h, p in actual.items() if p == 1), None)
            top2_actual = {h for h, p in actual.items() if p in (1, 2)}

            per_horse = []
            ok = True
            for row in race_rows:
                hd = _build_horse_data(row)
                try:
                    feats_v11 = FeatureEngineering.extract_features(
                        hd, all_results=[], conn=conn, venue=row['venue'] or 'ST',
                    )
                    feats_v10 = _v11_to_v10(feats_v11, row, conn)
                except Exception as e:
                    ok = False
                    break
                per_horse.append({
                    'name': row['horse_name'],
                    'v11': feats_v11, 'v10': feats_v10,
                })
            if not ok:
                continue

            for tag, pkl, key in (('v6.10', v6_10, 'v10'), ('v6.11', v6_11, 'v11')):
                scored = []
                for h in per_horse:
                    f = h[key]
                    w = _model_win_prob(pkl, f)
                    p = _model_place_prob(pkl, f)
                    s = (w * p) ** 0.5
                    scored.append((s, h['name']))
                scored.sort(key=lambda x: -x[0])
                top_names = [n for _, n in scored]
                if top_names[0] == winner:
                    win[tag] += 1; per_meeting[date][tag]['WIN'] += 1
                if set(top_names[:2]) >= top2_actual:
                    qin[tag] += 1; per_meeting[date][tag]['QIN'] += 1
                if actual_top3 <= set(top_names[:4]):
                    t4[tag] += 1; per_meeting[date][tag]['TOP-4'] += 1
                if actual_top3 <= set(top_names[:5]):
                    t5[tag] += 1; per_meeting[date][tag]['TOP-5'] += 1
            per_meeting[date]['n'] += 1
            n_races += 1

    n = n_races
    print("=" * 76)
    print(f"DIRECT TRAINED-MODEL A/B  —  {n} races, ML head only (geo-mean win×place)")
    print("=" * 76)
    print(f"{'Model':<8} {'WIN':>14} {'QIN':>14} {'TOP-4':>14} {'TOP-5':>14}")
    print("-" * 76)
    for tag in ('v6.10', 'v6.11'):
        w_pc = 100 * win[tag] / n if n else 0
        q_pc = 100 * qin[tag] / n if n else 0
        t4_pc = 100 * t4[tag] / n if n else 0
        t5_pc = 100 * t5[tag] / n if n else 0
        print(f"{tag:<8} {win[tag]:>4}/{n:<3} {w_pc:>5.1f}% "
              f"{qin[tag]:>4}/{n:<3} {q_pc:>5.1f}% "
              f"{t4[tag]:>4}/{n:<3} {t4_pc:>5.1f}% "
              f"{t5[tag]:>4}/{n:<3} {t5_pc:>5.1f}%")
    print("-" * 76)
    d_w = win['v6.11'] - win['v6.10']
    d_q = qin['v6.11'] - qin['v6.10']
    d_4 = t4['v6.11'] - t4['v6.10']
    d_5 = t5['v6.11'] - t5['v6.10']
    print(f"Δ (v6.11 − v6.10):  WIN {d_w:+d}   QIN {d_q:+d}   "
          f"TOP-4 {d_4:+d}   TOP-5 {d_5:+d}\n")

    # Per-meeting breakdown — shows whether deltas are concentrated or spread.
    print("=" * 76)
    print("PER-MEETING BREAKDOWN  (v6.11 − v6.10 hit count per metric)")
    print("=" * 76)
    print(f"{'Date':<14}{'N':>4}    {'WIN':>9}{'QIN':>9}{'TOP-4':>9}{'TOP-5':>9}")
    print("-" * 76)
    # Print oldest → newest for readability
    for date in sorted(per_meeting.keys()):
        pm = per_meeting[date]
        nn = pm['n']
        if nn == 0:
            continue
        dw = pm['v6.11']['WIN']   - pm['v6.10']['WIN']
        dq = pm['v6.11']['QIN']   - pm['v6.10']['QIN']
        d4 = pm['v6.11']['TOP-4'] - pm['v6.10']['TOP-4']
        d5 = pm['v6.11']['TOP-5'] - pm['v6.10']['TOP-5']
        def fmt(delta):
            if delta == 0: return '   ·    '
            return f"{delta:+d}".rjust(8)
        print(f"{date:<14}{nn:>4}    {fmt(dw):>9}{fmt(dq):>9}{fmt(d4):>9}{fmt(d5):>9}")
    print("-" * 76)
    print(f"{'TOTAL':<14}{n:>4}    {d_w:+9d}{d_q:+9d}{d_4:+9d}{d_5:+9d}\n")

    print("HONEST READ: pruning is expected to leave burned-data metrics ≈ flat")
    print("per memory feature-experiments-brier-vs-roi-pattern. ANY non-tiny")
    print("regression (say, ≥3 hits) on TOP-4 or TOP-5 is a real warning;")
    print("expect WIN/QIN to be within noise (3-5× discount per small-sample).")
    print("If TOP-N holds, the prune is safe to ship and the value is schema")
    print("parsimony + retrain speed, not lift.")
    print("Both models trained on/near these meetings — forward log via")
    print("scripts/validate_hk.py is the only honest verdict.")
    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
