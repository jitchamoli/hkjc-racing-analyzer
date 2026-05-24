"""Quantile-matched gate tuning — the ML statistician's recommended fix
for the v6.11 post-prune tier-inflation.

The +0.15 constant bump in betting/conviction.py was tuned on n=11 (one
morning's card). This script computes the correct gates by:
  1. Walking the last N meetings that have race results.
  2. Per race, computing top_2_share + top_3_share under BOTH v6.10 (backup
     at models/hkjc_racing_model.pkl.pre-prune-tier1) AND v6.11 (current).
  3. Building empirical CDFs of those values under each model.
  4. Finding the percentile each v6.10 gate threshold occupies in v6.10's
     own CDF, then picking the v6.11 value at that same percentile.

The result is a gate set under v6.11 that produces the SAME tier-rate
distribution as v6.10's original gates would have produced. This is the
statistically-defensible adjustment (vs the +0.15 morning patch).

Usage:
    python3 -m scripts.quantile_match_gates [--meetings N]    # default 30
"""
from __future__ import annotations

import argparse
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

# Current v6.10-era gates we want to quantile-match in v6.11
CURRENT_GATES = {
    'high_top2': 0.40,
    'high_top3': 0.55,
    'med_top2':  0.30,
    'med_top3':  0.42,
    'low_top2':  0.22,
}


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _wp(pkl, feats):
    m, c = pkl['model'], pkl.get('calibrator')
    proba = m.predict_proba(feats.reshape(1, -1))[0]
    raw = float(proba[1]) if len(proba) > 1 else float(proba[0])
    if c is None:
        return raw
    lg = float(logit(np.clip(raw, 1e-6, 1 - 1e-6)))
    return float(c.predict(np.array([lg]))[0])


def _pp(pkl, feats):
    pm, pc = pkl.get('place_model'), pkl.get('place_calibrator')
    if pm is None:
        return 0.25
    proba = pm.predict_proba(feats.reshape(1, -1))[0]
    raw = float(proba[1]) if len(proba) > 1 else float(proba[0])
    if pc is None:
        return raw
    lg = float(logit(np.clip(raw, 1e-6, 1 - 1e-6)))
    return float(pc.predict(np.array([lg]))[0])


def _hd(row):
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


def _v11_to_v10(v11_feats, row, conn):
    draw_adv = FeatureEngineering.calculate_draw_advantage(
        row['draw'] or 8, row['distance'] or 1200)
    speed_fig = FeatureEngineering.calculate_speed_fig_norm(
        row['horse_name'], conn=conn, before_date=row['race_date'])
    _, _, run_style = FeatureEngineering.calculate_running_position_features(
        row['horse_name'], conn=conn, before_date=row['race_date'])
    out = np.empty(26, dtype=np.float32)
    out[0:2]   = v11_feats[0:2]
    out[2]     = draw_adv
    out[3:20]  = v11_feats[2:19]
    out[20]    = speed_fig
    out[21:23] = v11_feats[19:21]
    out[23]    = run_style
    out[24:26] = v11_feats[21:23]
    return out


def _race_top_n_shares(per_horse, key):
    """Returns (top_2_share, top_3_share) using geo-mean of win × place as
    the per-horse score, then normalized to sum=1 (mirrors production)."""
    scores = []
    for h in per_horse:
        w = _wp(h['pkl'], h[key])
        p = _pp(h['pkl'], h[key])
        scores.append((w * p) ** 0.5)
    total = sum(scores) or 1.0
    norm = sorted([s / total for s in scores], reverse=True)
    t2 = sum(norm[:2])
    t3 = sum(norm[:3])
    return t2, t3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meetings', type=int, default=30,
                   help='Number of recent meetings to sample (default 30)')
    args = p.parse_args()

    if not V6_10_PATH.exists():
        print(f"ERROR: {V6_10_PATH} missing")
        return 1
    v6_10 = _load(V6_10_PATH)
    v6_11 = _load(V6_11_PATH)
    print(f"v6.10 ({V6_10_PATH.name}) n_features={v6_10.get('n_features')}")
    print(f"v6.11 ({V6_11_PATH.name}) n_features={v6_11.get('n_features')}\n")

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # Use race_results directly (prediction_log has only ~6 meetings results-filled).
    # Sample the last N meetings with at least 5 finished races.
    meetings = [r[0] for r in conn.execute("""
        SELECT race_date FROM race_results
        WHERE finish_position > 0
        GROUP BY race_date HAVING COUNT(*) >= 30
        ORDER BY race_date DESC LIMIT ?
    """, (args.meetings,)).fetchall()]
    print(f"Scanning {len(meetings)} meetings: {meetings[-1]} .. {meetings[0]}\n")

    t2_v10, t3_v10, t2_v11, t3_v11 = [], [], [], []
    n_races = 0
    for date in meetings:
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
            per_h10, per_h11 = [], []
            ok = True
            for row in race_rows:
                try:
                    f11 = FeatureEngineering.extract_features(
                        _hd(row), all_results=[], conn=conn,
                        venue=row['venue'] or 'ST')
                    f10 = _v11_to_v10(f11, row, conn)
                except Exception:
                    ok = False
                    break
                per_h10.append({'pkl': v6_10, 'v10': f10})
                per_h11.append({'pkl': v6_11, 'v11': f11})
            if not ok:
                continue
            t2a, t3a = _race_top_n_shares(per_h10, 'v10')
            t2b, t3b = _race_top_n_shares(per_h11, 'v11')
            t2_v10.append(t2a); t3_v10.append(t3a)
            t2_v11.append(t2b); t3_v11.append(t3b)
            n_races += 1

    print(f"Sampled {n_races:,} races\n")
    if n_races < 100:
        print("INSUFFICIENT — need ≥100 races for stable percentiles")
        return 1

    t2_v10 = np.array(t2_v10); t3_v10 = np.array(t3_v10)
    t2_v11 = np.array(t2_v11); t3_v11 = np.array(t3_v11)

    print(f"top_2_share:  v6.10 mean={t2_v10.mean():.3f} (sd {t2_v10.std():.3f})   "
          f"v6.11 mean={t2_v11.mean():.3f} (sd {t2_v11.std():.3f})")
    print(f"top_3_share:  v6.10 mean={t3_v10.mean():.3f} (sd {t3_v10.std():.3f})   "
          f"v6.11 mean={t3_v11.mean():.3f} (sd {t3_v11.std():.3f})")
    print()

    print("Quantile-matched gates (the percentile each v6.10 gate occupies in")
    print("v6.10's CDF, then the value at the SAME percentile in v6.11's CDF):\n")
    print(f"  {'Gate':<14} {'v6.10 cur':>10} {'%ile':>6}   "
          f"{'matched':>10} {'+0.15 patch':>12}   {'Δ vs patch':>10}")
    print("  " + "-" * 78)

    suggested = {}
    for name, cur in CURRENT_GATES.items():
        src = t2_v10 if name.endswith('top2') else t3_v10
        dst = t2_v11 if name.endswith('top2') else t3_v11
        pct = float((src < cur).mean()) * 100        # % of v6.10 races BELOW this gate
        matched = float(np.percentile(dst, pct))
        patch = cur + 0.15
        delta = matched - patch
        suggested[name] = matched
        print(f"  {name:<14} {cur:>10.3f} {pct:>5.1f}%   "
              f"{matched:>10.3f} {patch:>12.3f}   {delta:>+10.3f}")

    print()
    print("Interpretation:")
    print("  - 'matched' is the statistically-correct gate under v6.11 to keep")
    print("    the same race-rate at each tier as v6.10.")
    print("  - '+0.15 patch' is the morning eyeball-tuned value currently shipped.")
    print("  - 'Δ vs patch' > 0  ⇒  patch is too loose (more HIGHs than v6.10 had)")
    print("    'Δ vs patch' < 0  ⇒  patch is too tight (fewer HIGHs than v6.10 had)")
    print()
    print("Proposed conviction.py update:")
    print("  high_top2 = {:.3f}   # was {:.2f} (cur), patch {:.2f}".format(
        suggested['high_top2'], CURRENT_GATES['high_top2'],
        CURRENT_GATES['high_top2'] + 0.15))
    print("  high_top3 = {:.3f}".format(suggested['high_top3']))
    print("  med_top2  = {:.3f}".format(suggested['med_top2']))
    print("  med_top3  = {:.3f}".format(suggested['med_top3']))
    print("  low_top2  = {:.3f}".format(suggested['low_top2']))

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
