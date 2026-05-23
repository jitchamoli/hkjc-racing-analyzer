"""Cheap-diagnostic for the jockey's #1 recommendation (PR #52 review):
"mid-race 800m→400m sectional reveals trip-trouble / unspent finishers
that the current `late_kick` (positions-based) misses."

Per the project pattern (memory: project_sectional_closing_figure_killed),
test the candidate signal at the cheapest possible level BEFORE retraining.
Two kill-gates and one keep-gate:

  Gate A (redundancy with existing top feature):
    |r(mid_split_norm, late_kick)|  > 0.7  → REDUNDANT, KILL
  Gate B (signal vs outcome):
    |r(mid_split_norm, is_top3)|    < 0.02 → NO SIGNAL, KILL
  Gate C (unspent-finisher composite vs outcome):
    |r(mid_minus_late, is_top3)|    > 0.03 → KEEP for retrain test

Candidate feature definitions (computed per horse per HISTORICAL race that
has 3+ sectionals — 2nd-from-last = 800-400 segment regardless of distance):

  splits         = floats parsed from sectional_times string
  race_mean      = mean of splits (a per-race speed baseline)
  mid_split_norm = splits[-2] / race_mean    (>1 = slow in mid-race)
  last_split_norm= splits[-1] / race_mean    (>1 = slow in last-400)
  mid_minus_late = mid_split_norm - last_split_norm  ("unspent" proxy:
                   positive = held-up then sprinted; negative = used early)

Aggregation: mean across the horse's last 3 prior races with 3+ sectionals.

Output: correlation matrix + per-gate verdict + recommendation.

NOTE: This diagnostic operates on the DB only (no model retrain, no
training-pipeline change). It answers "is the signal real" in <1 minute.
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


def parse_sectionals(s: str) -> list[float]:
    if not s:
        return []
    try:
        return [float(x) for x in s.split()]
    except (ValueError, AttributeError):
        return []


def compute_mid_race_features(splits: list[float]) -> dict | None:
    """Returns dict with mid_split_norm, last_split_norm, mid_minus_late.
    Needs ≥3 splits (so 800-400 segment exists distinct from last-400)."""
    if len(splits) < 3:
        return None
    race_mean = sum(splits) / len(splits)
    if race_mean <= 0:
        return None
    mid = splits[-2] / race_mean
    last = splits[-1] / race_mean
    return {
        'mid_split_norm': mid,
        'last_split_norm': last,
        'mid_minus_late': mid - last,
    }


def existing_late_kick(running_position: str, field_size: int) -> float:
    """Mirrors ml/features.py:compute_running_position_features `late_kick`
    so we can compute the redundancy correlation without invoking the full
    feature pipeline."""
    if not running_position or field_size <= 1:
        return 0.5
    s = running_position.strip()
    if ' ' in s:
        try:
            positions = [int(x) for x in s.split()]
        except ValueError:
            return 0.5
    else:
        # Concatenated form "665" — single-digit per call.
        if not s.isdigit():
            return 0.5
        positions = [int(c) for c in s]
    if len(positions) < 2:
        return 0.5
    max_pos = float(max(field_size, max(positions)))
    delta = positions[-2] - positions[-1]
    late = 0.5 + delta / (2.0 * max_pos)
    return max(0.0, min(1.0, late))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # Step 1: build a per-horse history of mid-race feature observations.
    # Aggregation is the mean across the LAST 3 prior races for that horse
    # — same pattern as other recent-form features.
    print("Step 1: building per-horse mid-race history from race_results...")
    # Derive field_size per (race_date, race_number) since there's no
    # dedicated column. Count starters with a recorded finish_position.
    rows = conn.execute("""
        SELECT r.race_date, r.race_number, r.horse_name, r.sectional_times,
               r.running_position, r.is_top3, r.finish_position,
               (SELECT COUNT(*) FROM race_results r2
                WHERE r2.race_date=r.race_date AND r2.race_number=r.race_number
                  AND r2.finish_position > 0) AS field_size
        FROM race_results r
        WHERE r.sectional_times != '' AND r.sectional_times IS NOT NULL
          AND r.finish_position > 0
        ORDER BY r.horse_name, r.race_date
    """).fetchall()

    # Group by horse, store features per race in date order.
    per_horse_history = defaultdict(list)
    for r in rows:
        splits = parse_sectionals(r['sectional_times'])
        feats = compute_mid_race_features(splits)
        if feats is None:
            continue
        per_horse_history[r['horse_name']].append({
            'race_date': r['race_date'],
            **feats,
        })

    n_rows_with_sectionals = sum(len(v) for v in per_horse_history.values())
    print(f"  {n_rows_with_sectionals:,} horse-race observations with 3+ sectionals")
    print(f"  {len(per_horse_history):,} unique horses")
    print()

    # Step 2: walk all race outcomes; for each (horse, race), pull recent-form
    # mid-race aggregates from the 3 PRIOR races, plus the existing late_kick
    # computed from THIS race's running_position, plus the is_top3 outcome.
    print("Step 2: joining recent mid-race aggregates to outcomes...")
    outcomes = conn.execute("""
        SELECT r.race_date, r.race_number, r.horse_name, r.running_position,
               r.is_top3, r.finish_position,
               (SELECT COUNT(*) FROM race_results r2
                WHERE r2.race_date=r.race_date AND r2.race_number=r.race_number
                  AND r2.finish_position > 0) AS field_size
        FROM race_results r
        WHERE r.finish_position > 0 AND r.running_position != ''
        ORDER BY r.race_date
    """).fetchall()

    # Aligned arrays for correlations.
    mid_norm_xs, last_norm_xs, mid_minus_late_xs = [], [], []
    late_kick_xs, is_top3_ys = [], []

    for r in outcomes:
        if r['field_size'] is None or r['field_size'] <= 1:
            continue
        history = per_horse_history.get(r['horse_name'], [])
        prior = [h for h in history if h['race_date'] < r['race_date']][-3:]
        if len(prior) < 2:
            continue  # need at least 2 prior runs with sectionals for stable mean
        mid_mean = sum(h['mid_split_norm'] for h in prior) / len(prior)
        last_mean = sum(h['last_split_norm'] for h in prior) / len(prior)
        mml_mean = sum(h['mid_minus_late'] for h in prior) / len(prior)
        late = existing_late_kick(r['running_position'], r['field_size'])

        mid_norm_xs.append(mid_mean)
        last_norm_xs.append(last_mean)
        mid_minus_late_xs.append(mml_mean)
        late_kick_xs.append(late)
        is_top3_ys.append(float(r['is_top3'] or 0))

    n = len(mid_norm_xs)
    print(f"  {n:,} aligned outcome rows with ≥2 prior sectional runs")
    print()

    if n < 100:
        print("INSUFFICIENT DATA — under 100 aligned rows. Cannot run diagnostic.")
        return 1

    # Step 3: compute correlations.
    print("=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)
    print(f"  n = {n:,} aligned rows")
    print()
    r_redundancy = pearson(mid_norm_xs, late_kick_xs)
    r_signal_mid = pearson(mid_norm_xs, is_top3_ys)
    r_signal_mml = pearson(mid_minus_late_xs, is_top3_ys)
    r_signal_last = pearson(last_norm_xs, is_top3_ys)
    r_mid_vs_last = pearson(mid_norm_xs, last_norm_xs)
    r_mml_vs_late = pearson(mid_minus_late_xs, late_kick_xs)

    print(f"  Gate A (redundancy with existing late_kick):")
    print(f"     r(mid_split_norm,  late_kick)     = {r_redundancy:+.4f}")
    print(f"     r(mid_minus_late,  late_kick)     = {r_mml_vs_late:+.4f}")
    print()
    print(f"  Gate B (signal vs is_top3):")
    print(f"     r(mid_split_norm,  is_top3)       = {r_signal_mid:+.4f}")
    print(f"     r(last_split_norm, is_top3)       = {r_signal_last:+.4f}")
    print(f"     r(mid_minus_late,  is_top3)       = {r_signal_mml:+.4f}")
    print()
    print(f"  Internal:")
    print(f"     r(mid_split_norm,  last_split_norm)= {r_mid_vs_last:+.4f}")
    print()

    # Step 4: verdict.
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    verdicts = []

    if abs(r_redundancy) > 0.7 and abs(r_mml_vs_late) > 0.7:
        verdicts.append("KILL — Gate A: mid-race signal is redundant with "
                        f"existing late_kick (|r|={max(abs(r_redundancy), abs(r_mml_vs_late)):.2f})")
    else:
        print(f"  Gate A PASS: orthogonal to existing late_kick")
        print(f"     (|r|={max(abs(r_redundancy), abs(r_mml_vs_late)):.4f} < 0.7)")

    best_signal = max(abs(r_signal_mid), abs(r_signal_mml), abs(r_signal_last))
    if best_signal < 0.02:
        verdicts.append(f"KILL — Gate B: no detectable signal vs is_top3 "
                        f"(best |r|={best_signal:.4f} < 0.02)")
    elif abs(r_signal_mml) > 0.03:
        print(f"  Gate C PASS: mid_minus_late carries place-outcome signal")
        print(f"     (|r|={abs(r_signal_mml):.4f} > 0.03)")
    else:
        print(f"  Gate C MARGINAL: mid_minus_late |r|={abs(r_signal_mml):.4f}")
        print(f"     (< 0.03 threshold for 'worth retraining')")

    print()
    if verdicts:
        for v in verdicts:
            print(f"  {v}")
        print()
        print("  RECOMMENDATION: do NOT retrain. The mid-race sectional feature")
        print("  failed the cheap diagnostic; skip the retrain cycle.")
        conn.close()
        return 0

    if abs(r_signal_mml) > 0.03:
        print("  RECOMMENDATION: PROCEED to retrain test.")
        print("  Add mid_race_pace_norm + mid_minus_late as Features 25-26 in")
        print("  ml/feature_schema.py; retrain; compare top-4 capture rates on")
        print("  the holdout. Apply same retain/kill discipline as past feature")
        print("  experiments (place ECE, top-3/top-4 capture, NOT just Brier).")
    else:
        print("  RECOMMENDATION: MARGINAL. Signal exists but weak; consider")
        print("  combining with a second under-used signal before retraining.")

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
