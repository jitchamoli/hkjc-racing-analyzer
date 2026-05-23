"""Test the jockey's mid-race-sectional recommendation by re-ranking
production predictions on the last N meetings.

This is a SIGNAL-ADDITIVITY test, not a retrain. It answers:
  "If we add `mid_minus_late` as a re-ranking nudge on top of the
   current production calibrated_prob, does top-N capture improve?"

If the answer is "no improvement at any blend weight", the full retrain
will not help either (interactions can recover at most a small fraction
of what a clean additive blend cannot find). If the answer is "yes",
the retrain test is worth running.

Caveats acknowledged:
  - This is BURNED data — the model trained on (or near) these meetings.
    Burned hit-rates are upper bounds on what we'd expect forward.
  - Re-ranking ignores interactions a full retrain could pick up — so a
    negative result here doesn't strictly preclude retrain benefit, but
    a positive result is encouraging.
  - The blend weight λ is swept; reporting the BEST λ on burned data
    invites overfitting. We report the FULL sweep and pick λ=0.0 if it's
    competitive (honest baseline).
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from scripts.diagnose_mid_race_sectional import (  # noqa: E402
    compute_mid_race_features, parse_sectionals,
)


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # Step 1: build per-horse mid-race history (same as diagnostic).
    print("Step 1: building per-horse mid-race history...")
    rows = conn.execute("""
        SELECT race_date, race_number, horse_name, sectional_times
        FROM race_results
        WHERE sectional_times != '' AND sectional_times IS NOT NULL
          AND finish_position > 0
        ORDER BY horse_name, race_date
    """).fetchall()
    per_horse = defaultdict(list)
    for r in rows:
        feats = compute_mid_race_features(parse_sectionals(r['sectional_times']))
        if feats:
            per_horse[r['horse_name']].append({
                'race_date': r['race_date'], **feats,
            })
    print(f"  {len(per_horse):,} horses with sectional history\n")

    # Step 2: pull last 6 meetings of production predictions + results.
    meetings = [r[0] for r in conn.execute("""
        SELECT DISTINCT race_date FROM prediction_log
        WHERE results_filled=1 ORDER BY race_date DESC LIMIT 6
    """).fetchall()]
    print(f"Step 2: testing on {len(meetings)} meetings: "
          f"{meetings[-1]} .. {meetings[0]}\n")

    # For each (meeting, race): horses with their baseline rank, calibrated_prob,
    # actual_top3 outcome, and mid_minus_late aggregate from prior runs.
    races = []  # list of dicts: {date, rno, horses: [...], actual_top3_names: set}
    for date in meetings:
        rno_horses = defaultdict(list)
        for r in conn.execute("""
            SELECT race_number, horse_name, predicted_rank, calibrated_prob,
                   actual_position, actual_top3
            FROM prediction_log
            WHERE race_date=? AND results_filled=1
              AND predicted_rank IS NOT NULL AND calibrated_prob IS NOT NULL
        """, (date,)).fetchall():
            history = per_horse.get(r['horse_name'], [])
            prior = [h for h in history if h['race_date'] < date][-3:]
            mml = (sum(h['mid_minus_late'] for h in prior) / len(prior)
                   if len(prior) >= 2 else None)
            rno_horses[r['race_number']].append({
                'horse_name': r['horse_name'],
                'predicted_rank': r['predicted_rank'],
                'calibrated_prob': r['calibrated_prob'],
                'mid_minus_late': mml,
                'actual_position': r['actual_position'],
                'actual_top3': bool(r['actual_top3']),
            })
        for rno, horses in rno_horses.items():
            if len(horses) < 5:
                continue
            actual_top3_names = {h['horse_name'] for h in horses
                                 if h['actual_position'] in (1, 2, 3)}
            if len(actual_top3_names) != 3:
                continue  # need a clean podium
            races.append({
                'date': date, 'rno': rno, 'horses': horses,
                'actual_top3_names': actual_top3_names,
                'winner': next((h['horse_name'] for h in horses
                                if h['actual_position'] == 1), None),
            })

    print(f"  {len(races)} races with clean podium + valid predictions\n")
    if not races:
        print("INSUFFICIENT DATA")
        return 1

    # Coverage: how many horses have a usable mid_minus_late?
    total_horses = sum(len(r['horses']) for r in races)
    covered = sum(1 for r in races for h in r['horses']
                  if h['mid_minus_late'] is not None)
    print(f"  feature coverage: {covered}/{total_horses} horses "
          f"({100 * covered / total_horses:.1f}%)\n")

    # Step 3: compute z-score of mid_minus_late within EACH race (so the
    # nudge is race-internal, not absolute — fairer across distances).
    for race in races:
        vals = [h['mid_minus_late'] for h in race['horses']
                if h['mid_minus_late'] is not None]
        if len(vals) >= 2:
            mu = sum(vals) / len(vals)
            sd = statistics.pstdev(vals) or 1.0
            for h in race['horses']:
                h['mml_z'] = ((h['mid_minus_late'] - mu) / sd
                              if h['mid_minus_late'] is not None else 0.0)
        else:
            for h in race['horses']:
                h['mml_z'] = 0.0

    # Step 4: sweep blend weight λ. For each λ:
    #   composite_score = calibrated_prob + λ × mml_z × (cal_prob_scale)
    # where cal_prob_scale ≈ mean(cal_prob) so λ=1.0 means the z-shift can
    # move a horse roughly one "average prob unit". Then re-rank within
    # each race by composite_score descending → new top-N sets.
    cal_scale = sum(h['calibrated_prob'] for r in races for h in r['horses']) \
        / sum(len(r['horses']) for r in races)
    print(f"  calibrated_prob scale (mean) = {cal_scale:.4f}\n")

    LAMBDAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    def measure(lmbda):
        win, qin, top3_in_top4, top3_in_top5 = 0, 0, 0, 0
        for race in races:
            for h in race['horses']:
                h['_score'] = h['calibrated_prob'] + lmbda * h['mml_z'] * cal_scale
            ranked = sorted(race['horses'], key=lambda h: -h['_score'])
            top1 = ranked[0]['horse_name']
            top2 = {h['horse_name'] for h in ranked[:2]}
            top4 = {h['horse_name'] for h in ranked[:4]}
            top5 = {h['horse_name'] for h in ranked[:5]}
            actual_top2 = {next((h['horse_name'] for h in race['horses']
                                 if h['actual_position'] == p), None)
                           for p in (1, 2)} - {None}
            if top1 == race['winner']:
                win += 1
            if actual_top2 and actual_top2 <= top2:
                qin += 1
            if race['actual_top3_names'] <= top4:
                top3_in_top4 += 1
            if race['actual_top3_names'] <= top5:
                top3_in_top5 += 1
        return win, qin, top3_in_top4, top3_in_top5

    # Step 5: print baseline + each λ.
    n = len(races)
    print("=" * 78)
    print(f"RE-RANK CAPTURE TEST — {n} races, blend = cal_prob + λ × mml_z × scale")
    print("=" * 78)
    print(f"{'λ':>6} {'TOP-1 WIN':>14} {'QIN (top-2)':>14} {'TOP-4-CAP':>14} {'TOP-5-CAP':>14}")
    print("-" * 78)
    base = None
    for lmbda in LAMBDAS:
        w, q, t4, t5 = measure(lmbda)
        if base is None:
            base = (w, q, t4, t5)
            tag = ' (baseline)'
        else:
            tag = ''
        marker_w = " " if w == base[0] else ("+" if w > base[0] else "-")
        marker_q = " " if q == base[1] else ("+" if q > base[1] else "-")
        marker_4 = " " if t4 == base[2] else ("+" if t4 > base[2] else "-")
        marker_5 = " " if t5 == base[3] else ("+" if t5 > base[3] else "-")
        print(f"{lmbda:>6.2f} "
              f"{w:>4}/{n:<3} {100*w/n:>5.1f}%{marker_w} "
              f"{q:>4}/{n:<3} {100*q/n:>5.1f}%{marker_q} "
              f"{t4:>4}/{n:<3} {100*t4/n:>5.1f}%{marker_4} "
              f"{t5:>4}/{n:<3} {100*t5/n:>5.1f}%{marker_5}"
              f"{tag}")
    print()
    print("Legend: + = improved vs λ=0 baseline, - = regressed, blank = unchanged")
    print()

    # Step 6: honest verdict
    print("=" * 78)
    print("VERDICT (signal-additivity test, BURNED data — UPPER BOUND)")
    print("=" * 78)
    best = max(LAMBDAS, key=lambda l: sum(measure(l)))
    best_metrics = measure(best)
    base_metrics = measure(0.0)
    deltas = [b - a for a, b in zip(base_metrics, best_metrics)]
    print(f"  best λ = {best} (total hits across 4 metrics)")
    print(f"  vs baseline λ=0: ΔWIN={deltas[0]:+d}  ΔQIN={deltas[1]:+d}  "
          f"ΔTOP4={deltas[2]:+d}  ΔTOP5={deltas[3]:+d}")
    print()
    if sum(deltas) > 0 and any(d > 0 for d in deltas):
        print("  → SIGNAL ADDITIVITY POSITIVE on the burned 6-meeting holdout.")
        print("    A full retrain has a chance of moving forward top-N capture.")
        print("    Next step: add feature, retrain, evaluate on the FORWARD log")
        print("    (scripts/validate_hk.py) — that's the only honest measure.")
    elif sum(deltas) < 0:
        print("  → SIGNAL ADDITIVITY NEGATIVE on the burned holdout.")
        print("    Even with optimistic blending the feature HURTS. KILL.")
    else:
        print("  → SIGNAL ADDITIVITY NEUTRAL — no improvement at any λ.")
        print("    The diagnostic correlation is real but doesn't surface as")
        print("    actionable re-ranking lift. Retrain unlikely to help.")
        print("    Consistent with the project pattern: 'Brier improves, ROI")
        print("    doesn't' (memory: project_feature_experiments_pattern).")
    print()
    print("  HONEST CAVEAT: 6 meetings ≈ 40-50 races is small; even a")
    print("  +3 hit delta is well within sampling noise. Apply the small-sample")
    print("  inflation 3-5× discount rule (memory:")
    print("  feedback_small_sample_inflation_pattern).")

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
