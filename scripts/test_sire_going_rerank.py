"""Signal-additivity test for the candidate sire_top3_rate_on_going feature.

Same structure as test_mid_race_rerank.py — we re-rank the last 6 meetings'
production predictions by blending calibrated_prob with the candidate feature,
sweep blend weight λ, and report top-N capture.

This does NOT modify the production model. It answers the question:
  "If we had added sire-on-going as a re-ranking nudge to today's production
   predictions, would top-N capture have improved on the last 6 meetings?"

Caveats:
  - BURNED data — the production model was trained on or near these races.
  - A signal-additivity positive doesn't prove a retrain will help, but a
    signal-additivity negative is strong evidence the retrain won't either.
  - Sample is small (6 meetings ≈ 50-60 races). Apply the 3-5× small-sample
    discount rule (memory: feedback_small_sample_inflation_pattern).
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
from scripts.diagnose_sire_going import bucket_going  # noqa: E402


def _build_sire_going_lookup(conn):
    """Return a function: (sire, going_bucket, before_date) → smoothed top-3 rate.

    Implementation: pre-compute per-(sire, going_bucket) chronological lists
    of (race_date, is_top3) entries from race_results. For each query, count
    entries with race_date < before_date and apply Beta(2, 8) smoothing.
    """
    # horse → sire
    horse_to_sire = {
        r['horse_name']: r['sire']
        for r in conn.execute(
            "SELECT horse_name, sire FROM renavon_horses "
            "WHERE sire IS NOT NULL AND sire != ''"
        )
    }

    # (sire, bucket) → sorted list of (race_date, is_top3)
    timeline = defaultdict(list)
    for r in conn.execute("""
        SELECT race_date, horse_name, going, is_top3
        FROM race_results
        WHERE finish_position > 0 AND going != '' AND going IS NOT NULL
        ORDER BY race_date, race_number
    """):
        sire = horse_to_sire.get(r['horse_name'])
        bucket = bucket_going(r['going'])
        if not sire or not bucket:
            continue
        timeline[(sire, bucket)].append((r['race_date'], int(r['is_top3'] or 0)))

    def lookup(sire: str, bucket: str, before_date: str) -> float:
        if not sire or not bucket:
            return 0.20  # base rate fallback
        entries = timeline.get((sire, bucket), [])
        starts = 0
        top3s = 0
        for date, t3 in entries:
            if date >= before_date:
                break  # entries are sorted
            starts += 1
            top3s += t3
        return (top3s + 2) / (starts + 10)  # Beta(2, 8) — base 20%

    return horse_to_sire, lookup


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    print("Step 1: building horse→sire map + (sire, going) timelines...")
    horse_to_sire, lookup_sire = _build_sire_going_lookup(conn)
    print(f"  {len(horse_to_sire):,} horses with sire info\n")

    # Last 6 meetings of production predictions
    meetings = [r[0] for r in conn.execute("""
        SELECT DISTINCT race_date FROM prediction_log
        WHERE results_filled=1 ORDER BY race_date DESC LIMIT 6
    """).fetchall()]
    print(f"Step 2: testing on {len(meetings)} meetings: "
          f"{meetings[-1]} .. {meetings[0]}\n")

    races = []
    for date in meetings:
        # Get this meeting's going per race (from race_results, joined back)
        going_per_race = {}
        for r in conn.execute(
            "SELECT race_number, going FROM race_results "
            "WHERE race_date=? GROUP BY race_number",
            (date,),
        ):
            going_per_race[r['race_number']] = r['going']

        rno_horses = defaultdict(list)
        for r in conn.execute("""
            SELECT race_number, horse_name, predicted_rank, calibrated_prob,
                   actual_position, actual_top3
            FROM prediction_log
            WHERE race_date=? AND results_filled=1
              AND predicted_rank IS NOT NULL AND calibrated_prob IS NOT NULL
        """, (date,)).fetchall():
            sire = horse_to_sire.get(r['horse_name'])
            bucket = bucket_going(going_per_race.get(r['race_number'], ''))
            sire_rate = lookup_sire(sire, bucket, date) if sire and bucket else None
            rno_horses[r['race_number']].append({
                'horse_name': r['horse_name'],
                'predicted_rank': r['predicted_rank'],
                'calibrated_prob': r['calibrated_prob'],
                'sire_rate': sire_rate,
                'actual_position': r['actual_position'],
                'actual_top3': bool(r['actual_top3']),
            })
        for rno, horses in rno_horses.items():
            if len(horses) < 5:
                continue
            actual_top3_names = {h['horse_name'] for h in horses
                                 if h['actual_position'] in (1, 2, 3)}
            if len(actual_top3_names) != 3:
                continue
            races.append({
                'date': date, 'rno': rno, 'horses': horses,
                'actual_top3_names': actual_top3_names,
                'winner': next((h['horse_name'] for h in horses
                                if h['actual_position'] == 1), None),
            })

    print(f"  {len(races)} races with clean podium + valid predictions")
    total_horses = sum(len(r['horses']) for r in races)
    covered = sum(1 for r in races for h in r['horses']
                  if h['sire_rate'] is not None)
    print(f"  feature coverage: {covered}/{total_horses} horses "
          f"({100 * covered / total_horses:.1f}%)\n")

    # Z-score within each race so the blend is race-relative.
    for race in races:
        vals = [h['sire_rate'] for h in race['horses']
                if h['sire_rate'] is not None]
        if len(vals) >= 2:
            mu = sum(vals) / len(vals)
            sd = statistics.pstdev(vals) or 1.0
            for h in race['horses']:
                h['sire_z'] = ((h['sire_rate'] - mu) / sd
                               if h['sire_rate'] is not None else 0.0)
        else:
            for h in race['horses']:
                h['sire_z'] = 0.0

    cal_scale = sum(h['calibrated_prob'] for r in races for h in r['horses']) \
        / sum(len(r['horses']) for r in races)
    print(f"  calibrated_prob scale (mean) = {cal_scale:.4f}\n")

    LAMBDAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    def measure(lmbda):
        win, qin, t4, t5 = 0, 0, 0, 0
        for race in races:
            for h in race['horses']:
                h['_score'] = h['calibrated_prob'] + lmbda * h['sire_z'] * cal_scale
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
                t4 += 1
            if race['actual_top3_names'] <= top5:
                t5 += 1
        return win, qin, t4, t5

    n = len(races)
    print("=" * 78)
    print(f"RE-RANK TEST — {n} races, blend = cal_prob + λ × sire_z × scale")
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
        mw = " " if w == base[0] else ("+" if w > base[0] else "-")
        mq = " " if q == base[1] else ("+" if q > base[1] else "-")
        m4 = " " if t4 == base[2] else ("+" if t4 > base[2] else "-")
        m5 = " " if t5 == base[3] else ("+" if t5 > base[3] else "-")
        print(f"{lmbda:>6.2f} "
              f"{w:>4}/{n:<3} {100*w/n:>5.1f}%{mw} "
              f"{q:>4}/{n:<3} {100*q/n:>5.1f}%{mq} "
              f"{t4:>4}/{n:<3} {100*t4/n:>5.1f}%{m4} "
              f"{t5:>4}/{n:<3} {100*t5/n:>5.1f}%{m5}"
              f"{tag}")
    print()
    print("=" * 78)
    print("READING")
    print("=" * 78)
    # Honest reading: focus on TOP-4 (north star) and TOP-5 (BOX TRIO-5).
    base_w, base_q, base_t4, base_t5 = base
    for lmbda in LAMBDAS[1:]:
        w, q, t4, t5 = measure(lmbda)
        score = (t4 - base_t4) * 2 + (t5 - base_t5) * 2 + (w - base_w) + (q - base_q)
        if t4 >= base_t4 and t5 >= base_t5 and (t4 > base_t4 or t5 > base_t5 or w > base_w or q > base_q):
            print(f"  λ={lmbda}: STRICTLY IMPROVES at least one of "
                  f"(WIN, QIN, TOP-4, TOP-5) without regressing TOP-4 or TOP-5")
    print()
    print("  HONEST READ: this is BURNED data (model trained on or near these")
    print("  races). A positive here is necessary but not sufficient — the")
    print("  ONLY honest measure is forward log via scripts/validate_hk.py")
    print("  after N≥3 un-burned meetings.")
    print()
    print("  Apply small-sample discount: deltas <3 hits on 60 races are noise.")
    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
