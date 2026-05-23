"""Cheap-diagnostic for the handicap expert's #1 recommendation:
"sire-on-going strike rate as a feature — biggest single missing piece;
HK is a closed gene pool so per-sire sample sizes are big enough to be
real, unlike per-horse going history which is too sparse for smoothing."

Candidate feature: sire_top3_rate_on_going
  For each (horse, race) outcome, look up the horse's sire (renavon_horses)
  and compute the sire's top-3 rate on the SAME going bucket, using ONLY
  races with race_date < current_race.date (no leakage). Bayesian-smoothed
  with a Beta(2, 8) prior so the long-tail of 1-progeny sires shrinks to
  base rate.

Going buckets (normalised — the DB has both short-form codes and full
strings; standardising prevents tiny per-string buckets):
  FIRM     : GOOD, GOOD TO FIRM, FIRM, FAST, GD, G, GF, FT
  GY       : GOOD TO YIELDING, GY
  YIELDING : YIELDING, Y
  SOFT     : SOFT, HEAVY, S, H, WET-SLOW, WET SLOW, WS, RAIN AFFECTED, RA
  WET-FAST : WET-FAST, WF
  (anything unrecognised treated as its raw upper-case string)

Three kill-gates + one keep-gate:
  Gate A1 (redundancy with horse-level win rate):
    |r(sire_top3_rate_on_going, prior_win_rate)|  > 0.5  → KILL
  Gate A2 (redundancy with per-horse going history):
    |r(sire_top3_rate_on_going, going_performance)| > 0.7 → KILL
  Gate B  (signal vs outcome):
    |r(sire_top3_rate_on_going, is_top3)|         < 0.02 → KILL (no signal)
    |r(sire_top3_rate_on_going, is_top3)|         > 0.03 → PROCEED

Single-pass O(N) implementation: walk rows in (date, race) order,
accumulating per-(sire, going_bucket) counts AS WE GO. For each row
the diagnostic feature is computed from the CURRENT counters (= prior
state) — strictly leak-free without per-row subqueries.
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


GOING_BUCKETS = {
    'GOOD': 'FIRM', 'GOOD TO FIRM': 'FIRM', 'FIRM': 'FIRM', 'FAST': 'FIRM',
    'GD': 'FIRM', 'G': 'FIRM', 'GF': 'FIRM', 'FT': 'FIRM',
    'GOOD TO YIELDING': 'GY', 'GY': 'GY',
    'YIELDING': 'YIELDING', 'Y': 'YIELDING', 'YIELDING TO SOFT': 'YIELDING', 'YS': 'YIELDING',
    'SOFT': 'SOFT', 'HEAVY': 'SOFT', 'S': 'SOFT', 'H': 'SOFT',
    'WET-SLOW': 'SOFT', 'WET SLOW': 'SOFT', 'WS': 'SOFT',
    'RAIN AFFECTED': 'SOFT', 'RA': 'SOFT',
    'WET-FAST': 'WET-FAST', 'WET FAST': 'WET-FAST', 'WF': 'WET-FAST',
}


def bucket_going(g: str) -> str:
    if not g:
        return ''
    u = g.strip().upper()
    return GOING_BUCKETS.get(u, u)


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

    # Step 1: load horse → sire map from renavon_horses (99.7% coverage).
    print("Step 1: loading horse → sire map...")
    horse_to_sire = {}
    for r in conn.execute(
        "SELECT horse_name, sire FROM renavon_horses "
        "WHERE sire IS NOT NULL AND sire != ''"
    ).fetchall():
        horse_to_sire[r['horse_name']] = r['sire']
    print(f"  {len(horse_to_sire):,} horses with sire info\n")

    # Step 2: pull all race outcomes in chronological order.
    print("Step 2: scanning race_results in chronological order...")
    rows = conn.execute("""
        SELECT race_date, race_number, horse_name, going,
               is_top3, is_winner, finish_position
        FROM race_results
        WHERE finish_position > 0 AND going != '' AND going IS NOT NULL
        ORDER BY race_date, race_number
    """).fetchall()
    print(f"  {len(rows):,} total result rows\n")

    # Step 3: single-pass accumulation. For each row:
    #   1. compute candidate feature from CURRENT (prior) counters
    #   2. compute redundancy features (prior_win_rate, going_perf) same way
    #   3. record outcome
    #   4. THEN update counters with this row's outcome
    #
    # Per-(sire, bucket) state: (n_starts, n_top3)
    sire_going = defaultdict(lambda: [0, 0])
    # Per-horse state: (n_starts, n_wins, n_top3)
    horse_total = defaultdict(lambda: [0, 0, 0])
    # Per-(horse, bucket) state: (n_starts, n_top3)
    horse_going = defaultdict(lambda: [0, 0])

    xs_sire, xs_horse_win, xs_horse_going, ys = [], [], [], []
    skipped_no_sire = 0
    for r in rows:
        horse = r['horse_name']
        sire = horse_to_sire.get(horse)
        bucket = bucket_going(r['going'])
        if not sire or not bucket:
            skipped_no_sire += 1
            continue

        # Bayesian smoothing: Beta(2, 8) prior → base_rate 20%, prior weight 10.
        # This matches the place-base-rate in HK (top3 ≈ 25% on average) and
        # shrinks rare sires (≤10 starts on going) heavily toward it.
        sg_starts, sg_top3 = sire_going[(sire, bucket)]
        sire_t3_rate = (sg_top3 + 2) / (sg_starts + 10)

        # Horse prior win rate (Beta(1, 12) — base ~7.7% win rate)
        h_starts, h_wins, h_top3 = horse_total[horse]
        h_win_rate = (h_wins + 1) / (h_starts + 13)

        # Horse going performance (Beta(2, 6) — base ~25% top3)
        hg_starts, hg_top3 = horse_going[(horse, bucket)]
        h_going_rate = (hg_top3 + 2) / (hg_starts + 8)

        xs_sire.append(sire_t3_rate)
        xs_horse_win.append(h_win_rate)
        xs_horse_going.append(h_going_rate)
        ys.append(float(r['is_top3'] or 0))

        # Now update counters with this row (AFTER recording).
        sg_starts += 1
        if r['is_top3']:
            sg_top3 += 1
        sire_going[(sire, bucket)] = [sg_starts, sg_top3]

        h_starts += 1
        if r['is_winner']:
            h_wins += 1
        if r['is_top3']:
            h_top3 += 1
        horse_total[horse] = [h_starts, h_wins, h_top3]

        hg_starts += 1
        if r['is_top3']:
            hg_top3 += 1
        horse_going[(horse, bucket)] = [hg_starts, hg_top3]

    print(f"  {len(ys):,} rows with sire + going bucket")
    print(f"  {skipped_no_sire:,} skipped (no sire or unrecognised going)\n")

    # Coverage by sire sample size
    sire_starts = defaultdict(int)
    for (sire, _), (n, _) in sire_going.items():
        sire_starts[sire] += n
    big_sires = sum(1 for n in sire_starts.values() if n >= 100)
    huge_sires = sum(1 for n in sire_starts.values() if n >= 500)
    print(f"  {len(sire_starts):,} unique sires represented")
    print(f"    with ≥100 progeny-starts: {big_sires:,}")
    print(f"    with ≥500 progeny-starts: {huge_sires:,}\n")

    if len(ys) < 100:
        print("INSUFFICIENT DATA")
        return 1

    # Step 4: correlations
    print("=" * 72)
    print("CORRELATION MATRIX")
    print("=" * 72)
    print(f"  n = {len(ys):,} aligned rows\n")
    r_sig = pearson(xs_sire, ys)
    r_red_winrate = pearson(xs_sire, xs_horse_win)
    r_red_going = pearson(xs_sire, xs_horse_going)
    r_horse_win_vs_top3 = pearson(xs_horse_win, ys)
    r_horse_going_vs_top3 = pearson(xs_horse_going, ys)

    print("  Gate A1 (redundancy with horse-level win rate):")
    print(f"     r(sire_t3_on_going, horse_prior_win_rate) = {r_red_winrate:+.4f}")
    print()
    print("  Gate A2 (redundancy with per-horse going history):")
    print(f"     r(sire_t3_on_going, horse_going_rate)     = {r_red_going:+.4f}")
    print()
    print("  Gate B (signal vs is_top3):")
    print(f"     r(sire_t3_on_going,    is_top3)           = {r_sig:+.4f}")
    print()
    print("  Baselines (for comparison):")
    print(f"     r(horse_prior_win_rate, is_top3)          = {r_horse_win_vs_top3:+.4f}")
    print(f"     r(horse_going_rate,    is_top3)           = {r_horse_going_vs_top3:+.4f}")
    print()

    # Step 5: verdict
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    verdicts = []
    if abs(r_red_winrate) > 0.5:
        verdicts.append(
            f"KILL — Gate A1: sire-on-going overlaps horse win rate "
            f"|r|={abs(r_red_winrate):.2f} > 0.5"
        )
    else:
        print(f"  Gate A1 PASS: orthogonal to horse_prior_win_rate "
              f"(|r|={abs(r_red_winrate):.4f} < 0.5)")
    if abs(r_red_going) > 0.7:
        verdicts.append(
            f"KILL — Gate A2: redundant with per-horse going history "
            f"|r|={abs(r_red_going):.2f} > 0.7"
        )
    else:
        print(f"  Gate A2 PASS: orthogonal to horse_going_rate "
              f"(|r|={abs(r_red_going):.4f} < 0.7)")
    if abs(r_sig) < 0.02:
        verdicts.append(
            f"KILL — Gate B: no detectable signal vs is_top3 "
            f"|r|={abs(r_sig):.4f} < 0.02"
        )
    elif abs(r_sig) > 0.03:
        print(f"  Gate B PASS: sire_t3_on_going carries place-outcome signal "
              f"(|r|={abs(r_sig):.4f} > 0.03)")
    else:
        print(f"  Gate B MARGINAL: |r|={abs(r_sig):.4f} between 0.02-0.03 "
              f"threshold")
    print()
    if verdicts:
        for v in verdicts:
            print(f"  {v}")
        print()
        print("  RECOMMENDATION: do NOT retrain. The sire feature failed the")
        print("  cheap diagnostic; skip the retrain cycle.")
        conn.close()
        return 0
    if abs(r_sig) > 0.03:
        # Compare to existing baselines so the reader can scale it
        ratio_vs_horse_win = abs(r_sig) / abs(r_horse_win_vs_top3) if r_horse_win_vs_top3 else float('inf')
        print(f"  RECOMMENDATION: PROCEED to retrain test.")
        print(f"  Signal strength is {ratio_vs_horse_win:.0%} of horse_prior_win_rate's")
        print(f"  correlation with is_top3 — a meaningful fraction of an")
        print(f"  established top feature.")
        print()
        print(f"  Next step: add as Feature 26 (sire_top3_rate_on_going) in")
        print(f"  ml/feature_schema.py; retrain; measure on forward log via")
        print(f"  scripts/validate_hk.py.")
    else:
        print(f"  RECOMMENDATION: MARGINAL. Don't retrain on this signal alone.")
    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
