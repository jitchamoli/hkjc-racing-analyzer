"""Scale-fixed V2 performance-rating jockey ELO — SHADOW LOGGER ONLY.

Per user 2026-05-25 (memory: project_jockey_elo_v1_killed_v2_passed_2026-05-25):
  V2 passed cheap diagnostic at Gate B r=+0.0986 (sane top jockeys).
  CAVEAT: the V2 in diagnose_jockey_elo_v2.py uses
    expected = mp_win        (mean ~0.07)
    actual   = rank_score    (mean ~0.5)
  → +K·0.43 net drift per ride for every jockey → no convergence.
  Ordering still works but absolute ELO is cosmetic.

This script fixes that mismatch and runs ONLY as a shadow logger:

  expected = Harville-derived E[rank_score]
             (Plackett-Luce Monte Carlo, 200 samples/race, fixed seed)
  actual   = 1 - (finish_position - 1) / (field_size - 1)

Both expected and actual live in [0, 1] with race-conditional means
close to 0.5, so the per-ride delta has zero net drift and the ELO
converges around START_ELO.

What it does
============
  1. Single chronological pass over race_results (strict leak-free —
     pre-race state recorded BEFORE post-race update).
  2. Writes one CSV row per (race, horse) to
     `tracking_data/jockey_perf_elo_shadow.csv` with the columns the
     downstream evaluator will need to correlate with frozen
     `race_bet_ticket` REPLAY ROI + top-4/5 capture.
  3. Prints the same Gate A1/A2/B sanity numbers as the diagnostic so
     we can see whether the scale fix preserves the +0.0986 signal.

What it deliberately does NOT do
================================
  * Touch the model. No retrain, no calibrator refit.
  * Touch the dashboard. No bet/rank/stake influence.
  * Modify `race_results`, `race_bet_ticket`, or any production table.

The output CSV is decision support for an offline evaluator only.
Re-run after each meeting to refresh; idempotent (overwrites in place).
"""
from __future__ import annotations

import csv
import math
import os
import random
import sqlite3
import statistics
import sys
import tempfile
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

# ELO parameters (same as diagnostic so the comparison is apples-to-apples)
K_FACTOR = 32.0
START_ELO = 1500.0
HALF_LIFE_DAYS = 365.0
DECAY_RATE = math.log(2) / HALF_LIFE_DAYS

# Diagnostic gate parameters
MIN_OBS_FOR_GATE = 20
WINDOW_30D_DAYS = 30

# Monte Carlo for Harville E[rank_score]
PL_SAMPLES = 200
PL_SEED = 42

# Output
SHADOW_CSV = REPO_ROOT / 'tracking_data' / 'jockey_perf_elo_shadow.csv'
CSV_HEADER = [
    'race_date', 'race_number', 'venue', 'distance', 'class_type',
    'horse_number', 'horse_name', 'jockey',
    'win_odds', 'market_p_win_norm', 'expected_rank_score_harville',
    'jockey_perf_elo_pre_race',
    'jockey_career_rides_pre_race', 'jockey_career_wins_pre_race',
    'jockey_career_wr_pre_race', 'jockey_30d_wr_pre_race',
    'finish_position', 'field_size', 'actual_rank_score',
    'is_winner', 'is_top3',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def parse_date(s: str) -> datetime:
    return datetime.strptime(s.replace('/', '-'), '%Y-%m-%d')


def harville_expected_rank_scores(win_probs: list[float],
                                  n_samples: int = PL_SAMPLES,
                                  seed: int = PL_SEED) -> list[float]:
    """E[rank_score_i | PL(win_probs)] via Monte Carlo Plackett-Luce sampling.

    rank_score = 1 - (finish_position - 1) / (field_size - 1) ∈ [0, 1]

    Uses a deterministic seed per call so results are reproducible run-to-run.
    PL sampling: pick 1st-finisher by weighted draw, remove, repeat for 2nd,
    etc. O(N · n_samples) per race; for N≤14 and n_samples=200 this is fast.
    """
    N = len(win_probs)
    if N < 2:
        return [1.0] * N
    rng = random.Random(seed)
    total = [0.0] * N
    for _ in range(n_samples):
        remaining = list(range(N))
        # Local copy of probs by horse-index (immutable across draws)
        for rank in range(1, N + 1):
            # Weighted draw from remaining horses by their win_prob
            denom = 0.0
            for h in remaining:
                denom += win_probs[h]
            if denom <= 0:
                break
            u = rng.random() * denom
            cum = 0.0
            chosen_idx = len(remaining) - 1  # fallback
            for idx, h in enumerate(remaining):
                cum += win_probs[h]
                if cum >= u:
                    chosen_idx = idx
                    break
            h = remaining[chosen_idx]
            total[h] += 1.0 - (rank - 1) / (N - 1)
            remaining.pop(chosen_idx)
    return [s / n_samples for s in total]


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    """Tmp + os.replace so partial writes never overwrite an existing log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADER, lineterminator='\n')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))

    print("Step 1: scanning race_results in chronological order...")
    rows = conn.execute("""
        SELECT race_date, race_number, venue, distance, class_type,
               horse_number, horse_name, jockey, win_odds,
               is_winner, is_top3, finish_position
        FROM race_results
        WHERE jockey IS NOT NULL AND jockey != ''
          AND win_odds IS NOT NULL AND win_odds > 0
          AND is_top3 IS NOT NULL
          AND finish_position > 0
        ORDER BY race_date, race_number
    """).fetchall()
    print(f"  {len(rows):,} rows pulled")

    races: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for r in rows:
        races[(r[0], r[1])].append(r)
    sorted_keys = sorted(races.keys())
    print(f"  {len(sorted_keys):,} unique races")

    # State
    elo: dict[str, float] = defaultdict(lambda: START_ELO)
    last_date: dict[str, datetime] = {}
    career_rides: dict[str, int] = defaultdict(int)
    career_wins: dict[str, int] = defaultdict(int)
    recent_30d: dict[str, deque] = defaultdict(deque)

    # Gate samples (only for n≥MIN_OBS_FOR_GATE)
    xs_elo, xs_career_wr, xs_30d_wr, ys_top3 = [], [], [], []

    csv_rows: list[dict] = []
    skipped_low_obs = 0
    processed_races = 0

    for race_key in sorted_keys:
        race_rows = races[race_key]
        race_date_str = race_key[0]
        try:
            race_date = parse_date(race_date_str)
        except Exception:
            continue

        # Market normalised win probs
        raw_p = [1.0 / float(r[8]) for r in race_rows]
        total_p = sum(raw_p)
        if total_p <= 0:
            continue
        win_probs = [p / total_p for p in raw_p]
        field_size = len(race_rows)
        if field_size < 2:
            continue

        # Harville E[rank_score] for the expected side of the ELO update
        exp_rs = harville_expected_rank_scores(win_probs)

        # Decay each jockey's ELO to current race date BEFORE recording
        decayed_pre = []
        for r in race_rows:
            jky = r[7]
            prev = last_date.get(jky)
            if prev is not None:
                days = max(0, (race_date - prev).days)
                decay = math.exp(-DECAY_RATE * days)
                elo[jky] = 1500.0 + (elo[jky] - 1500.0) * decay
            decayed_pre.append(elo[jky])

        # Record samples + CSV rows (using PRE-UPDATE state)
        for (row, pre_elo, mp_win, mp_exp_rs) in zip(
                race_rows, decayed_pre, win_probs, exp_rs):
            (rd, rn, venue, distance, class_type,
             horse_num, horse_name, jky,
             win_odds, is_win, is_top3, fp) = row

            dq = recent_30d[jky]
            cutoff = race_date.toordinal() - WINDOW_30D_DAYS
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            n30 = len(dq)
            wr30 = (sum(int(w) for _, w in dq) / n30) if n30 else 0.0
            n_career = career_rides[jky]
            wr_career = (career_wins[jky] / n_career) if n_career else 0.0

            actual_rs = 1.0 - (fp - 1) / (field_size - 1)
            actual_rs = max(0.0, min(1.0, actual_rs))

            csv_rows.append({
                'race_date': rd,
                'race_number': rn,
                'venue': venue or '',
                'distance': distance or 0,
                'class_type': class_type or '',
                'horse_number': horse_num or 0,
                'horse_name': horse_name or '',
                'jockey': jky,
                'win_odds': f'{float(win_odds):.2f}',
                'market_p_win_norm': f'{mp_win:.6f}',
                'expected_rank_score_harville': f'{mp_exp_rs:.6f}',
                'jockey_perf_elo_pre_race': f'{pre_elo:.2f}',
                'jockey_career_rides_pre_race': n_career,
                'jockey_career_wins_pre_race': career_wins[jky],
                'jockey_career_wr_pre_race': f'{wr_career:.6f}',
                'jockey_30d_wr_pre_race': f'{wr30:.6f}',
                'finish_position': fp,
                'field_size': field_size,
                'actual_rank_score': f'{actual_rs:.6f}',
                'is_winner': int(bool(is_win)),
                'is_top3': int(bool(is_top3)),
            })

            if n_career >= MIN_OBS_FOR_GATE:
                xs_elo.append(pre_elo)
                xs_career_wr.append(wr_career)
                xs_30d_wr.append(wr30)
                ys_top3.append(float(is_top3 or 0))
            else:
                skipped_low_obs += 1

        # Post-race ELO update — SCALE-FIXED V2
        for (row, mp_exp_rs) in zip(race_rows, exp_rs):
            (_, _, _, _, _, _, _, jky, _, is_win, _, fp) = row
            actual_rs = 1.0 - (fp - 1) / (field_size - 1)
            actual_rs = max(0.0, min(1.0, actual_rs))
            elo[jky] += K_FACTOR * (actual_rs - mp_exp_rs)
            last_date[jky] = race_date
            career_rides[jky] += 1
            if is_win:
                career_wins[jky] += 1
            recent_30d[jky].append((race_date.toordinal(), bool(is_win)))

        processed_races += 1

    print(f"  Processed {processed_races:,} races, "
          f"{len(csv_rows):,} per-horse rows")
    print(f"  Eligible for gates: {len(ys_top3):,}  "
          f"(skipped {skipped_low_obs:,} with <{MIN_OBS_FOR_GATE} prior rides)")

    # Write shadow CSV (atomic)
    atomic_write_csv(SHADOW_CSV, csv_rows)
    print(f"\n✓ Wrote {SHADOW_CSV.relative_to(REPO_ROOT)} "
          f"({SHADOW_CSV.stat().st_size / 1024:.1f} KB)")

    # ---- Diagnostics: did the scale fix preserve the +0.0986 signal? ----
    if len(ys_top3) < 5000:
        print("INSUFFICIENT DATA FOR GATE REPORT")
        return 0

    print(f"\n{'=' * 70}")
    print("SCALE-FIXED V2 — GATE REPORT")
    print(f"{'=' * 70}")
    print(f"  n = {len(ys_top3):,}")
    print(f"  Distribution of jockey_perf_elo_pre_race:")
    print(f"    mean={statistics.mean(xs_elo):.1f}  "
          f"sd={statistics.stdev(xs_elo):.1f}  "
          f"min={min(xs_elo):.1f}  max={max(xs_elo):.1f}")

    eligible_finals = sorted(
        ((j, elo[j], career_rides[j], career_wins[j])
         for j in elo if career_rides[j] >= 100),
        key=lambda t: t[1], reverse=True,
    )
    print(f"\n  Top 5 jockeys by current ELO (≥100 career rides):")
    for j, e, n, w in eligible_finals[:5]:
        print(f"    {j:<25} ELO {e:7.1f}  rides {n:5d}  wins {w:4d} "
              f"(career {w/n*100:.1f}%)")
    print(f"  Bottom 5 jockeys by current ELO (≥100 career rides):")
    for j, e, n, w in eligible_finals[-5:]:
        print(f"    {j:<25} ELO {e:7.1f}  rides {n:5d}  wins {w:4d} "
              f"(career {w/n*100:.1f}%)")

    r_b   = pearson(xs_elo, ys_top3)
    r_a1  = pearson(xs_elo, xs_career_wr)
    r_a2  = pearson(xs_elo, xs_30d_wr)
    r_base_career = pearson(xs_career_wr, ys_top3)
    r_base_30d    = pearson(xs_30d_wr, ys_top3)

    print(f"\n  Gate A1 (vs career WR)   r = {r_a1:+.4f}  "
          f"{'PASS' if abs(r_a1) < 0.5 else 'KILL'}")
    print(f"  Gate A2 (vs 30d form)    r = {r_a2:+.4f}  "
          f"{'PASS' if abs(r_a2) < 0.7 else 'KILL'}")
    print(f"  Gate B  (vs is_top3)     r = {r_b:+.4f}  "
          f"{'PASS' if abs(r_b) > 0.03 else ('MARGINAL' if abs(r_b) > 0.02 else 'KILL')}")
    print()
    print("  Reference baselines on same sample:")
    print(f"    r(career_wr, is_top3) = {r_base_career:+.4f}")
    print(f"    r(30d_wr,    is_top3) = {r_base_30d:+.4f}")
    print(f"  Scale-broken V2 baseline (mp_win as expected): r = +0.0986")

    print(f"\n{'=' * 70}")
    print("SHADOW POLICY REMINDER")
    print(f"{'=' * 70}")
    print("  This CSV is decision support for an OFFLINE evaluator.")
    print("  No production feature, no retrain, no dashboard influence.")
    print("  Evaluate against frozen race_bet_ticket REPLAY ROI + top-4/5")
    print("  capture over N≥6 forward meetings (post-2026/05/20 cutoff)")
    print("  before ANY model decision.")

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
