"""Cheap diagnostic for the jockey-ELO candidate (roadmap stage 1 of 4).

Per the diagnostic-to-ship pipeline (memory: feedback_diagnostic_to_ship_pipeline),
new feature candidates must pass three correlation gates on the FULL DB
BEFORE any retrain or shadow-log experiment.

What this measures
==================
`jockey_elo_pre_race` — a market-anchored skill rating that updates each
race based on whether the jockey beat the market's expected win probability:

    expected = market_p = (1 / win_odds), normalized per race so Σ = 1
    actual   = 1 if jockey won, else 0
    delta    = K * (actual - expected)
    new_elo  = old_elo + delta

Time-decay: between races a jockey's rating drifts back toward 1500
(neutral) with a 365-day half-life. Handles long inactivity.

Why anchor on market: top HK jockeys get the best horses, so raw
win-rate confounds skill with mount quality (memory:
project_jt_market_sharpness_killed exposed exactly this trap on the
J×T candidate). Anchoring on market expectation isolates the
"above-market overperformance" signal — a jockey who wins a $20 horse
gets +K * 0.95 ELO; the same jockey winning a $1.4 fav only gets
+K * 0.30. Punishment is symmetric for losses on chalk.

Pre-race only: pre-race rating is recorded BEFORE applying the
post-race update. Strict leak-free — single chronological pass,
no per-row before_date subqueries needed.

Three gates (same shape as diagnose_jt_sharpness.py + diagnose_sire_going.py)
================================================================================
  A1  redundancy with jockey career win-rate    |r| > 0.5  → KILL
  A2  redundancy with jockey 30d form-rate      |r| > 0.7  → KILL
  B   signal vs is_top3                         |r| < 0.02 → KILL
                                                |r| > 0.03 → PROCEED to stage 2
"""
from __future__ import annotations

import math
import sqlite3
import statistics
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

# ELO parameters
K_FACTOR = 32.0          # standard chess K — moderate learn rate
START_ELO = 1500.0
HALF_LIFE_DAYS = 365.0   # exponential decay toward 1500 between rides
DECAY_RATE = math.log(2) / HALF_LIFE_DAYS

# Diagnostic parameters
MIN_OBS_FOR_GATE = 20    # require ≥20 prior rides before counting a row
WINDOW_30D_DAYS = 30     # 30-day form window for Gate A2 baseline


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
    # race_date format: YYYY-MM-DD or YYYY/MM/DD
    s = s.replace('/', '-')
    return datetime.strptime(s, '%Y-%m-%d')


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))

    print("Step 1: scanning race_results in chronological order...")
    # Single pass: jockey, win_odds, is_winner, is_top3, race_date, race_number
    rows = conn.execute("""
        SELECT race_date, race_number, jockey, win_odds, is_winner, is_top3
        FROM race_results
        WHERE jockey IS NOT NULL AND jockey != ''
          AND win_odds IS NOT NULL AND win_odds > 0
          AND is_top3 IS NOT NULL
        ORDER BY race_date, race_number
    """).fetchall()
    print(f"  {len(rows):,} rows with jockey + win_odds + is_top3")

    # Group rows by (race_date, race_number) so we can normalize market_p
    # per race (so probabilities sum to 1 within a race).
    races: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for r in rows:
        races[(r[0], r[1])].append(r)
    sorted_race_keys = sorted(races.keys())
    print(f"  {len(sorted_race_keys):,} unique races")

    # State per jockey
    elo: dict[str, float] = defaultdict(lambda: START_ELO)
    last_race_date: dict[str, datetime] = {}
    career_rides: dict[str, int] = defaultdict(int)
    career_wins: dict[str, int] = defaultdict(int)
    # 30-day rolling deque of (date, is_winner)
    recent_30d: dict[str, deque] = defaultdict(deque)

    # Collected per-row samples for gates
    xs_elo, xs_career_wr, xs_30d_wr, ys_top3 = [], [], [], []
    skipped_low_obs = 0

    for race_key in sorted_race_keys:
        race_rows = races[race_key]
        race_date_str = race_key[0]
        try:
            race_date = parse_date(race_date_str)
        except Exception:
            continue

        # Normalize market probabilities within this race
        raw_p = [1.0 / float(r[3]) for r in race_rows]
        total_p = sum(raw_p)
        if total_p <= 0:
            continue
        norm_p = [p / total_p for p in raw_p]

        # Decay each jockey's ELO toward 1500 based on days since last
        # ride, BEFORE recording pre-race rating for this row.
        decayed_pre = []
        for r in race_rows:
            jky = r[2]
            old_last = last_race_date.get(jky)
            if old_last is not None:
                days = max(0, (race_date - old_last).days)
                decay = math.exp(-DECAY_RATE * days)
                e = 1500.0 + (elo[jky] - 1500.0) * decay
                elo[jky] = e
            decayed_pre.append(elo[jky])

        # Record pre-race samples (leak-free — these reflect state BEFORE
        # the current race contributes to ratings).
        for (row, pre_elo, mp) in zip(race_rows, decayed_pre, norm_p):
            _, _, jky, _, is_win, is_top3 = row
            # Prune 30d deque for this jockey before reading
            dq = recent_30d[jky]
            cutoff = race_date.toordinal() - WINDOW_30D_DAYS
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            n30 = len(dq)
            wr30 = (sum(int(w) for _, w in dq) / n30) if n30 else 0.0
            n_career = career_rides[jky]
            wr_career = (career_wins[jky] / n_career) if n_career else 0.0

            if n_career >= MIN_OBS_FOR_GATE:
                xs_elo.append(pre_elo)
                xs_career_wr.append(wr_career)
                xs_30d_wr.append(wr30)
                ys_top3.append(float(is_top3 or 0))
            else:
                skipped_low_obs += 1

        # Apply post-race updates (jockeys' ELO + counters) using
        # normalized market expectation. Each race row contributes
        # K * (actual - expected_p) to its jockey's rating.
        for (row, mp) in zip(race_rows, norm_p):
            _, _, jky, _, is_win, _ = row
            actual = 1.0 if is_win else 0.0
            delta = K_FACTOR * (actual - mp)
            elo[jky] += delta
            last_race_date[jky] = race_date
            career_rides[jky] += 1
            if is_win:
                career_wins[jky] += 1
            recent_30d[jky].append((race_date.toordinal(), bool(is_win)))

    print(f"  Rows scored for correlation: {len(ys_top3):,}  "
          f"({skipped_low_obs:,} skipped — jockey had <{MIN_OBS_FOR_GATE} prior rides)")
    print(f"  Unique jockeys seen:        {len(elo):,}")
    eligible_jky = sum(1 for j, n in career_rides.items() if n >= MIN_OBS_FOR_GATE)
    print(f"  Jockeys that ever crossed n≥{MIN_OBS_FOR_GATE}: {eligible_jky:,}")

    if len(ys_top3) < 5000:
        print("INSUFFICIENT DATA")
        return 1

    # Distribution sanity
    print(f"\n  Distribution of jockey_elo_pre_race (when n≥{MIN_OBS_FOR_GATE}):")
    print(f"    mean={statistics.mean(xs_elo):.1f}  "
          f"sd={statistics.stdev(xs_elo):.1f}  "
          f"min={min(xs_elo):.1f}  max={max(xs_elo):.1f}")
    print(f"  Distribution of is_top3:")
    print(f"    mean={statistics.mean(ys_top3):.3f}  (base rate)")

    # Spot check: top 10 / bottom 10 jockeys by final ELO (eligible only)
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

    print("\n" + "=" * 70)
    print("CANDIDATE: jockey_elo_pre_race (market-anchored, time-decayed)")
    print("=" * 70)
    r_b   = pearson(xs_elo, ys_top3)
    r_a1  = pearson(xs_elo, xs_career_wr)
    r_a2  = pearson(xs_elo, xs_30d_wr)
    r_baseline_career = pearson(xs_career_wr, ys_top3)
    r_baseline_30d    = pearson(xs_30d_wr, ys_top3)
    print(f"  n = {len(ys_top3):,}")
    print(f"  Gate A1 (vs jockey career win-rate)   r = {r_a1:+.4f}  "
          f"{'PASS' if abs(r_a1) < 0.5 else 'KILL'}")
    print(f"  Gate A2 (vs jockey 30d form-rate)     r = {r_a2:+.4f}  "
          f"{'PASS' if abs(r_a2) < 0.7 else 'KILL'}")
    print(f"  Gate B  (vs is_top3)                  r = {r_b:+.4f}  "
          f"{'PASS' if abs(r_b) > 0.03 else ('MARGINAL' if abs(r_b) > 0.02 else 'KILL')}")
    print()
    print("  Baselines for scale:")
    print(f"    r(jockey_career_wr, is_top3)  = {r_baseline_career:+.4f}")
    print(f"    r(jockey_30d_wr,    is_top3)  = {r_baseline_30d:+.4f}")

    print("\nVERDICT:")
    if abs(r_a1) >= 0.5:
        print(f"  KILL — Gate A1 ({r_a1:+.4f} redundant with career win-rate)")
    elif abs(r_a2) >= 0.7:
        print(f"  KILL — Gate A2 ({r_a2:+.4f} redundant with 30d form-rate)")
    elif abs(r_b) < 0.02:
        print(f"  KILL — Gate B ({r_b:+.4f} no signal vs is_top3)")
    elif abs(r_b) > 0.03:
        print(f"  PROCEED — Gate B passes at r={r_b:+.4f}.")
        if abs(r_b) <= abs(r_baseline_30d) + 0.005:
            print(f"  CAVEAT: not materially stronger than the existing 30d "
                  f"baseline (r={r_baseline_30d:+.4f}). Stage 2 (re-rank "
                  f"signal-additivity on burned holdout) is the next gate; "
                  f"don't expect a big jump.")
        else:
            print(f"  ELO signal is +{abs(r_b) - abs(r_baseline_30d):+.4f} above "
                  f"the 30d baseline — promising orthogonal channel.")
        # Recent shipped features for scale:
        print(f"  Reference: sire_top3_rate_on_going shipped at r=+0.077, "
              f"mid_race_pace at r=+0.084.")
    else:
        print(f"  MARGINAL — Gate B at r={r_b:+.4f}; in the noise band.")

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
