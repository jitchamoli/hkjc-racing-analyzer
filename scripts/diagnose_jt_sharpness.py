"""Self-computable J×T sharpness feature — leak-free alternative to the
API-based jt_ae diagnostic (which suffered from lifetime contamination).

Walks race_results in chronological (race_date, race_number) order,
accumulates per-(jockey, trainer) running counters [rides, wins,
sum_market_prob] AS WE GO. For each row the feature is computed from
CURRENT (prior-state) counters → strictly leak-free in a single pass,
no per-row before_date subqueries needed.

a/e definition (same as theracingapi.com):
    actual    = wins
    expected  = sum of (1/win_win_odds)        # market-implied win count
    a/e       = actual / expected

Bayesian-smoothed toward neutral (ae=1.0):
    ae_smooth = (wins + α) / (expected + α)
with α=20 (moderate prior — combos with <20 prior rides anchor near 1.0,
heavy combos like Purton×Shum at 500+ rides barely move from raw).

Three gates (same as diagnose_draw_style.py + diagnose_sire_going.py):
  A1  redundancy with jockey win-rate (the easy baseline)  |r| > 0.5  → KILL
  A2  redundancy with raw J×T win-rate                     |r| > 0.7  → KILL
  B   signal vs is_top3                                    |r| < 0.02 → KILL
                                                           |r| > 0.03 → PROCEED
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

ALPHA = 20.0             # prior strength in "rides" units, anchor at ae=1.0
MIN_OBS_FOR_GATE = 20    # require this many prior rides before counting a row
                         # in correlation; below this, ae≈1.0 and just dilutes


def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))

    print(f"Step 1: scanning race_results in chronological order...")
    rows = conn.execute("""
        SELECT race_date, race_number, jockey, trainer, win_odds, is_winner, is_top3
        FROM race_results
        WHERE jockey IS NOT NULL AND jockey != ''
          AND trainer IS NOT NULL AND trainer != ''
          AND win_odds IS NOT NULL AND win_odds > 0
          AND is_top3 IS NOT NULL
        ORDER BY race_date, race_number
    """).fetchall()
    print(f"  {len(rows):,} rows with jockey + trainer + win_odds + is_top3")

    # Running counters
    jt_counts  = defaultdict(lambda: [0, 0, 0.0])   # (jky,trn) → [rides, wins, sum_p]
    jky_counts = defaultdict(lambda: [0, 0])         # jky → [rides, wins]

    xs_jt_ae, xs_jt_wr, xs_jky_wr, ys = [], [], [], []
    skipped_low_obs = 0
    for race_date, race_no, jky, trn, win_odds, is_win, is_top3 in rows:
        market_p = 1.0 / float(win_odds)
        jt_key = (jky, trn)
        jt_rides, jt_wins, jt_sum_p = jt_counts[jt_key]
        jky_rides, jky_wins = jky_counts[jky]

        if jt_rides >= MIN_OBS_FOR_GATE:
            ae_smooth   = (jt_wins + ALPHA) / (jt_sum_p + ALPHA)
            jt_raw_wr   = jt_wins / jt_rides if jt_rides else 0.0
            jky_raw_wr  = jky_wins / jky_rides if jky_rides else 0.0
            xs_jt_ae.append(ae_smooth)
            xs_jt_wr.append(jt_raw_wr)
            xs_jky_wr.append(jky_raw_wr)
            ys.append(float(is_top3 or 0))
        else:
            skipped_low_obs += 1

        # Update counters AFTER recording (strict leak-free)
        jt_rides += 1
        jt_sum_p += market_p
        if is_win:
            jt_wins += 1
        jt_counts[jt_key] = [jt_rides, jt_wins, jt_sum_p]
        jky_rides += 1
        if is_win:
            jky_wins += 1
        jky_counts[jky] = [jky_rides, jky_wins]

    print(f"  Rows scored for correlation: {len(ys):,}  "
          f"({skipped_low_obs:,} skipped — combo had <{MIN_OBS_FOR_GATE} prior rides)")
    print(f"  Unique J×T combos seen:     {len(jt_counts):,}")
    eligible_combos = sum(1 for c in jt_counts.values() if c[0] >= MIN_OBS_FOR_GATE)
    print(f"  Combos that ever crossed n≥{MIN_OBS_FOR_GATE}: {eligible_combos:,}")

    if len(ys) < 5000:
        print("INSUFFICIENT DATA")
        return 1

    # Distribution sanity
    import statistics
    print(f"\n  Distribution of ae_smooth (when n≥{MIN_OBS_FOR_GATE}):")
    print(f"    mean={statistics.mean(xs_jt_ae):.3f}  "
          f"sd={statistics.stdev(xs_jt_ae):.3f}  "
          f"min={min(xs_jt_ae):.3f}  max={max(xs_jt_ae):.3f}")
    print(f"  Distribution of is_top3:")
    print(f"    mean={statistics.mean(ys):.3f}  (base rate)")

    print("\n" + "=" * 70)
    print("CANDIDATE: jt_market_sharpness (leak-free self-computed a/e)")
    print("=" * 70)
    r_b   = pearson(xs_jt_ae, ys)
    r_a1  = pearson(xs_jt_ae, xs_jky_wr)
    r_a2  = pearson(xs_jt_ae, xs_jt_wr)
    print(f"  n = {len(ys):,}")
    print(f"  Gate A1 (vs jockey win-rate)   r = {r_a1:+.4f}  "
          f"{'PASS' if abs(r_a1) < 0.5 else 'KILL'}")
    print(f"  Gate A2 (vs J×T raw win-rate)  r = {r_a2:+.4f}  "
          f"{'PASS' if abs(r_a2) < 0.7 else 'KILL'}")
    print(f"  Gate B  (vs is_top3)           r = {r_b:+.4f}  "
          f"{'PASS' if abs(r_b) > 0.03 else ('MARGINAL' if abs(r_b) > 0.02 else 'KILL')}")
    print()
    print(f"  Baselines for scale:")
    print(f"    r(jockey_win_rate, is_top3)  = {pearson(xs_jky_wr, ys):+.4f}")
    print(f"    r(jt_raw_winrate,  is_top3)  = {pearson(xs_jt_wr,  ys):+.4f}")

    print("\nVERDICT:")
    if abs(r_a1) >= 0.5:
        print("  KILL — Gate A1 (redundant with jockey baseline)")
    elif abs(r_a2) >= 0.7:
        print("  KILL — Gate A2 (redundant with raw J×T win-rate)")
    elif abs(r_b) < 0.02:
        print("  KILL — Gate B (no signal vs outcome)")
    elif abs(r_b) > 0.03:
        # Compare against API-leaked Stage 1 number (was +0.033)
        delta_vs_leak = (r_b - 0.033)
        print(f"  PROCEED — Gate B passes at r={r_b:+.4f}.")
        if r_b < 0.04:
            print(f"  CAVEAT: only marginally above 0.03 threshold; recent shipped "
                  f"features (sire_top3 at +0.077, mid_race_pace at +0.084) were "
                  f"~2× stronger. Stage 2 (re-rank) is the next gate.")
        if abs(delta_vs_leak) > 0.005:
            print(f"  vs the leaked API a/e (r=+0.0333), this is {delta_vs_leak:+.4f}")
            if delta_vs_leak > 0:
                print("    → leak-free signal is STRONGER than the leaked version (good sign).")
            else:
                print("    → leak-free signal is WEAKER, suggesting the API a/e leak was "
                      "indeed inflating Stage 1.")
    else:
        print(f"  MARGINAL — Gate B at r={r_b:+.4f}; in noise band.")

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
