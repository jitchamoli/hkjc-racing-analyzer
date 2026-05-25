"""Cheap diagnostic — TWO variants of jockey ELO (roadmap stage 1 of 4).

Round 1 (diagnose_jockey_elo.py) ran the WIN-anchored ELO and got
Gate B=+0.0210 (MARGINAL). The smoking gun was Teetan / Chadwick at
the bottom of the table — top HK jockeys penalised for riding lots
of favourites (win/loss binary inflates chalk-rider losses).

This script tests two design variants per user spec (2026-05-25):

  V1 PLACE-ANCHORED ELO
    expected = market-implied top-3 prob (Harville/Plackett-Luce
               from per-race normalised win-odds — HKJC never
               published historical place odds)
    actual   = is_top3

  V2 PERFORMANCE-RATING ELO
    expected = market-implied normalised win prob
    actual   = 1 - (finish_position - 1) / (field_size - 1)
               (partial credit; 2nd in 12-horse field = 0.91)

Both:
  • single chronological pass — strict leak-free
  • pre-race rating recorded BEFORE post-race update
  • 365-day half-life decay toward 1500 between rides
  • K=32

Three gates (same as round 1):
  A1  redundancy with jockey career win-rate    |r| > 0.5  → KILL
  A2  redundancy with jockey 30d form-rate      |r| > 0.7  → KILL
  B   signal vs is_top3                         |r| < 0.02 → KILL
                                                |r| > 0.03 → PROCEED

Decision rule (per user 2026-05-25):
  • If NEITHER variant clears r ≥ +0.030 AND has sane top/bottom
    sanity → KILL jockey ELO cleanly, move to top-4 ranker.
  • If ONE clears → keep SHADOW-ONLY, test against frozen bet_ticket
    REPLAY ROI + top-4 capture before any production retrain.
  • DO NOT add to production features regardless.
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

K_FACTOR = 32.0
START_ELO = 1500.0
HALF_LIFE_DAYS = 365.0
DECAY_RATE = math.log(2) / HALF_LIFE_DAYS

MIN_OBS_FOR_GATE = 20
WINDOW_30D_DAYS = 30


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


def harville_top3(win_probs: list[float]) -> list[float]:
    """Plackett-Luce top-3 inclusion probability for each horse.

    Pr(i ∈ top 3) = Pr(i=1st) + Pr(i=2nd) + Pr(i=3rd)
                  = p_i
                  + Σ_{j≠i}     p_j · p_i / (1-p_j)
                  + Σ_{j≠i, k∉{i,j}} p_j · p_k · p_i / ((1-p_j)(1-p_j-p_k))

    Assumes win_probs already normalised (Σ = 1).
    """
    n = len(win_probs)
    top3 = [0.0] * n
    for i in range(n):
        pi = win_probs[i]
        # 1st place
        top3[i] += pi
        # 2nd place
        for j in range(n):
            if j == i:
                continue
            pj = win_probs[j]
            denom1 = 1.0 - pj
            if denom1 <= 0:
                continue
            top3[i] += pj * pi / denom1
            # 3rd place
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = win_probs[k]
                denom2 = denom1 - pk
                if denom2 <= 0:
                    continue
                top3[i] += pj * pk * pi / (denom1 * denom2)
    return top3


# ---------------------------------------------------------------------------
# Per-variant ELO state container — keeps both variants' state isolated
# while sharing the chronological scan, baselines, and gates.
# ---------------------------------------------------------------------------
class EloState:
    def __init__(self, name: str):
        self.name = name
        self.elo: dict[str, float] = defaultdict(lambda: START_ELO)
        self.last_date: dict[str, datetime] = {}
        self.xs_elo: list[float] = []
        self.xs_career_wr: list[float] = []
        self.xs_30d_wr: list[float] = []
        self.ys_top3: list[float] = []

    def decay_to(self, jky: str, race_date: datetime) -> float:
        prev = self.last_date.get(jky)
        if prev is not None:
            days = max(0, (race_date - prev).days)
            decay = math.exp(-DECAY_RATE * days)
            self.elo[jky] = 1500.0 + (self.elo[jky] - 1500.0) * decay
        return self.elo[jky]

    def record(self, pre_elo: float, career_wr: float, wr30: float,
               is_top3: int):
        self.xs_elo.append(pre_elo)
        self.xs_career_wr.append(career_wr)
        self.xs_30d_wr.append(wr30)
        self.ys_top3.append(float(is_top3))

    def update(self, jky: str, race_date: datetime,
               actual: float, expected: float):
        self.elo[jky] += K_FACTOR * (actual - expected)
        self.last_date[jky] = race_date


def report(state: EloState, career_rides: dict[str, int],
           career_wins: dict[str, int]) -> tuple[float, float, float]:
    """Print sanity check + gate results for one ELO variant. Returns
    (r_a1, r_a2, r_b) for the decision matrix."""
    print(f"\n{'='*70}")
    print(f"VARIANT: {state.name}")
    print(f"{'='*70}")
    if len(state.ys_top3) < 5000:
        print("INSUFFICIENT DATA")
        return 0.0, 0.0, 0.0

    print(f"  Distribution of pre-race ELO (when n≥{MIN_OBS_FOR_GATE}):")
    print(f"    mean={statistics.mean(state.xs_elo):.1f}  "
          f"sd={statistics.stdev(state.xs_elo):.1f}  "
          f"min={min(state.xs_elo):.1f}  max={max(state.xs_elo):.1f}")

    eligible_finals = sorted(
        ((j, state.elo[j], career_rides[j], career_wins[j])
         for j in state.elo if career_rides[j] >= 100),
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

    r_b  = pearson(state.xs_elo, state.ys_top3)
    r_a1 = pearson(state.xs_elo, state.xs_career_wr)
    r_a2 = pearson(state.xs_elo, state.xs_30d_wr)
    print(f"\n  n = {len(state.ys_top3):,}")
    print(f"  Gate A1 (vs jockey career win-rate)   r = {r_a1:+.4f}  "
          f"{'PASS' if abs(r_a1) < 0.5 else 'KILL'}")
    print(f"  Gate A2 (vs jockey 30d form-rate)     r = {r_a2:+.4f}  "
          f"{'PASS' if abs(r_a2) < 0.7 else 'KILL'}")
    print(f"  Gate B  (vs is_top3)                  r = {r_b:+.4f}  "
          f"{'PASS' if abs(r_b) > 0.03 else ('MARGINAL' if abs(r_b) > 0.02 else 'KILL')}")
    return r_a1, r_a2, r_b


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))

    print("Step 1: scanning race_results in chronological order...")
    rows = conn.execute("""
        SELECT race_date, race_number, jockey, win_odds,
               is_winner, is_top3, finish_position
        FROM race_results
        WHERE jockey IS NOT NULL AND jockey != ''
          AND win_odds IS NOT NULL AND win_odds > 0
          AND is_top3 IS NOT NULL
          AND finish_position > 0
        ORDER BY race_date, race_number
    """).fetchall()
    print(f"  {len(rows):,} rows with jockey + win_odds + finish_position")

    races: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for r in rows:
        races[(r[0], r[1])].append(r)
    sorted_race_keys = sorted(races.keys())
    print(f"  {len(sorted_race_keys):,} unique races")

    # Shared baseline state (used for Gate A1/A2 baselines + the
    # r(jockey_career_wr, is_top3) / r(jockey_30d_wr, is_top3) reference numbers).
    career_rides: dict[str, int] = defaultdict(int)
    career_wins: dict[str, int] = defaultdict(int)
    recent_30d: dict[str, deque] = defaultdict(deque)

    # Two variant states evolved in lockstep
    v1 = EloState("V1 — PLACE-ANCHORED (expected=top3 via Harville, actual=is_top3)")
    v2 = EloState("V2 — PERFORMANCE-RATING (expected=norm-win, actual=1-rank/field)")

    # Baseline references (career_wr, 30d_wr) → is_top3 — recomputed on the
    # SAME sample (only rows where career_rides >= 20) so the comparison
    # is apples-to-apples.
    xs_baseline_career: list[float] = []
    xs_baseline_30d: list[float] = []
    ys_baseline: list[float] = []

    skipped_low_obs = 0

    for race_key in sorted_race_keys:
        race_rows = races[race_key]
        race_date_str = race_key[0]
        try:
            race_date = parse_date(race_date_str)
        except Exception:
            continue

        # Per-race market structure
        raw_p = [1.0 / float(r[3]) for r in race_rows]
        total_p = sum(raw_p)
        if total_p <= 0:
            continue
        win_probs = [p / total_p for p in raw_p]
        top3_probs = harville_top3(win_probs)
        field_size = len(race_rows)
        # field_size of 1 makes the perf-rating denominator blow up — skip
        if field_size < 2:
            continue

        # Decay pre-race ratings + record samples for both variants
        for i, row in enumerate(race_rows):
            _, _, jky, _, is_win, is_top3, fp = row
            pre_v1 = v1.decay_to(jky, race_date)
            pre_v2 = v2.decay_to(jky, race_date)

            dq = recent_30d[jky]
            cutoff = race_date.toordinal() - WINDOW_30D_DAYS
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            n30 = len(dq)
            wr30 = (sum(int(w) for _, w in dq) / n30) if n30 else 0.0
            n_career = career_rides[jky]
            wr_career = (career_wins[jky] / n_career) if n_career else 0.0

            if n_career >= MIN_OBS_FOR_GATE:
                v1.record(pre_v1, wr_career, wr30, is_top3 or 0)
                v2.record(pre_v2, wr_career, wr30, is_top3 or 0)
                xs_baseline_career.append(wr_career)
                xs_baseline_30d.append(wr30)
                ys_baseline.append(float(is_top3 or 0))
            else:
                skipped_low_obs += 1

        # Post-race updates for both variants
        for (row, mp_win, mp_top3) in zip(race_rows, win_probs, top3_probs):
            _, _, jky, _, is_win, is_top3, fp = row
            # V1 — expected = top3 market prob, actual = is_top3
            v1.update(jky, race_date,
                      actual=float(is_top3 or 0),
                      expected=mp_top3)
            # V2 — expected = win market prob, actual = 1-(rank-1)/(field-1)
            rank_score = 1.0 - (fp - 1) / (field_size - 1)
            rank_score = max(0.0, min(1.0, rank_score))
            v2.update(jky, race_date,
                      actual=rank_score,
                      expected=mp_win)
            # Shared career/30d counters
            career_rides[jky] += 1
            if is_win:
                career_wins[jky] += 1
            recent_30d[jky].append((race_date.toordinal(), bool(is_win)))

    print(f"  Rows scored for correlation: {len(v1.ys_top3):,}  "
          f"({skipped_low_obs:,} skipped — jockey had <{MIN_OBS_FOR_GATE} prior rides)")
    print(f"  Unique jockeys seen:        {len(v1.elo):,}")

    # Run both gates side-by-side
    a1_v1, a2_v1, b_v1 = report(v1, career_rides, career_wins)
    a1_v2, a2_v2, b_v2 = report(v2, career_rides, career_wins)

    # Apples-to-apples baselines on the SAME sample
    r_baseline_career = pearson(xs_baseline_career, ys_baseline)
    r_baseline_30d    = pearson(xs_baseline_30d,    ys_baseline)
    print(f"\n{'='*70}")
    print("BASELINES (same sample, for scale)")
    print(f"{'='*70}")
    print(f"  r(jockey_career_wr, is_top3) = {r_baseline_career:+.4f}")
    print(f"  r(jockey_30d_wr,    is_top3) = {r_baseline_30d:+.4f}")
    print(f"  Round 1 (win-anchored ELO):  r = +0.0210  (MARGINAL)")
    print(f"  Reference shipped features:  sire_top3 +0.0772, "
          f"mid_race_pace +0.0838")

    # Decision matrix per user 2026-05-25 spec
    print(f"\n{'='*70}")
    print("DECISION MATRIX")
    print(f"{'='*70}")
    v1_pass = abs(b_v1) > 0.03 and abs(a1_v1) < 0.5 and abs(a2_v1) < 0.7
    v2_pass = abs(b_v2) > 0.03 and abs(a1_v2) < 0.5 and abs(a2_v2) < 0.7
    print(f"  V1 place-anchored:       Gate B = {b_v1:+.4f}  "
          f"{'PASS' if v1_pass else 'FAIL'}")
    print(f"  V2 performance-rating:   Gate B = {b_v2:+.4f}  "
          f"{'PASS' if v2_pass else 'FAIL'}")

    print(f"\nVERDICT:")
    if not v1_pass and not v2_pass:
        print(f"  KILL jockey-ELO cleanly. Neither variant cleared r ≥ +0.030 "
              f"vs is_top3. Move to top-4-set-aware ranker (roadmap stage 3) "
              f"as the next architectural swing.")
        return 0
    winner = max([("V1 place-anchored", b_v1, v1_pass),
                  ("V2 performance-rating", b_v2, v2_pass)],
                 key=lambda t: abs(t[1]) if t[2] else 0.0)
    name, r, _ = winner
    print(f"  {name} clears with Gate B = {r:+.4f}.")
    print(f"  Per user 2026-05-25 rule: SHADOW-ONLY. Set up offline scoring")
    print(f"  against frozen bet_ticket REPLAY ROI + top-4 capture before")
    print(f"  ANY production retrain. Do NOT add to production features yet.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
