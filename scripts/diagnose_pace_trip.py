"""Cheap diagnostic for GPT-5.5 audit Finding #5 — 5 pace/trip flags.

Per feedback_diagnostic_to_ship_pipeline: new features must pass cheap
correlation gates BEFORE re-rank/A/B/ship. Pace features have a 100%
kill rate so far in HK (closing_position_gain, jockey_book_quality,
race_shape) — structural reason: small fields + jockey-controlled
tempo + tight markets price these in pre-race. This diagnostic tests
GPT-5.5's 5 candidates at once.

Flags tested (per horse in each race):
  1. lone_speed              — horse is Leader AND only Leader in race
  2. in_hot_pace_race        — race has ≥3 Leaders (pace-collapse setup)
  3. buried_rail_closer      — horse is Closer AND draw 1-3
  4. wide_no_cover           — draw ≥9 AND distance ≥1200
  5. pace_collapse_candidate — Closer AND race has ≥3 Leaders

Gates per flag:
  A1: r(flag, is_winner) on full DB — must |r| ≥ 0.02 to survive
  A2: r(flag, is_top3)   on full DB — must |r| ≥ 0.02 to survive
  B : |r(flag, proxy)|  < 0.7 vs early_speed_proxy, field_rel_draw_proxy,
      late_kick_proxy (redundant flags die here)

Verdict: PASS only if A1 ≥ 0.02 AND A2 ≥ 0.02 AND no B collision ≥ 0.7.

Usage:
    python3 -m scripts.diagnose_pace_trip
"""
from __future__ import annotations
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import config

A_GATE = 0.02
B_GATE = 0.70


def _parse_running_position(rp: str) -> list[int]:
    """Parse '1 1 1' or '111' format. Returns [early, mid, late] or [].
    Same shape gotcha as DB notes: two formats exist."""
    if not rp:
        return []
    rp = rp.strip()
    if ' ' in rp:
        try:
            return [int(x) for x in rp.split()][:3]
        except ValueError:
            return []
    if rp.isdigit() and len(rp) in (3, 6):
        try:
            if len(rp) == 3:
                return [int(rp[0]), int(rp[1]), int(rp[2])]
            return [int(rp[0:2]), int(rp[2:4]), int(rp[4:6])]
        except ValueError:
            return []
    return []


def _classify(early: int) -> str:
    if early <= 2:
        return 'Leader'
    if early <= 5:
        return 'Stalker'
    if early <= 9:
        return 'Midfield'
    return 'Closer'


def _pearson(xs: list[float], ys: list[int]) -> float:
    n = len(xs)
    if n < 30:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0 or dy2 == 0:
        return 0.0
    return num / ((dx2 ** 0.5) * (dy2 ** 0.5))


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))
    rows = conn.execute("""
        SELECT race_date, race_number, horse_name, draw, distance,
               finish_position, is_winner, is_top3, running_position
        FROM race_results
        WHERE running_position != ''
          AND draw > 0
          AND distance > 0
    """).fetchall()
    print(f"Loaded {len(rows):,} race_results rows with running_position + draw + distance")

    races = defaultdict(list)
    for r in rows:
        rp = _parse_running_position(r[8])
        if len(rp) < 1:
            continue
        races[(r[0], r[1])].append({
            'horse_name': r[2],
            'draw': r[3],
            'distance': r[4],
            'finish_position': r[5],
            'is_winner': int(r[6] or 0),
            'is_top3': int(r[7] or 0),
            'early': rp[0],
            'style': _classify(rp[0]),
        })
    print(f"Grouped into {len(races):,} races")

    flag_vals: dict[str, list[float]] = {k: [] for k in (
        'lone_speed', 'in_hot_pace_race', 'buried_rail_closer',
        'wide_no_cover', 'pace_collapse_candidate',
    )}
    proxy_vals: dict[str, list[float]] = {k: [] for k in (
        'early_speed_proxy', 'field_rel_draw_proxy', 'late_kick_proxy',
    )}
    y_win: list[int] = []
    y_top3: list[int] = []

    for (_, _), horses in races.items():
        n = len(horses)
        n_leaders = sum(1 for h in horses if h['style'] == 'Leader')
        hot_pace = n_leaders >= 3
        for h in horses:
            flag_vals['lone_speed'].append(1.0 if (h['style'] == 'Leader' and n_leaders == 1) else 0.0)
            flag_vals['in_hot_pace_race'].append(1.0 if hot_pace else 0.0)
            flag_vals['buried_rail_closer'].append(1.0 if (h['style'] == 'Closer' and h['draw'] <= 3) else 0.0)
            flag_vals['wide_no_cover'].append(1.0 if (h['draw'] >= 9 and h['distance'] >= 1200) else 0.0)
            flag_vals['pace_collapse_candidate'].append(1.0 if (h['style'] == 'Closer' and hot_pace) else 0.0)
            # Proxies for Gate B redundancy check
            proxy_vals['early_speed_proxy'].append(1.0 / max(1, h['early']))
            proxy_vals['field_rel_draw_proxy'].append(
                1.0 - (h['draw'] - 1) / max(1, n - 1)
            )
            # late_kick proxy: improvement from early to finish position
            proxy_vals['late_kick_proxy'].append(
                (h['early'] - h['finish_position']) / max(1, n) if h['finish_position'] else 0.0
            )
            y_win.append(h['is_winner'])
            y_top3.append(h['is_top3'])

    print(f"\nFlag base rates ({len(y_win):,} rows):")
    for name, vals in flag_vals.items():
        rate = sum(vals) / len(vals)
        print(f"  {name:30s} {rate*100:.2f}%")

    print(f"\n{'='*92}")
    print(f"  PACE/TRIP DIAGNOSTIC — A1/A2/B gates  "
          f"(A pass: |r|≥{A_GATE}, B kill: |r|≥{B_GATE})")
    print(f"{'='*92}")
    print(f"  {'flag':<30}{'A1 (win)':>10}{'A2 (top3)':>12}"
          f"{'best collision':>30}{'verdict':>10}")
    print("  " + "-" * 90)

    for name, vals in flag_vals.items():
        a1 = _pearson(vals, y_win)
        a2 = _pearson(vals, y_top3)
        best_collision_r = 0.0
        best_collision_name = ''
        for pname, pvals in proxy_vals.items():
            r = _pearson(vals, [int(round(v)) for v in pvals]) if False else _pearson(vals, list(map(int, [round(v * 100) for v in pvals])))
            # Easier: compute r against the proxy float series directly.
            r = _pearson_float(vals, pvals)
            if abs(r) > abs(best_collision_r):
                best_collision_r = r
                best_collision_name = pname
        a1_ok = abs(a1) >= A_GATE
        a2_ok = abs(a2) >= A_GATE
        b_ok = abs(best_collision_r) < B_GATE
        verdict = 'PASS' if (a1_ok and a2_ok and b_ok) else 'KILL'
        coll_str = f"{best_collision_name}={best_collision_r:+.3f}"
        print(f"  {name:<30}{a1:>+10.4f}{a2:>+12.4f}"
              f"{coll_str:>30}{verdict:>10}")

    print()
    print("  Verdict rules: KILL on A1<0.02 OR A2<0.02 OR collision ≥0.70.")
    print("  Per feedback_diagnostic_to_ship_pipeline: only PASS flags advance")
    print("  to re-rank signal-additivity testing (next stage if any survive).")
    conn.close()
    return 0


def _pearson_float(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 30:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0 or dy2 == 0:
        return 0.0
    return num / ((dx2 ** 0.5) * (dy2 ** 0.5))


if __name__ == '__main__':
    sys.exit(main())
