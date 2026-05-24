"""Measure the cumulative ROI impact of the BOX TRIO-5 triple gate
(shipped 2026-05-23) over the last N meetings.

Replays each race twice — once with the OLD "always fires on top-5" rule,
once with the NEW triple-gate (heavy fav / LOW/SKIP conv / small field
skip) — using the same picks + same real HKJC dividends. The delta is
the gate's measured P&L impact on what the user actually bet.

Reads:
  results/race_data_YYYYMMDD_with_results.json   (per-meeting picks + finishers)
  race_dividends table                            (real TRIO dividends per HK$10)

Auto-runs in scripts.post_race after results ingest so the user sees
the cumulative gate impact after every meeting.

Usage:
    python3 -m scripts.gate_impact              # last 8 meetings (default)
    python3 -m scripts.gate_impact --last 12    # last 12
    python3 -m scripts.gate_impact --all        # every available meeting
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from betting.dividends import hkjc_declared_to_unit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / 'results'

# BOX TRIO-5 mechanics — must match dashboard/template.py
BOX_TRIO5_STAKE_PER_COMBO = 10.0  # HK$10 minimum
BOX_TRIO5_COMBOS = 10
BOX_TRIO5_TOTAL_STAKE = BOX_TRIO5_STAKE_PER_COMBO * BOX_TRIO5_COMBOS  # $100


def _trio_div_per1(conn, race_date: str, race_no: int,
                   finishers: list[dict]) -> float | None:
    """Look up TRIO dividend from race_dividends, return per-$1 stake.
    HKJC stores per HK$10; divide by 10."""
    if len(finishers) < 3:
        return None
    nums = sorted(h['number'] for h in finishers[:3])
    combo = ','.join(str(n) for n in nums)
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? "
        "AND race_number=? AND pool='TRIO' AND combination=?",
        (race_date, race_no, combo),
    ).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _triple_gate_skip(race: dict) -> tuple[bool, str]:
    """Return (skip, reason_label) for the F4-style triple gate."""
    horses = race.get('horses', [])
    top1_odds = horses[0].get('win_odds') if horses else 0
    top1_odds = top1_odds or 0
    tier = (race.get('conviction_tier') or 'SKIP').upper()
    field = len(horses)
    if 0 < top1_odds < 2.5:
        return True, 'heavy_fav'
    if tier not in ('EDGE', 'HIGH', 'MED'):
        return True, f'tier={tier}'
    if field < 8:
        return True, 'small_field'
    return False, ''


def _meeting_files(last_n: int | None) -> list[Path]:
    files = sorted(RESULTS_DIR.glob('race_data_*_with_results.json'))
    if last_n is None:
        return files
    return files[-last_n:]


def analyze(last_n: int | None) -> int:
    files = _meeting_files(last_n)
    if not files:
        print('No race_data_*_with_results.json files found.', file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(config.DB_PATH))

    per_meeting = defaultdict(lambda: {
        'old_stake': 0.0, 'old_pnl': 0.0, 'old_n': 0, 'old_hits': 0,
        'new_stake': 0.0, 'new_pnl': 0.0, 'new_n': 0, 'new_hits': 0,
        'gate_saved_pnl': 0.0,
    })
    skip_reasons: dict[str, int] = defaultdict(int)

    for fp in files:
        date_str = fp.stem.replace('race_data_', '').replace('_with_results', '')
        race_date = f'{date_str[0:4]}/{date_str[4:6]}/{date_str[6:8]}'
        payload = json.loads(fp.read_text())
        races = payload['races'] if isinstance(payload, dict) else payload

        for race in races:
            if not race.get('results_available'):
                continue
            horses = race.get('horses', [])
            if len(horses) < 5:
                continue

            finishers = sorted(
                [h for h in horses if (h.get('actual_position') or 0) > 0],
                key=lambda h: h['actual_position'],
            )
            if len(finishers) < 3:
                continue

            top3_actual = {h['name'] for h in finishers[:3]}
            top5_picks = {h['name'] for h in
                          sorted(horses, key=lambda h: h.get('rank', 99))[:5]}
            hit = top3_actual.issubset(top5_picks)

            div = _trio_div_per1(conn, race_date, race['race_number'], finishers)
            if div is None:
                continue  # no dividend data for this race; skip

            pnl = (BOX_TRIO5_STAKE_PER_COMBO * div - BOX_TRIO5_TOTAL_STAKE) \
                  if hit else -BOX_TRIO5_TOTAL_STAKE

            e = per_meeting[date_str]
            # OLD: always fires
            e['old_stake'] += BOX_TRIO5_TOTAL_STAKE
            e['old_pnl'] += pnl
            e['old_n'] += 1
            e['old_hits'] += int(hit)

            # NEW: triple-gated
            skip, reason = _triple_gate_skip(race)
            if skip:
                e['gate_saved_pnl'] += -pnl  # +ve if we avoided a loss
                skip_reasons[reason] += 1
            else:
                e['new_stake'] += BOX_TRIO5_TOTAL_STAKE
                e['new_pnl'] += pnl
                e['new_n'] += 1
                e['new_hits'] += int(hit)

    conn.close()

    if not per_meeting:
        print('No races with TRIO dividends found.', file=sys.stderr)
        return 1

    # Print report
    print(f"\n{'=' * 78}")
    print(f"  BOX TRIO-5 TRIPLE GATE — {len(per_meeting)} meetings "
          f"(stake HK${int(BOX_TRIO5_STAKE_PER_COMBO)}/combo × {BOX_TRIO5_COMBOS} combos)")
    print(f"{'=' * 78}")
    print(f"  {'Meeting':<12}{'OLD bets':>10}{'OLD ROI':>10}"
          f"{'NEW bets':>10}{'NEW ROI':>10}{'$saved':>11}")
    print('  ' + '-' * 74)

    tot = defaultdict(float)
    tot_old_n = tot_new_n = tot_old_hits = tot_new_hits = 0
    for date_str in sorted(per_meeting):
        e = per_meeting[date_str]
        old_roi = (e['old_pnl'] / e['old_stake'] * 100) if e['old_stake'] else 0
        new_roi_str = (f"{e['new_pnl'] / e['new_stake'] * 100:>+9.1f}%"
                       if e['new_stake'] else '       —')
        print(f"  {date_str[4:6]}/{date_str[6:8]}    "
              f"{e['old_n']:>7} {old_roi:>+9.1f}% "
              f"{e['new_n']:>9} {new_roi_str}  {e['gate_saved_pnl']:>+10.0f}")
        for k in ('old_stake', 'old_pnl', 'new_stake', 'new_pnl', 'gate_saved_pnl'):
            tot[k] += e[k]
        tot_old_n += e['old_n']; tot_new_n += e['new_n']
        tot_old_hits += e['old_hits']; tot_new_hits += e['new_hits']

    print('  ' + '-' * 74)
    old_roi = tot['old_pnl'] / tot['old_stake'] * 100 if tot['old_stake'] else 0
    new_roi = tot['new_pnl'] / tot['new_stake'] * 100 if tot['new_stake'] else 0
    print(f"  TOTAL      {tot_old_n:>7} {old_roi:>+9.1f}% "
          f"{tot_new_n:>9} {new_roi:>+9.1f}%  "
          f"{tot['new_pnl'] - tot['old_pnl']:>+10.0f}")
    print()
    print(f"  Hit rate:  OLD={tot_old_hits}/{tot_old_n} "
          f"({tot_old_hits/tot_old_n*100:.1f}%)  →  "
          f"NEW={tot_new_hits}/{tot_new_n} "
          f"({tot_new_hits/tot_new_n*100:.1f}%)")
    print(f"  Exposure:  OLD=${tot['old_stake']:.0f}  →  "
          f"NEW=${tot['new_stake']:.0f}  "
          f"(gate removed ${tot['old_stake'] - tot['new_stake']:.0f})")
    print(f"  Net P&L:   OLD={tot['old_pnl']:+.0f}  →  "
          f"NEW={tot['new_pnl']:+.0f}  "
          f"(improvement: {tot['new_pnl'] - tot['old_pnl']:+.0f} HKD = "
          f"{new_roi - old_roi:+.1f}pp ROI)")
    print()
    print(f"  Gate skipped {sum(skip_reasons.values())} bets.  "
          f"Reason mix: {dict(skip_reasons)}")
    print()
    print(f"  Caveat: data is historical (burned holdout). Forward signal may differ.")
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--last', type=int, default=8,
                   help='Restrict to last N meetings with merged-results data '
                        '(default: 8)')
    g.add_argument('--all', action='store_true',
                   help='Use every available meeting (overrides --last)')
    args = p.parse_args()
    return analyze(None if args.all else args.last)


if __name__ == '__main__':
    sys.exit(main())
