"""Measure the cumulative ROI impact of the heavy-fav-skip clause on
BOX TRIO-5 over the last N meetings.

Replays each race twice:
  PROD  — current production: tier + field-size gates only (heavy-fav-skip
          was removed 2026-05-24 after user UNGATE choice; see memory
          heavy-fav-skip-ungated-2026-05-24).
  GATED — deprecated alternative: PROD gates PLUS heavy-fav-skip
          (top1 < $2.50). What we'd save/lose by re-enabling the clause.

Uses the same picks + same real HKJC dividends in both arms. Delta is
the heavy-fav clause's measured P&L impact vs current production.

Pre-2026-05-24 history of this script labelled the arms "OLD" (ungated)
and "NEW" (gated). After production moved to ungated, the labels were
backwards relative to what's deployed — renamed to PROD/GATED to track
the deployment reality.

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
        'prod_stake': 0.0, 'prod_pnl': 0.0, 'prod_n': 0, 'prod_hits': 0,
        'gated_stake': 0.0, 'gated_pnl': 0.0, 'gated_n': 0, 'gated_hits': 0,
        'gated_delta_pnl': 0.0,
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

            # PROD: apply the tier + field-size gates production actually
            # ships (heavy-fav is the only gate the GATED arm adds — see
            # _triple_gate_skip). Before 2026-05-29 the PROD arm wrote
            # every race regardless of gates, so PROD ROI was artificially
            # depressed by SKIP/LOW/small-field bets production never made.
            # The whole impact comparison was off.
            tier = (race.get('conviction_tier') or 'SKIP').upper()
            field = len(horses)
            prod_skips = (tier not in ('EDGE', 'HIGH', 'MED')) or (field < 8)

            e = per_meeting[date_str]
            if not prod_skips:
                e['prod_stake'] += BOX_TRIO5_TOTAL_STAKE
                e['prod_pnl'] += pnl
                e['prod_n'] += 1
                e['prod_hits'] += int(hit)

            # GATED (deprecated alternative): PROD gates + heavy-fav-skip.
            # gated_delta_pnl is +ve when re-enabling the gate WOULD have
            # saved money vs current PROD (i.e., a skipped losing bet).
            skip, reason = _triple_gate_skip(race)
            if skip:
                # Only score the delta against PROD if PROD would have bet.
                if not prod_skips:
                    e['gated_delta_pnl'] += -pnl
                skip_reasons[reason] += 1
            else:
                e['gated_stake'] += BOX_TRIO5_TOTAL_STAKE
                e['gated_pnl'] += pnl
                e['gated_n'] += 1
                e['gated_hits'] += int(hit)

    conn.close()

    if not per_meeting:
        print('No races with TRIO dividends found.', file=sys.stderr)
        return 1

    # Print report
    print(f"\n{'=' * 78}")
    print(f"  BOX TRIO-5 — PROD (ungated) vs GATED (heavy-fav-skip, deprecated)")
    print(f"  {len(per_meeting)} meetings · stake HK${int(BOX_TRIO5_STAKE_PER_COMBO)}"
          f"/combo × {BOX_TRIO5_COMBOS} combos")
    print(f"{'=' * 78}")
    print(f"  {'Meeting':<12}{'PROD bets':>11}{'PROD ROI':>11}"
          f"{'GATED bets':>12}{'GATED ROI':>11}{'$re-gate':>10}")
    print('  ' + '-' * 74)

    tot = defaultdict(float)
    tot_prod_n = tot_gated_n = tot_prod_hits = tot_gated_hits = 0
    for date_str in sorted(per_meeting):
        e = per_meeting[date_str]
        prod_roi = (e['prod_pnl'] / e['prod_stake'] * 100) if e['prod_stake'] else 0
        gated_roi_str = (f"{e['gated_pnl'] / e['gated_stake'] * 100:>+9.1f}%"
                         if e['gated_stake'] else '       —')
        print(f"  {date_str[4:6]}/{date_str[6:8]}    "
              f"{e['prod_n']:>8} {prod_roi:>+9.1f}% "
              f"{e['gated_n']:>11} {gated_roi_str}  {e['gated_delta_pnl']:>+9.0f}")
        for k in ('prod_stake', 'prod_pnl', 'gated_stake', 'gated_pnl', 'gated_delta_pnl'):
            tot[k] += e[k]
        tot_prod_n += e['prod_n']; tot_gated_n += e['gated_n']
        tot_prod_hits += e['prod_hits']; tot_gated_hits += e['gated_hits']

    print('  ' + '-' * 74)
    prod_roi = tot['prod_pnl'] / tot['prod_stake'] * 100 if tot['prod_stake'] else 0
    gated_roi = tot['gated_pnl'] / tot['gated_stake'] * 100 if tot['gated_stake'] else 0
    print(f"  TOTAL      {tot_prod_n:>8} {prod_roi:>+9.1f}% "
          f"{tot_gated_n:>11} {gated_roi:>+9.1f}%  "
          f"{tot['gated_pnl'] - tot['prod_pnl']:>+9.0f}")
    print()
    print(f"  Hit rate:  PROD={tot_prod_hits}/{tot_prod_n} "
          f"({tot_prod_hits/tot_prod_n*100:.1f}%)  →  "
          f"GATED={tot_gated_hits}/{tot_gated_n} "
          f"({tot_gated_hits/tot_gated_n*100:.1f}%)")
    print(f"  Exposure:  PROD=${tot['prod_stake']:.0f}  →  "
          f"GATED=${tot['gated_stake']:.0f}  "
          f"(gate would skip ${tot['prod_stake'] - tot['gated_stake']:.0f})")
    print(f"  Net P&L:   PROD={tot['prod_pnl']:+.0f}  →  "
          f"GATED={tot['gated_pnl']:+.0f}  "
          f"(re-gating impact: {tot['gated_pnl'] - tot['prod_pnl']:+.0f} HKD = "
          f"{gated_roi - prod_roi:+.1f}pp ROI)")
    print()
    print(f"  Heavy-fav clause would skip {sum(skip_reasons.values())} bets.  "
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
