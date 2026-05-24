"""Paper-track a "longshot-inclusive TRIO" experiment across meetings.

Motivation (2026-05-20): a friend hit a ~120x TRIO that included a rank-7
$51 horse our top-5 box deliberately excludes. This logs — ON PAPER ONLY,
nothing wired into the dashboard or WA message — what WOULD have happened if
we boxed the model's top 5 PLUS one *steamed* rank 6-8 longshot (smart-money
longshot, not a random bolter). The thesis: market support on a longshot the
model under-rates may rescue TRIOs that the default top-5 box misses.

Selection rule (per race):
  - base   = model top 5 (ranks 1-5)
  - add    = the most-steamed horse among ranks 6-8 (steam_move.is_steam),
             tie-broken by largest % shortening. None steamed -> NO BET.
  - box    = 6 horses -> C(6,3)=20 TRIO combos.

ROI model (HKJC TRIO dividends are declared per HK$10 unit):
  - longshot box: 20 combos x $10 = $200 stake; payout = TRIO dividend if the
    actual top-3 set is inside the 6 numbers.
  - baseline TRIO-5: 10 combos x $10 = $100 stake (same races, for comparison).

Ship gate (mirrors [[n6_validation_rule]] + eval_swap_rule discipline):
  - < 6 meetings with a qualifying bet      -> INSUFFICIENT_DATA (keep paper-only)
  - >= 6 meetings AND box ROI > 0 AND beats baseline ROI -> SHIP candidate
  - >= 6 meetings AND box ROI <= 0           -> KILL

Usage:
    python3 scripts/track_longshot_trio.py 2026/05/20   # log one meeting
    python3 scripts/track_longshot_trio.py --backfill-all
    python3 scripts/track_longshot_trio.py --report
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.forward_common import bayesian_verdict

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / 'tracking_data' / 'longshot_trio_log.csv'
DB_PATH = REPO_ROOT / 'tracking_data' / 'horse_history.db'

LOG_HEADERS = [
    'race_date', 'race_number', 'variant',
    'longshot_num', 'longshot_name', 'longshot_rank', 'longshot_odds', 'steam_pct',
    'cold_num', 'cold_name', 'cold_rank', 'cold_odds', 'drift_pct',
    'box6_nums', 'actual_top3',
    'trio5_hit', 'box6_hit', 'rescued', 'longshot_finish',
]

# Variant 1 (calc_hot): top-5 + 1 steamed rank 6-8 → BOX TRIO-6 (20 combos).
LONGSHOT_RANKS = (6, 7, 8)

# Variant 2 (mix_hot_cold, 2026-05-24): top-1 + 1 most-steamed in rank 6-12
# + 1 most-drifted in rank 6-12 → STRAIGHT TRIO (1 combo). $10 lottery to
# complement the 20-combo calc_hot box. Mirrors dashboard's _bolterTrio().
MIX_LONGSHOT_RANKS = tuple(range(6, 13))


def _steam_pct(horse: dict) -> int:
    """Absolute % shortening from the steam label, e.g. 'STEAM -42% (...)' -> 42."""
    s = horse.get('steam_move') or {}
    m = re.search(r'-(\d+)%', s.get('label', '') if isinstance(s, dict) else '')
    return int(m.group(1)) if m else 0


def _movement_pct(horse: dict) -> float:
    """odds_movement.change_pct (signed). >0 = shortened, <0 = drifted."""
    om = horse.get('odds_movement') or {}
    return float(om.get('change_pct') or 0)


def _select(horses_sorted: list) -> dict | None:
    """Variant 1 (calc_hot): most-steamed rank 6-8 horse."""
    if len(horses_sorted) < max(LONGSHOT_RANKS):
        return None
    candidates = []
    for rank in LONGSHOT_RANKS:
        h = horses_sorted[rank - 1]
        s = h.get('steam_move') or {}
        if isinstance(s, dict) and s.get('is_steam'):
            candidates.append((rank, h))
    if not candidates:
        return None
    rank, horse = max(candidates, key=lambda rh: _steam_pct(rh[1]))
    return {'rank': rank, 'horse': horse, 'steam_pct': _steam_pct(horse)}


def _select_hot_in_range(horses_sorted: list, ranks: tuple[int, ...]) -> dict | None:
    """Most-steamed (largest positive change_pct) horse in given rank range."""
    cap = min(max(ranks), len(horses_sorted))
    candidates = []
    for rank in ranks:
        if rank > cap:
            break
        h = horses_sorted[rank - 1]
        s = h.get('steam_move') or {}
        if isinstance(s, dict) and s.get('is_steam'):
            candidates.append((rank, h))
    if not candidates:
        return None
    rank, horse = max(candidates, key=lambda rh: _movement_pct(rh[1]))
    return {'rank': rank, 'horse': horse, 'steam_pct': int(_movement_pct(horse))}


def _select_cold_in_range(horses_sorted: list, ranks: tuple[int, ...],
                          exclude_nums: set[int] | None = None) -> dict | None:
    """Most-drifted (most negative change_pct) horse in given rank range.

    Drift threshold: change_pct < -10% (mirrors odds_tracker 'Mild drift' floor).
    Skips horses already chosen by the hot selector.
    """
    cap = min(max(ranks), len(horses_sorted))
    exclude_nums = exclude_nums or set()
    candidates = []
    for rank in ranks:
        if rank > cap:
            break
        h = horses_sorted[rank - 1]
        if h.get('number') in exclude_nums:
            continue
        pct = _movement_pct(h)
        if pct < -10:
            candidates.append((rank, h, pct))
    if not candidates:
        return None
    rank, horse, pct = min(candidates, key=lambda rhp: rhp[2])
    return {'rank': rank, 'horse': horse, 'drift_pct': abs(int(pct))}


def _empty_row_fields() -> dict:
    """Default empty values for unused columns (variant-dependent)."""
    base = {k: '' for k in (
        'longshot_num', 'longshot_name', 'longshot_rank', 'longshot_odds',
        'steam_pct', 'cold_num', 'cold_name', 'cold_rank', 'cold_odds',
        'drift_pct', 'box6_nums', 'actual_top3', 'longshot_finish',
    )}
    base.update({'trio5_hit': 0, 'box6_hit': 0, 'rescued': 0})
    return base


def _box_hit(box_nums: list[int], actual_top3: list) -> bool:
    return bool(actual_top3) and all(n in box_nums for n in actual_top3)


def _rows_for_meeting(races: list, date: str) -> list[dict]:
    """One row per (race, qualifying variant) — 0, 1, or 2 rows per race."""
    rows = []
    for r in races:
        if not r.get('results_available'):
            continue
        top3 = r.get('actual_top3') or []
        if not top3:
            continue
        horses_sorted = sorted(r['horses'], key=lambda h: h.get('rank') or 99)

        # Variant 1: calc_hot (top-5 + steamed rank 6-8 → BOX TRIO-6)
        if len(horses_sorted) >= max(LONGSHOT_RANKS):
            sel = _select(horses_sorted)
            if sel:
                base5 = [h['number'] for h in horses_sorted[:5]]
                ls = sel['horse']
                box6 = base5 + [ls['number']]
                trio5_hit = _box_hit(base5, top3)
                box6_hit = _box_hit(box6, top3)
                row = _empty_row_fields()
                row.update({
                    'race_date': date,
                    'race_number': r.get('race_number'),
                    'variant': 'calc_hot',
                    'longshot_num': ls['number'],
                    'longshot_name': ls.get('name', ''),
                    'longshot_rank': sel['rank'],
                    'longshot_odds': ls.get('win_odds') or '',
                    'steam_pct': sel['steam_pct'],
                    'box6_nums': '|'.join(str(n) for n in box6),
                    'actual_top3': '|'.join(str(n) for n in top3),
                    'trio5_hit': int(trio5_hit),
                    'box6_hit': int(box6_hit),
                    'rescued': int(box6_hit and not trio5_hit),
                    'longshot_finish': ls.get('actual_position') or '',
                })
                rows.append(row)

        # Variant 2: mix_hot_cold (top-1 + steamed 6-12 + drifted 6-12 → STRAIGHT TRIO)
        if len(horses_sorted) >= 6:
            top1 = horses_sorted[0]
            hot = _select_hot_in_range(horses_sorted, MIX_LONGSHOT_RANKS)
            cold = _select_cold_in_range(
                horses_sorted, MIX_LONGSHOT_RANKS,
                exclude_nums={hot['horse']['number']} if hot else set(),
            )
            if hot and cold:
                hh = hot['horse']
                ch = cold['horse']
                trio3 = [top1['number'], hh['number'], ch['number']]
                # STRAIGHT TRIO with exactly 3 picks: hit iff set(top3) == set(trio3)
                hit = bool(top3) and set(top3) == set(trio3)
                row = _empty_row_fields()
                row.update({
                    'race_date': date,
                    'race_number': r.get('race_number'),
                    'variant': 'mix_hot_cold',
                    'longshot_num': hh['number'],
                    'longshot_name': hh.get('name', ''),
                    'longshot_rank': hot['rank'],
                    'longshot_odds': hh.get('win_odds') or '',
                    'steam_pct': hot['steam_pct'],
                    'cold_num': ch['number'],
                    'cold_name': ch.get('name', ''),
                    'cold_rank': cold['rank'],
                    'cold_odds': ch.get('win_odds') or '',
                    'drift_pct': cold['drift_pct'],
                    'box6_nums': '|'.join(str(n) for n in trio3),
                    'actual_top3': '|'.join(str(n) for n in top3),
                    'trio5_hit': 0,
                    'box6_hit': int(hit),
                    'rescued': 0,
                    'longshot_finish': hh.get('actual_position') or '',
                })
                rows.append(row)
    return rows


def _load_meeting(date: str) -> list | None:
    date_str = date.replace('/', '')
    path = REPO_ROOT / 'results' / f'race_data_{date_str}_with_results.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get('races') if isinstance(data, dict) else data


def _write_log(rows: list[dict], single_date: str | None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if LOG_PATH.exists() and single_date:
        with open(LOG_PATH) as f:
            existing = [r for r in csv.DictReader(f) if r.get('race_date') != single_date]
    all_rows = existing + rows if single_date else rows
    with open(LOG_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        w.writeheader()
        w.writerows(all_rows)


def track_one(date: str) -> int:
    races = _load_meeting(date)
    if races is None:
        print(f"No completed meeting at {date} (missing _with_results.json).")
        return 1
    rows = _rows_for_meeting(races, date)
    _write_log(rows, single_date=date)

    calc_rows = [r for r in rows if r['variant'] == 'calc_hot']
    mix_rows  = [r for r in rows if r['variant'] == 'mix_hot_cold']

    n_calc_hit = sum(r['box6_hit'] for r in calc_rows)
    n_calc_resc = sum(r['rescued'] for r in calc_rows)
    print(f"\n{date}  variant=calc_hot: {len(calc_rows)} qualifying races. "
          f"box6 hit {n_calc_hit}/{len(calc_rows)}; rescued {n_calc_resc}.")
    for r in calc_rows:
        flag = '  <-- RESCUED' if r['rescued'] else ('  (hit)' if r['box6_hit'] else '')
        print(f"   R{r['race_number']}: +#{r['longshot_num']} {r['longshot_name']} "
              f"(rank {r['longshot_rank']}, ${r['longshot_odds']}, -{r['steam_pct']}%) "
              f"finish {r['longshot_finish'] or '?'}{flag}")

    n_mix_hit = sum(r['box6_hit'] for r in mix_rows)
    print(f"\n{date}  variant=mix_hot_cold: {len(mix_rows)} qualifying races. "
          f"STRAIGHT TRIO hit {n_mix_hit}/{len(mix_rows)}.")
    for r in mix_rows:
        flag = '  (HIT)' if r['box6_hit'] else ''
        print(f"   R{r['race_number']}: hot #{r['longshot_num']} {r['longshot_name']} "
              f"(rank {r['longshot_rank']}, ${r['longshot_odds']}, -{r['steam_pct']}%)  "
              f"cold #{r['cold_num']} {r['cold_name']} "
              f"(rank {r['cold_rank']}, ${r['cold_odds']}, +{r['drift_pct']}%){flag}")
    return 0


def backfill_all() -> int:
    pattern = str(REPO_ROOT / 'results' / 'race_data_*_with_results.json')
    all_rows = []
    for p in sorted(glob.glob(pattern)):
        date_str = Path(p).stem.replace('race_data_', '').replace('_with_results', '')
        date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
        try:
            data = json.loads(Path(p).read_text())
            races = data.get('races') if isinstance(data, dict) else data
            all_rows.extend(_rows_for_meeting(races, date))
        except (json.JSONDecodeError, IOError):
            continue
    _write_log(all_rows, single_date=None)
    print(f"Backfilled {len(all_rows)} qualifying races across the log.")
    return report()


def _trio_dividend(date: str, race_no) -> float | None:
    """Min TRIO dividend (per $10) for a race, or None if not backfilled yet."""
    if not DB_PATH.exists():
        return None
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            "SELECT MIN(dividend) FROM race_dividends "
            "WHERE race_date=? AND race_number=? AND pool='TRIO'",
            (date, race_no)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    return row[0] if row and row[0] is not None else None


def _report_calc_hot(rows: list[dict]) -> dict:
    """Variant 1 report: top-5 + steamed rank 6-8 → BOX TRIO-6 (20 combos)."""
    if not rows:
        print("\nVariant 1 (calc_hot): no qualifying races.")
        return {}
    meetings = sorted({r['race_date'] for r in rows})
    n_q = len(rows)
    n_trio5 = sum(int(r['trio5_hit']) for r in rows)
    n_box6 = sum(int(r['box6_hit']) for r in rows)
    n_resc = sum(int(r['rescued']) for r in rows)

    box_stake = box_pay = base_stake = base_pay = 0.0
    n_priced = 0
    for r in rows:
        div = _trio_dividend(r['race_date'], int(r['race_number']))
        if div is None:
            continue
        n_priced += 1
        box_stake += 20 * 10
        base_stake += 10 * 10
        if int(r['box6_hit']):
            box_pay += div
        if int(r['trio5_hit']):
            base_pay += div

    print("\n" + "=" * 72)
    print("VARIANT 1: calc_hot  (top-5 + steamed rank 6-8 → BOX TRIO-6, 20 combos)")
    print("=" * 72)
    print(f"Meetings:      {len(meetings)}  ({meetings[0]} .. {meetings[-1]})")
    print(f"Qualifying:    {n_q}")
    print(f"  TRIO-5 base: {n_trio5}/{n_q} ({100*n_trio5/n_q:.0f}%)")
    print(f"  box6 hit:    {n_box6}/{n_q} ({100*n_box6/n_q:.0f}%)")
    print(f"  RESCUED:     {n_resc}  (box6 hit where TRIO-5 missed)")
    if n_priced:
        box_roi = (box_pay - box_stake) / box_stake * 100
        base_roi = (base_pay - base_stake) / base_stake * 100
        print(f"\nROI over {n_priced} priced races:")
        print(f"  longshot box:    {box_roi:+.0f}%  (won ${box_pay:.0f} on ${box_stake:.0f})")
        print(f"  baseline TRIO-5: {base_roi:+.0f}%  (won ${base_pay:.0f} on ${base_stake:.0f})")
    else:
        box_roi = base_roi = None
        print("\nROI: pending — no TRIO dividends backfilled yet.")
    print("\nVERDICT:", _verdict(len(meetings), box_roi, base_roi))
    return {'rows': rows, 'meetings': meetings}


def _report_mix_hot_cold(rows: list[dict]) -> dict:
    """Variant 2 report: top-1 + hot + cold → STRAIGHT TRIO (1 combo, $10 lottery).

    Same selection as dashboard's _bolterTrio() — the 🍀 lottery line.
    """
    if not rows:
        print("\nVariant 2 (mix_hot_cold): no qualifying races.")
        return {}
    meetings = sorted({r['race_date'] for r in rows})
    n_q = len(rows)
    n_hit = sum(int(r['box6_hit']) for r in rows)  # field reused for STRAIGHT TRIO hit

    mix_stake = mix_pay = 0.0
    n_priced = 0
    for r in rows:
        div = _trio_dividend(r['race_date'], int(r['race_number']))
        if div is None:
            continue
        n_priced += 1
        mix_stake += 1 * 10   # STRAIGHT TRIO = 1 combo at $10
        if int(r['box6_hit']):
            mix_pay += div

    print("\n" + "=" * 72)
    print("VARIANT 2: mix_hot_cold  (top-1 + steamed 6-12 + drifted 6-12 → STRAIGHT TRIO)")
    print("                         (same shape as dashboard 🍀 BOLTER TRIO)")
    print("=" * 72)
    print(f"Meetings:      {len(meetings)}  ({meetings[0]} .. {meetings[-1]})")
    print(f"Qualifying:    {n_q}")
    print(f"  STRAIGHT hit: {n_hit}/{n_q} ({100*n_hit/n_q:.1f}%)")
    if n_priced:
        mix_roi = (mix_pay - mix_stake) / mix_stake * 100
        print(f"\nROI over {n_priced} priced races (STRAIGHT TRIO, 1 combo × $10):")
        print(f"  mix straight: {mix_roi:+.0f}%  (won ${mix_pay:.0f} on ${mix_stake:.0f})")
        print("  (lottery shape — single hit can dwarf many meetings of losses)")
    else:
        mix_roi = None
        print("\nROI: pending — no TRIO dividends backfilled yet.")
    print("\nVERDICT:", _verdict(len(meetings), mix_roi, None))
    return {'rows': rows, 'meetings': meetings}


def report() -> int:
    if not LOG_PATH.exists():
        print("No log yet. Run with a date or --backfill-all first.")
        return 1
    with open(LOG_PATH) as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        print("Log is empty.")
        return 1

    # Backward-compat: rows from before the variant column default to calc_hot.
    for r in all_rows:
        if not r.get('variant'):
            r['variant'] = 'calc_hot'

    calc_rows = [r for r in all_rows if r['variant'] == 'calc_hot']
    mix_rows  = [r for r in all_rows if r['variant'] == 'mix_hot_cold']

    _report_calc_hot(calc_rows)
    _report_mix_hot_cold(mix_rows)
    rows = calc_rows  # legacy aggregate Bayesian section uses calc_hot only
    if not rows:
        return 0
    meetings = sorted({r['race_date'] for r in rows})
    box_stake = box_pay = base_stake = base_pay = 0.0
    n_priced = 0
    for r in rows:
        div = _trio_dividend(r['race_date'], int(r['race_number']))
        if div is None:
            continue
        n_priced += 1
        box_stake += 20 * 10
        base_stake += 10 * 10
        if int(r['box6_hit']):
            box_pay += div
        if int(r['trio5_hit']):
            base_pay += div
    box_roi = (box_pay - box_stake) / box_stake * 100 if box_stake else None
    base_roi = (base_pay - base_stake) / base_stake * 100 if base_stake else None

    # Bayesian per-meeting verdict (added 2026-05-23 — replaces the aggregate
    # N>=6 + ROI-compare heuristic above with optional-stopping-safe SHIP/KILL/
    # WATCH on per-meeting cumulative-ROI binary outcomes). Both verdicts are
    # printed so the older heuristic stays interpretable in the historical log.
    by_meeting = defaultdict(lambda: {'box_stake': 0.0, 'box_pay': 0.0,
                                       'base_stake': 0.0, 'base_pay': 0.0})
    for r in rows:
        div = _trio_dividend(r['race_date'], int(r['race_number']))
        if div is None:
            continue
        m = by_meeting[r['race_date']]
        m['box_stake'] += 20 * 10
        m['base_stake'] += 10 * 10
        if int(r['box6_hit']): m['box_pay'] += div
        if int(r['trio5_hit']): m['base_pay'] += div

    winners: list[int] = []
    box_cum = base_cum = box_stake_cum = base_stake_cum = 0.0
    for md in sorted(by_meeting):
        m = by_meeting[md]
        box_cum += (m['box_pay'] - m['box_stake'])
        base_cum += (m['base_pay'] - m['base_stake'])
        box_stake_cum += m['box_stake']
        base_stake_cum += m['base_stake']
        box_roi_cum = box_cum / box_stake_cum if box_stake_cum else 0
        base_roi_cum = base_cum / base_stake_cum if base_stake_cum else 0
        winners.append(1 if box_roi_cum > base_roi_cum else 0)

    v = bayesian_verdict(winners)
    print(f"\nBayesian (per-meeting cum-ROI, longshot-box vs baseline TRIO-5):")
    print(f"  meetings-won={v['wins']}/{v['n_meetings']}  "
          f"posterior={v['posterior']:.3f}  → {v['verdict']}")
    return 0


def _verdict(n_meetings: int, box_roi, base_roi) -> str:
    if n_meetings < 6:
        return (f"INSUFFICIENT_DATA — {n_meetings}/6 meetings. Keep paper-only; "
                "do NOT wire into dashboard/WA yet.")
    if box_roi is None:
        return "INSUFFICIENT_DATA — 6+ meetings but no priced races. Backfill dividends."
    if box_roi <= 0:
        return f"KILL — box ROI {box_roi:+.0f}% over 6+ meetings. Longshot leg is -EV."
    if base_roi is not None and box_roi <= base_roi:
        return (f"NEUTRAL — box ROI {box_roi:+.0f}% does not beat baseline TRIO-5 "
                f"{base_roi:+.0f}%. The longshot leg adds cost without edge.")
    return (f"SHIP candidate — box ROI {box_roi:+.0f}% > baseline "
            f"{base_roi:+.0f}% over 6+ meetings. Propose wiring in (re-confirm first).")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('date', nargs='?', help='YYYY/MM/DD — log one meeting')
    p.add_argument('--backfill-all', action='store_true',
                   help='Rebuild log from all completed meetings on disk')
    p.add_argument('--report', action='store_true',
                   help='Cumulative hit-rate + ROI + ship verdict')
    args = p.parse_args()
    if args.backfill_all:
        return backfill_all()
    if args.report:
        return report()
    if not args.date:
        p.print_help()
        return 1
    return track_one(args.date)


if __name__ == '__main__':
    sys.exit(main())
