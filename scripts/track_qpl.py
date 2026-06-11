"""Paper-track a QUINELLA PLACE (QPL) experiment across meetings.

Motivation (2026-06-10): user observes QPL bets (both picks finish top-3,
any order) hit far more often than straight Quinella (picks 1-2) and *feel*
like they pay well. A 6-meeting backtest confirmed the hit-rate half — QPL
top-2 hits ~2x straight QIN (20% vs 10%) — but blanket QPL ROI did NOT beat
straight QIN (-28% vs -25%) because the easier bet pays smaller dividends
(the "predictable hits more, pays least" law, see [[project_north_star_top4_ranking]]).

QPL maps onto the model's STRONGEST dimension (the place model, ECE 0.009 vs
0.053 win), so a SELECTIVE/gated QPL is more plausible than most ideas here.
This logs — ON PAPER ONLY, nothing wired into the dashboard or WA — what a
QPL bet WOULD have returned, to find out at N>=6 whether the user's instinct
is a real edge or the frequency illusion.

Two variants, each = the model's top-2 as a single QPL pair ($10, 1 combo):
  - gated_top2 : fire ONLY when conviction_tier != SKIP (the selective thesis)
  - all_top2   : fire every race (the blanket baseline, -28% in backtest)

Each is scored against REAL HKJC dividends (pool='QUINELLA PLACE', per HK$10)
and compared to the SAME pick played as a straight QUINELLA, so the report
shows whether QPL's extra hits actually beat straight QIN on ROI.

Ship gate (mirrors [[feedback_n6_validation_rule]] + track_longshot_trio):
  - < 6 meetings                                  -> INSUFFICIENT_DATA (paper-only)
  - >= 6 meetings AND QPL ROI > 0 AND beats QIN   -> SHIP candidate
  - >= 6 meetings AND QPL ROI <= 0                -> KILL

Usage:
    python3 scripts/track_qpl.py 2026/06/10     # log one meeting
    python3 scripts/track_qpl.py --backfill-all
    python3 scripts/track_qpl.py --report
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.forward_common import bayesian_verdict

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / 'tracking_data' / 'qpl_log.csv'
DB_PATH = REPO_ROOT / 'tracking_data' / 'horse_history.db'
STAKE = 10  # user's flat HK$10/combo habit ([[user_stake_habit_10hkd_minimum]])

LOG_HEADERS = [
    'race_date', 'race_number', 'variant', 'tier',
    'a_num', 'a_name', 'b_num', 'b_name',
    'actual_top3', 'qpl_hit', 'qin_hit',
]


def _load_meeting(date: str) -> list | None:
    date_str = date.replace('/', '')
    path = REPO_ROOT / 'results' / f'race_data_{date_str}_with_results.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get('races') if isinstance(data, dict) else data


def _rows_for_meeting(races: list, date: str) -> list[dict]:
    """Up to 2 rows per race (gated_top2 + all_top2)."""
    rows = []
    for r in races:
        if not r.get('results_available'):
            continue
        top3 = r.get('actual_top3') or []
        if len(top3) < 3:
            continue
        # actual_top3 is sorted by horse NUMBER, not finish order — derive the
        # true finishing order from actual_position for the top-2 (QIN baseline).
        fin = sorted(
            (int(h['actual_position']), int(h['number'])) for h in r['horses']
            if str(h.get('actual_position') or '').isdigit() and int(h['actual_position']) > 0
            and str(h.get('number') or '').isdigit())
        if len(fin) < 3:
            continue
        top3_set = {n for _, n in fin[:3]}
        top2_set = {n for _, n in fin[:2]}
        tier = (r.get('conviction_tier') or '').upper()
        horses_sorted = sorted(r['horses'], key=lambda h: h.get('rank') or 99)
        if len(horses_sorted) < 2:
            continue
        a, b = horses_sorted[0], horses_sorted[1]
        try:
            a_num, b_num = int(a['number']), int(b['number'])
        except (TypeError, ValueError, KeyError):
            continue
        qpl_hit = int(a_num in top3_set and b_num in top3_set)
        qin_hit = int({a_num, b_num} == top2_set)
        base = {
            'race_date': date, 'race_number': r.get('race_number'), 'tier': tier,
            'a_num': a_num, 'a_name': a.get('name', ''),
            'b_num': b_num, 'b_name': b.get('name', ''),
            'actual_top3': '|'.join(str(n) for n in top3),
            'qpl_hit': qpl_hit, 'qin_hit': qin_hit,
        }
        rows.append({**base, 'variant': 'all_top2'})
        if tier != 'SKIP':
            rows.append({**base, 'variant': 'gated_top2'})
    return rows


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


def _div(con, date: str, race_no, pool: str, a: int, b: int) -> float | None:
    """Dividend (per $1) for pair (a,b) in `pool`, or None if that exact pair
    was not a declared winning combination."""
    lo, hi = sorted((a, b))
    try:
        row = con.execute(
            "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
            "AND pool=? AND combination=?", (date, race_no, pool, f"{lo},{hi}")).fetchone()
    except sqlite3.Error:
        return None
    return row[0] / 10.0 if row and row[0] is not None else None  # per HK$10 -> per $1


def _race_priced(con, date: str, race_no, pool: str) -> bool:
    """True if `pool` results are declared for this race (so we can stake it).
    A losing pair simply has no winning-combo row, but the race IS scorable."""
    try:
        row = con.execute(
            "SELECT 1 FROM race_dividends WHERE race_date=? AND race_number=? "
            "AND pool=? LIMIT 1", (date, race_no, pool)).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def track_one(date: str) -> int:
    races = _load_meeting(date)
    if races is None:
        print(f"No completed meeting at {date} (missing _with_results.json).")
        return 1
    rows = _rows_for_meeting(races, date)
    _write_log(rows, single_date=date)
    for variant in ('gated_top2', 'all_top2'):
        vr = [r for r in rows if r['variant'] == variant]
        if not vr:
            continue
        nq = len(vr); qpl = sum(r['qpl_hit'] for r in vr); qin = sum(r['qin_hit'] for r in vr)
        print(f"\n{date}  {variant}: {nq} races  | QPL hit {qpl}/{nq}  vs  straight-QIN hit {qin}/{nq}")
        for r in vr:
            f = 'QPL✓' if r['qpl_hit'] else '   '
            print(f"   R{r['race_number']:<2} [{r['tier']:<4}] #{r['a_num']} {r['a_name']} + "
                  f"#{r['b_num']} {r['b_name']}  -> top3={r['actual_top3']}  {f}")
    return 0


def _report_variant(con, rows: list[dict]) -> tuple[float | None, float | None, list[int]]:
    """Return (qpl_roi, qin_roi, per-meeting winners) for a variant's rows."""
    meetings = sorted({r['race_date'] for r in rows})
    qpl_stake = qpl_pay = qin_stake = qin_pay = 0.0
    # per-meeting cumulative tracking for the Bayesian verdict
    cum_qpl_s = cum_qpl_p = cum_qin_s = cum_qin_p = 0.0
    winners = []
    n_priced = 0
    for d in meetings:
        mrows = [r for r in rows if r['race_date'] == d]
        for r in mrows:
            rn = int(r['race_number'])
            if not _race_priced(con, d, rn, 'QUINELLA PLACE'):
                continue  # results not declared yet — not scorable
            n_priced += 1
            qpl_stake += STAKE; cum_qpl_s += STAKE
            qin_stake += STAKE; cum_qin_s += STAKE
            qd = _div(con, d, rn, 'QUINELLA PLACE', int(r['a_num']), int(r['b_num']))
            nd = _div(con, d, rn, 'QUINELLA', int(r['a_num']), int(r['b_num']))
            if int(r['qpl_hit']) and qd:
                qpl_pay += qd * STAKE; cum_qpl_p += qd * STAKE
            if int(r['qin_hit']) and nd:
                qin_pay += nd * STAKE; cum_qin_p += nd * STAKE
        # cumulative-through-meeting ROI comparison
        if cum_qpl_s and cum_qin_s:
            qpl_roi_c = (cum_qpl_p - cum_qpl_s) / cum_qpl_s
            qin_roi_c = (cum_qin_p - cum_qin_s) / cum_qin_s
            winners.append(1 if qpl_roi_c > qin_roi_c else 0)
    qpl_roi = (qpl_pay - qpl_stake) / qpl_stake * 100 if qpl_stake else None
    qin_roi = (qin_pay - qin_stake) / qin_stake * 100 if qin_stake else None
    nq = len(rows)
    print(f"  races={nq}  priced={n_priced}  "
          f"QPL hit {sum(int(r['qpl_hit']) for r in rows)}/{nq} "
          f"({100*sum(int(r['qpl_hit']) for r in rows)/nq:.0f}%)  "
          f"QIN hit {sum(int(r['qin_hit']) for r in rows)}/{nq} "
          f"({100*sum(int(r['qin_hit']) for r in rows)/nq:.0f}%)")
    if qpl_roi is not None:
        print(f"  QPL ROI {qpl_roi:+.1f}%   |   straight-QIN ROI {qin_roi:+.1f}%   "
              f"(same picks, real dividends, flat ${STAKE})")
    else:
        print("  ROI pending — no QPL dividends backfilled yet.")
    return qpl_roi, qin_roi, winners


def report() -> int:
    if not LOG_PATH.exists():
        print("No qpl_log.csv yet. Run with a date or --backfill-all.")
        return 1
    with open(LOG_PATH) as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        print("qpl_log.csv is empty.")
        return 1
    con = sqlite3.connect(DB_PATH) if DB_PATH.exists() else None
    print("=" * 72)
    print("QPL (QUINELLA PLACE) SHADOW — paper-only forward experiment")
    print("=" * 72)
    for variant, desc in (('gated_top2', 'top-2 QPL, tier != SKIP (selective thesis)'),
                          ('all_top2', 'top-2 QPL every race (blanket baseline)')):
        vr = [r for r in all_rows if r['variant'] == variant]
        if not vr:
            continue
        meetings = sorted({r['race_date'] for r in vr})
        print(f"\n--- {variant}: {desc} ---")
        print(f"  meetings: {len(meetings)}  ({meetings[0]} .. {meetings[-1]})")
        qpl_roi, qin_roi, winners = (None, None, [])
        if con is not None:
            qpl_roi, qin_roi, winners = _report_variant(con, vr)
        bv = bayesian_verdict(winners)
        ship = 'INSUFFICIENT_DATA'
        if bv['n_meetings'] >= 6 and qpl_roi is not None:
            if qpl_roi > 0 and (qin_roi is None or qpl_roi > qin_roi):
                ship = 'SHIP candidate'
            elif qpl_roi <= 0:
                ship = 'KILL'
            else:
                ship = 'WATCH'
        print(f"  Bayesian P(QPL beats straight-QIN): {bv['posterior']:.3f} "
              f"({bv['wins']}/{bv['n_meetings']} meetings)  |  ship-gate: {ship}")
    if con is not None:
        con.close()
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
    print(f"Backfilled {len(all_rows)} variant-rows across the log.\n")
    return report()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('date', nargs='?', help='Meeting date YYYY/MM/DD to log')
    p.add_argument('--report', action='store_true', help='Cumulative report')
    p.add_argument('--backfill-all', action='store_true',
                   help='Rebuild log from every *_with_results.json then report')
    args = p.parse_args()
    if args.backfill_all:
        return backfill_all()
    if args.report:
        return report()
    if args.date:
        return track_one(args.date)
    p.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
