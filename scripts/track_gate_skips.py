"""Paper-track what the GATES throw away, per product, per race.

Motivation (2026-09-16): on 2026/05/24 R2 the model's top 4 were exactly the
first 4 home and the ticket was EMPTY — tier LOW, every horse SKIP. F4 BOX-4
would have paid $369 on $10. A one-off check across 172 races then showed the
races we SKIP had done no worse than the races we BET, but the gap collapsed
after removing one or two big dividends, i.e. noise. Nothing in the repo was
accumulating the comparison, so the question had no dataset behind it.

This logs, for every race with a frozen ticket, what the SAME model top-N box
would have returned at flat HK$10/combo — split by whether the gate actually
put that product on the ticket. It changes no bet and no ranking.

Read it as: does gating EARN its skips? If the SKIPPED column keeps beating
the BET column across meetings, the gates are costing money.

Ship gate (mirrors [[n6_validation_rule]]): >= 6 meetings before any gate
change, and discount ROI deltas >= 30pp at n < 50 by 3-5x
([[feedback_small_sample_inflation_pattern]]).

Usage:
    python3 -m scripts.track_gate_skips 2026/09/13     # log one meeting
    python3 -m scripts.track_gate_skips --backfill-all
    python3 -m scripts.track_gate_skips --report
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / 'tracking_data' / 'gate_skip_log.csv'
DB_PATH = REPO_ROOT / 'tracking_data' / 'horse_history.db'

LOG_HEADERS = ['race_date', 'race_number', 'product', 'decision', 'tier',
               'legs', 'stake', 'payout', 'hit']

# k = how many model ranks go in the box; need = finishers that must land in it.
PRODUCTS = {
    'QIN-4':  {'k': 4, 'need': 2, 'pool': 'QUINELLA',
               'labels': ('BOX QIN-4', 'BOX QIN-4 PROTECTION')},
    'TRIO-5': {'k': 5, 'need': 3, 'pool': 'TRIO',
               'labels': ('BOX TRIO-5',)},
    'F4-5':   {'k': 5, 'need': 4, 'pool': 'FIRST 4',
               'labels': ('F4 BOX-5',)},
}
_LABEL_TO_KEY = {lab: key for key, p in PRODUCTS.items() for lab in p['labels']}


def products_on_ticket(ticket: dict) -> set:
    """Product keys the gates actually STAKED. Paper lines are not a bet."""
    return {_LABEL_TO_KEY[b['label']]
            for b in (ticket or {}).get('bets', [])
            if b.get('label') in _LABEL_TO_KEY and not b.get('paper')}


def score_product(ranks: list, top4: list, dividends: dict, product: dict):
    """(stake, payout, hit) for boxing the model's top-k at $10/combo.

    None when the race cannot be scored at all: field too short to fill the
    box, too few finishers, or no dividend on record (an unpriced/overseas
    row must not be booked as a loss).
    """
    k, need, pool = product['k'], product['need'], product['pool']
    if len(ranks) < k or len(top4) < need:
        return None
    dividend = dividends.get(pool)
    if not dividend:
        return None
    hit = set(top4[:need]) <= set(ranks[:k])
    return comb(k, need) * 10.0, (dividend if hit else 0.0), hit


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

def _meeting_rows(conn, race_date: str) -> list:
    tickets = {rn: json.loads(js) for rn, js in conn.execute(
        "SELECT race_number, ticket_json FROM race_bet_ticket WHERE race_date=?", (race_date,))}
    if not tickets:
        return []
    ranks, finish, tiers = defaultdict(list), defaultdict(list), {}
    for rn, hno, pos, tier in conn.execute(
            "SELECT race_number, horse_number, actual_position, conviction_tier "
            "FROM prediction_log WHERE race_date=? ORDER BY race_number, predicted_rank", (race_date,)):
        if hno is None:
            continue
        ranks[rn].append(hno)
        tiers.setdefault(rn, tier)
        if pos and pos > 0:
            finish[rn].append((pos, hno))
    dividends = defaultdict(dict)
    for rn, pool, dv in conn.execute(
            "SELECT race_number, pool, dividend FROM race_dividends WHERE race_date=?", (race_date,)):
        dividends[rn].setdefault(pool, dv)

    rows = []
    for rn, ticket in sorted(tickets.items()):
        top4 = [h for _, h in sorted(finish.get(rn, []))[:4]]
        staked = products_on_ticket(ticket)
        for key, product in PRODUCTS.items():
            scored = score_product(ranks.get(rn, []), top4, dividends.get(rn, {}), product)
            if scored is None:
                continue
            stake, payout, hit = scored
            rows.append({
                'race_date': race_date, 'race_number': rn, 'product': key,
                'decision': 'bet' if key in staked else 'skipped',
                'tier': tiers.get(rn) or '', 'legs': ' '.join(str(n) for n in ranks[rn][:product['k']]),
                'stake': stake, 'payout': payout, 'hit': int(hit),
            })
    return rows


def log_meeting(race_date: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = _meeting_rows(conn, race_date)
    finally:
        conn.close()
    if not rows:
        print(f"  no frozen tickets / results for {race_date} — nothing logged")
        return 0
    existing = []
    if LOG_PATH.exists():
        with open(LOG_PATH, newline='') as fh:
            existing = [r for r in csv.DictReader(fh) if r['race_date'] != race_date]
    with open(LOG_PATH, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_HEADERS)
        w.writeheader()
        w.writerows(existing + rows)
    print(f"  logged {len(rows)} product-rows for {race_date} → {LOG_PATH.name}")
    return len(rows)


def backfill_all() -> None:
    conn = sqlite3.connect(DB_PATH)
    dates = [d for (d,) in conn.execute(
        "SELECT DISTINCT race_date FROM race_bet_ticket ORDER BY race_date")]
    conn.close()
    for d in dates:
        log_meeting(d)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def report() -> None:
    if not LOG_PATH.exists():
        print("no gate_skip_log.csv yet — run --backfill-all")
        return
    with open(LOG_PATH, newline='') as fh:
        rows = list(csv.DictReader(fh))
    meetings = sorted({r['race_date'] for r in rows})
    print("=" * 76)
    print("  GATE SKIPS — same model box, split by whether the gate staked it")
    print("=" * 76)
    print(f"  meetings: {len(meetings)}  ({meetings[0]} .. {meetings[-1]})   "
          f"flat HK$10/combo, real dividends, PAPER\n")
    print("  %-7s %-8s %6s %5s %8s %9s %11s" %
          ('product', 'decision', 'races', 'hits', 'ROI', 'med div', 'ROI -top1'))
    for key in PRODUCTS:
        for decision in ('bet', 'skipped'):
            sel = [r for r in rows if r['product'] == key and r['decision'] == decision]
            if not sel:
                continue
            stake = sum(float(r['stake']) for r in sel)
            payout = sum(float(r['payout']) for r in sel)
            wins = sorted((float(r['payout']) for r in sel if r['hit'] == '1'), reverse=True)
            roi = 100 * (payout - stake) / stake
            roi_dropped = 100 * (payout - (wins[0] if wins else 0) - stake) / stake
            print("  %-7s %-8s %6d %5d %7.1f%% %9.0f %10.1f%%" %
                  (key, decision, len(sel), len(wins), roi,
                   st.median(wins) if wins else 0, roi_dropped))
        print()
    print("  READ: if SKIPPED keeps beating BET, the gates are costing money.")
    print(f"  Ship gate: >= 6 meetings (have {len(meetings)}) AND the gap must survive")
    print("  the 'ROI -top1' column — one big dividend has flipped this before.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('race_date', nargs='?', help='YYYY/MM/DD')
    ap.add_argument('--backfill-all', action='store_true')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if args.backfill_all:
        backfill_all()
    elif args.race_date:
        log_meeting(args.race_date)
    if args.report or not (args.backfill_all or args.race_date):
        report()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
