"""Shadow the MODEL-ONLY ranking (pure_prob) against production's blended one.

User directive (2026-09-16): "I don't always want to rely on market voted
horses — sometimes horses which are not ranked well come in the top 3." They
are right that it happens: across 236 forward races, 148 podium finishers were
ranked 8th or worse by the market.

Production ranks by `calibrated_prob`, which is ~55% market. `pure_prob` is
the model's own pre-market opinion (betting/edge_calculator owns the one
definition). Ranking by it instead:

    ranking                 winner   top3-in-5   outsiders caught   TRIO-5 ROI
    production (blended)     27.1%       24.2%        16/148          -21.3%
    model-only (pure_prob)   21.2%       24.2%        26/148          -15.1%

Same podium capture, 60% more outsiders, less loss — but worse at naming the
winner, and the bootstrap CI on the ROI gap spans zero. So this SHADOWS: it
logs what the model-only box WOULD have returned and changes no bet, no
ranking, no ticket. Ship only if the gap survives ([[n6_validation_rule]]).

Usage:
    python3 -m scripts.shadow_pure_rank 2026/09/13     # log one meeting
    python3 -m scripts.shadow_pure_rank --backfill-all
    python3 -m scripts.shadow_pure_rank --report
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / 'results'
LOG_PATH = REPO_ROOT / 'tracking_data' / 'pure_rank_log.csv'
DB_PATH = REPO_ROOT / 'tracking_data' / 'horse_history.db'

BOX_K = 5          # BOX TRIO-5, the main box bet
BOX_NEED = 3
OUTSIDER_RANK = 8  # "not ranked well" = market rank 8 or worse

LOG_HEADERS = ['race_date', 'race_number', 'prod_legs', 'pure_legs', 'actual_top3',
               'outsiders', 'prod_hit', 'pure_hit', 'prod_outsiders_in_box',
               'pure_outsiders_in_box', 'dividend']


def pure_order(horses: list) -> list:
    """Horse numbers ranked by the model's own opinion, market excluded.

    Ties break by production rank (input order), never by incidental order —
    same determinism rule as the recommender's `pre_blend_rank` tiebreak.
    """
    idx = {id(h): i for i, h in enumerate(horses)}
    return [h['number'] for h in sorted(
        horses, key=lambda h: (-(h.get('pure_prob') or -1.0), idx[id(h)]))]


def market_outsiders(top3: set, horses: list, cutoff: int = OUTSIDER_RANK) -> set:
    """Podium horses the MARKET rated `cutoff`th or worse. Unpriced runners
    have no market opinion, so they are not counted either way."""
    priced = sorted((h for h in horses if (h.get('win_odds') or 0) > 0),
                    key=lambda h: h['win_odds'])
    rank = {h['number']: i + 1 for i, h in enumerate(priced)}
    return {n for n in top3 if rank.get(n, 0) >= cutoff}


def box_result(order: list, top3: set, dividend):
    """(stake, payout, hit) for a BOX TRIO-5 on `order`. None when unscorable."""
    if len(order) < BOX_K or not dividend:
        return None
    hit = top3 <= set(order[:BOX_K])
    return comb(BOX_K, BOX_NEED) * 10.0, (dividend if hit else 0.0), hit


def _rows_for_meeting(race_date: str) -> list:
    path = RESULTS_DIR / f"race_data_{race_date.replace('/', '')}_with_results.json"
    if not path.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    trio = {rn: dv for rn, dv in conn.execute(
        "SELECT race_number, dividend FROM race_dividends "
        "WHERE race_date=? AND pool='TRIO'", (race_date,))}
    conn.close()

    rows = []
    for r in json.load(open(path)).get('races', []):
        hs = r.get('horses') or []
        ar = r.get('actual_results') or []
        if len(ar) < 3 or len(hs) < BOX_K or not any(h.get('pure_prob') for h in hs):
            continue
        top3 = {x['number'] for x in ar[:3]}
        prod = [h['number'] for h in hs]
        pure = pure_order(hs)
        dividend = trio.get(r['race_number'])
        scored_prod = box_result(prod, top3, dividend)
        scored_pure = box_result(pure, top3, dividend)
        if scored_prod is None or scored_pure is None:
            continue
        outs = market_outsiders(top3, hs)
        rows.append({
            'race_date': race_date, 'race_number': r['race_number'],
            'prod_legs': ' '.join(map(str, prod[:BOX_K])),
            'pure_legs': ' '.join(map(str, pure[:BOX_K])),
            'actual_top3': ' '.join(map(str, sorted(top3))),
            'outsiders': ' '.join(map(str, sorted(outs))),
            'prod_hit': int(scored_prod[2]), 'pure_hit': int(scored_pure[2]),
            'prod_outsiders_in_box': len(outs & set(prod[:BOX_K])),
            'pure_outsiders_in_box': len(outs & set(pure[:BOX_K])),
            'dividend': dividend,
        })
    return rows


def log_meeting(race_date: str) -> int:
    rows = _rows_for_meeting(race_date)
    if not rows:
        print(f"  no scorable races for {race_date} — nothing logged")
        return 0
    existing = []
    if LOG_PATH.exists():
        with open(LOG_PATH, newline='') as fh:
            existing = [r for r in csv.DictReader(fh) if r['race_date'] != race_date]
    with open(LOG_PATH, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_HEADERS)
        w.writeheader()
        w.writerows(existing + rows)
    print(f"  logged {len(rows)} races for {race_date} → {LOG_PATH.name}")
    return len(rows)


def backfill_all() -> None:
    for p in sorted(RESULTS_DIR.glob('race_data_*_with_results.json')):
        ymd = p.name[10:18]
        log_meeting(f"{ymd[:4]}/{ymd[4:6]}/{ymd[6:]}")


def report() -> None:
    if not LOG_PATH.exists():
        print("no pure_rank_log.csv yet — run --backfill-all")
        return
    with open(LOG_PATH, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("pure_rank_log.csv is empty")
        return
    meetings = sorted({r['race_date'] for r in rows})
    stake = len(rows) * comb(BOX_K, BOX_NEED) * 10.0
    n_out = sum(len(r['outsiders'].split()) for r in rows if r['outsiders'])
    print("=" * 74)
    print("  MODEL-ONLY RANKING (pure_prob) vs PRODUCTION — shadow, BOX TRIO-5")
    print("=" * 74)
    print(f"  meetings: {len(meetings)}  ({meetings[0]} .. {meetings[-1]})   "
          f"races: {len(rows)}   flat HK$10/combo\n")
    print("  %-22s %6s %10s %18s" % ('ranking', 'hits', 'ROI', 'outsiders in box'))
    for lab, hit_key, out_key in (('production (blended)', 'prod_hit', 'prod_outsiders_in_box'),
                                  ('model-only (pure)', 'pure_hit', 'pure_outsiders_in_box')):
        hits = [r for r in rows if r[hit_key] == '1']
        payout = sum(float(r['dividend']) for r in hits)
        caught = sum(int(r[out_key]) for r in rows)
        print("  %-22s %6d %9.1f%% %13d/%-4d" %
              (lab, len(hits), 100 * (payout - stake) / stake, caught, n_out))
    print("\n  'outsiders' = podium horses the MARKET ranked "
          f"{OUTSIDER_RANK}th or worse.")
    print("  SHADOW ONLY — nothing here changes a bet. Ship only if the ROI gap")
    print("  holds AND survives dropping the biggest dividend.")


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
