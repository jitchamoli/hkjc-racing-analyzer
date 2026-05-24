"""Score tier_b_log.csv rows against actual race results.

Compares cumulative P&L of:
  - bet_now (current Tier B recommender: TRIO disabled, conviction 0.40/0.55)
  - bet_old (pre-Tier-B recommender: TRIO enabled, conviction 0.33/0.42)

Uses real HKJC dividends from race_dividends. Same 1u-stake convention as
data/database._real_exotic_pnl: WIN/PLACE/BANKER WIN = 1u, QIN/TRIO straight =
1 combo, BOX QIN-3/4 = 3/6 combos, TRIO-3 straight = 1 combo, BOX TRIO 4/5/6 =
4/10/20 combos.

The OLD code's TRIO branch always emits a STRAIGHT TRIO on the top-3 picks
(1 combo, 1u stake). The NEW code falls through to BOX Q4 / BOX Q / QIN.

Usage:
    python3 -m scripts.tier_b_evaluate '2026/05/17'   # score one meeting + print delta
    python3 -m scripts.tier_b_evaluate --report       # cumulative across all logged meetings
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config
from betting.dividends import hkjc_declared_to_unit
from scripts.forward_common import bayesian_verdict

LOG_PATH = REPO_ROOT / "tracking_data" / "tier_b_log.csv"

_BET_COMBOS = {
    'QIN':   {2: 1, 3: 3, 4: 6},
    'TRIO':  {3: 1, 4: 4, 5: 10, 6: 20},
}


def _lookup_actual_top(conn, date: str, race_no: int) -> tuple[list[int], list[int]]:
    """Return (top2_nums, top3_nums) from race_results for the given race."""
    rows = conn.execute(
        "SELECT finish_position, horse_number FROM race_results "
        "WHERE race_date=? AND race_number=? AND finish_position IN (1,2,3) "
        "ORDER BY finish_position",
        (date, race_no),
    ).fetchall()
    top3 = [int(r[1]) for r in rows]
    top2 = [int(r[1]) for r in rows[:2]]
    return top2, top3


def _lookup_win_dividend(conn, date: str, race_no: int, horse_number: int) -> float | None:
    """Return WIN dividend (per 1u stake) for the given horse, or None if missed."""
    row = conn.execute(
        "SELECT dividend FROM race_dividends "
        "WHERE race_date=? AND race_number=? AND pool='WIN' AND combination=?",
        (date, race_no, str(horse_number)),
    ).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _score_edge_shadow(conn, date: str, race_no: int,
                        horse_number: int, actual_top2: list[int]) -> tuple[float, float, bool]:
    """1u WIN stake on the EDGE candidate.

    Returns (pnl, stake, hit). Hit iff horse finished 1st (WIN pool).
    pnl = dividend - 1 on hit, -1 on miss. Stake = 1u always.
    """
    stake = 1.0
    won = bool(actual_top2) and horse_number == actual_top2[0]
    if not won:
        return -stake, stake, False
    dividend = _lookup_win_dividend(conn, date, race_no, horse_number)
    if dividend is None:
        # Hit but dividend missing — assume break-even (1u back) as honest fallback
        return 0.0, stake, True
    return dividend - stake, stake, True


def _real_pnl(conn, date, race_no, bet_type, leg_count, hit,
               winning_nums) -> tuple[float | None, float]:
    """Same dividend-lookup contract as data/database._real_exotic_pnl."""
    combos = _BET_COMBOS.get(bet_type, {}).get(leg_count)
    if combos is None:
        return None, 0.0
    stake = float(combos)
    if not hit:
        return -stake, stake
    pool = 'QUINELLA' if bet_type == 'QIN' else 'TRIO'
    nums = sorted(int(n) for n in winning_nums if n)
    combo = ','.join(str(n) for n in nums)
    row = conn.execute(
        "SELECT dividend FROM race_dividends "
        "WHERE race_date=? AND race_number=? AND pool=? AND combination=?",
        (date, race_no, pool, combo),
    ).fetchone()
    if not row:
        return None, stake
    dividend = hkjc_declared_to_unit(row[0])
    if dividend is None:
        return None, stake
    return dividend - stake, stake


def _score_bet(conn, date, race_no, bet_type, top3_picks, top4_picks,
                actual_top2, actual_top3) -> tuple[float, float, bool]:
    """Return (pnl, stake, hit). pnl=0 stake=0 hit=False if bet_type is SKIP/WIN/PLACE.

    Tier B comparison cares only about QIN/TRIO branches — those are what
    the rule cascade can shift between OLD and NEW code.
    """
    if bet_type == 'TRIO':
        # STRAIGHT TRIO on top-3 picks (OLD code path)
        legs = top3_picks
        hit = sorted(legs) == sorted(actual_top3) if len(actual_top3) == 3 else False
        pnl, stake = _real_pnl(conn, date, race_no, 'TRIO', 3, hit, legs)
        if pnl is None:
            return -1.0, 1.0, hit  # fallback: no dividend → assume losing 1u
        return pnl, stake, hit
    if bet_type == 'BOX Q4':
        legs = top4_picks
        legs_in_top2 = sum(1 for n in legs if n in actual_top2)
        hit = legs_in_top2 >= 2
        pnl, stake = _real_pnl(conn, date, race_no, 'QIN', 4, hit, actual_top2)
        if pnl is None:
            return -6.0, 6.0, hit
        return pnl, stake, hit
    if bet_type == 'BOX Q':
        legs = top3_picks
        legs_in_top2 = sum(1 for n in legs if n in actual_top2)
        hit = legs_in_top2 >= 2
        pnl, stake = _real_pnl(conn, date, race_no, 'QIN', 3, hit, actual_top2)
        if pnl is None:
            return -3.0, 3.0, hit
        return pnl, stake, hit
    if bet_type == 'QIN':
        legs = top3_picks[:2]
        legs_in_top2 = sum(1 for n in legs if n in actual_top2)
        hit = legs_in_top2 >= 2
        pnl, stake = _real_pnl(conn, date, race_no, 'QIN', 2, hit, actual_top2)
        if pnl is None:
            return -1.0, 1.0, hit
        return pnl, stake, hit
    # WIN / PLACE / BANKER WIN / SKIP — outside Tier B's effect
    return 0.0, 0.0, False


def report(filter_date: str | None = None) -> int:
    if not LOG_PATH.exists():
        print(f"No log yet at {LOG_PATH}. Run scripts.tier_b_shadow <date> first.")
        return 0
    with open(LOG_PATH) as f:
        rows = list(csv.DictReader(f))
    if filter_date:
        rows = [r for r in rows if r['race_date'] == filter_date]
    if not rows:
        print(f"No rows {('for ' + filter_date) if filter_date else ''} in log.")
        return 0

    conn = sqlite3.connect(str(config.DB_PATH))
    totals = defaultdict(lambda: {'bets': 0, 'hits': 0, 'pnl': 0.0, 'stake': 0.0})
    # Per-meeting NEW vs OLD totals — feeds the Bayesian verdict at end.
    per_meeting = defaultdict(lambda: {
        'new_pnl': 0.0, 'new_stake': 0.0,
        'old_pnl': 0.0, 'old_stake': 0.0,
    })
    # Marginal EDGE totals — only races where the LOWER threshold caught
    # a candidate that the HIGHER threshold did NOT. Answers "what does
    # +3pp catch above +5pp?" (audit fix 2026-05-17, ML expert finding LOW).
    marginal_totals = defaultdict(lambda: {'bets': 0, 'hits': 0, 'pnl': 0.0, 'stake': 0.0})
    races_scored = 0
    diff_rows = []  # races where OLD and NEW disagreed
    for r in rows:
        date = r['race_date']
        race_no = int(r['race_number'])
        actual_top2, actual_top3 = _lookup_actual_top(conn, date, race_no)
        if not actual_top2 or len(actual_top3) < 3:
            continue  # results not in yet
        top3_picks = [int(x) for x in r['top_3_horse_numbers'].split(',') if x]
        top4_picks = [int(x) for x in r['top_4_horse_numbers'].split(',') if x]

        for scheme, bet in (('NEW (Tier B)', r['bet_now']),
                            ('OLD (pre-audit)', r['bet_old'])):
            if bet in ('SKIP', 'WIN', 'PLACE', 'BANKER WIN'):
                continue
            pnl, stake, hit = _score_bet(conn, date, race_no, bet,
                                          top3_picks, top4_picks,
                                          actual_top2, actual_top3)
            totals[scheme]['bets'] += 1
            totals[scheme]['pnl'] += pnl
            totals[scheme]['stake'] += stake
            if hit:
                totals[scheme]['hits'] += 1
            # Also accumulate per-meeting NEW/OLD for the Bayesian verdict.
            mkey = 'new' if scheme == 'NEW (Tier B)' else 'old'
            per_meeting[date][f'{mkey}_pnl'] += pnl
            per_meeting[date][f'{mkey}_stake'] += stake

        if r['bet_now'] != r['bet_old']:
            diff_rows.append((date, race_no, r['bet_old'], r['bet_now']))

        # EDGE shadow — score 1u WIN bet on the candidate at each G3 cutoff.
        # Production cutoff is +5pp; +3 and +2 are observation-only until
        # we have N≥200 disagreements per the EDGE memory.
        edge_5 = (r.get('edge_5pp_horse') or '').strip()
        edge_3 = (r.get('edge_3pp_horse') or '').strip()
        edge_2 = (r.get('edge_2pp_horse') or '').strip()
        for thr_label, cand in (('EDGE @+5pp (prod)', edge_5),
                                 ('EDGE @+3pp (shadow)', edge_3),
                                 ('EDGE @+2pp (shadow)', edge_2)):
            if not cand:
                continue
            pnl, stake, hit = _score_edge_shadow(conn, date, race_no,
                                                  int(cand), actual_top2)
            totals[thr_label]['bets'] += 1
            totals[thr_label]['pnl'] += pnl
            totals[thr_label]['stake'] += stake
            if hit:
                totals[thr_label]['hits'] += 1

        # MARGINAL: what does each lower threshold catch beyond the higher?
        # +3pp marginal = candidate exists at +3pp but NOT at +5pp (different horse)
        # +2pp marginal = candidate exists at +2pp but NOT at +3pp
        for marg_label, lower, higher in (('+3pp ABOVE +5pp', edge_3, edge_5),
                                            ('+2pp ABOVE +3pp', edge_2, edge_3)):
            if lower and lower != higher:
                pnl, stake, hit = _score_edge_shadow(conn, date, race_no,
                                                      int(lower), actual_top2)
                marginal_totals[marg_label]['bets'] += 1
                marginal_totals[marg_label]['pnl'] += pnl
                marginal_totals[marg_label]['stake'] += stake
                if hit:
                    marginal_totals[marg_label]['hits'] += 1

        races_scored += 1
    conn.close()

    print(f"\nTier B forward-validation — {races_scored} races scored "
          f"({len(diff_rows)} where OLD vs NEW disagreed)")
    if diff_rows:
        print(f"\nDisagreements:")
        for date, race_no, old, new in diff_rows:
            print(f"  {date} R{race_no}: OLD={old} → NEW={new}")
    print(f"\n{'scheme':<22} {'bets':>5} {'hits':>5} {'hit%':>6} {'stake':>7} {'P&L':>9} {'ROI':>9}")
    print('-' * 75)
    # Render recommender bets first, then EDGE shadow scenarios
    order = ['NEW (Tier B)', 'OLD (pre-audit)',
             'EDGE @+5pp (prod)', 'EDGE @+3pp (shadow)', 'EDGE @+2pp (shadow)']
    for scheme in order:
        if scheme not in totals:
            continue
        d = totals[scheme]
        hit_pct = d['hits'] / d['bets'] * 100 if d['bets'] else 0
        roi = d['pnl'] / d['stake'] * 100 if d['stake'] else 0
        print(f"  {scheme:<20} {d['bets']:>5} {d['hits']:>5} {hit_pct:>5.1f}% "
              f"{f'${d['stake']:.0f}':>7} {f'${d['pnl']:+.1f}':>9} {roi:>+7.1f}%")

    if 'NEW (Tier B)' in totals and 'OLD (pre-audit)' in totals:
        n, o = totals['NEW (Tier B)'], totals['OLD (pre-audit)']
        roi_delta = (n['pnl']/n['stake']*100 if n['stake'] else 0) - \
                    (o['pnl']/o['stake']*100 if o['stake'] else 0)
        pnl_delta = n['pnl'] - o['pnl']
        verdict = 'NEW wins' if pnl_delta > 0 else ('OLD wins' if pnl_delta < 0 else 'tie')
        print(f"\nDelta (NEW - OLD): P&L ${pnl_delta:+.1f}, ROI {roi_delta:+.1f}pp → {verdict}")
        n_diff = len(diff_rows)
        if n_diff < 10:
            print(f"  ⚠️  Only {n_diff} disagreement(s) so far — collect ≥10 before drawing conclusions.")

        # Bayesian per-meeting verdict (added 2026-05-23) — optional-stopping
        # safe SHIP/KILL/WATCH on cumulative-ROI per meeting. Replaces the
        # aggregate "NEW wins / OLD wins / tie" verdict above as the primary
        # decision rule; that line stays for context.
        winners: list[int] = []
        new_cum_pnl = old_cum_pnl = 0.0
        new_cum_stake = old_cum_stake = 0.0
        for md in sorted(per_meeting):
            m = per_meeting[md]
            new_cum_pnl += m['new_pnl']; new_cum_stake += m['new_stake']
            old_cum_pnl += m['old_pnl']; old_cum_stake += m['old_stake']
            new_roi = new_cum_pnl / new_cum_stake if new_cum_stake else 0
            old_roi = old_cum_pnl / old_cum_stake if old_cum_stake else 0
            winners.append(1 if new_roi > old_roi else 0)
        v = bayesian_verdict(winners)
        print(f"\nBayesian (per-meeting cum-ROI, NEW vs OLD):")
        print(f"  meetings-won={v['wins']}/{v['n_meetings']}  "
              f"posterior={v['posterior']:.3f}  → {v['verdict']}")

    # Marginal EDGE table — only rows where lower threshold caught a horse
    # the higher one did not. Shows the incremental value of lowering G3.
    if marginal_totals:
        print(f"\n--- EDGE marginal-only (lower threshold catches above higher) ---")
        print(f"{'comparison':<22} {'bets':>5} {'hits':>5} {'hit%':>6} {'stake':>7} {'P&L':>9} {'ROI':>9}")
        print('-' * 75)
        for label in ('+3pp ABOVE +5pp', '+2pp ABOVE +3pp'):
            if label not in marginal_totals:
                continue
            d = marginal_totals[label]
            hit_pct = d['hits'] / d['bets'] * 100 if d['bets'] else 0
            roi = d['pnl'] / d['stake'] * 100 if d['stake'] else 0
            print(f"  {label:<20} {d['bets']:>5} {d['hits']:>5} {hit_pct:>5.1f}% "
                  f"{f'${d['stake']:.0f}':>7} {f'${d['pnl']:+.1f}':>9} {roi:>+7.1f}%")
        # Decision rule from EDGE memory: need ≥10 marginal fires per tier
        # before lowering production threshold
        m3_n = marginal_totals.get('+3pp ABOVE +5pp', {}).get('bets', 0)
        if m3_n < 10:
            print(f"  ⚠️  Only {m3_n} marginal +3pp fire(s) — need ≥10 before considering "
                  f"lowering production G3 to +3pp.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('race_date', nargs='?', default=None,
                   help='YYYY/MM/DD — score one meeting only')
    p.add_argument('--report', action='store_true',
                   help='Cumulative report across all logged meetings')
    args = p.parse_args()
    if args.report or not args.race_date:
        return report()
    return report(filter_date=args.race_date)


if __name__ == '__main__':
    sys.exit(main())
