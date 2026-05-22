"""Closing-line value (CLV) report — do our picks shorten (sharp) or drift?

REAL CLV: open->close odds movement on our recommended picks. This is the
fast, low-variance sharpness signal that the backtest's single-snapshot
"CLV" (log model/market at one point in time) is NOT. CLV tells us in a few
meetings what flat ROI takes 20+ to show: are we picking horses the market
subsequently backs?

Data:
  tracking_data/odds_history.db  — odds_snapshots (opening = first, closing =
                                   last snapshot per horse on race day)
  tracking_data/horse_history.db — prediction_log (our picks + outcomes)
Coverage starts 2026/05/03 (when live odds polling began); only horses with
>=2 snapshots have a measurable open->close move.

Metric:
  CLV% = (opening - closing) / opening * 100
  Positive  = pick SHORTENED after we picked it (market moved toward us →
              we "beat the closing line" → sharp).
  Negative  = pick DRIFTED (we faded a horse the market dropped).

Usage:
  python3 -m scripts.clv_report                 # PRIMARY (rank-1) picks
  python3 -m scripts.clv_report --rank 2        # rank-N picks
  python3 -m scripts.clv_report --since 2026/05/17
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ODDS_DB = REPO_ROOT / 'tracking_data' / 'odds_history.db'
HORSE_DB = REPO_ROOT / 'tracking_data' / 'horse_history.db'


def _open_close(since: str | None) -> dict:
    """Return {(date, race, horse_number): (opening_odds, closing_odds, n_snaps)}."""
    if not ODDS_DB.exists():
        return {}
    con = sqlite3.connect(ODDS_DB)
    q = ("SELECT race_date, race_number, horse_number, win_odds "
         "FROM odds_snapshots")
    params: tuple = ()
    if since:
        q += " WHERE race_date >= ?"
        params = (since,)
    q += " ORDER BY id ASC"
    opens, closes, counts = {}, {}, {}
    for date, race, horse, odds in con.execute(q, params):
        if not odds or odds <= 0:
            continue
        key = (date, race, horse)
        if key not in opens:
            opens[key] = odds
        closes[key] = odds
        counts[key] = counts.get(key, 0) + 1
    con.close()
    return {k: (opens[k], closes[k], counts[k]) for k in opens}


def _picks(rank: int, since: str | None) -> list:
    """rank-N picks from prediction_log with results filled."""
    con = sqlite3.connect(HORSE_DB)
    con.row_factory = sqlite3.Row
    q = ("SELECT race_date, race_number, horse_number, conviction_tier, "
         "win_odds, actual_position, actual_won, actual_top3 "
         "FROM prediction_log WHERE predicted_rank = ? AND results_filled = 1")
    params: list = [rank]
    if since:
        q += " AND race_date >= ?"
        params.append(since)
    rows = con.execute(q, params).fetchall()
    con.close()
    return rows


def _fmt_pct(x):
    return f"{x:+.1f}%"


def report(rank: int, since: str | None) -> int:
    oc = _open_close(since)
    if not oc:
        print(f"No odds snapshots found in {ODDS_DB}"
              + (f" since {since}" if since else "") + ".")
        return 1
    picks = _picks(rank, since)

    rows = []
    for p in picks:
        key = (p['race_date'], p['race_number'], p['horse_number'])
        if key not in oc:
            continue
        opening, closing, n = oc[key]
        if n < 2 or opening <= 0:
            continue  # no measurable move
        clv = (opening - closing) / opening * 100.0
        rows.append({
            'date': p['race_date'], 'race': p['race_number'],
            'tier': p['conviction_tier'] or '?',
            'opening': opening, 'closing': closing, 'clv': clv,
            'won': bool(p['actual_won']),
            'placed': bool(p['actual_top3']),
        })

    if not rows:
        print(f"No rank-{rank} picks with >=2 odds snapshots"
              + (f" since {since}" if since else "") + ". "
              "(Live odds polling started 2026/05/03; need multiple fetches/race.)")
        return 1

    n = len(rows)
    clvs = [r['clv'] for r in rows]
    mean_clv = sum(clvs) / n
    n_short = sum(1 for c in clvs if c > 0)
    n_won = sum(1 for r in rows if r['won'])
    n_placed = sum(1 for r in rows if r['placed'])

    # Predictive check: do shortened picks win/place more than drifted ones?
    short = [r for r in rows if r['clv'] > 0]
    drift = [r for r in rows if r['clv'] <= 0]
    def _rate(rs, key):
        return (sum(1 for r in rs if r[key]) / len(rs) * 100.0) if rs else float('nan')

    meetings = sorted({r['date'] for r in rows})
    print("\n" + "=" * 70)
    print(f"  CLV REPORT — rank-{rank} picks  ({n} picks, {len(meetings)} meetings)")
    print(f"  {meetings[0]} .. {meetings[-1]}")
    print("=" * 70)
    print(f"  Mean CLV:            {_fmt_pct(mean_clv)}   "
          f"({'picks SHORTEN on avg — sharp' if mean_clv > 0 else 'picks DRIFT on avg — fading the market'})")
    print(f"  Beat-the-line rate:  {n_short}/{n} ({100*n_short/n:.0f}% shortened)")
    print(f"  Win rate:            {n_won}/{n} ({100*n_won/n:.0f}%)   "
          f"Place rate: {n_placed}/{n} ({100*n_placed/n:.0f}%)")
    print()
    print("  Is CLV predictive? (win/place rate by odds move)")
    print(f"    SHORTENED picks (n={len(short)}): win {_rate(short,'won'):.0f}%  place {_rate(short,'placed'):.0f}%")
    print(f"    DRIFTED   picks (n={len(drift)}): win {_rate(drift,'won'):.0f}%  place {_rate(drift,'placed'):.0f}%")
    print("    → if SHORTENED win-rate > DRIFTED, our picks + the late market agree (good).")

    # Per-meeting CLV (the fast signal: watch this trend over coming weeks)
    print("\n  Per-meeting mean CLV:")
    for d in meetings:
        mr = [r['clv'] for r in rows if r['date'] == d]
        print(f"    {d}:  {_fmt_pct(sum(mr)/len(mr))}  (n={len(mr)})")
    print("\n  NOTE: positive & rising CLV = our picks are sharp; flat/negative = "
          "we're not beating the market. Far faster to read than flat ROI.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rank', type=int, default=1, help='predicted rank to analyse (default 1)')
    ap.add_argument('--since', default=None, help='YYYY/MM/DD lower bound')
    args = ap.parse_args()
    return report(args.rank, args.since)


if __name__ == '__main__':
    sys.exit(main())
