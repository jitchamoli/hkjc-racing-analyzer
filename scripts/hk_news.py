"""CLI: pull today's HK editorial/news from theracingapi.com.

Usage:
    python3 -m scripts.hk_news                       # whole card, all flags
    python3 -m scripts.hk_news 2026/05/24            # specific date (HKT)
    python3 -m scripts.hk_news --picks               # only races where our
                                                     # predictions JSON has a
                                                     # MED/HIGH/EDGE tier pick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.racing_api_hk_news import fetch_hk_news  # noqa: E402


def _load_our_picks(date_yyyy_mm_dd: str) -> dict:
    """Returns {race_number: (horse_name, tier)} from results/race_data_*.json
    for the actionable tiers."""
    fname = REPO_ROOT / 'results' / f"race_data_{date_yyyy_mm_dd.replace('/','').replace('-','')}.json"
    if not fname.exists():
        return {}
    data = json.loads(fname.read_text())
    races = data if isinstance(data, list) else data.get('races', [])
    picks = {}
    for r in races:
        tier = (r.get('conviction_tier') or '').upper()
        if tier not in ('MED', 'HIGH', 'EDGE'):
            continue
        horses = sorted(r.get('horses', []),
                        key=lambda h: -(h.get('calibrated_probability') or 0))
        if not horses:
            continue
        rno = r.get('race_number') or r.get('race_no')
        picks[rno] = (horses[0].get('name', ''), tier)
    return picks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('date', nargs='?', help='YYYY/MM/DD (HKT) — defaults to today')
    p.add_argument('--picks', action='store_true',
                   help='Only show races with MED/HIGH/EDGE picks in race_data_*.json')
    args = p.parse_args()

    date_iso = args.date.replace('/', '-') if args.date else None
    races = fetch_hk_news(date_yyyy_mm_dd=date_iso)
    if not races:
        print(f"No HK racecards from API for {date_iso or 'today/tomorrow'}.")
        return 0

    picks = {}
    if args.picks and args.date:
        picks = _load_our_picks(args.date)
        if not picks:
            print(f"No MED/HIGH/EDGE tiered picks in results/ for {args.date}.")
            return 0

    print(f"\nHK card {races[0].off_hkt.strftime('%Y-%m-%d')} — "
          f"{len(races)} races · {races[0].venue} · source: theracingapi.com\n")

    for r in races:
        if args.picks and r.race_number not in picks:
            continue

        pick = picks.get(r.race_number)
        header = (f"R{r.race_number}  {r.time} HKT  "
                  f"{r.venue} {r.distance}M {r.going}  field={r.field_size}")
        print(header)

        if pick:
            pick_name, tier = pick
            h = r.find(pick_name)
            if h:
                t14 = (f"{h.trainer_14d_wins}/{h.trainer_14d_runs} "
                       f"({h.trainer_14d_pct:.0f}%)" if h.trainer_14d_runs else 'n/a')
                gear_bit = f" gear={h.gear}" if h.gear else ''
                wt_bit   = f" wt={h.weight}" if h.weight else ''
                or_bit   = f" OR={h.rating}" if h.rating is not None else ''
                print(f"  >>> OUR {tier} PICK: #{h.number} {h.name}  "
                      f"draw {h.draw}, {h.jockey}, {h.trainer}{gear_bit}{wt_bit}{or_bit}, "
                      f"trainer-14d {t14}")
                if h.comment:
                    print(f"      COMMENT:   {h.comment}")
                if h.spotlight and h.spotlight != h.comment:
                    print(f"      SPOTLIGHT: {h.spotlight}")
            else:
                print(f"  >>> {pick_name} not found in racing-API runners "
                      f"(name mismatch — HKJC vs api spelling?)")

        for h in r.horses:
            if h.flags:
                print(f"   [{' · '.join(h.flags)}]  #{h.number} {h.name}")
        print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
