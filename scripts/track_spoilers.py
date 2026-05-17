"""Log every horse OUTSIDE our top-5 that finished top-3 — for spoiler
pattern research.

Builds the empirical dataset we need to find REAL spoiler-swap rules
across 50+ races. Without this, every swap rule proposal (rank-5,
VALUE WATCH, etc.) is N=1 speculation.

Per-race row written to tracking_data/spoilers.csv:
  date, race_number, horse_number, our_rank, finish_pos, win_odds, gear,
  edge_pp, ev_win, top_2_share, top_3_share, conviction_tier

Use: after each meeting, `python3 scripts/track_spoilers.py YYYY/MM/DD`
or `--backfill-all` to rebuild from all completed meetings on disk.

Analysis: open spoilers.csv in any tool. Look for clusters in
(gear, odds-range, model-rank, top_2_share-bucket) that have high
spoiler frequency. Anything found there is a candidate for
eval_swap_rule.py validation.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPOILERS_PATH = REPO_ROOT / 'tracking_data' / 'spoilers.csv'

HEADERS = [
    'race_date', 'race_number', 'horse_number', 'horse_name',
    'our_rank', 'finish_pos', 'win_odds', 'gear',
    'edge_pp', 'ev_win', 'calibrated_prob',
    'top_2_share', 'top_3_share', 'conviction_tier',
]


def _rows_for_meeting(races: list, date_str: str) -> list[dict]:
    """Return a list of spoiler rows for one meeting's races."""
    rows = []
    for r in races:
        if not r.get('results_available'):
            continue
        horses_sorted = sorted(r['horses'], key=lambda h: h.get('rank') or 99)
        for h in r['horses']:
            pos = h.get('actual_position')
            rank = h.get('rank')
            if pos is None or rank is None:
                continue
            # Spoiler = placed top-3 but ranked outside top-5
            if pos > 3 or rank <= 5:
                continue
            rows.append({
                'race_date': date_str,
                'race_number': r.get('race_number'),
                'horse_number': h.get('number'),
                'horse_name': h.get('name') or h.get('horse_name') or '',
                'our_rank': rank,
                'finish_pos': pos,
                'win_odds': h.get('win_odds') or '',
                'gear': h.get('gear') or '',
                'edge_pp': h.get('edge_pp') or '',
                'ev_win': h.get('ev_win') or '',
                'calibrated_prob': h.get('calibrated_prob') or '',
                'top_2_share': r.get('top_2_share') or '',
                'top_3_share': r.get('top_3_share') or '',
                'conviction_tier': r.get('conviction_tier') or '',
            })
    return rows


def _load_meeting(date: str) -> list | None:
    """date in YYYY/MM/DD format, returns list of races or None."""
    date_str = date.replace('/', '')
    path = REPO_ROOT / 'results' / f'race_data_{date_str}_with_results.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get('races') if isinstance(data, dict) else data


def _write_rows(rows: list[dict], date: str, append: bool = False) -> None:
    """Write rows, deduping on (race_date, race_number, horse_number)."""
    SPOILERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if SPOILERS_PATH.exists():
        with open(SPOILERS_PATH) as f:
            for r in csv.DictReader(f):
                # Drop existing rows for this date (re-fire idempotent)
                if not append and r.get('race_date') == date:
                    continue
                existing.append(r)
    with open(SPOILERS_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for r in existing:
            w.writerow(r)
        for r in rows:
            w.writerow(r)


def track_one(date: str) -> int:
    races = _load_meeting(date)
    if not races:
        print(f"No completed meeting at {date} (missing _with_results.json).")
        return 1
    rows = _rows_for_meeting(races, date)
    _write_rows(rows, date)
    print(f"{date}: logged {len(rows)} spoilers to {SPOILERS_PATH}")
    return 0


def backfill_all() -> int:
    pattern = str(REPO_ROOT / 'results' / 'race_data_*_with_results.json')
    files = sorted(glob.glob(pattern))
    SPOILERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for p in files:
        date_str = Path(p).stem.replace('race_data_', '').replace('_with_results', '')
        date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
        try:
            data = json.loads(Path(p).read_text())
            races = data.get('races') if isinstance(data, dict) else data
            all_rows.extend(_rows_for_meeting(races, date))
        except (json.JSONDecodeError, IOError):
            continue
    with open(SPOILERS_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"Backfilled {len(all_rows)} spoiler rows across {len(files)} meetings.")
    print(f"\nSummary by our_rank:")
    from collections import Counter
    by_rank = Counter(r['our_rank'] for r in all_rows)
    for rank in sorted(by_rank):
        print(f"  rank {rank:>2}: {by_rank[rank]:>4} spoilers")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('date', nargs='?', help='YYYY/MM/DD — log spoilers for one meeting')
    p.add_argument('--backfill-all', action='store_true',
                   help='Rebuild spoilers.csv from all completed meetings on disk')
    args = p.parse_args()
    if args.backfill_all:
        return backfill_all()
    if not args.date:
        p.print_help()
        return 1
    return track_one(args.date)


if __name__ == '__main__':
    sys.exit(main())
