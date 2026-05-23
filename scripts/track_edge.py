"""Track EDGE-tier fires + finish positions across all completed meetings.

Writes tracking_data/edge_stats.json with the cumulative count consumed by
the dashboard EDGE banner. Replaces the hardcoded "N=4" text — banner now
reads live stats so the count stays accurate without manual updates.

Per-EDGE-fire row also written to tracking_data/edge_log.csv for analysis.

Usage:
    python3 scripts/track_edge.py 2026/05/17   # log one meeting
    python3 scripts/track_edge.py --backfill-all
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = REPO_ROOT / 'tracking_data' / 'edge_stats.json'
LOG_PATH = REPO_ROOT / 'tracking_data' / 'edge_log.csv'

LOG_HEADERS = [
    'race_date', 'race_number', 'edge_horse_number', 'edge_horse_name',
    'edge_horse_rank', 'win_odds', 'edge_pp', 'finish_position',
]


def _rows_for_meeting(races: list, date_str: str) -> list[dict]:
    """Return EDGE fires for one meeting with the EDGE horse's finish_position."""
    rows = []
    for r in races:
        if not r.get('results_available') or not r.get('has_edge'):
            continue
        eh_num = r.get('edge_horse_number')
        if eh_num is None:
            continue
        # Find the EDGE horse to read its finish_position
        edge_horse = next((h for h in r['horses'] if h.get('number') == eh_num), None)
        if not edge_horse:
            continue
        rows.append({
            'race_date': date_str,
            'race_number': r.get('race_number'),
            'edge_horse_number': eh_num,
            'edge_horse_name': r.get('edge_horse_name') or edge_horse.get('name', ''),
            'edge_horse_rank': r.get('edge_horse_rank') or edge_horse.get('rank'),
            'win_odds': edge_horse.get('win_odds') or '',
            'edge_pp': edge_horse.get('edge_pp') or '',
            'finish_position': edge_horse.get('actual_position') or '',
        })
    return rows


def _load_meeting(date: str) -> list | None:
    date_str = date.replace('/', '')
    path = REPO_ROOT / 'results' / f'race_data_{date_str}_with_results.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get('races') if isinstance(data, dict) else data


def _write_log_and_stats(rows: list[dict], single_date: str = None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # If updating one date, drop existing rows for it first (dedup)
    existing = []
    if LOG_PATH.exists() and single_date:
        with open(LOG_PATH) as f:
            existing = [r for r in csv.DictReader(f) if r.get('race_date') != single_date]
    elif LOG_PATH.exists() and not rows:
        return  # nothing to write
    all_rows = existing + rows if single_date else rows
    # Atomic write: a concurrent dashboard regen reads this file; a torn read
    # silently wipes the EDGE banner, and a half-written CSV permanently
    # corrupts the cumulative count on the next dedup read.
    _logtmp = LOG_PATH.with_suffix(LOG_PATH.suffix + '.tmp')
    with open(_logtmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    _logtmp.replace(LOG_PATH)

    # Recompute aggregate stats
    n_fires = len(all_rows)
    n_placed = sum(1 for r in all_rows
                    if str(r.get('finish_position', '')).isdigit() and int(r['finish_position']) <= 3)
    n_won = sum(1 for r in all_rows
                 if str(r.get('finish_position', '')).isdigit() and int(r['finish_position']) == 1)
    stats = {
        'n_fires': n_fires,
        'n_placed': n_placed,
        'n_won': n_won,
        'last_updated': max((r['race_date'] for r in all_rows), default='') if all_rows else '',
    }
    _stmp = STATS_PATH.with_suffix('.json.tmp')
    _stmp.write_text(json.dumps(stats, indent=2))
    _stmp.replace(STATS_PATH)


def track_one(date: str) -> int:
    races = _load_meeting(date)
    if races is None:
        print(f"No completed meeting at {date} (missing _with_results.json).")
        return 1
    rows = _rows_for_meeting(races, date)
    _write_log_and_stats(rows, single_date=date)
    stats = json.loads(STATS_PATH.read_text())
    print(f"{date}: logged {len(rows)} EDGE fires. "
          f"Cumulative: {stats['n_fires']} fires, "
          f"{stats['n_placed']} placed, {stats['n_won']} won.")
    return 0


def backfill_all() -> int:
    pattern = str(REPO_ROOT / 'results' / 'race_data_*_with_results.json')
    files = sorted(glob.glob(pattern))
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
    _write_log_and_stats(all_rows, single_date=None)
    stats = json.loads(STATS_PATH.read_text())
    print(f"Backfilled {stats['n_fires']} EDGE fires across {len(files)} meetings.")
    print(f"  Placed top-3: {stats['n_placed']}/{stats['n_fires']}")
    print(f"  Won:          {stats['n_won']}/{stats['n_fires']}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('date', nargs='?', help='YYYY/MM/DD — log EDGE fires for one meeting')
    p.add_argument('--backfill-all', action='store_true',
                   help='Rebuild edge_log.csv from all completed meetings on disk')
    args = p.parse_args()
    if args.backfill_all:
        return backfill_all()
    if not args.date:
        p.print_help()
        return 1
    return track_one(args.date)


if __name__ == '__main__':
    sys.exit(main())
