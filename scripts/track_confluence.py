"""Track three-source confluence for WIN-sidebet forward validation.

Confluence = model rank-1 horse is also the racing API TIP AND has been
steamed (live odds ≥25% shorter than the bookmaker morning line). When
all three sources agree, hypothesis is that WIN sidebet is +EV. Pre-kill
context: pure steam (R8 MR DESIRA 2026-05-27) lost; steam + TIP + model
together (R4 BRIGHT DAY 2026-05-27) won. Need N≥10 meetings to validate.

Run AFTER `snapshot_racing_api.py` was captured for the same date
(racing API only carries today/tomorrow so the snapshot must exist).

Per-race row written to tracking_data/confluence_log.csv:
  date, race_number, rank1_number, rank1_name,
  morning_odds, live_odds, steam_pct, steam_yes,
  api_tip, tip_match, all_three,
  rank1_finish_pos, won, profit_at_10_flat

Use:
  python3 scripts/track_confluence.py 2026/05/30
  python3 scripts/track_confluence.py 2026/05/30 --seed   # write headers + skip
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / 'tracking_data' / 'racing_api_snapshots'
LOG_PATH = REPO_ROOT / 'tracking_data' / 'confluence_log.csv'

STEAM_THRESHOLD = 0.25  # live must be ≥25% shorter than morning to count as steamed

HEADERS = [
    'race_date', 'race_number', 'rank1_number', 'rank1_name',
    'morning_odds', 'live_odds', 'steam_pct', 'steam_yes',
    'api_tip', 'tip_match', 'all_three',
    'rank1_finish_pos', 'won', 'profit_at_10_flat',
]


def _norm(s: str) -> str:
    """Normalize horse names for fuzzy match across HKJC ↔ racing API."""
    return re.sub(r'[^A-Z0-9]', '', (s or '').upper())


def _row_for_race(race: dict, snapshot_race: dict) -> dict | None:
    """Compute confluence row for one race's rank-1 horse. Returns None
    if results not yet available."""
    if not race.get('results_available'):
        return None
    rank1 = next((h for h in race['horses'] if h.get('rank') == 1), None)
    if not rank1:
        return None
    morning_h = next(
        (h for h in snapshot_race.get('horses', [])
         if _norm(h.get('name', '')) == _norm(rank1.get('name', ''))),
        None,
    )
    morning = morning_h.get('morning_decimal') if morning_h else None
    live = float(rank1.get('win_odds') or 0)
    steam_pct = ((morning - live) / morning) if (morning and live) else None
    steam_yes = bool(steam_pct is not None and steam_pct >= STEAM_THRESHOLD)
    api_tip = snapshot_race.get('tip', '') or ''
    tip_match = bool(api_tip and _norm(api_tip) == _norm(rank1.get('name', '')))
    all_three = steam_yes and tip_match
    finish_pos = rank1.get('actual_position')
    won = finish_pos == 1
    profit = (10 * live - 10) if won else -10

    return {
        'race_date': race.get('race_date') or '',
        'race_number': race.get('race_number'),
        'rank1_number': rank1.get('number'),
        'rank1_name': rank1.get('name'),
        'morning_odds': f'{morning:.2f}' if morning else '',
        'live_odds': f'{live:.2f}',
        'steam_pct': f'{steam_pct:.3f}' if steam_pct is not None else '',
        'steam_yes': int(steam_yes),
        'api_tip': api_tip,
        'tip_match': int(tip_match),
        'all_three': int(all_three),
        'rank1_finish_pos': finish_pos,
        'won': int(won),
        'profit_at_10_flat': f'{profit:.2f}',
    }


def _write_rows(rows: list[dict], replace_date: str | None = None) -> None:
    """Write rows to CSV. When `replace_date` is set, any existing rows for
    that race_date are dropped first so re-running `./raceday finish` for
    the same meeting is idempotent (track_edge / track_spoilers follow the
    same pattern). Without dedup, the second run silently doubled every
    confluence row and corrupted the WIN-sidebet ROI.

    Header is written iff the file is new or being fully rewritten.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if replace_date and LOG_PATH.exists():
        existing = []
        with LOG_PATH.open('r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get('race_date') != replace_date:
                    existing.append(r)
        with LOG_PATH.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADERS)
            w.writeheader()
            for r in existing:
                w.writerow(r)
            for row in rows:
                w.writerow(row)
        return

    is_new = not LOG_PATH.exists()
    with LOG_PATH.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        if is_new:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('date', help='YYYY/MM/DD (HKT)')
    p.add_argument('--seed', action='store_true',
                   help='Create CSV with headers only and exit (for first-time setup).')
    args = p.parse_args()

    if args.seed:
        if LOG_PATH.exists():
            print(f'ℹ️  {LOG_PATH} already exists, no-op.')
            return 0
        _write_rows([])
        print(f'✅ seeded empty CSV: {LOG_PATH}')
        return 0

    date_str = args.date
    date_compact = date_str.replace('/', '').replace('-', '')

    snapshot_path = SNAPSHOT_DIR / f'{date_compact}.json'
    if not snapshot_path.exists():
        print(f'❌ no snapshot at {snapshot_path}. Run '
              f'`python3 scripts/snapshot_racing_api.py {date_str}` first '
              f'(must run before midnight HKT on race day).', file=sys.stderr)
        return 1

    results_path = REPO_ROOT / 'results' / f'race_data_{date_compact}_with_results.json'
    if not results_path.exists():
        print(f'❌ no results at {results_path}. Run '
              f'`python3 results_fetcher.py {date_str}` first.', file=sys.stderr)
        return 1

    snap = json.loads(snapshot_path.read_text())
    snap_by_race = {r['race_number']: r for r in snap.get('races', [])}

    results_doc = json.loads(results_path.read_text())
    races = results_doc.get('races') if isinstance(results_doc, dict) else results_doc

    rows = []
    for race in races:
        race['race_date'] = date_str
        sr = snap_by_race.get(race.get('race_number'))
        if not sr:
            continue
        row = _row_for_race(race, sr)
        if row:
            rows.append(row)

    _write_rows(rows, replace_date=date_str)
    n_all_three = sum(1 for r in rows if r['all_three'])
    n_won = sum(1 for r in rows if r['won'])
    print(f'✅ logged {len(rows)} rows to {LOG_PATH}')
    print(f'   all_three_confluent: {n_all_three}/{len(rows)} races · '
          f'rank-1 wins: {n_won}/{len(rows)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
