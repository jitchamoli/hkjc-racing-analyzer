"""Snapshot theracingapi.com HK racecard to disk for later confluence
scoring. Racing API only returns today/tomorrow — must run before
midnight HKT on race day.

Stores per-race: race_number, off_time, tip, verdict, betting_forecast,
horses[{number, name, morning_decimal}]. Morning decimal odds are parsed
from the `betting_forecast` string (the bookmaker's published morning
line) — racing API runner.odds[] is empty for HK on PRO tier.

Forward-validation gate (project_edge_pattern_n4): three-source confluence
= model rank-1 + steam (live ≥25% shorter than morning) + tip_match. R4
BRIGHT DAY 2026-05-27 hit all three and won ($3.7). R8 MR DESIRA had
steam only (TIP was different horse) and lost. N≥10 meetings to validate
WIN-sidebet ROI.

Use:
  python3 scripts/snapshot_racing_api.py 2026/05/30        # capture for date
  python3 scripts/snapshot_racing_api.py                   # capture for today
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / 'tracking_data' / 'racing_api_snapshots'


def parse_betting_forecast(s: str) -> dict[str, float]:
    """Parse "4/1 Find My Love, Bright Day, 6/1 Country Pride" into
    {'FIND MY LOVE': 5.0, 'BRIGHT DAY': 5.0, 'COUNTRY PRIDE': 7.0}.

    A bare name after a price inherits that price (racing API quirk).
    Returns decimal odds (frac + 1).
    """
    if not s:
        return {}
    out: dict[str, float] = {}
    current_price: float | None = None
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r'^(\d+)/(\d+)\s+(.+)$', chunk)
        if m:
            num, den, name = m.groups()
            current_price = int(num) / int(den) + 1.0
            out[name.strip().upper()] = current_price
        elif current_price is not None:
            out[chunk.strip().upper()] = current_price
    return out


def snapshot(date_str: str) -> Path:
    """Fetch racing API for date, save snapshot. Returns path."""
    from data.racing_api_hk_news import fetch_hk_news

    card = fetch_hk_news(date_str)
    if not card:
        print(f'❌ racing API returned 0 races for {date_str} '
              f'(API only carries today+tomorrow). Snapshot NOT written.',
              file=sys.stderr)
        sys.exit(1)

    snapshot_data = {
        'date': date_str,
        'captured_at': datetime.now().isoformat(timespec='seconds'),
        'races': [],
    }
    for race in card:
        forecast = parse_betting_forecast(getattr(race, 'betting_forecast', '') or '')
        horses_out = []
        for h in race.horses:
            horses_out.append({
                'number': h.number,
                'name': h.name,
                'morning_decimal': forecast.get(h.name.upper()),
            })
        snapshot_data['races'].append({
            'race_number': race.race_number,
            'off_hkt': race.off_hkt.strftime('%H:%M') if race.off_hkt else None,
            'tip': getattr(race, 'tip', '') or '',
            'verdict': getattr(race, 'verdict', '') or '',
            'betting_forecast': getattr(race, 'betting_forecast', '') or '',
            'horses': horses_out,
        })

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    date_compact = date_str.replace('/', '').replace('-', '')
    out_path = SNAPSHOT_DIR / f'{date_compact}.json'
    out_path.write_text(json.dumps(snapshot_data, indent=2, ensure_ascii=False))
    n_with_morning = sum(
        1 for r in snapshot_data['races']
        for h in r['horses']
        if h['morning_decimal'] is not None
    )
    n_total = sum(len(r['horses']) for r in snapshot_data['races'])
    print(f'✅ snapshot saved: {out_path}')
    print(f'   {len(card)} races, {n_with_morning}/{n_total} horses with morning odds, '
          f'{sum(1 for r in snapshot_data["races"] if r["tip"])} races with TIP')
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('date', nargs='?', default=None,
                   help='YYYY/MM/DD (HKT). Defaults to today (HKT).')
    args = p.parse_args()
    date_str = args.date or datetime.now().strftime('%Y/%m/%d')
    if '/' not in date_str:
        date_str = f'{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}'
    snapshot(date_str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
