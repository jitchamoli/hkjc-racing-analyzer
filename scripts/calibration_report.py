"""Read-only calibration audit of verified frozen pre-off predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import config

HKT = ZoneInfo('Asia/Hong_Kong')
BANDS = (0, .05, .10, .20, .30, .50, 1.01)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def eligible_race(day, race):
    ticket = race.get('bet_ticket') or {}
    if ticket.get('post_off_unverified'):
        return [], 'post_off_unverified'
    try:
        stamp = datetime.fromisoformat(ticket['frozen_at'])
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=HKT)
        off = datetime.strptime(day + ' ' + race['time'], '%Y/%m/%d %H:%M').replace(tzinfo=HKT)
        if stamp >= off:
            return [], 'not_verified_pre_off'
    except (KeyError, TypeError, ValueError):
        return [], 'missing_time_provenance'
    horses = race.get('horses') or []
    actual = race.get('actual_results') or []
    numbers = [h.get('number') for h in horses]
    outcomes = {r.get('number'): r.get('position') for r in actual}
    if (not horses or None in numbers or len(set(numbers)) != len(numbers)
            or len(outcomes) != len(actual) or set(numbers) != set(outcomes)
            or any(not isinstance(p, int) or p < 1 for p in outcomes.values())):
        return [], 'incomplete_or_withdrawn_field'
    probs = [h.get('calibrated_prob') for h in horses]
    if any(not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs):
        return [], 'invalid_probability'
    if abs(sum(probs) - 1) > .001:
        return [], 'non_normalized_probabilities'
    winners = sum(p == 1 for p in outcomes.values())
    if not winners:
        return [], 'missing_winner'
    return [dict(race_date=day, race_number=race['race_number'], number=h['number'],
                 tier=race.get('conviction_tier', 'UNKNOWN'), rank=h.get('rank'),
                 probability=p, outcome=(1 / winners if outcomes[h['number']] == 1 else 0),
                 frozen_at=ticket['frozen_at'], model_version=race.get('model_version', 'unknown'))
            for h, p in zip(horses, probs)], None


def block_interval(rows, statistic, seed=42):
    blocks = defaultdict(list)
    for row in rows:
        blocks[row['race_date']].append(row)
    if len(blocks) < 2:
        return None
    values = list(blocks.values())
    rng = np.random.default_rng(seed)
    samples = [statistic([r for i in rng.integers(0, len(values), len(values)) for r in values[i]])
               for _ in range(500)]
    return np.quantile(samples, [.025, .975]).tolist()


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        p = row['probability']
        idx = next(i for i in range(len(BANDS) - 1) if BANDS[i] <= p < BANDS[i + 1])
        groups[(row['tier'], idx)].append(row)
    report = []
    for (tier, idx), xs in sorted(groups.items()):
        gap = lambda ys: float(np.mean([r['outcome'] - r['probability'] for r in ys]))
        n_meetings = len({r['race_date'] for r in xs})
        report.append(dict(tier=tier, band=f'[{BANDS[idx]:.2f},{min(1, BANDS[idx+1]):.2f}' + (']' if idx == 5 else ')'),
                           runners=len(xs), races=len({(r['race_date'], r['race_number']) for r in xs}),
                           meetings=n_meetings, mean_probability=float(np.mean([r['probability'] for r in xs])),
                           observed=float(np.mean([r['outcome'] for r in xs])), gap=gap(xs),
                           gap_ci95=block_interval(xs, gap),
                           brier=float(np.mean([(r['probability'] - r['outcome']) ** 2 for r in xs])),
                           insufficient_evidence=n_meetings < 6 or len(xs) < 100))
    races = defaultdict(list)
    for row in rows:
        races[(row['race_date'], row['race_number'])].append(row)
    complete = [xs for xs in races.values() if abs(sum(r['probability'] for r in xs) - 1) <= .001
                and abs(sum(r['outcome'] for r in xs) - 1) <= .001]
    return {'groups': report, 'race_log_loss': (sum(-r['outcome'] * math.log(max(r['probability'], 1e-15))
                for xs in complete for r in xs) / len(complete) if complete else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=config.PROJECT_ROOT / 'results')
    parser.add_argument('--output', type=Path, default=config.PROJECT_ROOT / 'tracking_data/calibration_report.json')
    args = parser.parse_args()
    rows, exclusions = [], Counter()
    for path in sorted(args.results_dir.glob('race_data_*_with_results.json')):
        data = json.loads(path.read_text())
        for race in data['races']:
            eligible, reason = eligible_race(data['race_date'], race)
            rows.extend(eligible)
            if reason:
                exclusions[reason] += 1
    report = {'status': 'DIAGNOSTIC ONLY', 'dead_heat_convention': 'fractional win credit',
              'exclusions': dict(exclusions), 'all_runners': summarize(rows),
              'rank_one': summarize([r for r in rows if r['rank'] == 1])}
    atomic_json(args.output.with_name('calibration_records.json'), rows)
    atomic_json(args.output, report)
    print(f'Calibration: {len(rows)} eligible runners; exclusions={dict(exclusions)}; {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
