"""Isolated chronological top-four ranking diagnostic; never saves a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import comb
import re
import sqlite3

import numpy as np
from xgboost import XGBClassifier, XGBRanker

import config
from ml.training import build_training_data_from_db
from scripts.calibration_report import atomic_json, block_interval, eligible_race


def meeting_folds(dates, n_folds=3):
    meetings = np.array(sorted(set(dates)))
    start = len(meetings) // 2
    if start < 3 or len(meetings) - start < n_folds:
        raise ValueError('Not enough meetings for chronological folds')
    for test_dates in np.array_split(meetings[start:], n_folds):
        prior = meetings[meetings < test_dates[0]]
        split = max(1, int(len(prior) * .85))
        yield (np.flatnonzero(np.isin(dates, prior[:split])),
               np.flatnonzero(np.isin(dates, prior[split:])),
               np.flatnonzero(np.isin(dates, test_dates)))


def capture(order, positions):
    chosen = set(order[:5])
    return {'winner_top1': positions[order[0]] == 1,
            'winner_in5': any(positions[n] == 1 for n in chosen),
            'top3_in5': {n for n, p in positions.items() if p <= 3} <= chosen,
            'top4_in5': {n for n, p in positions.items() if p <= 4} <= chosen}


def settle_box(legs, dividends, need, unit=10):
    # Official dividends are declared per HK$10, including tied combinations.
    chosen = set(legs)
    payout = sum(float(div) * unit / 10 for combo, div in dividends
                 if len(ns := set(map(int, re.findall(r'\d+', combo)))) == need and ns <= chosen)
    return comb(len(chosen), need) * unit, payout


def pair_ticket(bet, order, dividends):
    need = {'TRIO': 3, 'FIRST 4': 4}.get(bet.get('pool'))
    if need is None or not dividends:
        return None
    try:
        legs = bet['legs']
        old = [int(l['number']) for l in legs]
        ranks = [int(l['rank']) for l in legs]
        unit = float(bet['unit_stake'])
        if (unit <= 0 or len(set(old)) != len(old) or len(set(ranks)) != len(ranks)
                or min(ranks) < 1 or max(ranks) > len(order)
                or len(old) < need or bet['combos'] != comb(len(old), need)):
            return None
        new = [order[i - 1] for i in ranks]
        return {'label': bet['label'], 'paper': bool(bet.get('paper')),
                'production': settle_box(old, dividends, need, unit),
                'ranker': settle_box(new, dividends, need, unit)}
    except (KeyError, TypeError, ValueError):
        return None


def summarize_pairs(rows):
    report = {}
    for metric in ('winner_top1', 'winner_in5', 'top3_in5', 'top4_in5'):
        delta = lambda xs: float(np.mean([x['ranker'][metric] - x['classifier'][metric] for x in xs]))
        report[metric] = {'classifier': float(np.mean([r['classifier'][metric] for r in rows])),
                          'ranker': float(np.mean([r['ranker'][metric] for r in rows])),
                          'paired_delta_ci95': block_interval(rows, delta)}
    for pool in ('TRIO', 'FIRST 4'):
        xs = [r for r in rows if pool in r['settlement']]
        if not xs:
            continue
        arms = {}
        for arm in ('classifier', 'ranker'):
            stake = sum(r['settlement'][pool][arm][0] for r in xs)
            payout = sum(r['settlement'][pool][arm][1] for r in xs)
            arms[arm] = {'stake': stake, 'payout': payout, 'roi': payout / stake - 1 if stake else None}
        delta = lambda ys: float(np.mean([r['settlement'][pool]['ranker'][1] - r['settlement'][pool]['classifier'][1] for r in ys]))
        report[pool] = {'races': len(xs), **arms, 'paired_payout_delta_ci95': block_interval(xs, delta)}
    return report


def run(n_folds, years_back):
    data = build_training_data_from_db(years_back=years_back, return_metadata=True)
    if len(data) != 6 or data[0] is None:
        raise ValueError('Runner-keyed research data unavailable')
    X, y, _, _, groups, meta = data
    if len(meta) != len(X) or not np.isfinite(X).all():
        raise ValueError('Invalid research feature/runner alignment')
    keys = [(m['race_date'], m['race_number'], m['horse_number']) for m in meta]
    counts = Counter(keys)
    bad_groups = {groups[i] for i, key in enumerate(keys) if counts[key] > 1 or not key[2]}
    keep = np.array([i for i, g in enumerate(groups) if g not in bad_groups])
    X, y, groups = X[keep], y[keep], groups[keep]
    meta = [meta[i] for i in keep]
    dates = np.array([m['race_date'] for m in meta])
    positions = np.array([m['finish_position'] for m in meta])
    relevance = (positions <= 4).astype(int)
    # Fixed settings: no evaluation-fold tuning or full-dataset recency weights.
    params = dict(n_estimators=100, max_depth=3, learning_rate=.05,
                  subsample=1, colsample_bytree=1, random_state=42, n_jobs=2)
    conn = sqlite3.connect(config.DB_PATH)
    divs = defaultdict(list)
    for day, race, pool, combo, dividend in conn.execute(
            "SELECT race_date,race_number,pool,combination,dividend FROM race_dividends WHERE pool IN ('TRIO','FIRST 4')"):
        divs[(day, race, pool)].append((combo, dividend))
    conn.close()
    frozen = {}
    for path in sorted((config.PROJECT_ROOT / 'results').glob('race_data_*_with_results.json')):
        data = json.loads(path.read_text())
        for race in data['races']:
            eligible, _ = eligible_race(data['race_date'], race)
            if eligible:
                frozen[(data['race_date'], race['race_number'])] = race['bet_ticket']
    rows, folds, exclusions = [], [], Counter()
    exclusions['invalid_runner_key_races'] = len(bad_groups)
    for fold, (train, val, test) in enumerate(meeting_folds(dates, n_folds)):
        assert max(dates[train]) < min(dates[val]) < min(dates[test])
        _, counts = np.unique(groups[train], return_counts=True)
        # Each contiguous race group has one label for every runner.
        sizes = [len(list(ix)) for ix in np.split(train, np.flatnonzero(groups[train][1:] != groups[train][:-1]) + 1)]
        assert len(sizes) == len(counts)
        classifier = XGBClassifier(objective='binary:logistic', **params)
        ranker = XGBRanker(objective='rank:ndcg', lambdarank_pair_method='topk',
                           lambdarank_num_pair_per_sample=4, **params)
        classifier.fit(X[train], y[train])
        ranker.fit(X[train], relevance[train], group=sizes)
        scores = {'classifier': classifier.predict_proba(X[test])[:, 1],
                  'ranker': ranker.predict(X[test])}
        folds.append({'fold': fold, 'train_until': max(dates[train]),
                      'reserved_validation_until': max(dates[val]),
                      'test_from': min(dates[test]), 'test_until': max(dates[test]),
                      'training_rows': len(train), 'training_races': len(sizes)})
        by_race = defaultdict(list)
        for local, global_idx in enumerate(test):
            by_race[(meta[global_idx]['race_date'], meta[global_idx]['race_number'])].append((local, global_idx))
        for (day, race), entries in by_race.items():
            pos = {meta[g]['horse_number']: int(positions[g]) for _, g in entries}
            if len(pos) < 5 or max(pos.values()) > len(pos) or 1 not in pos.values():
                exclusions['incomplete_field'] += 1
                continue
            orders = {arm: [meta[g]['horse_number'] for l, g in sorted(entries, key=lambda e: (-float(s[e[0]]), meta[e[1]]['horse_number']))]
                      for arm, s in scores.items()}
            row = {'race_date': day, 'race_number': race, 'fold': fold,
                   **{arm: {k: int(v) for k, v in capture(order, pos).items()} for arm, order in orders.items()},
                   'orders': orders, 'settlement': {}}
            for pool, need in (('TRIO', 3), ('FIRST 4', 4)):
                declared = divs.get((day, race, pool))
                if declared:
                    row['settlement'][pool] = {arm: settle_box(order[:5], declared, need) for arm, order in orders.items()}
                else:
                    exclusions['missing_' + pool] += 1
            row['paired_frozen_tickets'] = []
            for bet in frozen.get((day, race), {}).get('bets', []):
                declared = divs.get((day, race, bet.get('pool')))
                pair = pair_ticket(bet, orders['ranker'], declared)
                if pair is not None and all(l['number'] in pos for l in bet['legs']):
                    row['paired_frozen_tickets'].append(pair)
            rows.append(row)
        print(f'Fold {fold + 1}/{n_folds}: {len(train)} train rows, {len(test)} test rows', flush=True)
    if not rows:
        raise ValueError('No eligible walk-forward races')
    paired = {}
    for paper in (False, True):
        bets = [b for r in rows for b in r['paired_frozen_tickets'] if b['paper'] == paper]
        arms = {}
        for arm in ('production', 'ranker'):
            stake = sum(b[arm][0] for b in bets)
            payout = sum(b[arm][1] for b in bets)
            arms[arm] = {'stake': stake, 'payout': payout, 'roi': payout / stake - 1 if stake else None}
        paired_rows = []
        for row in rows:
            bs = [b for b in row['paired_frozen_tickets'] if b['paper'] == paper]
            if bs:
                paired_rows.append({'race_date': row['race_date'],
                                    'delta': sum(b['ranker'][1] - b['production'][1] for b in bs),
                                    'stake': sum(b['production'][0] for b in bs)})
        roi_delta = lambda xs: sum(x['delta'] for x in xs) / sum(x['stake'] for x in xs)
        paired['paper' if paper else 'system'] = {'tickets': len(bets), **arms,
                'meetings': len({r['race_date'] for r in paired_rows}),
                'paired_roi_delta_ci95': block_interval(paired_rows, roi_delta)}
    return {'status': 'HISTORICAL COUNTERFACTUAL ONLY - NOT PRODUCTION REPLAY',
            'objective': 'rank:ndcg with binary top-four relevance; not calibrated win probability',
            'feature_source': 'production as-of-date features, no odds feature; actual starters only',
            'limitations': ['Declared-field withdrawals are not reconstructed; prospective tickets required before promotion.',
                            'Fixed HK$10 top-five boxes; not a replay of production gate decisions.',
                            'Historical feature-source corrections may not represent data known at the time.'],
            'params': params, 'folds': folds, 'exclusions': dict(exclusions), 'paired_frozen_replay': paired,
            'summary': summarize_pairs(rows), 'races': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--years-back', type=int, default=2)
    p.add_argument('--output', type=str, default=str(config.PROJECT_ROOT / 'tracking_data/top4_diagnostic.json'))
    args = p.parse_args()
    if args.folds < 1 or args.years_back < 1:
        p.error('folds and years-back must be positive')
    report = run(args.folds, args.years_back)
    atomic_json(args.output, report)
    print(report['summary'])
    print(f'Research only: {args.output}')


if __name__ == '__main__':
    main()
