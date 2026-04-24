"""Snapshot-diff alert generation for the live rail.

State is carried across ticks in AlertState. Each (race, horse, type)
pair fires at most once per day — once EDGE_CLOSED emits for a race,
we also mark EDGE_OPENED as already-emitted for that race so the window
doesn't re-open on a subsequent tick. See spec §7.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple


@dataclass
class AlertState:
    """Carries dedup state across refresh ticks within a day."""
    emitted: Set[Tuple[int, int, str]] = field(default_factory=set)


def _already_emitted(state: AlertState, race: int, horse: int, typ: str) -> bool:
    return (race, horse, typ) in state.emitted


def _mark(state: AlertState, race: int, horse: int, typ: str) -> None:
    state.emitted.add((race, horse, typ))


def compute_alerts(current_races: List[Dict], previous_races: List[Dict],
                   state: AlertState) -> List[Dict]:
    """Diff current vs previous snapshots → alerts list (newest-first).

    Alert dict: type, race, horse (0 for race-level), ts, severity, detail.
    """
    prev_by_race = {r['race_number']: r for r in previous_races}
    out = []
    ts = int(time.time())

    for curr in current_races:
        race_num = curr['race_number']
        prev = prev_by_race.get(race_num, {})
        prev_horses = {h['number']: h for h in prev.get('horses', [])}

        # EDGE_OPENED / EDGE_CLOSED / EDGE_WIDENED (race-level, horse=0)
        curr_edge = curr.get('has_edge', False)
        prev_edge = prev.get('has_edge', False)
        if curr_edge and not prev_edge and not _already_emitted(state, race_num, 0, 'EDGE_OPENED'):
            out.append({'type': 'EDGE_OPENED', 'race': race_num, 'horse': 0,
                        'ts': ts, 'severity': 'strong', 'detail': None})
            _mark(state, race_num, 0, 'EDGE_OPENED')
        elif prev_edge and not curr_edge and not _already_emitted(state, race_num, 0, 'EDGE_CLOSED'):
            out.append({'type': 'EDGE_CLOSED', 'race': race_num, 'horse': 0,
                        'ts': ts, 'severity': 'mild', 'detail': None})
            _mark(state, race_num, 0, 'EDGE_CLOSED')
            _mark(state, race_num, 0, 'EDGE_OPENED')  # suppress same-day re-open

        if curr_edge and prev_edge:
            curr_best_pp = max((h.get('edge_pp', 0) for h in curr['horses']), default=0)
            prev_best_pp = max((h.get('edge_pp', 0) for h in prev.get('horses', [])), default=0)
            if (curr_best_pp - prev_best_pp) >= 2.0 and not _already_emitted(state, race_num, 0, 'EDGE_WIDENED'):
                out.append({'type': 'EDGE_WIDENED', 'race': race_num, 'horse': 0,
                            'ts': ts, 'severity': 'strong',
                            'detail': f'+{curr_best_pp - prev_best_pp:.1f}pp'})
                _mark(state, race_num, 0, 'EDGE_WIDENED')

        # STEAM / COLD per-horse
        for h in curr['horses']:
            num = h['number']
            curr_odds = h.get('win_odds', 0)
            prev_h = prev_horses.get(num, {})
            prev_odds = prev_h.get('win_odds', 0)
            if not (prev_odds and curr_odds):
                continue

            pct_change = (curr_odds - prev_odds) / prev_odds
            if pct_change <= -0.25 and not _already_emitted(state, race_num, num, 'STEAM'):
                out.append({'type': 'STEAM', 'race': race_num, 'horse': num,
                            'ts': ts, 'severity': 'strong',
                            'detail': f'{prev_odds:.1f} → {curr_odds:.1f}'})
                _mark(state, race_num, num, 'STEAM')
            elif pct_change >= 0.40 and not _already_emitted(state, race_num, num, 'COLD'):
                prev_sorted = sorted(
                    (ph for ph in prev.get('horses', []) if ph.get('win_odds', 0) > 0),
                    key=lambda x: x['win_odds'],
                )
                if prev_sorted and num in {ph['number'] for ph in prev_sorted[:2]}:
                    out.append({'type': 'COLD', 'race': race_num, 'horse': num,
                                'ts': ts, 'severity': 'mild',
                                'detail': f'{prev_odds:.1f} → {curr_odds:.1f}'})
                    _mark(state, race_num, num, 'COLD')

        # RACE_UPCOMING
        mtp = curr.get('minutes_to_post')
        if mtp is not None and 0 <= mtp <= 6 and not _already_emitted(state, race_num, 0, 'RACE_UPCOMING'):
            out.append({'type': 'RACE_UPCOMING', 'race': race_num, 'horse': 0,
                        'ts': ts, 'severity': 'mild', 'detail': f'{mtp}m to post'})
            _mark(state, race_num, 0, 'RACE_UPCOMING')

    return list(reversed(out))  # newest-first within tick
