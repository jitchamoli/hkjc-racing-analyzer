"""Tests for scripts/track_spoilers.py — empirical spoiler logger.

Pins the spoiler detection criteria (rank > 5 AND finished top-3) and
the dedup-on-rerun behavior so future refactors don't silently drift.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.track_spoilers import _rows_for_meeting, _write_rows, HEADERS, SPOILERS_PATH


def _race(rn, top_2=0.45, top_3=0.6, tier='HIGH', horses=None):
    return {
        'race_number': rn,
        'results_available': True,
        'top_2_share': top_2,
        'top_3_share': top_3,
        'conviction_tier': tier,
        'horses': horses or [],
    }


def _horse(num, rank, pos=None, odds=10.0, gear='B'):
    return {
        'number': num, 'rank': rank, 'actual_position': pos,
        'name': f'Horse{num}', 'win_odds': odds, 'gear': gear,
        'edge_pp': 1.5, 'ev_win': 0.05, 'calibrated_prob': 0.08,
    }


class TestSpoilerDetection:
    def test_rank_6_placed_top3_is_spoiler(self):
        races = [_race(1, horses=[
            _horse(1, 1, pos=4),
            _horse(2, 6, pos=2),  # spoiler: rank 6, placed 2nd
            _horse(3, 2, pos=1),
            _horse(4, 3, pos=3),
        ])]
        rows = _rows_for_meeting(races, '2026/05/17')
        assert len(rows) == 1
        assert rows[0]['horse_number'] == 2
        assert rows[0]['our_rank'] == 6
        assert rows[0]['finish_pos'] == 2

    def test_rank_5_inside_box_is_not_spoiler(self):
        # rank 5 is INSIDE the top-5 box, so it's not a spoiler even if placed
        races = [_race(1, horses=[
            _horse(1, 5, pos=1),
            _horse(2, 1, pos=2),
        ])]
        rows = _rows_for_meeting(races, '2026/05/17')
        assert len(rows) == 0

    def test_outside_top3_not_spoiler(self):
        # rank 7 finished 4th — outside top-3, not a spoiler
        races = [_race(1, horses=[
            _horse(1, 7, pos=4),
        ])]
        rows = _rows_for_meeting(races, '2026/05/17')
        assert len(rows) == 0

    def test_skips_races_without_results(self):
        races = [{
            'race_number': 1, 'results_available': False,
            'horses': [_horse(1, 6, pos=1)],
        }]
        rows = _rows_for_meeting(races, '2026/05/17')
        assert len(rows) == 0

    def test_skips_horses_without_position(self):
        races = [_race(1, horses=[
            _horse(1, 6, pos=None),  # no actual_position
            _horse(2, 7, pos=2),     # this is a spoiler
        ])]
        rows = _rows_for_meeting(races, '2026/05/17')
        assert len(rows) == 1
        assert rows[0]['horse_number'] == 2

    def test_multiple_spoilers_in_one_race(self):
        races = [_race(1, horses=[
            _horse(1, 1, pos=4),
            _horse(2, 8, pos=1),  # spoiler 1
            _horse(3, 6, pos=3),  # spoiler 2
            _horse(4, 2, pos=2),
        ])]
        rows = _rows_for_meeting(races, '2026/05/17')
        assert len(rows) == 2
        nums = {r['horse_number'] for r in rows}
        assert nums == {2, 3}

    def test_carries_race_level_context(self):
        races = [_race(1, top_2=0.42, top_3=0.65, tier='HIGH', horses=[
            _horse(1, 9, pos=2, odds=25.0, gear='B/TT'),
        ])]
        rows = _rows_for_meeting(races, '2026/05/17')
        r = rows[0]
        assert r['top_2_share'] == 0.42
        assert r['top_3_share'] == 0.65
        assert r['conviction_tier'] == 'HIGH'
        assert r['win_odds'] == 25.0
        assert r['gear'] == 'B/TT'


class TestDedup:
    def test_rerun_same_date_overwrites_prior_rows(self, tmp_path, monkeypatch):
        # Redirect SPOILERS_PATH to tmp
        import scripts.track_spoilers as ts
        monkeypatch.setattr(ts, 'SPOILERS_PATH', tmp_path / 'spoilers.csv')

        rows_v1 = [{h: 'x' for h in HEADERS}]
        rows_v1[0]['race_date'] = '2026/05/17'
        rows_v1[0]['race_number'] = '1'
        rows_v1[0]['horse_number'] = '99'  # mark as v1
        ts._write_rows(rows_v1, '2026/05/17')

        rows_v2 = [{h: 'x' for h in HEADERS}]
        rows_v2[0]['race_date'] = '2026/05/17'
        rows_v2[0]['race_number'] = '1'
        rows_v2[0]['horse_number'] = '88'  # mark as v2
        ts._write_rows(rows_v2, '2026/05/17')

        with open(ts.SPOILERS_PATH) as f:
            saved = list(csv.DictReader(f))
        # Only one row should remain (the v2 — v1 dropped on rerun)
        assert len(saved) == 1
        assert saved[0]['horse_number'] == '88'

    def test_different_dates_coexist(self, tmp_path, monkeypatch):
        import scripts.track_spoilers as ts
        monkeypatch.setattr(ts, 'SPOILERS_PATH', tmp_path / 'spoilers.csv')

        for date, num in [('2026/05/17', '11'), ('2026/05/20', '22')]:
            r = {h: '' for h in HEADERS}
            r.update({'race_date': date, 'race_number': '1', 'horse_number': num})
            ts._write_rows([r], date)

        with open(ts.SPOILERS_PATH) as f:
            saved = list(csv.DictReader(f))
        assert len(saved) == 2
        dates = sorted(r['race_date'] for r in saved)
        assert dates == ['2026/05/17', '2026/05/20']
