"""Locks the tier_b_shadow.py forward-validation logger.

Audit 2026-05-17 caught a HIGH-severity dedup bug: the script opened the
log in 'a' mode, so each fresh-odds re-run for the same date appended a
new copy of all races (1× → 3× in a single day, invalidating bet/hit/P&L
counts in the cumulative report).

These tests pin:
  - dedup on (race_date, race_number) across re-runs of the same date
  - bet_old vs bet_now logic (only differs when TRIO gate fires AND
    bet is not WIN/BANKER WIN — the cascade is WIN→TRIO→BOX*)
  - OLD vs NEW conviction tier transitions
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

import scripts.tier_b_shadow as tbs


@pytest.fixture
def temp_repo(monkeypatch, tmp_path):
    """Redirect REPO_ROOT + LOG_PATH so each test gets an isolated dir."""
    monkeypatch.setattr(tbs, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(tbs, 'LOG_PATH', tmp_path / 'tracking_data' / 'tier_b_log.csv')
    (tmp_path / 'results').mkdir()
    (tmp_path / 'tracking_data').mkdir()
    yield tmp_path


def _write_race_data(repo: Path, date: str, races: list) -> None:
    date_str = date.replace('/', '')
    path = repo / 'results' / f'race_data_{date_str}.json'
    with open(path, 'w') as f:
        json.dump(races, f)


def _read_log(repo: Path) -> list[dict]:
    log = repo / 'tracking_data' / 'tier_b_log.csv'
    if not log.exists():
        return []
    with open(log) as f:
        return list(csv.DictReader(f))


def _race(race_number: int, horses_data: list[dict]) -> dict:
    """Build a minimal race dict for tier_b_shadow.run() to consume."""
    return {
        'race_number': race_number,
        'has_edge': False,
        'odds_age_minutes': 0,
        'horses': horses_data,
    }


def _horse(rank: int, number: int, prob: float, win_odds: float = 5.0,
            bet: str = 'SKIP', edge_pp: float = -2.0,
            ev_win: float = -0.5, pure_prob: float = 0.10):
    return {
        'rank': rank, 'number': number, 'calibrated_prob': prob,
        'win_odds': win_odds, 'bet_recommendation': bet,
        'edge_pp': edge_pp, 'ev_win': ev_win, 'pure_prob': pure_prob,
    }


class TestDedup:
    def test_single_run_writes_one_row_per_race(self, temp_repo):
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.30, bet='QIN'),
                      _horse(2, 1, 0.20, bet='QIN'),
                      _horse(3, 5, 0.10)]),
        ])
        tbs.run('2026/05/17')
        rows = _read_log(temp_repo)
        assert len(rows) == 1
        assert rows[0]['race_number'] == '1'

    def test_re_run_same_date_does_not_duplicate(self, temp_repo):
        """The exact bug that prompted the audit fix."""
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.30, bet='QIN'),
                      _horse(2, 1, 0.20, bet='QIN'),
                      _horse(3, 5, 0.10)]),
        ])
        tbs.run('2026/05/17')
        tbs.run('2026/05/17')
        tbs.run('2026/05/17')
        rows = _read_log(temp_repo)
        # Three runs, same date — should still have exactly 1 row
        assert len(rows) == 1

    def test_different_date_appends_without_overwriting(self, temp_repo):
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.30, bet='QIN'),
                      _horse(2, 1, 0.20, bet='QIN'),
                      _horse(3, 5, 0.10)]),
        ])
        tbs.run('2026/05/17')
        _write_race_data(temp_repo, '2026/05/20', [
            _race(1, [_horse(1, 4, 0.25, bet='QIN'),
                      _horse(2, 6, 0.18, bet='QIN'),
                      _horse(3, 2, 0.12)]),
        ])
        tbs.run('2026/05/20')
        rows = _read_log(temp_repo)
        assert len(rows) == 2
        dates = sorted(r['race_date'] for r in rows)
        assert dates == ['2026/05/17', '2026/05/20']

    def test_re_run_one_date_preserves_other_dates(self, temp_repo):
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.30, bet='QIN'),
                      _horse(2, 1, 0.20, bet='QIN'),
                      _horse(3, 5, 0.10)]),
        ])
        _write_race_data(temp_repo, '2026/05/20', [
            _race(1, [_horse(1, 4, 0.25, bet='QIN'),
                      _horse(2, 6, 0.18, bet='QIN'),
                      _horse(3, 2, 0.12)]),
        ])
        tbs.run('2026/05/17')
        tbs.run('2026/05/20')
        tbs.run('2026/05/17')  # re-run the first date
        rows = _read_log(temp_repo)
        assert len(rows) == 2


class TestBetOldVsBetNow:
    def test_trio_gate_does_not_fire_means_same_bet(self, temp_repo):
        """Wide gaps → TRIO never would have fired → bet_old == bet_now."""
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.40, bet='QIN'),   # gap_1_2 = 0.20 (wide)
                      _horse(2, 1, 0.20, bet='QIN'),
                      _horse(3, 5, 0.10)]),
        ])
        tbs.run('2026/05/17')
        row = _read_log(temp_repo)[0]
        assert row['bet_now'] == row['bet_old'] == 'QIN'

    def test_trio_gate_fires_flat_field_means_bet_old_is_trio(self, temp_repo):
        """Tight gaps + top_3 > 0.32 → OLD TRIO would fire, NEW gives QIN/BOX."""
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.16, bet='QIN'),   # gap_1_2 = 0.00
                      _horse(2, 1, 0.16, bet='QIN'),    # gap_2_3 = 0.01
                      _horse(3, 5, 0.15)]),             # top_3 = 0.47 > 0.32
        ])
        tbs.run('2026/05/17')
        row = _read_log(temp_repo)[0]
        assert row['bet_old'] == 'TRIO'   # OLD recommender would have shipped TRIO
        assert row['bet_now'] != 'TRIO'   # NEW recommender did not (disabled)

    def test_win_bet_unaffected_by_trio_disable(self, temp_repo):
        """If bet_now is WIN (cascade fires before TRIO), bet_old == WIN too."""
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.40, bet='BANKER WIN'),
                      _horse(2, 1, 0.20),
                      _horse(3, 5, 0.10)]),
        ])
        tbs.run('2026/05/17')
        row = _read_log(temp_repo)[0]
        assert row['bet_now'] == 'WIN'  # rolled up from BANKER WIN
        assert row['bet_old'] == 'WIN'


class TestConvictionTierTransitions:
    def test_old_high_can_become_new_med(self, temp_repo):
        """top_2 = 0.36 → OLD HIGH (≥0.33), NEW MED (<0.40)."""
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.20, bet='QIN'),
                      _horse(2, 1, 0.16, bet='QIN'),
                      _horse(3, 5, 0.10)]),
        ])
        tbs.run('2026/05/17')
        row = _read_log(temp_repo)[0]
        # top_2_share = 0.36 → OLD HIGH (≥0.33) but NEW MED (<0.40, ≥0.30)
        assert row['tier_old'] == 'HIGH'
        assert row['tier_new'] == 'MED'

    def test_high_tier_unchanged_when_strong_enough(self, temp_repo):
        """top_2 = 0.50 → HIGH in both old (≥0.33) and new (≥0.40)."""
        _write_race_data(temp_repo, '2026/05/17', [
            _race(1, [_horse(1, 8, 0.30, bet='QIN'),
                      _horse(2, 1, 0.20, bet='QIN'),
                      _horse(3, 5, 0.10)]),
        ])
        tbs.run('2026/05/17')
        row = _read_log(temp_repo)[0]
        assert row['tier_old'] == row['tier_new'] == 'HIGH'

    def test_edge_takes_precedence_over_both(self, temp_repo):
        _write_race_data(temp_repo, '2026/05/17', [{
            'race_number': 1, 'has_edge': True, 'odds_age_minutes': 0,
            'horses': [_horse(1, 8, 0.30, bet='QIN'),
                       _horse(2, 1, 0.20, bet='QIN'),
                       _horse(3, 5, 0.10)],
        }])
        tbs.run('2026/05/17')
        row = _read_log(temp_repo)[0]
        assert row['tier_old'] == 'EDGE'
        assert row['tier_new'] == 'EDGE'
