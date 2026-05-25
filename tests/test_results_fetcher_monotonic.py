"""Monotonic-merge contract for `dashboard/results_fetcher.py`
(GPT-5.5 R11 2026-05-25).

Bug: `load_prediction_data()` loaded only the bare `race_data_YYYYMMDD.json`
and `merge_results_with_predictions()` set `results_available=False` for
every race missing from the fresh fetch. When `save_merged_data()`
overwrote `_with_results.json`, any race transiently dropped by the HKJC
scraper LOST its previously-merged result block (actual_results,
actual_winner, actual_top3, actual_win_odds, accuracy, per-horse
actual_position).

Fix: load the existing `_with_results.json` via `load_existing_merged()`
and pass it into `merge_results_with_predictions()`; races missing from
the fresh fetch but present in the existing merged file keep their prior
result fields (monotonic).

These tests pin the contract.
"""
from __future__ import annotations
import json
import os
import tempfile

import pytest

from dashboard.results_fetcher import ResultsFetcher


def _fresh_fetcher(tmp_path):
    """A real fetcher pointed at an isolated results dir."""
    f = ResultsFetcher.__new__(ResultsFetcher)
    f.results_dir = str(tmp_path)
    return f


def _race(race_no, with_horses=3):
    return {
        'race_number': race_no,
        'horses': [
            {'number': i + 1, 'name': f'H{i+1}', 'rank': i + 1,
             'bet_recommendation': 'QIN', 'win_odds': 5.0 + i}
            for i in range(with_horses)
        ],
    }


def _result_row(num, position, win_odds=5.0):
    return {'number': num, 'name': f'H{num}', 'position': position,
            'win_odds': win_odds}


def test_missing_race_in_fresh_fetch_preserves_prior_result(tmp_path):
    """The headline regression: R1 already merged; fresh fetch returns only
    R2; merge must KEEP R1's results not flip results_available=False."""
    fetcher = _fresh_fetcher(tmp_path)
    # Seed the existing _with_results.json with R1+R2 both merged.
    existing = {
        'race_date': '2026/05/24',
        'updated_at': '2026-05-24 18:00:00',
        'day_summary': {},
        'races': [
            {**_race(1), 'results_available': True,
             'actual_results': [_result_row(1, 1), _result_row(2, 2),
                                _result_row(3, 3)],
             'actual_winner': 1, 'actual_top3': [1, 2, 3],
             'actual_win_odds': 5.0, 'accuracy': {'win_correct': True},
             'horses': [
                 {'number': 1, 'name': 'H1', 'rank': 1, 'actual_position': 1},
                 {'number': 2, 'name': 'H2', 'rank': 2, 'actual_position': 2},
                 {'number': 3, 'name': 'H3', 'rank': 3, 'actual_position': 3},
             ]},
            {**_race(2), 'results_available': True,
             'actual_results': [_result_row(2, 1, 3.0)],
             'actual_winner': 2, 'actual_top3': [2],
             'actual_win_odds': 3.0, 'accuracy': {}},
        ],
    }
    (tmp_path / 'race_data_20260524_with_results.json').write_text(json.dumps(existing))

    # Now simulate a re-run where the fresh predictions JSON has both races
    # but the HKJC fetch only returned R2.
    predictions = [_race(1), _race(2)]
    actual_results_fresh = {
        2: [_result_row(2, 1, 3.0)],  # Only R2 returned this time
    }

    loaded_existing = fetcher.load_existing_merged('2026/05/24')
    assert 1 in loaded_existing and 2 in loaded_existing, \
        "load_existing_merged must surface both previously-merged races"

    merged = fetcher.merge_results_with_predictions(
        predictions, actual_results_fresh, loaded_existing
    )

    r1 = next(r for r in merged if r['race_number'] == 1)
    assert r1['results_available'] is True, (
        "R1 had results merged previously; transient miss must NOT flip it back to False"
    )
    assert r1['actual_winner'] == 1
    assert r1['actual_top3'] == [1, 2, 3]
    assert r1['actual_win_odds'] == 5.0
    # Per-horse actual_position must also be restored
    positions = {h['number']: h.get('actual_position') for h in r1['horses']}
    assert positions == {1: 1, 2: 2, 3: 3}, (
        f"actual_position lost on horses when restoring R1: {positions!r}"
    )


def test_fresh_fetch_takes_priority_over_existing_for_same_race(tmp_path):
    """When HKJC republishes a corrected result, the fresh fetch must
    overwrite the existing merged record — monotonic ≠ frozen."""
    fetcher = _fresh_fetcher(tmp_path)
    existing = {
        'race_date': '2026/05/24', 'updated_at': '', 'day_summary': {},
        'races': [{**_race(1), 'results_available': True,
                   'actual_results': [_result_row(1, 1)],
                   'actual_winner': 1, 'actual_top3': [1],
                   'actual_win_odds': 5.0,
                   'horses': [{'number': 1, 'name': 'H1', 'rank': 1,
                               'actual_position': 1}]}],
    }
    (tmp_path / 'race_data_20260524_with_results.json').write_text(json.dumps(existing))

    predictions = [_race(1)]
    # Fresh fetch reports a DIFFERENT winner — e.g., HKJC corrected DQ.
    actual_results_fresh = {
        1: [_result_row(2, 1, 7.0), _result_row(1, 2, 5.0), _result_row(3, 3)],
    }
    loaded = fetcher.load_existing_merged('2026/05/24')
    merged = fetcher.merge_results_with_predictions(
        predictions, actual_results_fresh, loaded
    )
    r1 = next(r for r in merged if r['race_number'] == 1)
    assert r1['actual_winner'] == 2, "fresh fetch must overwrite existing for races it returns"
    assert r1['actual_win_odds'] == 7.0


def test_missing_race_with_no_prior_result_stays_unavailable(tmp_path):
    """Race that was NEVER merged before AND is missing from fresh fetch
    must still flip results_available=False (no spurious True)."""
    fetcher = _fresh_fetcher(tmp_path)
    # No existing _with_results.json at all.
    predictions = [_race(1), _race(2)]
    actual_results_fresh = {2: [_result_row(2, 1)]}
    loaded = fetcher.load_existing_merged('2026/05/24')  # returns {}
    merged = fetcher.merge_results_with_predictions(
        predictions, actual_results_fresh, loaded
    )
    r1 = next(r for r in merged if r['race_number'] == 1)
    assert r1['results_available'] is False
    assert 'actual_winner' not in r1


def test_load_existing_merged_returns_empty_when_file_absent(tmp_path):
    fetcher = _fresh_fetcher(tmp_path)
    assert fetcher.load_existing_merged('2026/05/24') == {}


def test_load_existing_merged_skips_races_without_results_available(tmp_path):
    """Half-merged races (results_available=False) must NOT be returned as
    "frozen" — otherwise we'd restore a bogus result block on the next merge."""
    fetcher = _fresh_fetcher(tmp_path)
    existing = {
        'race_date': '2026/05/24', 'updated_at': '', 'day_summary': {},
        'races': [
            {**_race(1), 'results_available': True,
             'actual_results': [_result_row(1, 1)], 'actual_winner': 1,
             'horses': [{'number': 1, 'name': 'H1', 'rank': 1,
                         'actual_position': 1}]},
            {**_race(2), 'results_available': False},
        ],
    }
    (tmp_path / 'race_data_20260524_with_results.json').write_text(json.dumps(existing))
    loaded = fetcher.load_existing_merged('2026/05/24')
    assert 1 in loaded
    assert 2 not in loaded, "results_available=False race must not be cached as frozen"


def test_load_existing_merged_tolerates_corrupt_json(tmp_path):
    """A truncated/corrupt _with_results must not crash the next results
    fetch — return empty dict and continue (the fresh fetch results are
    still the primary source of truth)."""
    fetcher = _fresh_fetcher(tmp_path)
    (tmp_path / 'race_data_20260524_with_results.json').write_text('{ not valid json')
    assert fetcher.load_existing_merged('2026/05/24') == {}


def test_merge_works_without_existing_arg(tmp_path):
    """Backward-compat: existing_merged defaults to None / empty so callers
    that don't pass it (other scripts, ad-hoc usage) keep working."""
    fetcher = _fresh_fetcher(tmp_path)
    predictions = [_race(1), _race(2)]
    actual_results_fresh = {1: [_result_row(1, 1)]}
    merged = fetcher.merge_results_with_predictions(predictions, actual_results_fresh)
    r1 = next(r for r in merged if r['race_number'] == 1)
    r2 = next(r for r in merged if r['race_number'] == 2)
    assert r1['results_available'] is True
    assert r2['results_available'] is False
