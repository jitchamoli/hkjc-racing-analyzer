"""scripts/validate_hk — _load_cutoff format validation.

PR #52 strict code review HIGH #2: an empty or corrupted cutoff file
silently makes `race_date > ''` true for every row, contaminating the
un-burned forward-validation evidence with all burned pre-cutoff data.
The fix raises SystemExit with a clear message on malformed cutoffs.
"""
from __future__ import annotations

import pytest

from scripts.validate_hk import _load_cutoff


class _StubConn:
    """Minimal sqlite3-shaped stub. _load_cutoff only calls execute(...)
    in the fallback path (when CUTOFF_FILE doesn't exist), so tests that
    create the file should never reach this."""
    def execute(self, *_args, **_kwargs):
        raise AssertionError(
            '_load_cutoff should not query DB when CUTOFF_FILE is present'
        )


@pytest.fixture
def cutoff_path(tmp_path, monkeypatch):
    p = tmp_path / 'hk_forward_cutoff.txt'
    monkeypatch.setattr('scripts.validate_hk.CUTOFF_FILE', p)
    return p


def test_load_cutoff_accepts_valid_date(cutoff_path):
    cutoff_path.write_text('2026/05/20')
    assert _load_cutoff(_StubConn()) == '2026/05/20'


def test_load_cutoff_strips_whitespace(cutoff_path):
    cutoff_path.write_text('  2026/05/20\n')
    assert _load_cutoff(_StubConn()) == '2026/05/20'


def test_load_cutoff_rejects_empty_file(cutoff_path):
    cutoff_path.write_text('')
    with pytest.raises(SystemExit, match='not a YYYY/MM/DD date'):
        _load_cutoff(_StubConn())


def test_load_cutoff_rejects_whitespace_only(cutoff_path):
    cutoff_path.write_text('   \n\t  ')
    with pytest.raises(SystemExit, match='not a YYYY/MM/DD date'):
        _load_cutoff(_StubConn())


@pytest.mark.parametrize('bad', [
    '2026-05-20',        # wrong separator
    '20260520',          # no separator
    '5/20/2026',         # wrong order
    '2026/5/20',         # missing zero pad
    '2026/05/20 garbage',  # trailing junk
    'today',             # not a date at all
    '2026/05',           # incomplete
])
def test_load_cutoff_rejects_malformed(cutoff_path, bad):
    cutoff_path.write_text(bad)
    with pytest.raises(SystemExit, match='not a YYYY/MM/DD date'):
        _load_cutoff(_StubConn())
