"""`validate_hk --replay --report` must re-score, not serve a stale CSV.

Found 2026-10-04 restoring the emptied 2026/09/16 R7 ticket: after writing
the ticket back, `--replay --report` printed byte-identical totals, because
the re-score loop sits inside `if not args.report:` and `--report` reads
`hk_forward_validation_replay.csv` as-is.

That is the exact failure the loop's own comment says it exists to prevent —
dividends land via a separate backfill, so any meeting first scored before
its dividends arrived stays frozen as a total loss. `--report` is the
flag someone reaches for when they want the cumulative number, so it is
the worst place to serve stale scores.
"""
from __future__ import annotations

import pytest

import scripts.validate_hk as vh


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Stub out the DB and both report renderers; record re-score calls."""
    scored: list[str] = []
    written: list[str] = []

    monkeypatch.setattr(vh, '_load_cutoff', lambda conn: '2026/05/20')
    monkeypatch.setattr(vh, '_meeting_dates',
                        lambda conn, after=None, only_burned_through=None:
                        ['2026/09/16', '2026/10/01'])
    monkeypatch.setattr(vh, '_read_csv', lambda: [])
    monkeypatch.setattr(vh, '_read_replay_csv', lambda: [])
    monkeypatch.setattr(vh, '_append_csv', lambda d, acc: None)
    monkeypatch.setattr(vh, '_score_meeting',
                        lambda conn, d: {s[0]: (0.0, 0.0, 0) for s in vh.STRATS} | {'_races': 0})

    def _fake_score_replay(conn, d):
        scored.append(d)
        return {'labels': {}, '_races': 1}

    def _fake_write(d, replay):
        written.append(d)

    monkeypatch.setattr(vh, '_score_meeting_replay', _fake_score_replay)
    monkeypatch.setattr(vh, '_write_replay_row', _fake_write)
    monkeypatch.setattr(vh, '_report', lambda rows, title: None)
    monkeypatch.setattr(vh, '_report_replay', lambda rows, title: None)

    class _Conn:
        row_factory = None
        def execute(self, *a, **k): raise AssertionError('no direct DB use expected')
        def close(self): pass

    monkeypatch.setattr(vh.sqlite3, 'connect', lambda *a, **k: _Conn())
    return scored, written


def test_replay_with_report_rescores_every_meeting(harness, monkeypatch):
    scored, written = harness
    monkeypatch.setattr('sys.argv', ['validate_hk', '--replay', '--report'])

    assert vh.main() == 0

    assert scored == ['2026/09/16', '2026/10/01'], (
        f"--replay --report did not re-score; scored={scored!r}. A meeting whose "
        f"dividends or ticket changed after first scoring would stay stale."
    )
    assert written == ['2026/09/16', '2026/10/01']


def test_replay_without_report_still_rescores(harness, monkeypatch):
    """Regression guard: the existing (working) path must keep working."""
    scored, written = harness
    monkeypatch.setattr('sys.argv', ['validate_hk', '--replay'])

    assert vh.main() == 0
    assert scored == ['2026/09/16', '2026/10/01']
    assert written == ['2026/09/16', '2026/10/01']


def test_report_without_replay_does_not_rescore(harness, monkeypatch):
    """--report alone stays a cheap read of the synthetic CSV."""
    scored, written = harness
    monkeypatch.setattr('sys.argv', ['validate_hk', '--report'])

    assert vh.main() == 0
    assert scored == []
    assert written == []
