"""Meeting-wide going backfill: a blank per-race going must inherit the
meeting's modal going so the CHAOS soft-going filter fires consistently.

Regression for 2026/06/13 R5/R9 — those two races came back with going=''
(Playwright-fallback page-text miss) on a Yielding card and falsely escaped
CHAOS, staking bets they should not have.
"""
from dashboard.generator import backfill_meeting_going


class _Race:
    def __init__(self, going):
        self.going = going


def test_blank_going_inherits_modal():
    fetched = [_Race('Yielding'), _Race(''), _Race('Yielding'), _Race('')]
    backfill_meeting_going(fetched)
    assert [r.going for r in fetched] == ['Yielding', 'Yielding', 'Yielding', 'Yielding']


def test_modal_is_most_common_not_first():
    fetched = [_Race('Good'), _Race('Yielding'), _Race('Yielding'), _Race('')]
    backfill_meeting_going(fetched)
    assert fetched[-1].going == 'Yielding'  # modal, not the first-seen 'Good'


def test_existing_going_never_overwritten():
    fetched = [_Race('Yielding'), _Race('Good')]
    backfill_meeting_going(fetched)
    assert fetched[1].going == 'Good'


def test_all_blank_is_noop():
    fetched = [_Race(''), _Race('')]
    backfill_meeting_going(fetched)
    assert [r.going for r in fetched] == ['', '']


def test_exceptions_and_none_are_skipped():
    fetched = [_Race('Yielding'), Exception('boom'), None, _Race('')]
    backfill_meeting_going(fetched)  # must not raise
    assert fetched[0].going == 'Yielding'
    assert fetched[3].going == 'Yielding'
