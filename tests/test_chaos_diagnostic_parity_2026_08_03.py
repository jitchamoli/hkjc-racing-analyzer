"""The CHAOS diagnostic must measure the filter production actually ships.

Bug (code review 2026-08-03): `scripts/chaos_diagnostic_db.is_chaos_db` did a
substring match — `'YIELDING' in g` — so 'Good To Yielding' (and its short
code GY) FIRED. Production `betting.conviction.detect_chaos` was deliberately
NARROWED on 2026-07-15 to EXCLUDE GY, because on 154 full-DB races the
favourite strikes ~31% on GY, indistinguishable from firm.

So the diagnostic was measuring a strictly wider filter than the one deployed.
The "~-7pp expected lift" figure quoted in CLAUDE.md came from this script.
A diagnostic that disagrees with production produces numbers that cannot
inform a decision about production.

Fix: delegate to `detect_chaos` so the two can never diverge again. The
deliberately-augmented keyword sweep (RAIN / WET, labels that never appear in
live race-card JSON) stays available under an explicitly different name so it
is never mistaken for the shipped filter.
"""
import pytest

from betting.conviction import detect_chaos
from scripts.chaos_diagnostic_db import is_chaos_db


GOINGS = [
    'Good', 'Good To Firm', 'Firm', 'Fast', 'Good (AWT)', 'Sealed',
    'Good To Yielding', 'GY',
    'Yielding', 'Y', 'Yielding To Soft', 'YS', 'Soft', 'S',
    'Heavy', 'H', 'Wet Slow', 'WS', 'Wet-Fast', 'WF',
]


class TestDiagnosticMatchesProduction:
    @pytest.mark.parametrize('going', GOINGS)
    def test_same_verdict_as_production(self, going):
        prod = detect_chaos(going)[0]
        assert is_chaos_db(going) == prod, (
            f"diagnostic and production disagree on {going!r}: "
            f"diagnostic={is_chaos_db(going)} production={prod}"
        )

    def test_good_to_yielding_is_not_chaos(self):
        """The specific narrowing that the diagnostic was ignoring."""
        assert is_chaos_db('Good To Yielding') is False
        assert is_chaos_db('GY') is False

    def test_genuinely_soft_going_still_fires(self):
        for g in ('Yielding', 'Soft', 'Heavy', 'Wet Slow'):
            assert is_chaos_db(g) is True, f"{g} should be CHAOS"

    def test_firm_going_never_fires(self):
        for g in ('Good', 'Good To Firm', 'Firm', 'Fast'):
            assert is_chaos_db(g) is False, f"{g} should not be CHAOS"
