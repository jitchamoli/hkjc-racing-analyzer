r"""`last_6` arrives in three formats; only two were parsed correctly.

    live HKJC race card   '5/5/1/2/1/1'    slash-separated
    Renavon DB backfill   '4 11 9 5 0 8'   SPACE-separated   <-- was broken
    legacy concatenated   '13425'          digit-per-position

`parse_form_positions` branched on '/' and otherwise iterated CHARACTERS, so a
space-separated string with any double-digit placing split that placing into
two positions. '4 11 9 5 0 8' (4th, 11th, 9th, 5th, unplaced, 8th) parsed as
[4, 1, 1, 9, 5, 0, 8] — seven runs, and the 11th became two apparent WINS.

Scale: 95% of race_results rows are space-separated (Renavon wrote spaces for
2021-2025; the HKJC scraper writes slashes), and 32,696 of 54,662 last_6 values
(60%) contain a two-digit placing. It stayed hidden because production scores
the live card and the backtest's held-out fold is 2026 — both slash-form — so
the broken branch was never exercised by a current caller.

The legacy concatenated form must KEEP its digit-by-digit reading: '13425' is
five separate placings, not the number 13,425. Only the whitespace case changes.
"""
import pytest

from analytics import parse_form_positions


class TestSpaceSeparatedForm:
    def test_double_digit_placing_stays_one_position(self):
        """The bug: '11' must be eleventh, not two firsts."""
        assert parse_form_positions('4 11 9 5 0 8') == [4, 11, 9, 5, 0, 8]

    def test_several_double_digit_placings(self):
        assert parse_form_positions('0 11 8 11 13 9') == [0, 11, 8, 11, 13, 9]

    def test_single_digit_space_form(self):
        """Already worked by accident — must not regress."""
        assert parse_form_positions('0 8 4 6 2 7') == [0, 8, 4, 6, 2, 7]

    def test_irregular_whitespace(self):
        assert parse_form_positions('  4   11\t9 ') == [4, 11, 9]


class TestExistingFormatsUnchanged:
    def test_slash_form(self):
        assert parse_form_positions('1/3/10/2/5') == [1, 3, 10, 2, 5]

    def test_legacy_concatenated_stays_digit_by_digit(self):
        """'13425' is five placings, NOT the integer 13425."""
        assert parse_form_positions('13425') == [1, 3, 4, 2, 5]

    def test_empty_and_none(self):
        assert parse_form_positions('') == []
        assert parse_form_positions(None) == []


class TestAnalyzeFormDelegates:
    """analyze_form kept its own inline copy of the parse — the duplicate is
    exactly what CLAUDE.md rule 2 forbids, and it carried the same bug."""

    @staticmethod
    def _analyze(s):
        import inspect
        import core.analyzer as ca
        cls = next(o for _, o in vars(ca).items()
                   if inspect.isclass(o) and hasattr(o, 'analyze_form'))
        return cls.analyze_form(s)

    def test_bad_form_is_not_read_as_a_winner(self):
        """'4 11 9 5 0 8' has NO win. It must not score as 'Winner in form'."""
        score, reason = self._analyze('4 11 9 5 0 8')
        assert 'Winner' not in reason, f'phantom win from double digits: {reason}'
        assert score < 40, f'bad form scored {score:.1f}: {reason}'

    def test_terrible_form_is_not_strong_form(self):
        """11/8/11/13/9 previously read as 'Strong form: 5W in last 9'."""
        score, reason = self._analyze('0 11 8 11 13 9')
        assert 'Strong form' not in reason, f'phantom wins: {reason}'
        assert score < 40, f'terrible form scored {score:.1f}: {reason}'

    def test_space_and_slash_forms_agree(self):
        """The same six placings must score identically in either notation."""
        assert self._analyze('4 11 9 5 0 8') == self._analyze('4/11/9/5/0/8')

    def test_genuinely_strong_form_still_scores_high(self):
        score, reason = self._analyze('5 5 1 2 1 1')
        assert score > 70, f'strong form scored only {score:.1f}: {reason}'
