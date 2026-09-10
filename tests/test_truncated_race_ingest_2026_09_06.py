r"""A partially-published race must not be frozen into the DB.

HKJC publishes a result in two stages: the first four placings appear within
a minute of the all-clear (with win odds still 0.00), and the full field plus
dividends follow several minutes later. Observed four times on 2026/09/06 —
R5, R6, R7 and R10 each sat at 4 finishers for 3-11 minutes.

`_meeting_is_complete` only checks that stored race NUMBERS form a contiguous
1..N card, so a race ingested during that window is "complete" — its number is
present — and the re-scrape guard skips the date forever. That is the same
freeze bug the contiguity check was written to fix (2025/11/15 lost race 8),
one level down: at the FINISHER level rather than the race level. On 2026/09/06
only a manual finisher-count check before `raceday finish` prevented it.

Two signals separate a truncated race from a small one, both verified against
the full 4,769-race DB:

  * finisher count — no HK meeting race has fewer than 5 finishers. All 14
    races in the DB with <=4 rows are venue=NULL overseas runners (HK horses
    racing in Dubai/Saudi/Japan), which are legitimate and must NOT be flagged
    for re-scrape (they would never repair — there is no HKJC page for them).
  * win odds — 4,767 of 4,769 races carry odds on at least one row. A
    partially-published race carries 0.00 on every row. The 2 exceptions are
    both venue=NULL overseas rows, so the venue filter covers them too.
"""
from __future__ import annotations

import pytest

from data.scraper import truncated_races


def _row(n, finishers, odds_rows, venue='ST'):
    return {'race_number': n, 'finishers': finishers,
            'odds_rows': odds_rows, 'venue': venue}


class TestCatchesThePartialPublish:
    def test_four_finishers_no_odds_is_truncated(self):
        """The exact 2026/09/06 R10 state: 4 of 13 home, all odds 0.00."""
        assert truncated_races([_row(10, 4, 0)]) == [10]

    def test_four_finishers_with_odds_is_still_truncated(self):
        """Count alone is enough — no HK race finishes fewer than 5."""
        assert truncated_races([_row(10, 4, 4)]) == [10]

    def test_full_field_without_odds_is_truncated(self):
        """Odds alone is enough — a real result always carries them."""
        assert truncated_races([_row(10, 13, 0)]) == [10]

    def test_reports_every_bad_race_in_the_card(self):
        rows = [_row(1, 14, 14), _row(5, 4, 0), _row(6, 4, 0), _row(10, 13, 13)]
        assert truncated_races(rows) == [5, 6]


class TestDoesNotFlagLegitimateRaces:
    def test_small_but_real_hk_field_is_fine(self):
        """2026/09/06 R3 was a genuine 6-runner Group race."""
        assert truncated_races([_row(3, 6, 6)]) == []

    def test_five_finishers_is_the_floor_and_passes(self):
        assert truncated_races([_row(3, 5, 5)]) == []

    def test_non_finishers_do_not_trigger_it(self):
        """2026/09/06 R7: 8 finishers from 9 runners — one did not finish."""
        assert truncated_races([_row(7, 8, 8)]) == []

    def test_overseas_rows_are_never_flagged(self):
        """venue=NULL rows are HK horses racing abroad — 1-4 rows is correct
        and there is no HKJC results page to repair from, so flagging them
        would cause an endless re-scrape loop."""
        assert truncated_races([_row(1, 1, 1, venue=None)]) == []
        assert truncated_races([_row(1, 2, 0, venue='')]) == []

    def test_empty_input(self):
        assert truncated_races([]) == []


class TestAgainstTheRealDatabase:
    def test_no_stored_hk_race_is_flagged(self):
        """Every race currently in the DB must pass — otherwise the guard
        would send the scraper into a loop over historical meetings."""
        import sqlite3
        import config
        conn = sqlite3.connect(str(config.DB_PATH))
        rows = [{'race_number': r[1], 'finishers': r[2], 'odds_rows': r[3],
                 'venue': r[4]}
                for r in conn.execute(
                    """SELECT race_date, race_number, COUNT(*),
                              SUM(CASE WHEN win_odds>0 THEN 1 ELSE 0 END),
                              MAX(venue)
                       FROM race_results GROUP BY race_date, race_number""")]
        conn.close()
        bad = truncated_races(rows)
        assert bad == [], f'guard would re-scrape {len(bad)} historical races'
