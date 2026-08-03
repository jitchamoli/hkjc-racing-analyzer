"""An incompletely-scraped meeting must be re-scrapeable.

Bug (code review 2026-08-03): `scrape_meeting_results` gives up on a card
after EMPTY_TOLERANCE consecutive empty/unparseable races, so a transient HTTP
failure mid-card silently truncates the meeting. `main()` then guarded
re-runs with:

    if existing > 0: continue      # "already have this date"

Any rows at all meant the date was never re-scraped, so a partial meeting was
frozen forever. Real casualty in the production DB: 2025/11/15 holds races
1-7 and 9-10 — race 8 is absent from race_results AND race_dividends, and no
`scrape_history --date 2025/11/15` re-run would ever repair it.

The guard should skip only meetings that look COMPLETE: a contiguous 1..N
race sequence. A gap, or a card whose last race is missing, must re-scrape.
"""
from data.scraper import _meeting_is_complete


class TestMeetingCompleteness:
    def test_contiguous_card_is_complete(self):
        assert _meeting_is_complete([1, 2, 3, 4, 5, 6, 7, 8, 9]) is True

    def test_interior_gap_is_incomplete(self):
        """The real 2025/11/15 case — race 8 dropped mid-card."""
        assert _meeting_is_complete([1, 2, 3, 4, 5, 6, 7, 9, 10]) is False

    def test_multiple_gaps_are_incomplete(self):
        assert _meeting_is_complete([1, 2, 5, 6]) is False

    def test_missing_first_race_is_incomplete(self):
        assert _meeting_is_complete([2, 3, 4, 5]) is False

    def test_empty_meeting_is_incomplete(self):
        """No rows at all — nothing to skip, must scrape."""
        assert _meeting_is_complete([]) is False

    def test_single_race_meeting_is_complete(self):
        """Overseas/simulcast one-race days are legitimately complete."""
        assert _meeting_is_complete([1]) is True

    def test_duplicate_race_numbers_do_not_mask_a_gap(self):
        """Several runners per race means race numbers repeat."""
        assert _meeting_is_complete([1, 1, 1, 2, 2, 4, 4]) is False
