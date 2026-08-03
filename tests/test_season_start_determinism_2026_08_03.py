"""Season-to-date stats must be derived from the DATA, not the wall clock.

Bug (code review 2026-08-03): `HorseDatabase._current_season_start()` used
`datetime.now()`, while `_season_start_for(date_str)` — used by the leak-free
backtest path — derives the season from the race date. `rebuild_stats()` uses
the wall-clock one.

Two consequences:

1. NON-DETERMINISM. The same DB rebuilt on two different days aggregates
   jockey/trainer stats over different windows, so a backtest is not
   reproducible across a season boundary.

2. OFF-SEASON COLLAPSE, which is live right now. On 2026-08-03 the wall clock
   says the season began 2026/07/01, but the last race in the DB is
   2026/07/15 — so "season to date" spans five meetings. jockey_stats /
   trainer_stats are rebuilt over that stub and feed the rule engine's jockey
   and trainer factors for every prediction.

Deriving from MAX(race_date) fixes both: it is deterministic, and during the
off-season it keeps the just-completed season rather than an empty new one.
"""
import pytest

from data.database import HorseDatabase


@pytest.fixture
def db(tmp_path):
    d = HorseDatabase(db_path=str(tmp_path / "t.db"))
    yield d
    d.close()


def _add_race(d, date, name="Z Purton"):
    d.conn.execute(
        "INSERT INTO race_results (race_date, race_number, horse_name, jockey, "
        "trainer, finish_position, is_winner, is_top3) "
        "VALUES (?, 1, 'H', ?, 'T', 1, 1, 1)", (date, name))
    d.conn.commit()


class TestSeasonStartFromData:
    def test_uses_latest_race_date_not_today(self, db):
        """A DB whose latest race is mid-season must not use today's date.

        Uses a clearly mid-season date so this asserts DATA-derivation only,
        independent of where the Jul-1 boundary falls. (Separate known issue:
        HK seasons actually end in mid-July, so July meetings land in the
        FOLLOWING season under the Jul1-Jun30 convention this code uses.)
        """
        _add_race(db, "2026/03/15")
        assert db._current_season_start() == "2025/07/01"

    def test_season_boundary_is_july_first(self, db):
        _add_race(db, "2026/08/20")
        assert db._current_season_start() == "2026/07/01"

    def test_is_deterministic_across_calls(self, db):
        _add_race(db, "2026/07/15")
        assert db._current_season_start() == db._current_season_start()

    def test_matches_the_date_aware_helper(self, db):
        """The two helpers must agree on the same date — they diverged before."""
        _add_race(db, "2026/07/15")
        assert db._current_season_start() == HorseDatabase._season_start_for("2026/07/15")

    def test_empty_db_does_not_crash(self, db):
        got = db._current_season_start()
        assert got.endswith("/07/01") and len(got) == 10
