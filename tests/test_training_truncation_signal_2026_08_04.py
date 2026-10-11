r"""`looks_truncated` must detect a bad INGEST, not a small FIELD.

The 2026-08-03 fix gated on `estimate_field_size <= 6` as a proxy for "this
race lost its tail to a partial ingest". Measured against the live DB, that
drops 53 races in the training window — and **38 of them are genuine**:

    2026/03/01 R7  Group One 2000m, 6 runners, numbers 1-6, FIRST 4 declared
    2023/03/19 R7  G1 1400m, 5 runners, numbers 1-5, TIERCE + TRIO declared
    2022/05/01 R3  G3, 5 runners, numbers 1-5, 7 dividend rows

Complete fields with contiguous saddle numbers and declared dividends. The
heuristic removed ~11% of all Group races, and the deletion correlates with
the label (a 6-horse field has a ~1/6 base win rate vs ~1/12), so it is MNAR
deletion on a feature the model actually uses (`field_size` 0.034,
`field_rel_draw` 0.042). It also pushed the training minimum field size to 7,
so inference on a 6-runner Group race now extrapolates outside training
support. One train/predict skew was traded for another.

The real signature of the artefact is documented in memory
(`project_overseas_runner_rows`): HK horses racing abroad (Dubai/Saudi/Japan)
are stored WITHOUT a venue. Measured: 15 such races, and every one of the 38
genuine short fields has a venue. So venue is the discriminating signal, not
field size.

A conservative floor is kept as a backstop against a future truncated ingest:
the smallest genuine HK field in five years is 5 runners, so `<= 4` cannot
catch a real race (verified: 0 venue-present races with <= 4 finishers).
"""
import pytest

from ml.training import estimate_field_size, looks_truncated


def _rows(n, venue='Sha Tin', start=1):
    return [{'finish_position': p, 'venue': venue, 'horse_number': p}
            for p in range(start, start + n)]


class TestGenuineShortFieldsSurvive:
    def test_six_runner_group_race_is_kept(self):
        """2026/03/01 R7 — Group One, 6 runners, FIRST 4 declared."""
        assert looks_truncated(_rows(6)) is False

    def test_five_runner_group_race_is_kept(self):
        """2023/03/19 R7 — G1, 5 runners. The smallest genuine HK field."""
        assert looks_truncated(_rows(5)) is False

    def test_full_field_is_kept(self):
        assert looks_truncated(_rows(12)) is False


class TestArtefactsAreStillDropped:
    def test_missing_venue_is_dropped(self):
        """Overseas runners (Dubai/Saudi/Japan) carry no venue."""
        assert looks_truncated(_rows(2, venue=None)) is True

    def test_blank_venue_is_dropped(self):
        assert looks_truncated(_rows(2, venue='   ')) is True

    def test_missing_venue_key_entirely_is_dropped(self):
        assert looks_truncated([{'finish_position': 1}]) is True

    def test_absurdly_short_venue_present_race_is_dropped(self):
        """Backstop: 4 or fewer runners cannot be a real HK card."""
        assert looks_truncated(_rows(4)) is True

    def test_a_missing_venue_cannot_be_assigned_to_a_local_race(self):
        rows = _rows(6)
        rows[0]['venue'] = None
        assert looks_truncated(rows) is True

    def test_empty_input_is_not_truncated(self):
        assert looks_truncated([]) is False


class TestFieldSizeRecoveryUnchanged:
    def test_dropped_finisher_still_recovered_from_max_position(self):
        rows = [{'finish_position': p, 'venue': 'Sha Tin'} for p in (1, 2, 3, 12)]
        assert estimate_field_size(rows) == 12

    def test_row_count_wins_when_larger(self):
        assert estimate_field_size(_rows(10)) == 10


class TestAgainstTheLiveDatabase:
    """The claim that motivated this change, checked on real data."""

    def test_no_genuine_hk_race_is_dropped(self):
        import sqlite3
        import config
        if not config.DB_PATH.exists():
            pytest.skip('no local DB')
        conn = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT r.race_date, r.race_number, COUNT(*) n, MIN(r.venue) venue,
                   (SELECT COUNT(*) FROM race_dividends d
                     WHERE d.race_date=r.race_date
                       AND d.race_number=r.race_number) divs
              FROM race_results r
             WHERE r.race_date >= '2021/07/16' AND r.finish_position > 0
             GROUP BY r.race_date, r.race_number
        """).fetchall()
        conn.close()
        if not rows:
            pytest.skip('empty DB')

        dropped_with_dividends = [
            r for r in rows
            if looks_truncated([{'finish_position': 1, 'venue': r['venue']}] * r['n'])
            and r['divs'] > 0
        ]
        assert not dropped_with_dividends, (
            f"{len(dropped_with_dividends)} race(s) with declared HKJC dividends "
            f"would be dropped from training, e.g. "
            f"{[(r['race_date'], r['race_number'], r['n']) for r in dropped_with_dividends[:5]]}"
        )
