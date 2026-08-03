r"""`edge_filtered` must not mix two incompatible edge_pp scales.

The variant selects races by the rank-1 horse's `edge_pp`, read straight out of
the frozen results/race_data_*.json. But `edge_pp = (pure_prob - market_prob)`,
and pure_prob was recalibrated on 2026-08-03: before that it was under-confident
by roughly 2x, so every stored edge_pp carries a structural ~-10pp that
post-calibration meetings do not.

Backfilling the variant across the boundary therefore compares races scored on
two different rulers. The old meetings essentially never clear a positive floor
(mean edge_pp -10.28pp on the 634-race holdout), so the variant would look like
"bets almost nothing, then suddenly bets a lot" — an artefact of the recalibration
date, not of the market.

Meetings scored before the cutover are excluded outright, and the count is
reported rather than silently dropped.
"""
import pytest

from scripts.stake_ab import (
    race_meets_edge_floor, PURE_CALIBRATION_DATE, MEETING_DATE_KEY,
)


def _race(edge_pp: float, meeting_date: str = None) -> dict:
    race = {'horses': [
        {'rank': 1, 'edge_pp': edge_pp},
        {'rank': 2, 'edge_pp': edge_pp - 4.0},
    ]}
    if meeting_date is not None:
        race[MEETING_DATE_KEY] = meeting_date
    return race


class TestScaleGuard:
    def test_post_calibration_meeting_can_clear_the_floor(self):
        assert race_meets_edge_floor(_race(8.0, '2026/08/10'), 5.0) is True

    def test_pre_calibration_meeting_never_clears_the_floor(self):
        """Same +8pp, but measured on the old under-confident ruler."""
        assert race_meets_edge_floor(_race(8.0, '2026/07/15'), 5.0) is False

    def test_the_cutover_date_itself_counts_as_calibrated(self):
        assert race_meets_edge_floor(
            _race(8.0, PURE_CALIBRATION_DATE), 5.0) is True

    def test_unstamped_meeting_is_refused(self):
        """Unknown provenance is treated as pre-calibration, not as valid."""
        assert race_meets_edge_floor(_race(8.0), 5.0) is False

    def test_a_post_calibration_race_below_the_floor_still_fails(self):
        assert race_meets_edge_floor(_race(1.0, '2026/08/10'), 5.0) is False


class TestUnchangedBehaviour:
    def test_missing_rank_1_is_refused(self):
        race = {'horses': [{'rank': 2, 'edge_pp': 20.0}],
                MEETING_DATE_KEY: '2026/08/10'}
        assert race_meets_edge_floor(race, 5.0) is False

    def test_missing_edge_pp_is_refused(self):
        race = {'horses': [{'rank': 1}], MEETING_DATE_KEY: '2026/08/10'}
        assert race_meets_edge_floor(race, 5.0) is False

    def test_only_rank_1_counts(self):
        """A big edge on the rank-2 horse must not select the race."""
        race = {'horses': [{'rank': 1, 'edge_pp': -2.0},
                           {'rank': 2, 'edge_pp': 40.0}],
                MEETING_DATE_KEY: '2026/08/10'}
        assert race_meets_edge_floor(race, 5.0) is False


class TestStamping:
    def test_evaluate_date_stamps_every_race(self):
        """The guard is useless if evaluate_date forgets to stamp."""
        import ast
        import inspect
        from scripts import stake_ab

        src = inspect.getsource(stake_ab.evaluate_date)
        tree = ast.parse(src.lstrip())
        def _is_the_key(node):
            # Accept either race['_meeting_date'] or race[MEETING_DATE_KEY].
            if isinstance(node, ast.Constant):
                return node.value == MEETING_DATE_KEY
            return isinstance(node, ast.Name) and node.id == 'MEETING_DATE_KEY'

        stamps = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and _is_the_key(n.slice)
            and isinstance(n.ctx, ast.Store)
        ]
        assert stamps, (
            f"evaluate_date must stamp race[{MEETING_DATE_KEY!r}] so "
            "race_meets_edge_floor can tell which edge_pp scale it is reading"
        )
