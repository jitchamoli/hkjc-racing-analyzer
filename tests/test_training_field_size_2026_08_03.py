"""Training must not infer field_size from the number of rows it happens to have.

Bug (code review 2026-08-03): `build_training_data_from_db` computed

    field_sizes[rk] = field_sizes.get(rk, 0) + 1     # COUNT(*) of DB rows

while INFERENCE uses the declared field (`len(all_horses)` in core/analyzer).
When results ingest drops finishers — 241 of 4,759 races in the DB have
row_count < max(finish_position), plus an unknown number of truncated-tail
races the heuristic cannot see — training learns a field size that inference
would never produce. That is a CLAUDE.md ML Rule 2 (train/predict consistency)
violation on two features carrying ~0.080 combined importance in v6.11:
`field_size` (0.038) and `field_rel_draw` (0.042). It also skews the
`weight_vs_field` median for every horse in the affected race.

`max(finish_position)` is a lower bound on the true field that survives
dropped rows, so the field size must be at least that. This cannot recover a
race whose entire tail is missing (positions 1..4 stored of a 12-runner
field) — those are flagged, not silently trained on.
"""
import pytest

from ml.training import estimate_field_size, looks_truncated


class TestEstimateFieldSize:
    def test_complete_race_uses_the_row_count(self):
        rows = [{"finish_position": i} for i in range(1, 13)]
        assert estimate_field_size(rows) == 12

    def test_dropped_finisher_is_recovered_from_finish_position(self):
        """11 rows stored but someone finished 12th -> field was 12."""
        rows = [{"finish_position": i} for i in range(1, 13) if i != 5]
        assert estimate_field_size(rows) == 12

    def test_multiple_dropped_finishers(self):
        rows = [{"finish_position": i} for i in (1, 2, 3, 11, 14)]
        assert estimate_field_size(rows) == 14

    def test_dead_heat_does_not_inflate_the_field(self):
        """Two horses share 2nd; positions run 1,2,2,4 -> field is 4."""
        rows = [{"finish_position": p} for p in (1, 2, 2, 4)]
        assert estimate_field_size(rows) == 4

    def test_non_finishers_do_not_shrink_the_field(self):
        """A WV/PU horse stored with position 0 still occupied a stall."""
        rows = [{"finish_position": p} for p in (1, 2, 3, 0, 0)]
        assert estimate_field_size(rows) == 5

    def test_missing_positions_fall_back_to_row_count(self):
        rows = [{"finish_position": None} for _ in range(9)]
        assert estimate_field_size(rows) == 9

    def test_empty_race(self):
        assert estimate_field_size([]) == 0


class TestTruncatedTailDetection:
    """A race whose tail is entirely missing cannot be repaired by the above."""

    def test_truncated_tail_is_reported(self):

        # 4 rows, positions 1-4: consistent, but a 12-runner card lost 8 rows.
        rows = [{"finish_position": i} for i in range(1, 5)]
        assert looks_truncated(rows, typical_field=12) is True

    def test_normal_small_field_is_not_flagged(self):
        rows = [{"finish_position": i} for i in range(1, 10)]
        assert looks_truncated(rows, typical_field=12) is False
