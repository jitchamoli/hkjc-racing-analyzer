"""A failed blend-calibrator load must not be cached forever.

Bug (code review 2026-08-03): `_load_blend_calibrator` set

    _BLEND_CALIBRATOR_LOADED = True

BEFORE reading the file. So a missing sidecar, a checksum mismatch or an
unpickle error logged one line, cached None, and every subsequent race in the
process ran uncalibrated — a later fix to the artifact was never re-read.

That matters because the calibrator materially changes betting decisions: an
A/B over 400 fuzzed races moved the conviction tier in 22% of them and the
race bet type in 5. Production entry points (generate_predictions,
monitor/race_day) preflight and hard-exit, but scripts/backtest.py and the
shadow/compare scripts do not — and those produce the SHIP/KILL numbers.
There is also a TOCTOU window: preflight runs once at startup while the load
is lazy, so an auto-learn refit mid-process could poison the rest of the run.

Successful loads must still be cached (the file is read once per process).
"""
import pathlib

import pytest

import core.scorer as scorer


@pytest.fixture(autouse=True)
def _reset_cache():
    scorer.reset_blend_calibrator_cache()
    yield
    scorer.reset_blend_calibrator_cache()


_REAL_READ_BYTES = pathlib.Path.read_bytes


def _count_calibrator_reads(monkeypatch, counter, fail=False):
    def patched(self):
        if self.name == "blend_calibrator.pkl":
            counter["n"] += 1
            if fail:
                raise OSError("transient read failure")
        return _REAL_READ_BYTES(self)

    monkeypatch.setattr("pathlib.Path.read_bytes", patched)


class TestBlendCalibratorRetry:
    def test_failed_load_is_retried_on_the_next_call(self, monkeypatch):
        """A transient/bad artifact must not poison the whole process."""
        reads = {"n": 0}
        _count_calibrator_reads(monkeypatch, reads, fail=True)

        assert scorer._load_blend_calibrator() is None
        assert scorer._load_blend_calibrator() is None

        assert reads["n"] == 2, (
            f"failed load was cached — artifact read {reads['n']} time(s), so a "
            "later fix would never be picked up"
        )

    def test_recovery_after_a_transient_failure(self, monkeypatch):
        """Once the artifact reads cleanly again, calibration must resume."""
        reads = {"n": 0}
        _count_calibrator_reads(monkeypatch, reads, fail=True)
        assert scorer._load_blend_calibrator() is None

        monkeypatch.undo()
        assert scorer._load_blend_calibrator() is not None, (
            "pipeline stayed uncalibrated after the artifact recovered"
        )

    def test_successful_load_is_cached(self, monkeypatch):
        """Guard: the happy path must still read the artifact only once."""
        reads = {"n": 0}
        _count_calibrator_reads(monkeypatch, reads)

        first = scorer._load_blend_calibrator()
        second = scorer._load_blend_calibrator()

        assert first is second
        assert reads["n"] == 1, f"artifact re-read {reads['n']} times on the happy path"

    def test_production_calibrator_still_loads_and_verifies(self):
        """Sanity: the on-disk artifact is intact and usable."""
        iso = scorer._load_blend_calibrator()
        assert iso is not None, "production blend_calibrator.pkl failed to load"
        assert hasattr(iso, "predict")
